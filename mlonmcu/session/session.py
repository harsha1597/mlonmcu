#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Definition of a MLonMCU Run which represents a set of benchmarks in a session."""
import os,time
import shutil
import filelock
import tempfile
import multiprocessing
from datetime import datetime
from enum import Enum
from pathlib import Path
import concurrent.futures

from tqdm import tqdm
import pandas as pd  # <-- ADDED IMPORT
import numpy as np   # <-- ADDED IMPORT

from mlonmcu.session.run import Run
from mlonmcu.logging import get_logger
from mlonmcu.report import Report
from mlonmcu.config import filter_config
from mlonmcu.session.estimate.gcn import EstimatePostBuild
from mlonmcu.session.estimate.extract_graph import get_sw_flags
from .postprocess.postprocess import SessionPostprocess
from .run import RunStage
# vvv ADDED IMPORTS vvv
from mlonmcu.target.metrics import Metrics
from mlonmcu.artifact import Artifact, ArtifactFormat
# ^^^ ADDED IMPORTS ^^^

logger = get_logger()  # TODO: rename to get_mlonmcu_logger


class SessionStatus(Enum):  # TODO: remove?
    """Status type for a session."""

    CREATED = 0
    OPEN = 1
    CLOSED = 2
    ERROR = 3


class Session:
    """A session which wraps around multiple runs in a context."""

    DEFAULTS = {
        "report_fmt": "csv",
        "runtime_to_codesize":1 # Ratio of importance between runtime and codesize
    }

    def __init__(self, label=None, idx=None, archived=False, dest=None, config=None):
        self.timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.label = (
            label if isinstance(label, str) and len(label) > 0 else ("unnamed" + "_" + self.timestamp)
        )  # TODO: decide if named sessions should also get a timestamp?
        self.idx = idx
        self.config = config if config else {}
        self.config = filter_config(self.config, "session", self.DEFAULTS, set(), set())
        self.status = SessionStatus.CREATED
        self.opened_at = None
        self.closed_at = None
        self.runs = []
        self.report = None
        self.next_run_idx = 0
        self.archived = archived
        self.dir = Path(dest) if dest is not None else None
        self.tempdir = None
        self.session_lock = None
        self.estimator = None
        
    @property
    def runs_dir(self):
        return None if self.dir is None else (self.dir / "runs")

    def __enter__(self):
        if self.archived:
            logger.warning("Opening an already archived session is not recommended")
        else:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            self.close()

    @property
    def prefix(self):
        """get prefix property."""
        return f"[session-{self.idx}] " if self.idx else ""

    @property
    def report_fmt(self):
        """get report_fmt property."""
        return str(self.config["report_fmt"])

    def create_run(self, *args, **kwargs):
        """Factory method to create a run and add it to this session."""
        idx = len(self.runs)
        logger.debug("Creating a new run with id %s", idx)
        run = Run(*args, idx=idx, session=self, **kwargs)
        self.runs.append(run)
        return run

    #  def update_run(self): # TODO TODO
    #      pass

    def get_reports(self):
        """Returns a full report which includes all runs in this session."""
        if self.report:
            return self.report

        reports = [run.get_report() for run in self.runs]
        merged = Report()
        merged.add(reports)
        return merged

    def enumerate_runs(self):
        """Update run indices."""
        # Find start index
        max_idx = -1
        for run in self.runs:
            if run.archived:
                max_idx = max(max_idx, run.idx)
        run_idx = max_idx + 1
        last_run_idx = None
        for run in self.runs:
            if not run.archived:
                run.idx = run_idx
                run.init_directory()
                run_idx += 1
                last_run_idx = run.idx
        self.next_run_idx = run_idx
        if last_run_idx is not None:
            self.update_latest_run_symlink(last_run_idx)
    
     
    def update_latest_run_symlink(self, latest_run_idx):
        run_link = self.runs_dir / "latest"  # TODO: Create relative symlink using os.path.relpath for portability
        if os.path.islink(run_link):
            os.unlink(run_link)
        os.symlink(self.runs_dir / str(latest_run_idx), run_link)

    def request_run_idx(self):
        """Return next free run index."""
        ret = self.next_run_idx
        self.next_run_idx += 1
        # TODO: find a better approach for this
        return ret
    def get_built_files(self,run):
        import pickle
        from mlonmcu.models.model import Model, Program
        """Estimate runtime and codesize using a cost model."""
        # logger.debug("%s [%s] Processing stage ESTIMATE", run.prefix, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # logger.info(f"MLIF flags: {run.sw_flags}")
        debug_path = "/nfs/TUEIEDAscratch/ge85zic/graph_regressor/mlonmcu_graph/debug/"
        run_id=run.idx
        def get_tir_and_c_files(codegen_dir):
            tir_file = "default.tir"
            c_file = "default.c"
            tir_file_path = None
            c_file_path = None

            if not os.path.isdir(codegen_dir):
                logger.warning(f"Codegen directory not found for estimation: {codegen_dir}")
                return None, None

            contents = os.listdir(codegen_dir)
            if tir_file in contents:
                tir_file_path=os.path.join(codegen_dir, tir_file)
            else:
                logger.warning(f"TIR file {tir_file} not found in {codegen_dir}")
                return None,None
            
            if c_file in contents:
                c_file_path=os.path.join(codegen_dir, c_file)
            elif "codegen" in contents:
                # try to find c file with default prefix
                import glob
                c_files = glob.glob(os.path.join(codegen_dir, "codegen","host","src","default*.c"))
                # Locate the c file with the main definition
                c_file_w_main = None
                if len(c_files) == 3:
                    c_file_w_main = "default_lib2.c"
                elif len(c_files)>1:
                    c_file_w_main = "default_lib1.c"
                
                if c_file_w_main:
                    c_file_path_list = [path for path in c_files if c_file_w_main in os.path.basename(path)]
                    c_file_path = c_file_path_list[0] if len(c_file_path_list) == 1 else None
                elif len(c_files) == 1:
                    c_file_path = c_files[0]
            
            return tir_file_path, c_file_path
       
        name = "default"
        
        if isinstance(run.model, Model):
            if not run.completed[RunStage.BUILD]:
                logger.warning(f"Run {run.idx} has not completed BUILD stage. Skipping.")
                return None, None, None, run_id
            run.export_stage(RunStage.BUILD, optional=run.export_optional)
            
            # This should be initialized here for the artifact injection
            
            codegen_dir = run.dir if not run.stage_subdirs else (run.dir / "stages" / str(int(RunStage.BUILD)))
            
            sw_feat = run.sw_flags
            tir_file_path, c_file_path = get_tir_and_c_files(codegen_dir)
        
        else: # Program
            if not run.completed[RunStage.LOAD]:
                logger.warning(f"Run {run.idx} has not completed LOAD stage. Skipping.")
                return None, None, None, run_id
                
            # if RunStage.ESTIMATE not in run.artifacts_per_stage:
            #      run.artifacts_per_stage[RunStage.ESTIMATE] = {}
                 
            codegen_dir = run.dir if not run.stage_subdirs else (run.dir / "stages" / str(int(RunStage.LOAD))) # Changed from BUILD
            tir_file_path, c_file_path = get_tir_and_c_files(codegen_dir)
            sw_feat = run.sw_flags
            
        return tir_file_path, c_file_path, sw_feat, run_id
    
    def process_runs(
        self,
        until=RunStage.DONE,
        per_stage=False,
        print_report=False,
        num_workers=1,
        progress=False,
        export=False,
        context=None,
        estimate_postbuild = False,
        cost_model_path= "/nfs/TUEIEDAscratch/ge85zic/mlonmcu/mlonmcu/session/estimate/model/GNN_Estimator.pt"
        # runs_filter=None,
    ):
        """Process all runs in this session until a given stage."""

        # TODO: Add configurable callbacks for stage/run complete
        assert self.active, "Session needs to be opened first"

        self.enumerate_runs()
        self.report = None
        estimator_results={}
        runs = [run for run in self.runs ]#if runs_filter is None or run.idx in runs_filter]
        
        active_run_ids = {run.idx for run in runs} # This list gets filtered after estimate stage
        if estimate_postbuild:
            logger.info(f"Loading shared GNN estimator from {cost_model_path}...")
            self.estimator = EstimatePostBuild(cost_model_path)
            logger.info("Shared estimator loaded.")
            
        assert num_workers > 0, "num_workers can not be < 1"
        workers = []
        # results = []
        workers = []
        pbar = None  # Outer progress bar
        pbar2 = None  # Inner progress bar
        num_runs = len(self.runs)
        num_failures = 0
        stage_failures = {}
        worker_run_idx = []

        
        def filter_from_estimates(results_dict, filter_type="pareto", runtime_threshold=None, code_size_threshold=None, ratio=None):
            """Filter runs based on a dictionary of estimation results."""
            
            # 1. Create DataFrame from the results dictionary
            # The keys of the dict are the Run IDs
            # The values are the (runtime, codesize) tuples or None
            
            # Create a DataFrame from the dictionary
            # Orient='index' makes keys the index and tuples the columns
            df = pd.DataFrame.from_dict(
                results_dict, 
                orient='index', 
                columns=['Estimated Runtime [s]', 'Estimated Code Size [bytes]']
            )
            
            # Add the Run IDs (the index) as a regular column
            df['Run'] = df.index

            # 2. Use the exact same filtering logic as before
            runtime_col = "Estimated Runtime [s]"
            codesize_col = "Estimated Code Size [bytes]"
            
            # Drop rows where estimation failed (where the original value was None, now np.nan)
            df = df.dropna(subset=[runtime_col, codesize_col])
            if df.empty:
                logger.warning("No runs with valid estimates found. Skipping filtering.")
                return []  # Return no runs
                
            if filter_type == "threshold" and runtime_threshold is not None and code_size_threshold is not None:
                filtered_df = df[
                    (df[runtime_col] <= runtime_threshold)
                    & (df[codesize_col] <= code_size_threshold)
                ]
                filtered_runids = filtered_df["Run"].tolist()
            elif filter_type == "pareto":
                # Pareto front filtering
                filtered_indices = []
                # Use .iterrows() which iterates over rows of the DataFrame
                for i, row in df.iterrows():
                    dominated = False
                    # i is the Run ID (index)
                    for j, other_row in df.iterrows():
                        if i != j:
                            if (
                                other_row[runtime_col] <= row[runtime_col]
                                and other_row[codesize_col] <= row[codesize_col]
                                and (
                                    other_row[runtime_col] < row[runtime_col]
                                    or other_row[codesize_col] < row[codesize_col]
                                )
                            ):
                                dominated = True
                                break
                    if not dominated:
                        filtered_indices.append(i) # Add the Run ID (index)
                
                # Select the rows based on the index (Run ID)
                filtered_df = df.loc[filtered_indices]
                filtered_runids = filtered_df["Run"].tolist()
            elif filter_type == "weighted":
                if ratio is None:
                    logger.warning("Ratio not provided for 'weighted' filter. Defaulting to 0.5.")
                    ratio = 0.5
                scores = (
                    ratio * df[runtime_col] / df[runtime_col].max()
                    + (1 - ratio) * df[codesize_col] / df[codesize_col].max()
                )
                threshold = scores.median()  # Example: keep runs below median score
                filtered_df = df[scores <= threshold]
                filtered_runids = filtered_df["Run"].tolist()
            else:
                logger.warning(f"Unknown filter type: {filter_type}. Defaulting to 'pareto'.")
                # Pareto front filtering (default)
                filtered_indices = []
                for i, row in df.iterrows():
                    dominated = False
                    for j, other_row in df.iterrows():
                        if i != j:
                            if (
                                other_row[runtime_col] <= row[runtime_col]
                                and other_row[codesize_col] <= row[codesize_col]
                                and (
                                    other_row[runtime_col] < row[runtime_col]
                                    or other_row[codesize_col] < row[codesize_col]
                                )
                            ):
                                dominated = True
                                break
                    if not dominated:
                        filtered_indices.append(i)
                filtered_df = df.loc[filtered_indices]
                filtered_runids = filtered_df["Run"].tolist()
                
            return filtered_runids
        
        def _init_progress(total, msg="Processing..."):
            """Helper function to initialize a progress bar for the session."""
            return tqdm(
                total=total,
                desc=msg,
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}s]",
                leave=None,
            )
        
        def _update_progress(pbar, count=1):
            """Helper function to update the progress bar for the session."""
            pbar.update(count)

        def _close_progress(pbar):
            """Helper function to close the session progressbar, if available."""
            if pbar:
                pbar.close()
        def update_report_with_estimator_results(report, estimator_results,estimator_time):
            """Merges GNN estimator results into the main session report."""
            if not estimator_results:
                # No results to merge, return original report
                return report

            # 1. Convert the estimator_results dict to a DataFrame
            #    The keys (run_ids) will become the index.
            df_estimates = pd.DataFrame.from_dict(
                estimator_results,
                orient="index",
                columns=["Estimated Runtime [s]", "Estimated Code Size [bytes]"],
            )

            # 2. Name the index 'Run' to match the column in the main report
            df_estimates.index.name = "Run"

            # 3. Reset the index so 'Run' becomes a regular column for merging
            df_estimates = df_estimates.reset_index()
            df_estimates["Total Estimation Time [s]"] = estimator_time # Same for all runs
            # 4. Get the main part of the report DataFrame
            df_pre = report.pre_df

            # logger.info(f" df_main: {df_main}")
            # 5. Check that the main report has a 'Run' column
            if "Run" not in df_pre.columns:
                logger.warning("Main report DataFrame (main_df) does not have a 'Run' column. Unable to merge estimates.")
                return report

            # 6. Ensure the 'Run' columns are of a compatible type for merging.
            try:
                # Ensure both 'Run' columns are the same type before merging
                df_pre["Run"] = pd.to_numeric(df_pre["Run"])
                df_estimates["Run"] = pd.to_numeric(df_estimates["Run"])
            except (ValueError, TypeError):
                logger.warning("Could not convert 'Run' column to numeric. Merge might fail or be incorrect.")

            # 7. Merge the DataFrames
            #    We use a 'left' merge to keep all rows from the original report (df_main)
            #    and add the estimate columns. Runs without estimates will have NaN.
            df_merged = pd.merge(df_pre, df_estimates, on="Run", how="left")
            rec = df_merged.to_dict()
            report.set_pre(rec)
            # logger.info(f" Recs from main: {rec}")
            #    Assign the newly merged DataFrame back to the report's main_df attribute.
            # report.main_df = df_merged

            return report


        def _process(pbar, run, until, skip):
            """Helper function to invoke the run."""
            run.process(until=until, skip=skip, export=export)
            if not per_stage and run.has_stage(RunStage.POSTPROCESS) and RunStage.POSTPROCESS not in skip:
                # run.postprocess()
                run.process(until=RunStage.POSTPROCESS, start=RunStage.POSTPROCESS, skip=skip, export=export)
            if progress:
                _update_progress(pbar)

        def _join_workers(workers):
            """Helper function to collect all worker threads."""
            nonlocal num_failures
            results = []
            for i, w in enumerate(workers):
                try:
                    results.append(w.result())
                except Exception as e:
                    logger.exception(e)
                    logger.error("An exception was thrown by a worker during simulation")
                run_index = worker_run_idx[i]
                run = self.runs[run_index]
                if run.failing:
                    num_failures += 1
                    failed_stage = RunStage(run.next_stage).name
                    if failed_stage in stage_failures:
                        stage_failures[failed_stage].append(run_index)
                    else:
                        stage_failures[failed_stage] = [run_index]
            if progress:
                _close_progress(pbar)
            return results

        skipped_stages = [stage for stage in RunStage if not any(run.has_stage(stage) for run in runs)]
        def _used_stages(runs, until):
            """Determines the stages which are used by at least one run."""
            used = []
            
                    
            for stage_index in list(range(RunStage.LOAD, until + 1)) + [RunStage.POSTPROCESS]:
                stage = RunStage(stage_index)
                if any(run.has_stage(stage) for run in runs) and stage not in skipped_stages:
                    used.append(stage)
            return used

        used_stages = _used_stages(self.runs, until)


        with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
            if per_stage:
                if progress:
                    pbar2 = _init_progress(len(used_stages), msg="Processing stages")
                for stage in used_stages:
                    run_stage = RunStage(stage).name
                    if progress:
                        pbar = _init_progress(len(self.runs), msg=f"Processing stage {run_stage}")
                    else:
                        logger.info("%s Processing stage %s", self.prefix, run_stage)
                    for i, run in enumerate(self.runs):
                        if i == 0:
                            total_threads = min(len(self.runs), num_workers)
                            cpu_count = multiprocessing.cpu_count()
                            if (stage == RunStage.COMPILE) and run.compile_platform:
                                total_threads *= run.compile_platform.num_threads
                            if total_threads > 2 * cpu_count:
                                if pbar2:
                                    print()
                                logger.warning(
                                    "The chosen configuration leads to a maximum of %d threads being"
                                    + " processed which heavily exceeds the available CPU resources (%d)."
                                    + " It is recommended to lower the value of 'mlif.num_threads'!",
                                    total_threads,
                                    cpu_count,
                                )
                        if run.idx not in active_run_ids:
                            continue
                        if run.failing:
                            logger.warning("Skiping stage '%s' for failed run", run_stage)
                        else:
                            worker_run_idx.append(i)
                            workers.append(executor.submit(_process, pbar, run, until=stage, skip=skipped_stages))
                    _join_workers(workers)

                    
                    if stage == RunStage.BUILD and estimate_postbuild:
                        start_time = time.time()
                        logger.info("Batch estimating all runs after BUILD stage...")
                        run_ids=[]
                        c_files = []
                        tir_files=[]
                        sw_feats = []
                        
                        # Create a map to find runs by their ID
                        run_map = {run.idx: run for run in self.runs if run.idx in active_run_ids}
                        
                        for run_id in active_run_ids:
                            run = run_map[run_id]
                            tir_file_path, c_file_path, sw_feat, _ = self.get_built_files(run)
                            
                            if tir_file_path and c_file_path and sw_feat is not None:
                                run_ids.append(run_id)
                                c_files.append(c_file_path)
                                tir_files.append(tir_file_path)
                                sw_feats.append(sw_feat)
                            else:
                                logger.warning(f"Skipping estimation for Run {run_id}: Missing build files or SW features.")

                        if not c_files:
                             logger.error("No valid files found for estimation. Skipping estimate-based filtering.")
                             continue # Skip to the next stage
                             
                        # Run batch inference
                        logger.info(f"Running batch inference on {len(c_files)} runs...")
                        estimator_results = self.estimator.estimate(c_files,tir_files,sw_feats,run_ids)
                        # logger.info(f"Batch inference complete.{estimator_results}, {run_ids}")

                        runs_to_compile = filter_from_estimates(estimator_results, filter_type="pareto")
                        
                        logger.info(f"Proceeding with {len(runs_to_compile)} runs after Pareto filtering: {runs_to_compile}")
                        
                        # Update the set of active runs for all subsequent stages
                        active_run_ids = set(runs_to_compile)
                        end_time = time.time()
                        estimator_time = end_time - start_time
                        logger.info(f"Estimation time: {estimator_time} secs")
                    # ^^^ END OF NEW LOGIC ^^^
                    
                    workers = []
                    worker_run_idx = []
                    if progress:
                        _update_progress(pbar2)
                if progress:
                    _close_progress(pbar2)
            else:
                # ... (This is the non-per-stage logic, it will not run your filter) ...
                if progress:
                    pbar = _init_progress(len(self.runs), msg="Processing all runs")
                else:
                    logger.info(self.prefix + "Processing all stages")
                for i, run in enumerate(self.runs):
                    if i == 0:
                        total_threads = min(len(self.runs), num_workers)
                        cpu_count = multiprocessing.cpu_count()
                        if (
                            (until >= RunStage.COMPILE)
                            and run.compile_platform is not None
                            and run.compile_platform.name == "mlif"
                        ):
                            total_threads *= (
                                run.compile_platform.num_threads
                            )  # TODO: This should also be used for non-mlif platforms
                        if total_threads > 2 * cpu_count:
                            if pbar2:
                                print()
                            logger.warning(
                                "The chosen configuration leads to a maximum of %d threads being processed which"
                                + " heavily exceeds the available CPU resources (%d)."
                                + " It is recommended to lower the value of 'mlif.num_threads'!",
                                total_threads,
                                cpu_count,
                            )
                    worker_run_idx.append(i)
                    workers.append(executor.submit(_process, pbar, run, until=until, skip=skipped_stages))
                _join_workers(workers)
        
        if num_failures == 0:
            logger.info("All runs completed successfuly!")
        elif num_failures == num_runs:
            logger.error("All runs have failed to complete!")
        else:
            num_success = num_runs - num_failures
            logger.warning("%d out or %d runs completed successfully!", num_success, num_runs)
            summary = "\n".join(
                [
                    f"\t{stage}: \t{len(failed)} failed run(s): " + " ".join([str(idx) for idx in failed])
                    for stage, failed in stage_failures.items()
                    if len(failed) > 0
                ]
            )
            logger.info("Summary:\n%s", summary)

        report = self.get_reports()
        if estimate_postbuild:
            report = update_report_with_estimator_results(report, estimator_results,estimator_time)

        logger.info("Postprocessing session report")
        # Warning: currently we only support one instance of the same type of postprocess,
        # also it will be applied to all rows!
        session_postprocesses = []
        for run in self.runs:
            for postprocess in run.postprocesses:
                if isinstance(postprocess, SessionPostprocess):
                    if postprocess.name not in [p.name for p in session_postprocesses]:
                        session_postprocesses.append(postprocess)
        for postprocess in session_postprocesses:
            artifacts = postprocess.post_session(report)
            if artifacts is not None:
                for artifact in artifacts:
                    # Postprocess has an artifact: write to disk!
                    logger.debug("Writting postprocess artifact to disk: %s", artifact.name)
                    artifact.export(self.dir)
        report_file = Path(self.dir) / f"report.{self.report_fmt}"
        report.export(report_file)
        results_dir = context.environment.paths["results"].path
        results_file = results_dir / f"{self.label}.{self.report_fmt}"
        report.export(results_file)
        logger.info(self.prefix + "Done processing runs")
        self.report = report
        if print_report:
            logger.info("Report:\n%s", str(report.df))

        return num_failures == 0

    def discard(self):
        """Discard a run and remove its directory."""
        self.close()
        if self.dir.is_dir():
            logger.debug("Cleaning up discarded session")
            shutil.rmtree(self.dir)

    def __repr__(self):
        return f"Session(idx={self.idx},status={self.status},runs={self.runs})"

    @property
    def active(self):
        """Get active property."""
        return self.status == SessionStatus.OPEN

    @property
    def failing(self):
        """Get failng property."""

        # via report
        if self.report:
            df = self.report.df
            if "Failing" in df.columns:
                if df["Failing"].any():
                    return True
        # via runs
        if len(self.runs) > 0:
            for run in self.runs:
                if run.failing:
                    return True

        return False

    def open(self):
        """Open this run."""
        self.status = SessionStatus.OPEN
        self.opened_at = datetime.now()
        if self.dir is None:
            assert not self.archived
            self.tempdir = tempfile.TemporaryDirectory()
            self.dir = Path(self.tempdir.name)
        else:
            if not self.dir.is_dir():
                self.dir.mkdir(parents=True)
        label_file = self.dir / "label.txt"
        with open(label_file, "w") as f:
            f.write(self.label)
        self.session_lock = filelock.FileLock(os.path.join(self.dir, ".lock"))
        try:
            self.session_lock.acquire(timeout=10)
        except filelock.Timeout as err:
            raise RuntimeError("Lock on session could not be aquired.") from err
        if not os.path.exists(self.runs_dir):
            os.mkdir(self.runs_dir)

    def close(self, err=None):
        """Close this run."""
        if err:
            self.status = SessionStatus.ERROR
        else:
            self.status = SessionStatus.CLOSED
        self.closed_at = datetime.now()
        self.session_lock.release()
        os.remove(self.session_lock.lock_file)
        if self.tempdir:
            self.tempdir.cleanup()