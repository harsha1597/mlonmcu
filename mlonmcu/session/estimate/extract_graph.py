import concurrent.futures
from functools import partial
from time import gmtime, strftime
import sys
from glob import glob
import pickle
import re,os
from tqdm import tqdm
import dgl
import numpy as np
import torch
import pandas as pd
from tvm import meta_schedule as ms
import tvm
from tvm import runtime

from collections import Counter
import re
import textwrap
import ast, re


import logging
from multiprocessing import Pool
from functools import partial



def exp2m(x):
    return np.exp2(x)-1
def slog(x):
    return np.log2(x+1)
def map_features_to_values(feature_sizes, data_array,filter_names=[]):
    """
    Maps feature names to their corresponding actual values from an array.

    Args:
        feature_sizes (dict): A dictionary with feature names as keys
                              and their number of elements as values.
        data_array (np.ndarray): The array containing the feature data.

    Returns:
        dict: A dictionary mapping each feature name to its slice of values.
    """
    feature_value_map = {}
    filter = []
    current_index = 0
    
    
    total_feature_size = sum(feature_sizes.values())
    
    

    for feature_name, size in feature_sizes.items():
        start_index = current_index
        end_index = start_index + size
        current_index = end_index
        
        if feature_name in filter_names:
            
            filter.extend([False] * size)
            continue
        
        feature_value_map[feature_name] = data_array[start_index:end_index]
        
        filter.extend([True] * size)
        
        
        
        
    return feature_value_map,filter

features = {
    "float_ops": 7, 
    "int_ops": 7, 
    "bool_select": 2,
    "vect_loop": 11, 
    "unroll_loop": 11,
    "parallel_loop": 11,
    "gpu_feats": 8, 
    "memory_access_for_buffer1": 18,
    "memory_access_for_buffer2": 18,
    "memory_access_for_buffer3": 18,
    "memory_access_for_buffer4": 18,
    "memory_access_for_buffer5": 18,
    "arith_intensity": 0, 
    "group4": 4,
    "loop_group5": 3,
    
    "code_size":2 
    
    }

def get_main_tvm_function(filename):

    start_pattern = re.compile(r'^\s*\w[\w\s\*]*\btvmgen_default___tvm_main__\s*\([^;]*\)\s*\{')

    inside_function = False
    brace_count = 0
    function_lines = []

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if not inside_function:
                    
                    if start_pattern.search(line):
                        inside_function = True
                        brace_count = line.count("{") - line.count("}")
                        start_line = i
                        function_lines.append(line)
                else:
                    
                    function_lines.append(line)
                    brace_count += line.count("{") - line.count("}")
                    
                    
                    if brace_count == 0:
                        end_line = i
                        break
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        return None
    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}")
        return None

    if function_lines:
        
        return "".join(function_lines)
    else:
        logging.warning(f"Function 'tvmgen_default___tvm_main__' not found in {filename}.")
        return None

def parse_c_function_calls(file, target_function_name):
    """
    Parses C code to find a specific function and count the calls to other
    functions within its body.

    Args:
        source_code (str): A string containing the C source code.
        target_function_name (str): The name of the function to analyze.

    Returns:
        collections.Counter: A Counter object mapping function names to their
                             call counts. Returns None if the function is not found.
    """
    source_code = get_main_tvm_function(file)
    if not source_code:
        return None
    try:
        
        func_def_start = source_code.index(target_function_name)
        
        
        body_start_index = source_code.index('{', func_def_start) + 1
        
        
        open_braces = 1
        body_end_index = -1
        for i in range(body_start_index, len(source_code)):
            if source_code[i] == '{':
                open_braces += 1
            elif source_code[i] == '}':
                open_braces -= 1
            
            if open_braces == 0:
                body_end_index = i
                break
        
        if body_end_index == -1:
            logging.error(f"Error: Could not find matching closing brace for function '{target_function_name}'.")
            return None

        
        function_body = source_code[body_start_index:body_end_index]
        
        
        
        function_call_pattern = re.compile(r'(\w+)\(')
        all_calls = function_call_pattern.findall(function_body)
        
        
        return Counter(all_calls)

    except ValueError:
        logging.error(f"Error: Target function '{target_function_name}' not found in the source code.")
        return None

def get_io_bytes(func) -> dict:
    """
    Calculates input/output bytes.
    Inputs are all buffers starting with 'p'.
    The output is the *first* buffer found after the 'p' buffers.
    Returns a dictionary.
    """
    total_input_bytes = 0
    total_output_bytes = 0
    
    
    def _calculate_bytes(buffer) -> int:
        
        itemsize = runtime.DataType(buffer.dtype).bits // 8
        
        
        num_elements = 1
        
        if not buffer.shape:
            return itemsize
            
        for dim_expr in buffer.shape:
            
            num_elements *= dim_expr.value
            
        return itemsize * num_elements

    
    processing_inputs = True
    for param_var in func.params:
        
        if param_var not in func.buffer_map:
            continue
            
        buffer = func.buffer_map[param_var]
        param_name = param_var.name

        if processing_inputs:
            if param_name.startswith("p"):
                
                total_input_bytes += _calculate_bytes(buffer)
            else:
                
                
                processing_inputs = False
                total_output_bytes += _calculate_bytes(buffer)
                
                
                
                break
        
    return {'in': slog(total_input_bytes), 'out': slog(total_output_bytes)}

def load_tir(tir_file):
    """ Returns a list of IR Mods from a tir dump in mlonmcu"""
    funct=[]
    if not os.path.isfile(tir_file):
        logging.error(f"TIR path {tir_file} is not a file")
        return funct
    with open(tir_file, "r") as f:
        content = f.read()


        funct.extend([x for x in content.split("# from tvm.script import tir as T") if x.strip()])

    objs = []
    for tir_source_code in funct:
        if not tir_source_code.strip():
            continue
        try:
            obj = tvm.script.from_source(tir_source_code)
        except:
            
            try: 
                pattern = r"T\.realize\s*\(([^()]*)\)"
                replacement = r'T.realize(\1, "global", True)'
                new_code = re.sub(pattern, replacement, tir_source_code)
                obj = tvm.script.from_source(new_code)
                
            except Exception as e:
                logging.error(f"Error replacing T.realize: {tir_source_code}\n{e}")
                raise e
            
        objs.append(obj)

    ret={}
    for obj in objs:
        func_name = obj.attrs["global_symbol"]
        if isinstance(obj, tvm.tir.PrimFunc):
            default_name = "main"
            obj = tvm.IRModule({default_name: obj})
        assert isinstance(obj, tvm.IRModule)
        ret[func_name] = obj
    return ret


def tir_tofeats_mlp(ir_mods,filter):
    """
    ir_mods: List of mods
    filter_mask: 
    """
    def combine_bufferstores(feats,toolchain_feats):
        X_ = np.concatenate([slog(exp2m(feats).sum(axis=0)),slog(np.array(toolchain_feats))],dtype=np.float32)
        return X_
        
    target = tvm.target.Target("c")
    
    func_name_to_feats = {}
    for name,mod in ir_mods.items():
        
        io_map = get_io_bytes(mod["main"])
        
        toolchain_feats_ = []
        f=mod["main"]
        const_bytes = tvm.tir.analysis.calculate_constant_bytes(f, 16)*(10**-3) 
                        
        workspace_bytes = tvm.tir.analysis.calculate_workspace_bytes(f, 16) * (10**-3) 
        
            
            
            
            
        
        
        toolchain_feats_.extend([const_bytes,workspace_bytes])
        
        
        tune_ctx = ms.tune_context.TuneContext(
            mod=mod,
            target=target,
            
            
            
            
            
            
        )
        
        sched = tvm.tir.Schedule(mod)
        candidate = ms.MeasureCandidate(sch=sched, args_info=[])

        extractor = ms.feature_extractor.PerStoreFeature(arith_intensity_curve_num_samples=0)
        (dummy_feature,) = extractor.extract_from(
            tune_ctx,
            candidates=[candidate],
        )
        dummy_feature = dummy_feature.numpy() 
        feats_per_func =combine_bufferstores(dummy_feature,toolchain_feats_)
        
        
        
        
        if filter:
            assert feats_per_func.shape[0] == len(filter),f"{feats_per_func.shape}, {len(filter)}"
            
            func_name_to_feats[name] = {"feat":feats_per_func[filter],**io_map}
        else:
            
            func_name_to_feats[name] = {"feat":feats_per_func,**io_map}
            
    
    
    
    return func_name_to_feats


def create_dgl_graph(nodes, edges):
    """
    Constructs a DGL graph from the parsed node and edge lists,
    including node features and edge weights.

    Args:
        nodes (list): A list of node dictionaries. Each node must have 'id' and 'feats'.
        edges (list): A list of edge dictionaries. Each edge must have 
                      'source', 'target', and 'weight'.

    Returns:
        dgl.DGLGraph: The constructed DGL graph object.
    """
    if not nodes:
        return dgl.graph(([], []))

    
    
    original_ids = sorted([node['id'] for node in nodes])
    id_map = {orig_id: new_id for new_id, orig_id in enumerate(original_ids)}
    
    num_nodes = len(nodes)
    
    
    node_features = [None] * num_nodes
    for node in nodes:
        new_id = id_map[node['id']]
        
        node_features[new_id] = node['feat'] 

    
    source_ids = np.array([id_map[edge['source']] for edge in edges])
    dest_ids = np.array([id_map[edge['target']] for edge in edges])
    
    
    edge_weights = np.array([edge['weight'] for edge in edges])
    
    
    
    g = dgl.graph((source_ids, dest_ids), num_nodes=num_nodes)
    
    
    g.ndata['feat'] = torch.tensor(np.array(node_features), dtype=torch.float32)
    
    
    g.edata['weight'] = torch.tensor(np.array(edge_weights), dtype=torch.float32)
    
    
    return g

def get_sw_flags(report_path=None, sw_dict={},post_run=False):
    data = []
    ret_data = ()
    toolchain_map = {'gcc': 0,"llvm":1}
    sw_opt_maps={"s":0,"1":1,"2":2,"3":3}
    
    sw_flag_data={}
    if not post_run and sw_dict:
        
        SW_FLAGS=["toolchain",'optimize', 'lto','garbage_collect']
        
        try:
            
            sw_flag_data['toolchain'] = toolchain_map[str(sw_dict['toolchain']).strip()]
            sw_flag_data['opt'] = sw_opt_maps[str(sw_dict['optimize']).strip()]
            sw_flag_data['gc'] = int(sw_dict['garbage_collect'])
            sw_flag_data['lto'] = int(sw_dict['lto'])
            ret_data = (int(sw_flag_data["toolchain"]),int(sw_flag_data["opt"]),int(sw_flag_data["gc"]),
                        int(sw_flag_data["lto"]))
        except Exception as e:
            logging.error(f"Error processing sw_flags data: {e}, data: {data}")
        return ret_data
    elif report_path and post_run: # Get SW flags from report.csv after the run
        try:
            df = pd.read_csv(report_path)
        except FileNotFoundError:
            logging.error(f"Report.csv not found at: {report_path}")
            return None
        except Exception as e:
            logging.error(f"Error reading {report_path}: {e}")
            return None

        sw_flags = ['mlif.toolchain', 'mlif.optimize', 'mlif.garbage_collect', 'mlif.lto']
        run_swflagdata = {}
        cols = ["Run","Config","ROM code","Runtime [s]","Run Instructions"]
        
        if not all(col in df.columns for col in cols):
            logging.warning(f"Missing required columns in {report_path}. Found columns: {df.columns.tolist()}")
            return None
        
        df=df.dropna(subset=cols)
        
        for run,config,rom_code,run_time,run_instr in df[["Run","Config","ROM code","Runtime [s]","Run Instructions"]].values:
            config_str = str(config)
            config_str = re.sub(r"PosixPath\(['\"](.*?)['\"]\)", r"'\1'", config_str)
            try:
                config = ast.literal_eval(config_str)
            except Exception as e:
                logging.error(f"Error parsing config string for run {run}: {e}\nString was: {config_str}")
                continue

            run = int(run)
            run_data={}
            try:
                sw_flag_data = {flag: config[flag] for flag in sw_flags }
                sw_flag_data['mlif.toolchain'] = toolchain_map[str(sw_flag_data['mlif.toolchain']).strip()]
                sw_flag_data['mlif.optimize'] = sw_opt_maps[str(sw_flag_data['mlif.optimize'])]
                ret_data = (int(sw_flag_data["mlif.toolchain"]),int(sw_flag_data["mlif.optimize"]),
                            int(sw_flag_data["mlif.garbage_collect"]),int(sw_flag_data["mlif.lto"]))
                
                run_data["rom_code"] = rom_code
                run_data["run_time"] = run_time
                run_data["run_instr"] = run_instr
                run_data["sw_flags"] = ret_data
                run_swflagdata[run] = run_data
                
            except Exception as e:
                logging.error(f"Error processing data for run {run}: {e}, data: {config}")
        return run_swflagdata

def parse_graph_from_c_code(file, map_fname_to_feats):
    """
    Parses TVM-generated C code to extract a strictly linear computation graph
    based on the order of function calls.
    """
    pattern = re.compile(
        r"if \((tvmgen_default_[a-zA-Z0-9_]+)\((.*?)\)\s*!=\s*0\s*\)"
    )
    c_code = get_main_tvm_function(file) 
    if not c_code:
        logging.error(f"Could not get main function from {file}")
        return [], []
        
    matches = pattern.findall(c_code)
    if not matches:
        return [], []

    nodes = []
    edges = []
    node_counter = 0

    if not map_fname_to_feats:
        raise ValueError("map_fname_to_feats cannot be empty.")
        
    
    feat_dim = len(list(map_fname_to_feats.values())[0]["feat"])
    io_node_feats = np.array([0.0] * feat_dim)

    
    input_node_id = -1
    nodes.append({
        "id": input_node_id,
        "name": "Input",
        "feat": io_node_feats 
    })
    edges.append({
        "source": input_node_id,
        "target": input_node_id,
        "label": "input_self_loop",
        "weight": 0
    })

    
    previous_node_id = input_node_id
    previous_node_name = "Input"

    
    for func_name, _ in matches:  
        
        current_node_id = node_counter
        
        
        node_features = map_fname_to_feats.get(func_name)
        if node_features is None:
            raise ValueError(f"Function {func_name} not found in feature map.")

        nodes.append({
            "id": current_node_id,
            "name": func_name,
            "feat": node_features["feat"] 
        })
        
        
        edge_weight = 0
        edge_label = ""
        if previous_node_id == input_node_id:
            
            
            edge_weight = node_features.get('in', 0)
            edge_label = "model_input"
        else:
            
            
            prev_node_features = map_fname_to_feats.get(previous_node_name)
            edge_weight = prev_node_features.get('out', 0)
            edge_label = f"{previous_node_name}_output"
        
        
        edges.append({
            "source": previous_node_id,
            "target": current_node_id,
            "label": edge_label,
            "weight": edge_weight
        })
        
        
        previous_node_id = current_node_id
        previous_node_name = func_name
        node_counter += 1

    
    output_node_id = node_counter
    nodes.append({
        "id": output_node_id,
        "name": "Output",
        "feat": io_node_feats
    })
    edges.append({
        "source": output_node_id,
        "target": output_node_id,
        "label": "output_self_loop",
        "weight": 0
    })
    
    
    
    
    final_edge_weight = 0
    if previous_node_name != "Input": 
        last_op_features = map_fname_to_feats.get(previous_node_name)
        final_edge_weight = last_op_features.get('out', 0)
            
    edges.append({
        "source": previous_node_id, 
        "target": output_node_id,
        "label": "model_output",
        "weight": final_edge_weight
    })
        
    return nodes, edges

def extract_graph_in_run(c_file, tir_file, sw_feats,filter_rows):
    
    mods = load_tir(tir_file)
    if not mods:
        logging.warning(f"No TIR mods loaded from {tir_file}")
        return None
    func_name_to_feat = tir_tofeats_mlp(mods, filter_rows)
    nodes, edges = parse_graph_from_c_code(c_file, func_name_to_feat)
    g = create_dgl_graph(nodes, edges)

def extract_graph_from_session(session_path, filter_rows):
    X_path = []
    y_path = []
    global_sw_flags_path = []
    results = []
    run_order=[]
    def process_run_item(item, session_path, tir_filter):
        """
        Processes a single (k, v) item from the sw_flags dictionary.
        
        Returns a tuple (graph, y_data, sw_flag) on success, or None on failure.
        """
        
        k, v = item
        run_path = os.path.join(session_path,"runs", str(k))
        
        try:
            if not os.path.isdir(run_path):
                logging.warning(f"Directory not found, skipping: {run_path}")
                return None

            tir_file = [os.path.join(run_path, "default.tir")]
            
            c_files_dir = os.path.join(run_path, "codegen", "host", "src")
            c_files = os.listdir(c_files_dir)
            
            if len(c_files) == 3:
                c_file_path = f"codegen/host/src/default_lib2.c"
            else:
                c_file_path = f"codegen/host/src/default_lib1.c"
            c_file = os.path.join(run_path, c_file_path)
            
            mods = load_tir(tir_file)
            if not mods:
                logging.warning(f"No TIR mods loaded from {run_path}")
                return None
                
            func_name_to_feat = tir_tofeats_mlp(mods, tir_filter)
            
            nodes, edges = parse_graph_from_c_code(c_file, func_name_to_feat)
            if not nodes:
                logging.warning(f"No graph nodes parsed from {c_file}")
                return None

            g = create_dgl_graph(nodes, edges)
            
            
            y_data = [float(v["run_instr"]), float(v["rom_code"]), float(v["run_time"])]
            sw_flag_data = v["sw_flags"]
            
            
            return (g, y_data, sw_flag_data)
        
        except Exception as e:
            
            logging.error(f"Error processing run {run_path}: {e}", exc_info=True)
            return None

    report_csv = os.path.join(session_path, "report.csv")
    target_function = "tvmgen_default___tvm_main__"
    
    sw_flags = get_sw_flags(report_csv, post_run=True)
    if not sw_flags:
        logging.warning(f"Skipping session {session_path} due to missing or invalid SW flags.")
        
        return X_path, y_path, global_sw_flags_path,run_order 

    
    try:
        for item in sw_flags.items():
            
            result = process_run_item(item, session_path, filter_rows)
            
            if result is not None:
                
                g, y_data, sw_flag_data = result
                X_path.append(g)
                y_path.append(y_data)
                global_sw_flags_path.append(sw_flag_data)
                run_order.append(item[0])
        
        logging.info(f"Successfully processed {len(X_path)} items from session {session_path}.")
        
    except Exception as e:
        
        logging.error(f"Error processing session {session_path}: {e}", exc_info=True)
        
    
    
    return X_path, y_path, global_sw_flags_path,run_order


def process_path(path, filter):
    import time
    """
    Worker function to process a single session path.
    This replaces the body of the loop in get_graph_from_session.
    """
    X_path = []
    y_path = []
    global_sw_flags_path = []

    report_csv = os.path.join(path, "report.csv")
    target_function = "tvmgen_default___tvm_main__"
    
    sw_flags = get_sw_flags(report_csv, post_run=True)
    if not sw_flags:
        logging.warning(f"Skipping session {path} due to missing or invalid SW flags.")
        
        return X_path, y_path, global_sw_flags_path 

    try:
        for k, v in sw_flags.items():
            start_time = time.time()
            run_path = os.path.join(path, str(k))
            if os.path.isdir(run_path):
                try:
                    tir_file = os.path.join(run_path, "tir", "default.tir")
                    c_files = glob(os.path.join(run_path, "cfile", "*.c"))
                    if not c_files:
                        logging.warning(f"No .c file found in {run_path}")
                        continue
                    c_file = c_files[0] 
                    
                    mods = load_tir(tir_file)
                    if not mods:
                        logging.warning(f"No TIR mods loaded from {run_path}")
                        continue
                        
                    func_name_to_feat = tir_tofeats_mlp(mods, filter)
                    
                    nodes, edges = parse_graph_from_c_code(c_file, func_name_to_feat)
                    if not nodes:
                        logging.warning(f"No graph nodes parsed from {c_file}")
                        continue

                    g = create_dgl_graph(nodes, edges)
                    
                    X_path.append(g)
                    y_path.append([float(sw_flags[k]["run_instr"]), float(sw_flags[k]["rom_code"]), float(sw_flags[k]["run_time"])])
                    global_sw_flags_path.append(v["sw_flags"])
                    end_time = time.time()
                    logging.info(f"Processed run {k} in {end_time - start_time:.2f} seconds.")
                except Exception as e:
                    
                    logging.error(f"Error processing run {run_path}: {e}", exc_info=True)
                    continue
    except Exception as e:
        
        logging.error(f"Error processing session {path}: {e}", exc_info=True)
    
    
    return X_path, y_path, global_sw_flags_path

def extract_graph_parallel(paths, filter_rows,num_cores=os.cpu_count() or 1):
    
    logging.info(f"Using {num_cores} cores for processing.")
    
    worker_func = partial(process_path, filter=filter_rows)
    
    X = []
    y = []
    global_sw_flags = []

    logging.info("Starting parallel processing...")
    with Pool(processes=num_cores) as pool:
        
        results = list(tqdm(
            pool.imap_unordered(worker_func, paths), 
            total=len(paths),
            desc="Processing sessions"
        ))

    logging.info("Parallel processing finished. Aggregating results...")
    
    
    
    for res_tuple in results:
        X_path, y_path, global_sw_flags_path = res_tuple
        X.extend(X_path)
        y.extend(y_path)
        global_sw_flags.extend(global_sw_flags_path)
    
    logging.info(f"Aggregated a total of {len(X)} graphs.")
    
    if not y:
        logging.warning("No data was successfully processed. Output file will be empty.")
        sys.exit(0)

    y_np = np.array(y,dtype=np.float32)
    global_sw_flags_np = np.array(global_sw_flags)

    return X, y_np, global_sw_flags_np

if __name__ == "__main__":
    
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        filename='extract_graph.log',
        filemode='w' 
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO) 
    logging.getLogger('').addHandler(console)

    logging.info("Starting graph extraction process...")

    
    filter_rows_selection=["gpu_feats","vect_loop","float_ops"]

    
    total_size = sum(features.values()) 
    logging.info(f"Total feature size: {total_size}")
    
    dummy_features = np.arange(total_size) 

    
    _,filter_rows = map_features_to_values(features,dummy_features ,filter_rows_selection)
    
    graph_path = f"/nfs/TUEIEDAscratch/ge85zic/mlonmcu_env/temp/graph_data_funcs"
    paths = glob(os.path.join(graph_path, "*"))
    
    if not paths:
        logging.warning("No paths found. Exiting.")
        sys.exit(0)
        
    logging.info(f"Found {len(paths)} paths to process.")
    
    last_path = paths[-1] 
    X, y_np, global_sw_flags_np = extract_graph_parallel(paths, filter_rows)
    now = strftime("%Y%m%d_%H%M%S", gmtime())
    output_file = f"/nfs/TUEIEDAscratch/ge85zic/graph_regressor/data/graph_data_feats_run_instr_run_time_{now}.pkl"
    logging.info(f"Saving data to {output_file}...")
    with open(output_file,"wb") as f:
        pickle.dump((X, y_np, global_sw_flags_np, filter_rows,last_path), f)

    logging.info("All done.")