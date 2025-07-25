import os
import argparse
import multiprocessing
import logging

# import mlonmcu
from mlonmcu.context.context import MlonMcuContext
from mlonmcu.session.run import RunStage
from mlonmcu.session.postprocess.postprocess import SessionPostprocess
from mlonmcu.models.lookup import apply_modelgroups
from mlonmcu.logging import get_logger, set_log_level

logger = get_logger()


class CustomPostprocess(SessionPostprocess):  # RunPostprocess?
    """TODO"""

    def __init__(self, features=None, config=None):
        super().__init__("custom", features=features, config=config)

    def post_session(self, report):
        """Called at the end of a session."""
        # df = report.post_df.copy()
        # report.post_df = df


FRONTEND = "tflite"

TARGETS = [
    "spike",
    "host_x86",
    "etiss",
    "corstone300",
]

DEFAULT_TARGETS = [
    # "spike",
    # "host_x86",
    "etiss",
    "corstone300",
]

PLATFORM = "mlif"

BACKENDS = [
    "tvmaot",
]
DEFAULT_BACKENDS = [
    "tvmaot",
]

FEATURES = [
    "unpacked_api",
    "usmp",
    "none",
]

DEFAULT_FEATURES = [
    "unpacked_api",
    "usmp",
]


VALIDATE_FEATURES = ["validate", "debug"]


def get_backend_features(backend, enable_default=True, enable_unpacked_api=True, enable_usmp=True):
    backend_features = {
        "tvmaot": [
            [],
            *([["unpacked_api"]] if enable_unpacked_api else []),
            *([["usmp"]] if enable_usmp else []),
            *([["unpacked_api", "usmp"]] if enable_unpacked_api and enable_usmp else []),
        ],
    }
    return backend_features[backend]


def get_backend_config(backend, features):
    return (
        [
            {"usmp.algorithm": "greedy_by_size"},
            {"usmp.algorithm": "greedy_by_conflicts"},
            {"usmp.algorithm": "hill_climb"},
        ]
        if "usmp" in features
        else [{}]
    )


DEFAULT_CONFIG = {
    "mlif.num_threads": 4,
}

BACKEND_DEFAULT_CONFIG = {
    "tvmaot": {},
}

MODELS = [
    "sine_model",
    # "magic_wand",
    # "micro_speech",
    # "cifar10",
    # "simple_mnist",
    "aww",
    "vww",
    "resnet",
    "toycar",
]

POSTPROCESSES_0 = [
    "features2cols",
    "config2cols",
]

POSTPROCESSES_1 = [
    "rename_cols",
    "filter_cols",
]

POSTPROCESS_CONFIG = {
    "filter_cols.keep": [
        "Model",
        "Backend",
        "Target",
        "Cycles",
        "Runtime [s]",
        "Total ROM",
        "Total RAM",
        "ROM read-only",
        "ROM code",
        "ROM misc",
        "RAM data",
        "RAM zero-init data",
        "Incomplete",
        "Failing",
        "Features",
        # "Comment",
        "Validation",
        "Kernels",
        "Unpacked API",
        "USMP",
        "Algorithm",
        "RAM stack",
        "RAM heap",
    ],
    "rename_cols.mapping": {
        "feature_unpacked_api": "Unpacked API",
        "feature_usmp": "USMP",
        "config_usmp.algorithm": "Algorithm",
    },
    "filter_cols.drop_nan": True,
}


def gen_features(backend, features, validate=False):
    ret = []
    if validate:
        ret += VALIDATE_FEATURES
    ret += features
    return ret


def gen_config(backend, backend_config, features, enable_postprocesses=False):
    ret = {}
    ret.update(DEFAULT_CONFIG)
    ret.update(BACKEND_DEFAULT_CONFIG[backend])
    ret.update(backend_config)
    if enable_postprocesses:
        ret.update(POSTPROCESS_CONFIG)
    return ret


def benchmark(args):
    with MlonMcuContext() as context:
        user_config = context.environment.vars  # TODO: get rid of this workaround
        session = context.create_session()
        models = apply_modelgroups(args.models, context=context)
        for model in models:
            for backend in args.backend:
                for target in args.target:
                    enable_default = not args.skip_default
                    enable_unpacked_api = "unpacked_api" in args.feature
                    enable_usmp = "usmp" in args.feature
                    for backend_features in get_backend_features(
                        backend,
                        enable_default=enable_default,
                        enable_unpacked_api=enable_unpacked_api,
                        enable_usmp=enable_usmp,
                    ):
                        for backend_config in get_backend_config(backend, backend_features):
                            features = gen_features(backend, backend_features, validate=args.validate)
                            config = gen_config(backend, backend_config, features, enable_postprocesses=args.post)
                            config.update(user_config)  # TODO
                            print("RUN", model, config, features, backend, target)
                            run = session.create_run(config=config)
                            run.add_features_by_name(features, context=context)
                            run.add_platform_by_name(PLATFORM, context=context)
                            run.add_frontend_by_name(FRONTEND, context=context)
                            run.add_model_by_name(model, context=context)
                            run.add_backend_by_name(backend, context=context)
                            run.add_target_by_name(target, context=context)
                            if args.post:
                                run.add_postprocesses_by_name(POSTPROCESSES_0)
                                run.add_postprocesses_by_name(POSTPROCESSES_1, append=True)
        if args.noop:
            stage = RunStage.LOAD
        else:
            stage = RunStage.RUN
        session.process_runs(until=stage, num_workers=args.parallel, progress=args.progress, context=context)
        report = session.get_reports()
        report_file = args.output
        report.export(report_file)
        print()
        print(report.df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "models",
        metavar="model",
        type=str,
        # nargs="+",
        nargs="*",
        default=MODELS,
        help="Model to process",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        action="append",
        choices=BACKENDS,
        # default=DEFAULT_BACKENDS,
        default=[],
        help=f"Backends to use (default: {DEFAULT_BACKENDS})",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        action="append",
        choices=TARGETS,
        # default=DEFAULT_TARGETS,
        default=[],
        help=f"Targets to use (default: {DEFAULT_TARGETS}s)",
    )
    parser.add_argument(
        "-f",
        "--feature",
        type=str,
        action="append",
        choices=FEATURES,
        # default=default_features,
        default=[],
        help=f"features to use (default: {DEFAULT_FEATURES})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-default",
        dest="skip_default",
        action="store_true",
        help="Do not generate benchmarks for reference runs (default: %(default)s)",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Run postprocesses after the session (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="Display progress bar (default: %(default)s)",
    )
    parser.add_argument(
        "--parallel",
        metavar="THREADS",
        nargs="?",
        type=int,
        const=multiprocessing.cpu_count(),
        default=1,
        help="Use multiple threads to process runs in parallel (%(const)s if specified, else %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        type=str,
        default=os.path.join(os.getcwd(), "out.csv"),
        help="""Output CSV file (default: %(default)s)""",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed messages for easier debugging (default: %(default)s)",
    )
    parser.add_argument(
        "--noop",
        action="store_true",
        help="Skip processing runs. (default: %(default)s)",
    )
    args = parser.parse_args()
    if not args.backend:
        args.backend = DEFAULT_BACKENDS
    if not args.target:
        args.target = DEFAULT_TARGETS
    if not args.feature:
        args.feature = DEFAULT_FEATURES
    if "none" in args.feature:
        assert len(args.feature) == 1
        args.feature = []
    if args.verbose:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.INFO)
    benchmark(args)


if __name__ == "__main__":
    main()
