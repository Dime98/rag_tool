import argparse
import sys
from pathlib import Path

from rag_tool.core.benchmark import run_benchmark


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-benchmark-config",
        type=str,
        required=True,
        help="Path of benchmark config json.",
    )
    parser.add_argument(
        "-experiments-folder",
        type=str,
        required=True,
        help="Folder with experiments output.",
    )
    parser.add_argument("-save-to", type=str, required=True, help="Path to save benchmarking results.")
    parser.add_argument(
        "-experiment-summary",
        action="store_false",
        help="Summary of experiments configs.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run_benchmark(
        benchmark_config_path=Path(args.benchmark_config),
        experiments_folder=Path(args.experiments_folder),
        printout_experiments_summary=args.experiment_summary,
        save_to=Path(args.save_to),
    )
