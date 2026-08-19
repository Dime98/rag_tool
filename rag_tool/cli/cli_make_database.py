import argparse
import sys

from rag_tool.core.build_vector_store import build_vector_store


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        type=str,
        required=True,
        help="Path of configs for creating vector db.",
    )
    parser.add_argument(
        "-save-to",
        type=str,
        required=True,
        help="Output directory for experiment.",
    )
    parser.add_argument("-store-stats", action="store_true", help="Saves statistics.")
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    build_vector_store(config_path=args.config, save_to=args.save_to, store_stats=args.store_stats)
