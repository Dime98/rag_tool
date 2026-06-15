import argparse
import sys

from rag_tool.core.chat import chat_loop


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    chat_loop(args.config)
