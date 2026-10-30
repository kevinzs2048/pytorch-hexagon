"""Select a model adapter without importing its framework at startup."""
import argparse
import importlib


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hexagon_runtime", description="PyTorch models on Hexagon DSP")
    parser.add_argument("model", choices=("qwen3",), help="Model adapter")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Adapter command and options")
    args = parser.parse_args(argv)
    importlib.import_module("models." + args.model + ".cli").main(args.arguments)


if __name__ == "__main__":
    main()
