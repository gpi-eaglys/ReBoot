import argparse
from typing import Any

from lib.utils.enums import Dataset


def get_parser_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument("dataset", type=str, help=[dataset.name for dataset in Dataset])
    parser.add_argument("--subsample", type=int, default=None, help="Subsample the training dataset")
    parser.add_argument("--resize", type=int, default=None, help="Resize image datasets")
    # Network arguments
    parser.add_argument("--num-layers", type=int, default=2, help="Number of fully connected layers")
    parser.add_argument("--num-hidden", type=int, nargs="*", default=[32], help="Number of hidden units per layer")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay regularization")
    # Custom dataset arguments
    parser.add_argument("--input-dim", type=int, default=32, help="Input dimension for the custom dataset")
    parser.add_argument("--output-dim", type=int, default=32, help="Output dimension for the custom dataset")
    # Run arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of runs")
    return parser.parse_args()
