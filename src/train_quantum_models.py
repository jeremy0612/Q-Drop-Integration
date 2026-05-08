"""Canonical multi-dataset Torch graph trainer with Q-Drop support."""

import os
import sys

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.graph_training import build_train_parser, config_from_args, run_experiments


def parse_args():
    parser = build_train_parser(
        description="Train quantum models on MUTAG/PROTEINS",
        default_datasets=["mutag", "proteins"],
    )
    return config_from_args(parser.parse_args())


def main() -> None:
    config = parse_args()
    run_experiments(config)


if __name__ == "__main__":
    main()
