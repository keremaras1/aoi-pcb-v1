"""CLI script to generate synthetic PCB training and validation datasets.

Reads all parameters from config.json (or a custom path via --config) and
calls the data generator for whichever splits are enabled.

Usage::

    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --config path/to/config.json
"""

import argparse
import warnings

from aoi_pcb.config_loader import Config
from aoi_pcb.data.generator import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PCB datasets from config.json."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="libpng warning: iCCP: known incorrect sRGB profile")

    config = Config(args.config)
    backlayer_path = config.generator.image_sources.backlayer_path
    ic_path = config.generator.image_sources.ic_path

    if config.generator.gen_train:
        print("Generating training data...")
        train_args = config.get_init_kwargs("generator.train_data")
        generate_dataset(**train_args, backlayer_path=backlayer_path, ic_path=ic_path)
        print("Training data generation complete.")

    if config.generator.gen_val:
        print("Generating validation data...")
        val_args = config.get_init_kwargs("generator.val_data")
        generate_dataset(**val_args, backlayer_path=backlayer_path, ic_path=ic_path)
        print("Validation data generation complete.")


if __name__ == "__main__":
    main()
