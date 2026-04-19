"""CLI script for training the IC keypoint detection model.

Loads data, builds the MobileNetV2-based model, and runs the training loop
with early stopping and learning rate scheduling as configured in config.json.
The trained model and training logs are saved to the output directory.

Usage::

    python scripts/train.py
    python scripts/train.py --config path/to/config.json --output-dir experiments/run_1
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import callbacks

from aoi_pcb.config_loader import Config
from aoi_pcb.data.encoder import DataEncoder
from aoi_pcb.model.architecture import build_model
from aoi_pcb.model.loss import custom_loss
from aoi_pcb.model.metric import KeypointAlignmentMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the IC keypoint detection model."
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save the trained model and logs. "
            "Defaults to experiments/run_<timestamp>."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(args.config)

    # Resolve output directory
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("experiments") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Report available GPUs
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")

    # --- Data loading ---
    encoder = DataEncoder(config)
    data_dir = config.generator.train_data.data_dir
    label_path = Path(config.generator.train_data.labels_dir) / config.generator.train_data.label_file

    X, y, ref_coords, ref_center = encoder(data_dir, label_path)
    print(f"Training data shape: {X.shape}, Labels shape: {y.shape}")
    print(f"Reference coords: {ref_coords}")
    print(f"Reference center: {ref_center}")

    # --- Model setup ---
    input_shape = X.shape[1:]
    output_shape = y.shape[1]

    metric = KeypointAlignmentMetric(ref_center, ref_coords, config)
    model = build_model(input_shape, output_shape)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.training.optimizer_lr),
        loss=custom_loss,
        metrics=[metric.__call__],
    )

    # --- Callbacks ---
    es_args = config.get_init_kwargs("training.early_stopping")
    lr_args = config.get_init_kwargs("training.lr_schedule")
    log_csv_args = config.get_init_kwargs("training.log_csv")

    log_csv_args["filename"] = str(output_dir / log_csv_args.pop("file_name"))

    training_callbacks = [
        callbacks.EarlyStopping(**es_args),
        callbacks.ReduceLROnPlateau(**lr_args),
        callbacks.CSVLogger(**log_csv_args),
    ]

    # --- Training ---
    history = model.fit(
        X,
        y,
        validation_split=config.training.val_split,
        epochs=config.training.n_epochs,
        callbacks=training_callbacks,
    )

    # --- Save model ---
    model_path = output_dir / "model.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # --- Summary ---
    final_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    print(f"Final train loss: {final_loss:.6f} | val loss: {final_val_loss:.6f}")
    print(f"Trained for {len(history.history['loss'])} epochs.")


if __name__ == "__main__":
    main()
