"""CLI script for evaluating a trained IC keypoint detection model.

Loads a saved model, runs evaluation on the validation dataset, and
optionally saves visualisation images showing predicted vs actual keypoint
overlays on each PCB image.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --save-visuals
    python scripts/evaluate.py --model-path experiments/run_YYYYMMDD_HHMMSS/model.keras
    python scripts/evaluate.py --model-path experiments/run_YYYYMMDD_HHMMSS/model.keras --config path/to/other_config.json
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from aoi_pcb.config_loader import Config
from aoi_pcb.data.encoder import DataEncoder
from aoi_pcb.model.loss import custom_loss
from aoi_pcb.model.metric import KeypointAlignmentMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained IC keypoint detection model."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to the saved model file (.keras). "
            "Defaults to the most recently modified run in experiments/."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a JSON configuration file. "
            "Defaults to config.json inside the run directory."
        ),
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save prediction overlay images to <model-path>/visuals/.",
    )
    parser.add_argument(
        "--n-visuals",
        type=int,
        default=15,
        help="Number of prediction overlay images to save (default: 15).",
    )
    return parser.parse_args()


def draw_keypoints(
    image: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    img_width: int,
    normalized: bool,
) -> np.ndarray:
    """Draw actual (red) and predicted (blue) keypoint circles onto an image.

    Args:
        image: HxWx3 uint8 image array.
        actual: Flattened array of 8 ground-truth coordinates.
        predicted: Flattened array of 8 predicted coordinates.
        img_width: Image width; used to rescale normalized coordinates.
        normalized: Whether coordinates are in [0, 1] and need rescaling.

    Returns:
        Image array with circles drawn on it.
    """
    if normalized:
        actual = actual * img_width
        predicted = predicted * img_width

    corners_actual = np.rint(actual.reshape(-1, 2)).astype("int32")
    corners_predicted = np.rint(predicted.reshape(-1, 2)).astype("int32")

    img = image.copy()
    for corner in corners_actual:
        img = cv2.circle(img, tuple(corner), 4, (255, 0, 0), 2)   # red — ground truth
    for corner in corners_predicted:
        img = cv2.circle(img, tuple(corner), 4, (0, 0, 255), 2)   # blue — prediction

    return img


def main() -> None:
    args = parse_args()

    # Resolve model path — default to the most recently modified run
    if args.model_path:
        model_path = args.model_path
    else:
        runs = sorted(Path("experiments").glob("*/model.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise FileNotFoundError("No trained model found in experiments/. Run train.py first.")
        model_path = str(runs[0])
    print(f"Model: {model_path}")

    config_path = args.config or str(Path(model_path).parent / "config.json")
    config = Config(config_path)
    print(f"Config: {config_path}")

    # --- Load data ---
    encoder = DataEncoder(config)
    data_dir = config.generator.val_data.data_dir
    label_path = (
        Path(config.generator.val_data.labels_dir) / config.generator.val_data.label_file
    )

    X, y, ref_coords, ref_center = encoder(data_dir, label_path)
    print(f"Validation data shape: {X.shape}, Labels shape: {y.shape}")

    # --- Load model ---
    # compile=False skips Keras's attempt to deserialise the saved compile config,
    # which fails for @tf.function-decorated losses. We re-compile immediately after.
    metric = KeypointAlignmentMetric(ref_center, ref_coords, config)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss=custom_loss, metrics=[metric.alignment_metric])
    model.summary()

    # --- Evaluate ---
    results = model.evaluate(X, y, verbose=1)
    print("\nEvaluation results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.6f}")

    # --- Predict ---
    predictions = model.predict(X)
    print(f"\nPredictions shape: {predictions.shape}")

    # --- Save visualisations ---
    if args.save_visuals:
        visuals_dir = Path(model_path).parent / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        n = min(args.n_visuals, len(X))

        for idx in range(n):
            img = draw_keypoints(
                X[idx].copy(),
                actual=y[idx],
                predicted=predictions[idx],
                img_width=X.shape[1],
                normalized=config.encoder.normalize_labels,
            )
            out_path = visuals_dir / f"pred_{idx:04d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"Saved {n} visualisation images to {visuals_dir}")


if __name__ == "__main__":
    main()
