# AOI-PCB: Automated Optical Inspection for PCB Assembly

A deep learning system for detecting IC component misplacement on printed circuit boards, implementing the approach described in:

> *Automated Optical Inspection for Printed Circuit Board Assembly Manufacturing with Transfer Learning and Synthetic Data Generation* — [IEEE](https://ieeexplore.ieee.org/document/9837280)

## Overview

IC misplacement is a common defect in PCB assembly. This system localizes an IC by predicting the coordinates of its four corner keypoints in a PCB image, framing the problem as a regression task rather than classification. The key contributions are:

- **Synthetic training data**: No hand-labeled real images are needed. A reference IC is blended onto real PCB background images with randomized rotation and positional offset, producing 2000 training and 300 validation images.
- **Custom perpendicularity loss**: In addition to coordinate MSE, a penalty term enforces that the four predicted edge vectors form right angles — encouraging the model to output geometrically valid rectangles.
- **Lightweight deployment target**: A frozen MobileNetV2 backbone (pre-trained on ImageNet) is used for feature extraction. Only the small custom head is trained, keeping the parameter count low.

## Model Architecture

```
Input (256×256×3)
  → GaussianNoise(σ=0.1)
  → MobileNetV2 [frozen, ImageNet weights]
  → Dropout(0.3)
  → SeparableConv2D(8 filters, 5×5, ReLU)
  → Flatten
  → Dense(512, ReLU)
  → Dropout(0.1)
  → Dense(8)          ← 4 corners × (x, y), normalised to [0, 1]
```

## Results

Training stabilizes around epoch 600. On the validation set:

- Combined loss: ~0.0001
- Center position MAE: ~0.01 (normalized coordinates)
- The model reliably detects the IC footprint within the tolerance required for assembly inspection.

## Installation

Requires Python ≥ 3.10, pip > 19.0 (> 20.3 on macOS), and TensorFlow 2.18.

```bash
git clone <repo>
cd aoi-pcb-v1

# CPU only
pip install -e ".[dev]"

# Apple Silicon GPU (tensorflow-metal)
pip install -e ".[dev,metal]"

# Linux / WSL2 CUDA GPU
pip install -e ".[dev,cuda]"

# With notebook dependencies (matplotlib, jupyterlab)
pip install -e ".[dev,notebooks]"
```

## Usage

### 1. Generate the dataset

```bash
# Use default config.json
python scripts/generate_dataset.py

# Use a custom config
python scripts/generate_dataset.py --config path/to/config.json
```

Creates `datasets/training/` and `datasets/validation/` with PNG images and CSV label files. Source images (`pcb_images/ic.png`, `pcb_images/pcb_backlayer.png`) are committed to the repo.

### 2. Train

```bash
# Use default config.json; save to experiments/run_YYYYMMDD_HHMMSS/
python scripts/train.py

# Use a custom config and output directory
python scripts/train.py --config path/to/config.json --output-dir experiments/my_run
```

Each run saves `model.keras`, a training log CSV, and a `config.json` snapshot to its output directory.

### 3. Evaluate

```bash
# Auto-detect the most recently modified run
python scripts/evaluate.py

# Evaluate a specific run
python scripts/evaluate.py --model-path experiments/run_YYYYMMDD_HHMMSS/model.keras

# Save prediction overlay images
python scripts/evaluate.py --save-visuals --n-visuals 20
```

The config is loaded automatically from the run directory. Pass `--config` to override.

### Notebooks

Interactive walkthroughs are in `notebooks/`:
- `training.ipynb` — data generation → model architecture → training loop → loss curves
- `evaluation.ipynb` — load model → run predictions → visualise keypoint overlays → per-sample error histogram

## Configuration

All parameters are in `config.json`:

| Section     | Key parameters                                                                     |
|-------------|------------------------------------------------------------------------------------|
| `generator` | `dataset_size`, `rotation_angle`, `delta`, `seed`, output dirs, source image paths |
| `encoder`   | `normalize_data`, `normalize_labels`, `train_data_splice`                          |
| `training`  | `optimizer_lr`, `n_epochs`, `early_stopping.*`, `lr_schedule.*`                    |
| `metrics`   | `x_weight`, `y_weight`, `angle_weight`                                             |

## Testing

```bash
pytest tests/
```

The test suite covers the config loader, data utilities, encoder (including error branches via mocks), model architecture, custom loss, and alignment metric. `data/generator.py` is excluded from unit tests — it requires the source images and is validated end-to-end by running `generate_dataset.py`.

## Project Structure

```
aoi-pcb-v1/
├── config.json
├── pyproject.toml
├── pcb_images/              # Source images for synthesis (committed)
├── datasets/                # Generated data (gitignored, populate with generate_dataset.py)
├── experiments/             # Training runs (gitignored)
├── src/aoi_pcb/
│   ├── config_loader.py
│   ├── data/
│   │   ├── generator.py
│   │   ├── encoder.py
│   │   └── utils.py
│   └── model/
│       ├── architecture.py
│       ├── loss.py
│       └── metric.py
├── scripts/
│   ├── generate_dataset.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   ├── training.ipynb
│   └── evaluation.ipynb
└── tests/
```
