# AOI-PCB: Automated Optical Inspection for PCB Assembly

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![CI](https://github.com/keremaras/aoi-pcb-v1/actions/workflows/ci.yml/badge.svg)](https://github.com/keremaras/aoi-pcb-v1/actions/workflows/ci.yml)

A deep learning system for detecting IC component misplacement on printed circuit boards, implementing the approach described in:

> *Automated Optical Inspection for Printed Circuit Board Assembly Manufacturing with Transfer Learning and Synthetic Data Generation* — Saif, Aras & Giuseppi, MED 2022 — [IEEE Xplore](https://ieeexplore.ieee.org/document/9837280)

## Overview

IC misplacement is one of the most common defects in PCB assembly: a component lands at the wrong position or rotation, causing electrical failures that are costly to catch downstream. Traditional AOI systems rely on large hand-labeled image datasets that are expensive to collect and tightly coupled to specific board designs.

This project reframes the problem as a **keypoint regression task**: given a PCB image, predict the (x, y) coordinates of all four corners of the IC bounding box. Three design choices make this practical without real labeled data:

- **Synthetic training data** — A reference IC is blended onto real PCB background images with randomised rotation and offset, generating 2,000 training and 2,000 validation images programmatically. No hand-labelling required.
- **Custom perpendicularity loss** — In addition to coordinate MSE, a penalty term enforces that the four predicted corner vectors form right angles, encouraging geometrically valid (rectangular) outputs.
- **Frozen MobileNetV2 backbone** — Pre-trained ImageNet features are used as-is. Only the small custom regression head is trained, keeping the parameter count low and training fast.

## Pipeline

```mermaid
flowchart LR
  A["ic.png +<br/>pcb_backlayer.png"] --> B["Generator<br/>rotate + blend"]
  B --> C["Synthetic<br/>Dataset"]
  C --> D["DataEncoder"]
  D --> E["MobileNetV2<br/>frozen"]
  E --> F["Custom Head<br/>SepConv + Dense"]
  F --> G["8 keypoints"]
  G --> H["Custom Loss<br/>MSE + Perp"]
  H --> I["Trained<br/>model.keras"]
  I --> J["Evaluator<br/>overlay + metrics"]
```

| Stage | Source |
|-------|--------|
| Generator | `src/aoi_pcb/data/generator.py` |
| DataEncoder | `src/aoi_pcb/data/encoder.py` |
| MobileNetV2 + Custom Head | `src/aoi_pcb/model/architecture.py` |
| Custom Loss / Metric | `src/aoi_pcb/model/loss.py`, `metric.py` |
| Evaluator | `scripts/evaluate.py`, `notebooks/evaluation.ipynb` |

## Model Architecture

```
Input (256×256×3, uint8)
  → GaussianNoise(σ=0.1)
  → preprocess_input        [0, 255] → [−1, 1]
  → MobileNetV2 [frozen, ImageNet weights]
  → Dropout(0.3)
  → SeparableConv2D(8 filters, 5×5, ReLU)
  → Flatten
  → Dense(512, ReLU)
  → Dropout(0.1)
  → Dense(8)                ← 4 corners × (x, y), normalised to [0, 1]
```

## Results

Training stabilises around epoch 600. On the validation set:

- Combined loss: ~0.0001
- Center position MAE: ~0.01 (normalised coordinates)
- The model reliably detects the IC footprint within the tolerance required for assembly inspection.

## Installation

Requires Python ≥ 3.10 and TensorFlow 2.18.

```bash
git clone https://github.com/keremaras/aoi-pcb-v1.git
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
python scripts/generate_dataset.py
# or with a custom config:
python scripts/generate_dataset.py --config path/to/config.json
```

Creates `datasets/training/` and `datasets/validation/` with PNG images and CSV label files.

### 2. Train

```bash
python scripts/train.py
# or with a custom config and output directory:
python scripts/train.py --config path/to/config.json --output-dir experiments/my_run
```

Each run saves `model.keras`, a training log CSV, and a `config.json` snapshot to a timestamped directory under `experiments/`.

### 3. Evaluate

```bash
# Auto-detect the most recently modified run
python scripts/evaluate.py

# Evaluate a specific run
python scripts/evaluate.py --model-path experiments/run_YYYYMMDD_HHMMSS/model.keras

# Save prediction overlay images
python scripts/evaluate.py --save-visuals --n-visuals 20
```

### Notebooks

Interactive walkthroughs are in `notebooks/`:
- `training.ipynb` — data generation → model architecture → training loop → loss curves
- `evaluation.ipynb` — load model → run predictions → visualise keypoint overlays → per-sample error histogram

## Configuration

All parameters live in `config.json`:

| Section | Key parameters |
|---------|----------------|
| `generator` | `dataset_size`, `rotation_angle`, `delta`, `seed`, output dirs, source image paths |
| `encoder` | `normalize_data`, `normalize_labels`, `train_data_splice` |
| `training` | `optimizer_lr`, `n_epochs`, `early_stopping.*`, `lr_schedule.*` |
| `metrics` | `x_weight`, `y_weight`, `angle_weight` |

## Testing

```bash
pytest tests/
```

The test suite covers the config loader, data utilities, the full synthetic generation pipeline, encoder (including error branches), model architecture, custom loss, and alignment metric — with 100% statement coverage across all source modules. Tests are fully hermetic: no real PCB images or pre-generated datasets required.

## Project Structure

```
aoi-pcb-v1/
├── config.json
├── pyproject.toml
├── LICENSE
├── pcb_images/              # Reference images for synthesis
├── datasets/                # Generated data (gitignored — run generate_dataset.py)
├── experiments/             # Training runs  (gitignored — run train.py)
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

## Citation

```bibtex
@INPROCEEDINGS{9837280,
  author={Saif, Syed Saad and Aras, Kerem and Giuseppi, Alessandro},
  booktitle={2022 30th Mediterranean Conference on Control and Automation (MED)},
  title={Automated Optical Inspection for Printed Circuit Board Assembly Manufacturing
         with Transfer Learning and Synthetic Data Generation},
  year={2022},
  pages={318-323},
  doi={10.1109/MED54222.2022.9837280}
}
```
