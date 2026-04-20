"""Tests for the data processing pipeline: utils and encoder."""

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from aoi_pcb.data.utils import (
    MAX_PIXEL_VALUE,
    alphanum_key,
    normalize_values,
    rescale_values,
    sort_alphanumeric,
)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestNormalizeValues:
    def test_max_maps_to_one(self) -> None:
        data = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result = normalize_values(data)
        assert result.max() == pytest.approx(1.0)

    def test_zero_stays_zero(self) -> None:
        data = np.zeros((2, 2, 3), dtype=np.uint8)
        assert normalize_values(data).max() == 0.0

    def test_midpoint(self) -> None:
        data = np.array([[[128]]], dtype=np.uint8)
        result = normalize_values(data)
        assert result[0, 0, 0] == pytest.approx(128 / 255.0)

    def test_output_dtype_is_float(self) -> None:
        data = np.ones((4, 4, 3), dtype=np.uint8) * 100
        assert normalize_values(data).dtype in (np.float32, np.float64)


class TestRescaleValues:
    def test_one_maps_to_255(self) -> None:
        data = np.ones((2, 2, 3), dtype=np.float64)
        result = rescale_values(data)
        assert result.max() == 255

    def test_zero_stays_zero(self) -> None:
        data = np.zeros((2, 2, 3), dtype=np.float64)
        assert rescale_values(data).max() == 0

    def test_output_dtype_is_uint8(self) -> None:
        data = np.ones((2, 2, 3), dtype=np.float64) * 0.5
        assert rescale_values(data).dtype == np.uint8

    def test_roundtrip(self) -> None:
        original = np.array([[[0, 128, 255]]], dtype=np.uint8)
        roundtripped = rescale_values(normalize_values(original))
        # Allow ±1 due to floating-point rounding
        assert np.abs(original.astype(int) - roundtripped.astype(int)).max() <= 1


class TestMaxPixelValueConstant:
    def test_value_is_255(self) -> None:
        assert MAX_PIXEL_VALUE == 255.0


class TestSortAlphanumeric:
    def test_numeric_order(self, tmp_path: Path) -> None:
        # Create files whose lexicographic order differs from numeric order
        for name in ["pcb_10.png", "pcb_2.png", "pcb_1.png", "pcb_20.png"]:
            (tmp_path / name).touch()
        result = sort_alphanumeric(str(tmp_path))
        assert result == ["pcb_1.png", "pcb_2.png", "pcb_10.png", "pcb_20.png"]

    def test_single_file(self, tmp_path: Path) -> None:
        (tmp_path / "pcb_0.png").touch()
        assert sort_alphanumeric(str(tmp_path)) == ["pcb_0.png"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert sort_alphanumeric(str(tmp_path)) == []


class TestAlphanumKey:
    def test_splits_digits_from_text(self) -> None:
        key = alphanum_key("pcb_10.png")
        # Should contain the integer 10, not the string "10"
        assert 10 in key

    def test_lowercase(self) -> None:
        key = alphanum_key("FILE_1.PNG")
        assert "file_" in key or any(isinstance(k, str) and k == "file_" for k in key)


# ---------------------------------------------------------------------------
# DataEncoder — tested with small synthetic fixtures (no real dataset needed)
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(
    tmp_path: Path,
    n_images: int = 5,
    img_size: int = 32,
) -> tuple[Path, Path]:
    """Create a tiny synthetic image directory and CSV label file."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Write n_images solid-color PNGs
    for i in range(n_images):
        color = (i * 40, i * 20, 100)
        img = Image.new("RGB", (img_size, img_size), color)
        img.save(images_dir / f"pcb_{i}.png")

    # Write CSV: first row = reference points + center, rest = per-image corners
    label_path = tmp_path / "labels.csv"
    ref_corners = [(4, 4), (28, 4), (28, 28), (4, 28)]
    ref_center = (16, 16)

    with open(label_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ref_corners, ref_center])
        for _ in range(n_images):
            writer.writerow([(4, 4), (28, 4), (28, 28), (4, 28)])

    return images_dir, label_path



class _FakeConfig:
    """Minimal config stub for DataEncoder."""
    class _Encoder:
        train_data_splice = None
        normalize_data = False
        normalize_labels = False

    encoder = _Encoder()


class _FakeNormalisedConfig:
    class _Encoder:
        train_data_splice = None
        normalize_data = True
        normalize_labels = True

    encoder = _Encoder()


class TestDataEncoder:
    def test_output_shapes(self, tmp_path: Path) -> None:
        from aoi_pcb.data.encoder import DataEncoder

        images_dir, label_path = _make_synthetic_dataset(tmp_path, n_images=5, img_size=32)
        encoder = DataEncoder(_FakeConfig())
        X, y, ref_coords, ref_center = encoder(str(images_dir), str(label_path))

        assert X.shape == (5, 32, 32, 3)
        assert y.shape == (5, 8)          # 4 corners × 2 coords
        assert ref_coords.shape == (8,)
        assert ref_center.shape == (2,)

    def test_normalized_images_in_range(self, tmp_path: Path) -> None:
        from aoi_pcb.data.encoder import DataEncoder

        images_dir, label_path = _make_synthetic_dataset(tmp_path, n_images=3, img_size=32)
        encoder = DataEncoder(_FakeNormalisedConfig())
        X, y, ref_coords, ref_center = encoder(str(images_dir), str(label_path))

        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_normalized_labels_in_range(self, tmp_path: Path) -> None:
        from aoi_pcb.data.encoder import DataEncoder

        images_dir, label_path = _make_synthetic_dataset(tmp_path, n_images=3, img_size=32)
        encoder = DataEncoder(_FakeNormalisedConfig())
        _, y, ref_coords, ref_center = encoder(str(images_dir), str(label_path))

        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_splice_truncates_dataset(self, tmp_path: Path) -> None:
        from aoi_pcb.data.encoder import DataEncoder

        class _SplicedConfig:
            class _Encoder:
                train_data_splice = 3
                normalize_data = False
                normalize_labels = False
            encoder = _Encoder()

        images_dir, label_path = _make_synthetic_dataset(tmp_path, n_images=5, img_size=32)
        encoder = DataEncoder(_SplicedConfig())
        X, y, _, _ = encoder(str(images_dir), str(label_path))

        assert X.shape[0] == 3
        assert y.shape[0] == 3


# ---------------------------------------------------------------------------
# DataEncoder — validation error branches
# ---------------------------------------------------------------------------

class _NormalisedLabelsConfig:
    class _Encoder:
        train_data_splice = None
        normalize_data = False
        normalize_labels = True

    encoder = _Encoder()


class TestDataEncoderValidation:
    """Validation error branches — uses mocks to inject arrays directly,
    bypassing file I/O."""

    def test_out_of_bounds_labels_raise(self) -> None:
        """Labels > 1.0 after normalization raise ValueError."""
        from aoi_pcb.data.encoder import DataEncoder

        with patch('aoi_pcb.data.encoder.sort_alphanumeric', return_value=[]), \
             patch.object(DataEncoder, 'image_to_numpy', return_value=np.zeros((1, 32, 32, 3))), \
             patch.object(DataEncoder, 'coords_to_numpy', return_value=(
                 np.array([[1.09, 0.125, 0.875, 0.125, 0.125, 0.875, 0.875, 0.875]]),
                 np.array([0.125, 0.125, 0.875, 0.125, 0.125, 0.875, 0.875, 0.875]),
                 np.array([0.5, 0.5]),
             )):
            encoder = DataEncoder(_NormalisedLabelsConfig())
            with pytest.raises(ValueError, match="Labels are not normalized"):
                encoder("fake_dir", "fake_labels.csv")

    def test_out_of_bounds_ref_coords_raise(self) -> None:
        """Reference coords > 1.0 after normalization raise ValueError."""
        from aoi_pcb.data.encoder import DataEncoder

        with patch('aoi_pcb.data.encoder.sort_alphanumeric', return_value=[]), \
             patch.object(DataEncoder, 'image_to_numpy', return_value=np.zeros((1, 32, 32, 3))), \
             patch.object(DataEncoder, 'coords_to_numpy', return_value=(
                 np.array([[0.125, 0.125, 0.875, 0.125, 0.125, 0.875, 0.875, 0.875]]),
                 np.array([1.09, 0.125, 0.875, 0.125, 0.125, 0.875, 0.875, 0.875]),
                 np.array([0.5, 0.5]),
             )):
            encoder = DataEncoder(_NormalisedLabelsConfig())
            with pytest.raises(ValueError, match="Reference coordinates are not normalized"):
                encoder("fake_dir", "fake_labels.csv")

    def test_out_of_bounds_ref_center_raise(self) -> None:
        """Reference center > 1.0 after normalization raises ValueError."""
        from aoi_pcb.data.encoder import DataEncoder

        with patch('aoi_pcb.data.encoder.sort_alphanumeric', return_value=[]), \
             patch.object(DataEncoder, 'image_to_numpy', return_value=np.zeros((1, 32, 32, 3))), \
             patch.object(DataEncoder, 'coords_to_numpy', return_value=(
                 np.array([[0.125, 0.125, 0.875, 0.125, 0.125, 0.875, 0.875, 0.875]]),
                 np.array([0.125, 0.125, 0.875, 0.125, 0.125, 0.875, 0.875, 0.875]),
                 np.array([1.09, 0.5]),
             )):
            encoder = DataEncoder(_NormalisedLabelsConfig())
            with pytest.raises(ValueError, match="Reference center is not normalized"):
                encoder("fake_dir", "fake_labels.csv")

    def test_float_train_data_splice_raises(self) -> None:
        """Non-integer train_data_splice raises ValueError inside image_to_numpy."""
        from aoi_pcb.data.encoder import DataEncoder

        class _FloatSpliceConfig:
            class _Encoder:
                train_data_splice = 3.5
                normalize_data = False
                normalize_labels = False

            encoder = _Encoder()

        fake_img = Image.new("RGB", (32, 32))
        with patch('aoi_pcb.data.encoder.sort_alphanumeric', return_value=['pcb_0.png']), \
             patch('aoi_pcb.data.encoder.Image.open', return_value=fake_img):
            encoder = DataEncoder(_FloatSpliceConfig())
            with pytest.raises(ValueError, match="must be an integer"):
                encoder("fake_dir", "fake_labels.csv")
