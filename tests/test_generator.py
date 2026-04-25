"""Tests for synthetic PCB image generation pipeline (generator.py)."""

import csv
import math
import random as rnd
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aoi_pcb.data.generator import (
    add_alpha,
    add_tuple,
    blend,
    generate_dataset,
    get_layer,
    image_scale,
    img_generator,
    org_corners,
    pil_to_np,
)


@pytest.fixture
def fake_pcb_assets(tmp_path: Path) -> tuple[str, str]:
    """Create minimal solid-color backlayer and IC images for hermetic tests."""
    backlayer_path = tmp_path / "backlayer.png"
    ic_path = tmp_path / "ic.png"
    Image.new("RGB", (256, 256), color=(120, 80, 60)).save(backlayer_path)
    Image.new("RGB", (64, 64), color=(200, 200, 100)).save(ic_path)
    return str(backlayer_path), str(ic_path)


# ---------------------------------------------------------------------------
# add_tuple
# ---------------------------------------------------------------------------


class TestAddTuple:
    def test_element_wise_sum(self) -> None:
        assert add_tuple((1, 2, 3), (4, 5, 6)) == (5, 7, 9)

    def test_two_element_tuples(self) -> None:
        assert add_tuple((0, 0), (5, 7)) == (5, 7)

    def test_single_element(self) -> None:
        assert add_tuple((10,), (20,)) == (30,)


# ---------------------------------------------------------------------------
# add_alpha
# ---------------------------------------------------------------------------


class TestAddAlpha:
    def test_appends_alpha_channel(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        assert add_alpha(rgb).shape == (4, 4, 4)

    def test_alpha_is_fully_opaque(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        assert np.all(add_alpha(rgb)[:, :, 3] == 255)

    def test_rgb_channels_unchanged(self) -> None:
        rgb = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        np.testing.assert_array_equal(add_alpha(rgb)[:, :, :3], rgb)


# ---------------------------------------------------------------------------
# image_scale
# ---------------------------------------------------------------------------


class TestImageScale:
    def test_scale_by_factor_returns_none_rescale(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        scaled, factor = image_scale(img, shape=None, factor=2.0)
        assert scaled.shape == (200, 200, 3)
        assert factor is None

    def test_scale_by_shape(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        scaled, _ = image_scale(img, shape=(50, 50))
        assert scaled.shape[:2] == (50, 50)

    def test_rescale_factor_is_target_height_over_original_width(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200
        _, factor = image_scale(img, shape=(100, 100))
        assert factor == pytest.approx(100 / 200)

    def test_shape_with_wrong_length_raises(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape must have exactly 2 elements"):
            image_scale(img, shape=(100, 100, 3))


# ---------------------------------------------------------------------------
# pil_to_np
# ---------------------------------------------------------------------------


class TestPilToNp:
    def test_output_shape_is_width_height_channels(self) -> None:
        # pil_to_np reshapes using img.size which is (W, H) in PIL convention
        pil_img = Image.new("RGBA", (10, 8), color=(100, 150, 200, 255))
        arr = pil_to_np(pil_img)
        assert arr.shape == (10, 8, 4)

    def test_output_dtype_is_uint8(self) -> None:
        pil_img = Image.new("RGBA", (4, 4), color=(0, 0, 0, 255))
        assert pil_to_np(pil_img).dtype == np.uint8


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------


class TestBlend:
    def test_opaque_foreground_covers_background(self) -> None:
        ic = np.zeros((16, 16, 4), dtype=np.uint8)
        ic[:, :, 0] = 200  # red channel
        ic[:, :, 3] = 255  # fully opaque
        bg = np.zeros((16, 16, 4), dtype=np.uint8)
        bg[:, :, 2] = 200  # blue channel
        bg[:, :, 3] = 255  # fully opaque
        result = blend(ic, bg)
        assert result[8, 8, 0] == 200  # IC red dominates
        assert result[8, 8, 2] == 0  # background blue gone

    def test_transparent_foreground_shows_background(self) -> None:
        ic = np.zeros((16, 16, 4), dtype=np.uint8)
        ic[:, :, 3] = 0  # fully transparent
        bg = np.zeros((16, 16, 4), dtype=np.uint8)
        bg[:, :, 0] = 200
        bg[:, :, 3] = 255
        result = blend(ic, bg)
        assert result[8, 8, 0] == 200  # background shows through

    def test_returns_rgb_not_rgba(self) -> None:
        ic = np.full((8, 8, 4), 128, dtype=np.uint8)
        bg = np.full((8, 8, 4), 64, dtype=np.uint8)
        result = blend(ic, bg)
        assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# org_corners
# ---------------------------------------------------------------------------


class TestOrgCorners:
    def test_returns_both_original_and_rotated(self) -> None:
        original, rotated = org_corners(10, 10, np.array([128.0, 128.0]), 0.0)
        assert len(original) == 4
        assert len(rotated) == 4

    def test_original_order_is_tl_tr_bl_br(self) -> None:
        original, _ = org_corners(10, 10, np.array([128.0, 128.0]), 0.0)
        assert original[0] == (0, 0)  # TL
        assert original[1] == (10, 0)  # TR
        assert original[2] == (0, 10)  # BL
        assert original[3] == (10, 10)  # BR

    def test_zero_angle_translates_to_center(self) -> None:
        _, rotated = org_corners(10, 10, np.array([128.0, 128.0]), 0.0)
        # TL: (0,0) - (5,5) + (128,128) = (123, 123)
        assert rotated[0] == (123, 123)
        # TR: (10,0) - (5,5) + (128,128) = (133, 123)
        assert rotated[1] == (133, 123)

    def test_nonzero_angle_rotates_corners(self) -> None:
        _, zero_rot = org_corners(10, 10, np.array([128.0, 128.0]), 0.0)
        _, quarter_turn = org_corners(10, 10, np.array([128.0, 128.0]), math.pi / 4)
        assert zero_rot != quarter_turn


# ---------------------------------------------------------------------------
# get_layer
# ---------------------------------------------------------------------------


class TestGetLayer:
    def test_shape_parameter_resizes_to_target(self, fake_pcb_assets: tuple[str, str]) -> None:
        backlayer_path, _ = fake_pcb_assets
        h, w, rescale_factor, layer = get_layer(backlayer_path, shape=(256, 256))
        assert h == 256
        assert w == 256
        assert layer.shape == (256, 256, 4)
        # rescale_factor = shape[1] / original_width; fixture is 256x256 so ratio = 1.0
        assert rescale_factor == pytest.approx(1.0)

    def test_factor_parameter_scales_image(self, fake_pcb_assets: tuple[str, str]) -> None:
        _, ic_path = fake_pcb_assets
        # IC fixture is 64x64; factor=1.5 → 96x96
        h, w, rescale_factor, layer = get_layer(ic_path, shape=None, factor=1.5)
        assert h == 96
        assert w == 96
        assert rescale_factor is None

    def test_output_has_fully_opaque_alpha(self, fake_pcb_assets: tuple[str, str]) -> None:
        backlayer_path, _ = fake_pcb_assets
        _, _, _, layer = get_layer(backlayer_path, shape=(64, 64))
        assert layer.shape[2] == 4
        assert np.all(layer[:, :, 3] == 255)


# ---------------------------------------------------------------------------
# img_generator
# ---------------------------------------------------------------------------


def _make_rgba(h: int, w: int) -> np.ndarray:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    return arr


class TestImgGenerator:
    def test_output_shapes_and_types(self) -> None:
        img, ic_center, corners, angle = img_generator(
            alpha=5.0,
            delta_weight=0.1,
            back_height=256,
            back_width=256,
            backlayer=_make_rgba(256, 256),
            ic_height=64,
            ic_width=64,
            front_layer_base=_make_rgba(64, 64),
        )
        assert img.shape == (256, 256, 3)
        assert ic_center.shape == (2,)
        assert len(corners) == 4
        assert isinstance(angle, float)

    def test_angle_within_alpha_bounds(self) -> None:
        kwargs = dict(
            alpha=10.0,
            delta_weight=0.1,
            back_height=256,
            back_width=256,
            backlayer=_make_rgba(256, 256),
            ic_height=64,
            ic_width=64,
            front_layer_base=_make_rgba(64, 64),
        )
        for _ in range(10):
            _, _, _, angle = img_generator(**kwargs)
            assert -10.0 <= angle <= 10.0

    def test_seed_reproducibility(self) -> None:
        kwargs = dict(
            alpha=5.0,
            delta_weight=0.1,
            back_height=256,
            back_width=256,
            backlayer=_make_rgba(256, 256),
            ic_height=64,
            ic_width=64,
            front_layer_base=_make_rgba(64, 64),
        )
        np.random.seed(7)
        rnd.seed(7)
        _, center1, corners1, angle1 = img_generator(**kwargs)

        np.random.seed(7)
        rnd.seed(7)
        _, center2, corners2, angle2 = img_generator(**kwargs)

        assert angle1 == angle2
        np.testing.assert_array_equal(center1, center2)
        assert corners1 == corners2


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    def test_creates_images_and_csv(self, tmp_path: Path, fake_pcb_assets: tuple[str, str]) -> None:
        backlayer_path, ic_path = fake_pcb_assets

        generate_dataset(
            dataset_size=3,
            rotation_angle=5.0,
            delta=0.1,
            data_dir=str(tmp_path / "images"),
            labels_dir=str(tmp_path / "labels"),
            label_file="test.csv",
            backlayer_path=backlayer_path,
            ic_path=ic_path,
            seed=42,
        )

        assert len(list((tmp_path / "images").glob("pcb_*.png"))) == 3
        with open(tmp_path / "labels" / "test.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4  # 1 reference row + 3 data rows

    def test_seed_reproducibility(self, tmp_path: Path, fake_pcb_assets: tuple[str, str]) -> None:
        backlayer_path, ic_path = fake_pcb_assets
        common = dict(
            dataset_size=2,
            rotation_angle=10.0,
            delta=0.1,
            backlayer_path=backlayer_path,
            ic_path=ic_path,
            seed=99,
        )
        generate_dataset(
            **common,
            data_dir=str(tmp_path / "a/img"),
            labels_dir=str(tmp_path / "a/lbl"),
            label_file="out.csv",
        )
        generate_dataset(
            **common,
            data_dir=str(tmp_path / "b/img"),
            labels_dir=str(tmp_path / "b/lbl"),
            label_file="out.csv",
        )

        assert (tmp_path / "a/lbl/out.csv").read_text() == (tmp_path / "b/lbl/out.csv").read_text()

    def test_different_seeds_produce_different_output(
        self, tmp_path: Path, fake_pcb_assets: tuple[str, str]
    ) -> None:
        backlayer_path, ic_path = fake_pcb_assets
        base = dict(
            dataset_size=5,
            rotation_angle=10.0,
            delta=0.1,
            backlayer_path=backlayer_path,
            ic_path=ic_path,
        )
        generate_dataset(
            **base,
            seed=1,
            data_dir=str(tmp_path / "a/img"),
            labels_dir=str(tmp_path / "a/lbl"),
            label_file="out.csv",
        )
        generate_dataset(
            **base,
            seed=2,
            data_dir=str(tmp_path / "b/img"),
            labels_dir=str(tmp_path / "b/lbl"),
            label_file="out.csv",
        )

        assert (tmp_path / "a/lbl/out.csv").read_text() != (tmp_path / "b/lbl/out.csv").read_text()
