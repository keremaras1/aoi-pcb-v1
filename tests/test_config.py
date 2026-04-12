"""Tests for the Config loader."""

import json
import tempfile
from pathlib import Path

import pytest

from aoi_pcb.config_loader import Config


@pytest.fixture
def sample_config(tmp_path: Path) -> Path:
    """Write a minimal config JSON to a temp file and return its path."""
    data = {
        "training": {
            "optimizer_lr": 0.001,
            "n_epochs": 100,
            "early_stopping": {
                "patience": 10,
                "restore_best_weights": True,
            },
        },
        "metrics": {
            "x_weight": 1.0,
            "y_weight": 1.0,
            "angle_weight": 0.5,
        },
        "flat_key": "flat_value",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    return config_path


class TestConfigLoading:
    def test_loads_flat_value(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        assert config.flat_key == "flat_value"

    def test_loads_nested_attribute(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        assert config.training.optimizer_lr == 0.001
        assert config.training.n_epochs == 100

    def test_loads_deeply_nested_attribute(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        assert config.training.early_stopping.patience == 10
        assert config.training.early_stopping.restore_best_weights is True

    def test_from_dict(self) -> None:
        config = Config.from_dict({"a": 1, "b": {"c": 2}})
        assert config.a == 1
        assert config.b.c == 2


class TestGetInitKwargs:
    def test_single_level_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        kwargs = config.get_init_kwargs("metrics")
        assert kwargs["x_weight"] == 1.0
        assert kwargs["y_weight"] == 1.0
        assert kwargs["angle_weight"] == 0.5

    def test_dot_notation_nested_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        kwargs = config.get_init_kwargs("training.early_stopping")
        assert kwargs["patience"] == 10
        assert kwargs["restore_best_weights"] is True

    def test_raises_for_missing_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        with pytest.raises(ValueError, match="not found"):
            config.get_init_kwargs("nonexistent")

    def test_raises_for_missing_nested_key(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        with pytest.raises(ValueError, match="not found"):
            config.get_init_kwargs("training.nonexistent")

    def test_raises_when_key_points_to_scalar(self, sample_config: Path) -> None:
        config = Config(str(sample_config))
        with pytest.raises(ValueError, match="must point to a nested dictionary"):
            config.get_init_kwargs("flat_key")


class TestProjectConfig:
    """Smoke test that the real config.json loads without errors."""

    def test_real_config_loads(self) -> None:
        config = Config("config.json")
        assert hasattr(config, "generator")
        assert hasattr(config, "training")
        assert hasattr(config, "metrics")

    def test_real_config_image_sources(self) -> None:
        config = Config("config.json")
        assert hasattr(config.generator, "image_sources")
        assert config.generator.image_sources.backlayer_path.endswith(".png")
        assert config.generator.image_sources.ic_path.endswith(".png")
