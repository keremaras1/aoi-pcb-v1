"""Tests for the custom loss function and alignment metric."""

import numpy as np
import pytest
import tensorflow as tf

from aoi_pcb.model.loss import _EDGE_SELECTOR, _NORMS_SELECTOR, custom_loss
from aoi_pcb.model.metric import _EDGE_SELECTOR as METRIC_EDGE_SELECTOR
from aoi_pcb.model.metric import KeypointAlignmentMetric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_rectangle(batch_size: int = 2) -> tf.Tensor:
    """Return a batch of keypoints forming a centered axis-aligned rectangle.

    Order matches org_corners: [TL, TR, BL, BR].
    """
    corners = [0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8]
    return tf.constant([corners] * batch_size, dtype=tf.float32)


def _make_metric(ref_coords=None, ref_center=None) -> KeypointAlignmentMetric:
    """Build a KeypointAlignmentMetric with default reference values."""
    if ref_coords is None:
        ref_coords = np.array([0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8], dtype=np.float64)
    if ref_center is None:
        ref_center = np.array([0.5, 0.5], dtype=np.float64)

    class _FakeConfig:
        class metrics:
            x_weight = 1.0
            y_weight = 1.0
            angle_weight = 1.0

    return KeypointAlignmentMetric(ref_center, ref_coords, _FakeConfig())


# ---------------------------------------------------------------------------
# Custom loss
# ---------------------------------------------------------------------------


class TestCustomLossOutput:
    def test_returns_scalar(self) -> None:
        y = _perfect_rectangle()
        result = custom_loss(y, y)
        assert result.shape == ()

    def test_non_negative(self) -> None:
        y = _perfect_rectangle()
        noisy = y + tf.random.normal(y.shape, stddev=0.05)
        result = custom_loss(y, noisy)
        assert float(result) >= 0.0


class TestCustomLossMSEComponent:
    def test_identical_predictions_have_zero_mse(self) -> None:
        """When y_true == y_pred the MSE term is zero; only perp term remains."""
        y = _perfect_rectangle()
        loss_same = float(custom_loss(y, y))

        # Perturb predictions slightly — loss must increase
        noisy = y + tf.constant([[0.05] * 8] * 2, dtype=tf.float32)
        loss_noisy = float(custom_loss(y, noisy))

        assert loss_noisy > loss_same

    def test_larger_error_gives_larger_loss(self) -> None:
        y_true = _perfect_rectangle()
        small_err = y_true + 0.01
        large_err = y_true + 0.1
        assert float(custom_loss(y_true, large_err)) > float(custom_loss(y_true, small_err))


class TestCustomLossPerpComponent:
    def test_perfect_rectangle_has_low_perp_loss(self) -> None:
        """A true rectangle has perpendicular edges — perp term should be near zero."""
        y = _perfect_rectangle()
        loss = float(custom_loss(y, y))
        # MSE is 0 when y_true == y_pred, so loss ≈ perp term only
        assert loss < 1e-6


class TestModuleLevelConstants:
    def test_edge_selector_shape(self) -> None:
        assert _EDGE_SELECTOR.shape == (4, 4)

    def test_norms_selector_shape(self) -> None:
        assert _NORMS_SELECTOR.shape == (4, 4)

    def test_metric_edge_selector_shape(self) -> None:
        assert METRIC_EDGE_SELECTOR.shape == (4, 1)


# ---------------------------------------------------------------------------
# KeypointAlignmentMetric
# ---------------------------------------------------------------------------


class TestKeypointAlignmentMetricOutput:
    def test_output_is_scalar_like(self) -> None:
        metric = _make_metric()
        y = tf.cast(_perfect_rectangle(), dtype=tf.float32)
        result = metric(y, y)
        # Result is (1, 1) — squeeze to scalar for comparison
        assert result.numpy().size == 1

    def test_output_is_finite(self) -> None:
        metric = _make_metric()
        y_true = tf.cast(_perfect_rectangle(), dtype=tf.float32)
        y_pred = y_true + tf.random.normal(y_true.shape, stddev=0.02)
        result = float(tf.squeeze(metric(y_true, y_pred)))
        assert np.isfinite(result)


class TestKeypointAlignmentMetricPerfectPrediction:
    def test_perfect_prediction_gives_near_zero_center_error(self) -> None:
        """When predictions exactly match truth the x/y offset terms are zero."""
        ref_coords = np.array([0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8], dtype=np.float64)
        ref_center = np.array([0.5, 0.5], dtype=np.float64)

        class _UnitWeights:
            class metrics:
                x_weight = 1.0
                y_weight = 1.0
                angle_weight = 0.0  # isolate center error

        metric = KeypointAlignmentMetric(ref_center, ref_coords, _UnitWeights())

        # Use the reference corners as both y_true and y_pred
        y = tf.constant([ref_coords.tolist()], dtype=tf.float32)
        result = float(tf.squeeze(metric(y, y)))
        assert abs(result) < 1e-6


class TestKeypointAlignmentMetricWeights:
    def test_zero_weights_give_zero_metric(self) -> None:
        class _ZeroWeights:
            class metrics:
                x_weight = 0.0
                y_weight = 0.0
                angle_weight = 0.0

        metric = KeypointAlignmentMetric(
            np.array([0.5, 0.5]),
            np.array([0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8]),
            _ZeroWeights(),
        )
        y = tf.cast(_perfect_rectangle(), dtype=tf.float32)
        result = float(tf.squeeze(metric(y, y + 0.1)))
        assert result == pytest.approx(0.0)
