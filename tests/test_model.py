"""Tests for the model architecture."""

import numpy as np
import pytest
import tensorflow as tf

from aoi_pcb.model.architecture import (
    _CONV_DROPOUT,
    _DENSE_DROPOUT,
    _DENSE_UNITS,
    _GAUSSIAN_NOISE_STDDEV,
    _SEPARABLE_CONV_FILTERS,
    _SEPARABLE_CONV_KERNEL,
    build_model,
)

INPUT_SHAPE = (64, 64, 3)   # smaller than production (256,256,3) to keep tests fast
OUTPUT_SHAPE = 8             # 4 keypoints × 2 coordinates


@pytest.fixture(scope="module")
def model() -> tf.keras.Model:
    """Build one model instance shared across tests in this module."""
    return build_model(INPUT_SHAPE, OUTPUT_SHAPE)


class TestModelOutputShape:
    def test_output_shape(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((2, *INPUT_SHAPE))
        y = model(x, training=False)
        assert y.shape == (2, OUTPUT_SHAPE)

    def test_output_shape_batch_1(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((1, *INPUT_SHAPE))
        y = model(x, training=False)
        assert y.shape == (1, OUTPUT_SHAPE)

    def test_model_output_is_tensor(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((2, *INPUT_SHAPE))
        y = model(x, training=False)
        assert isinstance(y, tf.Tensor)


class TestBackboneFrozen:
    def test_backbone_is_not_trainable(self, model: tf.keras.Model) -> None:
        backbone = next(
            layer for layer in model.layers
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
        )
        assert backbone.trainable is False

    def test_backbone_weights_are_non_trainable(self, model: tf.keras.Model) -> None:
        backbone = next(
            layer for layer in model.layers
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower()
        )
        assert len(backbone.non_trainable_weights) > 0
        assert len(backbone.trainable_weights) == 0

    def test_custom_head_has_trainable_weights(self, model: tf.keras.Model) -> None:
        assert len(model.trainable_weights) > 0


class TestModelArchitectureConstants:
    def test_gaussian_noise_stddev(self) -> None:
        assert _GAUSSIAN_NOISE_STDDEV == 0.1

    def test_separable_conv_filters(self) -> None:
        # Must be 8 per the paper — independent of output_shape
        assert _SEPARABLE_CONV_FILTERS == 8

    def test_separable_conv_kernel(self) -> None:
        assert _SEPARABLE_CONV_KERNEL == 5

    def test_dense_units(self) -> None:
        assert _DENSE_UNITS == 512


class TestGradientFlow:
    def test_gradients_flow_through_trainable_weights(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((2, *INPUT_SHAPE))
        y_true = tf.random.normal((2, OUTPUT_SHAPE))

        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))

        grads = tape.gradient(loss, model.trainable_weights)
        assert any(g is not None for g in grads), "No gradients found for trainable weights."

    def test_frozen_weights_have_no_gradients(self, model: tf.keras.Model) -> None:
        x = tf.random.normal((2, *INPUT_SHAPE))
        y_true = tf.random.normal((2, OUTPUT_SHAPE))

        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))

        grads = tape.gradient(loss, model.non_trainable_weights)
        assert all(g is None for g in grads), "Gradients leaked into frozen backbone weights."


class TestModelName:
    def test_model_name(self, model: tf.keras.Model) -> None:
        assert model.name == "keypoint_detector"
