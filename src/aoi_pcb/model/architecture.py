"""MobileNetV2-based keypoint detection model architecture.

The model follows the design described in the paper: a frozen MobileNetV2
backbone (pre-trained on ImageNet) is used for feature extraction, followed
by a custom regression head that predicts the (x, y) coordinates of four
IC corner keypoints.
"""

import keras
from keras import layers

# Architecture hyperparameters — match values reported in the paper
_GAUSSIAN_NOISE_STDDEV: float = 0.1
_CONV_DROPOUT: float = 0.3
_DENSE_DROPOUT: float = 0.1
_DENSE_UNITS: int = 512
_SEPARABLE_CONV_FILTERS: int = 8  # Fixed per paper; independent of output dimension
_SEPARABLE_CONV_KERNEL: int = 5


def build_model(
    input_shape: tuple[int, ...],
    output_shape: int,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Build the MobileNetV2-based keypoint detector.

    The backbone (MobileNetV2) is loaded with the specified weights and frozen.
    Only the custom regression head is trained.

    Architecture::

        Input → GaussianNoise → MobileNetV2 (frozen) → Dropout
              → SeparableConv2D(8, 5×5, ReLU) → Flatten
              → Dense(512, ReLU) → Dropout → Dense(output_shape)

    Args:
        input_shape: Shape of a single input image, e.g. ``(256, 256, 3)``.
            Must be at least ``(160, 160, 3)`` for the 5×5 SeparableConv2D
            to fit after MobileNetV2's 32× spatial downsampling.
        output_shape: Number of output values. For 4 keypoints this is 8
            (4 corners × 2 coordinates).
        weights: Weights to load into MobileNetV2. Use ``"imagenet"`` for
            pre-trained weights (default) or ``None`` for random initialization
            (useful for testing without downloading weights).

    Returns:
        A Keras Model named ``"keypoint_detector"``.
    """
    backbone = keras.applications.MobileNetV2(
        weights=weights, include_top=False, input_shape=input_shape
    )
    backbone.trainable = False

    inputs = layers.Input(input_shape)

    # Gaussian noise improves robustness to minor image variations
    x = layers.GaussianNoise(_GAUSSIAN_NOISE_STDDEV)(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x)
    x = layers.Dropout(_CONV_DROPOUT)(x)

    # Separable convolution reduces parameter count for low-compute deployment
    x = layers.SeparableConv2D(
        _SEPARABLE_CONV_FILTERS, kernel_size=_SEPARABLE_CONV_KERNEL, strides=1, activation="relu"
    )(x)

    x = layers.Flatten()(x)
    x = layers.Dense(_DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(_DENSE_DROPOUT)(x)

    outputs = layers.Dense(output_shape)(x)

    return keras.Model(inputs, outputs, name="keypoint_detector")
