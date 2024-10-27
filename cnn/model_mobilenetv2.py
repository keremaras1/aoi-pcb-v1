import keras
from keras import layers


def build_model(input_shape, output_shape):
    # Load the pre-trained weights of MobileNetV2 and freeze the weights
    backbone = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    backbone.trainable = False

    inputs = layers.Input(input_shape)
    x = layers.GaussianNoise(0.1)(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x)
    x = layers.Dropout(0.3)(x)
    x = layers.SeparableConv2D(
        output_shape, kernel_size=5, strides=1, activation="relu"
    )(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_shape)(x)

    # x = layers.SeparableConv2D(
    #    OUTPUT, kernel_size=3, strides=1, activation="relu"
    # )(x)

    return keras.Model(inputs, outputs, name="keypoint_detector")