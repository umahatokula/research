"""Shared ResNet50V2-UNet architecture for all 3 initializations.

Per CHANNEL_COMPATIBILITY.md: ResNet50V2 (not ResNet34 -- unavailable in
tf.keras.applications and the third-party package that provides it is
unmaintained and incompatible with current TensorFlow) is used identically
across random / ImageNet / oil-spill-pretrained conditions, with only the
initialization differing. Input is single-channel (grayscale oil-spill /
VV-only flood).

Skip-connection and probe-depth layer names below were determined by
introspecting `tf.keras.applications.ResNet50V2(include_top=False,
input_shape=(512, 512, 3))` -- see git history / development notes for the
resolution-progression dump. These are architecture facts, not tunable
parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.models.stem_adapt import average_rgb_stem_to_grayscale

STEM_LAYER_NAME = "conv1_conv"

# Skip-connection layers, shallow to deep, used by the U-Net decoder.
SKIP_LAYER_NAMES = {
    "e1": "conv1_conv",  # stride 2
    "e2": "pool1_pool",  # stride 4
    "e3": "conv2_block3_out",  # stride 8
    "e4": "conv3_block4_out",  # stride 16
}

# Probe depths for Step 3 frozen-feature segmentation probing.
PROBE_DEPTH_LAYER_NAMES = {
    "early": "conv1_conv",
    "middle": "conv3_block4_out",
    "bottleneck": None,  # base.output (final activation, stride 32)
}

InitMode = Literal["random", "imagenet", "oilspill"]


def _build_raw_resnet50v2(input_shape: tuple[int, int, int], weights: str | None) -> tf.keras.Model:
    return tf.keras.applications.ResNet50V2(include_top=False, weights=weights, input_shape=input_shape)


def _imagenet_encoder_weights_grayscale(spatial_shape: tuple[int, int]) -> list[np.ndarray]:
    """Build a 3-channel ImageNet ResNet50V2, return its weights with the stem averaged to 1 channel."""
    h, w = spatial_shape
    imagenet_model = _build_raw_resnet50v2((h, w, 3), weights="imagenet")
    weights = []
    for layer in imagenet_model.layers:
        if layer.name == STEM_LAYER_NAME:
            kernel, bias = layer.get_weights()
            weights.append(average_rgb_stem_to_grayscale(kernel))
            weights.append(bias)
        else:
            weights.extend(layer.get_weights())
    return weights


def build_encoder(
    input_shape: tuple[int, int, int] = (512, 512, 1),
    init_mode: InitMode = "random",
    oilspill_weights_path: str | Path | None = None,
) -> tf.keras.Model:
    """Build the ResNet50V2 encoder at the given initialization, single-channel input.

    Returns a functional model with 5 named outputs: e1, e2, e3, e4 (skip
    connections, shallow to deep) and bottleneck (final activation).
    """
    if input_shape[-1] != 1:
        raise ValueError(f"Encoder expects single-channel input, got shape {input_shape}")

    base = _build_raw_resnet50v2(input_shape, weights=None)

    if init_mode == "random":
        pass  # base already has fresh random weights
    elif init_mode == "imagenet":
        base.set_weights(_imagenet_encoder_weights_grayscale(input_shape[:2]))
    elif init_mode == "oilspill":
        if oilspill_weights_path is None:
            raise ValueError("init_mode='oilspill' requires oilspill_weights_path")
        base.load_weights(str(oilspill_weights_path))
    else:
        raise ValueError(f"Unknown init_mode: {init_mode!r}")

    outputs = {name: base.get_layer(layer_name).output for name, layer_name in SKIP_LAYER_NAMES.items()}
    outputs["bottleneck"] = base.output
    return tf.keras.Model(inputs=base.input, outputs=outputs, name=f"resnet50v2_encoder_{init_mode}")


def _conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def _decoder_stage(x, skip, filters: int, name: str):
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_upsample")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    return _conv_block(x, filters, name)


def build_unet(
    input_shape: tuple[int, int, int] = (512, 512, 1),
    init_mode: InitMode = "random",
    oilspill_weights_path: str | Path | None = None,
    decoder_filters: tuple[int, ...] = (256, 128, 64, 32, 16),
) -> tf.keras.Model:
    """Full segmentation model: ResNet50V2 encoder (per init_mode) + U-Net decoder.

    The decoder is always freshly, randomly initialized regardless of
    init_mode -- only the encoder's initialization differs between the 3
    study conditions, per the handoff doc's "same architecture ... only
    initialization differs" requirement.
    """
    if len(decoder_filters) != 5:
        raise ValueError("decoder_filters must have exactly 5 stages (one per upsample step to input resolution)")

    encoder = build_encoder(input_shape, init_mode, oilspill_weights_path)
    inp = encoder.input
    e1, e2, e3, e4 = encoder.output["e1"], encoder.output["e2"], encoder.output["e3"], encoder.output["e4"]
    bottleneck = encoder.output["bottleneck"]

    x = _decoder_stage(bottleneck, e4, decoder_filters[0], "dec1")  # 16 -> 32
    x = _decoder_stage(x, e3, decoder_filters[1], "dec2")  # 32 -> 64
    x = _decoder_stage(x, e2, decoder_filters[2], "dec3")  # 64 -> 128
    x = _decoder_stage(x, e1, decoder_filters[3], "dec4")  # 128 -> 256
    x = _decoder_stage(x, inp, decoder_filters[4], "dec5")  # 256 -> 512 (skip = raw input)

    output = layers.Conv2D(1, 1, activation="sigmoid", name="segmentation_output")(x)
    return tf.keras.Model(inputs=inp, outputs=output, name=f"resnet50v2_unet_{init_mode}")


def build_probe_model(encoder: tf.keras.Model, depth: str) -> tf.keras.Model:
    """Frozen-feature segmentation probe for Step 3: Y_hat = Upsample(Conv1x1(F_l)).

    `encoder` must be a model built by build_encoder(). Its weights are
    frozen (trainable=False); only the new 1x1 conv is trained.
    """
    if depth not in PROBE_DEPTH_LAYER_NAMES:
        raise ValueError(f"depth must be one of {list(PROBE_DEPTH_LAYER_NAMES)}, got {depth!r}")

    encoder.trainable = False
    feature_map = encoder.output["bottleneck"] if depth == "bottleneck" else encoder.output[_probe_output_key(depth)]

    input_h = encoder.input.shape[1]
    x = layers.Conv2D(1, 1, activation="sigmoid", name=f"probe_{depth}_conv1x1")(feature_map)
    x = layers.Resizing(input_h, input_h, interpolation="bilinear", name=f"probe_{depth}_upsample")(x)
    return tf.keras.Model(inputs=encoder.input, outputs=x, name=f"probe_{depth}")


def _probe_output_key(depth: str) -> str:
    layer_name = PROBE_DEPTH_LAYER_NAMES[depth]
    for key, name in SKIP_LAYER_NAMES.items():
        if name == layer_name:
            return key
    raise AssertionError(f"probe depth {depth!r} (layer {layer_name!r}) not found among encoder outputs")
