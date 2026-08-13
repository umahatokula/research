import numpy as np
import pytest
import tensorflow as tf

from src.models.backbone_unet import (
    PROBE_DEPTH_LAYER_NAMES,
    build_encoder,
    build_probe_model,
    build_unet,
)

# Small input size keeps these tests fast on CPU; the architecture is fully
# convolutional so weight shapes/logic don't depend on spatial size.
SMALL_SHAPE = (64, 64, 1)


def test_build_encoder_random_output_keys_and_shapes():
    encoder = build_encoder(SMALL_SHAPE, init_mode="random")
    assert set(encoder.output.keys()) == {"e1", "e2", "e3", "e4", "bottleneck"}
    # Spatial resolutions should strictly decrease: e1 > e2 > e3 > e4 >= bottleneck
    res = {k: v.shape[1] for k, v in encoder.output.items()}
    assert res["e1"] > res["e2"] > res["e3"] > res["e4"]
    assert res["bottleneck"] <= res["e4"]


def test_build_encoder_rejects_multichannel_input():
    with pytest.raises(ValueError):
        build_encoder((64, 64, 3), init_mode="random")


def test_build_encoder_rejects_unknown_init_mode():
    with pytest.raises(ValueError):
        build_encoder(SMALL_SHAPE, init_mode="bogus")


def test_build_encoder_oilspill_requires_weights_path():
    with pytest.raises(ValueError):
        build_encoder(SMALL_SHAPE, init_mode="oilspill")


def test_oilspill_init_loads_previously_saved_weights(tmp_path):
    # Build a "pretrained" encoder, save it, then load it back via
    # init_mode="oilspill" and confirm the stem weights match exactly --
    # this is the actual code path used to attach the fresh oil-spill
    # pretraining checkpoint (see CHANNEL_COMPATIBILITY.md).
    source = build_encoder(SMALL_SHAPE, init_mode="random")
    weights_path = tmp_path / "oilspill_encoder.weights.h5"
    source.save_weights(str(weights_path))

    loaded = build_encoder(SMALL_SHAPE, init_mode="oilspill", oilspill_weights_path=weights_path)

    source_stem = source.get_layer("conv1_conv").get_weights()[0]
    loaded_stem = loaded.get_layer("conv1_conv").get_weights()[0]
    np.testing.assert_allclose(source_stem, loaded_stem)


def test_build_unet_output_shape_matches_input_spatially():
    model = build_unet(SMALL_SHAPE, init_mode="random")
    assert model.output_shape == (None, 64, 64, 1)


def test_build_unet_rejects_wrong_decoder_filter_count():
    with pytest.raises(ValueError):
        build_unet(SMALL_SHAPE, init_mode="random", decoder_filters=(64, 32))


def test_unet_forward_pass_output_in_sigmoid_range():
    model = build_unet(SMALL_SHAPE, init_mode="random")
    x = np.random.rand(1, 64, 64, 1).astype(np.float32)
    y = model(x, training=False).numpy()
    assert y.shape == (1, 64, 64, 1)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


@pytest.mark.parametrize("depth", list(PROBE_DEPTH_LAYER_NAMES.keys()))
def test_probe_model_output_shape_matches_input(depth):
    encoder = build_encoder(SMALL_SHAPE, init_mode="random")
    probe = build_probe_model(encoder, depth)
    x = np.random.rand(1, 64, 64, 1).astype(np.float32)
    y = probe(x, training=False).numpy()
    assert y.shape == (1, 64, 64, 1)


def test_probe_model_freezes_encoder_weights():
    encoder = build_encoder(SMALL_SHAPE, init_mode="random")
    probe = build_probe_model(encoder, "middle")
    encoder_layer_names = {layer.name for layer in encoder.layers}
    for layer in probe.layers:
        if layer.name in encoder_layer_names:
            assert not layer.trainable, f"encoder layer {layer.name} should be frozen in the probe model"


def test_probe_model_rejects_unknown_depth():
    encoder = build_encoder(SMALL_SHAPE, init_mode="random")
    with pytest.raises(ValueError):
        build_probe_model(encoder, "nonexistent_depth")
