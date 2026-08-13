from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.data.sen1floods11_dataset import (
    DB_CLIP_MAX,
    DB_CLIP_MIN,
    derive_flood_label,
    load_chip,
    load_permanent_water,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"
# Known-downloaded chip, verified present via manual inspection during development.
SAMPLE_CHIP = "Ghana_103272"

requires_real_data = pytest.mark.skipif(
    not (DATA_DIR / "S1Hand" / f"{SAMPLE_CHIP}_S1Hand.tif").exists(),
    reason="Sen1Floods11 data not downloaded locally (run src/data/download_sen1floods11.py)",
)


@requires_real_data
def test_load_chip_shapes():
    vv, label = load_chip(SAMPLE_CHIP, DATA_DIR)
    assert vv.shape == (512, 512, 1)
    assert label.shape == (512, 512, 1)


@requires_real_data
def test_load_chip_dtypes():
    vv, label = load_chip(SAMPLE_CHIP, DATA_DIR)
    assert vv.dtype == np.float32
    assert label.dtype == np.int16


@requires_real_data
def test_vv_values_in_unit_range():
    vv, _ = load_chip(SAMPLE_CHIP, DATA_DIR)
    assert vv.min() >= 0.0
    assert vv.max() <= 1.0


@requires_real_data
def test_label_values_are_valid_set():
    _, label = load_chip(SAMPLE_CHIP, DATA_DIR)
    assert set(np.unique(label)).issubset({-1, 0, 1})


@requires_real_data
def test_uses_band_0_not_band_1():
    # Manually replicate the VV (band 0) preprocessing and confirm it matches
    # load_chip's output -- catches an accidental band-index swap (VV vs VH).
    raw = tifffile.imread(DATA_DIR / "S1Hand" / f"{SAMPLE_CHIP}_S1Hand.tif")
    expected_vv = np.nan_to_num(raw[0], nan=DB_CLIP_MIN)
    expected_vv = np.clip(expected_vv, DB_CLIP_MIN, DB_CLIP_MAX)
    expected_vv = (expected_vv - DB_CLIP_MIN) / (DB_CLIP_MAX - DB_CLIP_MIN)

    vv, _ = load_chip(SAMPLE_CHIP, DATA_DIR)
    np.testing.assert_allclose(vv[..., 0], expected_vv, rtol=1e-6)

    # And confirm it does NOT match band 1 (VH) -- would fail silently if the
    # band index were swapped, since both bands are valid float32 dB arrays.
    wrong_vv = np.nan_to_num(raw[1], nan=DB_CLIP_MIN)
    wrong_vv = np.clip(wrong_vv, DB_CLIP_MIN, DB_CLIP_MAX)
    wrong_vv = (wrong_vv - DB_CLIP_MIN) / (DB_CLIP_MAX - DB_CLIP_MIN)
    assert not np.allclose(vv[..., 0], wrong_vv)


@requires_real_data
def test_load_chip_raises_on_missing_chip():
    with pytest.raises(FileNotFoundError):
        load_chip("Nonexistent_999999", DATA_DIR)


@requires_real_data
def test_load_permanent_water_shape_and_values():
    jrc = load_permanent_water(SAMPLE_CHIP, DATA_DIR)
    assert jrc.shape == (512, 512, 1)
    assert set(np.unique(jrc)).issubset({0, 1})


def test_derive_flood_label_subtracts_permanent_water():
    label = np.array([[-1, 0, 1, 1]])
    permanent_water = np.array([[0, 0, 0, 1]])  # last pixel is permanent water
    flood = derive_flood_label(label, permanent_water)
    np.testing.assert_array_equal(flood, [[-1, 0, 1, 0]])


def test_derive_flood_label_leaves_nodata_and_dry_pixels_untouched():
    label = np.array([[-1, -1, 0, 0]])
    permanent_water = np.array([[1, 0, 1, 0]])  # permanent water shouldn't affect -1 or 0 labels
    flood = derive_flood_label(label, permanent_water)
    np.testing.assert_array_equal(flood, label)


def test_derive_flood_label_does_not_mutate_input():
    label = np.array([[1, 1]])
    permanent_water = np.array([[1, 0]])
    original = label.copy()
    derive_flood_label(label, permanent_water)
    np.testing.assert_array_equal(label, original)


@requires_real_data
def test_load_chip_default_separates_permanent_water():
    _, flood_label = load_chip(SAMPLE_CHIP, DATA_DIR)  # default: separate_permanent_water=True
    _, raw_label = load_chip(SAMPLE_CHIP, DATA_DIR, separate_permanent_water=False)
    jrc = load_permanent_water(SAMPLE_CHIP, DATA_DIR)

    expected_flood = derive_flood_label(raw_label, jrc)
    np.testing.assert_array_equal(flood_label, expected_flood)

    # Flood label should never have MORE water pixels than the raw label.
    assert (flood_label == 1).sum() <= (raw_label == 1).sum()
