import numpy as np
import pytest

from src.models.stem_adapt import average_rgb_stem_to_grayscale


def test_output_shape_collapses_channel_axis_to_1():
    weights = np.random.randn(7, 7, 3, 64).astype(np.float32)
    result = average_rgb_stem_to_grayscale(weights)
    assert result.shape == (7, 7, 1, 64)


def test_averaging_is_correct_known_values():
    # A single output filter, 1x1 kernel, so the math is trivially checkable.
    weights = np.array([[[[1.0], [2.0], [3.0]]]])  # shape (1,1,3,1): values 1,2,3
    result = average_rgb_stem_to_grayscale(weights)
    assert result.shape == (1, 1, 1, 1)
    assert result[0, 0, 0, 0] == pytest.approx(2.0)  # mean(1,2,3) == 2


def test_preserves_dtype():
    weights = np.random.randn(3, 3, 3, 32).astype(np.float32)
    result = average_rgb_stem_to_grayscale(weights)
    assert result.dtype == np.float32


def test_rejects_non_3_channel_input():
    weights = np.random.randn(3, 3, 2, 16).astype(np.float32)
    with pytest.raises(ValueError):
        average_rgb_stem_to_grayscale(weights)


def test_rejects_wrong_ndim():
    weights = np.random.randn(3, 3, 3).astype(np.float32)
    with pytest.raises(ValueError):
        average_rgb_stem_to_grayscale(weights)


def test_equal_rgb_values_average_to_the_same_value():
    # If a filter weighted R, G, B identically (e.g. a grayscale-detector
    # filter already), averaging should reproduce that same weight exactly.
    weights = np.full((5, 5, 3, 8), 0.42, dtype=np.float32)
    result = average_rgb_stem_to_grayscale(weights)
    np.testing.assert_allclose(result, np.full((5, 5, 1, 8), 0.42, dtype=np.float32))
