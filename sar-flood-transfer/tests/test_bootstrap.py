import numpy as np
import pytest

from src.stats.bootstrap import paired_bootstrap_ci


def test_identical_arrays_give_zero_diff_and_nonsignificant():
    rng = np.random.default_rng(0)
    values = rng.normal(0.7, 0.1, size=50)
    result = paired_bootstrap_ci(values, values.copy(), n_boot=2000, seed=1)
    assert result["mean_diff"] == pytest.approx(0.0, abs=1e-10)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
    assert result["p_value"] > 0.05


def test_clear_positive_effect_detected():
    rng = np.random.default_rng(0)
    # a is consistently ~0.15 higher than b, with modest noise -- should be
    # a clear, statistically detectable effect.
    b = rng.normal(0.5, 0.03, size=60)
    a = b + 0.15 + rng.normal(0, 0.03, size=60)
    result = paired_bootstrap_ci(a, b, n_boot=5000, seed=1)
    assert result["mean_diff"] == pytest.approx(0.15, abs=0.03)
    assert result["ci_low"] > 0.0  # CI excludes zero
    assert result["p_value"] < 0.05


def test_no_effect_ci_includes_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, size=60)
    b = rng.normal(0.5, 0.1, size=60)  # same distribution, independent noise
    result = paired_bootstrap_ci(a, b, n_boot=5000, seed=1)
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_deterministic_for_same_seed():
    rng = np.random.default_rng(0)
    a = rng.normal(size=40)
    b = rng.normal(size=40)
    r1 = paired_bootstrap_ci(a, b, n_boot=1000, seed=42)
    r2 = paired_bootstrap_ci(a, b, n_boot=1000, seed=42)
    assert r1 == r2


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_rejects_too_few_observations():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.array([1.0]), np.array([2.0]))


def test_rejects_non_1d_arrays():
    with pytest.raises(ValueError):
        paired_bootstrap_ci(np.zeros((3, 2)), np.zeros((3, 2)))


def test_n_matches_input_length():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])
    result = paired_bootstrap_ci(a, b, n_boot=500, seed=0)
    assert result["n"] == 4
