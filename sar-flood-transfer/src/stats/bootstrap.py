"""Paired bootstrap over independent test scenes (handoff doc, Step 4).

Explicitly NOT a Wilcoxon test on n=3 seeds (too few observations), and NOT
a pixel-level test (pixels aren't independent, produces artificially tiny
p-values). Resampling unit is the test scene/chip -- the closest thing to
an independent observation available in this study.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Bootstrap CI and two-sided p-value for the paired difference (a - b).

    Args:
        values_a, values_b: 1D arrays of equal length, paired by index (e.g.
            per-scene metric values for two conditions on the SAME scenes,
            already averaged across seeds -- pairing must hold, index i in
            both arrays must refer to the same scene).
        n_boot: number of bootstrap resamples.
        ci: confidence level (e.g. 0.95 for a 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        mean_diff: observed mean(a - b) on the real (non-resampled) data.
        ci_low, ci_high: percentile bootstrap CI bounds on the mean difference.
        p_value: two-sided bootstrap p-value (proportion of resampled diffs
            crossing zero, doubled; capped at 1.0).
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape:
        raise ValueError(f"values_a and values_b must have the same shape, got {values_a.shape} vs {values_b.shape}")
    if values_a.ndim != 1:
        raise ValueError("values_a/values_b must be 1D")
    n = values_a.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 paired observations, got {n}")

    diffs = values_a - values_b
    observed_mean_diff = float(diffs.mean())

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    alpha = 1.0 - ci
    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided bootstrap p-value: proportion of the bootstrap distribution
    # on the opposite side of zero from the observed effect, doubled.
    if observed_mean_diff >= 0:
        p_value = 2 * np.mean(boot_means <= 0)
    else:
        p_value = 2 * np.mean(boot_means >= 0)
    p_value = float(min(p_value, 1.0))

    return {
        "mean_diff": observed_mean_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
        "n": n,
    }
