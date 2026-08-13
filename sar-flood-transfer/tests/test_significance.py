import numpy as np
import pandas as pd
import pytest

from src.stats.significance import (
    PRIMARY_LABEL_FRACTION,
    learning_curve_summary,
    load_per_chip_metrics,
    primary_hypothesis_test,
)


def _write_per_chip_csv(per_chip_dir, initialization, label_fraction, seed, chip_ious):
    per_chip_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "chip_id": list(chip_ious.keys()),
            "iou": list(chip_ious.values()),
            "f1": [v + 0.05 for v in chip_ious.values()],  # arbitrary but deterministic f1
        }
    )
    df.to_csv(per_chip_dir / f"{initialization}_frac{label_fraction}_seed{seed}.csv", index=False)


def test_load_per_chip_metrics_averages_across_seeds(tmp_path):
    # Same 2 chips, 3 seeds, known IoU values -> mean should be exact.
    for seed, ious in zip((0, 1, 2), ({"a": 0.6, "b": 0.4}, {"a": 0.8, "b": 0.6}, {"a": 0.7, "b": 0.5})):
        _write_per_chip_csv(tmp_path, "oilspill", 0.10, seed, ious)

    result = load_per_chip_metrics(tmp_path, "oilspill", 0.10)
    assert result.loc["a", "iou"] == pytest.approx((0.6 + 0.8 + 0.7) / 3)
    assert result.loc["b", "iou"] == pytest.approx((0.4 + 0.6 + 0.5) / 3)


def test_load_per_chip_metrics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_per_chip_metrics(tmp_path, "oilspill", 0.10)


def test_primary_hypothesis_test_detects_clear_effect(tmp_path):
    chips = [f"chip_{i}" for i in range(20)]
    rng = np.random.default_rng(0)
    base = dict(zip(chips, rng.uniform(0.4, 0.6, size=20)))

    for seed in (0, 1, 2):
        oilspill_ious = {c: base[c] + 0.15 for c in chips}  # oilspill consistently better
        imagenet_ious = {c: base[c] for c in chips}
        _write_per_chip_csv(tmp_path, "oilspill", PRIMARY_LABEL_FRACTION, seed, oilspill_ious)
        _write_per_chip_csv(tmp_path, "imagenet", PRIMARY_LABEL_FRACTION, seed, imagenet_ious)

    result = primary_hypothesis_test(tmp_path, n_boot=2000, bootstrap_seed=1)
    assert result["mean_diff"] == pytest.approx(0.15, abs=1e-6)
    assert result["ci_low"] > 0.0
    assert result["comparison"] == "oilspill_vs_imagenet"
    assert result["label_fraction"] == PRIMARY_LABEL_FRACTION
    assert result["n_chips"] == 20


def test_primary_hypothesis_test_uses_only_common_chips(tmp_path):
    for seed in (0, 1, 2):
        _write_per_chip_csv(tmp_path, "oilspill", PRIMARY_LABEL_FRACTION, seed, {"a": 0.6, "b": 0.5, "only_oilspill": 0.9})
        _write_per_chip_csv(tmp_path, "imagenet", PRIMARY_LABEL_FRACTION, seed, {"a": 0.5, "b": 0.4, "only_imagenet": 0.1})

    result = primary_hypothesis_test(tmp_path, n_boot=500, bootstrap_seed=0)
    assert result["n_chips"] == 2  # only 'a' and 'b' are common to both


def test_primary_hypothesis_test_raises_when_no_common_chips(tmp_path):
    for seed in (0, 1, 2):
        _write_per_chip_csv(tmp_path, "oilspill", PRIMARY_LABEL_FRACTION, seed, {"a": 0.6})
        _write_per_chip_csv(tmp_path, "imagenet", PRIMARY_LABEL_FRACTION, seed, {"b": 0.5})

    with pytest.raises(ValueError):
        primary_hypothesis_test(tmp_path, n_boot=100)


def test_learning_curve_summary_computes_mean_and_std(tmp_path):
    df = pd.DataFrame(
        {
            "initialization": ["random", "random", "random", "imagenet", "imagenet", "imagenet"],
            "label_fraction": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            "seed": [0, 1, 2, 0, 1, 2],
            "iou": [0.4, 0.5, 0.6, 0.6, 0.7, 0.8],
            "f1": [0.5, 0.6, 0.7, 0.7, 0.8, 0.9],
        }
    )
    csv_path = tmp_path / "step2.csv"
    df.to_csv(csv_path, index=False)

    summary = learning_curve_summary(csv_path)
    random_row = summary[summary["initialization"] == "random"].iloc[0]
    assert random_row["iou_mean"] == pytest.approx(0.5)
    assert random_row["n_seeds"] == 3
    imagenet_row = summary[summary["initialization"] == "imagenet"].iloc[0]
    assert imagenet_row["iou_mean"] == pytest.approx(0.7)
