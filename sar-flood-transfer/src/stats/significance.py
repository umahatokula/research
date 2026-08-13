"""Step 4: primary hypothesis test + supporting learning-curve summary.

Per the handoff doc: with 5 label fractions, either apply a Holm correction
across all 5, or (preferred, cleaner) declare ONE fraction as the primary
label-scarce comparison and present the rest as a descriptive learning curve
without individual significance claims. We take the preferred option.

Primary label fraction: 10%. Rationale: this is the paper's most
label-scarce condition, the sharpest test of whether oil-spill pretraining
helps when labels are hardest to get -- directly matching the paper's
"data-efficient" framing. 25% would be a reasonable alternative per the
handoff doc; 10% was chosen, not defaulted to, and is recorded here so the
choice is visible rather than buried in a function argument default.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.stats.bootstrap import paired_bootstrap_ci

PRIMARY_LABEL_FRACTION = 0.10
PRIMARY_COMPARISON = ("oilspill", "imagenet")


def load_per_chip_metrics(
    per_chip_dir: Path, initialization: str, label_fraction: float, seeds: tuple[int, ...] = (0, 1, 2)
) -> pd.DataFrame:
    """Average per-chip IoU/F1 across seeds for one (initialization, label_fraction) config.

    Returns a DataFrame indexed by chip_id with columns [iou, f1], averaged
    across the given seeds (handoff doc, Step 4: "average per-scene metrics
    across seeds" before computing paired differences).
    """
    per_chip_dir = Path(per_chip_dir)
    frames = []
    for seed in seeds:
        path = per_chip_dir / f"{initialization}_frac{label_fraction}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing per-chip results: {path}")
        frames.append(pd.read_csv(path).set_index("chip_id"))
    combined = pd.concat(frames)
    return combined.groupby("chip_id")[["iou", "f1"]].mean()


def primary_hypothesis_test(
    per_chip_dir: Path,
    label_fraction: float = PRIMARY_LABEL_FRACTION,
    metric: str = "iou",
    seeds: tuple[int, ...] = (0, 1, 2),
    n_boot: int = 10000,
    bootstrap_seed: int = 0,
) -> dict:
    """Paired scene-level bootstrap comparison: oilspill vs. imagenet, at the primary label fraction."""
    init_a, init_b = PRIMARY_COMPARISON
    metrics_a = load_per_chip_metrics(per_chip_dir, init_a, label_fraction, seeds)
    metrics_b = load_per_chip_metrics(per_chip_dir, init_b, label_fraction, seeds)

    common_chips = sorted(set(metrics_a.index) & set(metrics_b.index))
    if not common_chips:
        raise ValueError(f"No common test chips between {init_a} and {init_b} at label_fraction={label_fraction}")

    a_values = metrics_a.loc[common_chips, metric].to_numpy()
    b_values = metrics_b.loc[common_chips, metric].to_numpy()

    result = paired_bootstrap_ci(a_values, b_values, n_boot=n_boot, seed=bootstrap_seed)
    result.update(
        {
            "label_fraction": label_fraction,
            "metric": metric,
            "comparison": f"{init_a}_vs_{init_b}",
            "n_chips": len(common_chips),
        }
    )
    return result


def learning_curve_summary(step2_csv_path: Path) -> pd.DataFrame:
    """Descriptive mean +/- std IoU/F1 across seeds, per (initialization, label_fraction).

    No significance claims here -- this feeds the headline label-efficiency
    figure, per the handoff doc's "supporting learning-curve results
    without individual significance claims" for the non-primary fractions.
    """
    df = pd.read_csv(step2_csv_path)
    summary = (
        df.groupby(["initialization", "label_fraction"])
        .agg(
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
        .sort_values(["initialization", "label_fraction"])
    )
    return summary
