"""Headline figure: label-efficiency curve (handoff doc, Step 7).

IoU (or F1) vs. label fraction, one line per initialization, with +/-1 SD
bands across seeds (n=3) -- a descriptive uncertainty band, not the Step 4
formal bootstrap CI (which only exists at the single primary label
fraction). The primary-fraction comparison's bootstrap CI is optionally
annotated separately so the figure doesn't imply inferential precision it
doesn't have at the non-primary fractions.

Colors: fixed categorical assignment (never cycled/reordered), validated
colorblind-safe via the dataviz skill's palette validator (light mode: all
checks pass; the aqua slot's sub-3:1 contrast against the white figure
background is why every line also gets a direct end-of-line label, not just
a legend).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # this module only saves PNGs, never displays interactively --
# pinning a non-interactive backend avoids picking up a Tk backend that may not
# have a usable tk.tcl installed (observed intermittently in this environment).
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INIT_ORDER = ("random", "imagenet", "oilspill")
INIT_COLORS = {"random": "#2a78d6", "imagenet": "#eb6834", "oilspill": "#1baf7a"}
INIT_LABELS = {"random": "Random init", "imagenet": "ImageNet init", "oilspill": "Oil-spill-pretrained"}


def plot_label_efficiency_curve(
    learning_curve_df: pd.DataFrame,
    output_path: Path,
    metric: str = "iou",
    primary_result: dict | None = None,
    title: str | None = None,
) -> None:
    """Plot the label-efficiency curve and save as a 300 DPI PNG.

    Args:
        learning_curve_df: output of src.stats.significance.learning_curve_summary()
            -- columns [initialization, label_fraction, {metric}_mean, {metric}_std, n_seeds].
        output_path: where to save the PNG.
        metric: "iou" or "f1".
        primary_result: optional output of src.stats.significance.primary_hypothesis_test(),
            annotated as a bracket at the primary label fraction.
        title: figure title; a sensible default is used if omitted.
    """
    mean_col, std_col = f"{metric}_mean", f"{metric}_std"
    missing = {"initialization", "label_fraction", mean_col, std_col} - set(learning_curve_df.columns)
    if missing:
        raise ValueError(f"learning_curve_df missing required columns: {missing}")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for init in INIT_ORDER:
        subset = learning_curve_df[learning_curve_df["initialization"] == init].sort_values("label_fraction")
        if subset.empty:
            continue
        x = subset["label_fraction"] * 100  # as percent
        y = subset[mean_col]
        std = subset[std_col].fillna(0.0)
        color = INIT_COLORS[init]

        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=8, label=INIT_LABELS[init], zorder=3)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.15, linewidth=0, zorder=1)

        # Direct end-of-line label (relief for the aqua slot's low contrast,
        # and generally more readable than a legend alone).
        ax.annotate(
            INIT_LABELS[init],
            xy=(x.iloc[-1], y.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold",
        )

    ax.set_xlabel("Training label fraction (%)")
    ax.set_ylabel(metric.upper())
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_title(title or f"Flood-segmentation label efficiency ({metric.upper()})")
    ax.grid(True, which="both", color="#e0dfda", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Right margin for the direct end-of-line labels, which sit past x=100%.
    ax.set_xlim(right=ax.get_xlim()[1] * 1.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    if primary_result is not None:
        frac_pct = primary_result["label_fraction"] * 100
        ax.axvline(frac_pct, color="#52514e", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
        ci_low, ci_high = primary_result["ci_low"], primary_result["ci_high"]
        p = primary_result["p_value"]
        y_bottom = ax.get_ylim()[0]
        y_range = ax.get_ylim()[1] - y_bottom
        ax.annotate(
            f"primary comparison ({primary_result['comparison'].replace('_', ' ')}):\n"
            f"Δ{metric.upper()} = {primary_result['mean_diff']:.3f}, "
            f"95% CI [{ci_low:.3f}, {ci_high:.3f}], p={p:.3f}",
            xy=(frac_pct, y_bottom + 0.04 * y_range),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color="#52514e",
            va="bottom",
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    from src.stats.significance import PRIMARY_LABEL_FRACTION, learning_curve_summary, primary_hypothesis_test

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step2-csv", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "step2_training_matrix.csv"
    )
    parser.add_argument(
        "--per-chip-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "per_chip_metrics"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "figures"
    )
    args = parser.parse_args()

    curve_df = learning_curve_summary(args.step2_csv)
    try:
        primary = primary_hypothesis_test(args.per_chip_dir, label_fraction=PRIMARY_LABEL_FRACTION)
    except FileNotFoundError:
        primary = None
        print("Per-chip results for the primary comparison not found; plotting without CI annotation.")

    for metric in ("iou", "f1"):
        out_path = args.output_dir / f"label_efficiency_curve_{metric}.png"
        plot_label_efficiency_curve(curve_df, out_path, metric=metric, primary_result=primary)
        print(f"Saved {out_path}")
