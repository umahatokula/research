import pandas as pd
import pytest
from PIL import Image

from src.figures.label_efficiency_curve import plot_label_efficiency_curve


def _synthetic_learning_curve_df():
    rows = []
    fractions = [0.05, 0.10, 0.25, 0.50, 1.00]
    base = {"random": 0.35, "imagenet": 0.55, "oilspill": 0.60}
    for init, start in base.items():
        for i, frac in enumerate(fractions):
            rows.append(
                {
                    "initialization": init,
                    "label_fraction": frac,
                    "iou_mean": start + 0.08 * i,
                    "iou_std": 0.02 + 0.005 * i,
                    "f1_mean": start + 0.10 + 0.08 * i,
                    "f1_std": 0.02,
                    "n_seeds": 3,
                }
            )
    return pd.DataFrame(rows)


def test_plot_creates_file_at_correct_dpi(tmp_path):
    df = _synthetic_learning_curve_df()
    out_path = tmp_path / "curve.png"
    plot_label_efficiency_curve(df, out_path, metric="iou")

    assert out_path.exists()
    img = Image.open(out_path)
    dpi = img.info.get("dpi", (0, 0))
    assert dpi[0] == pytest.approx(300, abs=1)


def test_plot_works_for_f1_metric(tmp_path):
    df = _synthetic_learning_curve_df()
    out_path = tmp_path / "curve_f1.png"
    plot_label_efficiency_curve(df, out_path, metric="f1")
    assert out_path.exists()


def test_plot_with_primary_result_annotation(tmp_path):
    df = _synthetic_learning_curve_df()
    primary_result = {
        "mean_diff": 0.08,
        "ci_low": 0.02,
        "ci_high": 0.14,
        "p_value": 0.01,
        "label_fraction": 0.10,
        "comparison": "oilspill_vs_imagenet",
    }
    out_path = tmp_path / "curve_annotated.png"
    plot_label_efficiency_curve(df, out_path, metric="iou", primary_result=primary_result)
    assert out_path.exists()


def test_rejects_missing_columns(tmp_path):
    df = pd.DataFrame({"initialization": ["random"], "label_fraction": [0.1]})
    with pytest.raises(ValueError):
        plot_label_efficiency_curve(df, tmp_path / "bad.png", metric="iou")
