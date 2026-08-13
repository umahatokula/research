"""IoU / F1 for binary water segmentation, excluding no-data pixels.

Sen1Floods11 labels use -1 for "no data / cloud-masked". Those pixels must be
excluded from both loss and metrics (handoff doc, Step 1) -- they are not a
third class, they're missing ground truth.
"""

from __future__ import annotations

import numpy as np

IGNORE_LABEL = -1


def compute_iou_f1(y_true: np.ndarray, y_pred: np.ndarray, ignore_label: int = IGNORE_LABEL) -> dict[str, float]:
    """Binary IoU and F1 for the water class (label == 1), ignoring `ignore_label` pixels.

    Args:
        y_true: array of {ignore_label, 0, 1}, any shape.
        y_pred: array of {0, 1} (predicted class), same shape as y_true.
        ignore_label: label value to exclude from both numerator and denominator.

    Returns:
        {"iou": float, "f1": float}. Returns NaN for either metric if there are
        no valid (non-ignored) pixels, or no valid pixels of the water class
        for a metric that would otherwise divide 0/0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    valid = y_true != ignore_label
    yt = y_true[valid].astype(bool)
    yp = y_pred[valid].astype(bool)

    tp = np.sum(yt & yp)
    fp = np.sum(~yt & yp)
    fn = np.sum(yt & ~yp)

    union = tp + fp + fn
    if union == 0:
        # No water in ground truth AND none predicted -- genuinely undefined,
        # not a case where precision/recall's 0/0 convention should paper over it.
        return {"iou": float("nan"), "f1": float("nan")}

    iou = float(tp / union)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"iou": iou, "f1": f1}
