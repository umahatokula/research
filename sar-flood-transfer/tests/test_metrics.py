import numpy as np
import pytest

from src.training.metrics import compute_iou_f1


def test_perfect_prediction():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    result = compute_iou_f1(y_true, y_pred)
    assert result["iou"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_no_overlap():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    result = compute_iou_f1(y_true, y_pred)
    assert result["iou"] == pytest.approx(0.0)
    assert result["f1"] == pytest.approx(0.0)


def test_partial_overlap_known_values():
    # tp=2, fp=1, fn=1 -> iou = 2/4 = 0.5, precision=2/3, recall=2/3, f1=2/3
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 1])
    result = compute_iou_f1(y_true, y_pred)
    assert result["iou"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)


def test_ignore_label_excluded_from_computation():
    # Without exclusion, the -1 pixels would corrupt the count (cast to bool
    # would treat -1 as truthy). With exclusion, only the first 4 pixels count.
    y_true = np.array([1, 1, 1, 0, -1, -1])
    y_pred = np.array([1, 1, 0, 1, 1, 0])
    result = compute_iou_f1(y_true, y_pred)
    assert result["iou"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)


def test_all_ignored_returns_nan():
    y_true = np.full(4, -1)
    y_pred = np.array([0, 1, 0, 1])
    result = compute_iou_f1(y_true, y_pred)
    assert np.isnan(result["iou"])
    assert np.isnan(result["f1"])


def test_no_water_no_predictions_iou_nan_not_zero():
    # 0/0 union case (no water in truth, none predicted): union == 0, so IoU
    # should be NaN (undefined), not silently 0 or 1.
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    result = compute_iou_f1(y_true, y_pred)
    assert np.isnan(result["iou"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_iou_f1(np.array([0, 1]), np.array([0, 1, 1]))


def test_works_on_2d_arrays():
    y_true = np.array([[1, 0], [1, -1]])
    y_pred = np.array([[1, 0], [0, 0]])
    result = compute_iou_f1(y_true, y_pred)
    # valid pixels: (1,1)->tp, (0,0)->tn, (1,0)->fn ; union=1, iou=1/2
    assert result["iou"] == pytest.approx(0.5)
