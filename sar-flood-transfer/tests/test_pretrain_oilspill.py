from pathlib import Path

import numpy as np
import pytest

from src.models.backbone_unet import build_encoder
from src.training.pretrain_oilspill import pretrain_oilspill, split_train_val

DATASET_ROOT = Path(r"C:\Users\Umaha\Documents\phd\Oil_Spill_Detection_Dataset")

requires_real_data = pytest.mark.skipif(
    not (DATASET_ROOT / "train" / "images").exists(), reason="Krestenitis dataset not found locally"
)


def test_split_train_val_is_disjoint_and_covers_all_stems():
    stems = [f"img_{i:04d}" for i in range(20)]
    train, val = split_train_val(stems, val_fraction=0.2, seed=0)
    assert set(train) | set(val) == set(stems)
    assert set(train) & set(val) == set()
    assert len(val) == 4


def test_split_train_val_deterministic():
    stems = [f"img_{i:04d}" for i in range(20)]
    a = split_train_val(stems, 0.2, seed=1)
    b = split_train_val(stems, 0.2, seed=1)
    assert a == b


@requires_real_data
def test_pretrain_smoke_test_tiny_subset(tmp_path, monkeypatch):
    # Smoke test: 4 train stems, 2 val stems, 1 epoch, small input size.
    # Confirms the full pipeline (data -> model -> fit -> save) runs without
    # error and produces a loadable checkpoint -- NOT a claim about learned
    # oil-spill segmentation quality, which needs the real, full-scale run.
    import src.training.pretrain_oilspill as mod

    tiny_stems = ["img_0001", "img_0002", "img_0003", "img_0004", "img_0005", "img_0006"]
    monkeypatch.setattr(mod, "list_sample_stems", lambda images_dir: tiny_stems)

    output_path = tmp_path / "oilspill_encoder.weights.h5"
    history = pretrain_oilspill(
        DATASET_ROOT,
        output_path,
        input_size=(64, 64),
        batch_size=2,
        epochs=1,
        val_fraction=1 / 3,
        seed=0,
    )

    assert "loss" in history
    assert output_path.exists()

    # The saved checkpoint should be loadable via the actual "oilspill" init
    # path used by the main training matrix.
    loaded = build_encoder((64, 64, 1), init_mode="oilspill", oilspill_weights_path=output_path)
    x = np.random.rand(1, 64, 64, 1).astype(np.float32)
    out = loaded(x, training=False)
    assert out["bottleneck"].shape == (1, 2, 2, 2048)
