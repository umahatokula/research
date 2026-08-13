from pathlib import Path

import pandas as pd
import pytest

from src.data.download_sen1floods11 import download_splits
from src.training.train_flood_segmentation import get_chip_id_splits, run_single_training

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"
requires_real_data = pytest.mark.skipif(
    not (DATA_DIR / "S1Hand").exists(), reason="Sen1Floods11 data not downloaded locally"
)


@requires_real_data
def test_get_chip_id_splits_matches_official_counts():
    splits = get_chip_id_splits(DATA_DIR)
    assert len(splits["train"]) == 252
    assert len(splits["valid"]) == 89
    assert len(splits["test"]) == 90
    # Real chip IDs, not filenames -- shouldn't contain the .tif suffix.
    assert not any(cid.endswith(".tif") for cid in splits["train"])


@requires_real_data
def test_run_single_training_smoke_test_tiny_subset(tmp_path, monkeypatch):
    # Smoke test per handoff doc Step 2: confirm the full pipeline (nested
    # fractions -> model -> fit -> eval -> CSV row) runs end-to-end without
    # error, on a tiny real-data subset (2 train / 1 val / 1 test chip,
    # 1 epoch). NOT a claim about learned segmentation quality.
    real_splits = download_splits(DATA_DIR)
    from src.data.download_sen1floods11 import chip_id_from_s1_filename

    real_chip_splits = {name: [chip_id_from_s1_filename(s1) for s1, _ in pairs] for name, pairs in real_splits.items()}
    tiny_splits = {
        "train": real_chip_splits["train"][:2],
        "valid": real_chip_splits["valid"][:1],
        "test": real_chip_splits["test"][:1],
    }

    import src.training.train_flood_segmentation as mod

    monkeypatch.setattr(mod, "get_chip_id_splits", lambda data_dir: tiny_splits)

    output_csv = tmp_path / "step2_results.csv"
    per_chip_dir = tmp_path / "per_chip"

    result = run_single_training(
        initialization="random",
        label_fraction=1.0,
        seed=0,
        data_dir=DATA_DIR,
        output_csv=output_csv,
        per_chip_dir=per_chip_dir,
        batch_size=1,
        epochs=1,
    )

    assert "iou" in result and "f1" in result
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert list(df.columns) == ["initialization", "label_fraction", "seed", "iou", "f1"]
    assert len(df) == 1
    assert df.iloc[0]["initialization"] == "random"

    per_chip_files = list(per_chip_dir.glob("*.csv"))
    assert len(per_chip_files) == 1
    per_chip_df = pd.read_csv(per_chip_files[0])
    assert len(per_chip_df) == 1  # 1 test chip
    assert set(per_chip_df.columns) == {"chip_id", "iou", "f1"}


@requires_real_data
def test_run_single_training_rejects_oilspill_without_weights():
    with pytest.raises(ValueError):
        run_single_training(
            initialization="oilspill",
            label_fraction=1.0,
            seed=0,
            data_dir=DATA_DIR,
            output_csv=Path("unused.csv"),
        )
