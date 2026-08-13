from pathlib import Path

import pandas as pd
import pytest

from src.training.frozen_probe import DEPTHS, INITIALIZATIONS, SEEDS, build_probe_matrix, run_single_probe

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"
requires_real_data = pytest.mark.skipif(not (DATA_DIR / "S1Hand").exists(), reason="Sen1Floods11 data not downloaded")


def test_probe_matrix_has_27_configs():
    configs = build_probe_matrix()
    assert len(configs) == 27  # 3 inits x 3 depths x 3 seeds


def test_probe_matrix_covers_every_combination_exactly_once():
    configs = build_probe_matrix()
    expected = {(init, depth, seed) for init in INITIALIZATIONS for depth in DEPTHS for seed in SEEDS}
    assert set(configs) == expected
    assert len(configs) == len(set(configs))


@requires_real_data
def test_run_single_probe_smoke_test_tiny_subset(tmp_path, monkeypatch):
    import src.training.frozen_probe as mod
    from src.data.download_sen1floods11 import chip_id_from_s1_filename, download_splits

    real_splits = download_splits(DATA_DIR)
    real_chip_splits = {name: [chip_id_from_s1_filename(s1) for s1, _ in pairs] for name, pairs in real_splits.items()}
    tiny_splits = {
        "train": real_chip_splits["train"][:2],
        "valid": real_chip_splits["valid"][:1],
        "test": real_chip_splits["test"][:1],
    }
    monkeypatch.setattr(mod, "get_chip_id_splits", lambda data_dir: tiny_splits)

    output_csv = tmp_path / "step3_results.csv"
    result = run_single_probe(
        initialization="random",
        depth="bottleneck",
        seed=0,
        data_dir=DATA_DIR,
        output_csv=output_csv,
        batch_size=1,
        epochs=1,
    )

    assert "iou" in result and "f1" in result
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert list(df.columns) == ["initialization", "depth", "seed", "iou", "f1"]
    assert df.iloc[0]["depth"] == "bottleneck"


@requires_real_data
def test_run_single_probe_rejects_bad_depth():
    with pytest.raises(ValueError):
        run_single_probe(
            initialization="random", depth="not_a_depth", seed=0, data_dir=DATA_DIR, output_csv=Path("unused.csv")
        )


@requires_real_data
def test_run_single_probe_rejects_oilspill_without_weights():
    with pytest.raises(ValueError):
        run_single_probe(
            initialization="oilspill", depth="early", seed=0, data_dir=DATA_DIR, output_csv=Path("unused.csv")
        )
