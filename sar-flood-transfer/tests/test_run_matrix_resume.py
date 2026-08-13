import pandas as pd

import src.training.run_matrix as run_matrix_mod
from src.training.run_matrix import build_run_matrix, run_matrix


def test_run_matrix_skips_already_completed_configs(tmp_path, monkeypatch):
    output_csv = tmp_path / "results.csv"
    # Pre-populate with 2 of the 3 smoke-test configs already "done".
    pd.DataFrame(
        [
            {"initialization": "random", "label_fraction": 0.10, "seed": 0, "iou": 0.5, "f1": 0.6},
            {"initialization": "imagenet", "label_fraction": 0.10, "seed": 0, "iou": 0.6, "f1": 0.7},
        ]
    ).to_csv(output_csv, index=False)

    calls = []

    def fake_run_single_training(initialization, label_fraction, seed, **kwargs):
        calls.append((initialization, label_fraction, seed))
        # Simulate the real function appending its own row.
        pd.DataFrame(
            [{"initialization": initialization, "label_fraction": label_fraction, "seed": seed, "iou": 0.9, "f1": 0.9}]
        ).to_csv(output_csv, mode="a", header=False, index=False)
        return {"iou": 0.9, "f1": 0.9}

    monkeypatch.setattr(run_matrix_mod, "run_single_training", fake_run_single_training)

    run_matrix(
        data_dir=tmp_path,
        output_csv=output_csv,
        per_chip_dir=tmp_path / "per_chip",
        oilspill_weights_path=tmp_path / "weights.h5",
        smoke_test=True,
    )

    # Only the missing "oilspill" config should have actually run.
    assert calls == [("oilspill", 0.10, 0)]


def test_run_matrix_runs_everything_when_csv_is_fresh(tmp_path, monkeypatch):
    output_csv = tmp_path / "results.csv"
    calls = []

    def fake_run_single_training(initialization, label_fraction, seed, **kwargs):
        calls.append((initialization, label_fraction, seed))
        pd.DataFrame(
            [{"initialization": initialization, "label_fraction": label_fraction, "seed": seed, "iou": 0.9, "f1": 0.9}]
        ).to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
        return {"iou": 0.9, "f1": 0.9}

    monkeypatch.setattr(run_matrix_mod, "run_single_training", fake_run_single_training)

    run_matrix(
        data_dir=tmp_path,
        output_csv=output_csv,
        per_chip_dir=tmp_path / "per_chip",
        oilspill_weights_path=tmp_path / "weights.h5",
        smoke_test=True,
    )

    assert set(calls) == set(build_run_matrix(smoke_test=True))
