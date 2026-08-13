import pandas as pd

from src.data.label_fractions import DEFAULT_FRACTIONS
from src.training.run_matrix import (
    INITIALIZATIONS,
    SEEDS,
    SMOKE_TEST_LABEL_FRACTION,
    SMOKE_TEST_SEED,
    build_run_matrix,
    get_completed_configs,
)


def test_smoke_test_matrix_has_one_config_per_initialization():
    configs = build_run_matrix(smoke_test=True)
    assert len(configs) == len(INITIALIZATIONS)
    for init, frac, seed in configs:
        assert init in INITIALIZATIONS
        assert frac == SMOKE_TEST_LABEL_FRACTION
        assert seed == SMOKE_TEST_SEED


def test_full_matrix_has_45_configs():
    configs = build_run_matrix(smoke_test=False)
    assert len(configs) == 45  # 3 initializations x 5 fractions x 3 seeds


def test_full_matrix_covers_every_combination_exactly_once():
    configs = build_run_matrix(smoke_test=False)
    expected = {
        (init, frac, seed) for init in INITIALIZATIONS for frac in DEFAULT_FRACTIONS for seed in SEEDS
    }
    assert set(configs) == expected
    assert len(configs) == len(set(configs))  # no duplicates


def test_get_completed_configs_empty_when_no_csv(tmp_path):
    assert get_completed_configs(tmp_path / "does_not_exist.csv") == set()


def test_get_completed_configs_reads_existing_rows(tmp_path):
    csv_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {"initialization": "random", "label_fraction": 0.10, "seed": 0, "iou": 0.5, "f1": 0.6},
            {"initialization": "imagenet", "label_fraction": 0.25, "seed": 1, "iou": 0.7, "f1": 0.8},
        ]
    ).to_csv(csv_path, index=False)

    completed = get_completed_configs(csv_path)
    assert completed == {("random", 0.10, 0), ("imagenet", 0.25, 1)}


def test_get_completed_configs_handles_empty_csv(tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["initialization", "label_fraction", "seed", "iou", "f1"]).to_csv(csv_path, index=False)
    assert get_completed_configs(csv_path) == set()
