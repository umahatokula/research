from src.data.label_fractions import DEFAULT_FRACTIONS
from src.training.run_matrix import (
    INITIALIZATIONS,
    SEEDS,
    SMOKE_TEST_LABEL_FRACTION,
    SMOKE_TEST_SEED,
    build_run_matrix,
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
