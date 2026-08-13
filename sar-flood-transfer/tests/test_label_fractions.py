import pytest

from src.data.label_fractions import DEFAULT_FRACTIONS, build_nested_fractions


def _chip_ids(n=252):
    return [f"chip_{i}" for i in range(n)]


def test_counts_match_expected_fractions():
    subsets = build_nested_fractions(_chip_ids(252), seed=0)
    assert len(subsets[0.05]) == round(0.05 * 252)
    assert len(subsets[0.10]) == round(0.10 * 252)
    assert len(subsets[0.25]) == round(0.25 * 252)
    assert len(subsets[0.50]) == round(0.50 * 252)
    assert len(subsets[1.00]) == 252


def test_strict_nesting_holds():
    subsets = build_nested_fractions(_chip_ids(252), seed=0)
    fracs = sorted(subsets.keys())
    for smaller, larger in zip(fracs, fracs[1:]):
        assert set(subsets[smaller]).issubset(set(subsets[larger])), (
            f"{smaller} subset not nested inside {larger} subset"
        )


def test_100_percent_is_all_chips():
    ids = _chip_ids(252)
    subsets = build_nested_fractions(ids, seed=0)
    assert set(subsets[1.00]) == set(ids)


def test_deterministic_for_same_seed():
    ids = _chip_ids(252)
    a = build_nested_fractions(ids, seed=42)
    b = build_nested_fractions(ids, seed=42)
    assert a == b


def test_different_seeds_can_differ():
    ids = _chip_ids(252)
    a = build_nested_fractions(ids, seed=1)
    b = build_nested_fractions(ids, seed=2)
    # Not a hard guarantee for all inputs, but with 252 chips at 5% (13 chips)
    # the odds of an identical shuffle prefix by chance are astronomically low.
    assert set(a[0.05]) != set(b[0.05])


def test_nesting_holds_across_all_seeds_used_in_study():
    ids = _chip_ids(252)
    for seed in (0, 1, 2):
        subsets = build_nested_fractions(ids, seed=seed)
        fracs = sorted(subsets.keys())
        for smaller, larger in zip(fracs, fracs[1:]):
            assert set(subsets[smaller]).issubset(set(subsets[larger]))


def test_rejects_unsorted_fractions():
    with pytest.raises(ValueError):
        build_nested_fractions(_chip_ids(10), fractions=(0.5, 0.1), seed=0)


def test_rejects_empty_chip_list():
    with pytest.raises(ValueError):
        build_nested_fractions([], seed=0)


def test_rejects_out_of_range_fraction():
    with pytest.raises(ValueError):
        build_nested_fractions(_chip_ids(10), fractions=(0.0, 0.5), seed=0)
    with pytest.raises(ValueError):
        build_nested_fractions(_chip_ids(10), fractions=(0.5, 1.5), seed=0)


def test_default_fractions_used_when_unspecified():
    subsets = build_nested_fractions(_chip_ids(252), seed=0)
    assert set(subsets.keys()) == set(DEFAULT_FRACTIONS)


def test_small_fraction_never_produces_empty_subset():
    # 5% of a small chip list should still return at least 1 chip, not 0.
    subsets = build_nested_fractions(_chip_ids(5), fractions=(0.05, 1.0), seed=0)
    assert len(subsets[0.05]) >= 1
