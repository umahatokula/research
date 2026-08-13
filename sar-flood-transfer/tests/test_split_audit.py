import pytest

from src.data.split_audit import audit_split, parse_event


def test_parse_event_basic():
    assert parse_event("Ghana_103272") == "Ghana"
    assert parse_event("Bolivia_23014") == "Bolivia"


def test_parse_event_rejects_no_underscore():
    with pytest.raises(ValueError):
        parse_event("NoUnderscoreHere")


def test_no_leakage_case():
    splits = {
        "train": ["Ghana_1", "Ghana_2", "Bolivia_1"],
        "valid": ["Nigeria_1"],
        "test": ["Spain_1", "Spain_2"],
    }
    report = audit_split(splits)
    assert report["has_duplicate_leakage"] is False
    assert report["has_event_leakage"] is False
    assert report["duplicate_chip_ids"] == []
    assert report["chip_counts_per_partition"] == {"train": 3, "valid": 1, "test": 2}


def test_detects_duplicate_chip_across_partitions():
    splits = {
        "train": ["Ghana_1", "Ghana_2"],
        "test": ["Ghana_2", "Bolivia_1"],  # Ghana_2 duplicated
    }
    report = audit_split(splits)
    assert report["has_duplicate_leakage"] is True
    assert report["duplicate_chip_ids"] == ["Ghana_2"]


def test_detects_event_overlap_without_chip_duplication():
    # Different chip IDs, same event -- this is the leakage case the doc cares
    # about most (same acquisition/region split across train and test).
    splits = {
        "train": ["Ghana_1", "Ghana_2"],
        "test": ["Ghana_3", "Bolivia_1"],
    }
    report = audit_split(splits)
    assert report["has_duplicate_leakage"] is False
    assert report["has_event_leakage"] is True
    assert report["event_overlap"][("test", "train")] == {"Ghana"}


def test_event_overlap_keys_are_alphabetically_sorted_pairs():
    splits = {"train": ["Ghana_1"], "valid": ["Ghana_2"], "test": ["Ghana_3"]}
    report = audit_split(splits)
    assert set(report["event_overlap"].keys()) == {("test", "train"), ("test", "valid"), ("train", "valid")}


def test_per_partition_events_correct():
    splits = {"train": ["Ghana_1", "Bolivia_1", "Ghana_2"]}
    report = audit_split(splits)
    assert report["per_partition_events"]["train"] == {"Ghana", "Bolivia"}
