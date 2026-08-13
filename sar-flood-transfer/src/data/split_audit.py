"""Audit the official Sen1Floods11 train/val/test split for leakage.

Handoff doc, Step 1: check for (a) duplicate/overlapping chips across
partitions, (b) chips from the same acquisition appearing in multiple
partitions, (c) geographic proximity or flood-event overlap between train
and test. Document findings honestly -- do not claim a stronger geographic
guarantee than what actually holds, but don't block on fixing it either.

Chip IDs encode their event as a prefix ("Ghana_103272" -> event "Ghana").
Per the event metadata (Sen1Floods11_Metadata.geojson), each event has a
single Sentinel-1 acquisition date and a single bounding polygon -- so all
chips from one event come from the same acquisition and the same general
geographic area. This means "same event" is a reasonable, cheap proxy for
both (b) and (c): if two partitions share an event, they necessarily share
an acquisition and overlap geographically. This is a proxy, not a
chip-level geometric check (which would need the full per-chip STAC
catalog) -- documented as a limitation, not silently upgraded to a stronger
claim.
"""

from __future__ import annotations


def parse_event(chip_id: str) -> str:
    """'Ghana_103272' -> 'Ghana'. Chip IDs are '{Event}_{numeric id}'."""
    if "_" not in chip_id:
        raise ValueError(f"chip_id '{chip_id}' does not contain an event prefix")
    return chip_id.rsplit("_", 1)[0]


def audit_split(splits: dict[str, list[str]]) -> dict:
    """Audit a {partition_name: [chip_ids]} mapping for cross-partition leakage.

    Returns a report dict:
        duplicate_chip_ids: chip IDs present in more than one partition.
        per_partition_events: {partition: set of event names in it}.
        event_overlap: {(partition_a, partition_b): set of shared events},
            one entry per unordered pair of partitions with the pair sorted
            alphabetically by partition name.
        chip_counts_per_partition: {partition: chip count}.
        has_duplicate_leakage: bool.
        has_event_leakage: bool.
    """
    partition_names = sorted(splits.keys())

    chip_to_partitions: dict[str, list[str]] = {}
    for name in partition_names:
        for chip_id in splits[name]:
            chip_to_partitions.setdefault(chip_id, []).append(name)
    duplicate_chip_ids = sorted(cid for cid, parts in chip_to_partitions.items() if len(parts) > 1)

    per_partition_events = {name: {parse_event(cid) for cid in splits[name]} for name in partition_names}

    event_overlap: dict[tuple[str, str], set[str]] = {}
    for i, a in enumerate(partition_names):
        for b in partition_names[i + 1 :]:
            shared = per_partition_events[a] & per_partition_events[b]
            event_overlap[(a, b)] = shared

    chip_counts_per_partition = {name: len(splits[name]) for name in partition_names}

    return {
        "duplicate_chip_ids": duplicate_chip_ids,
        "per_partition_events": per_partition_events,
        "event_overlap": event_overlap,
        "chip_counts_per_partition": chip_counts_per_partition,
        "has_duplicate_leakage": len(duplicate_chip_ids) > 0,
        "has_event_leakage": any(len(shared) > 0 for shared in event_overlap.values()),
    }


def format_audit_report(report: dict) -> str:
    """Human-readable summary, for CHANNEL_COMPATIBILITY.md-style documentation."""
    lines = ["# Split Audit Report", ""]
    lines.append("## Chip counts per partition")
    for name, count in report["chip_counts_per_partition"].items():
        lines.append(f"- {name}: {count}")

    lines.append("")
    lines.append("## Duplicate chip IDs across partitions")
    if report["has_duplicate_leakage"]:
        lines.append(f"FOUND {len(report['duplicate_chip_ids'])} duplicate chip ID(s): {report['duplicate_chip_ids']}")
    else:
        lines.append("None found.")

    lines.append("")
    lines.append("## Event overlap across partitions (proxy for same-acquisition / geographic leakage)")
    if report["has_event_leakage"]:
        for (a, b), shared in report["event_overlap"].items():
            if shared:
                lines.append(f"- {a} <-> {b}: shared events = {sorted(shared)}")
    else:
        lines.append("None found -- no event appears in more than one partition.")

    lines.append("")
    lines.append("## Per-partition event counts")
    for name, events in report["per_partition_events"].items():
        lines.append(f"- {name}: {len(events)} distinct events -- {sorted(events)}")

    return "\n".join(lines)
