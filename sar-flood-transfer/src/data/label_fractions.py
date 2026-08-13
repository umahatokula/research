"""Nested label-fraction subsets of the training chips.

Handoff doc, Step 1: D_5% subset D_10% subset D_25% subset D_50% subset D_100%,
per seed. This isolates "more labels helped" from "different scenes got
sampled" -- nesting is achieved by shuffling the chip list once per seed and
taking increasing prefixes, which trivially guarantees strict nesting.
"""

from __future__ import annotations

import random

DEFAULT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)


def build_nested_fractions(
    chip_ids: list[str],
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> dict[float, list[str]]:
    """Return {fraction: [chip_ids]} where subsets are strictly nested by prefix.

    Args:
        chip_ids: the full training-set chip IDs (order doesn't matter -- this
            function shuffles them deterministically from `seed`).
        fractions: fractions to build, must be sorted ascending, each in (0, 1].
        seed: RNG seed. The shuffle (and hence the resulting subsets) differs
            per seed, but nesting is guaranteed within a given seed.

    Returns:
        Dict mapping each fraction to its chip-ID list, in shuffled order.
        `result[fractions[i]]` is always a prefix of `result[fractions[i+1]]`.
    """
    if not chip_ids:
        raise ValueError("chip_ids must be non-empty")
    if list(fractions) != sorted(fractions):
        raise ValueError(f"fractions must be sorted ascending, got {fractions}")
    for f in fractions:
        if not (0.0 < f <= 1.0):
            raise ValueError(f"each fraction must be in (0, 1], got {f}")

    shuffled = list(chip_ids)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    result: dict[float, list[str]] = {}
    for frac in fractions:
        count = max(1, round(frac * n))
        result[frac] = shuffled[:count]
    return result
