# Split Audit: Official Sen1Floods11 Hand-Labeled Split

Per the handoff doc's Step 1 requirement to audit the official split before
training and document findings honestly. Produced by
`src/data/split_audit.py`, run against the actual downloaded official split
(252 train / 89 valid / 90 test, 431 chips total).

## Duplicate chip IDs across partitions

None found. Every chip ID appears in exactly one partition.

## Event overlap across partitions — found, and it's substantial

The hand-labeled subset covers **10 distinct flood events**
(Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain,
Sri-Lanka, USA) out of 12 events in the full Sen1Floods11 metadata (the
other 2 events are not represented in the hand-labeled data at all).

**All 10 events appear in all three partitions**, at roughly the same
proportions in each:

| Event      | train | valid | test |
|------------|------:|------:|-----:|
| Ghana      | 31 | 11 | 11 |
| India      | 40 | 14 | 14 |
| Mekong     | 18 |  6 |  6 |
| Nigeria    | 10 |  4 |  4 |
| Pakistan   | 16 |  6 |  6 |
| Paraguay   | 39 | 14 | 14 |
| Somalia    | 15 |  5 |  6 |
| Spain      | 18 |  6 |  6 |
| Sri-Lanka  | 24 |  9 |  9 |
| USA        | 41 | 14 | 14 |

This is consistent with roughly a 70/15/15 per-event split ratio, applied
independently within each event, rather than an event-level holdout.

## What this means

**The official split does not hold out flood events.** Chips in the test
set come from the same Sentinel-1 acquisitions (same date, same general
geographic area — each event has one acquisition date and one bounding
region per the event metadata) as chips in the training set. A model can
therefore see chips from a given flood event's imagery during training and
be evaluated on other chips from that same imagery at test time.

This does **not** mean the test metric is meaningless — this is the
standard, widely-used official Sen1Floods11 benchmark split, and using it
keeps this study comparable to other work that reports on it (per the
handoff doc's instruction to use the official split "for benchmark
comparability"). But it does mean: **test-set IoU/F1 in this study should
not be interpreted as a measure of generalization to genuinely new flood
events/geographies** — only as a measure of performance on held-out chips
within already-seen events. This caveat must be stated explicitly in the
paper's Methods/Limitations section, not left implicit.

## Not built: chip-level geometric proximity check

This audit uses "shared event" as a proxy for "same acquisition /
geographically overlapping" (justified because each event has a single
Sentinel-1 acquisition date and a single bounding polygon in the source
metadata — see `src/data/split_audit.py` docstring). A true chip-level
geometric proximity check would require the full per-chip STAC catalog
(`catalog.zip` in the bucket), which was not built. Per the handoff doc,
an event-held-out split is an optional stretch goal, not a requirement —
this audit is honest about the event-overlap finding without building the
full geometric analysis or a custom event-held-out split.
