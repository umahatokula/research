# Handoff Note: SAR Cross-Task Transfer Learning — Oil Spill → Flood Segmentation

## Context for whoever picks this up
This is a research paper project. The experimental design has already been debated,
critiqued, and refined across two rounds of external review — what follows is the
**final, agreed protocol**, not a first draft to be second-guessed. Implement it as
specified; flag deviations rather than silently changing the design. The goal is a
Q1-journal-credible empirical paper (target: IEEE JSTARS, stretch: IEEE TGRS).

## Working Title
**"What Transfers in SAR? Data-Efficient Flood Segmentation from Oil-Spill-Pretrained
Representations"**

## Research Question
Under what label-availability conditions does a SAR representation learned for
oil-spill segmentation transfer positively to flood-water segmentation, compared to
random and ImageNet initialization — and which encoder depth (shallow/mid/deep)
carries the transferable signal?

## Contribution Statement (use this exact framing in code comments / eventual paper)
> A controlled empirical study of cross-task transfer from oil-spill SAR segmentation
> to flood-event surface-water segmentation, quantifying label-efficiency gains
> relative to random and ImageNet initialization, and localizing transferable
> information across encoder depth.

Do not call this a "benchmark" unless splits/checkpoints/code are actually released
publicly as a reusable protocol.

---

## STEP 0 (BLOCKING — do this before anything else)

**Channel-compatibility check between the existing oil-spill model and Sen1Floods11.**

The user has an existing trained oil-spill SAR segmentation model (referred to in
prior conversation as "SpillNet" — locate it in the user's project files/repos).
Determine:
- What channel configuration does it consume? (single-channel, VV-only, VV+VH,
  three-channel replicated, or derived/composite channels?)
- What preprocessing/normalization was used during its training?

Sen1Floods11 provides **2 channels: VV and VH**, at 10 m resolution, 512×512 chips.

**Decision rule**: use the largest genuinely compatible channel set between source
and target. If the oil-spill model's input doesn't match VV/VH directly, define and
document one of these adaptation strategies (pick the one that best preserves a fair
comparison across ALL THREE initializations — same input pipeline for all conditions,
no exceptions):
- Use VV only for every initialization (simplest, most defensible).
- Construct identical two-channel inputs for all three models.
- Adapt the oil-spill model's first convolution layer (e.g., average existing
  channel weights) and document this explicitly as a limitation.
- Replace the first convolution and exclude it from the "transferred weights" claim
  (state this clearly if chosen).

**Do not proceed to Step 1 until this is resolved and documented in a short
`CHANNEL_COMPATIBILITY.md` note in the repo.**

---

## STEP 1 — Data Pipeline

**Dataset: Sen1Floods11** (public, Cloud to Street — github.com/cloudtostreet/Sen1Floods11,
also mirrored on GCS per the repo README; check for current access instructions).

Key facts (verified):
- 4,831 total chips, but only **446 are hand-labeled** by experts — use hand-labeled
  data ONLY. Do not mix in the ~4,385 weakly-labeled (NDVI/MNDWI-thresholded) chips —
  this would confound the label-efficiency measurement, which is the whole point of
  the study.
- Official IID split of the hand-labeled subset: **252 train / 89 validation / 90 test**.
  Use this official split for benchmark comparability.
- Bands: VV, VH (band order: 0=VV, 1=VH). 512×512 pixels, 10 m resolution.
- Labels: 0 = no water, 1 = water, -1 = missing/cloud-masked (exclude -1 pixels from
  loss and metrics).
- A separate permanent-water layer is provided (JRC surface water data). **Use it to
  separate permanent water from flood water** if the paper is to claim "flood
  segmentation" rather than the more conservative "surface-water segmentation." If
  this separation is skipped, all writeups/code comments must say "surface-water
  segmentation," not "flood segmentation" — this is a precise scientific claim, not
  a wording preference.

**Split auditing (do this, it's cheap and important):**
Before training, audit the official split for:
- Duplicate or overlapping chips across train/val/test
- Chips from the same original satellite acquisition appearing in multiple partitions
- Geographic proximity or flood-event overlap between train and test

Document findings honestly. If the official split has cross-partition event overlap,
say so explicitly in code comments/notes — do not claim a stronger geographic
guarantee than what actually holds. A fully event-held-out split is a stretch goal,
not a requirement — don't block on building one unless time permits.

**Label-fraction subsets — must be NESTED, not independently resampled:**
Construct 5%, 10%, 25%, 50%, 100% subsets of the 252 training chips such that
`D_5% ⊂ D_10% ⊂ D_25% ⊂ D_50% ⊂ D_100%`. This ensures performance changes are
attributable to additional labels, not to different sampled scenes. Fix this
nesting per random seed (i.e., the nesting structure can differ across the 3 seeds,
but must be strictly nested within each seed).

---

## STEP 2 — Baseline Training Matrix

**Three initializations only** (do not add SAR self-supervised or foundation-model
baselines like SARATR-X at this stage — explicitly deferred to Future Work, see
below):
1. Random initialization
2. ImageNet initialization (standard pretrained backbone weights)
3. Oil-spill SAR pretraining (the existing model's weights, per Step 0's resolved
   channel-compatibility strategy)

**Training matrix**: 3 initializations × 5 label fractions × 3 seeds = **45 runs**.

Requirements for every run:
- Same architecture, same optimizer, same number of training epochs/early-stopping
  criterion, same augmentation — only the initialization and label fraction change.
  This is essential for a fair comparison; do not let hyperparameters drift between
  conditions.
- Same preprocessing/normalization pipeline across all three initializations (per
  Step 0's resolution).
- Report IoU and F1 on the held-out test set (90 chips) for every run.
- **Smoke-test first**: before committing to all 45 runs, run one configuration
  end-to-end (e.g., 10% label fraction, 1 seed, all 3 initializations) to confirm the
  pipeline works and produces sane metrics. Do not scale up until this passes.

**Output**: a results table/CSV with columns
`[initialization, label_fraction, seed, IoU, F1]` — this is the core data for the
label-efficiency curve (the paper's headline figure).

---

## STEP 3 — Frozen-Feature Segmentation Probing

Only run this AFTER Step 2's main results look sane and sensible.

For each of the 3 initializations, freeze the encoder and probe at **3 depths**:
early / middle / bottleneck (deep).

**Correct probe formulation for segmentation (NOT a classification-style linear
probe):**

```
Ŷ_l = Upsample( Conv_1x1( F_l ) )
```

where `F_l` is the frozen feature map at encoder depth `l`. Upsample to the original
label resolution before computing metrics. Report **IoU/F1** at each depth, not
generic "probe accuracy."

Matrix: 3 initializations × 3 depths × 3 seeds = **27 lightweight probe runs**
(cheap relative to full fine-tuning since the encoder is frozen).

Call this **"frozen-feature segmentation probing"** in code/comments — more precise
than "linear probing," which is borrowed from classification and doesn't directly
apply here.

---

## STEP 4 — Statistics

**Do NOT** run a Wilcoxon test on n=3 seed-level observations (too few observations,
inflated significance risk). Instead:

1. Train each configuration with 3 seeds (already done in Step 2).
2. Generate predictions for every held-out test scene under each condition.
3. Average predictions or per-scene metrics across seeds.
4. Compare oil-spill vs. ImageNet initialization using **paired scene-level
   differences** (not pixel-level — pixels are not independent observations and
   will produce artificially tiny p-values).
5. Compute confidence intervals via **paired bootstrap over independent test
   scenes/events** (not over seeds).

**Multiple comparisons**: since there are 5 label fractions, either:
- Apply Holm correction across the 5 comparisons, OR
- **(Preferred, cleaner)** Declare ONE label fraction (10% or 25%) as the primary
  label-scarce comparison for hypothesis testing, and present the other fractions
  as supporting learning-curve results without individual significance claims.

---

## Explicitly Out of Scope (Future Work — do not implement now)
- SAR self-supervised / foundation-model baselines (e.g., SARATR-X) — flag as a
  known limitation in the eventual paper, not a silent omission.
- Ship detection/segmentation as a second target task — single target (flood/
  surface-water) only for this paper.
- Full battery of representation-analysis methods (CKA, UMAP, LogME, LEEP,
  reinitialization sweeps) — only the frozen-feature probing above.
- Event-held-out split — stretch goal only if time permits after the core protocol
  is done.

---

## Suggested Order of Operations
1. **Step 0** (channel compatibility) — blocking, do first, produce
   `CHANNEL_COMPATIBILITY.md`.
2. **Step 1** (data pipeline: download, hand-labeled subset, split audit, nested
   label-fraction construction).
3. **Step 2 smoke test** (1 config, all 3 inits, 10% labels, 1 seed) before scaling.
4. **Step 2 full run** (45 runs).
5. **Step 3** (27 probe runs).
6. **Step 4** (bootstrap statistics on the Step 2 results).
7. Export all results as CSVs + publication-quality figures (300 DPI, labeled axes,
   legends, captions) — same standard as the prior conjunction-risk-paper notebook
   this user has already built, for consistency of style across their two papers.
   Headline figure: label-efficiency curve (IoU/F1 vs. label fraction, one line per
   initialization, with bootstrap CIs as shaded bands).

## When Code/Results Are Done
Do not draft the paper's Methods/Results/Discussion until real results exist —
Introduction and Related Work can be drafted in parallel since they don't depend on
outcomes, but the rest waits for actual numbers, mirroring how the user's prior
conjunction-risk paper was developed (results first, writing after, with findings
allowed to reshape claims rather than the other way around).
