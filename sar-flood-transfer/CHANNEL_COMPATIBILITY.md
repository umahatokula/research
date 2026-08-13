# Channel & Architecture Compatibility: Oil-Spill Pretraining → Sen1Floods11

This note supersedes the first-pass version of this document (which explored
adapting SpillNet's existing `best_model.keras` directly). Two rounds of
discussion with the user changed the resolved approach — recorded here in full,
per the handoff doc's rule to flag deviations rather than silently change the
design.

## Deviation from the handoff doc's Step 0 framing

The handoff doc's Step 0 assumes reuse of an existing trained oil-spill model
("SpillNet", `C:\Users\Umaha\Documents\phd\best_model.keras`). Investigation
found:

- SpillNet is a **custom** 5-block encoder-decoder (32→64→128→256→512 filters
  + a custom SE-style bottleneck) with no off-the-shelf ImageNet-pretrained
  checkpoint. The handoff doc's Step 2 requires "same architecture ... only
  initialization differs" across all 3 conditions (random / ImageNet /
  oil-spill). A real ImageNet-pretrained checkpoint only exists for named,
  standard architectures (ResNet, VGG, EfficientNet, etc.), not for SpillNet's
  custom shape.
- SpillNet was trained on the Krestenitis "Oil Spill Detection Dataset", whose
  images are **RGB JPEG renderings** of SAR imagery, not raw backscatter — the
  3 channels have no confirmed physical (polarization) meaning.

**Decision (confirmed with user): do not reuse SpillNet's existing weights.**
Instead, pretrain a fresh oil-spill segmentation model on the same underlying
dataset (Krestenitis), using the **same standard backbone architecture**
(ResNet34-UNet) as the random-init and ImageNet-init conditions.

Rationale:
- Gets literal, layer-for-layer architecture identity across all 3
  conditions — no "matched capacity" compromise, no weight-splicing.
- The ImageNet-init condition gets a genuine off-the-shelf pretrained
  checkpoint, used as intended.
- The oil-spill-init condition gets genuine pretraining rather than a
  stem-adapted transplant of weights from an architecturally incompatible
  source.
- Cost: one extra pretraining run (on Krestenitis) before Step 2 can start.
  This is outside the 45-run matrix and is comparatively cheap.

This paper's "oil-spill-pretrained representation" therefore refers to a
representation pretrained on the oil-spill segmentation task and dataset, not
literally to the previously-existing SpillNet checkpoint. This distinction
must be stated explicitly in the eventual paper's Methods section.

## Channel decision: VV-only (grayscale) on both sides

Sen1Floods11 provides VV and VH. The oil-spill (Krestenitis) dataset provides
3-channel RGB-JPEG-rendered SAR imagery with no confirmed per-channel
polarimetric meaning.

**Decision: use VV only, as grayscale (single channel), on both the oil-spill
side and the flood side.**

- Oil-spill side: Krestenitis RGB images are converted to single-channel
  grayscale before pretraining (standard grayscale conversion, e.g. luminance
  weighting or simple channel averaging).
- Flood side: Sen1Floods11's VH band is dropped; only VV is used.

Rationale:
- Avoids the RGB→VV/VH channel-count/representation mismatch entirely — both
  sides become genuinely single-channel, so no stem-adaptation or channel
  padding trick is needed for either arm.
- Reported to be consistent with common practice in the oil-spill SAR
  detection literature, where VV alone is frequently preferred over VV+VH
  (VH is reported noisier/less reliable for oil detection in several papers
  surfaced during a literature check). **The specific citations backing this
  claim have not been independently verified in this session and must be
  checked against the primary sources before this rationale is written into
  the paper's Methods section** — the channel decision itself is sound
  regardless, but the citations need confirming.
- Grayscale-averaging a stem for a real ImageNet-pretrained backbone (RGB → 1
  channel) is a well-established, unremarkable technique in transfer
  learning (the RGB channels being averaged are physically meaningful
  photographic channels) — unlike averaging SpillNet's non-physical
  RGB-JPEG-rendered SAR channels, which was the original concern in the
  superseded version of this document.

## Resulting architecture (applies to all 3 initializations identically)

- Backbone: ResNet34-UNet (or equivalent standard segmentation architecture
  with an available ImageNet checkpoint — final choice made at implementation
  time and recorded in the plan).
- Input: single-channel (grayscale oil-spill / VV-only flood), stem adapted
  from the backbone's native 3-channel RGB input via channel averaging.
- Initializations:
  1. **Random** — backbone weights randomly initialized (standard init, e.g.
     He/Kaiming), stem included.
  2. **ImageNet** — backbone loaded with real ImageNet-pretrained weights,
     stem averaged from the pretrained 3-channel stem to 1 channel.
  3. **Oil-spill** — backbone loaded with weights from the fresh
     Krestenitis-pretrained checkpoint (see above), stem included (this
     pretraining run itself starts from the same ImageNet-averaged-stem
     initialization, so downstream comparisons isolate the effect of
     oil-spill segmentation pretraining specifically, not just ImageNet vs.
     not).

## Framing for the paper

Do not refer to the oil-spill-pretrained condition as "SpillNet" or "the
existing model" in code, comments, or the eventual manuscript. Refer to it as
"the oil-spill-pretrained encoder" or equivalent, and state in Methods that it
was pretrained specifically for this study on the Krestenitis dataset using
the same backbone as the other two conditions, rather than reusing a
previously existing checkpoint.

## Flood vs. surface-water labeling (resolved separately, noted here for completeness)

The pipeline separates JRC permanent water from the hand-labeled water mask
(`src/data/sen1floods11_dataset.py::derive_flood_label`), so labels reflect
flood water specifically. This is what supports the paper's actual title
("...Flood Segmentation...") rather than the handoff doc's fallback
"surface-water segmentation" framing, which would apply if this step were
skipped. See `SPLIT_AUDIT.md` for the separate (and more consequential)
finding about event overlap in the official train/val/test split.
