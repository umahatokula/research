"""Sen1Floods11 chip loading: VV-only, per CHANNEL_COMPATIBILITY.md.

S1Hand GeoTIFFs are (2, 512, 512) float32, band 0 = VV, band 1 = VH, in dB.
We use VV only (band 0) -- VH is dropped entirely, on both the oil-spill and
flood sides of this study (see CHANNEL_COMPATIBILITY.md).

LabelHand GeoTIFFs are (512, 512) int16 with values {-1, 0, 1}: -1 is
no-data/cloud-masked and must be excluded from loss and metrics (handoff
doc, Step 1) -- it is kept as -1 here, not remapped, so every consumer
(metrics.py, loss functions) can filter on the same sentinel value.

By default, labels are converted to FLOOD water only: JRC permanent-water
pixels are subtracted out (relabeled 0), so a pixel reads 1 only if it's
water in the hand-labeled mask AND not permanent water per JRC. This is
what supports the paper's actual "flood segmentation" claim rather than
the more conservative "surface-water segmentation" fallback the handoff
doc requires if this step is skipped -- see CHANNEL_COMPATIBILITY.md and
the handoff doc's Step 1 for the exact rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

# Reference dB clip/rescale range used by the dataset's own Train.ipynb
# example, applied here to VV alone: clip to [-50, 1] dB, rescale to [0, 1].
DB_CLIP_MIN = -50.0
DB_CLIP_MAX = 1.0


def derive_flood_label(label: np.ndarray, permanent_water: np.ndarray) -> np.ndarray:
    """Subtract JRC permanent water from a raw water label, leaving flood water only.

    -1 (no-data) pixels pass through unchanged. Label-1 pixels that coincide
    with JRC permanent water become 0 (they're water, but not flood water).
    Label-0 pixels are unaffected.
    """
    flood = label.copy()
    is_permanent_water = permanent_water.astype(bool)
    flood[(label == 1) & is_permanent_water] = 0
    return flood


def load_chip(chip_id: str, data_dir: Path, separate_permanent_water: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Load one chip's VV image and label mask.

    Args:
        separate_permanent_water: if True (default), subtract JRC permanent
            water from the label so it reflects flood water only (see module
            docstring). If False, the raw water label is returned as-is, and
            any downstream writeup must say "surface-water segmentation".

    Returns:
        vv: float32 array, shape (512, 512, 1), values in [0, 1].
        label: int16 array, shape (512, 512, 1), values in {-1, 0, 1}.
    """
    data_dir = Path(data_dir)
    s1 = tifffile.imread(data_dir / "S1Hand" / f"{chip_id}_S1Hand.tif")
    if s1.ndim != 3 or s1.shape[0] < 2:
        raise ValueError(f"Unexpected S1Hand shape for {chip_id}: {s1.shape}")

    vv = np.nan_to_num(s1[0], nan=DB_CLIP_MIN)  # band 0 = VV
    vv = np.clip(vv, DB_CLIP_MIN, DB_CLIP_MAX)
    vv = (vv - DB_CLIP_MIN) / (DB_CLIP_MAX - DB_CLIP_MIN)
    vv = vv.astype(np.float32)[..., np.newaxis]

    label = tifffile.imread(data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif")
    if label.ndim != 2:
        raise ValueError(f"Unexpected LabelHand shape for {chip_id}: {label.shape}")
    label = label.astype(np.int16)[..., np.newaxis]

    if vv.shape[:2] != label.shape[:2]:
        raise ValueError(f"VV/label shape mismatch for {chip_id}: {vv.shape} vs {label.shape}")

    if separate_permanent_water:
        permanent_water = load_permanent_water(chip_id, data_dir)
        if permanent_water.shape[:2] != label.shape[:2]:
            raise ValueError(f"JRC/label shape mismatch for {chip_id}: {permanent_water.shape} vs {label.shape}")
        label = derive_flood_label(label, permanent_water)

    return vv, label


def load_permanent_water(chip_id: str, data_dir: Path) -> np.ndarray:
    """Load the JRC permanent-water layer for a chip (0/1), shape (512, 512, 1)."""
    data_dir = Path(data_dir)
    jrc = tifffile.imread(data_dir / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif")
    return jrc.astype(np.uint8)[..., np.newaxis]


def make_generator(chip_ids: list[str], data_dir: Path):
    """Python generator yielding (vv, label) pairs -- feeds tf.data.Dataset.from_generator."""

    def _gen():
        for chip_id in chip_ids:
            yield load_chip(chip_id, data_dir)

    return _gen


def make_tf_dataset(chip_ids: list[str], data_dir: Path, batch_size: int = 8, shuffle: bool = False, seed: int = 0):
    """Build a batched tf.data.Dataset of (vv, label) pairs for the given chips."""
    import tensorflow as tf  # local import: keep this module importable without TF for pure data-audit uses

    output_signature = (
        tf.TensorSpec(shape=(512, 512, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(512, 512, 1), dtype=tf.int16),
    )
    ds = tf.data.Dataset.from_generator(make_generator(chip_ids, data_dir), output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(chip_ids), 1), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
