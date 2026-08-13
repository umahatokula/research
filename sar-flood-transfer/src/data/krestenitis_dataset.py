"""Krestenitis Oil Spill Detection Dataset loading: grayscale, binary oil label.

Source: C:\\Users\\Umaha\\Documents\\phd\\Oil_Spill_Detection_Dataset (train:
1002 samples, test: 110). Images are RGB-JPEG renderings of SAR imagery
(1250x650); labels_1D are single-channel PNGs with 5 classes (0=sea,
1=oil spill, 2=look-alike, 3=ship, 4=land).

Per CHANNEL_COMPATIBILITY.md: images are converted to grayscale (single
channel, "VV-like"), and labels are collapsed to binary oil-spill-vs-not
(class 1 -> 1, everything else -> 0) -- this pretraining task only needs
"is this pixel oil spill", matching the paper's framing of "oil-spill
segmentation pretraining", not full 5-class semantic segmentation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

OIL_SPILL_CLASS = 1

# Directory names known NOT to be the Krestenitis dataset root, skipped when
# searching a shared parent directory (e.g. a Drive folder that also holds
# this project's own checkpoints/data/figures/results).
_NON_DATASET_DIR_NAMES = frozenset({"checkpoints", "data", "figures", "results"})


def find_dataset_root(base_dir: Path) -> Path | None:
    """Find the Krestenitis dataset root under base_dir, regardless of its name.

    Searches for a directory containing a `train/images` subfolder. This
    exists because the dataset's own top-level folder name isn't guaranteed
    (e.g. a zip re-upload produced "Oil Spill Detection Dataset" with
    spaces, not the "Oil_Spill_Detection_Dataset" name used locally --
    silent failure the first time this was deployed to Colab, since an
    empty-glob path doesn't raise, it just yields zero training samples).
    """
    base_dir = Path(base_dir)
    for root, dirs, _files in os.walk(base_dir):
        root = Path(root)
        if root == base_dir:
            dirs[:] = [d for d in dirs if d not in _NON_DATASET_DIR_NAMES]
        if "train" in dirs and (root / "train" / "images").is_dir():
            return root
    return None


def list_sample_stems(images_dir: Path) -> list[str]:
    """'img_0001.jpg' -> 'img_0001', for every image in images_dir, sorted."""
    images_dir = Path(images_dir)
    return sorted(p.stem for p in images_dir.glob("*.jpg"))


def load_sample(
    stem: str,
    images_dir: Path,
    labels_dir: Path,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    """Load one (grayscale image, binary oil label) pair, resized to target_size.

    Returns:
        image: float32 array, shape (H, W, 1), values in [0, 1].
        label: uint8 array, shape (H, W, 1), values in {0, 1} (1 = oil spill).
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    img = Image.open(images_dir / f"{stem}.jpg").convert("L").resize(target_size, Image.BILINEAR)
    image = (np.asarray(img, dtype=np.float32) / 255.0)[..., np.newaxis]

    lbl = Image.open(labels_dir / f"{stem}.png").resize(target_size, Image.NEAREST)
    raw_label = np.asarray(lbl)
    label = (raw_label == OIL_SPILL_CLASS).astype(np.uint8)[..., np.newaxis]

    return image, label


def make_generator(stems: list[str], images_dir: Path, labels_dir: Path, target_size: tuple[int, int] = (224, 224)):
    def _gen():
        for stem in stems:
            yield load_sample(stem, images_dir, labels_dir, target_size)

    return _gen


def make_tf_dataset(
    stems: list[str],
    images_dir: Path,
    labels_dir: Path,
    target_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
    shuffle: bool = False,
    seed: int = 0,
):
    """Build a batched tf.data.Dataset of (grayscale image, binary oil label) pairs."""
    import tensorflow as tf  # local import: keep this module importable without TF

    h, w = target_size
    output_signature = (
        tf.TensorSpec(shape=(h, w, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w, 1), dtype=tf.uint8),
    )
    ds = tf.data.Dataset.from_generator(
        make_generator(stems, images_dir, labels_dir, target_size), output_signature=output_signature
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(stems), 1), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
