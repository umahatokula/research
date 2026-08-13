from pathlib import Path

import numpy as np
import pytest

from src.data.krestenitis_dataset import list_sample_stems, load_sample

DATASET_ROOT = Path(r"C:\Users\Umaha\Documents\phd\Oil_Spill_Detection_Dataset")
IMAGES_DIR = DATASET_ROOT / "train" / "images"
LABELS_DIR = DATASET_ROOT / "train" / "labels_1D"

requires_real_data = pytest.mark.skipif(
    not IMAGES_DIR.exists(), reason="Krestenitis Oil Spill Detection Dataset not found locally"
)


@requires_real_data
def test_list_sample_stems_finds_all_training_images():
    stems = list_sample_stems(IMAGES_DIR)
    assert len(stems) == 1002
    assert stems[0] == "img_0001"


@requires_real_data
def test_load_sample_shapes_and_dtypes():
    stem = list_sample_stems(IMAGES_DIR)[0]
    image, label = load_sample(stem, IMAGES_DIR, LABELS_DIR, target_size=(224, 224))
    assert image.shape == (224, 224, 1)
    assert label.shape == (224, 224, 1)
    assert image.dtype == np.float32
    assert label.dtype == np.uint8


@requires_real_data
def test_image_values_in_unit_range():
    stem = list_sample_stems(IMAGES_DIR)[0]
    image, _ = load_sample(stem, IMAGES_DIR, LABELS_DIR)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


@requires_real_data
def test_label_is_binary():
    stem = list_sample_stems(IMAGES_DIR)[0]
    _, label = load_sample(stem, IMAGES_DIR, LABELS_DIR)
    assert set(np.unique(label)).issubset({0, 1})


@requires_real_data
def test_label_matches_oil_spill_class_only():
    # Pick a sample with more than one class present in the raw label, and
    # confirm the binary label is 1 exactly where the raw label is class 1.
    from PIL import Image

    for stem in list_sample_stems(IMAGES_DIR)[:20]:
        raw = np.asarray(Image.open(LABELS_DIR / f"{stem}.png"))
        if 1 in np.unique(raw) and len(np.unique(raw)) > 1:
            _, label = load_sample(stem, IMAGES_DIR, LABELS_DIR, target_size=raw.shape[::-1])
            expected = (raw == 1).astype(np.uint8)[..., np.newaxis]
            np.testing.assert_array_equal(label, expected)
            return
    pytest.skip("No sample with class-1 pixels found in first 20 stems")


@requires_real_data
def test_load_sample_raises_on_missing_stem():
    with pytest.raises(FileNotFoundError):
        load_sample("nonexistent_stem", IMAGES_DIR, LABELS_DIR)
