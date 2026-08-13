"""Step 0.5: pretrain a fresh oil-spill segmentation encoder on Krestenitis.

Per CHANNEL_COMPATIBILITY.md, this replaces reusing SpillNet's existing
weights: we pretrain a new model on the same underlying dataset
(Oil_Spill_Detection_Dataset / Krestenitis), using the exact ResNet50V2-UNet
architecture shared by all 3 study conditions, starting the encoder from the
ImageNet-averaged-stem initialization. Only the resulting ENCODER weights are
saved -- the decoder here is task-specific to oil-spill pretraining and is
discarded; the flood-segmentation U-Net gets its own freshly-initialized
decoder regardless of encoder init_mode (see backbone_unet.py).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import tensorflow as tf

from src.data.krestenitis_dataset import list_sample_stems, make_tf_dataset
from src.models.backbone_unet import attach_decoder, build_encoder

DEFAULT_INPUT_SIZE = (224, 224)


def split_train_val(stems: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(val_fraction * len(shuffled)))
    return shuffled[n_val:], shuffled[:n_val]


def pretrain_oilspill(
    dataset_root: Path,
    output_weights_path: Path,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    batch_size: int = 8,
    epochs: int = 20,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict:
    """Train and save the oil-spill-pretrained encoder. Returns the Keras history dict."""
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "train" / "images"
    labels_dir = dataset_root / "train" / "labels_1D"

    all_stems = list_sample_stems(images_dir)
    train_stems, val_stems = split_train_val(all_stems, val_fraction, seed)

    train_ds = make_tf_dataset(
        train_stems, images_dir, labels_dir, target_size=input_size, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_ds = make_tf_dataset(val_stems, images_dir, labels_dir, target_size=input_size, batch_size=batch_size)

    encoder = build_encoder((*input_size, 1), init_mode="imagenet")
    model = attach_decoder(encoder, name="oilspill_pretrain")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="oil_iou")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    output_weights_path = Path(output_weights_path)
    output_weights_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.save_weights(str(output_weights_path))

    return history.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path(r"C:\Users\Umaha\Documents\phd\Oil_Spill_Detection_Dataset")
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "checkpoints" / "oilspill_encoder.weights.h5",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hist = pretrain_oilspill(
        args.dataset_root,
        args.output_weights,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print("Final epoch metrics:", {k: v[-1] for k, v in hist.items()})
