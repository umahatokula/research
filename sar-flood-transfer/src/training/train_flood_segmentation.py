"""Step 2: one (initialization, label_fraction, seed) training + eval run.

Appends one row to the main results CSV with columns
[initialization, label_fraction, seed, iou, f1] (handoff doc, Step 2), and
separately writes per-test-chip metrics (needed for Step 4's paired
scene-level bootstrap, which the handoff doc requires instead of pixel-level
or n=3-seed significance testing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.download_sen1floods11 import chip_id_from_s1_filename, download_splits
from src.data.label_fractions import build_nested_fractions
from src.data.sen1floods11_dataset import load_chip, make_tf_dataset
from src.models.backbone_unet import InitMode, build_unet
from src.training.losses import masked_binary_crossentropy
from src.training.metrics import compute_iou_f1

DEFAULT_INPUT_SIZE = (512, 512)


def get_chip_id_splits(data_dir: Path) -> dict[str, list[str]]:
    """{'train': [...], 'valid': [...], 'test': [...]} of chip IDs (not filenames)."""
    raw_splits = download_splits(data_dir)
    return {name: [chip_id_from_s1_filename(s1) for s1, _ in pairs] for name, pairs in raw_splits.items()}


def evaluate_test_set(
    model: tf.keras.Model, test_chip_ids: list[str], data_dir: Path
) -> tuple[dict[str, float], pd.DataFrame]:
    """Aggregate (pooled-pixel) IoU/F1 plus a per-chip breakdown for downstream bootstrap stats."""
    per_chip_rows = []
    all_true, all_pred = [], []

    for chip_id in test_chip_ids:
        vv, label = load_chip(chip_id, data_dir)
        pred_prob = model(vv[np.newaxis, ...], training=False).numpy()[0]
        pred_binary = (pred_prob >= 0.5).astype(np.int16)

        chip_metrics = compute_iou_f1(label, pred_binary)
        per_chip_rows.append({"chip_id": chip_id, "iou": chip_metrics["iou"], "f1": chip_metrics["f1"]})

        all_true.append(label)
        all_pred.append(pred_binary)

    pooled = compute_iou_f1(np.concatenate([a.ravel() for a in all_true]), np.concatenate([a.ravel() for a in all_pred]))
    return pooled, pd.DataFrame(per_chip_rows)


def run_single_training(
    initialization: InitMode,
    label_fraction: float,
    seed: int,
    data_dir: Path,
    output_csv: Path,
    oilspill_weights_path: Path | None = None,
    per_chip_dir: Path | None = None,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    batch_size: int = 4,
    epochs: int = 30,
    patience: int = 5,
    learning_rate: float = 1e-4,
) -> dict[str, float]:
    """Train one (init, label_fraction, seed) configuration, evaluate on the test set,
    append the result row to output_csv, and return the aggregate metrics dict.
    """
    if initialization == "oilspill" and oilspill_weights_path is None:
        raise ValueError("initialization='oilspill' requires oilspill_weights_path")

    splits = get_chip_id_splits(data_dir)
    nested = build_nested_fractions(splits["train"], seed=seed)
    if label_fraction not in nested:
        raise ValueError(f"label_fraction {label_fraction} not among nested fractions {sorted(nested.keys())}")
    train_chip_ids = nested[label_fraction]

    train_ds = make_tf_dataset(train_chip_ids, data_dir, batch_size=batch_size, shuffle=True, seed=seed)
    val_ds = make_tf_dataset(splits["valid"], data_dir, batch_size=batch_size)

    model = build_unet((*input_size, 1), init_mode=initialization, oilspill_weights_path=oilspill_weights_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=masked_binary_crossentropy)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    aggregate, per_chip_df = evaluate_test_set(model, splits["test"], data_dir)

    row = pd.DataFrame(
        [{"initialization": initialization, "label_fraction": label_fraction, "seed": seed, **aggregate}]
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)

    if per_chip_dir is not None:
        per_chip_dir = Path(per_chip_dir)
        per_chip_dir.mkdir(parents=True, exist_ok=True)
        per_chip_df.to_csv(per_chip_dir / f"{initialization}_frac{label_fraction}_seed{seed}.csv", index=False)

    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization", required=True, choices=["random", "imagenet", "oilspill"])
    parser.add_argument("--label-fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "sen1floods11"
    )
    parser.add_argument("--output-csv", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "step2_training_matrix.csv")
    parser.add_argument(
        "--per-chip-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "per_chip_metrics"
    )
    parser.add_argument(
        "--oilspill-weights",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "checkpoints" / "oilspill_encoder.weights.h5",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    result = run_single_training(
        args.initialization,
        args.label_fraction,
        args.seed,
        args.data_dir,
        args.output_csv,
        oilspill_weights_path=args.oilspill_weights if args.initialization == "oilspill" else None,
        per_chip_dir=args.per_chip_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    print("Result:", result)
