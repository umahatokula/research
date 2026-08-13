"""Step 3: frozen-feature segmentation probing.

3 initializations x 3 depths (early/middle/bottleneck) x 3 seeds = 27 runs.
Run only after Step 2's main results look sane (handoff doc). Each probe
freezes the encoder and trains only a 1x1 conv + upsample head
(build_probe_model in backbone_unet.py) on the FULL training set (no
label-fraction dimension here -- that axis belongs to Step 2). Called
"frozen-feature segmentation probing", not "linear probing" (handoff doc:
more precise since this is a segmentation task, not classification).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.sen1floods11_dataset import load_chip, make_tf_dataset
from src.models.backbone_unet import PROBE_DEPTH_LAYER_NAMES, InitMode, build_encoder, build_probe_model
from src.training.losses import masked_binary_crossentropy
from src.training.metrics import compute_iou_f1
from src.training.train_flood_segmentation import get_chip_id_splits

INITIALIZATIONS = ("random", "imagenet", "oilspill")
DEPTHS = tuple(PROBE_DEPTH_LAYER_NAMES.keys())
SEEDS = (0, 1, 2)

DEFAULT_INPUT_SIZE = (512, 512)


def build_probe_matrix() -> list[tuple[str, str, int]]:
    """List of (initialization, depth, seed) configs -- all 27 combinations."""
    return list(itertools.product(INITIALIZATIONS, DEPTHS, SEEDS))


def get_completed_probe_configs(output_csv: Path) -> set[tuple[str, str, int]]:
    """Configs already present in output_csv, as (initialization, depth, seed) tuples."""
    output_csv = Path(output_csv)
    if not output_csv.exists():
        return set()
    df = pd.read_csv(output_csv)
    if df.empty:
        return set()
    return set(zip(df["initialization"], df["depth"], df["seed"].astype(int)))


def evaluate_probe_test_set(probe: tf.keras.Model, test_chip_ids: list[str], data_dir: Path) -> dict[str, float]:
    all_true, all_pred = [], []
    for chip_id in test_chip_ids:
        vv, label = load_chip(chip_id, data_dir)
        pred_prob = probe(vv[np.newaxis, ...], training=False).numpy()[0]
        pred_binary = (pred_prob >= 0.5).astype(np.int16)
        all_true.append(label)
        all_pred.append(pred_binary)
    return compute_iou_f1(np.concatenate([a.ravel() for a in all_true]), np.concatenate([a.ravel() for a in all_pred]))


def run_single_probe(
    initialization: InitMode,
    depth: str,
    seed: int,
    data_dir: Path,
    output_csv: Path,
    oilspill_weights_path: Path | None = None,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    batch_size: int = 4,
    epochs: int = 15,
    patience: int = 4,
    learning_rate: float = 1e-3,  # only the small probe head trains, can afford a higher LR
) -> dict[str, float]:
    if initialization == "oilspill" and oilspill_weights_path is None:
        raise ValueError("initialization='oilspill' requires oilspill_weights_path")
    if depth not in PROBE_DEPTH_LAYER_NAMES:
        raise ValueError(f"depth must be one of {DEPTHS}, got {depth!r}")

    splits = get_chip_id_splits(data_dir)

    encoder = build_encoder((*input_size, 1), init_mode=initialization, oilspill_weights_path=oilspill_weights_path)
    probe = build_probe_model(encoder, depth)
    probe.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=masked_binary_crossentropy)

    train_ds = make_tf_dataset(splits["train"], data_dir, batch_size=batch_size, shuffle=True, seed=seed)
    val_ds = make_tf_dataset(splits["valid"], data_dir, batch_size=batch_size)

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    probe.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    metrics = evaluate_probe_test_set(probe, splits["test"], data_dir)

    row = pd.DataFrame([{"initialization": initialization, "depth": depth, "seed": seed, **metrics}])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)

    return metrics


def run_probe_matrix(
    data_dir: Path,
    output_csv: Path,
    oilspill_weights_path: Path,
    batch_size: int = 4,
    epochs: int = 15,
) -> None:
    configs = build_probe_matrix()
    completed = get_completed_probe_configs(output_csv)
    remaining = [c for c in configs if c not in completed]

    print(
        f"{len(configs)} frozen-feature probe configuration(s) total, "
        f"{len(completed & set(configs))} already completed, {len(remaining)} remaining"
    )
    for i, (initialization, depth, seed) in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] init={initialization} depth={depth} seed={seed}")
        metrics = run_single_probe(
            initialization=initialization,
            depth=depth,
            seed=seed,
            data_dir=data_dir,
            output_csv=output_csv,
            oilspill_weights_path=oilspill_weights_path if initialization == "oilspill" else None,
            batch_size=batch_size,
            epochs=epochs,
        )
        print(f"  -> iou={metrics['iou']:.4f} f1={metrics['f1']:.4f}")
    print(f"\nDone. Results appended to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "sen1floods11"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "step3_probe_matrix.csv"
    )
    parser.add_argument(
        "--oilspill-weights",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "checkpoints" / "oilspill_encoder.weights.h5",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    run_probe_matrix(args.data_dir, args.output_csv, args.oilspill_weights, args.batch_size, args.epochs)
