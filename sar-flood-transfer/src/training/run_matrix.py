"""Step 2 orchestrator: the full 3 x 5 x 3 = 45-run training matrix.

Per the handoff doc: smoke-test ONE configuration (all 3 initializations,
10% labels, 1 seed) before scaling to the full matrix. --smoke-test runs
exactly that; without the flag, every (initialization, label_fraction, seed)
combination is run, in an order that surfaces problems early (smoke-test
config first, then the rest).

Resumable: a config already present in output_csv is skipped rather than
re-run. This matters for cloud execution, where a multi-hour job commonly
spans more than one Colab session (idle/runtime-length disconnects) -- an
interrupted run can be restarted with the same command and picks up where
it left off instead of redoing completed work.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.data.label_fractions import DEFAULT_FRACTIONS
from src.training.train_flood_segmentation import run_single_training

INITIALIZATIONS = ("random", "imagenet", "oilspill")
SEEDS = (0, 1, 2)
SMOKE_TEST_LABEL_FRACTION = 0.10
SMOKE_TEST_SEED = 0


def build_run_matrix(smoke_test: bool) -> list[tuple[str, float, int]]:
    """List of (initialization, label_fraction, seed) configs to run, in order."""
    if smoke_test:
        return [(init, SMOKE_TEST_LABEL_FRACTION, SMOKE_TEST_SEED) for init in INITIALIZATIONS]
    return list(itertools.product(INITIALIZATIONS, DEFAULT_FRACTIONS, SEEDS))


def get_completed_configs(output_csv: Path) -> set[tuple[str, float, int]]:
    """Configs already present in output_csv, as (initialization, label_fraction, seed) tuples."""
    output_csv = Path(output_csv)
    if not output_csv.exists():
        return set()
    df = pd.read_csv(output_csv)
    if df.empty:
        return set()
    return set(zip(df["initialization"], df["label_fraction"].astype(float), df["seed"].astype(int)))


def run_matrix(
    data_dir: Path,
    output_csv: Path,
    per_chip_dir: Path,
    oilspill_weights_path: Path,
    smoke_test: bool = True,
    batch_size: int = 4,
    epochs: int = 30,
) -> None:
    configs = build_run_matrix(smoke_test)
    completed = get_completed_configs(output_csv)
    remaining = [c for c in configs if c not in completed]

    print(
        f"{len(configs)} configuration(s) total {'(smoke test)' if smoke_test else '(FULL MATRIX)'}, "
        f"{len(completed & set(configs))} already completed, {len(remaining)} remaining"
    )

    for i, (initialization, label_fraction, seed) in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] init={initialization} label_fraction={label_fraction} seed={seed}")
        result = run_single_training(
            initialization=initialization,
            label_fraction=label_fraction,
            seed=seed,
            data_dir=data_dir,
            output_csv=output_csv,
            oilspill_weights_path=oilspill_weights_path if initialization == "oilspill" else None,
            per_chip_dir=per_chip_dir,
            batch_size=batch_size,
            epochs=epochs,
        )
        print(f"  -> iou={result['iou']:.4f} f1={result['f1']:.4f}")

    print(f"\nDone. Results appended to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "sen1floods11"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "step2_training_matrix.csv"
    )
    parser.add_argument(
        "--per-chip-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results" / "per_chip_metrics"
    )
    parser.add_argument(
        "--oilspill-weights",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "checkpoints" / "oilspill_encoder.weights.h5",
    )
    parser.add_argument("--full", action="store_true", help="Run the full 45-run matrix instead of the smoke test.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    run_matrix(
        args.data_dir,
        args.output_csv,
        args.per_chip_dir,
        args.oilspill_weights,
        smoke_test=not args.full,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
