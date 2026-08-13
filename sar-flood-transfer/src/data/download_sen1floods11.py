"""Download the Sen1Floods11 hand-labeled subset (446 chips) via public HTTPS.

The sen1floods11 GCS bucket is publicly readable, so plain HTTPS GETs work
without gsutil or any credentials. We only download the hand-labeled subset
(S1Hand, LabelHand, JRCWaterHand) plus the official train/val/test split
CSVs -- NOT the weakly-labeled subset, per the handoff doc.
"""

from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS_PREFIX = f"{BUCKET_BASE}/splits/flood_handlabeled"
DATA_PREFIX = f"{BUCKET_BASE}/data/flood_events/HandLabeled"

SPLIT_FILES = ("flood_train_data.csv", "flood_valid_data.csv", "flood_test_data.csv")


def _fetch_url(url: str, max_retries: int = 4, timeout: float = 30.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts") from last_error


def _download_to_file(url: str, dest: Path, skip_existing: bool = True) -> str:
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return f"skip {dest.name}"
    data = _fetch_url(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return f"ok   {dest.name}"


def download_splits(data_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """Download the 3 split CSVs and return {split_name: [(s1_fname, label_fname), ...]}."""
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[tuple[str, str]]] = {}
    for split_file in SPLIT_FILES:
        dest = splits_dir / split_file
        if not dest.exists():
            dest.write_bytes(_fetch_url(f"{SPLITS_PREFIX}/{split_file}"))
        rows = list(csv.reader(io.StringIO(dest.read_text())))
        split_name = split_file.replace("flood_", "").replace("_data.csv", "")  # train/valid/test
        result[split_name] = [(r[0], r[1]) for r in rows if r]
    return result


def chip_id_from_s1_filename(s1_filename: str) -> str:
    """'Ghana_103272_S1Hand.tif' -> 'Ghana_103272'."""
    return s1_filename.replace("_S1Hand.tif", "")


def download_hand_labeled_subset(
    data_dir: Path,
    max_workers: int = 8,
    include_jrc: bool = True,
) -> None:
    """Download S1Hand, LabelHand, and (optionally) JRCWaterHand for all 446 chips."""
    splits = download_splits(data_dir)
    all_chip_ids = sorted(
        {chip_id_from_s1_filename(s1) for pairs in splits.values() for s1, _ in pairs}
    )

    jobs: list[tuple[str, Path]] = []
    for chip_id in all_chip_ids:
        jobs.append((f"{DATA_PREFIX}/S1Hand/{chip_id}_S1Hand.tif", data_dir / "S1Hand" / f"{chip_id}_S1Hand.tif"))
        jobs.append(
            (f"{DATA_PREFIX}/LabelHand/{chip_id}_LabelHand.tif", data_dir / "LabelHand" / f"{chip_id}_LabelHand.tif")
        )
        if include_jrc:
            jobs.append(
                (
                    f"{DATA_PREFIX}/JRCWaterHand/{chip_id}_JRCWaterHand.tif",
                    data_dir / "JRCWaterHand" / f"{chip_id}_JRCWaterHand.tif",
                )
            )

    print(f"Downloading {len(jobs)} files for {len(all_chip_ids)} chips...")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_to_file, url, dest): url for url, dest in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - report and continue, don't abort the whole batch
                failures.append(url)
                print(f"FAILED: {url} ({exc})")
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done")

    if failures:
        raise RuntimeError(f"{len(failures)} file(s) failed to download: {failures[:10]}...")
    print("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "sen1floods11",
        help="Local directory to download into.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--no-jrc", action="store_true", help="Skip downloading the JRC permanent-water layer.")
    args = parser.parse_args()

    download_hand_labeled_subset(args.data_dir, max_workers=args.max_workers, include_jrc=not args.no_jrc)
