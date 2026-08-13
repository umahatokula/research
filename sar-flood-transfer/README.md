# What Transfers in SAR? Data-Efficient Flood Segmentation from Oil-Spill-Pretrained Representations

Status: **code complete, experiments not yet run.**

## Summary

A controlled empirical study of cross-task transfer from oil-spill SAR
segmentation to flood-event surface-water segmentation: does an
oil-spill-pretrained representation transfer positively to flood-water
segmentation under label-scarce conditions, compared to random and
ImageNet initialization, and which encoder depth carries the transferable
signal?

## Key documents

- [`handoff_to_claude_code.md`](handoff_to_claude_code.md) -- the original, externally-reviewed experimental protocol.
- [`CHANNEL_COMPATIBILITY.md`](CHANNEL_COMPATIBILITY.md) -- resolved input-representation and architecture decisions (VV-only, fresh oil-spill pretraining on a shared ResNet50V2-UNet rather than reusing an existing checkpoint).
- [`SPLIT_AUDIT.md`](SPLIT_AUDIT.md) -- audit of the official Sen1Floods11 train/val/test split; found real cross-partition flood-event overlap, documented as a limitation.

## Repo layout

```
sar-flood-transfer/
├── src/
│   ├── data/        # Sen1Floods11 + Krestenitis downloaders/loaders, split audit, label-fraction nesting
│   ├── models/       # shared ResNet50V2-UNet architecture, stem adaptation, frozen-feature probing
│   ├── training/      # oil-spill pretraining, single training runs, matrix orchestrators, losses, metrics
│   ├── stats/          # paired scene-level bootstrap, primary hypothesis test
│   └── figures/         # headline label-efficiency curve
├── tests/                # 98 tests, run against real downloaded data where applicable
├── colab_run_matrix.ipynb # cloud GPU execution notebook (this machine has no NVIDIA GPU)
├── results/               # output CSVs (populated after running the matrix)
└── figures/                # output PNGs (populated after running the matrix)
```

## Running

Local dev/testing (data pipeline, unit tests, architecture checks) works on
CPU:

```bash
pip install -r ../requirements.txt -r requirements-extra.txt
python -m pytest tests/
```

Actually training the matrix needs a GPU this machine doesn't have --
`colab_run_matrix.ipynb` clones this repo into Google Colab, mounts Drive
for persistent storage, and runs Step 0.5 (oil-spill pretraining), Step 2
(45-run training matrix), Step 3 (27-run frozen-feature probing), and Step
4 (statistics + headline figure). It's resumable across sessions.

## Data

Not committed (see `.gitignore`): Sen1Floods11 (downloaded automatically,
public GCS bucket) and the Krestenitis Oil Spill Detection Dataset (local
copy at `C:\Users\Umaha\Documents\phd\Oil_Spill_Detection_Dataset` on this
machine; uploaded to Drive for the Colab run).
