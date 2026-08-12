# Interpretable Machine Learning for Satellite Conjunction Risk Prediction: A Cross-Model SHAP Comparison on the ESA Kelvins Dataset

## Summary

Compares XGBoost, Random Forest, and Logistic Regression on satellite
conjunction (close-approach) risk classification using the ESA Kelvins
Collision Avoidance Challenge dataset, then compares SHAP feature attributions
across the three models. Key findings:

- CDM-derived features (`max_risk_estimate`, `max_risk_scaling`) leak future
  information despite low linear correlation with the target — a
  Pearson-correlation-based leakage screen would miss them.
- The three models disagree meaningfully in feature attribution despite
  broadly similar predictive performance, consistent with the
  interpretability-instability concern raised in the explainability
  literature.
- 10-fold cross-validation with paired Wilcoxon significance tests confirms
  the XGBoost/Random Forest vs. Logistic Regression performance gap is real,
  not a single-split artifact.
- Threshold/recall analysis shows 95% recall isn't reachable in the tested
  threshold range — relevant to deployment decisions, since F1 alone would
  understate this.

## Repo contents

```
conjunction-risk-shap/
├── conjunction_risk_shap.ipynb     # full analysis: preprocessing, models, CV, SHAP
├── conjunction_risk_paper_final.docx
├── figures/                        # paper figures (ROC, SHAP summaries, dependence, etc.)
├── results/                        # CV metrics, SHAP importances, threshold/recall tables
└── shap_arrays/                    # cached SHAP values per model (.npy)
```

## Data

This project uses the [ESA Kelvins Collision Avoidance Challenge
dataset](https://kelvins.esa.int/collision-avoidance-challenge/). Raw and
derived data files are **not** included in this repo (size limits, and
redistribution terms for the Kelvins dataset should be checked independently).
To reproduce:

1. Download `train_data.csv` / `test_data.csv` from the Kelvins challenge page.
2. Place them in this folder.
3. Run `conjunction_risk_shap.ipynb` top to bottom.

## Reproducing

```bash
pip install -r ../requirements.txt
jupyter notebook conjunction_risk_shap.ipynb
```

## Citation

_Citation details to be added on publication._
