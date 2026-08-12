# Research

Code, figures, and results accompanying my papers. One folder per paper.

| Paper | Summary | Status | Folder |
|---|---|---|---|
| Interpretable Machine Learning for Satellite Conjunction Risk Prediction | Cross-model SHAP comparison (XGBoost, Random Forest, Logistic Regression) on the ESA Kelvins collision-avoidance dataset; examines feature-attribution disagreement across models and a CDM-derived data-leakage finding. | Draft | [`conjunction-risk-shap/`](conjunction-risk-shap/) |

## Shared setup

Each paper folder is self-contained (notebook + `README.md` + `figures/` + `results/`).
Most papers share the same baseline Python stack:

```bash
pip install -r requirements.txt
```

If a specific paper needs extra packages, it'll have its own `requirements-extra.txt`.

## License

Code in this repo is MIT licensed (see `LICENSE`). This does not cover any
third-party datasets referenced by individual papers — see each paper's own
README for data provenance and usage terms.
