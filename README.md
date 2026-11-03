# Platform Abuse Detection & Risk Strategy Optimization

A reproducible offline data science project on the public **YelpChi** review graph. It moves from numeric and relational risk signals to a calibrated score, then to allow / manual review / block decisions under a review-capacity constraint.

**Status:** completed experiments, held-out evaluation, scenario simulator, four-tab Streamlit dashboard, notebooks, tests, and technical report. This is an **offline simulation**, not a production moderation system.

## What the project answers

1. Do graph relations improve ranking beyond a review's 32 anonymous features?
2. Which relation contributes most, and where does the model make errors?
3. How do false blocks, missed spam, reviewer accuracy and capacity change the preferred policy?

The selected model was **LightGBM with original features plus R-U-R and R-S-R aggregates**. It was selected on a separate model-validation split; the R-T-R relation was omitted after ablation. On the untouched test split, calibrated PR-AUC was **0.875** and ROC-AUC **0.967**. Under the *illustrative* cost settings (missed spam 1, false block 5, review 0.15, reviewer accuracy 100%, capacity 5%), the three-action policy had **0.659 expected recall**, **0.001 expected FPR**, and **60.1 cost units per 1,000 reviews**. These numbers are properties of this offline split and assumptions, not forecast platform outcomes. See the [technical report](reports/technical_report.md) for full comparisons and limits.

## Dataset and evaluation boundary

YelpChi contains 45,954 review nodes, 6,677 filtered labels, 32 numeric features, and three relation graphs: R-U-R (same user), R-S-R (same business and star rating), and R-T-R (same business and month). The label is Yelp's filtering flag, used as a **proxy** for suspicious reviews. The distributed graph does not include raw review text, full user profiles, device information, or a usable event timeline. Node IDs are internal row numbers. We do not infer what individual anonymous numeric features mean.

The protocol uses stratified, disjoint **55/15/10/10/10** node splits for training, model selection, calibration, strategy selection, and final test. The setting is **transductive**: graph structure and unlabeled node features are available across splits. Neighbor labels are never model features. Isotonic calibration fits only on the calibration split; cost thresholds fit only on the strategy split. This is neither a time-split nor a new-user evaluation.

## Reproduce locally

Python 3.11 or 3.12 is recommended. On macOS, the LightGBM wheel may need `brew install libomp`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python scripts/download_data.py
python -m abuse_risk.run
python -m abuse_risk.robustness
python scripts/make_notebooks.py
pytest -q
streamlit run dashboard/app.py
```

The download script fetches the [CARE-GNN YelpChi archive](https://github.com/YingtongDou/CARE-GNN/tree/master/data), checks SHA-256 of the zip and extracted `.mat`, and keeps raw files out of Git. Model binaries and raw/processed data are also ignored. Running the pipeline writes CSV/PNG/Markdown reports and score files used by the dashboard. It takes roughly a minute on a laptop, depending on hardware.

## Project map

| Path | Purpose |
|---|---|
| `src/abuse_risk/data.py` | `.mat` loader, validation, label-free graph feature pipeline, relation EDA |
| `src/abuse_risk/run.py` | split, model comparison, calibration, error analysis, strategies, reports |
| `src/abuse_risk/strategy.py` | expected cost model, top-K review, threshold optimizer |
| `src/abuse_risk/robustness.py` | simulated prevalence shift and feature perturbation |
| `dashboard/app.py` | risk overview, model evaluation, live strategy simulator, node explorer |
| `notebooks/` | executable EDA, ablation and error-analysis views |
| `reports/` | generated results and [robustness analysis](reports/robustness.md) |
| `tests/` | feature leakage and policy/capacity checks |

## Strategy model

Let score `p` be the calibrated risk estimate. S0 allows all; S1 blocks above one cutoff; S2 reviews the highest-scored `K` nodes; S3 chooses two cutoffs for allow, review and block. S1 and S3 minimize simulated cost on the strategy-validation split. S3 is constrained so the review band contains no more than the specified fraction of that split. Fixed cutoffs are applied to the untouched test split.

The cost model is `c_FN × expected missed spam + c_FP × expected false blocks + c_review × reviewed nodes`. Reviewer accuracy `q` is modeled as symmetric: a reviewed spam item is missed with probability `1−q`, and a reviewed normal item is incorrectly blocked with probability `1−q`. This is a simplifying assumption; no real review outcomes are available. All costs are unitless and hypothetical.

The dashboard recomputes cutoffs on strategy validation when sliders change, then evaluates the resulting policy on the test split. It displays test labels because it is an **offline analysis tool**; a live system would not have them at decision time. Repeatedly exploring test scenarios may inform future choices, so any new final claim would require another holdout.

## Main evidence and limitations

The raw all-graph LightGBM beat raw original-feature LightGBM by **0.076 PR-AUC** on this test split; a paired node bootstrap gives a 95% interval of **[0.061, 0.098]**. R-U-R is especially useful in the ablation table. Calibration improved Brier score for the selected model, while isotonic ties reduced PR-AUC from **0.888 raw** to **0.875 calibrated**. Error rates by graph degree and relation ablations are in the technical report. The [robustness note](reports/robustness.md) shows that even 1% feature noise changed PR-AUC in one simulation, so scores should not be treated as invariant.

Yelp's filtering flag may contain errors, graph-connected nodes weaken independent bootstrap assumptions, and the full transductive graph is visible during scoring. The benchmark cannot establish performance on future reviews, unseen users, or real operational cost. Automatic blocking in a real platform would need independent labels, prospective evaluation, reviewer data, appeal monitoring and human governance.

## Data credit

Dataset and graph schema: [CARE-GNN](https://github.com/YingtongDou/CARE-GNN) and [DGL FraudYelpDataset](https://www.dgl.ai/dgl_docs/en/0.8.x/generated/dgl.data.FraudYelpDataset.html). The data is downloaded from its source and not redistributed here. Code in this repository is MIT licensed.
