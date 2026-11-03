"""Reproduce the complete offline experiment: python -m abuse_risk.run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_yelp, make_features, relation_summary
from .metrics import ranking_metrics
from .strategy import Costs, compare_policies, optimize

ROOT = Path(__file__).resolve().parents[2]


def _json(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=_json, allow_nan=False))


def split_indices(y, seed):
    """Disjoint 55/15/10/10/10 stratified node splits."""
    all_ids = np.arange(len(y))
    train, other = train_test_split(all_ids, train_size=0.55, stratify=y, random_state=seed)
    model_val, other = train_test_split(other, train_size=1/3, stratify=y[other], random_state=seed + 1)
    calibration, other = train_test_split(other, train_size=1/3, stratify=y[other], random_state=seed + 2)
    strategy_val, test = train_test_split(other, train_size=0.5, stratify=y[other], random_state=seed + 3)
    return {"train": train, "model_val": model_val, "calibration": calibration,
            "strategy_val": strategy_val, "test": test}


def _model(name, seed):
    if name == "M0 logistic original":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    return lgb.LGBMClassifier(n_estimators=220, learning_rate=0.04, num_leaves=31,
                              min_child_samples=80, colsample_bytree=0.85,
                              reg_lambda=2, random_state=seed, verbosity=-1, n_jobs=4)


def bootstrap_pr_difference(y, p_graph, p_original, seed=42, n=300):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) == 2:
            differences.append(average_precision_score(y[ids], p_graph[ids]) -
                               average_precision_score(y[ids], p_original[ids]))
    return np.quantile(differences, [0.025, 0.975]).tolist()


def run(raw: Path, seed: int = 42):
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    print("Loading YelpChi and constructing label-free graph features...", flush=True)
    data = load_yelp(raw)
    features = make_features(data)
    indices = split_indices(data.y, seed)
    original = [c for c in features if c.startswith("f_")]
    relation_cols = {r: [c for c in features if c.startswith(r + "_")] for r in data.graphs}
    specifications = {"M0 logistic original": original, "M1 LightGBM original": original,
                      "M2 LightGBM all graph": list(features.columns)}
    for relation in data.graphs:
        specifications[f"M3 without {relation.upper()}"] = [c for c in features if c not in relation_cols[relation]]

    quality = {"nodes": len(data.y), "positive": int(data.y.sum()),
               "positive_rate": float(data.y.mean()), "feature_count": data.x.shape[1],
               "missing_feature_values": int(np.isnan(data.x).sum()),
               "nonfinite_feature_values": int((~np.isfinite(data.x)).sum()),
               "duplicate_feature_rows": int(pd.DataFrame(data.x).duplicated().sum()),
               "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
               "split_counts": {k: len(v) for k, v in indices.items()},
               "split_positive_rates": {k: float(data.y[v].mean()) for k, v in indices.items()}}
    _write_json(ROOT / "reports/data_quality.json", quality)
    rel = relation_summary(data)
    rel.to_csv(ROOT / "reports/relation_summary.csv", index=False)
    feature_association = pd.DataFrame([
        {"feature": c, "point_biserial_r": float(pointbiserialr(data.y[indices["train"]], features[c].iloc[indices["train"]]).statistic)}
        for c in original
    ]).sort_values("point_biserial_r", key=lambda s: s.abs(), ascending=False)
    feature_association.to_csv(ROOT / "reports/feature_association.csv", index=False)

    # The SQL table is a reproducible analytical view, not a leaked model input.
    con = duckdb.connect()
    review_table = pd.DataFrame({"review_id": np.arange(len(data.y)), "label": data.y})
    for relation in data.graphs:
        review_table[f"{relation}_degree"] = np.expm1(features[f"{relation}_log_degree"])
    con.register("reviews", review_table)
    sql_summary = con.execute("SELECT label, count(*) AS reviews, avg(rur_degree) AS mean_rur_degree, avg(rsr_degree) AS mean_rsr_degree, avg(rtr_degree) AS mean_rtr_degree FROM reviews GROUP BY label ORDER BY label").df()
    sql_summary.to_csv(ROOT / "reports/sql_risk_summary.csv", index=False)
    con.close()

    fitted = {}
    val_rows = []
    test_rows = []
    for name, cols in specifications.items():
        print(f"Training {name}...", flush=True)
        model = _model(name, seed)
        model.fit(features.iloc[indices["train"]][cols], data.y[indices["train"]])
        fitted[name] = (model, cols)
        pred_val = model.predict_proba(features.iloc[indices["model_val"]][cols])[:, 1]
        val_rows.append({"model": name, "feature_count": len(cols), **ranking_metrics(data.y[indices["model_val"]], pred_val)})
    val_df = pd.DataFrame(val_rows).sort_values("pr_auc", ascending=False)
    val_df.to_csv(ROOT / "reports/model_selection.csv", index=False)
    winner = str(val_df.iloc[0]["model"])
    model, cols = fitted[winner]
    calibration = IsotonicRegression(out_of_bounds="clip")
    p_cal_raw = model.predict_proba(features.iloc[indices["calibration"]][cols])[:, 1]
    calibration.fit(p_cal_raw, data.y[indices["calibration"]])

    test_predictions = {}
    for name, (candidate, candidate_cols) in fitted.items():
        raw_test = candidate.predict_proba(features.iloc[indices["test"]][candidate_cols])[:, 1]
        test_predictions[name] = raw_test
        test_rows.append({"model": name, "feature_count": len(candidate_cols),
                          **ranking_metrics(data.y[indices["test"]], raw_test)})
    test_df = pd.DataFrame(test_rows).sort_values("pr_auc", ascending=False)
    test_df.to_csv(ROOT / "reports/model_comparison_test.csv", index=False)

    p_strategy = calibration.predict(model.predict_proba(features.iloc[indices["strategy_val"]][cols])[:, 1])
    p_test = calibration.predict(test_predictions[winner])
    scores = {}
    for split, p in [("strategy_val", p_strategy), ("test", p_test)]:
        ids = indices[split]
        frame = pd.DataFrame({"review_id": ids, "label": data.y[ids], "score": p})
        for relation in data.graphs:
            frame[f"{relation}_degree"] = np.expm1(features[f"{relation}_log_degree"].iloc[ids].to_numpy()).round().astype(int)
        frame.to_csv(ROOT / f"artifacts/{split}_scores.csv", index=False)
        scores[split] = frame
    joblib.dump({"model": model, "columns": cols, "calibrator": calibration, "seed": seed}, ROOT / "artifacts/risk_model.joblib")

    costs = Costs()
    thresholds = optimize(data.y[indices["strategy_val"]], p_strategy, costs, 0.05)
    _write_json(ROOT / "artifacts/default_thresholds.json", thresholds)
    comparison = compare_policies(data.y[indices["test"]], p_test, costs, 0.05, thresholds)
    pd.DataFrame([{"policy": name, **metrics} for name, metrics in comparison.items()]).to_csv(ROOT / "reports/strategy_comparison.csv", index=False)
    sensitivity = []
    for capacity in [0.02, 0.05, 0.10]:
        for fp_multiplier in [1, 5, 10]:
            for q in [1.0, 0.9, 0.8]:
                scenario = Costs(false_positive=fp_multiplier, review_accuracy=q)
                cuts = optimize(data.y[indices["strategy_val"]], p_strategy, scenario, capacity, grid_size=30)
                result = compare_policies(data.y[indices["test"]], p_test, scenario, capacity, cuts)
                sensitivity.extend({"capacity": capacity, "fp_cost": fp_multiplier,
                                    "review_accuracy": q, "policy": name, **metrics}
                                   for name, metrics in result.items())
    pd.DataFrame(sensitivity).to_csv(ROOT / "reports/strategy_sensitivity.csv", index=False)

    # Grouped error analysis and compact case cards use only held-out nodes.
    test_frame = scores["test"].copy()
    test_frame["total_degree"] = test_frame[[f"{r}_degree" for r in data.graphs]].sum(axis=1)
    test_frame["degree_group"] = pd.qcut(test_frame.total_degree.rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    error_rows = []
    binary = p_test >= thresholds["s1_threshold"]
    for group, part in test_frame.groupby("degree_group", observed=True):
        loc = test_frame.index.get_indexer(part.index)
        yy, bb = part.label.to_numpy(), binary[loc]
        error_rows.append({"degree_group": str(group), "n": len(part),
                           "recall": float((bb & (yy == 1)).sum() / max((yy == 1).sum(), 1)),
                           "fpr": float((bb & (yy == 0)).sum() / max((yy == 0).sum(), 1))})
    pd.DataFrame(error_rows).to_csv(ROOT / "reports/error_by_degree.csv", index=False)
    fp_cases = test_frame[(test_frame.label == 0) & binary].sort_values("score", ascending=False).head(5)
    fn_cases = test_frame[(test_frame.label == 1) & ~binary].sort_values("score").head(5)
    pd.concat([fp_cases.assign(error_type="false_positive"), fn_cases.assign(error_type="false_negative")]).to_csv(ROOT / "reports/error_cases.csv", index=False)

    importance = pd.DataFrame({"feature": cols,
                               "importance": model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model[-1].coef_[0])})
    importance.sort_values("importance", ascending=False).to_csv(ROOT / "reports/feature_importance.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    PrecisionRecallDisplay.from_predictions(data.y[indices["test"]], p_test, ax=axes[0])
    RocCurveDisplay.from_predictions(data.y[indices["test"]], p_test, ax=axes[1])
    bins = np.linspace(0, 1, 11)
    bin_ids = np.minimum(np.digitize(p_test, bins) - 1, 9)
    binned = [(p_test[bin_ids == i].mean(), data.y[indices["test"]][bin_ids == i].mean()) for i in range(10) if (bin_ids == i).sum()]
    axes[2].plot([0, 1], [0, 1], "--", color="grey")
    axes[2].plot(*np.array(binned).T, marker="o")
    axes[2].set(xlabel="Mean predicted risk", ylabel="Observed spam rate", title="Reliability (test)")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/model_curves.png", dpi=150)
    plt.close(fig)

    metrics = {"selected_model": winner, "selected_features": len(cols), "seed": seed,
               "test_calibrated": ranking_metrics(data.y[indices["test"]], p_test),
               "test_raw": ranking_metrics(data.y[indices["test"]], test_predictions[winner]),
               "graph_vs_original_pr_auc_ci": bootstrap_pr_difference(data.y[indices["test"]],
                   test_predictions["M2 LightGBM all graph"], test_predictions["M1 LightGBM original"], seed),
               "default_costs": costs.__dict__, "default_thresholds": thresholds}
    _write_json(ROOT / "reports/summary.json", metrics)
    build_report(quality, rel, val_df, test_df, comparison, metrics)
    print(f"Done. Selected {winner}; test PR-AUC {metrics['test_calibrated']['pr_auc']:.3f}", flush=True)


def build_report(quality, rel, val_df, test_df, comparison, metrics):
    p = ROOT / "reports/technical_report.md"
    winner = metrics["selected_model"]
    s3 = comparison["S3 allow/review/block"]
    s0 = comparison["S0 allow all"]
    ci = metrics["graph_vs_original_pr_auc_ci"]
    content = f"""# YelpChi offline abuse risk study

## Question and data

Can relation features improve spam ranking and help allocate a limited review budget? We use the public YelpChi graph, with **{quality['nodes']:,}** review nodes, **{quality['positive']:,}** filtered reviews ({quality['positive_rate']:.1%}), 32 engineered numeric features, and R-U-R / R-S-R / R-T-R edges. The filtering flag is a proxy label, not an independently verified ground truth. No raw text, user identity, device data, or usable event timeline is present.

Source: [CARE-GNN data repository](https://github.com/YingtongDou/CARE-GNN). SHA-256 of the input `.mat`: `{quality['source_sha256']}`.

## Protocol

Stratified node split, seed {metrics['seed']}: train {quality['split_counts']['train']:,}, model selection {quality['split_counts']['model_val']:,}, calibration {quality['split_counts']['calibration']:,}, strategy selection {quality['split_counts']['strategy_val']:,}, untouched test {quality['split_counts']['test']:,}. The graph setting is **transductive**: features and edges for all nodes are visible, but labels outside the relevant training or validation split are never model inputs. There is no time-based or new-user generalization claim. Model hyperparameters are fixed; winner selected by validation PR-AUC. Isotonic calibration uses the separate calibration split. Thresholds use the separate strategy split. The test set is only used for final offline reporting.

## Data quality and graph exploration

Missing/nonfinite feature values: {quality['missing_feature_values']}/{quality['nonfinite_feature_values']}; duplicate numeric feature rows: {quality['duplicate_feature_rows']:,}. Relation edge label agreement is descriptive only and is never a predictor.

{rel.to_markdown(index=False, floatfmt='.3f')}

Feature association on training nodes: [`feature_association.csv`](feature_association.csv). SQL summary: [`sql_risk_summary.csv`](sql_risk_summary.csv). Numeric feature names are anonymized; individual features are not assigned semantic meanings.

## Ranking and calibration

Validation ranking used for model choice:

{val_df[['model','feature_count','pr_auc','roc_auc','precision_at_5pct','recall_at_5pct']].to_markdown(index=False, floatfmt='.3f')}

Held-out test ranking, raw model outputs:

{test_df[['model','pr_auc','roc_auc','precision_at_5pct','recall_at_5pct']].to_markdown(index=False, floatfmt='.3f')}

Selected: **{winner}**. Calibrated test PR-AUC {metrics['test_calibrated']['pr_auc']:.3f}, ROC-AUC {metrics['test_calibrated']['roc_auc']:.3f}, Brier {metrics['test_calibrated']['brier']:.3f}. Raw Brier {metrics['test_raw']['brier']:.3f}. The paired node-bootstrap 95% interval for the *raw* M2 minus M1 test PR-AUC is [{ci[0]:.3f}, {ci[1]:.3f}]; this describes split-level sampling uncertainty, not temporal or graph-independent uncertainty. See [`model_curves.png`](model_curves.png) and [`feature_importance.csv`](feature_importance.csv) for ranking, calibration, and split-count importance or coefficient magnitude. Importance is model-specific and is not a causal explanation.

## Risk decisions and business assumptions

Actions: allow, human review, or block. Base costs are hypothetical: false negative 1, false positive 5, review 0.15, reviewer accuracy 100%, capacity 5% of reviews. Reviewed positives missed and reviewed negatives incorrectly blocked are counted in expectation when reviewer accuracy is below 100%. S1 and S3 cutoffs are optimized on strategy validation; S2 reviews the highest-scored 5%. **No test labels are used to tune thresholds.**

| Policy | Review | Block | Expected recall | Expected FPR | Cost / 1,000 |
|:--|--:|--:|--:|--:|--:|
"""
    for name, r in comparison.items():
        content += f"| {name} | {r['review']} | {r['block']} | {r['expected_recall']:.3f} | {r['expected_fpr']:.3f} | {r['cost_per_1000']:.2f} |\n"
    content += f"""

At the base assumptions, S3 lowers simulated cost by {(s0['cost_per_1000'] - s3['cost_per_1000']):.2f} cost units per 1,000 reviews versus allowing all. This is an **offline cost proxy**, not measured dollars or a production impact estimate. See [`strategy_sensitivity.csv`](strategy_sensitivity.csv) for capacities 2/5/10%, false-positive cost ratios 1/5/10, and reviewer accuracies 100/90/80%.

## Error analysis and robustness

[`error_by_degree.csv`](error_by_degree.csv) breaks recall and FPR into node-degree quartiles; [`error_cases.csv`](error_cases.csv) contains internal node IDs, scores and degrees for high-confidence errors. These are diagnostic examples, not individual review explanations. Relation removal experiments appear in the M3 rows above. The sensitivity table simulates operational assumptions; it does not measure real drift. Prior shift, feature perturbation and new-user generalization remain future work because YelpChi lacks timestamps and user-group identities.

## Limits and next step

Yelp filtering labels may be imperfect. Shared graph connections mean node bootstrap intervals may be optimistic. The relation graph includes test-node structure, so results cannot claim performance on unseen users or future events. Costs and reviewer accuracy are illustrative. A real rollout would require independently reviewed labels, prospective time or group splits, reviewer outcome data, and monitored appeals/false blocks before automatic blocking.
"""
    p.write_text(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/YelpChi.mat")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.data, args.seed)


if __name__ == "__main__":
    main()
