"""Simulated prior shift and numeric-feature perturbation diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .data import load_yelp, make_features

ROOT = Path(__file__).resolve().parents[2]


def reweighted_metrics(y, p, target_prior):
    """Importance-weight test labels to a target class prior, assuming P(x|y) fixed."""
    observed = float(np.mean(y))
    weights = np.where(y == 1, target_prior / observed, (1 - target_prior) / (1 - observed))
    return {"target_prior": target_prior,
            "weighted_brier": float(np.average((p - y) ** 2, weights=weights)),
            "weighted_pr_auc": float(average_precision_score(y, p, sample_weight=weights)),
            "weighted_precision_at_0_5": float(np.sum(weights * (y == 1) * (p >= 0.5)) /
                                                   max(np.sum(weights * (p >= 0.5)), 1e-12))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/YelpChi.mat")
    args = parser.parse_args()
    bundle = joblib.load(ROOT / "artifacts/risk_model.joblib")
    test = pd.read_csv(ROOT / "artifacts/test_scores.csv")
    data = load_yelp(args.data)
    ids = test.review_id.to_numpy()
    baseline = test.score.to_numpy()
    rows = [reweighted_metrics(test.label.to_numpy(), baseline, prior) for prior in [0.05, 0.10, 0.145, 0.20, 0.30]]
    pd.DataFrame(rows).to_csv(ROOT / "reports/prior_shift.csv", index=False)

    rng = np.random.default_rng(42)
    std = np.std(data.x, axis=0)
    noise = rng.normal(size=data.x.shape).astype(np.float32) * (0.01 * std)
    data.x = data.x + noise
    perturbed = make_features(data).iloc[ids][bundle["columns"]]
    p_noisy = bundle["calibrator"].predict(bundle["model"].predict_proba(perturbed)[:, 1])
    perturbation = pd.DataFrame([{
        "noise_scale_feature_sd": 0.01,
        "mean_abs_score_change": float(np.mean(np.abs(p_noisy - baseline))),
        "p95_abs_score_change": float(np.quantile(np.abs(p_noisy - baseline), 0.95)),
        "share_action_changes_at_0_5": float(np.mean((p_noisy >= 0.5) != (baseline >= 0.5))),
        "pr_auc_before": float(average_precision_score(test.label, baseline)),
        "pr_auc_after": float(average_precision_score(test.label, p_noisy)),
        "brier_before": float(brier_score_loss(test.label, baseline)),
        "brier_after": float(brier_score_loss(test.label, p_noisy)),
    }])
    perturbation.to_csv(ROOT / "reports/feature_perturbation.csv", index=False)
    report = f"""# Simulated robustness diagnostics

## Class-prior shift

The test samples are importance weighted to target spam prevalences of 5%, 10%, 14.5%, 20%, and 30%, assuming class-conditional score distributions stay fixed. This is a scenario analysis, not observed drift. At 5% prevalence, weighted precision among scores >= 0.5 is {rows[0]['weighted_precision_at_0_5']:.3f}; at 30% it is {rows[-1]['weighted_precision_at_0_5']:.3f}. The same fixed cutoff can have very different precision when prevalence changes. See [`prior_shift.csv`](prior_shift.csv).

## Feature perturbation

Independent Gaussian noise at 1% of each original feature's standard deviation was added to every node, then all graph aggregates and held-out scores were recomputed. Mean absolute score movement is {perturbation.iloc[0]['mean_abs_score_change']:.3f}; 95th percentile {perturbation.iloc[0]['p95_abs_score_change']:.3f}; {perturbation.iloc[0]['share_action_changes_at_0_5']:.1%} of test nodes cross the 0.5 cutoff. PR-AUC changes from {perturbation.iloc[0]['pr_auc_before']:.3f} to {perturbation.iloc[0]['pr_auc_after']:.3f}. This is one fixed-seed perturbation, not a reliability guarantee. See [`feature_perturbation.csv`](feature_perturbation.csv).

## Monitoring design

For a prospective pilot, log score histogram, Population Stability Index versus a fixed reference window, review and block rates, reviewer-confirmed positive rate, appeals/overturns, and delayed false-positive/false-negative estimates. Stratify by graph degree and relation coverage. Alert thresholds should be calibrated on future operational data; YelpChi has no usable event timeline to set them.
"""
    (ROOT / "reports/robustness.md").write_text(report)
    print("Wrote prior-shift and feature-perturbation diagnostics")


if __name__ == "__main__":
    main()
