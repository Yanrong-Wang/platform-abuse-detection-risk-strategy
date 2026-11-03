"""Repeated strategy validation on fixed node and R-U-R group splits.

Run after the main experiment with ``python -m abuse_risk.validate_strategy``.
This is a robustness audit on YelpChi, not an independent external dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit

from .data import load_yelp, make_features
from .metrics import ranking_metrics
from .run import ROOT, _model, split_indices
from .strategy import Costs, compare_policies, optimize

NODE_SEEDS = tuple(range(101, 111))
GROUP_SEEDS = tuple(range(201, 206))
REVIEW_ACCURACIES = (1.0, 0.9, 0.8)
CAPACITY = 0.05


def group_split_indices(y: np.ndarray, groups: np.ndarray, seed: int):
    """Keep R-U-R connected components disjoint across all five splits."""
    remaining = np.arange(len(y))
    split = {}
    fractions = (("train", 0.55), ("model_val", 1 / 3),
                 ("calibration", 1 / 3), ("strategy_val", 0.5))
    for offset, (name, fraction) in enumerate(fractions):
        divider = GroupShuffleSplit(n_splits=1, train_size=fraction, random_state=seed + offset)
        left, right = next(divider.split(remaining, y[remaining], groups[remaining]))
        split[name], remaining = remaining[left], remaining[right]
    split["test"] = remaining
    group_sets = [set(groups[ids]) for ids in split.values()]
    if any(group_sets[i] & group_sets[j] for i in range(len(group_sets)) for j in range(i + 1, len(group_sets))):
        raise AssertionError("R-U-R group leakage between splits")
    if any(len(np.unique(y[ids])) < 2 for ids in split.values()):
        raise ValueError("A group split lacks one of the labels")
    return split


def fit_one(features: pd.DataFrame, y: np.ndarray, split: dict, seed: int):
    """Freeze the previously selected model specification; refit per split."""
    cols = [c for c in features if not c.startswith("rtr_")]
    model = _model("M3 without RTR", seed)
    model.fit(features.iloc[split["train"]][cols], y[split["train"]])
    calibrator = IsotonicRegression(out_of_bounds="clip")
    raw_calibration = model.predict_proba(features.iloc[split["calibration"]][cols])[:, 1]
    calibrator.fit(raw_calibration, y[split["calibration"]])
    scores = {name: calibrator.predict(model.predict_proba(features.iloc[split[name]][cols])[:, 1])
              for name in ("strategy_val", "test")}
    return scores


def run(data_path: Path):
    data = load_yelp(data_path)
    features = make_features(data)
    _, groups = connected_components(data.graphs["rur"], directed=False)
    rows = []
    for split_type, seeds in (("stratified_nodes", NODE_SEEDS), ("rur_component_groups", GROUP_SEEDS)):
        for seed in seeds:
            split = (split_indices(data.y, seed) if split_type == "stratified_nodes"
                     else group_split_indices(data.y, groups, seed))
            scores = fit_one(features, data.y, split, seed)
            test_ranking = ranking_metrics(data.y[split["test"]], scores["test"])
            for accuracy in REVIEW_ACCURACIES:
                costs = Costs(review_accuracy=accuracy)
                cutoffs = optimize(data.y[split["strategy_val"]], scores["strategy_val"],
                                   costs, CAPACITY, grid_size=60)
                validation = compare_policies(data.y[split["strategy_val"]], scores["strategy_val"],
                                              costs, CAPACITY, cutoffs)
                test = compare_policies(data.y[split["test"]], scores["test"], costs, CAPACITY, cutoffs)
                for policy, result in test.items():
                    rows.append({"split_type": split_type, "seed": seed,
                                 "review_accuracy": accuracy, "review_capacity": CAPACITY,
                                 "train_n": len(split["train"]), "test_n": len(split["test"]),
                                 "test_positive_rate": float(data.y[split["test"]].mean()),
                                 "test_pr_auc": test_ranking["pr_auc"],
                                 "test_brier": test_ranking["brier"], "policy": policy,
                                 "validation_cost_per_1000": validation[policy]["cost_per_1000"],
                                 **result})
            print(f"{split_type} seed {seed} complete", flush=True)
    report = pd.DataFrame(rows)
    output = ROOT / "reports/repeated_strategy_evaluation.csv"
    report.to_csv(output, index=False)
    write_summary(report)
    print(f"Wrote {output}")


def write_summary(report: pd.DataFrame):
    lines = ["# Repeated strategy audit", "",
             "Model specification is frozen to M3 without R-T-R, then refitted and recalibrated independently for each split. "
             "S3 thresholds are fitted on the strategy-validation subset of each split; S4 uses calibrated probabilities and the stated cost assumptions. "
             "Seeds 101–110 are stratified node splits. Seeds 201–205 hold out entire R-U-R connected components as a proxy for same-user groups. "
             "Every split includes separate train, model-validation, calibration, strategy-validation and test subsets. "
             "The group proxy does not establish future-user performance: R-S-R and R-T-R connections remain transductive, and the data has no usable timeline.", "",
             "Costs are hypothetical (missed spam 1, false block 5, review 0.15); review capacity is 5%. "
             "The three reviewer accuracies are simulated. All results reuse one historical dataset and are not independent deployments.", "",
             "| Split | Reviewer accuracy | Mean S3 recall | Mean S4 recall | Mean S3 FPR | Mean S4 FPR | Mean S3 cost/1k | Mean S4 cost/1k | S4 lower cost in |",
             "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for split_type in ("stratified_nodes", "rur_component_groups"):
        for accuracy in REVIEW_ACCURACIES:
            part = report[(report.split_type == split_type) & (report.review_accuracy == accuracy)]
            s3 = part[part.policy == "S3 allow/review/block"].set_index("seed")
            s4 = part[part.policy == "S4 budgeted expected loss"].set_index("seed")
            wins = int((s4.cost_per_1000 < s3.cost_per_1000).sum())
            lines.append(f"| {split_type} | {accuracy:.0%} | {s3.expected_recall.mean():.1%} | {s4.expected_recall.mean():.1%} | "
                         f"{s3.expected_fpr.mean():.3%} | {s4.expected_fpr.mean():.3%} | "
                         f"{s3.cost_per_1000.mean():.1f} | {s4.cost_per_1000.mean():.1f} | {wins}/{len(s3)} |")
    lines += ["", "The calibrated model's test PR-AUC across the 10 node splits averaged "
              f"{report[(report.split_type == 'stratified_nodes') & (report.review_accuracy == 1.0) & (report.policy == 'S4 budgeted expected loss')].test_pr_auc.mean():.3f}; "
              "across the 5 R-U-R group splits it averaged "
              f"{report[(report.split_type == 'rur_component_groups') & (report.review_accuracy == 1.0) & (report.policy == 'S4 budgeted expected loss')].test_pr_auc.mean():.3f}. "
              "These are split averages on one graph, not fresh data estimates."]
    lines += ["", "## Interpretation", "",
              "Compare recall, false-positive rate and cost together. Repeated splits measure sensitivity to partition choice, "
              "not a confidence interval for prospective performance; node tests overlap and graph observations are dependent. "
              "The S4 rule was developed after the original seed-42 test was inspected, so even this audit is corroborating evidence, "
              "not a fully independent confirmation. A future time window or external dataset with review outcomes is still needed.", ""]
    (ROOT / "reports/repeated_strategy_evaluation.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/YelpChi.mat")
    args = parser.parse_args()
    run(args.data)


if __name__ == "__main__":
    main()
