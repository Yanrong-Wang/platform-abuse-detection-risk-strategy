"""Paired audit of the original and refit/blended S4 pipelines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from .calibration import BlendedCalibrator
from .data import load_yelp, make_features
from .run import ROOT, _model, split_indices
from .strategy import (Costs, actions_from_thresholds, budgeted_expected_loss_actions,
                       evaluate, optimize)
from .validate_strategy import fit_one, group_split_indices


SPLITS = (
    ("development", "stratified_nodes", range(101, 111)),
    ("development", "rur_component_groups", range(201, 206)),
    ("fixed_check", "stratified_nodes", range(301, 311)),
    ("fixed_check", "rur_component_groups", range(401, 406)),
)
ACCURACIES = (1.0, 0.9, 0.8)
CAPACITY = 0.05


def fit_refit_blend_one(features: pd.DataFrame, y: np.ndarray, split: dict, seed: int):
    """Refit the frozen architecture, then calibrate on separate nodes."""
    cols = [c for c in features if not c.startswith("rtr_")]
    train_ids = np.r_[split["train"], split["model_val"]]
    model = _model("M3 without RTR", seed)
    model.fit(features.iloc[train_ids][cols], y[train_ids])
    raw_calibration = model.predict_proba(features.iloc[split["calibration"]][cols])[:, 1]
    calibrator = BlendedCalibrator().fit(raw_calibration, y[split["calibration"]])
    return {name: calibrator.predict(model.predict_proba(features.iloc[split[name]][cols])[:, 1])
            for name in ("strategy_val", "test")}


def run(data_path: Path):
    data = load_yelp(data_path)
    features = make_features(data)
    _, groups = connected_components(data.graphs["rur"], directed=False)
    rows = []
    for phase, split_type, seeds in SPLITS:
        for seed in seeds:
            split = (split_indices(data.y, seed) if split_type == "stratified_nodes"
                     else group_split_indices(data.y, groups, seed))
            baseline = fit_one(features, data.y, split, seed)["test"]
            improved = fit_refit_blend_one(features, data.y, split, seed)
            for accuracy in ACCURACIES:
                costs = Costs(review_accuracy=accuracy)
                thresholds = optimize(data.y[split["strategy_val"]], improved["strategy_val"],
                                      costs, CAPACITY)
                decisions = (
                    ("original_s4", budgeted_expected_loss_actions(baseline, costs, CAPACITY)),
                    ("refit_blend_s4", budgeted_expected_loss_actions(improved["test"], costs, CAPACITY)),
                    ("refit_blend_s3", actions_from_thresholds(
                        improved["test"], thresholds["s3_low"], thresholds["s3_high"])),
                )
                for pipeline, actions in decisions:
                    result = evaluate(data.y[split["test"]], actions, costs)
                    rows.append({"phase": phase, "split_type": split_type, "seed": seed,
                                 "review_accuracy": accuracy, "pipeline": pipeline,
                                 "train_n": len(split["train"]),
                                 "refit_train_n": len(split["train"]) + len(split["model_val"]),
                                 "test_n": len(split["test"]), **result})
            print(f"{phase} {split_type} seed {seed} complete", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(ROOT / "reports/strategy_stability_evaluation.csv", index=False)
    write_summary(report)


def write_summary(report: pd.DataFrame):
    lines = ["# Strategy stability audit", "",
             "The action rule and cost assumptions are identical in both pipelines: S4 expected-loss allocation, "
             "5% review capacity, missed-spam cost 1, false-block cost 5 and review cost 0.15. "
             "The original pipeline fits the frozen LightGBM on the 55% training subset and uses isotonic calibration. "
             "The improved pipeline refits the same model on train plus model-selection nodes (about 70% total) "
             "and averages isotonic and logistic calibration fitted on the separate calibration subset. "
             "Strategy-validation and test labels are not used in either fit. Both pipelines use the same test nodes in each paired comparison.", "",
             "Seeds 101–110 and 201–205 were examined while developing the change. The refit/blend combination "
             "was fixed before running seeds 301–310 and 401–405. These later partitions are a check on the same "
             "historical graph, not a new dataset. R-U-R component splits keep same-user connected groups disjoint. "
             "Reviewer accuracies are simulated.", "",
             "| Batch | Split | Accuracy | Original cost/1k | Improved cost/1k | Cost wins | Original recall | Improved recall | Original FPR | Improved FPR |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for phase in ("development", "fixed_check"):
        for split_type in ("stratified_nodes", "rur_component_groups"):
            for accuracy in ACCURACIES:
                subset = report[(report.phase == phase) & (report.split_type == split_type)
                                & (report.review_accuracy == accuracy)]
                old = subset[subset.pipeline == "original_s4"].set_index("seed")
                new = subset[subset.pipeline == "refit_blend_s4"].set_index("seed")
                wins = int((new.cost_per_1000 < old.cost_per_1000).sum())
                lines.append(f"| {phase} | {split_type} | {accuracy:.0%} | "
                             f"{old.cost_per_1000.mean():.1f} | {new.cost_per_1000.mean():.1f} | "
                             f"{wins}/{len(old)} | {old.expected_recall.mean():.1%} | "
                             f"{new.expected_recall.mean():.1%} | {old.expected_fpr.mean():.3%} | "
                             f"{new.expected_fpr.mean():.3%} |")
    lines += ["", "## S3 versus S4 on the improved scores", "",
              "Both policies now receive the same refit/blend risk scores. S3 thresholds are selected "
              "only on strategy validation for each reviewer-accuracy scenario.", "",
              "| Batch | Split | Accuracy | S3 cost/1k | S4 cost/1k | S4 cost wins | S3 recall | S4 recall | S3 FPR | S4 FPR |",
              "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for phase in ("development", "fixed_check"):
        for split_type in ("stratified_nodes", "rur_component_groups"):
            for accuracy in ACCURACIES:
                subset = report[(report.phase == phase) & (report.split_type == split_type)
                                & (report.review_accuracy == accuracy)]
                s3 = subset[subset.pipeline == "refit_blend_s3"].set_index("seed")
                s4 = subset[subset.pipeline == "refit_blend_s4"].set_index("seed")
                wins = int((s4.cost_per_1000 < s3.cost_per_1000).sum())
                lines.append(f"| {phase} | {split_type} | {accuracy:.0%} | "
                             f"{s3.cost_per_1000.mean():.1f} | {s4.cost_per_1000.mean():.1f} | "
                             f"{wins}/{len(s3)} | {s3.expected_recall.mean():.1%} | "
                             f"{s4.expected_recall.mean():.1%} | {s3.expected_fpr.mean():.3%} | "
                             f"{s4.expected_fpr.mean():.3%} |")
    lines += ["", "The fixed-check batch has lower **mean simulated S4 cost in all six** split/accuracy "
              "combinations versus the original scoring pipeline. It does not win every individual split, "
              "and mean recall falls in the 90% reviewer-accuracy R-U-R group case. With the improved scores "
              "at 80% reviewer accuracy, S3 and S4 have nearly equal mean cost:", ""]
    for split_type in ("stratified_nodes", "rur_component_groups"):
        subset = report[(report.phase == "fixed_check") & (report.split_type == split_type)
                        & (report.review_accuracy == 0.8)]
        s3 = subset[subset.pipeline == "refit_blend_s3"].set_index("seed")
        s4 = subset[subset.pipeline == "refit_blend_s4"].set_index("seed")
        wins = int((s4.cost_per_1000 < s3.cost_per_1000).sum())
        lines.append(f"- {split_type}: S3 {s3.cost_per_1000.mean():.1f} vs S4 "
                     f"{s4.cost_per_1000.mean():.1f} cost/1k; cost standard deviation "
                     f"{s3.cost_per_1000.std():.1f} vs {s4.cost_per_1000.std():.1f}; "
                     f"S4 wins {wins}/{len(s3)} paired splits.")
    lines += ["", "This supports the new scoring pipeline more strongly than it establishes a "
              "policy winner at 80%. Reviewer outcomes and costs are scenario assumptions.", ""]
    (ROOT / "reports/strategy_stability_evaluation.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/YelpChi.mat")
    args = parser.parse_args()
    run(args.data)


if __name__ == "__main__":
    main()
