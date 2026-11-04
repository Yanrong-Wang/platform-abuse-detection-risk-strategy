# YelpChi offline abuse risk study

## Question and data

Can relation features improve spam ranking and help allocate a limited review budget? We use the public YelpChi graph, with **45,954** review nodes, **6,677** filtered reviews (14.5%), 32 engineered numeric features, and R-U-R / R-S-R / R-T-R edges. The filtering flag is a proxy label, not an independently verified ground truth. No raw text, user identity, device data, or usable event timeline is present.

Source: [CARE-GNN data repository](https://github.com/YingtongDou/CARE-GNN). SHA-256 of the input `.mat`: `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42`.

## Protocol

Stratified node split, seed 42: train 25,274, model selection 6,893, calibration 4,595, strategy selection 4,596, fixed test 4,596. The graph setting is **transductive**: features and edges for all nodes are visible, but labels outside the relevant training or validation split are never model inputs. There is no time-based or new-user generalization claim. Model hyperparameters are fixed; winner selected by validation PR-AUC, then refitted on train plus model-selection nodes (32,167). The final probabilities average isotonic and logistic calibration, both fitted only on calibration nodes. Thresholds use only the strategy split. The refit and calibration change was developed after earlier results were inspected; the fixed test remains an iterative offline check.

## Data quality and graph exploration

Missing/nonfinite feature values: 0/0; duplicate numeric feature rows: 0. Relation edge label agreement is descriptive only and is never a predictor.

| relation   |   undirected_edges |   mean_degree |   median_degree |   isolated_nodes |   edge_label_agreement |   both_spam_edge_fraction |   positive_neighbor_spam_rate |
|:-----------|-------------------:|--------------:|----------------:|-----------------:|-----------------------:|--------------------------:|------------------------------:|
| RUR        |              49315 |         2.146 |           1.000 |            22123 |                  0.996 |                     0.018 |                         0.909 |
| RSR        |            3402743 |       148.093 |         152.000 |               40 |                  0.772 |                     0.026 |                         0.186 |
| RTR        |             573616 |        24.965 |          23.000 |              522 |                  0.759 |                     0.026 |                         0.176 |

Feature association on training nodes: [`feature_association.csv`](feature_association.csv). SQL summary: [`sql_risk_summary.csv`](sql_risk_summary.csv). Numeric feature names are anonymized; individual features are not assigned semantic meanings.

## Ranking and calibration

Validation ranking used for model choice:

| model                 |   feature_count |   pr_auc |   roc_auc |   precision_at_5pct |   recall_at_5pct |
|:----------------------|----------------:|---------:|----------:|--------------------:|-----------------:|
| M3 without RTR        |             162 |    0.894 |     0.972 |               0.988 |            0.340 |
| M2 LightGBM all graph |             227 |    0.888 |     0.970 |               0.988 |            0.340 |
| M3 without RSR        |             162 |    0.875 |     0.967 |               0.986 |            0.339 |
| M3 without RUR        |             162 |    0.839 |     0.953 |               0.977 |            0.336 |
| M1 LightGBM original  |              32 |    0.804 |     0.941 |               0.965 |            0.332 |
| M0 logistic original  |              32 |    0.386 |     0.774 |               0.519 |            0.179 |

Held-out test ranking, raw model outputs. The refit row is the final selected model after adding model-selection nodes to training:

| model                  |   pr_auc |   roc_auc |   precision_at_5pct |   recall_at_5pct |
|:-----------------------|---------:|----------:|--------------------:|-----------------:|
| M3 without RTR (refit) |    0.897 |     0.971 |               0.996 |            0.343 |
| M3 without RTR         |    0.888 |     0.968 |               0.996 |            0.343 |
| M2 LightGBM all graph  |    0.879 |     0.966 |               0.996 |            0.343 |
| M3 without RSR         |    0.868 |     0.963 |               0.991 |            0.341 |
| M3 without RUR         |    0.825 |     0.946 |               0.987 |            0.340 |
| M1 LightGBM original   |    0.803 |     0.939 |               0.965 |            0.332 |
| M0 logistic original   |    0.384 |     0.774 |               0.543 |            0.187 |

Selected: **M3 without RTR**. Calibrated test PR-AUC 0.897, ROC-AUC 0.971, Brier 0.040. Raw Brier 0.042. The paired node-bootstrap 95% interval for the *raw* M2 minus M1 test PR-AUC is [0.061, 0.098]; this describes split-level sampling uncertainty, not temporal or graph-independent uncertainty. See [`model_curves.png`](model_curves.png) and [`feature_importance.csv`](feature_importance.csv) for ranking, calibration, and split-count importance or coefficient magnitude. Importance is model-specific and is not a causal explanation.

## Risk decisions and business assumptions

Actions: allow, human review, or block. Base costs are hypothetical: false negative 1, false positive 5, review 0.15, reviewer accuracy 100%, capacity 5% of reviews. Reviewed positives missed and reviewed negatives incorrectly blocked are counted in expectation when reviewer accuracy is below 100%. S1 and S3 cutoffs are optimized on strategy validation; S2 reviews the highest-scored 5%. S4 first chooses the cheaper automatic action for each calibrated score, then assigns reviews to nodes with the largest positive reduction in expected loss, up to the capacity. This exactly optimizes **estimated** additive loss under the stated probability and reviewer assumptions; it does not use labels to assign actions. Cutoffs and the stated cost-based policy choice use strategy-validation labels, not test labels. The policy with lowest strategy-validation cost is **S4 budgeted expected loss**; see [`strategy_selection.csv`](strategy_selection.csv). Split and reviewer-quality comparisons appear in [`strategy_stability_evaluation.md`](strategy_stability_evaluation.md).

| Policy | Review | Block | Expected recall | Expected FPR | Strategy-val cost / 1,000 | Test cost / 1,000 |
|:--|--:|--:|--:|--:|--:|--:|
| S0 allow all | 0 | 0 | 0.0% | 0.000% | 145.13 | 145.34 |
| S1 block threshold | 0 | 394 | 56.3% | 0.458% | 86.38 | 83.12 |
| S2 top-K review | 229 | 0 | 34.1% | 0.000% | 103.43 | 103.21 |
| S3 allow/review/block | 164 | 298 | 64.4% | 0.102% | 66.45 | 61.49 |
| S4 budgeted expected loss | 229 | 346 | 74.0% | 0.255% | 61.22 | 56.21 |


At the base assumptions, S4 improves test expected recall by 0.096 versus S3 and lowers simulated cost by 5.28 cost units per 1,000 reviews. It reviews 65 more nodes and has 6 more false blocks. Its expected FPR is 0.255%, versus 0.102% for S3. S4 lowers simulated cost by 89.13 cost units per 1,000 reviews versus allowing all. This is an **exploratory offline cost proxy**, not measured dollars or a production impact estimate. See [`strategy_sensitivity.csv`](strategy_sensitivity.csv) for capacities 2/5/10%, false-positive cost ratios 1/5/10, and reviewer accuracies 100/90/80%. S4 relies on probability calibration and the assumed symmetric reviewer accuracy; score ties use deterministic input order.

## Error analysis and robustness

[`error_by_degree.csv`](error_by_degree.csv) breaks recall and FPR into node-degree quartiles; [`error_cases.csv`](error_cases.csv) contains internal node IDs, scores and degrees for high-confidence errors. These are diagnostic examples, not individual review explanations. Relation removal experiments appear in the M3 rows above. The sensitivity table simulates operational assumptions; it does not measure real drift. Simulated prior shift and feature perturbation are described in [`robustness.md`](robustness.md). Prospective drift and new-user generalization remain open because YelpChi lacks a usable timeline and complete user identities.

## Limits and next step

Yelp filtering labels may be imperfect. Shared graph connections mean node bootstrap intervals may be optimistic. The relation graph includes test-node structure, so results cannot claim performance on unseen users or future events. Costs and reviewer accuracy are illustrative. The [`repeated strategy audit`](repeated_strategy_evaluation.md) compares S3 and S4, while the [`stability audit`](strategy_stability_evaluation.md) compares scoring pipelines on paired splits. A real rollout would require independently reviewed labels, prospective time or external splits, reviewer outcome data, and monitored appeals/false blocks before automatic blocking.
