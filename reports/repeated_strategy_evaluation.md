# Repeated strategy audit

Model specification is frozen to M3 without R-T-R, then refitted and recalibrated independently for each split. S3 thresholds are fitted on the strategy-validation subset of each split; S4 uses calibrated probabilities and the stated cost assumptions. Seeds 101–110 are stratified node splits. Seeds 201–205 hold out entire R-U-R connected components as a proxy for same-user groups. Every split includes separate train, model-validation, calibration, strategy-validation and test subsets. The group proxy does not establish future-user performance: R-S-R and R-T-R connections remain transductive, and the data has no usable timeline.

Costs are hypothetical (missed spam 1, false block 5, review 0.15); review capacity is 5%. The three reviewer accuracies are simulated. All results reuse one historical dataset and are not independent deployments.

| Split | Reviewer accuracy | Mean S3 recall | Mean S4 recall | Mean S3 FPR | Mean S4 FPR | Mean S3 cost/1k | Mean S4 cost/1k | S4 lower cost in |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| stratified_nodes | 100% | 66.1% | 73.0% | 0.311% | 0.367% | 68.4 | 62.3 | 10/10 |
| stratified_nodes | 90% | 62.4% | 68.8% | 0.374% | 0.471% | 76.2 | 72.8 | 9/10 |
| stratified_nodes | 80% | 59.2% | 62.8% | 0.404% | 0.512% | 82.2 | 82.0 | 6/10 |
| rur_component_groups | 100% | 68.1% | 72.4% | 0.255% | 0.221% | 62.1 | 56.7 | 4/5 |
| rur_component_groups | 90% | 63.1% | 68.6% | 0.267% | 0.291% | 70.0 | 65.0 | 4/5 |
| rur_component_groups | 80% | 60.8% | 63.9% | 0.336% | 0.428% | 76.0 | 76.4 | 2/5 |

The calibrated model's test PR-AUC across the 10 node splits averaged 0.870; across the 5 R-U-R group splits it averaged 0.879. These are split averages on one graph, not fresh data estimates.

## Interpretation

Compare recall, false-positive rate and cost together. Repeated splits measure sensitivity to partition choice, not a confidence interval for prospective performance; node tests overlap and graph observations are dependent. The S4 rule was developed after the original seed-42 test was inspected, so even this audit is corroborating evidence, not a fully independent confirmation. A future time window or external dataset with review outcomes is still needed.
