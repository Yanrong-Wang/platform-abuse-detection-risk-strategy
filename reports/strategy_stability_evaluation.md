# Strategy stability audit

The action rule and cost assumptions are identical in both pipelines: S4 expected-loss allocation, 5% review capacity, missed-spam cost 1, false-block cost 5 and review cost 0.15. The original pipeline fits the frozen LightGBM on the 55% training subset and uses isotonic calibration. The improved pipeline refits the same model on train plus model-selection nodes (about 70% total) and averages isotonic and logistic calibration fitted on the separate calibration subset. Strategy-validation and test labels are not used in either fit. Both pipelines use the same test nodes in each paired comparison.

Seeds 101–110 and 201–205 were examined while developing the change. The refit/blend combination was fixed before running seeds 301–310 and 401–405. These later partitions are a check on the same historical graph, not a new dataset. R-U-R component splits keep same-user connected groups disjoint. Reviewer accuracies are simulated.

| Batch | Split | Accuracy | Original cost/1k | Improved cost/1k | Cost wins | Original recall | Improved recall | Original FPR | Improved FPR |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| development | stratified_nodes | 100% | 62.3 | 59.6 | 8/10 | 73.0% | 74.7% | 0.367% | 0.361% |
| development | stratified_nodes | 90% | 72.8 | 69.1 | 9/10 | 68.8% | 69.9% | 0.471% | 0.421% |
| development | stratified_nodes | 80% | 82.0 | 78.8 | 8/10 | 62.8% | 64.5% | 0.512% | 0.480% |
| development | rur_component_groups | 100% | 56.7 | 53.1 | 4/5 | 72.4% | 74.8% | 0.221% | 0.214% |
| development | rur_component_groups | 90% | 65.0 | 63.9 | 3/5 | 68.6% | 69.7% | 0.291% | 0.297% |
| development | rur_component_groups | 80% | 76.4 | 73.4 | 3/5 | 63.9% | 65.0% | 0.428% | 0.380% |
| fixed_check | stratified_nodes | 100% | 60.6 | 56.4 | 10/10 | 72.2% | 74.5% | 0.298% | 0.277% |
| fixed_check | stratified_nodes | 90% | 70.8 | 67.0 | 10/10 | 68.6% | 69.8% | 0.413% | 0.365% |
| fixed_check | stratified_nodes | 80% | 80.4 | 76.9 | 9/10 | 62.4% | 64.7% | 0.454% | 0.442% |
| fixed_check | rur_component_groups | 100% | 64.6 | 58.8 | 5/5 | 71.7% | 72.4% | 0.368% | 0.259% |
| fixed_check | rur_component_groups | 90% | 76.3 | 69.1 | 5/5 | 68.8% | 67.6% | 0.546% | 0.341% |
| fixed_check | rur_component_groups | 80% | 85.0 | 78.4 | 5/5 | 62.2% | 62.4% | 0.579% | 0.402% |

## S3 versus S4 on the improved scores

Both policies now receive the same refit/blend risk scores. S3 thresholds are selected only on strategy validation for each reviewer-accuracy scenario.

| Batch | Split | Accuracy | S3 cost/1k | S4 cost/1k | S4 cost wins | S3 recall | S4 recall | S3 FPR | S4 FPR |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| development | stratified_nodes | 100% | 67.6 | 59.6 | 10/10 | 66.5% | 74.7% | 0.326% | 0.361% |
| development | stratified_nodes | 90% | 73.9 | 69.1 | 9/10 | 64.7% | 69.9% | 0.411% | 0.421% |
| development | stratified_nodes | 80% | 79.7 | 78.8 | 6/10 | 61.3% | 64.5% | 0.431% | 0.480% |
| development | rur_component_groups | 100% | 60.5 | 53.1 | 5/5 | 68.0% | 74.8% | 0.209% | 0.214% |
| development | rur_component_groups | 90% | 66.9 | 63.9 | 5/5 | 66.1% | 69.7% | 0.294% | 0.297% |
| development | rur_component_groups | 80% | 72.7 | 73.4 | 2/5 | 63.1% | 65.0% | 0.343% | 0.380% |
| fixed_check | stratified_nodes | 100% | 64.8 | 56.4 | 10/10 | 68.3% | 74.5% | 0.316% | 0.277% |
| fixed_check | stratified_nodes | 90% | 71.1 | 67.0 | 10/10 | 65.4% | 69.8% | 0.370% | 0.365% |
| fixed_check | stratified_nodes | 80% | 77.0 | 76.9 | 5/10 | 61.3% | 64.7% | 0.370% | 0.442% |
| fixed_check | rur_component_groups | 100% | 66.3 | 58.8 | 5/5 | 65.1% | 72.4% | 0.250% | 0.259% |
| fixed_check | rur_component_groups | 90% | 72.5 | 69.1 | 4/5 | 63.3% | 67.6% | 0.330% | 0.341% |
| fixed_check | rur_component_groups | 80% | 78.6 | 78.4 | 2/5 | 61.4% | 62.4% | 0.410% | 0.402% |

The fixed-check batch has lower **mean simulated S4 cost in all six** split/accuracy combinations versus the original scoring pipeline. It does not win every individual split, and mean recall falls in the 90% reviewer-accuracy R-U-R group case. With the improved scores at 80% reviewer accuracy, S3 and S4 have nearly equal mean cost:

- stratified_nodes: S3 77.0 vs S4 76.9 cost/1k; cost standard deviation 5.1 vs 4.9; S4 wins 5/10 paired splits.
- rur_component_groups: S3 78.6 vs S4 78.4 cost/1k; cost standard deviation 6.6 vs 5.7; S4 wins 2/5 paired splits.

This supports the new scoring pipeline more strongly than it establishes a policy winner at 80%. Reviewer outcomes and costs are scenario assumptions.
