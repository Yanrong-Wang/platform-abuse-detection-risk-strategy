# Simulated robustness diagnostics

## Class-prior shift

The test samples are importance weighted to target spam prevalences of 5%, 10%, 14.5%, 20%, and 30%, assuming class-conditional score distributions stay fixed. This is a scenario analysis, not observed drift. At 5% prevalence, weighted precision among scores >= 0.5 is 0.612; at 30% it is 0.928. The same fixed cutoff can have very different precision when prevalence changes. See [`prior_shift.csv`](prior_shift.csv).

## Feature perturbation

Independent Gaussian noise at 1% of each original feature's standard deviation was added to every node, then all graph aggregates and held-out scores were recomputed. Mean absolute score movement is 0.019; 95th percentile 0.113; 1.3% of test nodes cross the 0.5 cutoff. PR-AUC changes from 0.875 to 0.846. This is one fixed-seed perturbation, not a reliability guarantee. See [`feature_perturbation.csv`](feature_perturbation.csv).

## Monitoring design

For a prospective pilot, log score histogram, Population Stability Index versus a fixed reference window, review and block rates, reviewer-confirmed positive rate, appeals/overturns, and delayed false-positive/false-negative estimates. Stratify by graph degree and relation coverage. Alert thresholds should be calibrated on future operational data; YelpChi has no usable event timeline to set them.
