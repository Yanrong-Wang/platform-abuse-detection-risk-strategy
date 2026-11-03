# Resume claim evidence and remaining gap

This file records what can be claimed from the public YelpChi project. It is a note for future resume edits, not a claim of real platform impact.

## Current evidence

- The original model experiment uses disjoint train, model-validation, calibration, strategy-validation and fixed test node splits. Its calibrated test PR-AUC is 0.875 and ROC-AUC is 0.967 under a transductive graph setting.
- The initial two-threshold S3 policy and later expected-loss S4 policy were compared with a simulated 5% review-capacity limit and hypothetical costs. The one-split S4 comparison was developed after the original test had been inspected and is exploratory.
- A [repeated audit](../reports/repeated_strategy_evaluation.md) refits the frozen selected model on 10 stratified node splits and 5 R-U-R component group splits. With assumed 100% reviewer accuracy, S4 has lower simulated cost than S3 in 10/10 node splits and 4/5 group splits. At assumed 90% accuracy it wins 9/10 and 4/5; at 80% it wins only 6/10 and 2/5. This instability makes a blanket “10% cost reduction” claim inappropriate.

## What is still missing for a real-world impact claim

- A new time-based or external review dataset. YelpChi lacks a usable event timeline and complete user identities.
- Independently adjudicated abuse labels and measured reviewer accuracy. Yelp filtering flags are proxy labels, and current review outcomes are simulated.
- Real operational costs, appeal outcomes, and a prospective pilot. Current cost units are hypothetical.

Do not write “reduced platform fraud by X%,” “saved $Y,” or “production-ready.” Resume bullets may describe the built pipeline, calibrated offline metrics, budgeted strategy simulator, and sensitivity analyses, with offline/simulated qualifiers where needed.

## Suggested two resume bullets

- Built a reproducible YelpChi review-abuse detection pipeline for 45,954 reviews, engineering relation-specific graph features and calibrating LightGBM risk scores; achieved 0.875 PR-AUC on the fixed offline node test split.
- Developed a budget-constrained allow/review/block decision simulator and evaluated five policies across review-capacity and cost scenarios; audited the base strategy over 15 node and group splits under three reviewer-accuracy assumptions.
