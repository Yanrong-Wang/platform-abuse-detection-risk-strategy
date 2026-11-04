"""Run with: streamlit run dashboard/app.py"""
from pathlib import Path
import json
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abuse_risk.strategy import Costs, compare_policies, optimize  # noqa: E402

st.set_page_config(page_title="YelpChi Abuse Risk Lab", layout="wide")
st.title("YelpChi · Abuse Risk Lab")
st.caption("Offline simulation on public historical graph data. Costs and reviewer accuracy are assumptions, not measured platform outcomes.")


@st.cache_data
def load_outputs():
    report = ROOT / "reports"
    artifacts = ROOT / "artifacts"
    return {
        "summary": json.loads((report / "summary.json").read_text()),
        "quality": json.loads((report / "data_quality.json").read_text()),
        "relations": pd.read_csv(report / "relation_summary.csv"),
        "models": pd.read_csv(report / "model_comparison_test.csv"),
        "selection": pd.read_csv(report / "model_selection.csv"),
        "degrees": pd.read_csv(report / "error_by_degree.csv"),
        "importance": pd.read_csv(report / "feature_importance.csv"),
        "strategy": pd.read_csv(artifacts / "strategy_val_scores.csv"),
        "test": pd.read_csv(artifacts / "test_scores.csv"),
        "stability": (pd.read_csv(report / "strategy_stability_evaluation.csv")
                      if (report / "strategy_stability_evaluation.csv").exists() else None),
    }


try:
    output = load_outputs()
except FileNotFoundError:
    st.error("Run `python -m abuse_risk.run` first to generate the experiment artifacts.")
    st.stop()

overview, evaluation, simulator, cases = st.tabs(["Risk overview", "Model evaluation", "Strategy simulator", "Case explorer"])

with overview:
    q = output["quality"]
    a, b, c = st.columns(3)
    a.metric("Review nodes", f"{q['nodes']:,}")
    b.metric("Filtered labels", f"{q['positive']:,}")
    c.metric("Positive rate", f"{q['positive_rate']:.1%}")
    st.plotly_chart(px.histogram(output["test"], x="score", color="label", nbins=30,
                                 barmode="overlay", title="Calibrated risk scores · held-out test"), width="stretch")
    st.subheader("Graph relations")
    st.dataframe(output["relations"], hide_index=True, width="stretch")
    st.caption("Label agreement is descriptive EDA only. Neighbor labels are never used as model features.")

with evaluation:
    summary = output["summary"]
    st.write(f"Selected on model-validation PR-AUC, then refitted on train plus model-validation nodes: **{summary['selected_model']}**. Final scores average isotonic and logistic calibration.")
    a, b, c = st.columns(3)
    a.metric("Calibrated PR-AUC", f"{summary['test_calibrated']['pr_auc']:.3f}")
    b.metric("ROC-AUC", f"{summary['test_calibrated']['roc_auc']:.3f}")
    c.metric("Brier score", f"{summary['test_calibrated']['brier']:.3f}")
    st.plotly_chart(px.bar(output["models"], x="model", y="pr_auc", title="Test PR-AUC by feature ablation"), width="stretch")
    st.dataframe(output["models"], hide_index=True, width="stretch")
    d, e = st.columns(2)
    d.plotly_chart(px.bar(output["degrees"], x="degree_group", y="recall", title="Recall by graph degree"), width="stretch")
    e.plotly_chart(px.bar(output["importance"].head(15), x="importance", y="feature", orientation="h",
                            title="Selected model feature split counts"), width="stretch")
    st.image(str(ROOT / "reports/model_curves.png"), caption="Held-out PR, ROC and reliability curves")

with simulator:
    st.write("Change the assumptions. S1/S3 thresholds are re-optimized on the **strategy-validation** split; S4 allocates reviews by per-case expected loss. All policies are then evaluated on the **fixed test** split using the refit/blend risk scores.")
    left, right = st.columns(2)
    with left:
        capacity = st.slider("Review capacity (% of reviews)", 0, 20, 5, 1) / 100
        fp_cost = st.slider("False-block cost / missed-spam cost", 1.0, 20.0, 5.0, 0.5)
    with right:
        review_cost = st.slider("Cost per manual review", 0.0, 1.0, 0.15, 0.05)
        accuracy = st.slider("Reviewer accuracy", 0.5, 1.0, 1.0, 0.05)
    costs = Costs(false_positive=fp_cost, review=review_cost, review_accuracy=accuracy)
    strategy = output["strategy"]
    test = output["test"]
    cuts = optimize(strategy.label.to_numpy(), strategy.score.to_numpy(), costs, capacity, grid_size=35)
    validation_results = compare_policies(strategy.label.to_numpy(), strategy.score.to_numpy(), costs, capacity, cuts)
    selected_policy = min(validation_results, key=lambda name: validation_results[name]["cost"])
    results = compare_policies(test.label.to_numpy(), test.score.to_numpy(), costs, capacity, cuts)
    comparison = pd.DataFrame([{"policy": name, **value} for name, value in results.items()])
    st.write(f"Lowest simulated cost on strategy validation: **{selected_policy}**")
    st.dataframe(comparison[["policy", "allow", "review", "block", "expected_recall", "expected_fpr", "cost_per_1000"]].style.format({"expected_recall": "{:.1%}", "expected_fpr": "{:.2%}", "cost_per_1000": "{:.1f}"}), hide_index=True, width="stretch")
    st.plotly_chart(px.bar(comparison, x="policy", y="cost_per_1000", title="Simulated cost per 1,000 reviews"), width="stretch")
    st.caption(f"Chosen cutoffs: S1 = {cuts['s1_threshold']:.3f}; S3 allow < {cuts['s3_low']:.3f}, review below {cuts['s3_high']:.3f}, block at or above it. Review outcomes are expected values at the selected accuracy.")
    if output["stability"] is not None:
        st.subheader("Stability across reviewer accuracy")
        fixed = output["stability"].query("phase == 'fixed_check' and pipeline in ['original_s4', 'refit_blend_s4']")
        paired = fixed.pivot_table(index=["split_type", "review_accuracy"], columns="pipeline",
                                   values="cost_per_1000", aggfunc="mean")
        paired = paired.rename(columns={"original_s4": "Original S4", "refit_blend_s4": "Refit/blend S4"})
        st.dataframe(paired.style.format("{:.1f}"), width="stretch")
        st.caption("Mean simulated cost per 1,000 reviews across the fixed-check partitions; both pipelines use the same test nodes within each partition.")

with cases:
    st.write("Only held-out internal node IDs are shown. The dataset has no original review text or personal records.")
    test = output["test"]
    selected = st.selectbox("Internal review node ID", sorted(test.review_id.tolist()))
    row = test.loc[test.review_id == selected].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Risk score", f"{row.score:.3f}")
    b.metric("Filtered label", int(row.label))
    c.metric("Total relation degree", int(row.rur_degree + row.rsr_degree + row.rtr_degree))
    st.dataframe(pd.DataFrame({"relation": ["R-U-R", "R-S-R", "R-T-R"],
                               "neighbor_count": [row.rur_degree, row.rsr_degree, row.rtr_degree]}),
                 hide_index=True, width="stretch")
    st.caption("Feature importance is global model behavior; it cannot explain why this individual node received its score.")
