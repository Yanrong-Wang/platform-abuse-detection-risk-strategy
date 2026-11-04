"""Create lightweight, executable exploration notebooks from saved experiment files."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.write(notebook, OUT / name)


save("01_data_exploration.ipynb", [
    nbf.v4.new_markdown_cell("# YelpChi data exploration\n\nRun the pipeline first. These plots are descriptive; neighbor labels are never used in the model."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd, json, matplotlib.pyplot as plt\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nquality = json.loads((ROOT/'reports/data_quality.json').read_text())\nrelations = pd.read_csv(ROOT/'reports/relation_summary.csv')\nquality, relations"),
    nbf.v4.new_code_cell("pd.Series({'normal': quality['nodes']-quality['positive'], 'filtered': quality['positive']}).plot.bar(title='Class balance'); plt.ylabel('Nodes'); plt.show()"),
    nbf.v4.new_code_cell("association = pd.read_csv(ROOT/'reports/feature_association.csv')\nassociation.head(12).sort_values('point_biserial_r').plot.barh(x='feature', y='point_biserial_r', title='Training-only feature association'); plt.show()"),
    nbf.v4.new_code_cell("pd.read_csv(ROOT/'reports/sql_risk_summary.csv')"),
    nbf.v4.new_markdown_cell("**Hypothesis:** R-U-R may help beyond the 32 anonymous features. The experiment tests this with relation ablations. High degree alone is not evidence of abuse."),
])

save("02_feature_analysis.ipynb", [
    nbf.v4.new_markdown_cell("# Graph feature ablation\n\nAll comparisons are on the same held-out split. Model selection happened on a separate validation split."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd, matplotlib.pyplot as plt\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nselection = pd.read_csv(ROOT/'reports/model_selection.csv')\ntest = pd.read_csv(ROOT/'reports/model_comparison_test.csv')\nselection, test"),
    nbf.v4.new_code_cell("test.sort_values('pr_auc').plot.barh(x='model', y='pr_auc', title='Held-out PR-AUC'); plt.show()"),
    nbf.v4.new_code_cell("pd.read_csv(ROOT/'reports/feature_importance.csv').head(20)"),
    nbf.v4.new_markdown_cell("The importance values are LightGBM split counts. They describe the fitted model, not causal effects or individual explanations."),
])

save("03_error_and_strategy.ipynb", [
    nbf.v4.new_markdown_cell("# Error patterns and strategy tradeoffs\n\nScores and case IDs below are from the held-out test split. Strategy thresholds were selected elsewhere."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd, matplotlib.pyplot as plt\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nerrors = pd.read_csv(ROOT/'reports/error_by_degree.csv')\nerrors"),
    nbf.v4.new_code_cell("errors.plot(x='degree_group', y=['recall','fpr'], kind='bar', title='Errors by degree quartile'); plt.show()"),
    nbf.v4.new_code_cell("pd.read_csv(ROOT/'reports/error_cases.csv')"),
    nbf.v4.new_code_cell("comparison = pd.read_csv(ROOT/'reports/strategy_comparison.csv')\ncomparison[['policy','review','block','expected_recall','expected_fpr','cost_per_1000']]"),
    nbf.v4.new_code_cell("sensitivity = pd.read_csv(ROOT/'reports/strategy_sensitivity.csv')\nsensitivity.query('review_accuracy == 0.9 and capacity == 0.05').pivot(index='fp_cost', columns='policy', values='cost_per_1000')"),
    nbf.v4.new_code_cell("stability = pd.read_csv(ROOT/'reports/strategy_stability_evaluation.csv')\ncheck = stability.query(\"phase == 'fixed_check'\")\ncheck.pivot_table(index=['split_type','review_accuracy'], columns='pipeline', values='cost_per_1000', aggfunc='mean').round(1)"),
    nbf.v4.new_markdown_cell("All cost units are hypothetical. The historical graph and proxy labels cannot establish prospective business impact."),
])

print(f"Wrote notebooks to {OUT}")
