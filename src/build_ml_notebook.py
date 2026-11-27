"""Builds and executes notebooks/05_machine_learning.ipynb (outputs embedded).
Shows the saved Stage 05 results, re-checks the saved model on the test set and runs the
prediction script on fictional profiles.
Full recomputation: cd src && python preprocessing.py && python train_models.py &&
  python model_evaluation.py && python feature_importance.py && python fairness_analysis.py
Usage:  cd src && python build_ml_notebook.py"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "05_machine_learning.ipynb"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
F = "../reports/figures/stage_05/"
img = lambda *n: code("\n".join(f"display(Image('{F}{x}.png'))" for x in n))   # noqa: E731
cells = [
    md("# Stage 05 — Machine Learning & Predictive Modeling\nGeorgia Income Analytics — Gender & Income Study.\n\n"
       "**Target:** estimated real net monthly earnings of employees (band point estimate, 2025 GEL), "
       "Geostat LFS 2020–2025. Models learn log earnings; predictions back-transformed with smearing. "
       "Split 70/15/15 **by household**; hyper-parameters tuned with household-grouped CV on the training set; "
       "test set used once. **Model explanations are not causal evidence.**"),
    code("import sys, json, numpy as np, pandas as pd, joblib\nfrom IPython.display import Image, display\n"
         "sys.path.insert(0,'../src')\nimport preprocessing as pp, model_evaluation as me, predict_income as pi\n"
         "pd.set_option('display.width',200,'display.max_columns',30,'display.max_colwidth',60)\nR='../reports/'"),
    md("## 1. Data audit"), code("df = pp.load_ml_frame()\npart = pd.read_csv(str(pp.SPLIT_FILE).format(p=pp.MAIN_PERIOD)).part\n"
                                 "print(json.dumps(pp.audit(df, part), indent=1, default=str))"),
    md("## 2. Model comparison (test set)"),
    code("c = pd.read_csv(R+'ml_model_comparison.csv')\nc[c.split=='test'][['experiment','model','MAE','RMSE','R2','MedAE','R2_log','selected_best']]"),
    code("pd.read_csv(R+'ml_cv_results.csv')[['experiment','model','cv_mae_mean','cv_mae_std','best_params','fit_seconds']]"),
    img("03_model_performance_comparison", "09_cross_validation_results", "08_learning_curve"),
    code("display(pd.read_csv(R+'ml_temporal_vs_random.csv'))\ndisplay(pd.read_csv(R+'ml_replication_2017_2019.csv'))"),
    md("## 3. Best model diagnostics"), img("01_actual_vs_predicted", "02_residual_distribution"),
    md("## 4. Gender included vs excluded"),
    code("display(pd.read_csv(R+'ml_gender_impact.csv'))\ndisplay(pd.read_csv(R+'ml_fairness_by_gender.csv'))"),
    img("06_gender_prediction_comparison", "07_error_distribution_by_gender", "11_fairness_by_age_education"),
    md("## 5. Feature importance and SHAP"),
    code("display(pd.read_csv(R+'ml_permutation_importance.csv'))\ndisplay(pd.read_csv(R+'ml_shap_importance.csv'))\n"
         "display(pd.read_csv(R+'ml_shap_gender.csv'))"),
    img("04_feature_importance", "05_shap_summary", "05b_shap_by_feature", "05c_partial_dependence"),
    md("## 6. Saved model: reload and re-check; prediction script (fictional profiles)"),
    code("m = joblib.load('../models/best_income_model.joblib')\nte = df[part.to_numpy()=='test']\n"
         "print(me.metrics(te[pp.TARGET], m.predict(te[pp.FEATURES_A])))\n"
         "pi.predict(pd.DataFrame(pi.DEMO))[['gender','age','education','occupation','predicted_net_monthly_earnings_gel_2025']]"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
nbf.write(nb, NB)
print("written", NB)
