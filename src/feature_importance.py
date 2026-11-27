"""
Stage 05 — Feature importance and explainability (explains the MODEL, not causes).

  * Permutation importance on the TEST set, by original feature (the raw column is permuted,
    so all one-hot columns of a categorical feature move together); scoring = MAE in GEL.
  * SHAP values (log-earnings scale) for the best model; one-hot SHAP values summed back to the
    original feature; beeswarm of the most influential encoded columns.
  * Partial dependence (GEL) for age, potential experience and usual weekly hours.

Outputs: reports/ml_permutation_importance.csv, reports/ml_shap_importance.csv,
         reports/figures/stage_05/04_feature_importance.png, 05_shap_summary.png,
         05b_shap_by_feature.png, 05c_partial_dependence.png
Usage:   cd src && python feature_importance.py     (after train_models.py)
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

import eda
import model_evaluation as me
import preprocessing as pp

N_PERM, N_SHAP = 8000, 3000


def test_frame():
    df = pp.load_ml_frame()
    part = pd.read_csv(str(pp.SPLIT_FILE).format(p=pp.MAIN_PERIOD)).part
    return df[part.to_numpy() == "test"].reset_index(drop=True)


def original_feature(encoded: str, features: list[str]) -> str:
    name = encoded.split("__", 1)[1]
    if name.startswith("missingindicator_"):
        name = name.replace("missingindicator_", "")
    for f in sorted(features, key=len, reverse=True):
        if name == f or name.startswith(f + "_"):
            return f
    return name


def permutation(models: dict, te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = te.sample(min(N_PERM, len(te)), random_state=pp.RANDOM_STATE)
    for exp, (model, feats) in models.items():
        r = permutation_importance(model, sub[feats], sub[pp.TARGET], scoring="neg_mean_absolute_error",
                                   n_repeats=5, random_state=pp.RANDOM_STATE, n_jobs=1)
        for f, m, s in zip(feats, r.importances_mean, r.importances_std):
            rows.append(dict(experiment=exp, feature=f, mae_increase_gel=m, sd=s))
    return pd.DataFrame(rows).sort_values(["experiment", "mae_increase_gel"], ascending=[True, False])


def shap_values(model, te: pd.DataFrame, feats: list[str]):
    sub = te.sample(min(N_SHAP, len(te)), random_state=pp.RANDOM_STATE)
    prep, reg = model.named_steps["prep"], model.named_steps["model"].regressor_
    X = pd.DataFrame(prep.transform(sub[feats]), columns=prep.get_feature_names_out())
    if hasattr(reg, "feature_importances_"):
        expl = shap.TreeExplainer(reg)
        sv = expl.shap_values(X)
    else:
        bg = shap.sample(X, 300, random_state=pp.RANDOM_STATE)
        sv = shap.LinearExplainer(reg, bg).shap_values(X)
    sv = pd.DataFrame(np.asarray(sv), columns=X.columns)
    grouped = sv.T.groupby([original_feature(c, feats) for c in sv.columns]).sum().T
    return X, sv, grouped, sub


def run():
    best = pd.read_csv(me.REPORTS / "ml_model_comparison.csv").query("selected_best").model.iloc[0]
    te = test_frame()
    models = {"A_gender_included": (joblib.load(pp.MODELS_DIR / "best_income_model.joblib"), pp.FEATURES_A),
              "B_gender_excluded": (joblib.load(pp.MODELS_DIR / "best_income_model_gender_excluded.joblib"),
                                    pp.FEATURES_B)}
    perm = permutation(models, te)
    perm.to_csv(me.REPORTS / "ml_permutation_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, (exp, g) in zip(axes, perm.groupby("experiment")):
        g = g.sort_values("mae_increase_gel")
        ax.barh(g.feature, g.mae_increase_gel, xerr=g.sd, color="#4C72B0" if exp.startswith("A") else "#999999")
        ax.set(title=f"Permutation importance — {best}, {exp.replace('_', ' ')}",
               xlabel="Increase in test MAE when the feature is shuffled (GEL/month)")
    eda.caption(fig, me.SRC + " Importance = how much the MODEL relies on a feature for prediction — not a causal effect.")
    me._save(fig, "04_feature_importance.png")

    model, feats = models["A_gender_included"]
    X, sv, grouped, sub = shap_values(model, te, feats)
    imp = grouped.abs().mean().sort_values(ascending=False)
    g_sh = grouped["gender"]
    shap_tab = pd.DataFrame({"feature": imp.index, "mean_abs_shap_log": imp.values,
                             "approx_pct_effect": (np.exp(imp.values) - 1) * 100})
    shap_tab.to_csv(me.REPORTS / "ml_shap_importance.csv", index=False)
    gender_shap = pd.DataFrame({"gender": sub.gender.values, "shap_gender": g_sh.values}).groupby("gender") \
        .shap_gender.agg(["mean", "std", "count"])
    gender_shap["pct"] = (np.exp(gender_shap["mean"]) - 1) * 100
    gender_shap.to_csv(me.REPORTS / "ml_shap_gender.csv")

    plt.figure()
    top = sv.abs().mean().sort_values(ascending=False).index[:20]
    shap.summary_plot(sv[top].to_numpy(), X[top], feature_names=[c.split("__", 1)[1] for c in top], show=False,
                      plot_size=(10, 7))
    fig = plt.gcf()
    plt.title(f"SHAP summary — {best} (gender included), top 20 encoded features, log-earnings scale")
    eda.caption(fig, me.SRC + f" {len(sub)} test rows. SHAP explains the model's prediction relative to its average "
                "prediction; it is not causal evidence.")
    me._save(fig, "05_shap_summary.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    s = imp.sort_values()
    ax.barh(s.index, s.values, color=["#C44E52" if i == "gender" else "#4C72B0" for i in s.index])
    ax.set(title="Mean |SHAP| by original feature (log-earnings scale)", xlabel="Mean |SHAP value| (log points)")
    eda.caption(fig, me.SRC + " One-hot SHAP values summed per original feature.")
    me._save(fig, "05b_shap_by_feature.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    sample = te.sample(3000, random_state=pp.RANDOM_STATE)
    PartialDependenceDisplay.from_estimator(model, sample[feats], ["age", "potential_experience", "working_hours"],
                                            ax=ax, kind="average", grid_resolution=30)
    for a in ax:
        a.set_ylabel("Predicted GEL/month")
    fig.suptitle(f"Partial dependence — {best} (gender included)")
    eda.caption(fig, me.SRC + " Average model prediction when one feature is varied and the others are kept as observed.")
    me._save(fig, "05c_partial_dependence.png")
    return perm, shap_tab, gender_shap


if __name__ == "__main__":
    p, s, g = run()
    pd.set_option("display.width", 200)
    print(p.round(2).to_string(index=False))
    print(s.round(4).to_string(index=False))
    print(g)
