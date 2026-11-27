"""
Stage 05 — Evaluation metrics and evaluation figures.

metrics(): MAE, MSE, RMSE, R^2, Median Absolute Error (GEL, unweighted = the ML standard),
           plus survey-weighted MAE and R^2 on the log scale for reference.
Figures (reports/figures/stage_05/):
  01 actual vs predicted, 02 residual distribution, 03 model performance comparison,
  08 learning curve, 09 cross-validation results, (10 temporal validation)
Usage:  cd src && python model_evaluation.py     (after train_models.py)
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, learning_curve

import data_collection as dc
import eda

REPORTS = dc.PROJECT_ROOT / "reports"
FIG = REPORTS / "figures" / "stage_05"
SRC = ("Source: Geostat LFS microdata 2020-2025; employees; target = estimated real net monthly earnings "
       "(band point estimate, 2025 GEL); household-grouped test set.")


def metrics(y_true, y_pred, weight=None) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = mean_squared_error(y_true, y_pred)
    out = dict(n=len(y_true), MAE=mean_absolute_error(y_true, y_pred), MSE=mse, RMSE=float(np.sqrt(mse)),
               R2=r2_score(y_true, y_pred), MedAE=median_absolute_error(y_true, y_pred),
               R2_log=r2_score(np.log(y_true), np.log(y_pred)),
               mean_error=float(np.mean(y_pred - y_true)))
    if weight is not None:
        w = np.asarray(weight, float)
        out["MAE_weighted"] = float(np.average(np.abs(y_true - y_pred), weights=w))
    return out


def load_predictions():
    import preprocessing as pp
    return pd.read_csv(pp.PRED_FILE)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    import preprocessing as pp
    comp = pd.read_csv(REPORTS / "ml_model_comparison.csv")
    cvres = pd.read_csv(REPORTS / "ml_cv_results.csv")
    best = comp[comp.selected_best].model.iloc[0]
    pr = load_predictions()
    te = pr[pr.part == "test"]
    y = pp.TARGET
    sns.set_theme(style="whitegrid", font_scale=0.9)

    # 01 actual vs predicted (best, A) — banded target -> show distribution per actual value
    p = te[f"pred|A_gender_included|{best}"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(te[y], p, s=4, alpha=.15, color="#4C72B0")
    lim = [0, np.percentile(te[y], 99.5) * 1.05]
    ax.plot(lim, lim, color="black", lw=.8, ls="--")
    ax.set(xlim=lim, ylim=lim, title=f"Actual vs predicted earnings — {best} (gender included), test set",
           xlabel="Actual (band point estimate), GEL/month", ylabel="Predicted, GEL/month")
    ax = axes[1]
    bins = pd.qcut(p, 20, duplicates="drop")
    g = pd.DataFrame({"pred": p, "act": te[y]}).groupby(bins, observed=True).mean()
    ax.plot(g.pred, g.act, "o-", color="#55A868")
    ax.plot(lim, lim, color="black", lw=.8, ls="--")
    ax.set(xlim=lim, ylim=lim, title="Calibration: mean actual within 20 prediction bins",
           xlabel="Mean predicted, GEL/month", ylabel="Mean actual, GEL/month")
    eda.caption(fig, SRC + " Vertical stripes = earnings bands (the survey publishes bands, not amounts).")
    _save(fig, "01_actual_vs_predicted.png")

    # 02 residuals
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    r = p - te[y]
    sns.histplot(r, bins=80, ax=axes[0], color="#4C72B0")
    axes[0].axvline(0, color="black", lw=.8)
    axes[0].set(title="Residuals (predicted - actual), GEL", xlabel="GEL/month", xlim=np.percentile(r, [0.5, 99.5]))
    rl = np.log(p) - np.log(te[y])
    sns.histplot(rl, bins=80, ax=axes[1], color="#8172B2")
    axes[1].axvline(0, color="black", lw=.8)
    axes[1].set(title="Residuals on log scale", xlabel="log(predicted) - log(actual)")
    eda.caption(fig, SRC + f" Model: {best}, gender included.")
    _save(fig, "02_residual_distribution.png")

    # 03 performance comparison (test)
    t = comp[comp.split == "test"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)   # model names only on the first panel
    for ax, m, lab in zip(axes, ["MAE", "RMSE", "R2"], ["MAE, GEL (lower = better)", "RMSE, GEL (lower = better)",
                                                       "R² (higher = better)"]):
        sns.barplot(data=t, y="model", x=m, hue="experiment", ax=ax, palette=["#4C72B0", "#BBBBBB"])
        ax.set(title=lab, xlabel=m, ylabel="")
        lo = t[m].min()
        ax.set_xlim(lo * 0.9 if m != "R2" else max(0, lo - 0.05), t[m].max() * 1.02)
        ax.legend(fontsize=7, frameon=False, title=None) if ax is axes[0] else ax.get_legend().remove()
    eda.caption(fig, SRC + " Same split and search for every model; blue = gender included, grey = excluded.")
    _save(fig, "03_model_performance_comparison.png")

    # 08 learning curve (best, A) on the training set with household-grouped CV
    df = pp.load_ml_frame()
    part = pd.read_csv(str(pp.SPLIT_FILE).format(p=pp.MAIN_PERIOD)).part
    tr = df[part.to_numpy() == "train"]
    model = joblib.load(pp.MODELS_DIR / "best_income_model.joblib")
    sizes, tr_sc, va_sc = learning_curve(model, tr[pp.FEATURES_A], tr[y], groups=tr.household_id,
                                         cv=GroupKFold(3), train_sizes=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                                         scoring="neg_mean_absolute_error", n_jobs=1)
    lc = pd.DataFrame({"train_size": sizes, "train_mae": -tr_sc.mean(1), "train_sd": tr_sc.std(1),
                       "cv_mae": -va_sc.mean(1), "cv_sd": va_sc.std(1)})
    lc.to_csv(REPORTS / "ml_learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(lc.train_size, lc.train_mae, lc.train_sd, fmt="o-", label="training folds", color="#4C72B0")
    ax.errorbar(lc.train_size, lc.cv_mae, lc.cv_sd, fmt="s-", label="validation folds (household-grouped CV)",
                color="#C44E52")
    ax.set(title=f"Learning curve — {best} (gender included)", xlabel="Training rows", ylabel="MAE, GEL/month")
    ax.legend(frameon=False)
    eda.caption(fig, SRC + " Converging curves with a small gap = the model is limited by the information in the "
                "features (and by banded earnings), not by sample size.")
    _save(fig, "08_learning_curve.png")

    # 09 CV results
    rows = []
    for _, r_ in cvres.iterrows():
        for k, v in enumerate(json.loads(r_.cv_mae_folds)):
            rows.append(dict(model=r_.model, experiment=r_.experiment, fold=k + 1, MAE=v))
    cv = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.stripplot(data=cv, x="MAE", y="model", hue="experiment", dodge=True, ax=ax, size=7,
                  palette=["#4C72B0", "#BBBBBB"])
    sns.pointplot(data=cv, x="MAE", y="model", hue="experiment", dodge=.4, ax=ax, errorbar=None,
                  markers="|", linestyles="none", palette=["#1f3b63", "#777777"], legend=False)
    ax.set(title="3-fold household-grouped cross-validation on the training set (best hyper-parameters)",
           xlabel="MAE, GEL/month", ylabel="")
    ax.legend(frameon=False, fontsize=8)
    eda.caption(fig, SRC + " Each dot = one fold; spread = stability.")
    _save(fig, "09_cross_validation_results.png")

    # 10 temporal validation
    tv = pd.read_csv(REPORTS / "ml_temporal_validation.csv")
    random_te = comp[(comp.split == "test")][["model", "experiment", "MAE", "R2"]]
    m = tv.merge(random_te, on=["model", "experiment"], suffixes=("_temporal", "_random"))
    m.to_csv(REPORTS / "ml_temporal_vs_random.csv", index=False)
    return comp, lc, m


if __name__ == "__main__":
    c, lc, m = run()
    pd.set_option("display.width", 250)
    print(lc.round(1).to_string(index=False))
    print(m.round(3).to_string(index=False))
