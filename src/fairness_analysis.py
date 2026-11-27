"""
Stage 05 — Gender impact and fairness of the income models (test set only).

1. Accuracy by sex (A = gender included, B = excluded): n, mean actual, mean predicted,
   mean prediction error (pred - actual; > 0 = over-prediction), MAE, RMSE, MAE relative to
   mean actual; household-cluster bootstrap CI for the women-minus-men difference.
2. Same by age group x sex and education x sex.
3. Predicted gender gap: actual gap vs gap in predictions of A and B.
4. Gender flip (model A): predict every person with the other sex, all else equal ->
   the model-implied difference "at equal observed characteristics" (a model property, not causal).
5. Proxy check: how well do the NON-gender features predict sex (ROC AUC)? A high AUC means
   removing gender does not remove gender information (occupation, industry, hours act as proxies).

Outputs: reports/ml_fairness_by_gender.csv, ml_fairness_by_group.csv, ml_gender_impact.csv,
         reports/figures/stage_05/06_gender_prediction_comparison.png,
         07_error_distribution_by_gender.png, 11_fairness_by_age_education.png
Usage:   cd src && python fairness_analysis.py     (after train_models.py)
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import eda
import model_evaluation as me
import preprocessing as pp

B = 300
EXP = {"A_gender_included": "A (gender included)", "B_gender_excluded": "B (gender excluded)"}


def group_metrics(actual, pred) -> dict:
    e = pred - actual
    return dict(n=len(actual), mean_actual=actual.mean(), mean_predicted=pred.mean(), mean_error=e.mean(),
                MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), MAE_relative_pct=100 * np.abs(e).mean() / actual.mean())


def bootstrap_diff(te: pd.DataFrame, col: str, stat) -> tuple[float, float]:
    """95% CI of stat(women) - stat(men) with households resampled (Poisson bootstrap)."""
    codes, uniq = pd.factorize(te.household_id)
    rng = np.random.default_rng(pp.RANDOM_STATE)
    out = []
    f = (te.gender == "female").to_numpy()
    for _ in range(B):
        w = rng.poisson(1, len(uniq))[codes].astype(float)
        out.append(stat(te[col].to_numpy()[f], te[pp.TARGET].to_numpy()[f], w[f])
                   - stat(te[col].to_numpy()[~f], te[pp.TARGET].to_numpy()[~f], w[~f]))
    return tuple(np.percentile(out, [2.5, 97.5]))


def _mae(p, a, w):
    return np.average(np.abs(p - a), weights=w)


def _me(p, a, w):
    return np.average(p - a, weights=w)


def run():
    comp = pd.read_csv(me.REPORTS / "ml_model_comparison.csv")
    best = comp[comp.selected_best].model.iloc[0]
    pr = me.load_predictions()
    te = pr[pr.part == "test"].reset_index(drop=True)
    y = pp.TARGET
    te["age_group"] = pd.cut(te.age, eda.AGE_BINS, labels=eda.AGE_LABELS).astype(object).where(te.age >= 18, "15-17")

    rows, grp = [], []
    for name in comp.model.unique():
        for exp in EXP:
            col = f"pred|{exp}|{name}"
            for sex, g in te.groupby("gender"):
                rows.append(dict(model=name, experiment=exp, gender=sex, **group_metrics(g[y], g[col])))
            lo, hi = bootstrap_diff(te, col, _mae)
            rows.append(dict(model=name, experiment=exp, gender="women - men (MAE diff 95% CI)",
                             MAE=rows[-2]["MAE"] - rows[-1]["MAE"], ci_low=lo, ci_high=hi))   # [-2]=female, [-1]=male
            lo, hi = bootstrap_diff(te, col, _me)
            rows.append(dict(model=name, experiment=exp, gender="women - men (mean error diff 95% CI)",
                             mean_error=rows[-3]["mean_error"] - rows[-2]["mean_error"], ci_low=lo, ci_high=hi))
            if name == best:
                for dim in ("age_group", "education"):
                    for (k, sex), g in te.groupby([dim, "gender"]):
                        if len(g) >= 30:
                            grp.append(dict(model=name, experiment=exp, dimension=dim, group=k, gender=sex,
                                            **group_metrics(g[y], g[col])))
    fair = pd.DataFrame(rows)
    fair.to_csv(me.REPORTS / "ml_fairness_by_gender.csv", index=False)
    grp = pd.DataFrame(grp)
    grp.to_csv(me.REPORTS / "ml_fairness_by_group.csv", index=False)

    # gender impact: gaps, flip, proxies
    df = pp.load_ml_frame()
    part = pd.read_csv(str(pp.SPLIT_FILE).format(p=pp.MAIN_PERIOD)).part.to_numpy()
    tr, tst = df[part == "train"], df[part == "test"].reset_index(drop=True)
    mA = joblib.load(pp.MODELS_DIR / "best_income_model.joblib")
    f, m = te.gender == "female", te.gender == "male"
    gap = lambda s: (s[m].mean() - s[f].mean()) / s[m].mean() * 100   # noqa: E731
    flipped = tst[pp.FEATURES_A].copy()
    flipped["gender"] = flipped.gender.map({"female": "male", "male": "female"})
    base, alt = mA.predict(tst[pp.FEATURES_A]), mA.predict(flipped)
    fw = (tst.gender == "female").to_numpy()
    clf = HistGradientBoostingClassifier(random_state=pp.RANDOM_STATE)
    clf_pipe = pp.build_preprocessor(pp.FEATURES_B)
    Xtr = clf_pipe.fit_transform(tr[pp.FEATURES_B])
    clf.fit(Xtr, (tr.gender == "female").astype(int))
    auc = roc_auc_score((tst.gender == "female").astype(int), clf.predict_proba(clf_pipe.transform(tst[pp.FEATURES_B]))[:, 1])
    impact = pd.DataFrame([
        dict(measure="actual gap in test set (mean, %)", value=gap(te[y])),
        dict(measure=f"gap in predictions, model A {best} (%)", value=gap(te[f"pred|A_gender_included|{best}"])),
        dict(measure=f"gap in predictions, model B {best} (%)", value=gap(te[f"pred|B_gender_excluded|{best}"])),
        dict(measure="gender flip: women predicted as men, mean change (%)", value=100 * (alt[fw] / base[fw] - 1).mean()),
        dict(measure="gender flip: men predicted as women, mean change (%)", value=100 * (alt[~fw] / base[~fw] - 1).mean()),
        dict(measure="gender flip: model-implied gap at equal characteristics (%)",
             value=100 * (1 - np.exp(np.mean(np.log(np.where(fw, base, alt))) - np.mean(np.log(np.where(fw, alt, base)))))),
        dict(measure="proxy check: ROC AUC predicting sex from the other features", value=auc),
        dict(measure="Stage 04 reference: full regression M7, female coefficient (%)", value=-16.685),
    ])
    impact.to_csv(me.REPORTS / "ml_gender_impact.csv", index=False)
    plots(te, fair, grp, best)
    return fair, grp, impact


def plots(te, fair, grp, best):
    y = pp.TARGET
    # 06 gender prediction comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    rows = []
    for sex in ("female", "male"):
        g = te[te.gender == sex]
        rows += [dict(gender=sex, series="actual", value=g[y].mean()),
                 dict(gender=sex, series="predicted A (gender incl.)", value=g[f"pred|A_gender_included|{best}"].mean()),
                 dict(gender=sex, series="predicted B (gender excl.)", value=g[f"pred|B_gender_excluded|{best}"].mean())]
    sns.barplot(data=pd.DataFrame(rows), x="gender", y="value", hue="series", ax=axes[0],
                palette=["#333333", "#4C72B0", "#BBBBBB"])
    axes[0].set(title=f"Mean actual vs predicted earnings by sex — {best}", ylabel="GEL/month", xlabel="")
    axes[0].legend(frameon=False, fontsize=8)
    fb = fair[(fair.model == best) & fair.gender.isin(["female", "male"])]
    sns.barplot(data=fb, x="gender", y="mean_error", hue="experiment", ax=axes[1], palette=["#4C72B0", "#BBBBBB"])
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set(title="Mean prediction error (predicted - actual)", ylabel="GEL/month", xlabel="")
    axes[1].legend(frameon=False, fontsize=8)
    eda.caption(fig, me.SRC + " Model B over-predicts women and under-predicts men: without gender it cannot "
                "reproduce the part of the gap not carried by other features.")
    me._save(fig, "06_gender_prediction_comparison.png")

    # 07 error distribution by gender
    long = pd.concat([pd.DataFrame({"gender": te.gender, "experiment": EXP[e],
                                    "error": te[f"pred|{e}|{best}"] - te[y]}) for e in EXP])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.violinplot(data=long, x="error", y="experiment", hue="gender", split=True, inner="quartile", cut=0, ax=ax,
                   palette={"female": eda.SEX_COLORS["female"], "male": eda.SEX_COLORS["male"]})
    ax.axvline(0, color="black", lw=.8)
    ax.set(xlim=np.percentile(long.error, [1, 99]), title=f"Prediction error distribution by sex — {best}",
           xlabel="Predicted - actual, GEL/month", ylabel="")
    eda.caption(fig, me.SRC + " Dashed lines = quartiles.")
    me._save(fig, "07_error_distribution_by_gender.png")

    # 11 fairness by age / education
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, dim, order in [(axes[0], "age_group", ["15-17"] + eda.AGE_LABELS), (axes[1], "education", eda.EDU_ORDER)]:
        q = grp[grp.dimension == dim].copy()
        q["label"] = q.gender + ", " + q.experiment.map({"A_gender_included": "A", "B_gender_excluded": "B"})
        sns.pointplot(data=q, x="group", y="mean_error", hue="label", order=[o for o in order if o in set(q.group)],
                      ax=ax, palette=["#C44E52", "#E5A0A2", "#4C72B0", "#8DB0D6"], dodge=.3)
        ax.axhline(0, color="black", lw=.8)
        ax.set(title=f"Mean prediction error by {dim.replace('_', ' ')} and sex", ylabel="Predicted - actual, GEL/month",
               xlabel="")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(frameon=False, fontsize=8)
    eda.caption(fig, me.SRC + f" Model {best}; groups with < 30 test rows omitted.")
    me._save(fig, "11_fairness_by_age_education.png")


if __name__ == "__main__":
    f, g, i = run()
    pd.set_option("display.width", 220)
    print(f.round(2).to_string(index=False))
    print(i.round(3).to_string(index=False))
