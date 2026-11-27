"""
Stage 04 — Oaxaca-Blinder decomposition of the gender gap in log real earnings.

Twofold decomposition (weighted, LFS employees, per period):
    gap = mean(log y | men) - mean(log y | women)
        = (Xbar_m - Xbar_f) b*                      EXPLAINED   (different characteristics)
        + Xbar_m (b_m - b*) + Xbar_f (b* - b_f)     UNEXPLAINED (different returns + intercepts)
Reference coefficients b* = pooled regression of both sexes WITH a female dummy
(Neumark 1988 / Jann 2008) — avoids attributing the group difference to the covariates.
Sensitivity: b* = men's coefficients and b* = women's coefficients (index-number problem).

Detailed EXPLAINED part by blocks of variables (sum over all dummies of a block, which is
invariant to the omitted category): experience, education, occupation, region & urban,
hours & part-time, industry, job type (sector, informality, contract), survey year.
The detailed UNEXPLAINED part is not reported for categorical blocks (not identified —
depends on the omitted category).

The UNEXPLAINED component is NOT a measure of discrimination: it also contains every
difference not observed in the data (tenure, actual experience, career breaks,
firm, effort, preferences, measurement error of banded earnings).

Uncertainty: household-cluster Poisson bootstrap (B = 200) of all components.
Outputs: reports/oaxaca_results.csv, reports/figures/stage_04/07_oaxaca_blinder_decomposition.png
Usage:   cd src && python decomposition.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy

import eda
import regression_analysis as ra
import survey_analysis as sa

B = 200
BLOCK_PATTERNS = {
    "experience": ["exp"], "education": ["education_eda"], "occupation": ["occupation_label"],
    "region & urban": ["region_eda", "urban_rural"], "hours & part-time": ["log_hours", "part_time"],
    "industry": ["industry_label"], "job type (sector, informality, contract)": ["sector_cat", "informal", "contract"],
    "survey year": ["year_cat"],
}


def block_of(col: str) -> str:
    for b, pats in BLOCK_PATTERNS.items():
        if any(p in col for p in pats):
            return b
    raise KeyError(col)


def wls(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def decompose(X: np.ndarray, y: np.ndarray, female: np.ndarray, w: np.ndarray, cols: list[str],
              reference: str = "pooled") -> dict:
    m, f = female == 0, female == 1
    bm, bf = wls(X[m], y[m], w[m]), wls(X[f], y[f], w[f])
    if reference == "pooled":
        bp = wls(np.column_stack([X, female]), y, w)[:-1]       # pooled with group dummy, dummy dropped
    elif reference == "male":
        bp = bm
    else:
        bp = bf
    xm, xf = np.average(X[m], axis=0, weights=w[m]), np.average(X[f], axis=0, weights=w[f])
    gap = np.average(y[m], weights=w[m]) - np.average(y[f], weights=w[f])
    contrib = (xm - xf) * bp
    expl = contrib.sum()
    unexpl = xm @ (bm - bp) + xf @ (bp - bf)
    out = {"gap": gap, "explained": expl, "unexplained": unexpl}
    for c, v in zip(cols, contrib):
        if c == "Intercept":
            continue
        k = block_of(c)
        out[f"explained: {k}"] = out.get(f"explained: {k}", 0.0) + v
    return out


def run():
    rows, boot_store = [], {}
    for p in sa.PERIODS:
        d = ra.estimation_sample(sa.load_period(p))
        rhs = ra.full_rhs(d)
        X = patsy.dmatrix(rhs, d, return_type="dataframe")
        cols = list(X.columns)
        Xn, y, fem, w = X.to_numpy(), d.log_y.to_numpy(), d.female.to_numpy(), d.w.to_numpy()
        est = {ref: decompose(Xn, y, fem, w, cols, ref) for ref in ("pooled", "male", "female")}
        codes, counts = sa.poisson_bootstrap_counts(d.cluster.to_numpy(), B)   # not "C": patsy uses C()
        reps = pd.DataFrame([decompose(Xn, y, fem, w * counts[b, codes], cols) for b in range(B)])
        boot_store[p] = reps
        for k, v in est["pooled"].items():
            lo, hi = np.percentile(reps[k], [2.5, 97.5])
            rows.append(dict(period=p, component=k, reference="pooled (Neumark)", log_points=v,
                             ci_low=lo, ci_high=hi, se=reps[k].std(),
                             share_of_gap_pct=100 * v / est["pooled"]["gap"],
                             pct_equivalent=(np.exp(v) - 1) * 100 if k in ("gap", "explained", "unexplained") else np.nan))
        for ref in ("male", "female"):
            for k in ("explained", "unexplained"):
                rows.append(dict(period=p, component=k, reference=f"{ref} coefficients (sensitivity)",
                                 log_points=est[ref][k], share_of_gap_pct=100 * est[ref][k] / est[ref]["gap"]))
        rows.append(dict(period=p, component="n", reference="", log_points=len(d)))
    r = pd.DataFrame(rows)
    r.to_csv(sa.REPORTS / "oaxaca_results.csv", index=False)
    plot(r)
    return r


def plot(r: pd.DataFrame):
    sa.FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharex=True, gridspec_kw={"wspace": 0.75})
    for ax, p in zip(axes, sa.PERIODS):
        q = r[(r.period == p) & (r.reference == "pooled (Neumark)")].set_index("component")
        det = q[q.index.str.startswith("explained: ")].sort_values("log_points")
        labels = ["TOTAL GAP", "Unexplained", "Explained (total)"] + [i.replace("explained: ", "  ") for i in det.index]
        vals = [q.loc["gap"], q.loc["unexplained"], q.loc["explained"]] + [det.loc[i] for i in det.index]
        colors = ["#333333", "#C44E52", "#4C72B0"] + ["#8DB0D6" if v.log_points > 0 else "#E5A0A2" for v in vals[3:]]
        yy = np.arange(len(vals))[::-1]
        for yv, v, c in zip(yy, vals, colors):
            ax.barh(yv, v.log_points * 100, color=c)
            ax.errorbar(v.log_points * 100, yv, xerr=[[100 * (v.log_points - v.ci_low)], [100 * (v.ci_high - v.log_points)]],
                        color="black", capsize=3, lw=1)
            ax.text(max(v.ci_high * 100, 0) + 0.6, yv, f"{100 * v.log_points:.1f} ({v.share_of_gap_pct:.0f}%)",
                    va="center", fontsize=8)
        ax.set_yticks(yy, labels)
        ax.set_xlim(min(-8, ax.get_xlim()[0]), q.loc["gap"].ci_high * 100 + 14)   # room for value labels
        ax.axvline(0, color="black", lw=.8)
        ax.set(title=f"Oaxaca-Blinder decomposition, {p.replace('_', '-')}",
               xlabel="Log points x 100 (approx. % gap); (share of total gap)")
    eda.caption(fig, "Source: Geostat LFS microdata; employees; log real net monthly earnings; weighted; pooled reference "
                "coefficients (Neumark); 95% CI from household-cluster bootstrap (B=200). Negative explained part = "
                "women have characteristics associated with HIGHER pay (e.g. education). Unexplained is not proof "
                "of discrimination.")
    fig.savefig(sa.FIG / "07_oaxaca_blinder_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(run().round(4).to_string(index=False))
