"""
Stage 04 — Regression analysis of individual earnings (not causal).

Dependent variable: real (2025 GEL) net monthly earnings of employees, LFS.
Models (weighted, household-clustered robust SE; same estimation sample within a period):
  M1 simple            log y ~ female
  M2 + experience      + potential experience, experience^2      (no tenure in the data)
  M3 + education       + education level
  M4 + occupation      + ISCO major group
  M5 + region          + region, urban/rural
  M6 + hours           + log usual weekly hours, part-time
  M7 full              + industry, ownership sector, informality, written contract, survey year
  L1 / L7              same as M1 / M7 with earnings in GEL (simple / multiple linear regression)
  H7                   log HOURLY earnings, full controls without hours (usual hours >= 10)
  IR7                  INTERVAL regression of log earnings on the band bounds (no midpoint
                       assumption; top band open) — robustness check of M7

Interpretation in log models: women earn  (exp(b_female) - 1) * 100 %  relative to men with
the same observed characteristics.

Diagnostics: R^2, adjusted R^2 (weighted), generalized VIF (Fox & Monette) per term,
Breusch-Pagan heteroskedasticity test (-> robust SE).

Outputs: reports/regression_results.csv, reports/regression_diagnostics.csv,
         reports/vif_results.csv, reports/figures/stage_04/05_regression_coefficient_plot.png
Usage:   cd src && python regression_analysis.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.stats.diagnostic import het_breuschpagan

import eda
import survey_analysis as sa

BLOCKS = [
    ("M1 gender only", ""),
    ("M2 + experience", "exp + exp2"),
    ("M3 + education", "C(education_eda)"),
    ("M4 + occupation", "C(occupation_label)"),
    ("M5 + region", "C(region_eda) + C(urban_rural)"),
    ("M6 + hours", "log_hours + part_time"),
    ("M7 full", "C(industry_label) + C(sector_cat) + INFORMAL + C(contract) + C(year_cat)"),
]


def blocks_for(d: pd.DataFrame) -> list[tuple[str, str]]:
    """Informality terms that are identifiable in this period (no constant / collinear columns)."""
    inf = "informal_flag" + (" + informal_unknown" if d.informal_unknown.nunique() > 1 else "")
    return [(n, b.replace("INFORMAL", inf)) for n, b in BLOCKS]


def full_rhs(d: pd.DataFrame) -> str:
    return " + ".join(b for _, b in blocks_for(d) if b)


def estimation_sample(d: pd.DataFrame) -> pd.DataFrame:
    """Common sample: rows with every regressor observed (no imputation)."""
    need = ["log_y", "exp", "education_eda", "occupation_label", "region_eda", "urban_rural", "log_hours",
            "part_time", "industry_label"]
    s = d.dropna(subset=need)
    s = s[np.isfinite(s.log_hours)]
    return s.reset_index(drop=True)


def pct(b):
    return (np.exp(b) - 1) * 100


def tidy(r, model, period, log_outcome=True) -> pd.DataFrame:
    ci = r.conf_int()
    t = pd.DataFrame({"period": period, "model": model, "term": r.params.index, "coef": r.params.values,
                      "se": r.bse.values, "t": r.tvalues.values, "p_value": r.pvalues.values,
                      "ci_low": ci[0].values, "ci_high": ci[1].values})
    if log_outcome:
        t["pct_effect"], t["pct_ci_low"], t["pct_ci_high"] = pct(t.coef), pct(t.ci_low), pct(t.ci_high)
    t["n"], t["r2"], t["adj_r2"] = int(r.nobs), r.rsquared, r.rsquared_adj
    t["cov_type"] = "cluster (household), weighted"
    return t


# ------------------------------------------------------- interval regression ---
def interval_regression(X: np.ndarray, lo: np.ndarray, hi: np.ndarray, w: np.ndarray, groups: np.ndarray,
                        start: np.ndarray):
    """Weighted Gaussian interval regression of log earnings with cluster-robust sandwich covariance.
    P_i = Phi((log hi - x b)/s) - Phi((log lo - x b)/s); lo = 0 -> -inf, hi = inf -> +inf."""
    a = np.where(lo > 0, np.log(np.maximum(lo, 1e-12)), -np.inf)
    c = np.where(np.isfinite(hi), np.log(np.where(np.isfinite(hi), hi, 1)), np.inf)
    w = w / w.mean()

    def parts(theta):
        b, s = theta[:-1], np.exp(theta[-1])
        xb = X @ b
        zl, zu = (a - xb) / s, (c - xb) / s
        P = np.clip(stats.norm.cdf(zu) - stats.norm.cdf(zl), 1e-300, None)
        pl, pu = stats.norm.pdf(zl), stats.norm.pdf(zu)
        zpl = np.where(np.isfinite(zl), np.nan_to_num(zl) * pl, 0.0)     # z*phi(z) -> 0 at +-inf
        zpu = np.where(np.isfinite(zu), np.nan_to_num(zu) * pu, 0.0)
        g_b = (-(pu - pl) / (s * P))[:, None] * X
        g_s = -(zpu - zpl) / P
        return np.log(P), np.column_stack([g_b, g_s])

    def nll(theta):
        ll, g = parts(theta)
        return -np.sum(w * ll), -(w[:, None] * g).sum(axis=0)

    n = len(w)
    # optimise the MEAN log-likelihood with BFGS and a tight gradient tolerance (L-BFGS-B stopped early
    # with ~60 parameters in a check: |grad| 3e-3); then covariance from the SUM log-likelihood
    opt = optimize.minimize(lambda t: tuple(v / n for v in nll(t)), start, jac=True, method="BFGS",
                            options={"gtol": 1e-7, "maxiter": 5000})
    th = opt.x
    # Hessian by finite differences of the analytic gradient
    k, eps = len(th), 1e-5
    H = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = eps
        H[:, j] = (nll(th + e)[1] - nll(th - e)[1]) / (2 * eps)
    H = (H + H.T) / 2
    _, g = parts(th)
    sg = pd.DataFrame(w[:, None] * g).groupby(groups).sum().to_numpy()
    Hinv = np.linalg.inv(H)
    V = Hinv @ (sg.T @ sg) @ Hinv
    return th, V, opt.success


# ------------------------------------------------------------------ diagnostics ---
def gvif(d: pd.DataFrame, rhs: str) -> pd.DataFrame:
    """Generalized VIF per term (Fox & Monette 1992); GVIF^(1/(2df)) comparable to sqrt(VIF)."""
    X = patsy.dmatrix(rhs, d, return_type="dataframe")
    info = X.design_info
    X = X.drop(columns="Intercept")
    R = np.corrcoef(X.to_numpy(), rowvar=False)
    detR = np.linalg.slogdet(R)[1]
    cols = list(X.columns)
    out = []
    for term, sl in info.term_name_slices.items():
        if term == "Intercept":
            continue
        idx = [cols.index(c) for c in info.column_names[sl]]
        rest = [i for i in range(len(cols)) if i not in idx]
        g = np.exp(np.linalg.slogdet(R[np.ix_(idx, idx)])[1] + np.linalg.slogdet(R[np.ix_(rest, rest)])[1] - detR)
        out.append(dict(term=term, df=len(idx), gvif=g, gvif_adj=g ** (1 / (2 * len(idx)))))
    return pd.DataFrame(out)


def run():
    coefs, diags, vifs, ladder = [], [], [], []
    for p in sa.PERIODS:
        d = estimation_sample(sa.load_period(p))
        rhs = "female"
        FULL_RHS = full_rhs(d)
        for name, block in blocks_for(d):
            rhs = rhs + (" + " + block if block else "")
            r = sa.wls_cluster(f"log_y ~ {rhs}", d)
            t = tidy(r, name, p)
            coefs.append(t)
            ladder.append(t[t.term == "female"])
            diags.append(dict(period=p, model=name, outcome="log real earnings", n=int(r.nobs), r2=r.rsquared,
                              adj_r2=r.rsquared_adj, female_coef=r.params["female"],
                              female_pct=pct(r.params["female"]), female_p=r.pvalues["female"]))
        # level models (GEL)
        for name, rr in [("L1 simple linear (GEL)", "female"), ("L7 multiple linear (GEL)", f"female + {FULL_RHS}")]:
            r = sa.wls_cluster(f"y ~ {rr}", d)
            t = tidy(r, name, p, log_outcome=False)
            coefs.append(t)
            diags.append(dict(period=p, model=name, outcome="real earnings GEL", n=int(r.nobs), r2=r.rsquared,
                              adj_r2=r.rsquared_adj, female_coef=r.params["female"],
                              female_pct=100 * r.params["female"] / r.params["Intercept"] if name.startswith("L1")
                              else np.nan, female_p=r.pvalues["female"]))
        # hourly model
        rhs_h = "female + " + " + ".join(b for n_, b in blocks_for(d) if b and not n_.startswith("M6"))
        dh = d[d.working_hours >= 10].reset_index(drop=True)
        r = sa.wls_cluster(f"log_hourly ~ {rhs_h}", dh)
        t = tidy(r, "H7 log hourly earnings (hours>=10)", p)
        coefs.append(t)
        ladder.append(t[t.term == "female"])
        diags.append(dict(period=p, model="H7 log hourly earnings (hours>=10)", outcome="log real hourly earnings",
                          n=int(r.nobs), r2=r.rsquared, adj_r2=r.rsquared_adj, female_coef=r.params["female"],
                          female_pct=pct(r.params["female"]), female_p=r.pvalues["female"]))
        # interval regression (robustness)
        full = sa.wls_cluster(f"log_y ~ female + {FULL_RHS}", d)
        X = patsy.dmatrix(f"female + {FULL_RHS}", d, return_type="dataframe")
        start = np.append(full.params[X.columns].to_numpy(), np.log(np.sqrt(full.scale)))
        th, V, ok = interval_regression(X.to_numpy(), d.lo_real.to_numpy(), d.hi_real.to_numpy(),
                                        d.w.to_numpy(), d.cluster.to_numpy(), start)
        se = np.sqrt(np.diag(V))
        names = list(X.columns) + ["log_sigma"]
        t = pd.DataFrame({"period": p, "model": "IR7 interval regression (full)", "term": names, "coef": th, "se": se,
                          "t": th / se, "p_value": 2 * stats.norm.sf(np.abs(th / se)),
                          "ci_low": th - 1.96 * se, "ci_high": th + 1.96 * se})
        t["pct_effect"], t["pct_ci_low"], t["pct_ci_high"] = pct(t.coef), pct(t.ci_low), pct(t.ci_high)
        t["n"], t["r2"], t["adj_r2"], t["cov_type"] = len(d), np.nan, np.nan, "cluster sandwich, weighted MLE"
        coefs.append(t)
        ladder.append(t[t.term == "female"])
        diags.append(dict(period=p, model="IR7 interval regression (full)", outcome="log real earnings (interval)",
                          # by NAME: patsy puts categorical terms before numeric ones, so position 1 is not 'female'
                          n=len(d), female_coef=th[names.index("female")], female_pct=pct(th[names.index("female")]),
                          female_p=t.p_value.iloc[names.index("female")],
                          converged=ok, sigma=np.exp(th[-1])))
        # diagnostics on the full log model
        Xf = patsy.dmatrix(f"female + {FULL_RHS}", d, return_type="dataframe")
        ols = sm.OLS(d.log_y, Xf).fit()
        bp = het_breuschpagan(ols.resid, Xf)
        diags.append(dict(period=p, model="M7 full — Breusch-Pagan", outcome="log real earnings", n=len(d),
                          bp_lm=bp[0], bp_p=bp[1], note="p < 0.05 -> heteroskedasticity -> robust (cluster) SE used"))
        v = gvif(d, f"female + {FULL_RHS}")
        v.insert(0, "period", p)
        vifs.append(v)
    co = pd.concat(coefs, ignore_index=True)
    dg = pd.DataFrame(diags)
    vf = pd.concat(vifs, ignore_index=True)
    co.to_csv(sa.REPORTS / "regression_results.csv", index=False)
    dg.to_csv(sa.REPORTS / "regression_diagnostics.csv", index=False)
    vf.to_csv(sa.REPORTS / "vif_results.csv", index=False)
    plot(pd.concat(ladder), co)
    return co, dg, vf


def plot(ladder: pd.DataFrame, co: pd.DataFrame):
    sa.FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    models = list(dict.fromkeys(ladder.model))
    for p, col, off in [("2017_2019", "#999999", -0.17), ("2020_2025", "#55A868", 0.17)]:
        q = ladder[ladder.period == p].set_index("model").reindex(models)
        y = np.arange(len(models)) + off
        ax.errorbar(q.pct_effect, y, xerr=[q.pct_effect - q.pct_ci_low, q.pct_ci_high - q.pct_effect], fmt="o",
                    color=col, capsize=3, label=p.replace("_", "-"))
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=.8)
    ax.set(title="Female coefficient as % difference vs men (95% CI)", xlabel="% (exp(b) - 1)")
    ax.legend(frameon=False)
    ax = axes[1]
    full = co[(co.model == "M7 full") & (co.period == "2020_2025") & (co.term != "Intercept")
              & co.term.str.contains("female|education|occupation|log_hours|part_time")
              & ~co.term.str.contains("Higher professional")].copy()   # n=12 in 2020-25: uninformative CI
    full["label"] = full.term.str.replace(r"C\((\w+)\)\[T\.(.*)\]", r"\2", regex=True)
    full = full.sort_values("pct_effect")
    ax.errorbar(full.pct_effect, range(len(full)), xerr=[full.pct_effect - full.pct_ci_low,
                                                          full.pct_ci_high - full.pct_effect],
                fmt="o", color="#4C72B0", capsize=2)
    ax.set_yticks(range(len(full)), full.label, fontsize=8)
    ax.axvline(0, color="black", lw=.8)
    ax.set(title="Full model M7, 2020-2025: selected coefficients (% effect, 95% CI)",
           xlabel="% difference vs reference (Below secondary; Armed forces/other ISCO); log_hours: % per +1 log-point (hours x 2.7)")
    eda.caption(fig, "Source: Geostat LFS microdata; employees; log real net monthly earnings; weighted least squares "
                "with household-clustered SE. Same sample across M1-M7 within a period. Associations, not causal effects.")
    fig.savefig(sa.FIG / "05_regression_coefficient_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    co, dg, vf = run()
    pd.set_option("display.width", 250)
    print(dg.round(4).to_string(index=False))
    print(vf.round(2).to_string(index=False))
