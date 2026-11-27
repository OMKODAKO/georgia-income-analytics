"""
Stage 04 — Hypothesis tests, effect sizes, gap confidence intervals.

Every hypothesis is tested two ways and both are reported:
  * CLASSICAL (scipy): Welch t, Mann-Whitney U, one-way / Welch ANOVA, Kruskal-Wallis, chi-square.
    They assume independent observations and are UNWEIGHTED -> shown for reference only.
  * SURVEY-ADJUSTED (primary): weighted regression with household-clustered SE
    (see survey_analysis.py); Wald tests for joint hypotheses; cluster bootstrap for gap CIs.
All tests two-sided, alpha = 0.05; Benjamini-Hochberg FDR over all tests in the file.

Outcome: real (2025) net monthly earnings of employees. Main scale = log (right-skewed);
level GEL also reported.  Separate analysis per period (2017-2019, 2020-2025).

Outputs: reports/hypothesis_results.csv, reports/effect_sizes.csv,
         reports/gap_confidence_intervals.csv, reports/subgroup_gaps.csv,
         reports/figures/stage_04/*.png
Usage:   cd src && python hypothesis_testing.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

import eda
import survey_analysis as sa

ALPHA = 0.05
B_MEAN, B_MEDIAN = 1000, 200
HYP = {
    "H1": ("Mean earnings of women and men are equal", "Mean earnings of women and men differ"),
    "H2": ("No gender gap after controlling for age", "A gender gap remains after controlling for age"),
    "H3": ("No gender gap after controlling for education", "A gender gap remains after controlling for education"),
    "H4": ("No gender gap within the same occupation group", "A gender gap exists within occupation groups"),
    "H5": ("Mean earnings do not differ between regions", "Mean earnings differ between at least two regions"),
    "H6": ("Mean earnings do not differ between age groups", "Mean earnings differ between age groups"),
    "H7": ("Mean earnings do not differ between education levels", "Mean earnings differ between education levels"),
    "H8": ("Sex and labour-market category are independent", "Sex and labour-market category are associated"),
}


# ------------------------------------------------------------- effect sizes ---
def cohens_d(a, b, wa=None, wb=None):
    """(mean_b - mean_a) / pooled SD; weighted if weights given."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    wa = np.ones_like(a) if wa is None else np.asarray(wa, float)
    wb = np.ones_like(b) if wb is None else np.asarray(wb, float)
    ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
    va = np.average((a - ma) ** 2, weights=wa) * len(a) / (len(a) - 1)
    vb = np.average((b - mb) ** 2, weights=wb) * len(b) / (len(b) - 1)
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return (mb - ma) / sp


def hedges_g(d, n1, n2):
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * j
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    return g, g - 1.96 * se, g + 1.96 * se


def rank_biserial(u, n1, n2):
    """r = 2U/(n1*n2) - 1 = P(x1 > x2) - P(x1 < x2), U = Mann-Whitney U of sample 1 (here: men)."""
    return 2 * u / (n1 * n2) - 1


def eta_squared(groups):
    allv = np.concatenate(groups)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
    ss_t = ((allv - allv.mean()) ** 2).sum()
    return ss_b / ss_t


def epsilon_squared_kw(h, n):
    return h / ((n ** 2 - 1) / (n + 1))


def cramers_v(chi2, table):
    n = table.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def magnitude_d(d):
    d = abs(d)
    return "negligible" if d < .2 else "small" if d < .5 else "medium" if d < .8 else "large"


# --------------------------------------------------------------- the tests ---
def main_gender_tests(d: pd.DataFrame, p: str, res: list, eff: list):
    f, m = d[d.female == 1], d[d.female == 0]
    for scale, col in [("level GEL", "y"), ("log", "log_y")]:
        # assumptions
        lev = stats.levene(f[col], m[col], center="median")
        skew_f, skew_m = stats.skew(f[col]), stats.skew(m[col])
        t = stats.ttest_ind(m[col], f[col], equal_var=False)
        ci = t.confidence_interval(0.95)
        res.append(dict(period=p, hypothesis="H1", scale=scale, test="Welch t-test (classical, unweighted)",
                        statistic=t.statistic, df=t.df, p_value=t.pvalue,
                        estimate=m[col].mean() - f[col].mean(), ci_low=ci.low, ci_high=ci.high,
                        estimate_meaning="men minus women",
                        assumptions=f"Levene p={lev.pvalue:.2g} (variances differ -> Welch); "
                                    f"skew women {skew_f:.2f}, men {skew_m:.2f}; n large -> CLT"))
        r = sa.wls_cluster(f"{col} ~ female", d)
        lo, hi = r.conf_int().loc["female"]
        res.append(dict(period=p, hypothesis="H1", scale=scale,
                        test="Weighted regression, household-clustered SE (survey-adjusted, PRIMARY)",
                        statistic=r.tvalues["female"], df=r.df_resid, p_value=r.pvalues["female"],
                        estimate=-r.params["female"], ci_low=-hi, ci_high=-lo, estimate_meaning="men minus women",
                        assumptions="weights + clustering; no strata/PSU published"))
        dd = cohens_d(f[col], m[col])
        dw = cohens_d(f[col], m[col], f.w, m.w)
        g, glo, ghi = hedges_g(dd, len(f), len(m))
        eff.append(dict(period=p, comparison="men vs women", scale=scale, measure="Cohen's d (unweighted)",
                        value=dd, ci_low=np.nan, ci_high=np.nan, magnitude=magnitude_d(dd)))
        eff.append(dict(period=p, comparison="men vs women", scale=scale, measure="Cohen's d (weighted)",
                        value=dw, ci_low=np.nan, ci_high=np.nan, magnitude=magnitude_d(dw)))
        eff.append(dict(period=p, comparison="men vs women", scale=scale, measure="Hedges' g (unweighted)",
                        value=g, ci_low=glo, ci_high=ghi, magnitude=magnitude_d(g)))
    u = stats.mannwhitneyu(m.y, f.y, alternative="two-sided")
    rb = rank_biserial(u.statistic, len(m), len(f))
    res.append(dict(period=p, hypothesis="H1", scale="rank", test="Mann-Whitney U (classical, unweighted)",
                    statistic=u.statistic, df=np.nan, p_value=u.pvalue, estimate=rb, ci_low=np.nan, ci_high=np.nan,
                    estimate_meaning="rank-biserial r = P(man > woman) - P(woman > man)",
                    assumptions="many ties (banded earnings); tie-corrected normal approximation"))
    eff.append(dict(period=p, comparison="men vs women", scale="rank", measure="rank-biserial r",
                    value=rb, ci_low=np.nan, ci_high=np.nan,
                    magnitude="small" if abs(rb) < .3 else "medium" if abs(rb) < .5 else "large"))


def adjusted_gender_tests(d, p, res):
    for h, ctrl, label in [("H2", "C(age_group)", "age group"), ("H3", "C(education_eda)", "education"),
                           ("H4", "C(occupation_label)", "occupation (ISCO major group)")]:
        r = sa.wls_cluster(f"log_y ~ female + {ctrl}", d)
        lo, hi = r.conf_int().loc["female"]
        b = r.params["female"]
        res.append(dict(period=p, hypothesis=h, scale="log",
                        test=f"Weighted regression log earnings ~ female + {label} (clustered SE)",
                        statistic=r.tvalues["female"], df=r.df_resid, p_value=r.pvalues["female"],
                        estimate=(1 - np.exp(b)) * 100, ci_low=(1 - np.exp(hi)) * 100, ci_high=(1 - np.exp(lo)) * 100,
                        estimate_meaning="% women earn less than men with the same " + label + " (1 - exp(b))",
                        assumptions=f"n={int(r.nobs)}"))


def group_tests(d, p, res, eff):
    for h, col, label in [("H5", "region_eda", "region"), ("H6", "age_group", "age group"),
                          ("H7", "education_eda", "education")]:
        g = d[d[col].notna()]
        if col == "age_group":
            g = g[g.age_group != "15-17"]          # tiny group, reported separately in Stage 03
        if col == "education_eda":
            g = g[g.education_eda != "Higher professional"]   # n < 30 in both periods
        groups = [x.log_y.to_numpy() for _, x in g.groupby(col)]
        lev = stats.levene(*groups, center="median")
        a = stats.f_oneway(*groups)
        wa = anova_oneway(g.log_y, g[col], use_var="unequal")
        kw = stats.kruskal(*groups)
        e2, eps2 = eta_squared(groups), epsilon_squared_kw(kw.statistic, len(g))
        res += [
            dict(period=p, hypothesis=h, scale="log", test=f"One-way ANOVA by {label} (classical)",
                 statistic=a.statistic, df=len(groups) - 1, p_value=a.pvalue, estimate=e2,
                 estimate_meaning="eta-squared", assumptions=f"Levene p={lev.pvalue:.2g} -> unequal variances"),
            dict(period=p, hypothesis=h, scale="log", test=f"Welch ANOVA by {label} (classical)",
                 statistic=wa.statistic, df=wa.df[0], p_value=wa.pvalue, estimate=e2, estimate_meaning="eta-squared",
                 assumptions="robust to unequal variances"),
            dict(period=p, hypothesis=h, scale="log", test=f"Kruskal-Wallis by {label} (classical)",
                 statistic=kw.statistic, df=len(groups) - 1, p_value=kw.pvalue, estimate=eps2,
                 estimate_meaning="epsilon-squared", assumptions="rank-based; ties from bands"),
        ]
        r = sa.wls_cluster(f"log_y ~ C({col})", g)
        names = [n for n in r.params.index if n.startswith("C(")]
        w = r.wald_test(" = 0, ".join(names) + " = 0", scalar=True)
        res.append(dict(period=p, hypothesis=h, scale="log",
                        test=f"Wald test of {label} dummies, weighted, clustered SE (PRIMARY)",
                        statistic=float(w.statistic), df=len(names), p_value=float(w.pvalue),
                        estimate=r.rsquared, estimate_meaning="weighted R-squared of the group factor",
                        assumptions="survey-adjusted approximation"))
        eff.append(dict(period=p, comparison=f"between {label} groups", scale="log", measure="eta-squared",
                        value=e2, magnitude="small" if e2 < .06 else "medium" if e2 < .14 else "large"))
        eff.append(dict(period=p, comparison=f"between {label} groups", scale="log", measure="epsilon-squared (KW)",
                        value=eps2, magnitude="small" if eps2 < .06 else "medium" if eps2 < .14 else "large"))


def design_wald_distribution(cat: pd.Series, female: pd.Series, w: pd.Series, cluster: pd.Series):
    """Survey-adjusted test that the WEIGHTED distribution of `cat` is the same for women and men.
    delta = p_men - p_women (K-1 categories); covariance by linearisation: influence values
    summed within households, V = C/(C-1) * sum_c z_c z_c' ; Wald = delta' V^-1 delta ~ chi2(K-1)."""
    D = pd.get_dummies(cat).to_numpy(float)[:, :-1]
    w, fem = w.to_numpy(float), female.to_numpy(int)
    z = np.zeros_like(D)
    delta = np.zeros(D.shape[1])
    for g, sign in ((0, 1.0), (1, -1.0)):
        m = fem == g
        W = w[m].sum()
        pg = (w[m, None] * D[m]).sum(0) / W
        delta += sign * pg
        z[m] = sign * w[m, None] * (D[m] - pg) / W
    zc = pd.DataFrame(z).groupby(cluster.to_numpy()).sum().to_numpy()
    C = len(zc)
    V = C / (C - 1) * zc.T @ zc
    stat = float(delta @ np.linalg.pinv(V) @ delta)
    return stat, D.shape[1], float(stats.chi2.sf(stat, D.shape[1]))


def chi_square_tests(p, res, eff):
    """Sex x labour-market categories, all persons 15+ (individuals table) and employees."""
    import sqlite3
    con = sqlite3.connect(sa.DB)
    ind = pd.read_sql(f"SELECT gender, labour_status, employment_status, is_employee, occupation, part_time, "
                      f"weight_annual, household_key FROM individuals WHERE period='{p}'", con)
    emp = ind[ind.labour_status == "employed"]
    ee = ind[ind.is_employee == 1]
    for name, data, col in [("labour status (15+)", ind, "labour_status"),
                            ("status in employment (employed)", emp, "employment_status"),
                            ("occupation group (employees)", ee, "occupation"),
                            ("part-time (employees)", ee, "part_time")]:
        t = pd.crosstab(data.gender, data[col])
        t = t.loc[:, t.min() >= 5]                       # expected-count rule (sparse columns dropped)
        chi2, pv, dof, exp = stats.chi2_contingency(t)
        v = cramers_v(chi2, t)
        wsh = data.groupby(["gender", col]).weight_annual.sum().unstack(fill_value=0)
        wsh = (wsh.T / wsh.sum(axis=1)).T.round(3)
        res.append(dict(period=p, hypothesis="H8", scale="categorical",
                        test=f"Chi-square: sex x {name} (classical, unweighted)", statistic=chi2, df=dof,
                        p_value=pv, estimate=v, estimate_meaning="Cramer's V",
                        assumptions=f"min expected count {exp.min():.0f}; weighted female shares: "
                                    + "; ".join(f"{c}={wsh.loc['female', c]:.2f}/{wsh.loc['male', c]:.2f}"
                                                for c in t.columns)[:300]))
        dd = data[data[col].isin(t.columns)]
        st, df_, pw = design_wald_distribution(dd[col].astype(str), (dd.gender == "female").astype(int),
                                               dd.weight_annual, dd.household_key)
        res.append(dict(period=p, hypothesis="H8", scale="categorical",
                        test=f"Design-based Wald: sex x {name} (weighted, household-clustered, PRIMARY)",
                        statistic=st, df=df_, p_value=pw, estimate=v, estimate_meaning="Cramer's V (unweighted)",
                        assumptions="linearisation variance; no strata/PSU published"))
        eff.append(dict(period=p, comparison=f"sex x {name}", scale="categorical", measure="Cramer's V", value=v,
                        magnitude="negligible" if v < .1 else "small" if v < .3 else "medium" if v < .5 else "large"))


# ----------------------------------------------------------- gap intervals ---
def _band_inputs(d):
    return d.lo_real.to_numpy(), d.hi_real.to_numpy(), d.pareto_alpha.to_numpy()


def median_gap(d, wcol_values=None, hourly=False, iters=45):
    out = {}
    for sex in (0, 1):
        s = d[d.female == sex]
        scale = 1 / (s.working_hours * eda.WEEKS_PER_MONTH) if hourly else 1
        w = s.w.to_numpy() if wcol_values is None else wcol_values[s.index]
        out[sex] = eda.interval_quantile(s.lo_real * scale, s.hi_real * scale, s.pareto_alpha, w, .5, iters)
    return (out[0] - out[1]) / out[0] * 100


def gap_intervals(d, p) -> list[dict]:
    rows = []
    boot = sa.bootstrap_mean_gap(d, "y", B_MEAN)
    est = eda.gap_pct(eda.wmean(d[d.female == 0].y, d[d.female == 0].w), eda.wmean(d[d.female == 1].y, d[d.female == 1].w))
    rows.append(dict(period=p, measure="mean gap %", estimate=est, ci_low=np.percentile(boot, 2.5),
                     ci_high=np.percentile(boot, 97.5), se=boot.std(), method=f"cluster Poisson bootstrap B={B_MEAN}"))
    # delta-method check via regression on level: gap = -b / (a) ... reported for validation
    r = sa.wls_cluster("y ~ female", d)
    a, b = r.params["Intercept"], r.params["female"]
    V = r.cov_params().loc[["Intercept", "female"], ["Intercept", "female"]].to_numpy()
    grad = np.array([b / a ** 2, -1 / a]) * 100
    se = float(np.sqrt(grad @ V @ grad))
    rows.append(dict(period=p, measure="mean gap % (delta method check)", estimate=-b / a * 100,
                     ci_low=-b / a * 100 - 1.96 * se, ci_high=-b / a * 100 + 1.96 * se, se=se,
                     method="delta method, weighted regression, clustered SE"))
    codes, C = sa.poisson_bootstrap_counts(d.cluster.to_numpy(), B_MEDIAN)
    for label, hourly, sub in [("median gap %", False, d), ("hourly median gap %", True, d[d.working_hours > 0])]:
        sub = sub.reset_index(drop=True)
        cs, Cs = sa.poisson_bootstrap_counts(sub.cluster.to_numpy(), B_MEDIAN)
        est = median_gap(sub, hourly=hourly)
        reps = np.array([median_gap(sub, sub.w.to_numpy() * Cs[b, cs], hourly=hourly) for b in range(B_MEDIAN)])
        rows.append(dict(period=p, measure=label, estimate=est, ci_low=np.percentile(reps, 2.5),
                         ci_high=np.percentile(reps, 97.5), se=reps.std(),
                         method=f"cluster Poisson bootstrap B={B_MEDIAN}; quantiles interpolated within bands"))
    for y in eda.PERIODS[p]:
        s = d[d.survey_year == y].reset_index(drop=True)
        bt = sa.bootstrap_mean_gap(s, "y", B_MEAN)
        e = eda.gap_pct(eda.wmean(s[s.female == 0].y, s[s.female == 0].w), eda.wmean(s[s.female == 1].y, s[s.female == 1].w))
        rows.append(dict(period=p, measure=f"mean gap % {y}", estimate=e, ci_low=np.percentile(bt, 2.5),
                         ci_high=np.percentile(bt, 97.5), se=bt.std(), method=f"cluster Poisson bootstrap B={B_MEAN}"))
    return rows


def subgroup_gaps(d, p) -> list[dict]:
    """Mean gap % with bootstrap CI and survey-adjusted test of the female coefficient, per subgroup."""
    rows = []
    for col, label in [("age_group", "age"), ("education_eda", "education"), ("region_eda", "region"),
                       ("occupation_label", "occupation")]:
        for key, g in d.groupby(col):
            nf, nm = (g.female == 1).sum(), (g.female == 0).sum()
            rel = eda.reliability(nf, nm)
            if rel == "suppressed":
                rows.append(dict(period=p, dimension=label, group=key, n_female=nf, n_male=nm, reliability=rel))
                continue
            g = g.reset_index(drop=True)
            bt = sa.bootstrap_mean_gap(g, "y", 500)
            est = eda.gap_pct(eda.wmean(g[g.female == 0].y, g[g.female == 0].w),
                              eda.wmean(g[g.female == 1].y, g[g.female == 1].w))
            r = sa.wls_cluster("log_y ~ female", g)
            rows.append(dict(period=p, dimension=label, group=key, n_female=nf, n_male=nm, reliability=rel,
                             gap_mean_pct=est, ci_low=np.percentile(bt, 2.5), ci_high=np.percentile(bt, 97.5),
                             log_coef_female=r.params["female"], p_value=r.pvalues["female"]))
    return rows


# ------------------------------------------------------------------- plots ---
def plots(ci: pd.DataFrame, sub: pd.DataFrame, eff: pd.DataFrame, official: pd.DataFrame):
    sa.FIG.mkdir(parents=True, exist_ok=True)
    src = "Source: Geostat LFS microdata; employees, net monthly earnings (real 2025 GEL); weighted; 95% CI from " \
          "household-cluster bootstrap."
    # 1 CI comparison
    fig, ax = plt.subplots(figsize=(10, 4.8))
    main = ci[ci.measure.isin(["mean gap %", "median gap %", "hourly median gap %"])]
    ypos, labels = [], []
    for i, (_, r) in enumerate(main.iterrows()):
        y = i
        c = {"2017_2019": "#999999", "2020_2025": "#55A868"}[r.period]
        ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="o", color=c,
                    capsize=4)
        ax.text(r.ci_high + 0.4, y, f"{r.estimate:.1f}% [{r.ci_low:.1f}; {r.ci_high:.1f}]", va="center", fontsize=8)
        labels.append(f"{r.measure} ({r.period.replace('_', '-')})")
        ypos.append(y)
    off = official.dropna(subset=["official_adjusted_gpg_monthly_pct"])
    ax.axvspan(off.official_adjusted_gpg_monthly_pct.min(), off.official_adjusted_gpg_monthly_pct.max(),
               color="#DDDDDD", alpha=.6, label="Geostat adjusted GPG (monthly) range 2020-25")
    ax.set_yticks(ypos, labels)
    ax.set(title="Unadjusted gender gap with 95% confidence intervals", xlabel="Gap, % of men's earnings",
           xlim=(0, max(main.ci_high) + 12))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    eda.caption(fig, src + " Gap = (men - women) / men.", y=-0.12)
    fig.savefig(sa.FIG / "01_confidence_interval_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 2-4 by dimension
    for i, (dim, title) in enumerate([("age", "age group"), ("education", "education"), ("region", "region")], 2):
        s = sub[(sub.dimension == dim) & sub.gap_mean_pct.notna()]
        fig, ax = plt.subplots(figsize=(10, 0.45 * s.group.nunique() + 2.5))
        groups = list(dict.fromkeys(s.sort_values("gap_mean_pct").group))
        for k, (p, col, off_) in enumerate([("2017_2019", "#999999", -0.18), ("2020_2025", "#55A868", 0.18)]):
            q = s[s.period == p].set_index("group").reindex(groups)
            yv = np.arange(len(groups)) + off_
            ax.errorbar(q.gap_mean_pct, yv, xerr=[q.gap_mean_pct - q.ci_low, q.ci_high - q.gap_mean_pct],
                        fmt="o", color=col, capsize=3, label=p.replace("_", "-"))
        ax.set_yticks(range(len(groups)), [g + (" *" if (s[s.group == g].reliability == "low_n").any() else "")
                                           for g in groups])
        ax.axvline(0, color="black", lw=.8)
        ax.set(title=f"Gender pay gap (mean) by {title}, with 95% CI", xlabel="Gap, % of men's earnings")
        ax.legend(frameon=False)
        eda.caption(fig, src + " * = fewer than 100 observations for one sex. Groups with < 30 omitted.")
        fig.savefig(sa.FIG / f"0{i}_gender_pay_gap_by_{dim}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 6 effect sizes
    e = eff[eff.measure.isin(["Hedges' g (unweighted)", "Cohen's d (weighted)", "eta-squared", "Cramer's V",
                              "rank-biserial r"])].copy()
    e["label"] = e.comparison + " | " + e.measure + " | " + e.scale
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(e) + 2))
    e = e.sort_values(["measure", "value"])
    ax.barh(e.label + " (" + e.period.str.replace("_", "-") + ")", e.value,
            color=e.period.map({"2017_2019": "#999999", "2020_2025": "#55A868"}))
    ax.set(title="Effect sizes (practical significance)", xlabel="Effect size (d/g: SD units; others 0-1)")
    eda.caption(fig, "Benchmarks: d/g 0.2 small, 0.5 medium, 0.8 large; eta^2 0.01/0.06/0.14; Cramer's V 0.1/0.3/0.5. "
                "Grey = 2017-2019, green = 2020-2025.")
    fig.savefig(sa.FIG / "06_effect_size_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    res, eff, cis, subs = [], [], [], []
    for p in sa.PERIODS:
        d = sa.load_period(p)
        main_gender_tests(d, p, res, eff)
        adjusted_gender_tests(d, p, res)
        group_tests(d, p, res, eff)
        chi_square_tests(p, res, eff)
        cis += gap_intervals(d, p)
        subs += subgroup_gaps(d, p)
    r = pd.DataFrame(res)
    r.insert(2, "H0", r.hypothesis.map(lambda h: HYP[h][0]))
    r.insert(3, "H1_alternative", r.hypothesis.map(lambda h: HYP[h][1]))
    r["alpha"], r["two_sided"] = ALPHA, True
    sub = pd.DataFrame(subs)
    # Benjamini-Hochberg over all tests (main + subgroup)
    allp = pd.concat([r.p_value, sub.p_value.dropna()])
    rej, padj, _, _ = multipletests(allp, alpha=ALPHA, method="fdr_bh")
    r["p_value_bh"], r["reject_h0_bh"] = padj[:len(r)], rej[:len(r)]
    sub.loc[sub.p_value.notna(), "p_value_bh"] = padj[len(r):]
    sub.loc[sub.p_value.notna(), "reject_h0_bh"] = rej[len(r):]
    eff = pd.DataFrame(eff)
    ci = pd.DataFrame(cis)
    r.to_csv(sa.REPORTS / "hypothesis_results.csv", index=False)
    eff.to_csv(sa.REPORTS / "effect_sizes.csv", index=False)
    ci.to_csv(sa.REPORTS / "gap_confidence_intervals.csv", index=False)
    sub.to_csv(sa.REPORTS / "subgroup_gaps.csv", index=False)
    official = pd.read_csv(sa.REPORTS / "gender_income_comparison.csv")
    plots(ci, sub, eff, official)
    return r, eff, ci, sub


if __name__ == "__main__":
    r, e, c, s = run()
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(r[["period", "hypothesis", "scale", "test", "statistic", "p_value", "p_value_bh", "estimate",
             "ci_low", "ci_high"]].to_string(index=False))
    print(c.to_string(index=False))
    print(e.to_string(index=False))
