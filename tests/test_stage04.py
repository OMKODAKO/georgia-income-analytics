"""Stage 04 validation: hypotheses, tests, p-values, CIs, effect sizes, regression, weights, BH.
Results in reports/ are compared with INDEPENDENT recomputations (numpy / scipy / statsmodels)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import decomposition as dcp            # noqa: E402
import hypothesis_testing as ht        # noqa: E402
import regression_analysis as ra       # noqa: E402
import survey_analysis as sa           # noqa: E402

REP = ROOT / "reports"
HYP = pd.read_csv(REP / "hypothesis_results.csv")
EFF = pd.read_csv(REP / "effect_sizes.csv")
CI = pd.read_csv(REP / "gap_confidence_intervals.csv")
REG = pd.read_csv(REP / "regression_results.csv")
OAX = pd.read_csv(REP / "oaxaca_results.csv")
SUB = pd.read_csv(REP / "subgroup_gaps.csv")


@pytest.fixture(scope="module")
def d2():
    return sa.load_period("2020_2025")


# 1. hypotheses ---------------------------------------------------------------------------
def test_hypotheses_formulated():
    for p in sa.PERIODS:
        assert set(HYP[HYP.period == p].hypothesis) == {f"H{i}" for i in range(1, 9)}
    assert HYP.H0.notna().all() and HYP.H1_alternative.notna().all()
    assert (HYP.H0 != HYP.H1_alternative).all()
    assert HYP.two_sided.all() and (HYP.alpha == 0.05).all()
    assert HYP.groupby(["period", "hypothesis"]).test.apply(lambda s: s.str.contains("PRIMARY|clustered").any()).all()


# 2. statistical tests work --------------------------------------------------------------
def test_welch_detects_known_difference_and_not_null():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 4000), rng.normal(0.3, 2, 4000)
    assert stats.ttest_ind(b, a, equal_var=False).pvalue < 1e-6
    same = [stats.ttest_ind(rng.normal(0, 1, 500), rng.normal(0, 1, 500), equal_var=False).pvalue for _ in range(400)]
    assert 0.02 < np.mean(np.array(same) < 0.05) < 0.09          # type-I error close to alpha


def test_welch_result_recomputed(d2):
    row = HYP[(HYP.period == "2020_2025") & (HYP.test.str.startswith("Welch t")) & (HYP.scale == "log")].iloc[0]
    t = stats.ttest_ind(d2[d2.female == 0].log_y, d2[d2.female == 1].log_y, equal_var=False)
    assert row.statistic == pytest.approx(t.statistic, rel=1e-9)
    assert row.estimate == pytest.approx(d2[d2.female == 0].log_y.mean() - d2[d2.female == 1].log_y.mean())


def test_mann_whitney_and_rank_biserial(d2):
    row = HYP[(HYP.period == "2020_2025") & HYP.test.str.startswith("Mann")].iloc[0]
    u = stats.mannwhitneyu(d2[d2.female == 0].y, d2[d2.female == 1].y)
    n1, n2 = (d2.female == 0).sum(), (d2.female == 1).sum()
    assert row.estimate == pytest.approx(2 * u.statistic / (n1 * n2) - 1)
    assert 0 < row.estimate < 1                                     # men rank higher


# 3. p-values ----------------------------------------------------------------------------
def test_clustered_p_value_matches_normal_reference(d2):
    r = sa.wls_cluster("log_y ~ female", d2)
    assert r.pvalues["female"] == pytest.approx(2 * stats.norm.sf(abs(r.tvalues["female"])), rel=1e-6, abs=1e-300)
    row = HYP[(HYP.period == "2020_2025") & (HYP.hypothesis == "H1") & HYP.test.str.contains("PRIMARY")
              & (HYP.scale == "log")].iloc[0]
    assert row.statistic == pytest.approx(r.tvalues["female"])


def test_p_values_valid():
    assert HYP.p_value.between(0, 1).all() and HYP.p_value_bh.between(0, 1).all()


# 4. confidence intervals ----------------------------------------------------------------
def test_ci_contain_estimate_and_bootstrap_agrees_with_delta():
    assert ((CI.ci_low <= CI.estimate) & (CI.estimate <= CI.ci_high)).all()
    for p in sa.PERIODS:
        b = CI[(CI.period == p) & (CI.measure == "mean gap %")].iloc[0]
        dm = CI[(CI.period == p) & (CI.measure == "mean gap % (delta method check)")].iloc[0]
        assert b.estimate == pytest.approx(dm.estimate, abs=1e-6)
        assert abs(b.ci_low - dm.ci_low) < 0.5 and abs(b.ci_high - dm.ci_high) < 0.5


def test_regression_ci_formula():
    r = REG[REG.model == "M7 full"]
    assert np.allclose(r.ci_low, r.coef - stats.norm.ppf(.975) * r.se, atol=1e-6)
    assert np.allclose(r.pct_effect, (np.exp(r.coef) - 1) * 100)


# 5. effect sizes ------------------------------------------------------------------------
def test_effect_size_formulas():
    a, b = np.array([1., 2, 3, 4]), np.array([3., 4, 5, 6])
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    assert ht.cohens_d(a, b) == pytest.approx((b.mean() - a.mean()) / sp)
    g, lo, hi = ht.hedges_g(1.0, 4, 4)
    assert g == pytest.approx(1.0 * (1 - 3 / (4 * 8 - 9))) and lo < g < hi
    groups = [np.array([1., 2, 3]), np.array([4., 5, 6])]
    assert ht.eta_squared(groups) == pytest.approx(13.5 / 17.5)
    t = pd.DataFrame([[10, 20], [30, 40]])
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    assert ht.cramers_v(chi2, t) == pytest.approx(np.sqrt(chi2 / 100))


def test_effect_size_recomputed(d2):
    e = EFF[(EFF.period == "2020_2025") & (EFF.measure == "Cohen's d (weighted)") & (EFF.scale == "log")].value.iloc[0]
    f, m = d2[d2.female == 1], d2[d2.female == 0]
    assert e == pytest.approx(ht.cohens_d(f.log_y, m.log_y, f.w, m.w))
    assert 0.2 < e < 0.8                                            # small-to-medium, as reported


# 6. regression coefficients -------------------------------------------------------------
def test_wls_coefficients_match_numpy(d2):
    s = ra.estimation_sample(d2)
    X = patsy.dmatrix("female + exp + exp2 + C(education_eda)", s, return_type="dataframe")
    sw = np.sqrt(s.w.to_numpy())
    beta = np.linalg.lstsq(X.to_numpy() * sw[:, None], s.log_y.to_numpy() * sw, rcond=None)[0]
    r = sa.wls_cluster("log_y ~ female + exp + exp2 + C(education_eda)", s)
    assert np.allclose(beta, r.params[X.columns].to_numpy(), atol=1e-8)


def test_simple_model_equals_weighted_mean_difference(d2):
    s = ra.estimation_sample(d2)
    m1 = REG[(REG.period == "2020_2025") & (REG.model == "M1 gender only") & (REG.term == "female")].coef.iloc[0]
    diff = np.average(s[s.female == 1].log_y, weights=s[s.female == 1].w) - \
        np.average(s[s.female == 0].log_y, weights=s[s.female == 0].w)
    assert m1 == pytest.approx(diff, abs=1e-9)


def test_same_sample_across_ladder():
    lad = REG[REG.model.str.match(r"M\d") & (REG.term == "female")]
    assert lad.groupby("period").n.nunique().eq(1).all()


def test_no_problematic_multicollinearity():
    v = pd.read_csv(REP / "vif_results.csv")
    assert (v.gvif_adj < np.sqrt(10)).all()                    # GVIF^(1/2df) < sqrt(10) ~ VIF < 10


def test_interval_regression_converged_and_matches_ols_without_controls(d2):
    dg = pd.read_csv(REP / "regression_diagnostics.csv")
    assert dg[dg.model.str.startswith("IR7")].converged.astype(bool).all()
    # diagnostics must report the coefficient of 'female' (looked up by name, not by position)
    for p in sa.PERIODS:
        coef = REG[(REG.period == p) & REG.model.str.startswith("IR7") & (REG.term == "female")].coef.iloc[0]
        assert dg[(dg.period == p) & dg.model.str.startswith("IR7")].female_coef.iloc[0] == pytest.approx(coef)
    s = ra.estimation_sample(d2)
    X = patsy.dmatrix("female", s, return_type="dataframe")
    ols = sa.wls_cluster("log_y ~ female", s)
    th, V, ok = ra.interval_regression(X.to_numpy(), s.lo_real.to_numpy(), s.hi_real.to_numpy(), s.w.to_numpy(),
                                       s.cluster.to_numpy(), np.append(ols.params.to_numpy(), np.log(0.6)))
    assert ok and th[1] == pytest.approx(ols.params["female"], abs=0.01)


# 7. Oaxaca-Blinder ----------------------------------------------------------------------
def test_oaxaca_identity_and_link_to_regression():
    for p in sa.PERIODS:
        o = OAX[(OAX.period == p) & (OAX.reference == "pooled (Neumark)")].set_index("component").log_points
        assert o["explained"] + o["unexplained"] == pytest.approx(o["gap"], abs=1e-9)
        det = o[o.index.str.startswith("explained: ")].sum()
        assert det == pytest.approx(o["explained"], abs=1e-9)
        b = REG[(REG.period == p) & (REG.model == "M7 full") & (REG.term == "female")].coef.iloc[0]
        assert o["unexplained"] == pytest.approx(-b, abs=1e-6)


def test_oaxaca_matches_statsmodels(d2):
    from statsmodels.stats.oaxaca import OaxacaBlinder
    s = ra.estimation_sample(d2).sample(6000, random_state=3).reset_index(drop=True)
    X = patsy.dmatrix("exp + exp2 + C(education_eda) + log_hours", s, return_type="dataframe")
    mine = dcp.decompose(X.to_numpy(), s.log_y.to_numpy(), s.female.to_numpy(), np.ones(len(s)), list(X.columns))
    ex = X.copy()
    ex["female"] = s.female.to_numpy()
    ref = OaxacaBlinder(s.log_y.to_numpy(), ex.to_numpy(), bifurcate=ex.shape[1] - 1, hasconst=True,
                        swap=True).two_fold(two_fold_type="pooled").params
    assert mine["unexplained"] == pytest.approx(ref[0], abs=1e-8)
    assert mine["explained"] == pytest.approx(ref[1], abs=1e-8)
    assert mine["gap"] == pytest.approx(ref[2], abs=1e-8)


# 8. survey weights ----------------------------------------------------------------------
def test_weights_used_in_gap(d2):
    w = CI[(CI.period == "2020_2025") & (CI.measure == "mean gap %")].estimate.iloc[0]
    f, m = d2[d2.female == 1], d2[d2.female == 0]
    weighted = (np.average(m.y, weights=m.w) - np.average(f.y, weights=f.w)) / np.average(m.y, weights=m.w) * 100
    unweighted = (m.y.mean() - f.y.mean()) / m.y.mean() * 100
    assert w == pytest.approx(weighted) and abs(w - unweighted) > 0.1


def test_design_summary():
    s = pd.read_csv(REP / "survey_design_summary.csv")
    assert (~s.strata_available).all() and (~s.psu_available).all()
    assert (s.kish_deff_weights > 1).all()
    assert (s.se_female_weighted_cluster > s.se_female_naive_ols).all()   # naive SE understate uncertainty


def test_bootstrap_resamples_whole_households(d2):
    codes, counts = sa.poisson_bootstrap_counts(d2.cluster.to_numpy(), 3)
    rep = counts[0, codes]
    assert (pd.Series(rep).groupby(d2.cluster.to_numpy()).nunique() == 1).all()


# 9. multiple testing --------------------------------------------------------------------
def test_benjamini_hochberg_manual():
    p = np.concatenate([HYP.p_value.to_numpy(), SUB.p_value.dropna().to_numpy()])
    n = len(p)
    o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    manual = np.empty(n)
    manual[o] = np.minimum(adj, 1)
    got = np.concatenate([HYP.p_value_bh.to_numpy(), SUB.p_value_bh.dropna().to_numpy()])
    assert np.allclose(got, manual, rtol=1e-9, atol=1e-300)
    assert (got >= p - 1e-15).all()


def test_small_groups_not_tested():
    assert SUB.loc[SUB.reliability == "suppressed", "p_value"].isna().all()


def test_notebook_executed():
    import nbformat
    nb = nbformat.read(ROOT / "notebooks" / "04_statistical_analysis.ipynb", as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and all(c.get("outputs") for c in code[1:])          # first cell = imports only
    assert all(c.get("execution_count") for c in code)               # every cell was executed
    assert not any(o.get("output_type") == "error" for c in code for o in c.outputs)


def test_figures_created():
    figs = sorted(sa.FIG.glob("0[1-7]_*.png"))
    assert len(figs) == 7 and all(f.stat().st_size > 20_000 for f in figs)
