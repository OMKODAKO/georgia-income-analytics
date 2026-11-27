"""
Stage 04 — Survey design and shared estimation tools.

What the public LFS files allow (Stage 01/02 findings):
  * sampling weights  : yes  (weight_annual = quarterly personal weight / 4)
  * clusters          : households only — 2020-2025 DiaryID (same household across its
                        interviews); 2017-2019 only UID, which changes every quarter, so
                        repeated interviews of one household are NOT linked (clustering weaker)
  * strata / PSU      : NOT published  -> a full design-based (Taylor linearisation with
                        strata and PSUs) analysis is impossible.

Approximation used everywhere in Stage 04 ("survey-adjusted" = this, not a full design):
  * point estimates weighted with weight_annual
  * standard errors cluster-robust by household (Liang-Zeger sandwich), which is the
    linearisation variance without strata; ignoring strata usually makes SEs conservative,
    ignoring PSU clustering above the household makes them too small -> stated as limitation
  * cluster (Poisson) bootstrap for ratios and quantiles (gap %, median gap)

Usage:  cd src && python survey_analysis.py     -> reports/survey_design_summary.csv
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import data_collection as dc
import eda

DB = dc.PROJECT_ROOT / "database" / "income_data.db"
REPORTS = dc.PROJECT_ROOT / "reports"
FIG = REPORTS / "figures" / "stage_04"
PERIODS = eda.PERIODS
RNG_SEED = 20261127


_CACHE: dict = {}


def load_period(p: str) -> pd.DataFrame:
    """Employees with a valid earnings band, with analysis variables (no imputation)."""
    if "eda" not in _CACHE:
        _CACHE["eda"] = eda.load()
    d = _CACHE["eda"][p].copy().reset_index(drop=True)
    d["female"] = (d.gender == "female").astype(int)
    d["y"] = d.income_amount_real_2025                      # real 2025 GEL (band point estimate)
    d["log_y"] = np.log(d.y)
    defl = d.income_amount_real_2025 / d.income_amount
    d["lo_real"] = d.income_lower * defl                   # interval bounds in real GEL
    d["hi_real"] = d.income_upper.fillna(np.inf) * defl
    d["hourly"] = d.y / (d.working_hours * eda.WEEKS_PER_MONTH)
    d["log_hourly"] = np.log(d.hourly)
    d["log_hours"] = np.log(d.working_hours)
    # potential experience, centred -> removes the structural collinearity of exp and exp^2
    d["exp"] = d.potential_experience - d.potential_experience.mean()
    d["exp2"] = d.exp ** 2 / 100
    d["informal_cat"] = d.informal.map({0: "formal", 1: "informal"}).fillna("not_defined")
    # informality is defined only for non-agricultural jobs: in 2020-25 "not defined" == agriculture
    # (perfectly collinear with the industry dummy) -> use a 0/1 flag + an 'unknown' flag outside agriculture
    d["informal_flag"] = (d.informal == 1).astype(int)
    d["informal_unknown"] = (d.informal.isna() & (d.industry_nace2 != 1)).astype(int)
    d["contract"] = d.contract_written.map({1: "written", 0: "oral"}).fillna("unknown")
    d["sector_cat"] = d.sector.fillna("unknown")
    d["year_cat"] = d.survey_year.astype(str)
    d["cluster"] = d.household_key.astype("int64")
    d["w"] = d.weight_annual
    return d


def wls_cluster(formula: str, d: pd.DataFrame, weights: str = "w"):
    """Weighted least squares with household-clustered robust SE (rows with NA dropped first)."""
    import patsy
    y, X = patsy.dmatrices(formula, d, return_type="dataframe", NA_action="drop")
    idx = y.index
    model = smf.wls(formula, d.loc[idx], weights=d.loc[idx, weights])
    return model.fit(cov_type="cluster", cov_kwds={"groups": d.loc[idx, "cluster"].to_numpy()})


def poisson_bootstrap_counts(clusters: np.ndarray, B: int, seed: int = RNG_SEED):
    """Cluster Poisson bootstrap: each household gets a Poisson(1) multiplicity per replicate.
    Returns (codes, counts[B, n_clusters]) where codes maps rows -> cluster column."""
    codes, uniq = pd.factorize(clusters)
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, size=(B, len(uniq))).astype(np.float32)
    return codes, counts


def bootstrap_mean_gap(d: pd.DataFrame, value="y", B=500, seed=RNG_SEED) -> np.ndarray:
    """Replicates of the mean gap % = (m_male - m_female) / m_male * 100 (weighted)."""
    codes, C = poisson_bootstrap_counts(d.cluster.to_numpy(), B, seed)
    out = {}
    for sex in (0, 1):
        s = d[d.female == sex]
        n = C.shape[1]
        sy = np.bincount(codes[s.index.map(d.index.get_loc)], weights=(s.w * s[value]).to_numpy(), minlength=n)
        sw = np.bincount(codes[s.index.map(d.index.get_loc)], weights=s.w.to_numpy(), minlength=n)
        out[sex] = (C @ sy) / (C @ sw)
    return (out[0] - out[1]) / out[0] * 100


def design_summary() -> pd.DataFrame:
    rows = []
    for p in PERIODS:
        d = load_period(p)
        w = d.w
        kish = 1 + (w.std(ddof=0) / w.mean()) ** 2
        naive = smf.ols("log_y ~ female", d).fit()
        wh = smf.wls("log_y ~ female", d, weights=d.w).fit(cov_type="HC1")
        wc = wls_cluster("log_y ~ female", d)
        rows.append(dict(
            period=p, n=len(d), n_clusters=d.cluster.nunique(), mean_cluster_size=round(len(d) / d.cluster.nunique(), 2),
            cluster_variable="DiaryID (household, all interviews)" if p == "2020_2025"
            else "UID (household within one quarter only)",
            strata_available=False, psu_available=False,
            weight_cv=round(w.std(ddof=0) / w.mean(), 3), kish_deff_weights=round(kish, 3),
            effective_n_weights=round(len(d) / kish),
            se_female_naive_ols=naive.bse["female"], se_female_weighted_hc1=wh.bse["female"],
            se_female_weighted_cluster=wc.bse["female"],
            deff_se_cluster_vs_naive=round((wc.bse["female"] / naive.bse["female"]) ** 2, 3),
            coef_female_unweighted=naive.params["female"], coef_female_weighted=wc.params["female"]))
    s = pd.DataFrame(rows)
    s.to_csv(REPORTS / "survey_design_summary.csv", index=False)
    return s


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(design_summary().T.to_string())
