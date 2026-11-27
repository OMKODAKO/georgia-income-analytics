"""Stage 03 EDA validation (run after src/eda.py)."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eda  # noqa: E402

REP = ROOT / "reports"
GC = pd.read_csv(REP / "gender_income_comparison.csv")
DESC = pd.read_csv(REP / "descriptive_statistics.csv")


@pytest.fixture(scope="module")
def data():
    return eda.load()


# 1. counts match the source datasets ------------------------------------------------
@pytest.mark.parametrize("period", eda.PERIODS)
def test_counts_match_database_and_ml_dataset(data, period):
    con = sqlite3.connect(eda.DB)
    n_db = con.execute(f"SELECT COUNT(*) FROM v_ml_lfs_employees_{period}").fetchone()[0]
    n_ml = len(pd.read_csv(ROOT / "data" / "final" / f"ml_lfs_employees_{period}.csv.gz", usecols=["individual_id"]))
    assert len(data[period]) == n_db == n_ml
    d = DESC[(DESC.period == period) & DESC.scope.str.startswith("pooled") & (DESC.sex == "all")
             & (DESC.weighting == "unweighted")]
    assert int(d.n.iloc[0]) == n_db


def test_yearly_counts_sum_to_period(data):
    for p, years in eda.PERIODS.items():
        rows = GC[(GC.period == p) & (GC.year != "pooled")]
        assert sorted(rows.year.astype(int)) == years
        assert (rows.n_female + rows.n_male).sum() == len(data[p])


# 2. gender categories ------------------------------------------------------------------
def test_gender_categories(data):
    for p in eda.PERIODS:
        assert set(data[p].gender) == {"female", "male"}
    assert set(DESC.sex) == {"female", "male", "all"}
    for f in ["age_analysis", "education_analysis", "regional_analysis", "occupation_analysis"]:
        t = pd.read_csv(REP / f"{f}.csv")
        assert {"n_female", "n_male", "female_mean", "male_mean"} <= set(t.columns)


# 3. income calculations ----------------------------------------------------------------
def test_weighted_helpers():
    assert eda.wmean([1, 3], [1, 3]) == pytest.approx(2.5)
    assert eda.wstd([1, 1, 1], [1, 2, 3]) == 0
    assert eda.wquantile([10, 20, 30], [1, 1, 1], 0.5) == 20


def test_interval_quantile_uniform_and_pareto():
    # one closed band 400-600 -> median 500, quartiles 450/550
    assert eda.interval_quantile([400], [600], [np.nan], [1], 0.5) == pytest.approx(500, abs=0.01)
    assert eda.interval_quantile([400], [600], [np.nan], [1], 0.25) == pytest.approx(450, abs=0.01)
    # open top band, Pareto alpha=3, L=2000: median = 2000 * 2**(1/3)
    assert eda.interval_quantile([2000], [np.inf], [3.0], [1], 0.5) == pytest.approx(2000 * 2 ** (1 / 3), rel=1e-4)
    # equal weights in 0-100 and 100-200 -> median at the boundary 100
    assert eda.interval_quantile([0, 100], [100, 200], [np.nan] * 2, [1, 1], 0.5) == pytest.approx(100, abs=0.01)


def test_yearly_mean_recomputed_from_database():
    con = sqlite3.connect(eda.DB)
    d = pd.read_sql("SELECT gender, income_amount_real_2025 x, weight_annual w FROM v_ml_lfs_employees_2020_2025 "
                    "WHERE survey_year=2025", con)
    for sex in ["female", "male"]:
        g = d[d.gender == sex]
        expected = np.sum(g.x * g.w) / g.w.sum()
        got = GC[(GC.year == "2025")][f"{sex}_mean_real_weighted"].iloc[0]
        assert got == pytest.approx(expected, abs=0.01)


def test_quantiles_ordered():
    d = DESC.dropna(subset=["p25", "median", "p75"])
    assert ((d["min"] <= d.p25 + 1e-6) & (d.p25 <= d["median"]) & (d["median"] <= d.p75) &
            (d.p75 <= d["max"] + 1e-6)).all()


# 4. percentage gap ---------------------------------------------------------------------
def test_gap_formula():
    assert eda.gap_pct(1000, 750) == pytest.approx(25)
    assert eda.gap_pct(800, 1000) == pytest.approx(-25)
    assert np.isnan(eda.gap_pct(0, 5))


def test_gap_columns_consistent():
    for tag in ["real_weighted", "real_unweighted"]:
        exp = (GC[f"male_mean_{tag}"] - GC[f"female_mean_{tag}"]) / GC[f"male_mean_{tag}"] * 100
        assert np.allclose(GC[f"gap_mean_pct_{tag}"], exp, atol=0.02)
        exp = (GC[f"male_median_{tag}"] - GC[f"female_median_{tag}"]) / GC[f"male_median_{tag}"] * 100
        assert np.allclose(GC[f"gap_median_pct_{tag}"], exp, atol=0.02)


def test_small_groups_flagged_and_suppressed():
    for f in ["age_analysis", "education_analysis", "occupation_analysis", "regional_analysis"]:
        t = pd.read_csv(REP / f"{f}.csv")
        n = t[["n_female", "n_male"]].min(axis=1)
        assert (t.reliability == np.select([n < 30, n < 100], ["suppressed", "low_n"], "ok")).all()
        assert t.loc[t.reliability == "suppressed", "gap_mean_pct"].isna().all()


# 5. figures ------------------------------------------------------------------------------
def test_figures_created():
    pngs = sorted((REP / "figures").glob("[0-9][0-9]_*.png"))
    assert len(pngs) >= 12 and all(p.stat().st_size > 20_000 for p in pngs)
    idx = pd.read_csv(REP / "figures" / "figure_index.csv")
    assert {p.name for p in pngs} <= set(idx.file)
    assert (REP / "figures" / "10_income_trends_by_year_interactive.html").exists()


def test_notebook_executed():
    import nbformat
    nb = nbformat.read(ROOT / "notebooks" / "03_exploratory_data_analysis.ipynb", as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code and all(c.get("outputs") for c in code)
    assert not any(o.get("output_type") == "error" for c in code for o in c.outputs)


# 6. CSV files saved ------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["descriptive_statistics", "gender_income_comparison", "regional_analysis",
                                  "education_analysis", "occupation_analysis", "age_analysis",
                                  "working_hours_analysis", "correlation_matrix", "income_band_distribution",
                                  "income_groups_quintiles", "hies_household_income_by_head_sex"])
def test_csv_saved(name):
    t = pd.read_csv(REP / f"{name}.csv")
    assert len(t) > 0


# 7. survey weights -----------------------------------------------------------------------
def test_weighted_and_unweighted_both_reported():
    assert set(DESC.weighting) == {"weighted", "unweighted"}
    w = DESC[DESC.weighting == "weighted"]
    assert w.population.notna().all() and DESC[DESC.weighting == "unweighted"].population.isna().all()


def test_pooled_weights_represent_average_year(data):
    for p, years in eda.PERIODS.items():
        d = data[p]
        assert d.w_pooled.sum() == pytest.approx(d.groupby("survey_year").weight_annual.sum().mean(), rel=1e-9)


# 8. no mixing of income types ----------------------------------------------------------
def test_individual_and_household_income_not_mixed(data):
    con = sqlite3.connect(eda.DB)
    assert con.execute("SELECT DISTINCT income_unit FROM income").fetchall() == [("individual",)]
    for p in eda.PERIODS:
        assert not {"household_income", "household_income_real_2025"} & set(data[p].columns)
    h = pd.read_csv(REP / "hies_household_income_by_head_sex.csv")
    assert set(h.income_unit) == {"household (per capita)"}
    assert set(DESC.income_type) == {"individual net monthly earnings (employees)"}
    assert not any("household" in c for c in GC.columns)


def test_periods_not_pooled_together():
    assert set(DESC.period) == {"2017_2019", "2020_2025"}
    pooled = DESC[DESC.scope.str.startswith("pooled")]
    assert set(pooled.scope) == {"pooled 2017_2019", "pooled 2020_2025"}
    assert (pooled.income_basis == "real_2025").all()          # nominal GEL never pooled across years
