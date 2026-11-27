"""Stage 06 dashboard tests: data layer (numbers consistent with Stages 03-04) and the Streamlit app
(headless, streamlit.testing.v1.AppTest)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
import data as D  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
GC = pd.read_csv(ROOT / "reports" / "gender_income_comparison.csv")


# ---- data layer ---------------------------------------------------------------------
def test_unknown_period_rejected():
    with pytest.raises(ValueError):
        D.apply_filters("2017_2025")


@pytest.mark.parametrize("period", list(D.PERIODS))
def test_periods_never_mixed_and_weights_pooled(period):
    d = D.apply_filters(period)
    assert set(d.period) == {period}
    assert d.w.sum() == pytest.approx(d.groupby("survey_year").weight_annual.sum().mean(), rel=1e-9)


@pytest.mark.parametrize("period", list(D.PERIODS))
def test_overview_matches_stage03(period):
    s = D.gender_summary(D.apply_filters(period))
    ref = GC[(GC.period == period) & (GC.year == "pooled")].iloc[0]
    assert s["female"]["mean"] == pytest.approx(ref.female_mean_real_weighted, abs=0.01)
    assert s["male"]["mean"] == pytest.approx(ref.male_mean_real_weighted, abs=0.01)
    assert s["gap_mean"] == pytest.approx(ref.gap_mean_pct_real_weighted, abs=0.01)
    assert s["gap_median"] == pytest.approx(ref.gap_median_pct_real_weighted, abs=0.05)


def test_yearly_series_match_stage03():
    y = D.by_year(D.apply_filters("2020_2025")).set_index("year")
    for _, r in GC[(GC.period == "2020_2025") & (GC.year != "pooled")].iterrows():
        assert y.loc[int(r.year), "gap_mean"] == pytest.approx(r.gap_mean_pct_real_weighted, abs=0.01)


def test_filters_restrict_rows():
    d = D.apply_filters("2020_2025", years=[2025], regions=["Tbilisi"], education=["Master"])
    assert set(d.survey_year) == {2025} and set(d.region_eda) == {"Tbilisi"} and set(d.education_eda) == {"Master"}
    assert d.w.sum() == pytest.approx(d.weight_annual.sum())          # one year -> annual weights


def test_small_cells_suppressed():
    d = D.apply_filters("2020_2025", years=[2025], ages=["15-17"], regions=["Guria"])
    s = D.gender_summary(d)
    assert s["reliability"] == "suppressed" and math.isnan(s["gap_mean"])
    g = D.by_group(D.apply_filters("2020_2025"), "education_eda", D.eda.EDU_ORDER)
    hp = g[g.group == "Higher professional"].iloc[0]
    assert hp.reliability == "suppressed" and math.isnan(hp.female_mean) and math.isnan(hp.gap_mean)


def test_band_shares_sum_to_one():
    b = D.band_shares(D.apply_filters("2017_2019"))
    assert np.allclose(b.sum(), 1)


def test_quartiles_ordered():
    s = D.gender_summary(D.apply_filters("2020_2025", regions=["Kakheti"]))
    for sex in ("female", "male"):
        assert s[sex]["p25"] <= s[sex]["median"] <= s[sex]["p75"]


# ---- Streamlit app -------------------------------------------------------------------
@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    assert not app.exception
    assert len(app.tabs) == 6
    labels = [m.label for m in app.metric]
    assert labels[:4] == ["ქალი — საშუალო", "კაცი — საშუალო", "სხვაობა (საშუალო)", "სხვაობა (მედიანა)"]
    assert app.metric[2].value == "25.5%"


def test_filter_changes_results(app):
    # sidebar multiselects in order: 0 years, 1 age groups, 2 regions, 3 education, 4 occupation
    app.sidebar.multiselect[2].select("Tbilisi").run()
    assert not app.exception
    assert app.metric[0].value != "756 ₾"
    app.sidebar.multiselect[2].unselect("Tbilisi").run()
    assert app.metric[0].value == "756 ₾"


def test_period_switch(app):
    app.sidebar.radio[0].set_value("2017_2019").run()
    assert not app.exception and app.metric[2].value == "27.9%"
    app.sidebar.radio[0].set_value("2020_2025").run()


def test_tiny_or_empty_selection_shows_message_not_numbers():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.multiselect[0].set_value([2025])
    at.sidebar.multiselect[1].set_value(["15-17"])
    at.sidebar.multiselect[2].set_value(["Guria"])
    at.run()
    assert not at.exception
    msgs = [e.value for e in at.error] + [w.value for w in at.warning]
    assert any(("მცირე შერჩევა" in m) or ("მონაცემი არ არის" in m) for m in msgs)
    assert not any(m.label == "სხვაობა (საშუალო)" for m in at.metric)       # no numbers shown


def test_prediction_form(app):
    btn = [b for b in app.button if b.label == "პროგნოზი"]
    assert btn
    btn[0].click().run()
    assert not app.exception
    vals = {m.label: m.value for m in app.metric}
    assert vals["ქალი"].endswith("₾") and vals["კაცი"].endswith("₾") and vals["სხვაობა"].endswith("%")


def test_small_sample_error_message():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.multiselect[4].set_value(["Armed forces / other"])     # 16 women in 2020-2025
    at.run()
    assert not at.exception
    assert any("მცირე შერჩევა" in e.value for e in at.error)
    assert not any(m.label == "სხვაობა (საშუალო)" for m in at.metric)
