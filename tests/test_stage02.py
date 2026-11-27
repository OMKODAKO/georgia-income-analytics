"""Stage 02 data-quality tests (database, cleaning, standardisation).

Run after the pipeline:  variable_mapping -> data_cleaning -> data_validation -> database
    .venv/bin/python -m pytest -q tests/test_stage02.py
"""
import sqlite3
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data_collection as dc          # noqa: E402
import database                       # noqa: E402
import data_cleaning                  # noqa: E402
import income_standardization as inc  # noqa: E402
import variable_mapping as vm         # noqa: E402

DB = database.DB
TABLES = pd.read_csv(ROOT / "reports" / "stage01_table_profile.csv")
YEARS = range(2017, 2026)


@pytest.fixture(scope="module")
def con():
    c = sqlite3.connect(DB)
    c.execute("PRAGMA foreign_keys = ON")
    yield c
    c.close()


def q(con, sql):
    return con.execute(sql).fetchall()


def raw_rows(ds, year, table):
    t = TABLES[(TABLES.dataset == ds) & (TABLES.year == year) & (TABLES.canonical_table == table)]
    return int(t.n_rows.iloc[0])


# 1. database is created ---------------------------------------------------------
def test_01_database_exists_and_opens(con):
    assert DB.exists() and DB.stat().st_size > 0
    assert q(con, "PRAGMA integrity_check") == [("ok",)]


def test_01_schema_builds_in_memory():
    c = sqlite3.connect(":memory:")
    c.executescript(database.SCHEMA)
    assert {"data_sources", "individuals", "income", "households", "household_income"} <= \
        {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}


# 2. tables are created correctly -------------------------------------------------
SPEC_COLUMNS = {
    "data_sources": {"source_id", "source_name", "source_url", "survey_year", "collection_date", "file_name"},
    "individuals": {"individual_id", "source_id", "survey_year", "gender", "age", "region", "education",
                    "occupation", "employment_status", "working_hours"},
    "income": {"income_id", "individual_id", "income_type", "income_amount", "currency", "income_period"},
    "households": {"household_id", "household_size", "household_income", "household_region", "survey_year"},
}


@pytest.mark.parametrize("table,cols", SPEC_COLUMNS.items())
def test_02_spec_columns_present(con, table, cols):
    have = {r[1] for r in q(con, f"PRAGMA table_info({table})")}
    assert cols <= have, cols - have


def test_02_weights_and_design_ids_kept(con):
    for t, cols in [("individuals", {"weight_quarter", "weight_annual", "uid", "member_no", "household_key",
                                     "quarter_no"}),
                    ("households", {"weight_quarter", "weight_annual", "uid", "diary_id", "quarter_no"})]:
        assert cols <= {r[1] for r in q(con, f"PRAGMA table_info({t})")}


# 3. data load correctly -----------------------------------------------------------
@pytest.mark.parametrize("year", YEARS)
def test_03_row_counts_match_raw(con, year):
    assert q(con, f"SELECT COUNT(*) FROM individuals WHERE survey_year={year}")[0][0] == raw_rows("lfs", year, "ecstat")
    assert q(con, f"SELECT COUNT(*) FROM households WHERE survey_year={year}")[0][0] == raw_rows("hies", year, "sysschedule")
    assert q(con, f"SELECT COUNT(*) FROM hies_members WHERE survey_year={year}")[0][0] == raw_rows("hies", year, "tblshinda02")


def test_03_data_sources_complete(con):
    assert q(con, "SELECT COUNT(*) FROM data_sources")[0][0] == len(dc.SOURCES)


# 4. primary keys unique ----------------------------------------------------------
@pytest.mark.parametrize("table,pk", [("individuals", "individual_id"), ("income", "income_id"),
                                      ("income", "individual_id"), ("households", "household_id"),
                                      ("hies_members", "member_id"), ("data_sources", "source_id"),
                                      ("household_income", "household_id || '|' || income_type")])
def test_04_primary_keys_unique(con, table, pk):
    n, d = q(con, f"SELECT COUNT(*), COUNT(DISTINCT {pk}) FROM {table}")[0]
    assert n == d and n > 0


# 5. foreign keys work ------------------------------------------------------------
def test_05_no_fk_violations(con):
    assert q(con, "PRAGMA foreign_key_check") == []


def test_05_fk_enforced_on_bad_insert():
    c = sqlite3.connect(":memory:")
    c.executescript(database.SCHEMA)
    c.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO income (individual_id, income_type, income_unit, income_status, currency, "
                  "income_period, gross_net) VALUES ('nobody','x','individual','refused','GEL','m','net')")


# 6. income stored correctly ------------------------------------------------------
def test_06_individual_income_definition(con):
    assert q(con, "SELECT DISTINCT currency, income_unit, income_period, gross_net FROM income") == \
        [("GEL", "individual", "last_worked_month", "net")]


def test_06_amount_only_for_valid_and_within_band(con):
    assert q(con, "SELECT COUNT(*) FROM income WHERE income_status!='valid' AND income_amount IS NOT NULL")[0][0] == 0
    assert q(con, "SELECT COUNT(*) FROM income WHERE income_status='valid' AND income_amount IS NULL")[0][0] == 0
    bad = q(con, "SELECT COUNT(*) FROM income WHERE income_status='valid' AND income_upper IS NOT NULL "
                 "AND NOT (income_amount > income_lower AND income_amount <= income_upper)")[0][0]
    assert bad == 0
    assert q(con, "SELECT MIN(income_amount) FROM income WHERE is_top_band=1")[0][0] > 2000


def test_06_real_income_uses_cpi(con):
    cpi = pd.read_csv(ROOT / "reports" / "cpi_annual.csv").set_index("year").deflator_to_2025
    d = pd.read_sql("SELECT i.survey_year y, inc.income_amount a, inc.income_amount_real_2025 r FROM income inc "
                    "JOIN individuals i USING(individual_id) WHERE inc.income_status='valid'", con)
    assert np.allclose(d.r, d.a * d.y.map(cpi))


def test_06_cpi_matches_official_annual_inflation():
    # Geostat CPI page: annual average to annual average (%), 2018-2025
    official = {2018: 102.6, 2019: 104.9, 2020: 105.2, 2021: 109.6, 2022: 111.9, 2023: 102.5, 2024: 101.1,
                2025: 103.9}
    c = inc.cpi_annual().set_index("year").cpi_avg
    for y, v in official.items():
        assert round(100 * c[y] / c[y - 1], 1) == pytest.approx(v, abs=0.1), y


def test_06_band_nesting_for_common_scheme():
    assert inc.check_nesting()


@pytest.mark.parametrize("year", [2017, 2021, 2025])
def test_06_household_income_equals_raw_total(con, year):
    path = next((dc.RAW_DIR / "household_survey" / "extracted" / str(year)).rglob("tblincomes.sav"))
    raw, _ = pyreadstat.read_sav(str(path), usecols=["Shemosavalisul"])
    db = q(con, f"SELECT SUM(household_income) FROM households WHERE survey_year={year}")[0][0]
    assert db == pytest.approx(raw.Shemosavalisul.sum(), rel=1e-9)
    long = q(con, f"SELECT SUM(hi.income_amount) FROM household_income hi JOIN households h USING(household_id) "
                  f"WHERE h.survey_year={year} AND hi.income_type='inc_total'")[0][0]
    assert long == pytest.approx(db, rel=1e-9)


def test_06_household_and_individual_income_never_mixed(con):
    ind_cols = {r[1] for r in q(con, "PRAGMA table_info(individuals)")}
    assert not {"household_income", "income_amount"} & ind_cols
    assert q(con, "SELECT DISTINCT income_unit FROM household_income_types") == [("household",)]


# 7. gender coded correctly -------------------------------------------------------
def test_07_gender_codes(con):
    for t in ("individuals", "hies_members"):
        assert set(q(con, f"SELECT DISTINCT gender_code, gender FROM {t}")) == {(1, "female"), (2, "male")}


@pytest.mark.parametrize("year", [2017, 2018, 2023, 2025])
def test_07_gender_counts_equal_raw(con, year):
    path = next(p for p in (dc.RAW_DIR / "labour_force_survey" / "extracted" / str(year)).rglob("*.sav")
                if data_cleaning.canonical_table(p.stem) == "ecstat")
    raw, _ = pyreadstat.read_sav(str(path), usecols=["Sex"])
    db = dict(q(con, f"SELECT gender_code, COUNT(*) FROM individuals WHERE survey_year={year} GROUP BY 1"))
    assert db == raw.Sex.value_counts().astype(int).to_dict()


# 8. missing values handled correctly ---------------------------------------------
def test_08_undocumented_code_10_is_missing(con):
    rows = q(con, "SELECT income_status, COUNT(*), COUNT(income_amount) FROM income i JOIN individuals p "
                  "USING(individual_id) WHERE p.survey_year=2017 AND band_code_original=10 AND p.is_employee=1 "
                  "GROUP BY 1")
    assert rows == [("undocumented_code", 62, 0)]
    # non-employees with code 10 keep their own status and get no amount either
    # (code 10 is a valid band, 1001-1500 GEL, from 2020 — undocumented only in 2017)
    assert q(con, "SELECT COUNT(income_amount) FROM income i JOIN individuals p USING(individual_id) "
                  "WHERE p.survey_year=2017 AND band_code_original=10")[0][0] == 0


def test_08_refusals_and_zero_kept_distinct(con):
    st = dict(q(con, "SELECT income_status, COUNT(*) FROM income GROUP BY 1"))
    emp = "AND income_status != 'value_for_non_employee'"   # non-employees with a code keep their own status
    assert st["refused"] == q(con, f"SELECT COUNT(*) FROM income WHERE band_code_original=88 {emp}")[0][0]
    assert st["no_earnings_last_month"] == \
        q(con, f"SELECT COUNT(*) FROM income WHERE band_code_original=0 {emp}")[0][0]
    assert q(con, "SELECT COUNT(income_amount) FROM income WHERE band_code_original IN (0, 88)")[0][0] == 0
    # 2023+: refusals stored as system-missing -> missing_nonresponse, never imputed
    assert q(con, "SELECT COUNT(*) FROM income WHERE income_status='missing_nonresponse' "
                  "AND (band_code_original IS NOT NULL OR income_amount IS NOT NULL)")[0][0] == 0


def test_08_no_imputation_of_sex_or_hours(con):
    assert q(con, "SELECT COUNT(*) FROM individuals WHERE gender IS NULL")[0][0] == 0   # none missing in raw
    n = q(con, "SELECT COUNT(*) FROM individuals WHERE working_hours IS NULL AND working_hours_original IS NOT NULL")
    log = pd.read_csv(ROOT / "reports" / "cleaning_log.csv")
    assert n[0][0] == log[(log.variable == "hours_usual") & (log.action == "set_missing")].n_affected.sum()


def test_08_hies_zero_income_is_zero_not_null(con):
    assert q(con, "SELECT COUNT(*) FROM households WHERE household_income IS NULL")[0][0] == 0
    assert q(con, "SELECT COUNT(*) FROM households WHERE flag_zero_total_income=1 AND household_income != 0")[0][0] == 0


def test_08_every_change_logged():
    log = pd.read_csv(ROOT / "reports" / "cleaning_log.csv")
    assert {"dataset", "year", "variable", "rule", "n_affected", "action"} <= set(log.columns)
    assert set(log.year) == set(YEARS) and {"lfs", "hies"} == set(log.dataset)


# 9. raw data unchanged -----------------------------------------------------------
@pytest.mark.parametrize("src", dc.SOURCES, ids=lambda s: s.key)
def test_09_raw_files_unchanged(src):
    assert dc.sha256(dc.local_path(src)) == dc.load_manifest()[src.key]["sha256"]


@pytest.mark.parametrize("src", [s for s in dc.SOURCES if s.fmt == "zip"], ids=lambda s: s.key)
def test_09_extracted_copies_unchanged(src):
    folder = dc.RAW_DIR / dc.FOLDERS[src.group] / "extracted" / str(src.year)
    with zipfile.ZipFile(dc.local_path(src)) as z:
        for info in z.infolist():
            if not info.is_dir():
                assert zlib.crc32((folder / info.filename).read_bytes()) & 0xFFFFFFFF == info.CRC


def test_09_no_derived_files_in_raw():
    allowed = {"manifest.json", "download_log.csv"}
    for p in dc.RAW_DIR.rglob("*"):
        if p.is_file() and p.parent == dc.RAW_DIR:
            assert p.name in allowed, p


# 10. years pooled only when compatible --------------------------------------------
def test_10_periods_never_mixed(con):
    for p, years in [("2017_2019", {2017, 2018, 2019}), ("2020_2025", set(range(2020, 2026)))]:
        a = pd.read_csv(ROOT / "data" / "final" / f"lfs_analytical_{p}.csv.gz", usecols=["year_source_file", "period"])
        assert set(a.year_source_file) == years and set(a.period) == {p}
        assert {r[0] for r in q(con, f"SELECT DISTINCT survey_year FROM v_lfs_analytical_{p}")} == years


def test_10_pool_period_rejects_wrong_years():
    frame = pd.DataFrame({"year_source_file": [2019]})
    with pytest.raises(ValueError):
        data_cleaning.pool_period([frame], "2020_2025")


def test_10_pool_period_rejects_incompatible_variable(monkeypatch):
    m = vm.mapping()
    m.loc[m.harmonised == "sex", "compatible_within_period"] = False
    monkeypatch.setattr(vm, "mapping", lambda: m)
    frames = [pd.DataFrame({"year_source_file": [y]}) for y in (2017, 2018, 2019)]
    with pytest.raises(ValueError, match="not poolable"):
        data_cleaning.pool_period(frames, "2017_2019")


def test_10_mapping_validated_against_raw():
    m = pd.read_csv(ROOT / "reports" / "variable_mapping.csv")
    assert m.original_label.notna().all()          # every mapped variable exists in raw files
    across = m[(m.dataset == "lfs") & ~m.compatible_across_periods].harmonised.unique()
    assert {"earnings_band", "status_code", "employed"} <= set(across)


def test_10_ml_dataset_individual_lfs_only():
    for p in ("2017_2019", "2020_2025"):
        ml = pd.read_csv(ROOT / "data" / "final" / f"ml_lfs_employees_{p}.csv.gz")
        assert ml.individual_id.str.startswith("LFS").all() and set(ml.period) == {p}
        assert ml.earn_mid_real_2025.notna().all() and (ml.earn_mid_real_2025 > 0).all()
