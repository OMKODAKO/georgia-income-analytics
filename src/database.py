"""
Stage 02 — SQLite database  database/income_data.db

Target architecture (from the project spec) + what the real data support:

  data_sources       one row per official file (34)                      PK source_id
  individuals        LFS persons 15+, one row per person-interview        PK individual_id  FK source_id
  income             LFS INDIVIDUAL net earnings (employees)              PK income_id      FK individual_id
  households         HIES households                                      PK household_id   FK source_id
  household_income   HIES HOUSEHOLD income by source (long format)        PK (household_id, income_type) FK household_id
  household_income_types  definition of each household income item         PK income_type
  hies_members       HIES household members (no income at person level)   PK member_id      FK household_id
  + variable_mapping, cleaning_log, income_definitions, cpi_annual, pareto_top_band,
    outlier_flags, data_quality_summary                                   (documentation tables)
  + views v_lfs_analytical_<period>, v_ml_lfs_employees_<period>

Individual and household income are in DIFFERENT tables (income vs household_income)
and are never combined.  Survey weights and design ids (uid, household_key, quarter)
are kept.  Original codes are stored next to harmonised labels.

Usage:  cd src && python database.py          (rebuilds the database from data/processed)
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

import data_collection as dc
import income_standardization as inc

DB = dc.PROJECT_ROOT / "database" / "income_data.db"
PROCESSED = dc.PROJECT_ROOT / "data" / "processed"
REPORTS = dc.PROJECT_ROOT / "reports"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE data_sources (
    source_id        TEXT PRIMARY KEY,
    dataset          TEXT NOT NULL,          -- lfs | hies | meta | agg
    source_name      TEXT NOT NULL,
    source_url       TEXT NOT NULL,
    listing_page     TEXT NOT NULL,
    survey_year      INTEGER,
    collection_date  TEXT NOT NULL,          -- UTC download time
    file_name        TEXT NOT NULL,
    file_format      TEXT NOT NULL,
    file_bytes       INTEGER NOT NULL,
    sha256           TEXT NOT NULL
);

CREATE TABLE individuals (
    individual_id            TEXT PRIMARY KEY,
    source_id                TEXT NOT NULL REFERENCES data_sources(source_id),
    source_row               INTEGER NOT NULL,   -- row in the original ECSTAT file
    survey_year              INTEGER NOT NULL,
    period                   TEXT NOT NULL CHECK (period IN ('2017_2019','2020_2025')),
    quarter_no               INTEGER,
    month                    INTEGER,
    uid                      INTEGER NOT NULL,   -- original UID
    member_no                INTEGER NOT NULL,   -- original MemberNo
    household_key            INTEGER,            -- DiaryID (2020+) / UID (2017-19): cluster id
    gender                   TEXT CHECK (gender IN ('female','male')),
    gender_code              INTEGER CHECK (gender_code IN (1,2)),   -- original Sex code
    age                      INTEGER,
    age_group                TEXT,
    region                   TEXT,
    region_code              INTEGER,            -- original Geostat LFS code
    region_harmonised        TEXT,
    urban_rural              TEXT,
    education                TEXT,               -- harmonised group
    education_code           INTEGER,            -- original 13-level code
    years_schooling_approx   REAL,
    potential_experience     REAL,
    occupation               INTEGER,            -- ISCO major group (Occupation_converted)
    occupation_isco4         INTEGER,
    industry_nace2           INTEGER,
    sector                   TEXT,
    sector_code              INTEGER,
    labour_status            TEXT,
    employment_status        TEXT,               -- harmonised status in main job
    status_code              INTEGER,            -- original (B7_Status / Status)
    is_employee              INTEGER NOT NULL CHECK (is_employee IN (0,1)),
    working_hours            REAL,               -- usual weekly hours, main job (impossible -> NULL)
    working_hours_original   REAL,
    part_time                INTEGER,
    informal                 INTEGER,
    contract_written         INTEGER,
    household_size           INTEGER,
    relationship             INTEGER,
    marital_status           INTEGER,
    weight_quarter           REAL NOT NULL CHECK (weight_quarter > 0),
    weight_annual            REAL NOT NULL,
    flag_age_out_of_range    INTEGER, flag_age_mismatch_demographic INTEGER,
    flag_duplicate_key       INTEGER, flag_year_mismatch INTEGER,
    UNIQUE (source_id, uid, member_no)
);

CREATE TABLE income (
    income_id                INTEGER PRIMARY KEY,
    individual_id            TEXT NOT NULL UNIQUE REFERENCES individuals(individual_id),
    income_type              TEXT NOT NULL,      -- lfs_net_earnings_main_job
    income_unit              TEXT NOT NULL CHECK (income_unit = 'individual'),
    income_status            TEXT NOT NULL,      -- valid | refused | missing_nonresponse | ...
    band_code_original       REAL,               -- raw code incl. 0 / 88 / 10
    band_code                INTEGER,            -- valid bands only
    band_code_common         INTEGER,            -- 2020-25 mapped on 2017-19 bands
    income_lower             REAL,
    income_upper             REAL,               -- NULL = open top band
    income_amount            REAL,               -- point estimate (midpoint / Pareto), nominal
    income_amount_real_2025  REAL,
    amount_basis             TEXT,
    is_top_band              INTEGER,
    currency                 TEXT NOT NULL CHECK (currency = 'GEL'),
    income_period            TEXT NOT NULL,      -- last worked month
    gross_net                TEXT NOT NULL,
    CHECK (income_status = 'valid' OR income_amount IS NULL),
    CHECK (income_amount IS NULL OR income_amount > 0)
);

CREATE TABLE households (
    household_id             TEXT PRIMARY KEY,
    source_id                TEXT NOT NULL REFERENCES data_sources(source_id),
    source_row               INTEGER NOT NULL,
    survey_year              INTEGER NOT NULL,
    quarter_no               INTEGER,
    interview_month          TEXT,
    uid                      INTEGER NOT NULL,
    diary_id                 INTEGER,
    household_size           INTEGER,
    eq_adult                 REAL,
    household_income         REAL,               -- total income, monthly average, GEL, nominal
    household_income_real_2025 REAL,
    household_income_per_capita REAL,
    household_region         TEXT,
    region_code              INTEGER,            -- original HIES RegNo (different coding from LFS)
    region_harmonised        TEXT,
    urban_rural              TEXT,
    weight_quarter           REAL NOT NULL CHECK (weight_quarter > 0),
    weight_annual            REAL NOT NULL,
    head_sex                 TEXT, head_age INTEGER, head_education TEXT,
    n_members_listed         INTEGER, n_female INTEGER, n_male INTEGER,
    flag_zero_total_income   INTEGER, flag_negative_income INTEGER,
    flag_income_identity_items INTEGER, flag_income_identity_total INTEGER, flag_size_mismatch INTEGER,
    UNIQUE (source_id, uid)
);

CREATE TABLE household_income_types (
    income_type              TEXT PRIMARY KEY,
    original_variable        TEXT NOT NULL,
    is_income                INTEGER NOT NULL,   -- 0 = other cash inflow (borrowing, property sale)
    income_unit              TEXT NOT NULL CHECK (income_unit = 'household'),
    currency                 TEXT NOT NULL CHECK (currency = 'GEL'),
    income_period            TEXT NOT NULL
);

CREATE TABLE household_income (          -- composite PK, WITHOUT ROWID: 1.75 M rows, compact
    household_id             TEXT NOT NULL REFERENCES households(household_id),
    income_type              TEXT NOT NULL REFERENCES household_income_types(income_type),
    income_amount            REAL NOT NULL,
    income_amount_real_2025  REAL,
    PRIMARY KEY (household_id, income_type)
) WITHOUT ROWID;

CREATE TABLE hies_members (
    member_id                TEXT PRIMARY KEY,
    household_id             TEXT NOT NULL REFERENCES households(household_id),
    source_id                TEXT NOT NULL REFERENCES data_sources(source_id),
    survey_year              INTEGER NOT NULL,
    member_no                INTEGER NOT NULL,
    relationship             INTEGER,
    gender                   TEXT CHECK (gender IN ('female','male')),
    gender_code              INTEGER CHECK (gender_code IN (1,2)),
    age                      INTEGER,
    education                TEXT,
    education_code           INTEGER
);

CREATE INDEX ix_ind_year ON individuals(survey_year);
CREATE INDEX ix_ind_period ON individuals(period, is_employee);
CREATE INDEX ix_hh_year ON households(survey_year);
CREATE INDEX ix_mem_hh ON hies_members(household_id);
"""

VIEWS = """
CREATE VIEW v_lfs_analytical_{p} AS
    SELECT i.*, inc.income_status, inc.band_code, inc.band_code_common, inc.income_lower, inc.income_upper,
           inc.income_amount, inc.income_amount_real_2025, inc.is_top_band
    FROM individuals i LEFT JOIN income inc ON inc.individual_id = i.individual_id
    WHERE i.period = '{p}';
CREATE VIEW v_ml_lfs_employees_{p} AS
    SELECT * FROM v_lfs_analytical_{p} WHERE is_employee = 1 AND income_status = 'valid';
"""

HH_INCOME_ITEMS = {  # harmonised column -> (original variable, is_income)
    "inc_hired_employment": ("ShemDaq", 1), "inc_self_employment": ("ShemTviTdasaqm", 1),
    "inc_agriculture_sales": ("Shem_Sof", 1), "inc_property": ("Qonebidan", 1),
    "inc_pensions_benefits": ("PensStipDaxm", 1), "inc_remittances_abroad": ("Ucxoetidan", 1),
    "inc_gifts": ("Axloblebisagan", 1), "inc_cash_and_transfers": ("ShemosavaliDaTransf", 1),
    "inc_non_cash": ("ArafuladiMoxmareba", 1), "inc_total": ("Shemosavalisul", 1),
    "inflow_property_disposal": ("QonebisGayidvit", 0), "inflow_borrowing_savings": ("SesxAnDanazog", 0),
    "inflow_other_cash": ("SxvaFuladiSaxsrebi", 0), "inflows_cash_total": ("Fuladisaxsrebi", 0),
    "inflows_total": ("Saxsrebi_Sul", 0),
}


def _bool_to_int(df):
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


def sources_table() -> pd.DataFrame:
    m = dc.load_manifest()
    return pd.DataFrame([dict(source_id=k, dataset=v["group"], source_name=v["description"],
                              source_url=v["url"], listing_page=v["page"], survey_year=v["year"],
                              collection_date=v["downloaded_utc"], file_name=v["local_file"],
                              file_format=v["fmt"], file_bytes=v["bytes"], sha256=v["sha256"])
                         for k, v in m.items()])


def individuals_and_income(y: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(PROCESSED / f"lfs_clean_{y}.csv.gz", low_memory=False)
    ind = d.rename(columns={"year_source_file": "survey_year", "sex": "gender", "sex_code": "gender_code",
                            "education_group": "education", "occupation_isco1": "occupation",
                            "hours_usual": "working_hours", "hours_usual_original": "working_hours_original"})
    cols = [c for c in ind.columns if c in IND_COLS]
    ind = _bool_to_int(ind[cols].copy())
    asked = d[d.is_employee | d.earnings_status.eq("value_for_non_employee")]
    i = pd.DataFrame({
        "individual_id": asked.individual_id, "income_type": "lfs_net_earnings_main_job",
        "income_unit": "individual", "income_status": asked.earnings_status,
        "band_code_original": asked.earnings_band_original, "band_code": asked.earnings_band,
        "band_code_common": asked.earnings_band_common, "income_lower": asked.earn_lower,
        "income_upper": asked.earn_upper.where(asked.earn_upper != float("inf")),
        "income_amount": asked.earn_mid_nominal, "income_amount_real_2025": asked.earn_mid_real_2025,
        "amount_basis": asked.earn_is_top_band.map({True: "pareto_top_band_mean", False: "band_midpoint"})
        .where(asked.earnings_status.eq("valid")),
        "is_top_band": asked.earn_is_top_band.astype(int), "currency": "GEL",
        "income_period": "last_worked_month", "gross_net": "net"})
    return ind, i


IND_COLS = ["individual_id", "source_id", "source_row", "survey_year", "period", "quarter_no", "month", "uid",
            "member_no", "household_key", "gender", "gender_code", "age", "age_group", "region", "region_code",
            "region_harmonised", "urban_rural", "education", "education_code", "years_schooling_approx",
            "potential_experience", "occupation", "occupation_isco4", "industry_nace2", "sector", "sector_code",
            "labour_status", "employment_status", "status_code", "is_employee", "working_hours",
            "working_hours_original", "part_time", "informal", "contract_written", "household_size",
            "relationship", "marital_status", "weight_quarter", "weight_annual", "flag_age_out_of_range",
            "flag_age_mismatch_demographic", "flag_duplicate_key", "flag_year_mismatch"]


def households_tables(y: int, deflator: float):
    h = pd.read_csv(PROCESSED / f"hies_households_clean_{y}.csv.gz", low_memory=False)
    hh = h.rename(columns={"year": "survey_year", "inc_total": "household_income",
                           "inc_total_real_2025": "household_income_real_2025",
                           "inc_total_per_capita": "household_income_per_capita",
                           "region": "household_region", "head_education_group": "head_education"})
    hh["interview_month"] = hh.interview_month.astype(str)
    keep = ["household_id", "source_id", "source_row", "survey_year", "quarter_no", "interview_month", "uid",
            "diary_id", "household_size", "eq_adult", "household_income", "household_income_real_2025",
            "household_income_per_capita", "household_region", "region_code", "region_harmonised",
            "urban_rural", "weight_quarter", "weight_annual", "head_sex", "head_age", "head_education",
            "n_members_listed", "n_female", "n_male", "flag_zero_total_income", "flag_negative_income",
            "flag_income_identity_items", "flag_income_identity_total", "flag_size_mismatch"]
    hh = _bool_to_int(hh[keep].copy())
    long = h.melt(id_vars=["household_id"], value_vars=list(HH_INCOME_ITEMS), var_name="income_type",
                  value_name="income_amount")
    long["income_amount_real_2025"] = long.income_amount * deflator
    m = pd.read_csv(PROCESSED / f"hies_members_clean_{y}.csv.gz")
    mem = m.rename(columns={"individual_id": "member_id", "year": "survey_year", "sex": "gender",
                            "sex_code": "gender_code", "education_group": "education"})[
        ["member_id", "household_id", "source_id", "survey_year", "member_no", "relationship", "gender",
         "gender_code", "age", "education", "education_code"]]
    return hh, long, mem


def build(db=DB) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()                      # derived artefact: always rebuilt from processed data
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    con.execute("PRAGMA foreign_keys = ON")

    def append(df, table):
        df.to_sql(table, con, if_exists="append", index=False, chunksize=20000)

    append(sources_table(), "data_sources")
    append(pd.DataFrame([dict(income_type=k, original_variable=v[0], is_income=v[1], income_unit="household",
                              currency="GEL", income_period="monthly_average_reference_quarter")
                         for k, v in HH_INCOME_ITEMS.items()]), "household_income_types")
    cpi = pd.read_csv(REPORTS / "cpi_annual.csv").set_index("year").deflator_to_2025
    for y in range(2017, 2026):
        ind, i = individuals_and_income(y)
        append(ind, "individuals")
        append(i, "income")
        hh, long, mem = households_tables(y, cpi[y])
        append(hh, "households")
        append(long, "household_income")
        append(mem, "hies_members")
    for name, path in [("variable_mapping", "variable_mapping.csv"), ("cleaning_log", "cleaning_log.csv"),
                       ("income_definitions", "income_definitions.csv"), ("cpi_annual", "cpi_annual.csv"),
                       ("pareto_top_band", "pareto_top_band.csv"),
                       ("data_quality_summary", "data_quality_summary.csv")]:
        pd.read_csv(REPORTS / path).to_sql(name, con, index=False)
    pd.read_csv(PROCESSED / "outlier_flags.csv.gz").to_sql("outlier_flags", con, index=False)
    for p in ("2017_2019", "2020_2025"):
        con.executescript(VIEWS.format(p=p))
    con.commit()
    con.execute("VACUUM")
    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    if bad:
        raise RuntimeError(f"foreign key violations: {bad[:5]}")
    return con


def describe(con) -> pd.DataFrame:
    rows = []
    for (name, typ) in con.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
                                   "ORDER BY type, name").fetchall():
        n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
        rows.append(dict(object=name, type=typ, rows=n, columns=len(cols)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    con = build()
    print(describe(con).to_string(index=False))
    print("size MB:", round(DB.stat().st_size / 1e6, 1))
