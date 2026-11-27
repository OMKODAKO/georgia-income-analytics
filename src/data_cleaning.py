"""
Stage 02 — Data cleaning (each source and year separately).

Principles
  * raw files are only READ (data/raw is never written)
  * no row is deleted: problems are FLAGGED (flag_* columns) and every change is logged
    in reports/cleaning_log.csv (dataset, year, step, variable, rule, n_affected, action)
  * original codes are kept next to every recoded variable (*_code / *_original)
  * missing income is never imputed; unknown sex is never filled
  * real zero is distinguished from missing (LFS code 0 = "no earnings last month",
    HIES 0 GEL = household reported no income of that type)

Layers written
  data/processed/  clean data per source and year   (lfs_clean_<y>, hies_households_clean_<y>,
                                                     hies_members_clean_<y>)  .csv.gz
  data/final/      analytical datasets               lfs_analytical_2017_2019 / _2020_2025
                                                     hies_households_analytical_2017_2025
                   ML datasets (individual earnings, LFS only, per period)
                                                     ml_lfs_employees_2017_2019 / _2020_2025

Pooling rule: LFS years are pooled only inside a methodological period, and only
variables whose mapping says compatible_within_period=True are carried
(checked in pool_period()).  The two periods are never stacked (project decision).

Usage:  cd src && python data_cleaning.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

import data_collection as dc
import income_standardization as inc
import variable_mapping as vm
from data_profile import canonical_table

PROCESSED = dc.PROJECT_ROOT / "data" / "processed"
FINAL = dc.PROJECT_ROOT / "data" / "final"
REPORTS = dc.PROJECT_ROOT / "reports"
LOG_PATH = REPORTS / "cleaning_log.csv"

_log: list[dict] = []


def log(dataset, year, step, variable, rule, n, action, detail=""):
    _log.append(dict(dataset=dataset, year=year, step=step, variable=variable, rule=rule,
                     n_affected=int(n), action=action, detail=detail))


# ----------------------------------------------------------------- lookups ---
LFS_REGION = {11: "Tbilisi", 15: "Adjara A/R", 23: "Guria", 26: "Imereti", 29: "Kakheti",
              32: "Mtskheta-Mtianeti", 35: "Racha-Lechkhumi and Kvemo Svaneti",
              38: "Samegrelo-Zemo Svaneti", 41: "Samtskhe-Javakheti", 44: "Kvemo Kartli", 47: "Shida Kartli"}
HIES_REGION = {0: "Kakheti", 1: "Tbilisi", 2: "Shida Kartli", 3: "Kvemo Kartli", 5: "Samtskhe-Javakheti",
               7: "Adjara A/R", 8: "Guria", 9: "Samegrelo-Zemo Svaneti", 10: "Imereti",
               11: "Mtskheta-Mtianeti", 13: "Racha-Lechkhumi and Kvemo Svaneti"}
IMERETI_RACHA = "Imereti + Racha-Lechkhumi & Kvemo Svaneti"


def region_harmonised(name: pd.Series) -> pd.Series:
    """Racha is published separately only in some years -> merge with Imereti everywhere."""
    return name.replace({"Imereti": IMERETI_RACHA, "Racha-Lechkhumi and Kvemo Svaneti": IMERETI_RACHA})


EDU_GROUP = {1: "primary_or_less", 2: "primary_or_less", 3: "primary_or_less", 4: "primary_or_less",
             5: "lower_secondary", 6: "upper_secondary", 7: "vocational", 8: "vocational", 9: "vocational",
             10: "higher_professional", 11: "bachelor", 12: "master_or_doctor", 13: "master_or_doctor"}
# nominal years of schooling in the Georgian system (approximation, documented in the report)
EDU_YEARS = {1: 0, 2: 0, 3: 0, 4: 6, 5: 9, 6: 12, 7: 11, 8: 12, 9: 13, 10: 14, 11: 16, 12: 18, 13: 21}

STATUS = {  # period -> code -> harmonised status in employment (main job)
    "2017_2019": {1: "employee", 2: "self_employed", 3: "self_employed", 4: "other", 5: "contributing_family",
                  6: "other", 97: "other"},
    "2020_2025": {1: "employee", 2: "self_employed", 3: "contributing_family", 4: "other",
                  5: "contributing_family", 97: "other"},
}
SECTOR = {1: "state", 2: "private", 3: "other", 4: "other", 97: "other"}   # 98 = don't know -> missing
SEX = {1: "female", 2: "male"}


# ----------------------------------------------------------------- reading ---
def extracted(group: str, year: int) -> Path:
    return dc.RAW_DIR / dc.FOLDERS[group] / "extracted" / str(year)


def read_table(group: str, year: int, canonical: str, cols: list[str] | None = None) -> pd.DataFrame:
    path = next(p for p in extracted(group, year).rglob("*.sav") if canonical_table(p.stem) == canonical)
    df, _ = pyreadstat.read_sav(str(path), usecols=cols)
    df.attrs["file"] = str(path.relative_to(dc.PROJECT_ROOT))
    return df


def to_int_key(df: pd.DataFrame, col: str, dataset: str, year: int, table: str) -> pd.DataFrame:
    """Coerce an id column to integer, logging text->number conversions and failures."""
    if df[col].dtype == object or str(df[col].dtype).startswith("str"):
        log(dataset, year, "types", f"{table}.{col}", "id stored as text", len(df), "type_cast", "str -> Int64")
    num = pd.to_numeric(df[col], errors="coerce")
    bad = num.isna() & df[col].notna()
    if bad.any():
        log(dataset, year, "types", f"{table}.{col}", "non-numeric id", bad.sum(), "set_missing")
    df[col] = num.round().astype("Int64")
    return df


# --------------------------------------------------------------------- LFS ---
def load_lfs_year(year: int) -> pd.DataFrame:
    """Merge the year's tables onto ECSTAT (persons 15+) using the mapping table."""
    m = vm.resolve("lfs", year)
    by_table: dict[str, dict[str, str]] = {}
    for harm, (tab, var) in m.items():
        by_table.setdefault(tab, {})[var] = harm
    ec = read_table("lfs", year, "ecstat")
    src_file = ec.attrs["file"]
    base = ec[list(by_table["ecstat"])].rename(columns=by_table["ecstat"])
    base["source_row"] = np.arange(len(base))
    base = to_int_key(base, "uid", "lfs", year, "ecstat")
    base["member_no"] = base.member_no.astype("Int64")
    for tab, cols in by_table.items():
        if tab == "ecstat":
            continue
        keys = ["UID", "MemberNo"] if tab in ("demographic", "personal_weights") else ["UID"]
        extra = ["Age"] if tab == "demographic" else []
        t = read_table("lfs", year, tab, list(dict.fromkeys(keys + list(cols) + extra)))
        t = to_int_key(t, "UID", "lfs", year, tab)
        no_id = t.UID.isna().sum()
        if no_id:
            log("lfs", year, "linkage", f"{tab}.UID", "row without household id cannot be linked",
                no_id, "excluded_from_merge", "raw row kept untouched in data/raw")
            t = t[t.UID.notna()]
        t = t.rename(columns={**cols, "UID": "uid", "MemberNo": "member_no", "Age": "age_demographic"})
        if "member_no" in t:
            t["member_no"] = t.member_no.astype("Int64")
        on = ["uid", "member_no"] if "member_no" in t and tab != "familysize" else ["uid"]
        t = t.drop(columns=[c for c in t.columns if c in base.columns and c not in on])
        before = len(base)
        base = base.merge(t, on=on, how="left", validate="many_to_one" if on == ["uid"] else "one_to_one")
        assert len(base) == before
    base.insert(0, "year_source_file", year)
    base["source_id"] = f"lfs_{year}"
    base["source_file"] = src_file
    return base


def clean_lfs_year(year: int) -> pd.DataFrame:
    d = load_lfs_year(year)
    period = vm.period_of(year)
    d["period"] = period
    n = len(d)
    log("lfs", year, "input", "*", "ECSTAT persons 15+", n, "none")

    # duplicates ---------------------------------------------------------------
    dup = d.duplicated(["uid", "member_no"], keep=False)
    d["flag_duplicate_key"] = dup
    log("lfs", year, "duplicates", "uid+member_no", "duplicate person key", dup.sum(), "flag")
    full_dup = d.drop(columns=["source_row"]).duplicated(keep=False)
    log("lfs", year, "duplicates", "*", "fully identical rows", full_dup.sum(), "flag")

    # year consistency --------------------------------------------------------
    wrong_year = d.year.notna() & (d.year != year)
    log("lfs", year, "validity", "year", "interview year != file year", wrong_year.sum(), "flag")
    d["flag_year_mismatch"] = wrong_year

    # sex -----------------------------------------------------------------------
    bad = ~d.sex.isin([1, 2])
    log("lfs", year, "validity", "sex", "code not in {1,2} -> missing (never imputed)", bad.sum(), "set_missing")
    d["sex_code"] = d.sex.where(~bad).astype("Int64")
    d["sex"] = d.sex_code.map(SEX)

    # age -----------------------------------------------------------------------
    d["flag_age_out_of_range"] = ~d.age.between(15, 110)
    log("lfs", year, "validity", "age", "outside 15-110 (ECSTAT universe is 15+)", d.flag_age_out_of_range.sum(),
        "flag")
    mism = d.age_demographic.notna() & (d.age_demographic != d.age)
    d["flag_age_mismatch_demographic"] = mism
    log("lfs", year, "consistency", "age", "ECSTAT age != demographic-table age", mism.sum(), "flag")
    d["age"] = d.age.astype("Int64")
    d["age_group"] = pd.cut(d.age.astype(float), [14, 24, 34, 44, 54, 64, 200],
                            labels=["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)

    # region --------------------------------------------------------------------
    d["region_code"] = d.region_code.astype("Int64")
    d["region"] = d.region_code.map(LFS_REGION)
    unk = d.region.isna() & d.region_code.notna()
    log("lfs", year, "validity", "region_code", "unknown region code", unk.sum(), "flag")
    d["region_harmonised"] = region_harmonised(d.region)
    log("lfs", year, "harmonisation", "region", "Racha (35) merged with Imereti (26) for comparability",
        (d.region_code == 35).sum(), "recode", "original kept in region_code / region")
    d["urban_rural"] = d.urban_rural.map({1: "urban", 2: "rural"})

    # education -----------------------------------------------------------------
    d["education_code"] = d.education_code.astype("Int64")
    bad = d.education_code.notna() & ~d.education_code.isin(list(EDU_GROUP))
    log("lfs", year, "validity", "education_code", "code outside 1-13", bad.sum(), "set_missing")
    d.loc[bad, "education_code"] = pd.NA
    d["education_group"] = d.education_code.map(EDU_GROUP)
    d["years_schooling_approx"] = d.education_code.map(EDU_YEARS)
    d["potential_experience"] = (d.age - d.years_schooling_approx - 6).clip(lower=0)

    # labour status ---------------------------------------------------------------
    d["labour_status"] = np.select([d.employed == 1, d.unemployed == 1], ["employed", "unemployed"],
                                   "outside_labour_force")
    d["status_code"] = d.status_code.astype("Int64")
    d["employment_status"] = d.status_code.map(STATUS[period]).where(d.employed == 1)
    unk = (d.employed == 1) & d.status_code.notna() & d.employment_status.isna()
    log("lfs", year, "validity", "status_code", "undocumented status code", unk.sum(), "flag",
        ",".join(map(str, sorted(d.loc[unk, "status_code"].dropna().unique()))))
    d["is_employee"] = (d.employed == 1) & (d.status_code == 1)

    # occupation / industry / sector ---------------------------------------------
    d["occupation_isco1"] = d.occupation_isco1.astype("Int64")
    d["occupation_isco4"] = d.occupation_isco4.astype("Int64")
    d["industry_nace2"] = d.industry_nace2.astype("Int64")
    d["sector_code"] = d.sector_code.astype("Int64")
    d["sector"] = d.sector_code.map(SECTOR)
    log("lfs", year, "missing", "sector_code", "98 'do not know' -> missing", (d.sector_code == 98).sum(),
        "set_missing", "code kept in sector_code")

    # hours -----------------------------------------------------------------------
    h = d.hours_usual
    impossible = h.notna() & ((h <= 0) | (h > 168))
    log("lfs", year, "validity", "hours_usual", "<=0 or >168 h/week (impossible)", impossible.sum(), "set_missing")
    d["hours_usual_original"] = h
    d["hours_usual"] = h.where(~impossible)
    d["part_time"] = d.full_part_time.map({1: 0, 2: 1})
    d["informal"] = d.informal.astype("Int64")
    d["contract_written"] = d.contract_type.map({1: 1, 2: 0})

    # earnings --------------------------------------------------------------------
    top = inc.TOP_CODE[period]
    band = d.earnings_band
    d["earnings_band_original"] = band
    status = np.select(
        [~d.is_employee & band.isna(),
         ~d.is_employee & band.notna(),
         band.between(1, top),
         band == inc.NO_EARNINGS,
         band == inc.REFUSED,
         (year == 2017) & (band == inc.UNDOCUMENTED_2017),
         band.isna()],
        ["not_asked", "value_for_non_employee", "valid", "no_earnings_last_month", "refused",
         "undocumented_code", "missing_nonresponse"], "invalid_code")
    d["earnings_status"] = status
    for s in ["valid", "no_earnings_last_month", "refused", "undocumented_code", "missing_nonresponse",
              "value_for_non_employee", "invalid_code"]:
        k = (d.earnings_status == s).sum()
        if k or s in ("valid", "missing_nonresponse"):
            log("lfs", year, "income", "earnings_band", f"earnings_status = {s}", k,
                "classify" if s == "valid" else "set_missing",
                {"undocumented_code": "code 10: not in labels/questionnaire -> missing (decision)",
                 "missing_nonresponse": "employee without answer (from 2023 refusals are stored this way)",
                 "no_earnings_last_month": "real zero, kept separate from missing"}.get(s, ""))
    d["earnings_band"] = band.where(d.earnings_status == "valid").astype("Int64")

    # weights ---------------------------------------------------------------------
    badw = d.weight_quarter.isna() | (d.weight_quarter <= 0)
    log("lfs", year, "validity", "weight_quarter", "missing or non-positive weight", badw.sum(), "flag")
    d["weight_annual"] = d.weight_quarter / 4
    d["household_key"] = d.diary_id.astype("Int64") if "diary_id" in d else d.uid
    d["individual_id"] = ("LFS" + str(year) + "_" + d.uid.astype(str) + "_" + d.member_no.astype(str))
    return d


# -------------------------------------------------------------------- HIES ---
def clean_hies_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = vm.resolve("hies", year)
    tabs: dict[str, dict[str, str]] = {}
    for harm, (tab, var) in m.items():
        tabs.setdefault(tab, {})[var] = harm
    hh = read_table("hies", year, "sysschedule").rename(columns=tabs["sysschedule"])
    src_file = hh.attrs["file"]
    hh = hh[list(tabs["sysschedule"].values())]
    hh["source_row"] = np.arange(len(hh))
    hh = to_int_key(hh, "uid", "hies", year, "sysschedule")
    log("hies", year, "input", "*", "households (sysschedule)", len(hh), "none")
    for tab in ("familysize", "tblincomes"):
        t = read_table("hies", year, tab, ["UID"] + list(tabs[tab]))
        t = to_int_key(t, "UID", "hies", year, tab).rename(columns={**tabs[tab], "UID": "uid"})
        before = len(hh)
        hh = hh.merge(t, on="uid", how="left", validate="one_to_one")
        assert len(hh) == before
        log("hies", year, "linkage", tab, "household without row in table", hh[t.columns[1]].isna().sum(), "flag")

    # year / date
    q = read_table("hies", year, "123")
    qy = dict(zip(q.QuartNo.astype(int), q.YarNo.astype(int)))
    hh["year"] = hh.quarter_no.astype(int).map(qy)
    hh["interview_month"] = pd.to_datetime(hh.interview_month.astype(str), errors="coerce")
    log("hies", year, "types", "interview_month", "MonthYear parsed to date (date in 2017, text later)",
        len(hh), "type_cast")
    log("hies", year, "validity", "year", "quarter year != file year", (hh.year != year).sum(), "flag")

    # region
    hh["region_code"] = hh.region_code.astype("Int64")
    hh["region"] = hh.region_code.map(HIES_REGION)
    log("hies", year, "validity", "region_code", "unknown region code",
        (hh.region.isna() & hh.region_code.notna()).sum(), "flag")
    hh["region_harmonised"] = region_harmonised(hh.region)
    log("hies", year, "harmonisation", "region", "code 10 (Imereti, incl. Racha in some years) and 13 (Racha) "
        "merged", hh.region_code.isin([10, 13]).sum(), "recode", "original kept in region_code")
    hh["urban_rural"] = hh.urban_rural.map({1: "urban", 2: "rural"})
    hh["weight_annual"] = hh.weight_quarter / 4

    # income checks: negative, accounting identities, zero vs missing
    inc_cols = [c for c in hh.columns if c.startswith(("inc_", "inflow"))]
    hh["flag_negative_income"] = (hh[inc_cols] < 0).any(axis=1)
    log("hies", year, "validity", "income", "negative amount (impossible for 'received' items)",
        hh.flag_negative_income.sum(), "flag", "kept as reported; exclude via flag_negative_income")
    items = ["inc_hired_employment", "inc_self_employment", "inc_agriculture_sales", "inc_property",
             "inc_pensions_benefits", "inc_remittances_abroad", "inc_gifts"]
    d1 = (hh[items].sum(axis=1) - hh.inc_cash_and_transfers).abs() > 0.5
    d2 = (hh.inc_cash_and_transfers + hh.inc_non_cash - hh.inc_total).abs() > 0.5
    hh["flag_income_identity_items"], hh["flag_income_identity_total"] = d1, d2
    log("hies", year, "consistency", "inc_cash_and_transfers", "!= sum of items 1-7 (tol 0.5 GEL)", d1.sum(), "flag")
    log("hies", year, "consistency", "inc_total", "!= cash_and_transfers + non_cash (tol 0.5 GEL)", d2.sum(), "flag")
    hh["flag_zero_total_income"] = hh.inc_total == 0
    log("hies", year, "income", "inc_total", "total income = 0 (real zero, not missing)",
        hh.flag_zero_total_income.sum(), "flag")
    hh["inc_total_per_capita"] = hh.inc_total / hh.household_size
    hh["inc_total_per_eq_adult"] = hh.inc_total / hh.eq_adult

    # members (tblshinda02) and head of household
    mem = read_table("hies", year, "tblshinda02", ["UID"] + list(tabs["tblshinda02"]))
    mem = to_int_key(mem, "UID", "hies", year, "tblshinda02").rename(columns={**tabs["tblshinda02"], "UID": "uid"})
    mem["member_no"] = mem.member_no.astype("Int64")
    log("hies", year, "input", "*", "household members (tblshinda02)", len(mem), "none")
    bad = ~mem.sex.isin([1, 2])
    log("hies", year, "validity", "sex", "code not in {1,2} -> missing", bad.sum(), "set_missing")
    mem["sex_code"] = mem.sex.where(~bad).astype("Int64")
    mem["sex"] = mem.sex_code.map(SEX)
    mem["flag_age_out_of_range"] = ~mem.age.between(0, 110)
    log("hies", year, "validity", "age", "outside 0-110", mem.flag_age_out_of_range.sum(), "flag")
    mem["age"] = mem.age.astype("Int64")
    mem["education_code"] = mem.education_code.astype("Int64")
    mem["education_group"] = mem.education_code.map(EDU_GROUP)
    log("hies", year, "missing", "education_code", "education missing (mostly children)",
        mem.education_code.isna().sum(), "none")
    dup = mem.duplicated(["uid", "member_no"]).sum()
    log("hies", year, "duplicates", "uid+member_no", "duplicate member key", dup, "flag")
    mem["individual_id"] = "HIES" + str(year) + "_" + mem.uid.astype(str) + "_" + mem.member_no.astype(str)
    mem["household_id"] = "HIES" + str(year) + "_" + mem.uid.astype(str)
    mem["year"] = year
    mem["source_id"] = f"hies_{year}"

    heads = mem[mem.relationship == 1]
    nheads = heads.groupby("uid").size()
    log("hies", year, "consistency", "relationship", "households without exactly one head",
        (~hh.uid.isin(nheads[nheads == 1].index)).sum(), "flag")
    heads = heads.drop_duplicates("uid")[["uid", "sex", "age", "education_group"]].rename(
        columns={"sex": "head_sex", "age": "head_age", "education_group": "head_education_group"})
    hh = hh.merge(heads, on="uid", how="left")
    cnt = mem.groupby("uid").agg(n_members_listed=("member_no", "size"),
                                 n_female=("sex_code", lambda s: (s == 1).sum()),
                                 n_male=("sex_code", lambda s: (s == 2).sum()))
    hh = hh.merge(cnt, on="uid", how="left")
    diff = hh.n_members_listed != hh.household_size
    hh["flag_size_mismatch"] = diff
    log("hies", year, "consistency", "household_size", "FamilySize != members listed in Shinda02",
        diff.sum(), "flag", "absent members (NotPresentNow) may explain differences")
    hh["household_id"] = "HIES" + str(year) + "_" + hh.uid.astype(str)
    hh["source_id"] = f"hies_{year}"
    hh["source_file"] = src_file
    return hh, mem


# ------------------------------------------------------------------ layers ---
def pool_period(frames: list[pd.DataFrame], period: str) -> pd.DataFrame:
    """Stack years of ONE period, after checking the mapping allows it for every mapped variable."""
    mp = vm.mapping()
    mp = mp[(mp.dataset == "lfs") & (mp.period == period)]
    incompatible = mp[~mp.compatible_within_period].harmonised.unique()
    if len(incompatible):
        raise ValueError(f"variables not poolable within {period}: {incompatible}")
    years = sorted({int(f.year_source_file.iloc[0]) for f in frames})
    expected = list(range(2017, 2020) if period == "2017_2019" else range(2020, 2026))
    if years != expected:
        raise ValueError(f"{period}: years {years} != {expected}")
    return pd.concat(frames, ignore_index=True)


LFS_KEEP = ["individual_id", "source_id", "source_file", "source_row", "year_source_file", "period", "year",
            "quarter_no", "month", "uid", "member_no", "household_key", "sex", "sex_code", "age", "age_group",
            "relationship", "marital_status", "household_size", "region_code", "region", "region_harmonised",
            "urban_rural", "education_code", "education_group", "years_schooling_approx", "potential_experience",
            "labour_status", "employed", "unemployed", "hired", "status_code", "employment_status",
            "is_employee", "occupation_isco1", "occupation_isco4", "industry_nace2", "sector_code", "sector",
            "hours_usual", "hours_usual_original", "part_time", "informal", "contract_written",
            "earnings_band_original", "earnings_status", "earnings_band", "earnings_band_common",
            "earn_lower", "earn_upper", "earn_mid_nominal", "earn_mid_real_2025", "earn_is_top_band",
            "weight_quarter", "weight_annual",
            "flag_duplicate_key", "flag_year_mismatch", "flag_age_out_of_range", "flag_age_mismatch_demographic"]

ML_FEATURES = ["sex", "age", "education_group", "years_schooling_approx", "potential_experience",
               "region_harmonised", "urban_rural", "occupation_isco1", "industry_nace2", "sector",
               "hours_usual", "part_time", "informal", "contract_written", "household_size",
               "marital_status", "year", "quarter_no"]
ML_TARGETS = ["earnings_band", "earnings_band_common", "earn_lower", "earn_upper", "earn_mid_real_2025",
              "log_earn_mid_real_2025"]


def ml_dataset(analytical: pd.DataFrame) -> pd.DataFrame:
    """Employees with a valid earnings band. No imputation: feature missingness kept as NaN
    plus missing_* indicators; categorical features left as labels (encoding = Stage 05)."""
    e = analytical[analytical.is_employee & analytical.earnings_status.eq("valid")].copy()
    e["log_earn_mid_real_2025"] = np.log(e.earn_mid_real_2025)
    for c in ML_FEATURES:
        if e[c].isna().any():
            e[f"missing_{c}"] = e[c].isna().astype(int)
    keep = ["individual_id", "household_key", "period", "weight_annual"] + ML_TARGETS + ML_FEATURES + \
        [c for c in e.columns if c.startswith("missing_")]
    return e[keep]


def write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def run() -> dict:
    _log.clear()
    cpi = inc.cpi_annual()
    cpi.to_csv(REPORTS / "cpi_annual.csv", index=False)
    inc.INCOME_DEFINITIONS.to_csv(REPORTS / "income_definitions.csv", index=False)

    lfs = {y: clean_lfs_year(y) for y in range(2017, 2026)}
    # Pareto top-band means need all employees of each year with valid bands
    allv = pd.concat([d[d.earnings_status == "valid"][["year_source_file", "period", "earnings_band",
                                                        "weight_annual"]] for d in lfs.values()])
    top = inc.pareto_top(allv.rename(columns={"year_source_file": "year"}))
    top.to_csv(REPORTS / "pareto_top_band.csv", index=False)
    out = {}
    for y, d in lfs.items():
        d = inc.add_lfs_amounts(d.assign(year=d.year.fillna(y)), cpi, top)
        log("lfs", y, "income", "earn_mid_real_2025", f"deflated with CPI {y}->2025",
            d.earn_mid_real_2025.notna().sum(), "derive",
            f"factor {cpi.set_index('year').deflator_to_2025[y]:.4f}")
        lfs[y] = d[LFS_KEEP]
        write(lfs[y], PROCESSED / f"lfs_clean_{y}.csv.gz")
    for period, years in [("2017_2019", range(2017, 2020)), ("2020_2025", range(2020, 2026))]:
        a = pool_period([lfs[y] for y in years], period)
        write(a, FINAL / f"lfs_analytical_{period}.csv.gz")
        ml = ml_dataset(a)
        write(ml, FINAL / f"ml_lfs_employees_{period}.csv.gz")
        out[f"lfs_analytical_{period}"], out[f"ml_lfs_employees_{period}"] = a, ml

    hh_all, mem_all = [], []
    for y in range(2017, 2026):
        hh, mem = clean_hies_year(y)
        hh = inc.add_hies_real(hh, cpi, ["inc_total", "inc_cash_and_transfers", "inc_hired_employment",
                                         "inc_total_per_capita", "inc_total_per_eq_adult"])
        write(hh, PROCESSED / f"hies_households_clean_{y}.csv.gz")
        write(mem, PROCESSED / f"hies_members_clean_{y}.csv.gz")
        hh_all.append(hh)
        mem_all.append(mem)
    # HIES: all mapped variables identical/compatible across 2017-2025 (cross-year check) -> pooled
    hh_a = pd.concat(hh_all, ignore_index=True)
    write(hh_a, FINAL / "hies_households_analytical_2017_2025.csv.gz")
    out["hies_households"], out["hies_members"] = hh_a, pd.concat(mem_all, ignore_index=True)
    out["lfs_years"] = lfs
    out["cpi"], out["pareto"] = cpi, top
    pd.DataFrame(_log).to_csv(LOG_PATH, index=False)
    return out


if __name__ == "__main__":
    o = run()
    for k, v in o.items():
        if isinstance(v, pd.DataFrame):
            print(f"{k:35} {len(v):>9,} rows  {v.shape[1]} cols")
    print(o["pareto"].to_string(index=False))
