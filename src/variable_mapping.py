"""
Stage 02 — Variable Mapping Table.

Maps every harmonised variable used by the project to its ORIGINAL table and
variable name in each survey year, plus a compatibility decision:

  compatible_within_period   may years inside one methodological period be pooled?
  compatible_across_periods  may 2017-2019 and 2020-2025 be pooled for this variable?

LFS periods (methodology change, Geostat: 13th ICLS until 2019, 19th/20th ICLS from 2020):
  P1 = 2017-2019, P2 = 2020-2025.  By project decision the periods are analysed SEPARATELY;
  the across-period flag documents what *could* be compared, never triggers a merge.

Every mapping row is validated against the raw-data inventory
(reports/stage01_variable_profile.csv): a mapping to a variable that does not exist
raises an error — no variables are invented.

Usage:  cd src && python variable_mapping.py   → reports/variable_mapping.csv
"""
from __future__ import annotations

import pandas as pd

import data_collection as dc

PROFILE = dc.PROJECT_ROOT / "reports" / "stage01_variable_profile.csv"
OUT = dc.PROJECT_ROOT / "reports" / "variable_mapping.csv"

P1, P2 = range(2017, 2020), range(2020, 2026)
LFS_YEARS, HIES_YEARS = range(2017, 2026), range(2017, 2026)


def period_of(year: int) -> str:
    return "2017_2019" if year <= 2019 else "2020_2025"


# (harmonised, {years: (canonical_table, original_variable)}, within, across, note)
LFS_MAP = [
    ("year", {P1: ("id", "Year"), P2: ("demographic", "Year")}, True, True, "interview year"),
    ("quarter_no", {P1: ("id", "QuarterNo"), P2: ("demographic", "QuarterNo")}, True, True,
     "Geostat quarter code (83 = 2017 Q1)"),
    ("month", {P1: ("id", "Month"), P2: ("ecstat", "Month")}, True, True, "interview month"),
    ("uid", {LFS_YEARS: ("ecstat", "UID")}, True, True,
     "household id within ONE quarter; text in some 2018 tables -> coerced to integer"),
    ("member_no", {LFS_YEARS: ("ecstat", "MemberNo")}, True, True, "member number within household"),
    ("diary_id", {P2: ("ecstat", "DiaryID")}, True, False,
     "household id across interviews; not published for 2017-2019"),
    ("sex", {LFS_YEARS: ("ecstat", "Sex")}, True, True, "1=female 2=male in every year"),
    ("age", {LFS_YEARS: ("ecstat", "Age")}, True, True, "completed years; ECSTAT covers 15+"),
    ("relationship", {LFS_YEARS: ("demographic", "Relationship")}, True, True, "relation to household head"),
    ("marital_status", {LFS_YEARS: ("demographic", "Marital_Status")}, True, True, ""),
    ("household_size", {P1: ("familysize", "Family_Size"), P2: ("demographic", "Family_Size")}, True, True, ""),
    ("region_code", {P1: ("id", "Region"), P2: ("ecstat", "Region")}, True, True,
     "same Geostat codes; 35 Racha-Lechkhumi only from 2019 -> region_harmonised merges 35 into Imereti"),
    ("urban_rural", {P1: ("id", "Urban_Rural"), P2: ("ecstat", "Urban_Rural")}, True, True, "1=urban 2=rural"),
    ("education_code", {P1: ("ecstat", "I1_Education"), P2: ("ecstat", "Education")}, True, True,
     "same 13-level national classification in both periods"),
    ("employed", {LFS_YEARS: ("ecstat", "Employed")}, True, False,
     "definition of employment changed in 2020 (subsistence producers excluded)"),
    ("unemployed", {LFS_YEARS: ("ecstat", "Unemployed")}, True, False, "ILO strict; definition changed 2020"),
    ("hired", {LFS_YEARS: ("ecstat", "Hired")}, True, True, "hired employee flag"),
    ("status_code", {P1: ("ecstat", "B7_Status"), P2: ("ecstat", "Status")}, True, False,
     "ICSE-93 (P1) vs new status question (P2): codes differ; only 'employee'=1 is comparable"),
    ("occupation_isco1", {LFS_YEARS: ("ecstat", "Occupation_converted")}, True, False,
     "major group; label ISCO-88 in P1; code 0 'Armed forces' (P1) vs 'Other' (P2)"),
    ("occupation_isco4", {P1: ("ecstat", "B5_Occupation"), P2: ("ecstat", "Occupation")}, True, False,
     "4-digit code"),
    ("industry_nace2", {P1: ("ecstat", "Brunch_2"), range(2020, 2021): ("ecstat", "Brunch_converted_2"),
                        range(2021, 2025): ("ecstat", "Brunch_converted"),
                        range(2025, 2026): ("ecstat", "Branch_converted")}, True, True,
     "NACE Rev.2 section (21 codes) in every year; variable renamed 3 times"),
    ("sector_code", {P1: ("ecstat", "B6_Sector"), P2: ("ecstat", "Sector_ownership")}, True, False,
     "P2 adds NGO (3) and international org (4) -> sector_harmonised: state/private/other"),
    ("hours_usual", {P1: ("ecstat", "C2_M_Usually_hours"), P2: ("ecstat", "M_Usually_hours")}, True, True,
     "usual weekly hours, main job"),
    ("full_part_time", {LFS_YEARS: ("ecstat", "Full_time_and_part_time")}, True, True, "1=full 2=part"),
    ("informal", {LFS_YEARS: ("ecstat", "Informal_employment")}, True, False,
     "non-agricultural employment only; employment definition changed 2020"),
    ("contract_type", {P1: ("ecstat", "B11_Agreement_type"), P2: ("ecstat", "B12_Agreement_type")}, True, False,
     "written vs oral; question numbering/filters changed"),
    ("earnings_band", {P1: ("ecstat", "B17_B18_Net_earnings"), P2: ("ecstat", "B24_B25_Net_earnings")},
     True, False, "net monthly earnings band, main job: 9 bands (P1) vs 12 bands (P2); "
                  "earnings_band_common maps P2 onto the P1 bands (exact nesting)"),
    ("weight_quarter", {P1: ("personal_weights", "P_Weights"), P2: ("ecstat", "P_Weights")}, True, True,
     "personal weight; weight_annual = weight_quarter / 4"),
]

HIES_MAP = [
    ("uid", {HIES_YEARS: ("sysschedule", "UID")}, True, True, "household id within a quarter"),
    ("diary_id", {HIES_YEARS: ("sysschedule", "DiaryID")}, True, True, "household id over the survey"),
    ("quarter_no", {HIES_YEARS: ("sysschedule", "QuartNo")}, True, True, "year via table 123"),
    ("interview_month", {HIES_YEARS: ("sysschedule", "MonthYear")}, True, True,
     "date in 2017, text in later years -> parsed to date"),
    ("region_code", {HIES_YEARS: ("sysschedule", "RegNo")}, True, True,
     "HIES coding (differs from LFS!); code 10 = Imereti+Racha in 2017-18 & 2020-22, Racha=13 "
     "in 2019 & 2023-25 -> region_harmonised merges 10 and 13"),
    ("urban_rural", {HIES_YEARS: ("sysschedule", "UrbanOrRural")}, True, True, "1=urban 2=rural"),
    ("weight_quarter", {HIES_YEARS: ("sysschedule", "Weights")}, True, True,
     "HOUSEHOLD weight (no person weights 2017+); weight_annual = /4"),
    ("household_size", {HIES_YEARS: ("familysize", "FamilySize")}, True, True, ""),
    ("eq_adult", {HIES_YEARS: ("familysize", "Eq_Adult")}, True, True, "equivalent adults (Geostat scale)"),
    ("inc_hired_employment", {HIES_YEARS: ("tblincomes", "ShemDaq")}, True, True, "household, monthly avg, GEL"),
    ("inc_self_employment", {HIES_YEARS: ("tblincomes", "ShemTviTdasaqm")}, True, True, ""),
    ("inc_agriculture_sales", {HIES_YEARS: ("tblincomes", "Shem_Sof")}, True, True, ""),
    ("inc_property", {HIES_YEARS: ("tblincomes", "Qonebidan")}, True, True, ""),
    ("inc_pensions_benefits", {HIES_YEARS: ("tblincomes", "PensStipDaxm")}, True, True, ""),
    ("inc_remittances_abroad", {HIES_YEARS: ("tblincomes", "Ucxoetidan")}, True, True, ""),
    ("inc_gifts", {HIES_YEARS: ("tblincomes", "Axloblebisagan")}, True, True, ""),
    ("inc_cash_and_transfers", {HIES_YEARS: ("tblincomes", "ShemosavaliDaTransf")}, True, True, "= sum of items 1-7"),
    ("inflow_property_disposal", {HIES_YEARS: ("tblincomes", "QonebisGayidvit")}, True, True, "not income"),
    ("inflow_borrowing_savings", {HIES_YEARS: ("tblincomes", "SesxAnDanazog")}, True, True, "not income"),
    ("inflow_other_cash", {HIES_YEARS: ("tblincomes", "SxvaFuladiSaxsrebi")}, True, True, "not income"),
    ("inc_non_cash", {HIES_YEARS: ("tblincomes", "ArafuladiMoxmareba")}, True, True, ""),
    ("inc_total", {HIES_YEARS: ("tblincomes", "Shemosavalisul")}, True, True, "total income, monthly avg"),
    ("inflows_cash_total", {HIES_YEARS: ("tblincomes", "Fuladisaxsrebi")}, True, True, ""),
    ("inflows_total", {HIES_YEARS: ("tblincomes", "Saxsrebi_Sul")}, True, True, "cash + non-cash inflows"),
    # members
    ("member_no", {HIES_YEARS: ("tblshinda02", "MemberNo")}, True, True, ""),
    ("relationship", {HIES_YEARS: ("tblshinda02", "Relations")}, True, True, "1 = head"),
    ("sex", {HIES_YEARS: ("tblshinda02", "Gender")}, True, True, "1=female 2=male"),
    ("age", {HIES_YEARS: ("tblshinda02", "Age")}, True, True, ""),
    ("education_code", {HIES_YEARS: ("tblshinda02", "Education")}, True, True, "13 levels; missing for children"),
]


def expand(dataset: str, spec: list) -> pd.DataFrame:
    rows = []
    for harm, years_map, within, across, note in spec:
        for years, (table, var) in years_map.items():
            for y in years:
                rows.append(dict(dataset=dataset, harmonised=harm, year=y, period=period_of(y) if dataset == "lfs"
                                 else "2017_2025", source_table=table, source_variable=var,
                                 compatible_within_period=within,
                                 compatible_across_periods=across if dataset == "lfs" else within,
                                 note=note))
    return pd.DataFrame(rows)


def mapping() -> pd.DataFrame:
    return pd.concat([expand("lfs", LFS_MAP), expand("hies", HIES_MAP)], ignore_index=True)


def resolve(dataset: str, year: int) -> dict[str, tuple[str, str]]:
    """{harmonised: (table, original variable)} for one dataset-year."""
    m = mapping()
    m = m[(m.dataset == dataset) & (m.year == year)]
    return {r.harmonised: (r.source_table, r.source_variable) for r in m.itertuples()}


def validate(m: pd.DataFrame) -> pd.DataFrame:
    """Attach the original label/type from the raw inventory; fail on any invented variable."""
    prof = pd.read_csv(PROFILE, dtype=str, keep_default_na=False)
    prof["year"] = prof.year.astype(int)
    prof = prof[["dataset", "year", "canonical_table", "variable", "label", "spss_type", "value_labels"]]
    out = m.merge(prof, left_on=["dataset", "year", "source_table", "source_variable"],
                  right_on=["dataset", "year", "canonical_table", "variable"], how="left")
    missing = out[out.variable.isna()]
    if len(missing):
        raise ValueError("mapping refers to variables absent from raw data:\n"
                         + missing[["dataset", "year", "source_table", "source_variable"]].to_string())
    return out.drop(columns=["canonical_table", "variable"]).rename(
        columns={"label": "original_label", "spss_type": "original_type", "value_labels": "original_value_labels"})


def run() -> pd.DataFrame:
    m = validate(mapping())
    m.to_csv(OUT, index=False)
    return m


if __name__ == "__main__":
    m = run()
    print(f"{len(m)} mapping rows, {m.harmonised.nunique()} harmonised variables → {OUT}")
    print(m.groupby(["dataset", "harmonised"]).source_variable.agg(lambda s: " | ".join(dict.fromkeys(s)))
          .to_string())
