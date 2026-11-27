"""
Stage 01 — Data Dictionaries (one per dataset), generated from the raw files.

Every entry is derived from the SPSS metadata / data profile — nothing is typed in by
hand except the short verified notes in KEY_NOTES (each one is backed by a check in
reports/stage01_integrity_checks.csv or by the official codebook).

Outputs (reports/data_dictionary/):
  lfs_data_dictionary.csv                  one row per (table, variable), all years
  hies_data_dictionary.csv                 idem
  official_aggregates_dictionary.csv       one row per Excel sheet
Columns: Variable_Name, Table, Meaning, Data_Type, Possible_Values, Missing_Values,
         Years_Present, Cross_Year_Status, Unit_or_Level, Source, Notes

Requires: data_profile.py and data_integrity.py to have run.
"""
from __future__ import annotations

import json
import re

import pandas as pd

import data_collection as dc
from data_profile import VAR_OUT
from data_integrity import XYEAR_OUT

OUT_DIR = dc.PROJECT_ROOT / "reports" / "data_dictionary"

SOURCE = {
    "lfs": f"Geostat, Labour Force Survey databases 2017-2025 — {dc.LFS_DB_PAGE}",
    "hies": f"Geostat, Household Incomes and Expenditures Survey databases 2017-2025 — {dc.HIES_DB_PAGE}",
}

# Codes whose label marks a non-substantive answer (treated as missing in later stages)
SPECIAL = re.compile(r"refus|do not know|don.t know|not applicable|cannot define|no answer", re.I)

# Level of the observation, by table (individual vs household is essential: never mix them)
LEVEL = {
    "lfs": {"id": "household", "familysize": "household", "demographic": "person",
            "ecstat": "person (15+)", "personal_weights": "person (15+)", "deadmen": "person (deceased)"},
    "hies": {"123": "quarter lookup", "sysschedule": "household", "familysize": "household",
             "tblincomes": "household", "tblexpenditures": "household", "tblshinda01": "household",
             "tblshinda02": "person", "deadmen": "person (deceased)", "tblavgprices": "quarter x product",
             "tblconsumption": "household x product", "conspurch_03": "household x product",
             "tblshinda03_1": "household diary line", "tblshinda04": "household x month x item",
             "tblshinda04_1": "household x month x item"},
}

# Verified notes for the variables the study depends on
KEY_NOTES = {
    ("lfs", "Sex"): "GENDER: 1=Female, 2=Male in every year (label wording Woman/Women/Female varies; codes identical).",
    ("hies", "Gender"): "GENDER: 1=Female, 2=Male in every year.",
    ("lfs", "B17_B18_Net_earnings"): "INCOME (individual): net earnings from MAIN job, LAST WORKED MONTH, GEL, "
        "EMPLOYEES only. 9 bands 2017-2019. Exact amount asked (B17) but published only as band. "
        "88=refused (labelled from 2018). Code 10 in 2017 is undocumented. Missing share in this dictionary is over ALL persons 15+ (question asked to employees only); non-response among employees: reports/stage00_lfs_earnings_coverage.csv.",
    ("lfs", "B24_B25_Net_earnings"): "INCOME (individual): net earnings from MAIN job, LAST WORKED MONTH, GEL, "
        "EMPLOYEES only. 12 bands 2020-2025 (scheme differs from 2017-2019). 0=no earnings last month, "
        "88=refused; from 2023 refusals are system-missing instead of 88. Missing share in this dictionary is over ALL persons 15+ (question asked to employees only); non-response among employees: reports/stage00_lfs_earnings_coverage.csv.",
    ("lfs", "P_Weights"): "WEIGHT: personal weight, persons 15+. Quarterly weights sum to the 15+ population; "
        "annual estimates = weight / 4 (Geostat codebook). Sum/4 = 3.01 M (2017), 2.96 M (2025).",
    ("hies", "Weights"): "WEIGHT: HOUSEHOLD sampling weight (no person weights in public 2017-2025 HIES). "
        "Each quarter sums to all households (1.06 M in 2017, 1.16 M in 2025); annual = weight / 4.",
    ("hies", "ShemDaq"): "INCOME (HOUSEHOLD, not individual): income from hired employment of all members, "
        "average monthly, GEL.",
    ("hies", "Shemosavalisul"): "INCOME (HOUSEHOLD): total income, average monthly, GEL "
        "(cash income + transfers + non-cash). Must not be used as individual income.",
    ("hies", "ShemosavaliDaTransf"): "INCOME (HOUSEHOLD): cash income and transfers = sum of items 1-7, monthly avg, GEL.",
    ("lfs", "UID"): "ID of household within ONE quarter; changes at each interview. Stored as text in some 2018 tables.",
    ("lfs", "DiaryID"): "ID of household across all its interviews (2020+): use for clustering standard errors.",
    ("hies", "UID"): "ID of household within a quarter.",
    ("hies", "DiaryID"): "ID of household across the whole survey period.",
    ("lfs", "Occupation_converted"): "ISCO major group. Label says ISCO-88 in 2017-2019; code 0 = 'Armed Forces' "
        "early, 'Other' later.",
    ("lfs", "Region"): "Region code; 35 (Racha-Lechkhumi & Kvemo Svaneti) appears from 2019 only.",
    ("hies", "MonthYear", "tblshinda04"): " month the operation refers to = 1-3 months BEFORE the interview "
        "(retrospective quarter), so it can fall in the previous calendar year.",
}
KEY_NOTES[("hies", "MonthYear", "tblshinda04_1")] = KEY_NOTES[("hies", "MonthYear", "tblshinda04")]


def _fmt_labels(js: str) -> str:
    if not js:
        return ""
    d = json.loads(js)
    def k(x):
        try:
            return float(x)
        except ValueError:
            return float("inf")
    return "; ".join(f"{float(c):g}={lab}" if c.replace('.', '', 1).isdigit() else f"{c}={lab}"
                     for c, lab in sorted(d.items(), key=lambda kv: k(kv[0])))


def _range(series) -> str:
    vals = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    return f"{vals.min():g} – {vals.max():g}" if len(vals) else ""


def build_survey(ds: str, v: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    xi = x[x.dataset == ds].set_index(["canonical_table", "variable"])
    for (tab, var), g in v[v.dataset == ds].groupby(["canonical_table", "variable"], sort=False):
        g = g.sort_values("year")
        last = g.iloc[-1]
        labels = last.value_labels
        # possible values
        if labels:
            possible = _fmt_labels(labels)
            rng = _range(g["min"].tolist() + g["max"].tolist())
            if last.n_unique and int(last.n_unique) > 40:
                possible = f"numeric {rng}; labelled codes: {possible}"
        elif last.spss_type == "string" or last.pandas_dtype in ("object", "str"):
            possible = f"text / date, up to {max(map(int, g.n_unique))} distinct values"
        else:
            possible = f"numeric {_range(g['min'].tolist() + g['max'].tolist())}"
        undoc = sorted({c for c in g.undocumented_codes if c})
        if undoc:
            possible += f"  [codes in data without label: {' | '.join(f'{y}: {c}' for y, c in zip(g.year, g.undocumented_codes) if c)}]"
        # missing values
        miss = pd.to_numeric(g.pct_missing)
        special = []
        if labels:
            special = [f"{float(c):g}={l}" for c, l in json.loads(labels).items()
                       if SPECIAL.search(l) and c.replace('.', '', 1).isdigit()]
        missing = (f"system-missing {miss.min():.1f}%" if miss.min() == miss.max()
                   else f"system-missing {miss.min():.1f}–{miss.max():.1f}% by year")
        if special:
            missing += f"; non-response codes: {', '.join(special)}"
        um = {u for u in g.user_missing if u}
        if um:
            missing += f"; SPSS user-missing: {' | '.join(sorted(um))}"
        types = sorted({f"{t} ({m})" for t, m in zip(g.spss_type, g.spss_measure)})
        dtypes = sorted(set(g.pandas_dtype))
        status = xi.loc[(tab, var), "status"] if (tab, var) in xi.index else ""
        rows.append(dict(
            Variable_Name=var, Table=tab,
            Meaning=last.label or "(no label in file)",
            Data_Type=f"SPSS {' / '.join(types)}; pandas {' / '.join(dtypes)}",
            Possible_Values=possible, Missing_Values=missing,
            Years_Present=f"{g.year.iloc[0]}–{g.year.iloc[-1]}" if len(g) == int(g.year.iloc[-1]) - int(g.year.iloc[0]) + 1
            else ",".join(g.year),
            Cross_Year_Status=status,
            Unit_or_Level=LEVEL[ds].get(tab, ""),
            Source=SOURCE[ds],
            Notes=KEY_NOTES.get((ds, var, tab), KEY_NOTES.get((ds, var), "")),
        ))
    return pd.DataFrame(rows)


def build_aggregates() -> pd.DataFrame:
    rows = []
    for src in dc.SOURCES:
        if src.group != "agg":
            continue
        path = dc.local_path(src)
        for sheet, df in pd.read_excel(path, sheet_name=None, header=None).items():
            title = next((str(c) for c in df.iloc[:, 0].dropna().astype(str) if c.strip()), "")
            def _years(cells):
                return sorted({int(float(c)) for c in pd.Series(cells).dropna().astype(str)
                               if re.fullmatch(r"(19|20)\d{2}(\.0)?", c.strip())})
            # years sit either across the header rows or down the first column (e.g. CPI)
            years = max(_years(df.dropna(how="all").head(4).values.ravel()),
                        _years(df.iloc[:, 0].values),
                        _years(re.findall(r"\d{4}", f"{sheet} {title}")), key=len)
            unit = re.search(r"\(([^)]*(GEL|%|=100)[^)]*)\)|(GEL|%|2010=100)", title + " " + src.description)
            rows.append(dict(
                Variable_Name=src.key, Table=f"{dc.local_path(src).name} / {sheet}",
                Meaning=title.replace("\n", " ")[:200], Data_Type="Excel table (aggregate statistics)",
                Possible_Values=f"{df.dropna(how='all').shape[0]} rows x {df.dropna(how='all', axis=1).shape[1]} cols",
                Missing_Values="", Years_Present=f"{years[0]}–{years[-1]}" if years else "",
                Cross_Year_Status="", Unit_or_Level=(unit.group(0) if unit else "") + " — national aggregate",
                Source=f"Geostat — {src.page} ({src.url})", Notes=src.description))
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v = pd.read_csv(VAR_OUT, dtype=str, keep_default_na=False)
    x = pd.read_csv(XYEAR_OUT, dtype=str, keep_default_na=False)
    out = {"lfs": build_survey("lfs", v, x), "hies": build_survey("hies", v, x),
           "official_aggregates": build_aggregates()}
    for name, df in out.items():
        df.to_csv(OUT_DIR / f"{name}_data_dictionary.csv", index=False)
        print(f"{name}: {len(df)} entries")
    return out


if __name__ == "__main__":
    run()
