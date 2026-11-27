"""
Stage 01 — Technical profile of every raw SPSS table (read-only).

For every .sav table of every downloaded year it records
  table level   : format, file size, encoding, rows, columns, missing cells,
                  full-row duplicates, key duplicates, detected data year(s),
                  label-encoding problems, Georgian characters found
  variable level: label, SPSS type/measure, pandas dtype, n / missing / unique,
                  min / max, frequent values, value labels, user-missing
                  definitions, codes present in data but not in value labels

Outputs (derived files — raw data are never modified):
  reports/stage01_table_profile.csv
  reports/stage01_variable_profile.csv

Usage:  cd src && python data_profile.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

import data_collection as dc

REPORTS = dc.PROJECT_ROOT / "reports"
TABLE_OUT = REPORTS / "stage01_table_profile.csv"
VAR_OUT = REPORTS / "stage01_variable_profile.csv"

GEORGIAN = re.compile(r"[Ⴀ-ჿᲐ-Ჿⴀ-⴯]")
# Typical mojibake produced when UTF-8 Georgian/Latin text is decoded as cp1252/latin-1
MOJIBAKE = re.compile(r"�|áƒ|Ã.|Ð.|â€")


def canonical_table(name: str) -> str:
    """Harmonise table names that change spelling between years.
    e.g. LFS_ECSTAT_ENG_2020 -> ecstat, LFS_Demograpics -> demographic,
         LFS_Family_size -> familysize, LFS_Personal_weights -> personal_weights"""
    n = name.lower()
    n = re.sub(r"^lfs_", "", n)
    n = re.sub(r"_eng_\d{4}$", "", n)
    n = {"demograpics": "demographic", "family_size": "familysize"}.get(n, n)
    return n


# Natural keys (used only to *count* duplicates, never to drop rows)
KEYS = {
    "lfs": {"id": ["UID"], "familysize": ["UID"], "demographic": ["UID", "MemberNo"],
            "ecstat": ["UID", "MemberNo"], "personal_weights": ["UID", "MemberNo"],
            "deadmen": ["UID", "MemberNo"]},
    "hies": {"123": ["QuartNo"], "familysize": ["UID"], "sysschedule": ["UID"],
             "tblexpenditures": ["UID"], "tblincomes": ["UID"], "tblshinda01": ["UID"],
             "tblshinda02": ["UID", "MemberNo"], "deadmen": ["UID", "MemberNo"],
             "tblavgprices": ["QuartNo", "ProductCode"],
             "tblconsumption": ["UID", "ProductCode"], "conspurch_03": ["UID", "ProductCode"],
             "tblshinda01_animals": ["UID", "Animal"], "tblshinda01_comfort": ["UID", "Comfort"],
             "tblshinda01_cultures": ["UID", "Culture"], "tblshinda01_lands": ["UID", "LandNo"],
             "tblshinda01_subjects": ["UID", "Subject"],
             "tblshinda04": ["UID", "MonthYear", "TableNo", "ItemNo", "Opt"],
             "tblshinda04_1": ["UID", "MonthYear", "TableNo", "ItemNo"]},
}

LFS_FIRST_QUARTER_2017 = 83   # Geostat quarter code: 83 = 2017 Q1 (LFS_ID codebook)


def quarter_to_year(q: float) -> int:
    return 2017 + (int(q) - LFS_FIRST_QUARTER_2017) // 4


def detect_years(df: pd.DataFrame, group: str, hies_quarters: dict | None) -> set[int]:
    """Data year(s) as recorded *inside* the table."""
    if "Year" in df.columns:
        return set(df["Year"].dropna().astype(int).unique())
    if "YarNo" in df.columns:          # HIES 123 lookup table covers all past quarters
        return set()
    for qcol in ("QuarterNo", "QuartNo"):
        if qcol in df.columns:
            qs = df[qcol].dropna().unique()
            if group == "hies" and hies_quarters:
                return {hies_quarters.get(int(q), -1) for q in qs}
            return {quarter_to_year(q) for q in qs}
    if "MonthYear" in df.columns:
        return set(pd.to_datetime(df["MonthYear"], errors="coerce").dt.year.dropna().astype(int).unique())
    return set()


def _json(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str) if x else ""


def profile_variable(s: pd.Series, var: str, meta) -> dict:
    labels = meta.variable_value_labels.get(var, {})
    n_missing = int(s.isna().sum())
    nn = s.dropna()
    rec = dict(variable=var, label=meta.column_names_to_labels.get(var) or "",
               spss_type=meta.readstat_variable_types.get(var, ""),
               spss_measure=meta.variable_measure.get(var, ""),
               spss_format=meta.original_variable_types.get(var, ""),
               pandas_dtype=str(s.dtype), n=len(s), n_missing=n_missing,
               pct_missing=round(100 * n_missing / len(s), 2) if len(s) else 0.0,
               n_unique=int(nn.nunique()), min="", max="", top_values="",
               value_labels=_json(labels),
               user_missing=_json(meta.missing_user_values.get(var) or meta.missing_ranges.get(var)),
               undocumented_codes="", label_encoding_issue=False)
    if len(nn):
        try:
            rec["min"], rec["max"] = str(nn.min()), str(nn.max())
        except TypeError:
            pass
        if rec["n_unique"] <= 15:
            rec["top_values"] = _json({str(k): int(v) for k, v in nn.value_counts().items()})
    if labels and pd.api.types.is_numeric_dtype(s):
        extra = sorted(set(nn.unique()) - set(labels))
        if extra and len(extra) <= 20:          # >20 means a continuous var with a few labelled codes
            rec["undocumented_codes"] = ",".join(f"{v:g}" for v in extra)
    text = rec["label"] + " " + " ".join(map(str, labels.values()))
    rec["label_encoding_issue"] = bool(MOJIBAKE.search(text))
    return rec


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables, variables = [], []
    for src in dc.SOURCES:
        if src.fmt != "zip":
            continue
        folder = dc.RAW_DIR / dc.FOLDERS[src.group] / "extracted" / str(src.year)
        if not folder.exists():
            raise FileNotFoundError(f"{folder} — run inspect_raw.py first (extracts the ZIPs)")
        savs = sorted(folder.rglob("*.sav"))
        hq = None
        if src.group == "hies":       # quarter -> year lookup shipped with each HIES year
            qdf, _ = pyreadstat.read_sav(str(next(p for p in savs if p.stem == "123")))
            hq = dict(zip(qdf.QuartNo.astype(int), qdf.YarNo.astype(int)))
        for sav in savs:
            df, meta = pyreadstat.read_sav(str(sav))
            ct = canonical_table(sav.stem)
            key = [k for k in KEYS[src.group].get(ct, []) if k in df.columns]
            years = detect_years(df, src.group, hq)
            vrecs = [profile_variable(df[c], c, meta) for c in df.columns]
            obj_text = " ".join(df[c].dropna().astype(str).str.cat(sep=" ")[:200000]
                                for c in df.columns if df[c].dtype == object)
            tables.append(dict(
                dataset=src.group, year=src.year, table=sav.stem, canonical_table=ct,
                file=str(sav.relative_to(dc.PROJECT_ROOT)), file_bytes=sav.stat().st_size,
                format=dc.detect_format(sav), encoding=meta.file_encoding,
                n_rows=len(df), n_cols=df.shape[1],
                n_missing_cells=int(df.isna().sum().sum()),
                pct_missing_cells=round(100 * df.isna().to_numpy().mean(), 2) if df.size else 0.0,
                n_full_row_duplicates=int(df.duplicated().sum()),
                key="+".join(key), n_key_duplicates=int(df.duplicated(key).sum()) if key else "",
                years_in_data=",".join(map(str, sorted(years))),
                year_ok=(years == {src.year}) if years else "",
                n_label_encoding_issues=sum(v["label_encoding_issue"] for v in vrecs),
                n_georgian_chars=len(GEORGIAN.findall(obj_text + " ".join(v["label"] + v["value_labels"]
                                                                         for v in vrecs))),
            ))
            for v in vrecs:
                variables.append(dict(dataset=src.group, year=src.year, table=sav.stem,
                                      canonical_table=ct, **v))
            print(f"{src.group} {src.year} {sav.stem:28} {len(df):>8} rows  dup={tables[-1]['n_key_duplicates']}"
                  f"  years={tables[-1]['years_in_data']}")
    t, v = pd.DataFrame(tables), pd.DataFrame(variables)
    t.to_csv(TABLE_OUT, index=False)
    v.to_csv(VAR_OUT, index=False)
    return t, v


if __name__ == "__main__":
    run()
