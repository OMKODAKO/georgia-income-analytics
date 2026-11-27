"""
Stage 00 — Feasibility metrics for individual earnings by sex in the LFS.

For each year 2017-2025 computes, for *employees* in their main job:
  * unweighted n by sex
  * share refusing the earnings question (code 88) / no earnings (code 0)
  * weighted share in each earnings band by sex
  * weighted share in the open top band ("More than 2000 GEL")
  * the weighted median earnings band by sex (a coarse first signal only)

Output: reports/stage00_lfs_earnings_coverage.csv
        reports/stage00_lfs_band_distribution.csv

This is NOT the pay-gap analysis (Stage 04); it only tests whether the data
can support it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

import data_collection as dc

EXTRACT = dc.RAW_DIR / dc.FOLDERS["lfs"] / "extracted"
REPORTS = dc.PROJECT_ROOT / "reports"

# Per-period names of the variables we need (taken from the variable inventory).
SPEC = {
    "2017_2019": dict(ecstat="LFS_ECSTAT", earn="B17_B18_Net_earnings", status="B7_Status",
                      weights_table="LFS_Personal_[Ww]eights", top_code=9),
    "2020_2025": dict(ecstat="LFS_ECSTAT_ENG_*", earn="B24_B25_Net_earnings", status="Status",
                      weights_table=None, top_code=12),
}
REFUSED = 88


def period(year: int) -> str:
    return "2017_2019" if year <= 2019 else "2020_2025"


def find(year: int, pattern: str) -> Path:
    hits = sorted((EXTRACT / str(year)).rglob(f"{pattern}.sav"))
    if not hits:
        raise FileNotFoundError(f"{pattern}.sav for {year}")
    return hits[0]


def load_employees(year: int) -> tuple[pd.DataFrame, dict]:
    s = SPEC[period(year)]
    df, meta = pyreadstat.read_sav(str(find(year, s["ecstat"])))
    if s["weights_table"]:
        w, _ = pyreadstat.read_sav(str(find(year, s["weights_table"])))
        # 2018 stores UID as text in ECSTAT but numeric in the weights table: harmonise
        for t in (df, w):
            t["UID"] = pd.to_numeric(t["UID"], errors="raise").astype("int64")
        df = df.merge(w[["UID", "MemberNo", "P_Weights"]], on=["UID", "MemberNo"], how="left",
                      validate="one_to_one")
    # Employees in main job (ICSE code 1 in both periods)
    emp = df[(df["Employed"] == 1) & (df[s["status"]] == 1)].copy()
    emp = emp.rename(columns={s["earn"]: "earn_band"})
    emp["w"] = emp["P_Weights"] / 4  # annual estimate = quarterly weights / 4 (Geostat codebook)
    labels = {int(k): v for k, v in meta.variable_value_labels.get(s["earn"], {}).items()}
    return emp[["Sex", "Age", "earn_band", "w"]], labels


def weighted_median_band(bands: pd.Series, w: pd.Series) -> float:
    order = np.argsort(bands.values)
    b, ww = bands.values[order], w.values[order]
    cum = np.cumsum(ww) / ww.sum()
    return float(b[np.searchsorted(cum, 0.5)])


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    cov, dist = [], []
    for year in range(2017, 2026):
        emp, labels = load_employees(year)
        top = SPEC[period(year)]["top_code"]
        for sex, g in emp.groupby("Sex"):
            sex_lbl = {1: "Female", 2: "Male"}[int(sex)]
            valid = g[(g.earn_band >= 1) & (g.earn_band <= top)]
            # codes present in data but absent from value labels and questionnaire (e.g. "10" in 2017)
            undocumented = g[~g.earn_band.isin(list(labels) + [REFUSED, 0]) & g.earn_band.notna()]
            cov.append(dict(
                year=year, sex=sex_lbl, n_employees=len(g),
                weighted_employees_thousand=round(g.w.sum() / 1000, 1),
                # from 2023 refusals are stored as system-missing instead of code 88
                pct_missing=round(100 * g.earn_band.isna().mean(), 2),
                pct_refused=round(100 * (g.earn_band == REFUSED).mean(), 2),
                pct_no_earnings_code0=round(100 * (g.earn_band == 0).mean(), 2),
                n_undocumented_code=len(undocumented),
                undocumented_codes=",".join(map(str, sorted(undocumented.earn_band.unique().astype(int)))),
                n_valid_band=len(valid),
                pct_top_band_weighted=round(100 * valid.w[valid.earn_band == top].sum() / valid.w.sum(), 2),
                median_band=labels.get(int(weighted_median_band(valid.earn_band, valid.w)), "?"),
            ))
            shares = valid.groupby("earn_band").w.sum() / valid.w.sum()
            for b, sh in shares.items():
                dist.append(dict(year=year, sex=sex_lbl, band_code=int(b),
                                 band=labels.get(int(b), "UNDOCUMENTED"), weighted_share=round(sh, 4)))
    cov_df, dist_df = pd.DataFrame(cov), pd.DataFrame(dist)
    cov_df.to_csv(REPORTS / "stage00_lfs_earnings_coverage.csv", index=False)
    dist_df.to_csv(REPORTS / "stage00_lfs_band_distribution.csv", index=False)
    return cov_df, dist_df


if __name__ == "__main__":
    c, _ = run()
    pd.set_option("display.width", 250)
    print(c.to_string(index=False))
