"""
Stage 06 — Data layer of the dashboard (no Streamlit code here, so it can be unit-tested).

All numbers are computed live from the project database (database/income_data.db) with the
same rules as Stage 03-04:
  * LFS employees with a valid earnings band; real 2025 GEL; weighted
  * the two methodological periods (2017-2019, 2020-2025) are NEVER mixed
  * several years pooled -> weight_annual / number of selected years
  * mean = band point estimate; median / quartiles interpolated within bands
  * gap % = (men - women) / men
  * small cells: < 30 women or men -> suppressed, < 100 -> flagged "low n"
Stage 04-05 results are read from reports/ (they are not recomputed here).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import eda  # noqa: E402

REPORTS = ROOT / "reports"
PERIODS = {"2017_2019": "2017–2019", "2020_2025": "2020–2025"}
MIN_N, LOW_N = eda.MIN_N, eda.LOW_N
SOURCE = ("წყარო: საქსტატი, სამუშაო ძალის კვლევის მიკრომონაცემები; დაქირავებულები, ხელზე ასაღები თვიური ხელფასი "
          "ძირითად სამუშაოზე (ინტერვალის შეფასება), 2025 წლის ფასებში, შეწონილი.")


@lru_cache(maxsize=1)
def employees() -> pd.DataFrame:
    """Both periods, one row per employee-interview with a valid earnings band."""
    d = eda.load()
    frames = []
    for p in PERIODS:
        x = d[p].copy()
        defl = x.income_amount_real_2025 / x.income_amount
        x["lo_real"] = x.income_lower * defl
        x["hi_real"] = x.income_upper.fillna(np.inf) * defl
        x["sex"] = x.gender.map({"female": "ქალი", "male": "კაცი"})
        frames.append(x)
    return pd.concat(frames, ignore_index=True)


def options(period: str) -> dict:
    d = employees()
    d = d[d.period == period]
    return dict(years=sorted(d.survey_year.unique().tolist()),
                ages=[a for a in ["15-17"] + eda.AGE_LABELS if a in set(d.age_group)],
                regions=sorted(d.region_eda.dropna().unique().tolist()),
                education=[e for e in eda.EDU_ORDER if e in set(d.education_eda)],
                occupation=[o for o in eda.ISCO1.values() if o in set(d.occupation_label)])


def apply_filters(period: str, years=None, ages=None, regions=None, education=None, occupation=None) -> pd.DataFrame:
    """Filter ONE period; pooled weight = weight_annual / number of selected years."""
    if period not in PERIODS:
        raise ValueError(f"unknown period {period}")
    d = employees()
    d = d[d.period == period]
    for col, sel in (("survey_year", years), ("age_group", ages), ("region_eda", regions),
                     ("education_eda", education), ("occupation_label", occupation)):
        if sel:
            d = d[d[col].isin(sel)]
    d = d.copy()
    d["w"] = d.weight_annual / max(d.survey_year.nunique(), 1)
    return d


def _stats(g: pd.DataFrame, w: str = "w") -> dict:
    if len(g) == 0:
        return dict(n=0, population=0, mean=np.nan, median=np.nan, p25=np.nan, p75=np.nan)
    q = {k: eda.interval_quantile(g.lo_real, g.hi_real, g.pareto_alpha, g[w], v, 45)
         for k, v in (("p25", .25), ("median", .5), ("p75", .75))}
    return dict(n=len(g), population=float(g[w].sum()), mean=eda.wmean(g.income_amount_real_2025, g[w]), **q)


def gender_summary(d: pd.DataFrame) -> dict:
    f, m = _stats(d[d.gender == "female"]), _stats(d[d.gender == "male"])
    rel = eda.reliability(f["n"], m["n"])
    out = dict(female=f, male=m, reliability=rel,
               gap_mean=eda.gap_pct(m["mean"], f["mean"]), gap_median=eda.gap_pct(m["median"], f["median"]))
    if rel == "suppressed":
        out["gap_mean"] = out["gap_median"] = np.nan
    return out


def by_group(d: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(col):
        s = gender_summary(g)
        rows.append(dict(group=k, n_female=s["female"]["n"], n_male=s["male"]["n"], reliability=s["reliability"],
                         female_mean=s["female"]["mean"], male_mean=s["male"]["mean"],
                         female_median=s["female"]["median"], male_median=s["male"]["median"],
                         gap_mean=s["gap_mean"], gap_median=s["gap_median"]))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    for c in ("female_mean", "male_mean", "female_median", "male_median"):
        t.loc[t.reliability == "suppressed", c] = np.nan
    if order:
        t["_o"] = t.group.map({k: i for i, k in enumerate(order)})
        t = t.sort_values("_o").drop(columns="_o")
    return t.reset_index(drop=True)


def by_year(d: pd.DataFrame) -> pd.DataFrame:
    """Per survey year, with that year's annual weights (not pooled)."""
    rows = []
    for y, g in d.groupby("survey_year"):
        s = gender_summary(g.assign(w=g.weight_annual))
        rows.append(dict(year=int(y), female_mean=s["female"]["mean"], male_mean=s["male"]["mean"],
                         female_median=s["female"]["median"], male_median=s["male"]["median"],
                         gap_mean=s["gap_mean"], gap_median=s["gap_median"], reliability=s["reliability"]))
    return pd.DataFrame(rows)


def band_shares(d: pd.DataFrame) -> pd.DataFrame:
    labels = {1: "≤100", 2: "101–200", 3: "201–400", 4: "401–600", 5: "601–800", 6: "801–1000",
              7: "1001–1500", 8: "1501–2000", 9: ">2000"}
    t = d.groupby(["sex", "band_code_common"]).w.sum().unstack(0).fillna(0)
    t = t / t.sum()
    t.index = [labels[int(i)] for i in t.index]
    return t


def report_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / f"{name}.csv")


def model_metadata() -> dict:
    return json.loads((ROOT / "models" / "model_metadata.json").read_text())


# labels shown in the UI (Georgian); data keep the English codes
EDU_KA = {"Below secondary": "საშუალოზე დაბალი", "Secondary general": "ზოგადი საშუალო", "Vocational": "პროფესიული",
          "Higher professional": "უმაღლესი პროფესიული", "Bachelor": "ბაკალავრი", "Master": "მაგისტრი",
          "Doctor": "დოქტორი"}
OCC_KA = {"Armed forces / other": "შეიარაღებული ძალები / სხვა", "Managers": "მენეჯერები",
          "Professionals": "პროფესიონალები", "Technicians & assoc. prof.": "ტექნიკოსები",
          "Clerical support": "საოფისე პერსონალი", "Service & sales": "მომსახურება და ვაჭრობა",
          "Skilled agricultural": "სოფლის მეურნეობა", "Craft & trades": "ხელოსნები",
          "Plant & machine operators": "დანადგარების ოპერატორები", "Elementary occupations": "ელემენტარული პროფესიები"}
REGION_KA = {"Tbilisi": "თბილისი", "Adjara A/R": "აჭარა", "Guria": "გურია", "Imereti": "იმერეთი",
             "Kakheti": "კახეთი", "Mtskheta-Mtianeti": "მცხეთა-მთიანეთი",
             "Racha-Lechkhumi and Kvemo Svaneti": "რაჭა-ლეჩხუმი და ქვ. სვანეთი",
             "Samegrelo-Zemo Svaneti": "სამეგრელო-ზემო სვანეთი", "Samtskhe-Javakheti": "სამცხე-ჯავახეთი",
             "Kvemo Kartli": "ქვემო ქართლი", "Shida Kartli": "შიდა ქართლი",
             "Imereti + Racha-Lechkhumi & Kvemo Svaneti": "იმერეთი + რაჭა-ლეჩხუმი"}
