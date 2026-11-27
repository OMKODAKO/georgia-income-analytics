"""
Stage 02 — Data validation and outlier investigation (no data are changed or removed).

1. data quality summary per layer and year  -> reports/data_quality_summary.csv
   rows, columns, primary-key uniqueness, missing % of key variables, invalid
   values, flags, outliers
2. outliers — three methods, each FLAGGED, never deleted
     IQR        value < Q1 - 1.5*IQR  or  > Q3 + 1.5*IQR   (income on log scale)
     Z-score    |z| > 3                                    (income on log scale)
     domain     substantive limits (e.g. > 80 usual hours/week)
   -> data/processed/outlier_flags.csv.gz  (record id, variable, method, value)
   -> reports/outlier_summary.csv
3. investigation: is an extreme value real or an error?
   -> reports/outlier_investigation.csv  + distribution plots in reports/figures/

Earnings in LFS are bands, so they have no numeric outliers; instead the
plausibility of band x hours combinations is checked.

Usage:  cd src && python data_validation.py      (requires data_cleaning.py output)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import data_collection as dc

PROCESSED = dc.PROJECT_ROOT / "data" / "processed"
FINAL = dc.PROJECT_ROOT / "data" / "final"
REPORTS = dc.PROJECT_ROOT / "reports"
FIG = REPORTS / "figures"


def load():
    lfs = {p: pd.read_csv(FINAL / f"lfs_analytical_{p}.csv.gz", low_memory=False)
           for p in ("2017_2019", "2020_2025")}
    hh = pd.read_csv(FINAL / "hies_households_analytical_2017_2025.csv.gz", low_memory=False)
    mem = pd.concat([pd.read_csv(PROCESSED / f"hies_members_clean_{y}.csv.gz") for y in range(2017, 2026)],
                    ignore_index=True)
    return lfs, hh, mem


# ---------------------------------------------------------------- outliers ---
def iqr_bounds(x: pd.Series) -> tuple[float, float]:
    q1, q3 = x.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def flag_outliers(df: pd.DataFrame, id_col: str, var: str, group: str, dataset: str,
                  log_scale=False, domain=None, positive_only=False) -> pd.DataFrame:
    """Return long table of outlier flags for `var`, computed within each `group` (year)."""
    out = []
    for g, d in df.groupby(group):
        x = d[var].astype(float)
        if positive_only:
            x = x.where(x > 0)
        xs = np.log(x) if log_scale else x
        xs = xs.dropna()
        if xs.empty:
            continue
        lo, hi = iqr_bounds(xs)
        z = (xs - xs.mean()) / xs.std()
        for method, mask in [("iqr", (xs < lo) | (xs > hi)), ("zscore", z.abs() > 3)]:
            idx = mask[mask].index
            out.append(pd.DataFrame({"dataset": dataset, "group": g, "record_id": d.loc[idx, id_col],
                                     "variable": var, "method": method, "value": d.loc[idx, var],
                                     "direction": np.where(xs[idx] > xs.median(), "high", "low")}))
        if domain is not None:
            idx = d.index[domain(d[var])]
            out.append(pd.DataFrame({"dataset": dataset, "group": g, "record_id": d.loc[idx, id_col],
                                     "variable": var, "method": "domain", "value": d.loc[idx, var],
                                     "direction": "high"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def outliers(lfs, hh):
    flags = []
    for p, d in lfs.items():
        emp = d[d.labour_status == "employed"]
        flags.append(flag_outliers(emp, "individual_id", "hours_usual", "year", f"lfs_{p}",
                                   domain=lambda s: s > 80))
        flags.append(flag_outliers(d, "individual_id", "age", "year", f"lfs_{p}"))
        flags.append(flag_outliers(d, "individual_id", "household_size", "year", f"lfs_{p}"))
        flags.append(flag_outliers(d, "individual_id", "weight_quarter", "year", f"lfs_{p}", log_scale=True))
    for v in ["inc_total_per_capita", "inc_total"]:
        flags.append(flag_outliers(hh, "household_id", v, "year", "hies", log_scale=True, positive_only=True))
    flags.append(flag_outliers(hh, "household_id", "household_size", "year", "hies"))
    f = pd.concat(flags, ignore_index=True)
    f.to_csv(PROCESSED / "outlier_flags.csv.gz", index=False, compression="gzip")
    summ = f.groupby(["dataset", "variable", "method", "direction"]).size().rename("n_flagged").reset_index()
    summ.to_csv(REPORTS / "outlier_summary.csv", index=False)
    return f, summ


def investigate(lfs, hh, flags) -> pd.DataFrame:
    """Evidence on whether extreme values are genuine or errors."""
    rows = []
    for p, d in lfs.items():
        emp = d[d.labour_status == "employed"]
        long = emp[emp.hours_usual > 80]
        rows.append(dict(dataset=f"lfs_{p}", check="hours > 80/week",
                         n=len(long), share_pct=round(100 * len(long) / emp.hours_usual.notna().sum(), 2),
                         evidence=f"median {long.hours_usual.median():g} h; top occupations (ISCO1): "
                                  f"{long.occupation_isco1.value_counts().head(3).to_dict()}; "
                                  f"{(long.hours_usual % 7 == 0).mean():.0%} are multiples of 7 (daily x 7)",
                         verdict="concentrated in services/sales, elementary and operator occupations; mostly 7-day schedules (multiple of 7) -> plausible long hours, not entry errors; kept, flagged"))
        e = d[d.earnings_status == "valid"]
        odd = e[(e.earn_is_top_band) & (e.hours_usual < 15)]
        rows.append(dict(dataset=f"lfs_{p}", check="top earnings band with < 15 h/week",
                         n=len(odd), share_pct=round(100 * len(odd) / len(e), 3),
                         evidence=f"occupations: {odd.occupation_isco1.value_counts().head(3).to_dict()}",
                         verdict="very rare; no evidence of error; kept, to be checked in sensitivity analysis"))
        old = d[d.age > 100]
        rows.append(dict(dataset=f"lfs_{p}", check="age > 100", n=len(old),
                         share_pct=round(100 * len(old) / len(d), 4),
                         evidence=f"ages {sorted(old.age.unique().tolist())}; employed: {(old.labour_status == 'employed').sum()}",
                         verdict="plausible centenarians, none in earnings analysis"))
        w = d.weight_quarter
        rows.append(dict(dataset=f"lfs_{p}", check="weight max / median", n=int(w.notna().sum()),
                         share_pct=np.nan, evidence=f"ratio {w.max() / w.median():.1f}",
                         verdict="design weights; extreme ratios kept (never trimmed without Geostat design)"))
    # HIES income
    top = flags[(flags.dataset == "hies") & (flags.variable == "inc_total_per_capita") &
                (flags.method == "iqr") & (flags.direction == "high")]
    t = hh[hh.household_id.isin(top.record_id)]
    src_cols = ["inc_hired_employment", "inc_self_employment", "inc_agriculture_sales", "inc_property",
                "inc_pensions_benefits", "inc_remittances_abroad", "inc_gifts", "inc_non_cash"]
    dominant = t[src_cols].idxmax(axis=1).value_counts().head(4).to_dict()
    rows.append(dict(dataset="hies", check="high per-capita income (IQR on log)", n=len(t),
                     share_pct=round(100 * len(t) / len(hh), 2),
                     evidence=f"median household size {t.household_size.median():g} vs {hh.household_size.median():g}; "
                              f"Tbilisi share {(t.region == 'Tbilisi').mean():.0%} vs {(hh.region == 'Tbilisi').mean():.0%}; "
                              f"dominant source {dominant}; identity errors {int(t.flag_income_identity_total.sum())}",
                     verdict="income accounts add up (0 identity errors); smaller households; large share driven by gifts / agricultural sales (lumpy one-off income) -> genuine but volatile; kept, flagged"))
    low = flags[(flags.dataset == "hies") & (flags.variable == "inc_total_per_capita") &
                (flags.method == "iqr") & (flags.direction == "low")]
    tl = hh[hh.household_id.isin(low.record_id)]
    rows.append(dict(dataset="hies", check="low per-capita income (IQR on log)", n=len(tl),
                     share_pct=round(100 * len(tl) / len(hh), 2),
                     evidence=f"median {tl.inc_total_per_capita.median():.1f} GEL/person; "
                              f"non-cash share {(tl.inc_non_cash / tl.inc_total).median():.0%}",
                     verdict="very low cash incomes with consistent accounts; no evidence of entry error; kept, flagged"))
    z = hh[hh.flag_zero_total_income]
    rows.append(dict(dataset="hies", check="total income = 0", n=len(z), share_pct=round(100 * len(z) / len(hh), 2),
                     evidence=f"households with other cash inflows (borrowing/property sale) > 0: "
                              f"{int(((z.inflow_borrowing_savings + z.inflow_property_disposal + z.inflow_other_cash) > 0).sum())}",
                     verdict="real zero income (lived on borrowing/savings); kept as 0, not missing"))
    inc_cols = [c for c in hh.columns if c.startswith(("inc_", "inflow")) and not c.endswith("real_2025")
                and "per_" not in c]
    neg = hh[(hh[inc_cols] < 0).any(axis=1)]
    rows.append(dict(dataset="hies", check="negative income component", n=len(neg), share_pct=np.nan,
                     evidence="; ".join(f"{r.household_id}: " + ", ".join(f"{c}={r[c]:.1f}" for c in inc_cols if r[c] < 0)
                                        for _, r in neg.iterrows()),
                     verdict="IMPOSSIBLE value (gift received cannot be negative) -> probable data error; raw kept, flag_negative_income=True, exclude from income statistics"))
    r = pd.DataFrame(rows)
    r.to_csv(REPORTS / "outlier_investigation.csv", index=False)
    return r


def plots(lfs, hh):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (p, d) in zip(axes, lfs.items()):
        emp = d[d.labour_status == "employed"]
        sns.histplot(emp.hours_usual.dropna(), bins=range(0, 150, 4), ax=ax, color="#4C72B0")
        ax.axvline(80, color="#C44E52", ls="--", lw=1, label="domain limit 80 h")
        lo, hi = iqr_bounds(emp.hours_usual.dropna())
        ax.axvline(hi, color="#DD8452", ls=":", lw=1, label=f"IQR upper {hi:g} h")
        ax.set(title=f"Usual weekly hours, employed — LFS {p.replace('_', '–')}", xlabel="hours", ylabel="persons")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "stage02_hours_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    d = hh[hh.inc_total_per_capita_real_2025 > 0]
    # log_scale=True: quartiles and 1.5*IQR whiskers computed on log values = the method used for flags
    sns.boxplot(data=d, x="year", y="inc_total_per_capita_real_2025", ax=ax, color="#8DB0D6",
                log_scale=True, flierprops=dict(marker=".", markersize=2, alpha=.4))
    ax.set(title="HIES household income per capita (real, 2025 GEL, log scale) — outliers = points",
           xlabel="survey year", ylabel="GEL / person / month")
    fig.tight_layout()
    fig.savefig(FIG / "stage02_hies_income_outliers.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------- quality summary ---
def pk_unique(df, cols):
    return bool(~df.duplicated(cols).any())


def quality_summary(lfs, hh, mem, flags) -> pd.DataFrame:
    rows = []
    for p, d in lfs.items():
        for y, g in d.groupby("year_source_file"):
            emp = g[g.is_employee]
            rows.append(dict(
                dataset="lfs", layer="analytical", period=p, year=y, n_rows=len(g), n_cols=g.shape[1],
                primary_key="individual_id", pk_unique=pk_unique(g, ["individual_id"]),
                n_duplicate_keys=int(g.duplicated(["uid", "member_no"]).sum()),
                pct_missing_sex=round(100 * g.sex.isna().mean(), 2),
                pct_missing_age=round(100 * g.age.isna().mean(), 2),
                pct_missing_region=round(100 * g.region.isna().mean(), 2),
                pct_missing_education=round(100 * g.education_code.isna().mean(), 2),
                n_employees=len(emp),
                pct_missing_hours_employees=round(100 * emp.hours_usual.isna().mean(), 2),
                pct_earnings_valid_employees=round(100 * emp.earnings_status.eq("valid").mean(), 2),
                pct_earnings_refused_or_missing=round(100 * emp.earnings_status.isin(
                    ["refused", "missing_nonresponse", "undocumented_code"]).mean(), 2),
                n_invalid_sex=int(g.sex_code.isna().sum()),
                n_age_out_of_range=int(g.flag_age_out_of_range.sum()),
                n_age_mismatch=int(g.flag_age_mismatch_demographic.sum()),
                n_outlier_flags=int(flags[(flags.dataset == f"lfs_{p}") & (flags.group == y)].shape[0]),
                weighted_population_15plus=round(g.weight_annual.sum())))
    for y, g in hh.groupby("year"):
        m = mem[mem.year == y]
        rows.append(dict(
            dataset="hies", layer="analytical (households)", period="2017_2025", year=y, n_rows=len(g),
            n_cols=g.shape[1], primary_key="household_id", pk_unique=pk_unique(g, ["household_id"]),
            n_duplicate_keys=int(g.duplicated("uid").sum()),
            pct_missing_sex=round(100 * m.sex.isna().mean(), 2), pct_missing_age=round(100 * m.age.isna().mean(), 2),
            pct_missing_region=round(100 * g.region.isna().mean(), 2),
            pct_missing_education=round(100 * m.education_code.isna().mean(), 2),
            n_members=len(m), pct_zero_total_income=round(100 * g.flag_zero_total_income.mean(), 2),
            n_income_identity_errors=int(g.flag_income_identity_items.sum() + g.flag_income_identity_total.sum()),
            n_households_without_single_head=int(g.head_sex.isna().sum()),
            n_size_mismatch=int(g.flag_size_mismatch.sum()),
            n_outlier_flags=int(flags[(flags.dataset == "hies") & (flags.group == y)].shape[0]),
            weighted_households=round(g.weight_annual.sum())))
    for p in ("2017_2019", "2020_2025"):
        ml = pd.read_csv(FINAL / f"ml_lfs_employees_{p}.csv.gz")
        rows.append(dict(dataset="lfs", layer="ml", period=p, year="all", n_rows=len(ml), n_cols=ml.shape[1],
                         primary_key="individual_id", pk_unique=pk_unique(ml, ["individual_id"]),
                         pct_missing_sex=round(100 * ml.sex.isna().mean(), 2),
                         pct_missing_age=round(100 * ml.age.isna().mean(), 2),
                         pct_missing_hours_employees=round(100 * ml.hours_usual.isna().mean(), 2)))
    s = pd.DataFrame(rows)
    s.to_csv(REPORTS / "data_quality_summary.csv", index=False)
    return s


def run():
    lfs, hh, mem = load()
    flags, summ = outliers(lfs, hh)
    inv = investigate(lfs, hh, flags)
    plots(lfs, hh)
    q = quality_summary(lfs, hh, mem, flags)
    return q, summ, inv


if __name__ == "__main__":
    q, s, i = run()
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(q.to_string(index=False))
    print(s.to_string(index=False))
    print(i.to_string(index=False))
