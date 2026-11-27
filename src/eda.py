"""
Stage 03 — Exploratory Data Analysis (descriptive only: no hypothesis tests, no models).

Data
  * INDIVIDUAL income: LFS employees with a valid net-earnings band
    (views v_ml_lfs_employees_<period> in database/income_data.db).
    Amount = band midpoint, open top band = Pareto mean (Stage 02) -> an ESTIMATE.
    Two methodological periods, 2017-2019 and 2020-2025, are analysed SEPARATELY.
  * HOUSEHOLD income (HIES) is used in ONE separate section (by sex of household head)
    and is never mixed with individual earnings.

Weights
  * every statistic is given weighted (weight_annual) and unweighted
  * pooled period statistics use  w_pooled = weight_annual / number_of_years_in_period
    (each year counts once, as an average year)

Gap definitions
  gap_mean_pct   = (male_mean   - female_mean)   / male_mean   * 100
  gap_median_pct = (male_median - female_median) / male_median * 100

Reliability rule for subgroups (per sex, unweighted n):
  n >= 100 -> "ok";  30 <= n < 100 -> "low_n" (no firm conclusion);  n < 30 -> "suppressed"

Usage:  cd src && python eda.py
"""
from __future__ import annotations

import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import data_collection as dc

DB = dc.PROJECT_ROOT / "database" / "income_data.db"
REPORTS = dc.PROJECT_ROOT / "reports"
FIG = REPORTS / "figures"
PERIODS = {"2017_2019": [2017, 2018, 2019], "2020_2025": [2020, 2021, 2022, 2023, 2024, 2025]}
SEX_COLORS = {"female": "#C44E52", "male": "#4C72B0"}
WEEKS_PER_MONTH = 52 / 12
MIN_N, LOW_N = 30, 100

SOURCE_LFS = ("Source: Geostat Labour Force Survey microdata ({p}); employees, net monthly earnings in main job "
              "(band midpoints, Pareto top band); weighted.")
SOURCE_HIES = "Source: Geostat Household Incomes and Expenditures Survey 2017-2025; household income, weighted."

ISCO1 = {0: "Armed forces / other", 1: "Managers", 2: "Professionals", 3: "Technicians & assoc. prof.",
         4: "Clerical support", 5: "Service & sales", 6: "Skilled agricultural", 7: "Craft & trades",
         8: "Plant & machine operators", 9: "Elementary occupations"}
NACE2 = {1: "Agriculture", 2: "Mining", 3: "Manufacturing", 4: "Electricity, gas", 5: "Water, waste",
         6: "Construction", 7: "Trade, repair", 8: "Transport, storage", 9: "Accommodation, food",
         10: "Information, communication", 11: "Finance, insurance", 12: "Real estate",
         13: "Professional, scientific", 14: "Administrative, support", 15: "Public administration",
         16: "Education", 17: "Health, social work", 18: "Arts, recreation", 19: "Other services",
         20: "Households as employers", 21: "Extraterritorial org."}
# EDA education mapping — only categories that exist in the source (13-level national code)
EDU_EDA = {1: "Below secondary", 2: "Below secondary", 3: "Below secondary", 4: "Below secondary",
           5: "Below secondary", 6: "Secondary general", 7: "Vocational", 8: "Vocational", 9: "Vocational",
           10: "Higher professional", 11: "Bachelor", 12: "Master", 13: "Doctor"}
EDU_ORDER = ["Below secondary", "Secondary general", "Vocational", "Higher professional", "Bachelor",
             "Master", "Doctor"]
AGE_BINS = [17, 24, 34, 44, 54, 64, 200]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
HOURS_BINS = [0, 19, 34, 40, 48, 60, 200]
HOURS_LABELS = ["1-19", "20-34", "35-40", "41-48", "49-60", "61+"]


# ------------------------------------------------------------------ helpers ---
def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x) & ~np.isnan(w)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m])) if m.any() else np.nan


def wquantile(x, w, q):
    """Weighted quantile: smallest x whose cumulative weight share >= q."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x) & ~np.isnan(w)
    if not m.any():
        return np.nan
    o = np.argsort(x[m])
    xs, cw = x[m][o], np.cumsum(w[m][o]) / np.sum(w[m])
    return float(xs[min(np.searchsorted(cw, q), len(xs) - 1)])


def wstd(x, w):
    mu = wmean(x, w)
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x)
    return float(np.sqrt(np.sum(w[m] * (x[m] - mu) ** 2) / np.sum(w[m])))


def gap_pct(male, female):
    """(male - female) / male * 100; works for scalars and Series; NaN when male is 0/NaN."""
    if isinstance(male, (pd.Series, np.ndarray)):
        male = pd.Series(male, dtype=float).replace(0, np.nan)
        return (male - female) / male * 100
    return (male - female) / male * 100 if male and not np.isnan(male) else np.nan


def summarize(x, w=None) -> dict:
    """Full descriptive set; weighted if w is given."""
    x = np.asarray(x, float)
    if w is None:
        w = np.ones_like(x)
    w = np.asarray(w, float)
    return dict(n=int(len(x)), population=round(float(w.sum())) if w is not None else np.nan,
                mean=wmean(x, w), sd=wstd(x, w), min=float(np.nanmin(x)), p25=wquantile(x, w, .25),
                median=wquantile(x, w, .5), p75=wquantile(x, w, .75), max=float(np.nanmax(x)),
                p10=wquantile(x, w, .10), p90=wquantile(x, w, .90))


def interval_quantile(lo, hi, alpha, w, q, iters=60):
    """Quantile of a mixture of per-person distributions: uniform on (lo, hi] for closed bands,
    Pareto(alpha) above lo for the open top band (hi = inf).  Standard grouped-data method:
    interpolates WITHIN bands instead of jumping between band midpoints."""
    lo, hi, alpha, w = (np.asarray(v, float) for v in (lo, hi, alpha, w))
    m = ~np.isnan(lo) & ~np.isnan(w)
    lo, hi, alpha, w = lo[m], hi[m], alpha[m], w[m]
    if not len(lo):
        return np.nan
    top = np.isinf(hi)
    width = np.where(top, 1.0, hi - lo)

    def cdf(x):
        f = np.where(top, np.where(x > lo, 1 - (lo / np.maximum(x, 1e-12)) ** alpha, 0.0),
                     np.clip((x - lo) / width, 0, 1))
        return np.sum(w * f) / np.sum(w)
    a_, b_ = 0.0, float(np.nanmax(np.where(top, lo * 50, hi)))
    for _ in range(iters):
        mid = (a_ + b_) / 2
        a_, b_ = (mid, b_) if cdf(mid) < q else (a_, mid)
    return (a_ + b_) / 2


def summarize_bands(g: pd.DataFrame, w=None, basis="real") -> dict:
    """Descriptives for banded earnings.  mean/sd/min/max use the band point estimate
    (midpoint, Pareto mean for the top band); quantiles are interpolated within bands.
    basis: nominal | real (2025 GEL) | hourly_real (real / (usual weekly hours * 52/12))"""
    scale = np.ones(len(g))
    if basis in ("real", "hourly_real"):
        scale = (g.income_amount_real_2025 / g.income_amount).to_numpy()
    if basis == "hourly_real":
        scale = scale / (g.working_hours.to_numpy() * WEEKS_PER_MONTH)
    ww = np.ones(len(g)) if w is None else np.asarray(w, float)
    ok = ~np.isnan(scale)
    x = (g.income_amount.to_numpy() * scale)[ok]
    lo, hi = (g.income_lower.to_numpy() * scale)[ok], (g.income_upper.fillna(np.inf).to_numpy() * scale)[ok]
    al, ww = g.pareto_alpha.to_numpy()[ok], ww[ok]
    if not len(x):
        return dict(n=0, population=0, mean=np.nan, sd=np.nan, min=np.nan, p10=np.nan, p25=np.nan,
                    median=np.nan, p75=np.nan, p90=np.nan, max=np.nan)
    qs = {k: interval_quantile(lo, hi, al, ww, q) for k, q in
          [("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9)]}
    return dict(n=int(len(x)), population=round(float(ww.sum())), mean=wmean(x, ww), sd=wstd(x, ww),
                min=float(x.min()), max=float(x.max()), **qs)


def reliability(n_f, n_m):
    n = min(n_f, n_m)
    return "suppressed" if n < MIN_N else ("low_n" if n < LOW_N else "ok")


def caption(fig, text, y=-0.07):
    """Source / explanation note placed BELOW the axes (bbox_inches='tight' keeps it in the file)."""
    import textwrap
    fig.text(0.01, y, textwrap.fill(text, width=int(fig.get_figwidth() * 17)), fontsize=7.5,
             color="#555555", ha="left", va="top")


# --------------------------------------------------------------------- data ---
def load() -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(DB)
    out = {}
    alpha = pd.read_sql("SELECT year, alpha FROM pareto_top_band", con).set_index("year").alpha
    for p, years in PERIODS.items():
        d = pd.read_sql(f"""SELECT individual_id, household_key, survey_year, period, gender, age, education_code, region,
                                   region_harmonised, urban_rural, occupation, industry_nace2, sector,
                                   working_hours, part_time, informal, contract_written, household_size,
                                   years_schooling_approx, potential_experience, weight_annual,
                                   band_code, band_code_common, income_lower, income_upper,
                                   income_amount, income_amount_real_2025, is_top_band
                            FROM v_ml_lfs_employees_{p}""", con)
        d["w_pooled"] = d.weight_annual / len(years)
        d["pareto_alpha"] = d.survey_year.map(alpha)
        d["age_group"] = pd.cut(d.age, AGE_BINS, labels=AGE_LABELS).astype(object).where(d.age >= 18, "15-17")
        d["education_eda"] = d.education_code.map(EDU_EDA)
        d["occupation_label"] = d.occupation.map(ISCO1)
        d["industry_label"] = d.industry_nace2.map(NACE2)
        # P1: Racha only published separately from 2019 -> harmonised; P2: original regions
        d["region_eda"] = d.region_harmonised if p == "2017_2019" else d.region
        d["hours_band"] = pd.cut(d.working_hours, HOURS_BINS, labels=HOURS_LABELS).astype(object)
        d["hourly_nominal"] = d.income_amount / (d.working_hours * WEEKS_PER_MONTH)
        d["hourly_real_2025"] = d.income_amount_real_2025 / (d.working_hours * WEEKS_PER_MONTH)
        out[p] = d
    out["hies"] = pd.read_sql("""SELECT household_id, survey_year, head_sex, household_size, household_income,
                                        household_income_real_2025, household_income_per_capita, weight_annual,
                                        flag_negative_income, urban_rural
                                 FROM households""", con)
    con.close()
    return out


# ----------------------------------------------------------------- analyses ---
def descriptive(data) -> pd.DataFrame:
    rows = []
    for p, d in ((k, v) for k, v in data.items() if k in PERIODS):
        scopes = [(str(y), d[d.survey_year == y], "weight_annual") for y in PERIODS[p]] + \
                 [(f"pooled {p}", d, "w_pooled")]
        for scope, g, wcol in scopes:
            for sex in ["female", "male", "all"]:
                s = g if sex == "all" else g[g.gender == sex]
                for basis in ["nominal", "real"]:
                    if scope.startswith("pooled") and basis == "nominal":
                        continue           # pooling nominal GEL across years is not meaningful
                    for weighting in ["weighted", "unweighted"]:
                        st = summarize_bands(s, s[wcol] if weighting == "weighted" else None, basis)
                        if weighting == "unweighted":
                            st["population"] = np.nan
                        rows.append(dict(period=p, scope=scope, sex=sex,
                                         income_basis="real_2025" if basis == "real" else basis, weighting=weighting,
                                         **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()},
                                         income_type="individual net monthly earnings (employees)",
                                         currency="GEL", amount_note="mean/sd/min/max: band midpoint (Pareto mean in top band); "
                                         "quantiles: interpolated within bands"))
    return pd.DataFrame(rows)


def official_reference() -> pd.DataFrame:
    """Geostat enterprise-survey average nominal (gross) earnings by sex + official adjusted GPG (LFS)."""
    src = dc.local_path(next(s for s in dc.SOURCES if s.key == "wages_by_sex_annual"))
    s = pd.read_excel(src, sheet_name="NACE2", header=None)
    years = s.iloc[2, 1:].astype(float).astype(int).tolist()
    vals = {}
    sex = None
    for _, r in s.iterrows():
        lab = str(r.iloc[0]).strip() if pd.notna(r.iloc[0]) else str(r.iloc[1]).strip()
        if lab in ("Women", "Men"):
            sex = lab
        elif str(r.iloc[0]).strip() == "Total" and sex and sex not in vals:
            vals[sex] = pd.to_numeric(r.iloc[1:1 + len(years)], errors="coerce").tolist()
    ref = pd.DataFrame({"year": years, "official_enterprise_female": vals.get("Women"),
                        "official_enterprise_male": vals.get("Men")})
    ref["official_enterprise_gap_pct"] = gap_pct(ref.official_enterprise_male, ref.official_enterprise_female)
    g = pd.read_excel(dc.local_path(next(s for s in dc.SOURCES if s.key == "wages_gpg")), header=None)
    adj = {}
    for c in range(1, g.shape[1], 2):
        y = pd.to_numeric(g.iloc[1, c], errors="coerce")
        if pd.notna(y):
            adj[int(y)] = (float(g.iloc[3, c]), float(g.iloc[3, c + 1]))
    ref["official_adjusted_gpg_hourly_pct"] = ref.year.map(lambda y: adj.get(y, (np.nan,))[0])
    ref["official_adjusted_gpg_monthly_pct"] = ref.year.map(lambda y: adj.get(y, (np.nan, np.nan))[1])
    return ref


def gender_comparison(data) -> pd.DataFrame:
    rows = []
    for p, d in ((k, v) for k, v in data.items() if k in PERIODS):
        scopes = [(y, d[d.survey_year == y], "weight_annual") for y in PERIODS[p]] + [(f"pooled", d, "w_pooled")]
        for scope, g, wcol in scopes:
            f, m = g[g.gender == "female"], g[g.gender == "male"]
            r = dict(period=p, year=scope, n_female=len(f), n_male=len(m))
            for basis in ["nominal", "real"]:
                if scope == "pooled" and basis == "nominal":
                    continue
                for wt in ["w", "u"]:
                    wf = f[wcol] if wt == "w" else None
                    wm = m[wcol] if wt == "w" else None
                    sf, sm = summarize_bands(f, wf, basis), summarize_bands(m, wm, basis)
                    tag = f"{basis}_{'weighted' if wt == 'w' else 'unweighted'}"
                    r.update({f"female_mean_{tag}": sf["mean"], f"male_mean_{tag}": sm["mean"],
                              f"female_median_{tag}": sf["median"], f"male_median_{tag}": sm["median"],
                              f"gap_mean_pct_{tag}": gap_pct(sm["mean"], sf["mean"]),
                              f"gap_median_pct_{tag}": gap_pct(sm["median"], sf["median"])})
            hf, hm = f[f.working_hours > 0], m[m.working_hours > 0]
            shf, shm = summarize_bands(hf, hf[wcol], "hourly_real"), summarize_bands(hm, hm[wcol], "hourly_real")
            r.update(dict(female_hourly_mean_real=shf["mean"], male_hourly_mean_real=shm["mean"],
                          gap_hourly_mean_pct=gap_pct(shm["mean"], shf["mean"]),
                          gap_hourly_median_pct=gap_pct(shm["median"], shf["median"]),
                          female_mean_hours=wmean(f.working_hours, f[wcol]),
                          male_mean_hours=wmean(m.working_hours, m[wcol]),
                          female_share_employees=f[wcol].sum() / g[wcol].sum(),
                          female_share_top_band=f.loc[f.is_top_band == 1, wcol].sum() /
                          max(g.loc[g.is_top_band == 1, wcol].sum(), 1e-9)))
            rows.append(r)
    out = pd.DataFrame(rows)
    ref = official_reference()
    out = out.merge(ref.assign(year=ref.year.astype(object)), on="year", how="left")
    return out.round(2)


def band_distribution(data) -> pd.DataFrame:
    """Weighted share of employees in each common band (nominal GEL bounds), per year and sex."""
    labels = {1: "<=100", 2: "101-200", 3: "201-400", 4: "401-600", 5: "601-800", 6: "801-1000",
              7: "1001-1500", 8: "1501-2000", 9: ">2000"}
    rows = []
    for p, d in ((k, v) for k, v in data.items() if k in PERIODS):
        for (y, sex), g in d.groupby(["survey_year", "gender"]):
            sh = g.groupby("band_code_common").weight_annual.sum() / g.weight_annual.sum()
            for b, v in sh.items():
                rows.append(dict(period=p, year=y, sex=sex, band_common=int(b), band_label=labels[int(b)],
                                 weighted_share=round(v, 4)))
    return pd.DataFrame(rows)


def income_groups(data) -> pd.DataFrame:
    """Low / high income groups: weighted quintiles of real income (pooled period); women's share per quintile."""
    rows = []
    for p, d in ((k, v) for k, v in data.items() if k in PERIODS):
        sc = d.income_amount_real_2025 / d.income_amount
        cuts = [interval_quantile(d.income_lower * sc, d.income_upper.fillna(np.inf) * sc, d.pareto_alpha,
                                  d.w_pooled, q) for q in (.2, .4, .6, .8)]
        q = np.searchsorted(cuts, d.income_amount_real_2025.values, side="left") + 1
        d = d.assign(quintile=q)
        for k, g in d.groupby("quintile"):
            rows.append(dict(period=p, quintile=int(k), real_upper_bound=round(cuts[k - 1], 1) if k < 5 else np.nan,
                             population_share=round(g.w_pooled.sum() / d.w_pooled.sum(), 3),
                             female_share=round(g.loc[g.gender == "female", "w_pooled"].sum() / g.w_pooled.sum(), 3),
                             n=len(g)))
    return pd.DataFrame(rows)


def group_gap(d: pd.DataFrame, group: str, p: str, order=None, label_map=None) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(group, dropna=True):
        f, m = g[g.gender == "female"], g[g.gender == "male"]
        rel = reliability(len(f), len(m))
        sf = summarize_bands(f, f.w_pooled) if len(f) else {}
        sm = summarize_bands(m, m.w_pooled) if len(m) else {}
        sa = summarize_bands(g, g.w_pooled)
        r = dict(period=p, group_variable=group, group=key, n_female=len(f), n_male=len(m), reliability=rel,
                 female_share=round(f.w_pooled.sum() / g.w_pooled.sum(), 3),
                 all_mean=sa["mean"], all_median=sa["median"], all_p25=sa["p25"], all_p75=sa["p75"],
                 female_mean=sf.get("mean", np.nan), male_mean=sm.get("mean", np.nan),
                 female_median=sf.get("median", np.nan), male_median=sm.get("median", np.nan),
                 female_sd=sf.get("sd", np.nan), male_sd=sm.get("sd", np.nan),
                 female_mean_hours=wmean(f.working_hours, f.w_pooled), male_mean_hours=wmean(m.working_hours, m.w_pooled))
        r["gap_mean_pct"] = gap_pct(r["male_mean"], r["female_mean"])
        r["gap_median_pct"] = gap_pct(r["male_median"], r["female_median"])
        if rel == "suppressed":
            for k in ("female_mean", "male_mean", "female_median", "male_median", "gap_mean_pct", "gap_median_pct"):
                r[k] = np.nan
        if label_map:
            r["mapping"] = label_map.get(key, "")
        rows.append(r)
    out = pd.DataFrame(rows)
    if order:
        out["_o"] = out.group.map({k: i for i, k in enumerate(order)})
        out = out.sort_values("_o").drop(columns="_o")
    out["income_basis"] = "real 2025 GEL, weighted (pooled period)"
    return out.round(2)


def all_group_tables(data):
    t = {}
    for name, col, order, lab in [
        ("age", "age_group", ["15-17"] + AGE_LABELS, None),
        ("education", "education_eda", EDU_ORDER,
         {c: ", ".join(str(k) for k, v in EDU_EDA.items() if v == c) for c in EDU_ORDER}),
        ("region", "region_eda", None, None),
        ("occupation", "occupation_label", list(ISCO1.values()), None),
        ("industry", "industry_label", None, None),
        ("hours", "hours_band", HOURS_LABELS, None),
        ("part_time", "part_time", None, None),
    ]:
        frames = []
        for p in PERIODS:
            g = group_gap(data[p], col, p, order, lab)
            if name == "education":
                g["mapping"] = g.group.map(lambda c: "source codes " + lab.get(c, ""))
            frames.append(g)
        t[name] = pd.concat(frames, ignore_index=True)
    t["part_time"]["group"] = t["part_time"].group.map({0: "full-time", 1: "part-time"})
    t["region"] = t["region"].sort_values(["period", "all_mean"], ascending=[True, False])
    t["occupation"] = t["occupation"].sort_values(["period", "all_mean"], ascending=[True, False])
    return t


def correlations(data) -> dict[str, pd.DataFrame]:
    out = {}
    for p in PERIODS:
        d = data[p]
        x = pd.DataFrame({
            "log_real_income": np.log(d.income_amount_real_2025), "female": (d.gender == "female").astype(int),
            "age": d.age, "years_schooling": d.years_schooling_approx,
            "potential_experience": d.potential_experience, "weekly_hours": d.working_hours,
            "part_time": d.part_time, "informal": d.informal, "written_contract": d.contract_written,
            "public_sector": (d.sector == "state").astype(int), "urban": (d.urban_rural == "urban").astype(int),
            "tbilisi": (d.region == "Tbilisi").astype(int), "household_size": d.household_size,
            "isco_skill_rank": 10 - d.occupation.where(d.occupation > 0)})   # 1 managers .. 9 elementary -> high=skilled
        out[p] = x.corr(method="spearman").round(3)
    return out


def hies_by_head_sex(data) -> pd.DataFrame:
    h = data["hies"]
    h = h[(h.flag_negative_income == 0) & h.head_sex.notna()]
    rows = []
    for (y, s), g in h.groupby(["survey_year", "head_sex"]):
        pc = g.household_income_real_2025 / g.household_size
        st = summarize(pc, g.weight_annual)
        rows.append(dict(year=y, head_sex=s, n_households=len(g), mean_per_capita_real=st["mean"],
                         median_per_capita_real=st["median"],
                         mean_household_size=wmean(g.household_size, g.weight_annual),
                         income_unit="household (per capita)", currency="GEL (2025 prices)"))
    r = pd.DataFrame(rows)
    piv = r.pivot(index="year", columns="head_sex", values="mean_per_capita_real")
    r["gap_mean_pct_per_capita"] = r.year.map(gap_pct(piv["male"], piv["female"]))
    return r.round(2)


# -------------------------------------------------------------------- plots ---
def _save(fig, name):
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(data, gc, bands, t, corr, hies) -> list[dict]:
    FIG.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    index = []

    def reg(name, title, explanation):
        index.append(dict(file=name, title=title, explanation=explanation))

    yr = gc[gc.year != "pooled"].copy()
    yr["year"] = yr.year.astype(int)

    # 1-2 mean / median by sex per year
    for i, (stat, ttl) in enumerate([("mean", "Average"), ("median", "Median")], start=1):
        fig, ax = plt.subplots(figsize=(10, 4.6))
        x = np.arange(len(yr))
        ax.bar(x - 0.2, yr[f"female_{stat}_real_weighted"], 0.4, label="Women", color=SEX_COLORS["female"])
        ax.bar(x + 0.2, yr[f"male_{stat}_real_weighted"], 0.4, label="Men", color=SEX_COLORS["male"])
        for xi, g in zip(x, yr[f"gap_{stat}_pct_real_weighted"]):
            ax.text(xi, max(yr[f"male_{stat}_real_weighted"]) * 1.04, f"{g:.0f}%", ha="center", fontsize=8)
        top = max(yr[f"male_{stat}_real_weighted"])
        ax.set_ylim(0, top * 1.15)                      # room for the gap labels
        ax.grid(axis="x", visible=False)
        ax.axvline(2.5, color="grey", ls="--", lw=1)
        ax.text(2.55, top * 1.13, "methodology change (2020)", fontsize=7.5, color="grey", va="top")
        ax.set_xticks(x, yr.year)
        ax.set(title=f"{ttl} net monthly earnings of employees by sex, 2017-2025 (gap % above bars)",
               xlabel="Survey year", ylabel="GEL per month (2025 prices)")
        ax.legend(frameon=False, loc="lower right")
        caption(fig, SOURCE_LFS.format(p="2017-2025") + " Real = CPI-deflated to 2025. Periods are not directly "
                "comparable (dashed line).")
        name = f"01_male_vs_female_{stat}_income.png" if i == 1 else "02_male_vs_female_median_income.png"
        _save(fig, name)
        reg(name, f"Male vs female {stat} income",
            f"Weighted {stat} real earnings by sex for every year; the percentage is the gap relative to men.")

    # 3 histogram of bands
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, (p, years) in zip(axes, [("2017_2019", 2019), ("2020_2025", 2025)]):
        b = bands[(bands.period == p) & (bands.year == years)]
        piv = b.pivot(index="band_label", columns="sex", values="weighted_share").reindex(
            ["<=100", "101-200", "201-400", "401-600", "601-800", "801-1000", "1001-1500", "1501-2000", ">2000"])
        piv.plot.bar(ax=ax, color=[SEX_COLORS["female"], SEX_COLORS["male"]], width=0.8)
        ax.set(title=f"Distribution of net monthly earnings, {years}", xlabel="Earnings band (GEL, nominal)",
               ylabel="Share of employees")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.legend(["Women", "Men"], frameon=False)
    caption(fig, SOURCE_LFS.format(p="2019, 2025") + " Bars = weighted share of each sex in each band "
            "(the survey publishes bands, not amounts).")
    _save(fig, "03_income_distribution_histogram.png")
    reg("03_income_distribution_histogram.png", "Income distribution histogram",
        "Banded distribution: women are over-represented in the low bands, men in the high bands.")

    # 4 weighted boxplot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, p in zip(axes, PERIODS):
        d = data[p]
        stats = []
        for sex in ["female", "male"]:
            s = d[d.gender == sex]
            st = summarize_bands(s, s.w_pooled)
            stats.append(dict(label="Women" if sex == "female" else "Men", med=st["median"], q1=st["p25"],
                              q3=st["p75"], whislo=st["p10"], whishi=st["p90"], mean=st["mean"], fliers=[]))
        bp = ax.bxp(stats, showmeans=True, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], SEX_COLORS.values()):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set(title=f"Earnings by sex, {p.replace('_', '-')} pooled", ylabel="GEL per month (2025 prices)")
    caption(fig, SOURCE_LFS.format(p="2017-2025") + " Box = weighted P25-P75, line = median, triangle = mean, "
            "whiskers = P10-P90; quantiles interpolated within earnings bands.")
    _save(fig, "04_gender_income_boxplot.png")
    reg("04_gender_income_boxplot.png", "Gender income boxplot", "Weighted quartiles per sex and period.")

    # 5-8 group comparisons (P2 main, real)
    def grouped(tab, title, fname, xlabel, rotate=0, horizontal=False):
        g = tab[(tab.period == "2020_2025") & (tab.reliability != "suppressed")].copy()
        g["label"] = g.group.astype(str) + np.where(g.reliability == "low_n", " *", "")
        fig, ax = plt.subplots(figsize=(11, 5 if not horizontal else 6))
        m = g.melt(id_vars="label", value_vars=["female_mean", "male_mean"], var_name="sex", value_name="v")
        m["sex"] = m.sex.map({"female_mean": "Women", "male_mean": "Men"})
        if horizontal:
            sns.barplot(data=m, y="label", x="v", hue="sex", palette=[SEX_COLORS["female"], SEX_COLORS["male"]], ax=ax)
            ax.set(xlabel="Mean GEL per month (2025 prices)", ylabel=xlabel)
        else:
            sns.barplot(data=m, x="label", y="v", hue="sex", palette=[SEX_COLORS["female"], SEX_COLORS["male"]], ax=ax)
            ax.set(ylabel="Mean GEL per month (2025 prices)", xlabel=xlabel)
            ax.tick_params(axis="x", rotation=rotate)
            for i, (_, r) in enumerate(g.iterrows()):
                ax.text(i, max(r.male_mean, r.female_mean) * 1.02, f"{r.gap_mean_pct:.0f}%", ha="center", fontsize=8)
        ax.set_title(title)
        ax.legend(frameon=False, title=None)
        caption(fig, SOURCE_LFS.format(p="2020-2025 pooled") + " Gap % = (men - women) / men. "
                "* = fewer than 100 observations for one sex: no firm conclusion.")
        _save(fig, fname)

    grouped(t["age"], "Mean earnings by age group and sex, 2020-2025 (gap % above bars)",
            "05_income_by_age_group.png", "Age group")
    reg("05_income_by_age_group.png", "Income by age group", "Mean real earnings by sex in each age group.")
    grouped(t["education"], "Mean earnings by education and sex, 2020-2025 (gap % above bars)",
            "06_income_by_education.png", "Education (source categories)", rotate=15)
    reg("06_income_by_education.png", "Income by education", "Mean real earnings by sex for each education level.")
    grouped(t["region"], "Mean earnings by region and sex, 2020-2025", "07_regional_income_comparison.png",
            "Region", horizontal=True)
    reg("07_regional_income_comparison.png", "Regional income comparison", "Mean real earnings by sex and region.")
    grouped(t["occupation"], "Mean earnings by occupation (ISCO major group) and sex, 2020-2025",
            "08_occupation_income_comparison.png", "Occupation", horizontal=True)
    reg("08_occupation_income_comparison.png", "Occupation income comparison",
        "Mean real earnings by sex per occupation group.")

    # 9 gap by region, both periods
    fig, ax = plt.subplots(figsize=(10, 5))
    r = t["region"][t["region"].reliability != "suppressed"]
    piv = r.pivot(index="group", columns="period", values="gap_mean_pct").sort_values("2020_2025")
    piv.plot.barh(ax=ax, color=["#BBBBBB", "#55A868"])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Gender pay gap (mean, %) by region", xlabel="Gap % = (men - women) / men", ylabel="Region")
    # legend built from the bar containers (column order) — not from all artists (the 0-line is one too)
    ax.legend(ax.containers, [c.replace("_", "-") for c in piv.columns], frameon=False)
    caption(fig, SOURCE_LFS.format(p="2017-2019 and 2020-2025, pooled per period") + " In 2017-2019 Racha is "
            "merged with Imereti (published separately only from 2019).")
    _save(fig, "09_gender_pay_gap_by_region.png")
    reg("09_gender_pay_gap_by_region.png", "Gender pay gap by region", "Unadjusted mean gap per region and period.")

    # 10 trends
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for sex in ["female", "male"]:
        lab = "Women" if sex == "female" else "Men"
        for per in PERIODS:
            s = yr[yr.period == per]
            ax.plot(s.year, s[f"{sex}_mean_real_weighted"], "-o", color=SEX_COLORS[sex],
                    label=f"{lab}, real" if per == "2020_2025" else None)
            ax.plot(s.year, s[f"{sex}_mean_nominal_weighted"], ":", color=SEX_COLORS[sex],
                    label=f"{lab}, nominal" if per == "2020_2025" else None)
    ax.axvline(2019.5, color="grey", ls="--", lw=1)
    ax.set(title="Mean net monthly earnings: real (2025 prices) vs nominal", xlabel="Survey year", ylabel="GEL per month")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax = axes[1]
    for per in PERIODS:
        s = yr[yr.period == per]
        ax.plot(s.year, s.gap_mean_pct_real_weighted, "-o", color="#55A868",
                label="Mean gap (this study, LFS)" if per == "2020_2025" else None)
        ax.plot(s.year, s.gap_median_pct_real_weighted, "-s", color="#8172B2",
                label="Median gap (this study, LFS)" if per == "2020_2025" else None)
        ax.plot(s.year, s.gap_hourly_median_pct, "--^", color="#CCB974",
                label="Hourly median gap (this study)" if per == "2020_2025" else None)
    ax.plot(yr.year, yr.official_enterprise_gap_pct, ":", color="black", label="Geostat enterprise survey (gross, mean)")
    ax.plot(yr.year, yr.official_adjusted_gpg_monthly_pct, "x", color="black", ms=7,
            label="Geostat adjusted GPG, monthly (LFS)")
    ax.axvline(2019.5, color="grey", ls="--", lw=1)
    ax.set(title="Gender pay gap, 2017-2025", xlabel="Survey year", ylabel="Gap, % of men's earnings")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    caption(fig, SOURCE_LFS.format(p="2017-2025") + " Official references: Geostat wage statistics "
            "(enterprise survey, gross) and Geostat adjusted GPG table. Dashed line = methodology change.", y=-0.2)
    _save(fig, "10_income_trends_by_year.png")
    reg("10_income_trends_by_year.png", "Income trends by year",
        "Real and nominal means by sex; unadjusted mean, median and hourly-median gaps with official references.")

    # 11 scatter: female vs male mean per occupation x region cell
    d = data["2020_2025"]
    cells = []
    for (o, rgn), g in d.groupby(["occupation_label", "region_eda"]):
        f, m = g[g.gender == "female"], g[g.gender == "male"]
        if min(len(f), len(m)) >= MIN_N:
            cells.append(dict(occupation=o, region=rgn, female=wmean(f.income_amount_real_2025, f.w_pooled),
                              male=wmean(m.income_amount_real_2025, m.w_pooled), n=len(g)))
    cells = pd.DataFrame(cells)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    sns.scatterplot(data=cells, x="male", y="female", size="n", hue="occupation", ax=ax, sizes=(20, 300), alpha=.8)
    lim = [0, max(cells.male.max(), cells.female.max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.text(lim[1] * 0.6, lim[1] * 0.63, "equal pay line", rotation=45, fontsize=8)
    ax.set(xlim=lim, ylim=lim, title="Women's vs men's mean earnings in the same occupation and region, 2020-2025",
           xlabel="Men, mean GEL per month (2025 prices)", ylabel="Women, mean GEL per month (2025 prices)")
    ax.legend(fontsize=6.5, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    caption(fig, SOURCE_LFS.format(p="2020-2025 pooled") + f" Each point = occupation x region cell with >= {MIN_N} "
            "women and men. Points below the line: women earn less.")
    _save(fig, "11_gender_income_scatter.png")
    reg("11_gender_income_scatter.png", "Gender income scatter plot",
        f"{len(cells)} occupation x region cells; share below the equal-pay line = "
        f"{(cells.female < cells.male).mean():.0%}.")

    # 12 correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (p, c) in zip(axes, corr.items()):
        sns.heatmap(c, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                    annot_kws={"size": 6.5}, cbar=ax is axes[-1], yticklabels=ax is axes[0])
        ax.set_title(f"Spearman correlations, employees {p.replace('_', '-')}")
    caption(fig, SOURCE_LFS.format(p="2017-2025") + " Unweighted Spearman rank correlations; log_real_income = log of "
            "real 2025 GEL earnings. Correlation is not causation.")
    _save(fig, "12_correlation_heatmap.png")
    reg("12_correlation_heatmap.png", "Correlation heatmap", "Rank correlations between earnings and characteristics.")

    # 13 hours: monthly vs hourly gap by hours band
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    h = t["hours"][(t["hours"].period == "2020_2025") & (t["hours"].reliability != "suppressed")]
    x = np.arange(len(h))
    axes[0].bar(x - .2, h.female_mean, .4, color=SEX_COLORS["female"], label="Women")
    axes[0].bar(x + .2, h.male_mean, .4, color=SEX_COLORS["male"], label="Men")
    axes[0].set_xticks(x, h.group)
    axes[0].set(title="Mean monthly earnings by usual weekly hours, 2020-2025", xlabel="Usual weekly hours (main job)",
                ylabel="GEL per month (2025 prices)")
    axes[0].legend(frameon=False)
    s = yr
    for per in PERIODS:                       # periods drawn separately (methodology break)
        q = s[s.period == per]
        first = per == "2020_2025"
        axes[1].plot(q.year, q.gap_median_pct_real_weighted, "-o", color="#55A868",
                     label="Monthly, median" if first else None)
        axes[1].plot(q.year, q.gap_hourly_median_pct, "-^", color="#CCB974", label="Hourly, median" if first else None)
        axes[1].plot(q.year, q.gap_hourly_mean_pct, ":", color="#CCB974",
                     label="Hourly, mean (sensitive to very short hours)" if first else None)
    axes[1].axvline(2019.5, color="grey", ls="--", lw=1)
    axes[1].set(title="Monthly vs hourly gender gap", xlabel="Survey year", ylabel="Gap, %")
    axes[1].legend(frameon=False)
    caption(fig, SOURCE_LFS.format(p="2017-2025") + " Hourly = monthly earnings / (usual weekly hours x 52/12); "
            "an approximation (banded earnings, usual hours).")
    _save(fig, "13_working_hours_and_hourly_gap.png")
    reg("13_working_hours_and_hourly_gap.png", "Working hours and hourly earnings",
        "Separates the monthly gap from the hourly gap.")

    # 14 HIES household income by head sex
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for s_, lab in [("female", "Female-headed households"), ("male", "Male-headed households")]:
        g = hies[hies.head_sex == s_]
        ax.plot(g.year, g.mean_per_capita_real, "-o", color=SEX_COLORS[s_], label=lab)
    ax.set(title="HOUSEHOLD income per capita by sex of household head (HIES)", xlabel="Survey year",
           ylabel="GEL per person per month (2025 prices)")
    ax.legend(frameon=False)
    caption(fig, SOURCE_HIES + " Household-level income: NOT individual earnings and not comparable with the LFS figures.")
    _save(fig, "14_hies_household_income_by_head_sex.png")
    reg("14_hies_household_income_by_head_sex.png", "Household income by head's sex (HIES)",
        "Separate household-level view.")

    # interactive (Plotly)
    try:
        import plotly.express as px
        long = yr.melt(id_vars=["year", "period"], value_vars=["female_mean_real_weighted", "male_mean_real_weighted"],
                       var_name="sex", value_name="GEL")
        long["sex"] = long.sex.str.split("_").str[0]
        fig = px.line(long, x="year", y="GEL", color="sex", line_dash="period", markers=True,
                      color_discrete_map=SEX_COLORS,
                      title="Mean net monthly earnings by sex (real 2025 GEL) — Geostat LFS, employees, weighted")
        fig.write_html(FIG / "10_income_trends_by_year_interactive.html", include_plotlyjs="cdn")
        fig = px.scatter(cells, x="male", y="female", size="n", color="occupation", hover_data=["region"],
                         title="Women vs men mean earnings by occupation x region, 2020-2025 (real 2025 GEL)")
        fig.add_shape(type="line", x0=0, y0=0, x1=lim[1], y1=lim[1], line=dict(dash="dash"))
        fig.write_html(FIG / "11_gender_income_scatter_interactive.html", include_plotlyjs="cdn")
        index += [dict(file="10_income_trends_by_year_interactive.html", title="Interactive trends", explanation=""),
                  dict(file="11_gender_income_scatter_interactive.html", title="Interactive scatter", explanation="")]
    except ImportError:
        pass
    return index


# --------------------------------------------------------------------- run ---
def run() -> dict:
    data = load()
    desc = descriptive(data)
    gc = gender_comparison(data)
    bands = band_distribution(data)
    groups = income_groups(data)
    t = all_group_tables(data)
    corr = correlations(data)
    hies = hies_by_head_sex(data)
    idx = plot_all(data, gc, bands, t, corr, hies)

    desc.to_csv(REPORTS / "descriptive_statistics.csv", index=False)
    gc.to_csv(REPORTS / "gender_income_comparison.csv", index=False)
    t["region"].to_csv(REPORTS / "regional_analysis.csv", index=False)
    t["education"].to_csv(REPORTS / "education_analysis.csv", index=False)
    t["occupation"].to_csv(REPORTS / "occupation_analysis.csv", index=False)
    t["age"].to_csv(REPORTS / "age_analysis.csv", index=False)
    pd.concat([t["hours"], t["part_time"]]).to_csv(REPORTS / "working_hours_analysis.csv", index=False)
    t["industry"].to_csv(REPORTS / "industry_analysis.csv", index=False)
    bands.to_csv(REPORTS / "income_band_distribution.csv", index=False)
    groups.to_csv(REPORTS / "income_groups_quintiles.csv", index=False)
    pd.concat({p: c for p, c in corr.items()}, names=["period", "variable"]).to_csv(REPORTS / "correlation_matrix.csv")
    hies.to_csv(REPORTS / "hies_household_income_by_head_sex.csv", index=False)
    pd.DataFrame(idx).to_csv(FIG / "figure_index.csv", index=False)
    return dict(data=data, desc=desc, gc=gc, bands=bands, groups=groups, tables=t, corr=corr, hies=hies, figures=idx)


if __name__ == "__main__":
    r = run()
    pd.set_option("display.width", 250)
    cols = ["period", "year", "n_female", "n_male", "female_mean_real_weighted", "male_mean_real_weighted",
            "gap_mean_pct_real_weighted", "gap_median_pct_real_weighted", "gap_mean_pct_real_unweighted",
            "gap_hourly_mean_pct", "official_enterprise_gap_pct", "official_adjusted_gpg_monthly_pct"]
    print(r["gc"][cols].to_string(index=False))
    print(len(r["figures"]), "figures")
