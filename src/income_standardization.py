"""
Stage 02 — Income standardisation.

Every income variable gets an explicit definition (unit, period, currency,
net/gross, nominal/real) — see INCOME_DEFINITIONS — and real values in 2025 prices.

LFS earnings (individual)
  * the public files hold only BANDS of net monthly earnings (main job, last worked month)
  * band bounds per period: BANDS["2017_2019"] (9 bands), BANDS["2020_2025"] (12 bands)
  * earnings_band_common: 2020-2025 bands mapped onto the 2017-2019 bands; possible
    because every new band nests exactly inside an old one (checked in check_nesting())
  * point estimate (only for descriptive use; interval methods are preferred later):
        closed band   -> midpoint (lower + upper) / 2   e.g. 401-600 -> 500
        first band    -> 50  (0 < x <= 100)
        open top band -> Pareto mean  L * a / (a - 1)  with  L = 2000 and
                         a = ln((f_prev + f_top) / f_top) / ln(L / L_prev)
                         (f = weighted employees in the last two bands, L_prev = 1500),
                         estimated per survey year.  If a <= 1 the mean is undefined:
                         then 1.5 * L is used and flagged.
HIES incomes (household) — continuous GEL amounts, monthly averages; only deflated.

Deflation (official Geostat CPI, 2010 = 100, monthly, Georgia):
        CPI_y           = mean of the 12 monthly indices of year y
        real_2025(x, y) = x * CPI_2025 / CPI_y
  The survey year's annual average is used because LFS earnings refer to the
  "last worked month" (not always the interview month) and HIES incomes are
  quarterly averages.  Base year 2025 = latest complete CPI year.

Usage:  cd src && python income_standardization.py   → reports/cpi_annual.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_collection as dc

REPORTS = dc.PROJECT_ROOT / "reports"
BASE_YEAR = 2025
TOP_FALLBACK_MULT = 1.5

# band code -> (lower exclusive, upper inclusive) in GEL; upper None = open
BANDS = {
    "2017_2019": {1: (0, 100), 2: (100, 200), 3: (200, 400), 4: (400, 600), 5: (600, 800),
                  6: (800, 1000), 7: (1000, 1500), 8: (1500, 2000), 9: (2000, None)},
    "2020_2025": {1: (0, 100), 2: (100, 150), 3: (150, 200), 4: (200, 300), 5: (300, 400),
                  6: (400, 500), 7: (500, 600), 8: (600, 800), 9: (800, 1000), 10: (1000, 1500),
                  11: (1500, 2000), 12: (2000, None)},
}
TOP_CODE = {"2017_2019": 9, "2020_2025": 12}
# special codes of the earnings variable (from value labels / questionnaire)
NO_EARNINGS, REFUSED = 0, 88
UNDOCUMENTED_2017 = 10        # present in 2017 data, not in labels or questionnaire -> missing (decision 2026-11-27)

# P2 band -> P1 band (common scheme for comparisons only)
COMMON_FROM_P2 = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9}

INCOME_DEFINITIONS = pd.DataFrame([
    dict(income_type="lfs_net_earnings_main_job", dataset="lfs", unit="individual",
         period="last worked month", currency="GEL", gross_net="net",
         form="band (interval); amount = band midpoint / Pareto estimate", nominal_real="nominal + real_2025",
         who="employees in main job",
         source_variable="B17_B18_Net_earnings (2017-19), B24_B25_Net_earnings (2020-25)",
         definition="basic wages and salaries, bonuses, premiums, incentives, payment in kind (LFS questionnaire B17)"),
    *[dict(income_type=t, dataset="hies", unit="household", period="monthly average over the reference quarter",
           currency="GEL", gross_net="as reported (cash received)", form="amount",
           nominal_real="nominal + real_2025", who="all households", source_variable=v, definition=d)
      for t, v, d in [
          ("hies_inc_hired_employment", "ShemDaq", "income from hired employment, all members"),
          ("hies_inc_self_employment", "ShemTviTdasaqm", "income from self-employment"),
          ("hies_inc_agriculture_sales", "Shem_Sof", "income from selling agricultural production"),
          ("hies_inc_property", "Qonebidan", "property income (leasing, interest)"),
          ("hies_inc_pensions_benefits", "PensStipDaxm", "pensions, scholarships, assistance"),
          ("hies_inc_remittances_abroad", "Ucxoetidan", "remittances from abroad"),
          ("hies_inc_gifts", "Axloblebisagan", "money received as gift"),
          ("hies_inc_cash_and_transfers", "ShemosavaliDaTransf", "cash income and transfers = items 1-7"),
          ("hies_inc_non_cash", "ArafuladiMoxmareba", "non-cash income (own production consumed etc.)"),
          ("hies_inc_total", "Shemosavalisul", "total income = cash income and transfers + non-cash"),
      ]],
])


def band_bounds(period: str, code) -> tuple[float, float]:
    b = BANDS[period].get(int(code)) if pd.notna(code) else None
    if b is None:
        return (np.nan, np.nan)
    return (float(b[0]), np.inf if b[1] is None else float(b[1]))


def check_nesting() -> bool:
    """Every 2020-2025 band lies inside its common 2017-2019 band."""
    for p2, p1 in COMMON_FROM_P2.items():
        lo2, hi2 = band_bounds("2020_2025", p2)
        lo1, hi1 = band_bounds("2017_2019", p1)
        if not (lo2 >= lo1 and hi2 <= hi1):
            return False
    return True


def cpi_annual() -> pd.DataFrame:
    """Annual average CPI (2010=100), Georgia, from the official Geostat table."""
    src = next(s for s in dc.SOURCES if s.key == "cpi_2010")
    # sheet name in the Geostat file has a trailing space: "Georgia "
    sheets = pd.read_excel(dc.local_path(src), sheet_name=None, header=None)
    raw = next(v for k, v in sheets.items() if k.strip().lower() == "georgia")
    rows = []
    for _, r in raw.iterrows():
        y = pd.to_numeric(r.iloc[0], errors="coerce")
        months = pd.to_numeric(r.iloc[1:13], errors="coerce")
        if pd.notna(y) and 1990 < y < 2100:
            rows.append(dict(year=int(y), n_months=int(months.notna().sum()), cpi_avg=months.mean()))
    c = pd.DataFrame(rows)
    base = c.loc[(c.year == BASE_YEAR) & (c.n_months == 12), "cpi_avg"]
    if base.empty:
        raise ValueError(f"CPI for base year {BASE_YEAR} incomplete")
    c["deflator_to_2025"] = base.iloc[0] / c.cpi_avg
    c.loc[c.n_months < 12, "deflator_to_2025"] = np.nan      # never deflate with a partial year
    return c


def pareto_top(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year Pareto estimate of the mean of the open top band.
    df: employees with valid bands; columns year, period, earnings_band, weight_annual."""
    out = []
    for (year, period), g in df.groupby(["year", "period"]):
        top = TOP_CODE[period]
        f_top = g.loc[g.earnings_band == top, "weight_annual"].sum()
        f_prev = g.loc[g.earnings_band == top - 1, "weight_annual"].sum()
        L, Lp = BANDS[period][top][0], BANDS[period][top - 1][0]
        alpha = np.log((f_prev + f_top) / f_top) / np.log(L / Lp) if f_top > 0 else np.nan
        ok = bool(np.isfinite(alpha) and alpha > 1)
        mean = L * alpha / (alpha - 1) if ok else TOP_FALLBACK_MULT * L
        out.append(dict(year=year, period=period, n_top=int((g.earnings_band == top).sum()),
                        weighted_top=round(f_top), weighted_prev=round(f_prev),
                        alpha=round(alpha, 3) if np.isfinite(alpha) else np.nan,
                        top_mean=round(mean, 1), method="pareto" if ok else "fallback_1.5xL"))
    return pd.DataFrame(out)


def add_lfs_amounts(df: pd.DataFrame, cpi: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    """Adds earn_lower/upper, earnings_band_common, earn_mid_nominal, earn_mid_real_2025.
    Only rows with earnings_status == 'valid' get amounts."""
    d = df.copy()
    valid = d.earnings_status.eq("valid")
    lo, hi = zip(*[band_bounds(p, c) if v else (np.nan, np.nan)
                   for p, c, v in zip(d.period, d.earnings_band, valid)])
    d["earn_lower"], d["earn_upper"] = lo, hi
    d["earnings_band_common"] = np.where(
        valid, np.where(d.period == "2017_2019", d.earnings_band,
                        d.earnings_band.map(COMMON_FROM_P2)), np.nan)
    mid = (d.earn_lower + d.earn_upper) / 2
    mid = mid.where(d.earn_lower > 0, 50.0).where(valid)          # first band 0-100 -> 50
    topmean = d.year.map(top.set_index("year").top_mean)
    mid = mid.where(~np.isinf(d.earn_upper), topmean)
    d["earn_mid_nominal"] = mid.where(valid)
    d["earn_is_top_band"] = valid & np.isinf(d.earn_upper)
    defl = d.year.map(cpi.set_index("year").deflator_to_2025)
    d["earn_mid_real_2025"] = d.earn_mid_nominal * defl
    return d


def add_hies_real(df: pd.DataFrame, cpi: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    defl = d.year.map(cpi.set_index("year").deflator_to_2025)
    for c in cols:
        d[f"{c}_real_2025"] = d[c] * defl
    return d


if __name__ == "__main__":
    c = cpi_annual()
    c.to_csv(REPORTS / "cpi_annual.csv", index=False)
    INCOME_DEFINITIONS.to_csv(REPORTS / "income_definitions.csv", index=False)
    print("nesting ok:", check_nesting())
    print(c[c.year >= 2016].to_string(index=False))
