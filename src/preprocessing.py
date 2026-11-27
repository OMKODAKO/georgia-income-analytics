"""
Stage 05 — ML data, features, leakage-safe preprocessing and household-grouped splits.

Target (honest name): ESTIMATED REAL NET MONTHLY EARNINGS of employees, main job, 2025 GEL.
  The LFS publishes earnings only in bands; the target is the band point estimate
  (midpoint; Pareto mean for the open top band) deflated with the CPI (Stage 02).
  -> within-band variation is unobservable, which caps achievable accuracy.
  Models learn log(earnings) and predict in GEL with Duan's smearing correction
  (LogTargetRegressor), so metrics are reported in GEL.

Features (no copies of the target — band codes, bounds, hourly earnings and income-derived
variables are excluded):
  gender (Model A only), age, potential experience, education, occupation (ISCO major group),
  industry (NACE section), region, urban/rural, ownership sector, informality, written
  contract, usual weekly hours, part-time, household size, survey year.
  "Employment status" is constant (all are employees); the job-type variables
  (sector / informality / contract) describe the employment relationship instead.

Split: 70 / 15 / 15 by HOUSEHOLD (DiaryID) — the same person is interviewed in several
quarters/years, so all rows of one household go to the same part (GroupShuffleSplit).
All imputation / scaling / encoding is fitted inside the sklearn Pipeline on training data only.

Usage:  cd src && python preprocessing.py   (prints the dataset audit and writes the split file)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_collection as dc
import survey_analysis as sa

RANDOM_STATE = 42
MAIN_PERIOD = "2020_2025"
MODELS_DIR = dc.PROJECT_ROOT / "models"
SPLIT_FILE = dc.PROJECT_ROOT / "data" / "final" / "ml_split_{p}.csv"
PRED_FILE = dc.PROJECT_ROOT / "data" / "final" / f"ml_predictions_{MAIN_PERIOD}.csv.gz"

NUMERIC = ["age", "potential_experience", "working_hours", "household_size", "survey_year"]
CATEGORICAL = ["education", "occupation", "industry", "region", "urban_rural", "sector", "informality",
               "contract", "part_time"]
GENDER = ["gender"]
FEATURES_A = GENDER + NUMERIC + CATEGORICAL          # Model A — gender included
FEATURES_B = NUMERIC + CATEGORICAL                   # Model B — gender excluded
TARGET = "earnings_real_2025"
# columns that must NEVER be features (target or derived from it)
LEAKAGE_COLUMNS = ["band_code", "band_code_common", "income_lower", "income_upper", "income_amount",
                   "income_amount_real_2025", "is_top_band", "hourly_nominal", "hourly_real_2025", "y", "log_y",
                   "lo_real", "hi_real", "hourly", "log_hourly", TARGET]


def load_ml_frame(period: str = MAIN_PERIOD) -> pd.DataFrame:
    """One row per employee-interview with a valid earnings band (real data only)."""
    d = sa.load_period(period)
    out = pd.DataFrame({
        "individual_id": d.individual_id, "household_id": d.cluster, "weight": d.w,
        "gender": d.gender, "age": d.age.astype(float), "potential_experience": d.potential_experience,
        "working_hours": d.working_hours, "household_size": d.household_size.astype(float),
        "survey_year": d.survey_year.astype(float),
        "education": d.education_eda, "occupation": d.occupation_label, "industry": d.industry_label,
        "region": d.region_eda, "urban_rural": d.urban_rural, "sector": d.sector,
        "informality": d.informal.map({0: "formal", 1: "informal"}).fillna("not_defined (agriculture/unknown)"),
        "contract": d.contract_written.map({1: "written", 0: "oral"}),
        "part_time": d.part_time.map({0: "full-time", 1: "part-time"}),
        TARGET: d.y,
    })
    return out.reset_index(drop=True)


def split(df: pd.DataFrame, period: str = MAIN_PERIOD, save: bool = True) -> pd.Series:
    """70/15/15 household-grouped split; returns a Series 'train'/'validation'/'test'."""
    g = df.household_id.to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr, rest = next(first.split(df, groups=g))
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    va_r, te_r = next(second.split(rest, groups=g[rest]))
    part = pd.Series("train", index=df.index)
    part.iloc[rest[va_r]] = "validation"
    part.iloc[rest[te_r]] = "test"
    if save:
        pd.DataFrame({"individual_id": df.individual_id, "household_id": df.household_id, "part": part}) \
            .to_csv(str(SPLIT_FILE).format(p=period), index=False)
    return part


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """Imputation, scaling and one-hot encoding — fitted on training data only (inside Pipeline)."""
    num = [c for c in features if c in NUMERIC]
    cat = [c for c in features if c in CATEGORICAL + GENDER]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], verbose_feature_names_out=True)


class LogTargetRegressor(RegressorMixin, BaseEstimator):   # mixin first (sklearn >= 1.6 tags)
    """Fits `regressor` on log(y); predicts exp(f(x)) * smearing, smearing = mean(exp(train residuals))
    (Duan 1983) so predictions estimate the MEAN in GEL rather than the median."""

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y, **fit_params):
        y = np.asarray(y, float)
        self.regressor_ = clone(self.regressor).fit(X, np.log(y), **fit_params)
        resid = np.log(y) - self.regressor_.predict(X)
        self.smearing_ = float(np.mean(np.exp(resid)))
        return self

    def predict_log(self, X):
        return self.regressor_.predict(X)

    def predict(self, X):
        return np.exp(self.predict_log(X)) * self.smearing_


def make_pipeline(regressor, features: list[str]) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(features)), ("model", LogTargetRegressor(regressor))])


def audit(df: pd.DataFrame, part: pd.Series) -> dict:
    a = dict(rows=len(df), households=int(df.household_id.nunique()),
             women=int((df.gender == "female").sum()), men=int((df.gender == "male").sum()),
             years=sorted(df.survey_year.astype(int).unique().tolist()),
             target=f"{TARGET}: band point estimate, real 2025 GEL, net, monthly (last worked month)",
             target_distinct_values=int(df[TARGET].nunique()),
             missing={c: int(df[c].isna().sum()) for c in FEATURES_A if df[c].isna().any()},
             split_sizes=part.value_counts().to_dict(),
             households_in_two_parts=int(pd.DataFrame({"h": df.household_id, "p": part})
                                         .groupby("h").p.nunique().gt(1).sum()),
             leakage_columns_in_features=sorted(set(FEATURES_A) & set(LEAKAGE_COLUMNS)))
    return a


if __name__ == "__main__":
    df = load_ml_frame()
    part = split(df)
    print(json.dumps(audit(df, part), indent=2, default=str))
