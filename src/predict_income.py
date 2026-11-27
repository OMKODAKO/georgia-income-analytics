"""
Stage 05 — Predict estimated real net monthly earnings with the saved model.

The model was trained on Geostat LFS 2020-2025 employees. It predicts the expected earnings
(2025 GEL, net, main job) of a PROFILE — it is a research tool, not a basis for decisions about
real individuals, and its predictions are associations, not causal effects.

Input: CSV or JSON (list of records) with the columns in models/model_metadata.json
       ("features_raw"). Unknown categories are allowed (encoded as all-zero) but reported.
Usage:
  python predict_income.py --demo                         # fictional example profiles
  python predict_income.py --input people.csv --output predictions.csv
  python predict_income.py --input people.json --gender-excluded
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"

# FICTIONAL profiles for demonstration — not records from the survey.
DEMO = [
    dict(gender="female", age=34, potential_experience=12, working_hours=40, household_size=4, survey_year=2025,
         education="Master", occupation="Professionals", industry="Education", region="Tbilisi",
         urban_rural="urban", sector="state", informality="formal", contract="written", part_time="full-time"),
    dict(gender="male", age=34, potential_experience=12, working_hours=40, household_size=4, survey_year=2025,
         education="Master", occupation="Professionals", industry="Education", region="Tbilisi",
         urban_rural="urban", sector="state", informality="formal", contract="written", part_time="full-time"),
    dict(gender="female", age=52, potential_experience=34, working_hours=48, household_size=3, survey_year=2025,
         education="Vocational", occupation="Service & sales", industry="Trade, repair", region="Imereti",
         urban_rural="urban", sector="private", informality="informal", contract="oral", part_time="full-time"),
]


def load(gender_excluded: bool = False):
    import preprocessing  # noqa: F401  (LogTargetRegressor must be importable for unpickling)
    meta = json.loads((MODELS / "model_metadata.json").read_text())
    path = MODELS / ("best_income_model_gender_excluded.joblib" if gender_excluded else "best_income_model.joblib")
    return joblib.load(path), meta


def validate(df: pd.DataFrame, meta: dict, gender_excluded: bool) -> list[str]:
    feats = [f for f in meta["features_raw"] if not (gender_excluded and f == "gender")]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    notes = []
    for c, allowed in meta["categories"].items():
        if c in df and c in feats:
            bad = set(df[c].dropna().astype(str)) - set(allowed)
            if bad:
                notes.append(f"{c}: unknown categories {sorted(bad)} (treated as unseen)")
    for c, (lo, hi) in meta["numeric_ranges"].items():
        if c in df and ((df[c] < lo) | (df[c] > hi)).any():
            notes.append(f"{c}: values outside training range [{lo:g}, {hi:g}] — extrapolation")
    return notes


def predict(records: pd.DataFrame, gender_excluded: bool = False) -> pd.DataFrame:
    model, meta = load(gender_excluded)
    notes = validate(records, meta, gender_excluded)
    feats = [f for f in meta["features_raw"] if not (gender_excluded and f == "gender")]
    pred = model.predict(records[feats])
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("non-finite prediction")
    out = records.copy()
    out["predicted_net_monthly_earnings_gel_2025"] = np.round(pred.astype(np.float64), 1)   # xgboost returns float32
    out.attrs["notes"] = notes
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--output")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--gender-excluded", action="store_true")
    a = ap.parse_args()
    if a.demo:
        df = pd.DataFrame(DEMO)
    elif a.input:
        df = pd.read_json(a.input) if a.input.endswith(".json") else pd.read_csv(a.input)
    else:
        ap.error("give --input or --demo")
    out = predict(df, a.gender_excluded)
    for n in out.attrs["notes"]:
        print("NOTE:", n)
    if a.output:
        out.to_csv(a.output, index=False)
    print(out[["gender", "age", "education", "occupation", "region", "working_hours",
               "predicted_net_monthly_earnings_gel_2025"]].to_string(index=False))


if __name__ == "__main__":
    main()
