"""
Stage 05 — Train and compare income models (2020-2025 LFS employees; replication 2017-2019).

Protocol (identical for every model and for both experiments):
  * same household-grouped split: train 70% / validation 15% / test 15%   (preprocessing.split)
  * hyper-parameter search ONLY on the training set: RandomizedSearchCV / GridSearchCV with
    3-fold GroupKFold by household, scoring = MAE in GEL
  * model selection on the VALIDATION set (+ CV stability, train-validation gap)
  * the TEST set is used once, at the end, for the reported metrics
  * Experiment A = gender included, Experiment B = gender excluded (same split, same search)

Algorithms: Linear Regression, Ridge, Random Forest, Gradient Boosting (sklearn's histogram
gradient-boosting implementation, HistGradientBoostingRegressor — the classic
GradientBoostingRegressor is single-threaded and too slow for ~57k rows x 80 columns with a CV
search), XGBoost.

Outputs:
  models/best_income_model.joblib                (best algorithm, gender included)
  models/best_income_model_gender_excluded.joblib (same algorithm, gender excluded)
  models/model_metadata.json                     (config, features, training metadata, metrics)
  models/experiments/*.joblib                    (all fitted pipelines except large forests)
  data/final/ml_predictions_2020_2025.csv.gz     (validation + test predictions of every model)
  reports/ml_model_comparison.csv, reports/ml_cv_results.csv, reports/ml_temporal_validation.csv,
  reports/ml_replication_2017_2019.csv
Usage:  cd src && python train_models.py
"""
from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, RandomizedSearchCV
from xgboost import XGBRegressor

import model_evaluation as me
import preprocessing as pp

MODELS = {
    "Linear Regression": (LinearRegression(), None, None),
    "Ridge": (Ridge(), "grid", {"model__regressor__alpha": [0.01, 0.1, 1, 10, 100, 1000]}),
    "Random Forest": (RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=pp.RANDOM_STATE,
                                            max_samples=0.5),
                      "random", {"model__regressor__max_depth": [12, 18, 25, None],
                                 "model__regressor__min_samples_leaf": [5, 10, 20, 40],
                                 "model__regressor__max_features": [0.2, 0.33, 0.5]}),
    "Gradient Boosting": (HistGradientBoostingRegressor(random_state=pp.RANDOM_STATE, early_stopping=False),
                          "random", {"model__regressor__learning_rate": loguniform(0.02, 0.2),
                                     "model__regressor__max_iter": randint(200, 700),
                                     "model__regressor__max_leaf_nodes": randint(15, 63),
                                     "model__regressor__min_samples_leaf": randint(20, 200),
                                     "model__regressor__l2_regularization": loguniform(1e-3, 10)}),
    "XGBoost": (XGBRegressor(tree_method="hist", n_jobs=-1, random_state=pp.RANDOM_STATE, verbosity=0),
                "random", {"model__regressor__learning_rate": loguniform(0.02, 0.2),
                           "model__regressor__n_estimators": randint(200, 800),
                           "model__regressor__max_depth": randint(3, 9),
                           "model__regressor__min_child_weight": randint(1, 30),
                           "model__regressor__subsample": uniform(0.6, 0.4),
                           "model__regressor__colsample_bytree": uniform(0.5, 0.5),
                           "model__regressor__reg_lambda": loguniform(0.1, 20)}),
}
N_ITER = {"Random Forest": 6, "Gradient Boosting": 10, "XGBoost": 10}
EXPERIMENTS = {"A_gender_included": pp.FEATURES_A, "B_gender_excluded": pp.FEATURES_B}
SAVE_EXCLUDE = {"Random Forest"}            # forests are large; saved only if selected as best


def fit_one(name, features, X_tr, y_tr, groups):
    base, kind, space = MODELS[name]
    pipe = pp.make_pipeline(base, features)
    cv = GroupKFold(n_splits=3)
    t0 = time.time()
    if kind is None:
        search = GridSearchCV(pipe, {}, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1, refit=True)
    elif kind == "grid":
        search = GridSearchCV(pipe, space, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1, refit=True)
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=N_ITER[name], cv=cv, scoring="neg_mean_absolute_error",
                                    n_jobs=1, refit=True, random_state=pp.RANDOM_STATE)
    search.fit(X_tr, y_tr, groups=groups)
    i = search.best_index_
    folds = [-search.cv_results_[f"split{k}_test_score"][i] for k in range(3)]
    return search.best_estimator_, dict(best_params={k.replace("model__regressor__", ""): v
                                                    for k, v in search.best_params_.items()},
                                        cv_mae_folds=folds, cv_mae_mean=float(np.mean(folds)),
                                        cv_mae_std=float(np.std(folds)), fit_seconds=round(time.time() - t0, 1))


def run():
    pp.MODELS_DIR.mkdir(exist_ok=True)
    (pp.MODELS_DIR / "experiments").mkdir(exist_ok=True)
    df = pp.load_ml_frame()
    part = pp.split(df)
    tr, va, te = (df[part == s] for s in ("train", "validation", "test"))
    y = pp.TARGET
    preds = df.loc[part != "train", ["individual_id", "household_id", "weight", "gender", "age", "education",
                                     "survey_year", y]].assign(part=part[part != "train"])
    rows, cvrows, fitted = [], [], {}
    for exp, feats in EXPERIMENTS.items():
        for name in MODELS:
            model, info = fit_one(name, feats, tr[feats], tr[y], tr.household_id)
            fitted[(exp, name)] = model
            for s, data in (("train", tr), ("validation", va), ("test", te)):
                m = me.metrics(data[y], model.predict(data[feats]), data.weight)
                rows.append(dict(experiment=exp, model=name, split=s, **m))
            preds[f"pred|{exp}|{name}"] = model.predict(df.loc[part != "train", feats])
            cvrows.append(dict(experiment=exp, model=name, **{k: (json.dumps(v, default=float) if isinstance(v, (dict, list)) else v)
                                                              for k, v in info.items()}))
            if name not in SAVE_EXCLUDE:
                joblib.dump(model, pp.MODELS_DIR / "experiments" / f"{exp}__{name.replace(' ', '_')}.joblib", compress=3)
            print(f"{exp:18} {name:18} val MAE {rows[-2]['MAE']:.1f}  cv {info['cv_mae_mean']:.1f}±{info['cv_mae_std']:.1f}"
                  f"  ({info['fit_seconds']}s)")
    comp = pd.DataFrame(rows)
    cvres = pd.DataFrame(cvrows)
    best = select_best(comp, cvres)
    for exp, path in (("A_gender_included", "best_income_model.joblib"),
                      ("B_gender_excluded", "best_income_model_gender_excluded.joblib")):
        joblib.dump(fitted[(exp, best)], pp.MODELS_DIR / path, compress=3)
    comp["selected_best"] = comp.model == best
    comp.to_csv(me.REPORTS / "ml_model_comparison.csv", index=False)
    cvres.to_csv(me.REPORTS / "ml_cv_results.csv", index=False)
    preds.to_csv(pp.PRED_FILE, index=False, compression="gzip")
    temporal = temporal_validation(df, best, cvres)
    repl = replication(best, cvres)
    write_metadata(df, part, best, comp, cvres, fitted[("A_gender_included", best)], temporal, repl)
    return comp, cvres, best


def select_best(comp: pd.DataFrame, cvres: pd.DataFrame) -> str:
    """Lowest validation MAE (experiment A) among models whose CV std and train-validation gap are not
    worse than 2x the median of all models — accuracy, stability and generalisation together."""
    a = comp[comp.experiment == "A_gender_included"].pivot(index="model", columns="split", values="MAE")
    a["gap"] = a.validation - a.train
    c = cvres[cvres.experiment == "A_gender_included"].set_index("model").cv_mae_std
    ok = a[(a.gap.abs() <= 2 * a.gap.abs().median() + 5) & (c <= 2 * c.median() + 1)]
    return (ok if len(ok) else a).validation.idxmin()


def params_of(best, cvres, exp="A_gender_included"):
    p = json.loads(cvres[(cvres.experiment == exp) & (cvres.model == best)].best_params.iloc[0])
    return {f"model__regressor__{k}": v for k, v in p.items()}


def temporal_validation(df, best, cvres) -> pd.DataFrame:
    """Train 2020-2023, test 2024-2025 (households seen in the training years removed from the test)."""
    rows = []
    early, late = df[df.survey_year <= 2023], df[df.survey_year >= 2024]
    late = late[~late.household_id.isin(early.household_id)]
    for name in (best, "Linear Regression"):
        for exp, feats in EXPERIMENTS.items():
            pipe = pp.make_pipeline(MODELS[name][0], feats).set_params(**params_of(name, cvres, exp))
            pipe.fit(early[feats], early[pp.TARGET])
            rows.append(dict(model=name, experiment=exp, train_years="2020-2023", test_years="2024-2025",
                             n_train=len(early), n_test=len(late),
                             **me.metrics(late[pp.TARGET], pipe.predict(late[feats]), late.weight)))
    t = pd.DataFrame(rows)
    t.to_csv(me.REPORTS / "ml_temporal_validation.csv", index=False)
    return t


def replication(best, cvres) -> pd.DataFrame:
    """Same protocol on the 2017-2019 period (separate methodology; households linked only within a quarter)."""
    df = pp.load_ml_frame("2017_2019")
    part = pp.split(df, "2017_2019")
    tr, te = df[part == "train"], df[part == "test"]
    rows = []
    for name in (best, "Linear Regression"):
        for exp, feats in EXPERIMENTS.items():
            pipe = pp.make_pipeline(MODELS[name][0], feats).set_params(**params_of(name, cvres, exp))
            pipe.fit(tr[feats], tr[pp.TARGET])
            rows.append(dict(period="2017_2019", model=name, experiment=exp, n_train=len(tr), n_test=len(te),
                             **me.metrics(te[pp.TARGET], pipe.predict(te[feats]), te.weight)))
    r = pd.DataFrame(rows)
    r.to_csv(me.REPORTS / "ml_replication_2017_2019.csv", index=False)
    return r


def write_metadata(df, part, best, comp, cvres, model, temporal, repl):
    prep = model.named_steps["prep"]
    meta = dict(
        model_name=best, file="models/best_income_model.joblib",
        companion_file="models/best_income_model_gender_excluded.joblib",
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_source="Geostat Labour Force Survey microdata 2020-2025 (employees, main job)",
        target=pp.TARGET, target_definition="estimated real net monthly earnings, 2025 GEL (band point estimate); "
                                            "model fitted on log scale, predictions back-transformed with Duan smearing",
        smearing_factor=model.named_steps["model"].smearing_,
        features_raw=pp.FEATURES_A, numeric_features=pp.NUMERIC, categorical_features=pp.CATEGORICAL + pp.GENDER,
        features_after_preprocessing=list(prep.get_feature_names_out()),
        categories={c: sorted(map(str, df[c].dropna().unique())) for c in pp.CATEGORICAL + pp.GENDER},
        numeric_ranges={c: [float(df[c].min()), float(df[c].max())] for c in pp.NUMERIC},
        best_params=json.loads(cvres[(cvres.experiment == "A_gender_included") & (cvres.model == best)].best_params.iloc[0]),
        split=dict(method="GroupShuffleSplit by household (DiaryID)", random_state=pp.RANDOM_STATE,
                   sizes=part.value_counts().to_dict()),
        metrics={s: comp[(comp.experiment == "A_gender_included") & (comp.model == best) & (comp.split == s)]
                 .drop(columns=["experiment", "model", "split"]).iloc[0].to_dict() for s in ("train", "validation", "test")},
        temporal_validation=temporal.to_dict(orient="records"), replication_2017_2019=repl.to_dict(orient="records"),
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                      pandas=pd.__version__, numpy=np.__version__),
        intended_use="research: descriptive prediction of earnings bands; NOT for decisions about individuals; "
                     "not causal")
    (pp.MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=float, ensure_ascii=False))


if __name__ == "__main__":
    comp, cvres, best = run()
    pd.set_option("display.width", 250)
    print("BEST:", best)
    print(comp[comp.split != "train"].round(3).to_string(index=False))
