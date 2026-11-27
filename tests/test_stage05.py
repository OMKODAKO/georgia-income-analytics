"""Stage 05 ML tests (run after train_models.py, model_evaluation.py, feature_importance.py,
fairness_analysis.py)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import model_evaluation as me   # noqa: E402
import predict_income as pi     # noqa: E402
import preprocessing as pp      # noqa: E402

REP = ROOT / "reports"
COMP = pd.read_csv(REP / "ml_model_comparison.csv")
CV = pd.read_csv(REP / "ml_cv_results.csv")
PRED = me.load_predictions()
SPLIT = pd.read_csv(str(pp.SPLIT_FILE).format(p=pp.MAIN_PERIOD))
META = json.loads((pp.MODELS_DIR / "model_metadata.json").read_text())
BEST = COMP[COMP.selected_best].model.iloc[0]


@pytest.fixture(scope="module")
def df():
    return pp.load_ml_frame()


# 1. pipeline ----------------------------------------------------------------------------
def test_pipeline_fits_and_predicts(df):
    s = df.sample(3000, random_state=0)
    pipe = pp.make_pipeline(Ridge(), pp.FEATURES_A).fit(s[pp.FEATURES_A], s[pp.TARGET])
    p = pipe.predict(s[pp.FEATURES_A])
    assert np.isfinite(p).all() and (p > 0).all()
    Xt = pipe.named_steps["prep"].transform(s[pp.FEATURES_A])
    assert Xt.shape[0] == 3000 and not np.isnan(Xt).any()


def test_log_target_smearing(df):
    s = df.sample(3000, random_state=1)
    pipe = pp.make_pipeline(Ridge(), pp.FEATURES_A).fit(s[pp.FEATURES_A], s[pp.TARGET])
    m = pipe.named_steps["model"]
    assert m.smearing_ > 1                                     # exp(E[log]) < E[y]: correction raises predictions
    assert np.allclose(pipe.predict(s[pp.FEATURES_A]),
                       np.exp(m.predict_log(pipe.named_steps["prep"].transform(s[pp.FEATURES_A]))) * m.smearing_)


# 2. missing values ----------------------------------------------------------------------
def test_missing_and_unknown_values_handled(df):
    model = joblib.load(pp.MODELS_DIR / "best_income_model.joblib")
    x = df[pp.FEATURES_A].head(5).copy()
    x.loc[x.index[0], "working_hours"] = np.nan
    x.loc[x.index[1], "education"] = np.nan
    x.loc[x.index[2], "region"] = "Atlantis (unseen)"
    p = model.predict(x)
    assert np.isfinite(p).all() and (p > 0).all()


# 3. no train/test leakage ---------------------------------------------------------------
def test_households_not_shared_between_parts():
    assert SPLIT.groupby("household_id").part.nunique().max() == 1
    sizes = SPLIT.part.value_counts(normalize=True)
    assert abs(sizes.train - .70) < .02 and abs(sizes.validation - .15) < .02 and abs(sizes.test - .15) < .02


def test_no_target_derived_features():
    assert not set(pp.FEATURES_A) & set(pp.LEAKAGE_COLUMNS)
    assert pp.TARGET not in pp.FEATURES_A


def test_preprocessing_fitted_on_training_rows_only(df):
    model = joblib.load(pp.MODELS_DIR / "best_income_model.joblib")
    scaler = model.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    tr = df[SPLIT.part.to_numpy() == "train"]
    assert scaler.n_samples_seen_ == len(tr)                    # 9. test/validation never seen when fitting
    assert scaler.mean_[pp.NUMERIC.index("age")] == pytest.approx(tr.age.mean())


# 4. models train ------------------------------------------------------------------------
def test_all_models_trained_for_both_experiments():
    assert COMP.model.nunique() == 5 and COMP.experiment.nunique() == 2
    assert len(COMP) == 5 * 2 * 3
    assert (COMP.R2 > 0).all()
    for f in (pp.MODELS_DIR / "experiments").glob("*.joblib"):
        assert joblib.load(f) is not None


# 5. metrics -----------------------------------------------------------------------------
def test_metric_formulas():
    y, p = np.array([100., 200, 300, 400]), np.array([110., 190, 330, 360])
    m = me.metrics(y, p)
    assert m["MAE"] == pytest.approx(22.5) and m["MSE"] == pytest.approx((100 + 100 + 900 + 1600) / 4)
    assert m["RMSE"] == pytest.approx(np.sqrt(675)) and m["MedAE"] == pytest.approx(20)
    assert m["R2"] == pytest.approx(1 - 2700 / np.sum((y - y.mean()) ** 2))


def test_reported_test_metrics_recomputed():
    te = PRED[PRED.part == "test"]
    for (exp, name), r in COMP[COMP.split == "test"].set_index(["experiment", "model"]).iterrows():
        m = me.metrics(te[pp.TARGET], te[f"pred|{exp}|{name}"])
        assert r.MAE == pytest.approx(m["MAE"]) and r.R2 == pytest.approx(m["R2"])


# 6. A and B comparable ------------------------------------------------------------------
def test_experiments_share_split_and_protocol():
    for name in COMP.model.unique():
        a = COMP[(COMP.model == name) & (COMP.experiment == "A_gender_included")].set_index("split").n
        b = COMP[(COMP.model == name) & (COMP.experiment == "B_gender_excluded")].set_index("split").n
        assert (a == b).all()
    assert set(CV.groupby("model").experiment.nunique()) == {2}
    assert pp.FEATURES_B == [f for f in pp.FEATURES_A if f != "gender"]


# 7. save / load -------------------------------------------------------------------------
def test_saved_model_reproduces_stored_predictions(df):
    model = joblib.load(pp.MODELS_DIR / "best_income_model.joblib")
    te = df[SPLIT.part.to_numpy() == "test"]
    p = model.predict(te[pp.FEATURES_A])
    assert np.allclose(p, PRED[PRED.part == "test"][f"pred|A_gender_included|{BEST}"].to_numpy(), rtol=1e-6)
    assert META["model_name"] == BEST and META["features_raw"] == pp.FEATURES_A
    for k in ("best_params", "metrics", "split", "versions", "smearing_factor", "features_after_preprocessing"):
        assert k in META


# 8. predictions are real numbers --------------------------------------------------------
@pytest.mark.parametrize("excluded", [False, True])
def test_predict_script_returns_finite_gel(excluded):
    out = pi.predict(pd.DataFrame(pi.DEMO), gender_excluded=excluded)
    p = out.predicted_net_monthly_earnings_gel_2025
    assert p.dtype.kind == "f" and np.isfinite(p).all() and p.between(50, 6000).all()


def test_predict_script_rejects_missing_columns():
    with pytest.raises(ValueError):
        pi.predict(pd.DataFrame(pi.DEMO).drop(columns=["education"]))


# 9. test set not used for hyper-parameter search ---------------------------------------
def test_search_used_training_rows_only():
    n_train = (SPLIT.part == "train").sum()
    for f in (pp.MODELS_DIR / "experiments").glob("*.joblib"):
        m = joblib.load(f)
        assert m.named_steps["prep"].named_transformers_["num"].named_steps["scale"].n_samples_seen_ == n_train
    assert (CV.cv_mae_std >= 0).all() and CV.best_params.notna().all()


# 10. fairness metrics -------------------------------------------------------------------
def test_fairness_metrics_recomputed():
    f = pd.read_csv(REP / "ml_fairness_by_gender.csv")
    te = PRED[PRED.part == "test"]
    for (name, exp), _ in f.groupby(["model", "experiment"]):
        for sex in ("female", "male"):
            g = te[te.gender == sex]
            e = g[f"pred|{exp}|{name}"] - g[pp.TARGET]
            r = f[(f.model == name) & (f.experiment == exp) & (f.gender == sex)].iloc[0]
            assert r.n == len(g) and r.MAE == pytest.approx(e.abs().mean()) and r.mean_error == pytest.approx(e.mean())
            assert r.mean_actual == pytest.approx(g[pp.TARGET].mean())


def test_figures_created():
    names = ["01_actual_vs_predicted", "02_residual_distribution", "03_model_performance_comparison",
             "04_feature_importance", "05_shap_summary", "06_gender_prediction_comparison",
             "07_error_distribution_by_gender", "08_learning_curve", "09_cross_validation_results"]
    for n in names:
        assert (me.FIG / f"{n}.png").stat().st_size > 20_000, n


def test_notebook_executed():
    import nbformat
    nb = nbformat.read(ROOT / "notebooks" / "05_machine_learning.ipynb", as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.get("execution_count") for c in code)
    assert not any(o.get("output_type") == "error" for c in code for o in c.outputs)
