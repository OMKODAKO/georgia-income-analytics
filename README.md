# Georgia Income Analytics — Gender & Income Study

Reproducible study of the gender earnings gap in Georgia using official Geostat microdata.

**Main result (2020-2025, employees):** women earn 25.5% less than men on average (median 28.8%; hourly median 18.9%); with equal observed characteristics the gap is ~17%. See `reports/final_research_report.md`.

| Stage | Status | Report |
|---|---|---|
| 00 Research & Feasibility | ✅ | `reports/stage00_feasibility_report.md` |
| 01 Data Collection | ✅ | `reports/stage_01_report.md` |
| 02 Database & Cleaning | ✅ | `reports/stage_02_report.md` |
| 03 EDA | ✅ | `reports/stage_03_report.md` |
| 04 Statistical Analysis | ✅ | `reports/stage_04_report.md` |
| 05 Machine Learning | ✅ | `reports/stage_05_report.md` |
| 06 Dashboard | ✅ | `reports/stage_06_report.md` |
| 07 Final Research Report | ✅ | **`reports/final_research_report.md`** |

## Reports
- **Final research report:** [`reports/final_research_report.md`](reports/final_research_report.md)
- **Plain-language report (Georgian):** [`reports/plain_language_report.html`](reports/plain_language_report.html)
- Stage reports: `reports/stage00_feasibility_report.md`, `reports/stage_01_report.md` … `reports/stage_06_report.md`

## Data
The raw Geostat files (~800 MB), the SQLite database and the processed tables are **not stored in this
repository**. They are public data and are re-created by the pipeline below: `src/data_collection.py`
downloads every file from geostat.ge and checks it against `data/raw/manifest.json` (URL, size, SHA-256).
Source: National Statistics Office of Georgia (Geostat), www.geostat.ge.

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/data_collection.py           # download raw data (skips files already present)
.venv/bin/python src/data_collection.py --check-links
cd src
../.venv/bin/python inspect_raw.py                # extract ZIPs + variable inventory
../.venv/bin/python feasibility_lfs.py            # Stage 00 earnings coverage
../.venv/bin/python data_profile.py               # technical profile of every table
../.venv/bin/python data_integrity.py             # reliability + cross-year consistency
../.venv/bin/python data_dictionary.py            # data dictionaries
../.venv/bin/python variable_mapping.py           # Stage 02: harmonised variable mapping
../.venv/bin/python data_cleaning.py              # clean / analytical / ML layers + cleaning log
../.venv/bin/python data_validation.py            # quality summary + outliers
../.venv/bin/python database.py                   # SQLite database/income_data.db
../.venv/bin/python eda.py                        # Stage 03: EDA tables + reports/figures/
../.venv/bin/python build_eda_notebook.py         # executes notebooks/03_exploratory_data_analysis.ipynb
../.venv/bin/python survey_analysis.py            # Stage 04: survey design summary
../.venv/bin/python hypothesis_testing.py         # tests, effect sizes, gap CIs (bootstrap)
../.venv/bin/python regression_analysis.py        # regression ladder, interval regression, diagnostics
../.venv/bin/python decomposition.py              # Oaxaca-Blinder
../.venv/bin/python build_stats_notebook.py       # executes notebooks/04_statistical_analysis.ipynb
../.venv/bin/python preprocessing.py              # Stage 05: ML data audit + household-grouped split
../.venv/bin/python train_models.py               # 5 models x (gender in / out), CV search (~9 min)
../.venv/bin/python model_evaluation.py           # metrics, learning curve, CV, temporal validation
../.venv/bin/python feature_importance.py         # permutation importance, SHAP, partial dependence
../.venv/bin/python fairness_analysis.py          # errors by sex / age / education, gender flip, proxies
../.venv/bin/python build_ml_notebook.py          # executes notebooks/05_machine_learning.ipynb
../.venv/bin/python predict_income.py --demo      # use the saved model (fictional profiles)
cd .. && .venv/bin/python -m pytest -q tests
```

## Dashboard
```bash
.venv/bin/streamlit run dashboard/app.py     # http://localhost:8501
```

## Layout
```
data/raw/labour_force_survey/   LFS 2017-2025 ZIPs (+ extracted/ byte-identical copies)
data/raw/household_survey/      HIES 2017-2025 ZIPs (+ extracted/)
data/raw/metadata/              codebooks, methodology, questionnaires
data/raw/official_aggregates/   Geostat wage, gender-pay-gap and CPI tables
data/raw/manifest.json          URL, listing page, size, SHA-256, download time per file
data/raw/download_log.csv       every download / skip / failure / link check
data/processed/                 clean data per source and year (.csv.gz)
data/final/                     analytical datasets (per LFS period, HIES) and ML datasets (LFS employees)
database/income_data.db         SQLite: data_sources, individuals, income, households, household_income, ...
dashboard/                      Streamlit app (app.py) + data layer (data.py)
models/                         best_income_model.joblib (+ gender-excluded twin), model_metadata.json
src/                            pipeline code
reports/                        reports, profiles, data_dictionary/
tests/                          automated checks per stage
```
Raw files are never modified; all derived outputs are written outside `data/raw/` (except the extracted copies).

Data source: National Statistics Office of Georgia (Geostat), www.geostat.ge
