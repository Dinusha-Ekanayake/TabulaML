# Changelog

All notable changes to TabulaML. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [0.2.0] - 2026-11-20

### Added
- **Forecasting (time series)** as a third problem type in step 1:
  - Features are built from past values only: target lags, rolling means, calendar parts, and other columns lagged by the horizon.
  - Validation is always time-ordered.
  - Model checks: duplicate time points, series length, irregular gaps.
  - Baselines: **Naive** and **Seasonal naive**.
  - **Hybrid RF + ARIMA**: ARIMA for the time pattern, plus a Random Forest on its residuals.
  - Results show "beats the naive baseline by X %" and a chart of the test period.
  - Step 6 can forecast the next periods, with a chart and a CSV download.
- **Several targets at once:**
  - multi-label and multi-output classification;
  - multi-output regression;
  - averaged metrics, exact-match accuracy and Hamming loss;
  - per-target tables, per-target diagnostics and `prediction_<target>` columns.
- **Background training:**
  - a live progress card (model and fold, elapsed and remaining time);
  - a **Cancel** button that keeps the models that already finished.
- **Training-time estimate** before training: a range, a speed level, and warnings about slow choices.
- **Row cap** (step 3, Evaluation settings): random sample, whole groups, or the latest rows.
- **Per-column imputation choice** in Custom preprocessing, with type-aware options and bulk apply.
- **Uploads:** JSON (records, JSON lines, `{column: [...]}`) and Parquet. Thousands marks with apostrophes (`1'234.5`).
- **Warnings** for small datasets and small test sets.
- **Step 1:** problem-type cards and the standard workflow strip.

### Changed
- `prepare()` handles several targets and builds forecast features. Stratification is used only for single-target classification.
- Progress is reported per fold instead of per model.

## [0.1.0] - 2026-11-20

### Added
- Six-step Streamlit wizard: problem type, data, preprocessing, models, results, predict.
- Schema-driven preprocessing with no dataset-specific code. Leakage-safe pipelines.
- Holdout-first workflow with random, grouped and time-ordered validation.
- Leakage review, an Automatic or Custom pipeline, and a pipeline preview.
- Models:
  - 18 models;
  - voting and stacking;
  - an RF + ANN hybrid;
  - default, manual or auto-tuned hyperparameters.
- Results:
  - a cross-validation leaderboard and out-of-fold diagnostics;
  - a decision threshold and calibration;
  - a locked test set and permutation importance.
- Predict:
  - single records and batch files, with a quality report;
  - `.joblib` and `.pkl` model bundles.
- Bundled public Sri Lankan datasets (tea auction, Colombo dengue, plus a Kaggle telco fetcher), each with a dataset card.
- Light and dark theme picker, a fixed footer, and Streamlit Community Cloud configuration.
