# TabulaML — Project Definition

**Full title:** TabulaML: An Interactive Machine Learning Workbench with Generalized Data Preprocessing and Ensemble Model Comparison

## 1. Problem statement

Practitioners and students who want to compare Random Forest with other ensemble learners (XGBoost, LightGBM, CatBoost, bagging, stacking) on their own tabular data repeat the same work for every dataset: they clean messy columns, choose encodings, handle missing values and class imbalance, avoid data leakage, train several models, compare metrics and then wire up a way to make predictions. That work is slow and easy to get wrong, especially preprocessing and leakage control. Most of all, it is usually hard-coded to one dataset.

TabulaML is one interactive application for this. A user picks a problem type, loads any supported tabular dataset, and gets a **schema-driven preprocessing pipeline that is generated automatically for that dataset and can be adjusted**. They then train and compare one or more models, inspect evaluation metrics and use the trained models to make predictions. The application code never changes per dataset.

## 2. Objectives

1. Build a **schema-driven, dataset-independent preprocessing engine**. It detects column types, cleans common data-quality problems and builds a model-appropriate pipeline without any dataset-specific code.
2. Provide an **Automatic mode** (recommended defaults, each with a written explanation) and a **Custom mode** (global and per-column overrides).
3. Guarantee **leakage-safe validation**: every learned preprocessing step is fitted only inside training folds, and the test set is held out before anything is fitted.
4. Support **classification and regression** with individual models, bagging, boosting, voting, stacking and an RF + ANN hybrid.
5. Present **comparative evaluation**: a leaderboard, cross-validation scores with their spread, test-set metrics, diagnostic plots and feature importance.
6. Allow **prediction** with any trained model through a single-record form or a batch CSV, using the same fitted pipeline.
7. Ship **two bundled Sri Lankan-context sample datasets** (one for classification, one for regression) that go through exactly the same workflow as uploaded data.
8. Show **generalization** by running the same app unchanged on other public datasets, and compare RF with gradient-boosting methods.

## 3. Scope

### In scope
- Tabular datasets from CSV, XLSX or XLS files, up to about 200 MB and 1 million cells by default (configurable).
- Problem types: binary classification, multiclass classification and regression.
- Column types: numeric, categorical, boolean, datetime, identifier (dropped), constant (dropped) and free text. Free text is dropped by default, with an optional simple length/word-count feature.
- Datetime feature extraction: year, month, day-of-week, day-of-year and an "age in days" relative to a reference date.

### Out of scope (stated as limitations)
- Images, audio, long free text (NLP), graph data.
- Time-series forecasting with temporal cross-validation. The RF + ARIMA hybrid is an optional stretch module (Phase S3), not part of the core.
- Multi-label classification, ranking, clustering.
- Training on a cluster or GPU; everything runs on a single CPU machine.

TabulaML is defensibly "general" across **supported tabular classification and regression datasets**, not across every possible dataset. Unsupported inputs are detected and explained instead of being processed silently.

## 4. Users and user stories

| # | As a… | I want to… | So that… |
|---|---|---|---|
| U1 | student/analyst | choose classification or regression first | the app only offers valid targets, models and metrics |
| U2 | user | upload my own CSV/Excel file **or** load a Sri Lankan sample | I can try the workflow without my own data |
| U3 | user | see a data-quality report (types, missing values, cardinality, outliers, warnings) | I understand my data before modelling |
| U4 | user | accept automatic preprocessing or override it per column | I keep control without writing code |
| U5 | user | select one or many models, including bagging, boosting, voting and stacking | I can compare RF against XGBoost, LightGBM and CatBoost |
| U6 | user | press **Train & Predict** and watch progress | I know the run is working |
| U7 | user | see a leaderboard, CV mean ± std, test metrics and plots | I can choose the best model with evidence |
| U8 | user | enter one record in a form, or upload a batch CSV, and get predictions (with probabilities) | I can actually use the model |
| U9 | user | download predictions, the results report and the trained model bundle | I can reuse the outputs |

## 5. Functional requirements

### FR-1 Problem type (screen 1)
- Choose Classification or Regression. This choice filters targets, models, metrics and imbalance options.

### FR-2 Data source (screen 2)
- **Upload my dataset:** CSV, XLSX or XLS. Encoding fallback (utf-8, then utf-8-sig, then latin-1), delimiter sniffing, choice of Excel sheet.
- **Use a Sri Lankan sample dataset:** loads the bundled CSV and its dataset card. Samples use the same code path as uploads.
- Dataset preview (first rows, shape, memory use).
- Target column selection, validated against the problem type:
  - Classification: at least 2 classes, and each class has at least `cv_folds` rows. A warning is shown if there are more than 50 classes.
  - Regression: the target must be numeric (or numeric after cleaning), with more than 10 unique values. A warning is shown if it looks like class labels.
  - Rows with a missing target are dropped, and the number dropped is reported.

### FR-3 Data profile and preprocessing (screen 3) — the core of the project
See §6.

### FR-4 Model selection dashboard (screen 4)
- Model cards grouped as **Baselines/Linear**, **Trees & Bagging**, **Boosting**, **Other (SVM/KNN/ANN)** and **Ensembles (Voting / Stacking / Hybrid)**.
- Multi-select, with a "Recommended set" shortcut (RF, XGBoost, LightGBM, CatBoost, Logistic/Ridge).
- Voting and stacking use base learners chosen by the user. The stacking meta-learner is Logistic Regression or Ridge.
- Run settings: test size, CV folds, random seed, primary metric, and an optional quick hyperparameter search (stretch).
- A **Train & Predict** button with a progress indicator for each model. If one model fails, its error is shown and the remaining models still run.

### FR-5 Results (screen 5)
- A leaderboard sorted by the primary metric: CV mean ± std, test score, all secondary metrics, fit time and predict time. The best model is highlighted.
- Comparison charts: a bar chart of test scores and a box plot of CV fold scores for each model.
- Per-model detail:
  - Classification: confusion matrix, ROC curve, PR curve and classification report.
  - Regression: predicted vs actual and residual plots.
  - Both: permutation feature importance on the original columns.
- Download: leaderboard CSV and a full JSON run report.

### FR-6 Predict (screen 6)
- Choose a trained model (the best one is the default).
- **Single prediction:** a form generated automatically from the schema. Numbers get number inputs prefilled with the median, categories get dropdowns of seen values plus free text, booleans get checkboxes and dates get date pickers. Classification shows class probabilities.
- **Batch prediction:** upload a CSV. Columns are validated: missing required columns are an error, extra columns are ignored with a warning, and types are coerced by the same cleaner. Predictions and probabilities are appended, and the result can be downloaded.
- Download the model bundle (`.joblib` with metadata).

## 6. Generalized preprocessing requirements (pass/fail criterion)

| ID | Requirement | Behaviour |
|---|---|---|
| P1 | Generic cleaning | Trim whitespace; turn placeholder tokens (`"", "?", "NA", "N/A", "na", "null", "None", "-", "--", "nan"`) into missing; parse numeric strings with currency prefixes or thousand separators (`"Rs. 1,250"` becomes `1250`); map boolean-like values (`yes/no, y/n, true/false, t/f`, and `1/0` when those are the only values) to booleans; optionally normalise category case. |
| P2 | Dynamic schema detection | Infer each column as numeric, categorical, boolean, datetime, identifier, constant or text from its values, not only its dtype. |
| P3 | User type overrides | Any inferred type can be changed in the UI, and the pipeline is rebuilt. |
| P4 | Dynamic target selection | Any valid column can be the target (see FR-2). |
| P5 | Feature selection | Include or exclude any column. IDs, constant columns, columns more than 60 % missing and suspected leakage columns are excluded by default, and each exclusion is explained. |
| P6 | Missing values | Numeric: median, mean, constant or KNN, plus an optional missing-indicator column. Categorical: most frequent or the constant `"__missing__"`. Datetime: extracted parts are imputed as numeric. |
| P7 | Categorical handling | Low cardinality: one-hot (linear/neural models) or ordinal (tree models), with rare categories grouped below a frequency threshold. High cardinality (more than `high_card_threshold`, default 20): cross-fitted target encoding. **Unseen categories at prediction time never crash.** |
| P8 | Numeric processing | Scaling (standard/robust/min-max/none), Yeo-Johnson for skewed columns (\|skew\| > 1), IQR/percentile outlier clipping. These are applied only where they help the model family (see P10). |
| P9 | Imbalance (classification only) | None, class weights, SMOTE or random undersampling. Resampling is applied only to training folds, inside the pipeline. |
| P10 | Model-aware pipelines | **Tree profile** (RF, ExtraTrees, GB, XGB, LGBM, CatBoost, DT, Bagging): impute plus ordinal/target encoding, no scaling or power transforms. **Scale-sensitive profile** (linear, SVM, KNN, MLP): impute, one-hot/target encoding, outlier clipping, Yeo-Johnson and scaling. |
| P11 | Target handling | Classification labels are encoded to 0..k-1 and decoded back for display. For regression, a `log1p` target transform is recommended when the target skew is above 1 and the target is non-negative. |
| P12 | Leakage safety | Split first. All fitted steps live inside one Pipeline. CV fits the whole pipeline in each fold. A leakage detector flags features that are nearly identical to the target, whose missingness predicts the target, or that are highly correlated with it. |
| P13 | Consistent inference | The exact fitted pipeline, cleaner included, is reused for single and batch prediction and is saved in the model bundle. |
| P14 | Unsupported input | Clear errors for: no usable features, a target that does not suit the problem type, a file too large, fewer than 30 rows after cleaning, or all features being text or image paths. |
| P15 | Explainability | Automatic mode outputs a human-readable explanation for every decision (for example, "`monthly_charge`: 4.1 % missing, skew 2.3, so median imputation plus Yeo-Johnson (scale-sensitive models only)"). |

## 7. Sample datasets (local, Sri Lankan)

**Genuine, publicly available Sri Lankan datasets** (used for the main results):

| Key | Task | Rows | Target | Source and licence |
|---|---|---|---|---|
| `lk_colombo_dengue` | Regression | 279 weeks | `dengue_cases` | R package *colmozzie* (Thiyanga Talagala), compiled from the Weekly Epidemiological Reports of the Epidemiology Unit, Ministry of Health, Sri Lanka; weather variables per week. CC0. Bundled. |
| `lk_telco_churn` | Binary classification | 15,000 (stratified sample of 65,005) | `churn` (~23 % Yes) | Kaggle "Real World Customer Churn Dataset in Telco Domain" (Lasal Jayawardena): anonymised usage of customers of a large Sri Lankan operator, Jan-Mar 2023. CC0. Built locally by `scripts/fetch_public_datasets.py` because Kaggle needs a free login. |

**Synthetic Sri Lankan-context datasets** (preprocessing stress tests, always labelled "Synthetic"):

| Key | Task | Rows | Target | Purpose |
|---|---|---|---|---|
| `lk_synthetic_telecom_churn` | Binary classification | ~5,000 | `churned` | Mixed types and deliberate messiness |
| `lk_synthetic_patient_los` | Regression | ~3,000 | `length_of_stay_days` | Right-skewed target (log1p), RF + ANN hybrid |

The synthetic data is deliberately messy so the preprocessing engine has something to prove: missing values, placeholder tokens, `"Rs. 1,250"` strings, inconsistent case and whitespace, `Y/yes/No` booleans, outliers, duplicate rows, an ID column, a date column and **one planted leakage column per dataset** (`disconnection_date` / `total_bill_lkr`). The real telco data keeps one post-outcome column (`active_days_61_90_any_service`) as a real-world leakage example. The leakage detector flags all three.

**Adding another dataset:** drop a CSV and a dataset-card JSON into `data/samples/`. It is discovered automatically; no code changes.

Every sample has a **dataset card** shown in the app: source and provenance, Sri Lankan context, licence, collection period, feature dictionary with units, target definition, known limitations, anonymisation notes, and columns recommended for exclusion.

## 8. Non-functional requirements

- **Reproducibility:** one `random_state` drives the splits, models and samplers. The run report stores library versions.
- **Performance:** the recommended set (5 models, 5-fold CV) on the 5k-row churn sample finishes in under 2 minutes on a laptop CPU.
- **Robustness:** one failing model never aborts the run, and there are no unhandled exceptions in the UI.
- **Quality:** at least 80 % test coverage on `tabulaml/` excluding `ui/`; ruff clean; functions under 50 lines; files under 400 lines.
- **Security:** uploaded files are parsed with pandas only (never `eval` or pickle from users). Model bundles are only loaded if the app itself produced them, and the UI warns about this.

## 9. Evaluation metrics

| Task | Metrics | Default primary |
|---|---|---|
| Binary classification | Accuracy, Balanced accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC (average precision), MCC, Log-loss; confusion matrix, ROC and PR curves | F1 (if minority < 35 %), otherwise ROC-AUC |
| Multiclass classification | Accuracy, Balanced accuracy, macro and weighted P/R/F1, ROC-AUC (OvR macro), MCC, Log-loss | F1-macro |
| Regression | MAE, RMSE, R², MedAE, MAPE (only when there are no zero targets) | RMSE |

## 10. Research component (from the lecture)

- **RF vs gradient boosting (XGBoost, LightGBM, CatBoost):** a repeated-CV benchmark on both local samples and three standard public datasets, reporting mean ± std, fit time and a Wilcoxon signed-rank test between RF and the best booster.
- **Hybridisation:** RF + ANN for tabular regression, implemented as (a) stacking RF + MLP and (b) a residual hybrid where the MLP learns the RF residuals. RF + ARIMA for time series is an optional stretch module.
- **Strengths and limitations on tabular vs image/text data:** a discussion section in the report.

## 11. Deliverables

1. A working Streamlit app (`streamlit run app.py`).
2. A Python package `tabulaml/` with tests (at least 80 % coverage).
3. Two local sample datasets with dataset cards and a generator script.
4. Benchmark script and results (`reports/benchmark_results.csv` plus figures).
5. README, user guide, and an architecture document with diagrams.
6. A project report covering the problem, data, preprocessing design, experiments, results, discussion, limitations and AI-use disclosure if required.
