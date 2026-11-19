# TabulaML user guide

Start the app with `streamlit run app.py`. The sidebar shows the six steps. You can always go back, but changing an earlier choice clears the later steps, so results never mix settings. **Start over** resets everything.

## 1. Problem type
Choose **Classification** (predict a category, for example churn Yes/No) or **Regression** (predict a number, for example weekly dengue cases). This choice decides which targets, models and metrics are offered.

## 2. Data
- **Upload my dataset:** CSV, TXT, XLSX or XLS, up to 200 MB and 1 million cells. The delimiter and text encoding are detected automatically. Excel files let you pick a sheet. Duplicate column names are renamed, and rows with more values than headers are rejected with a message rather than silently misaligned.
- **Use a Sri Lankan sample dataset:** pick a sample, read its dataset card (source, licence, features, limitations), then **Load this sample**. Real public datasets show a green badge and synthetic ones an amber badge.
- **Target column:** the app checks it against the problem type. For example, regression needs a numeric target, and every class needs enough rows for the test split plus cross-validation. If the column looks like the other problem type, the app says so.

## 3. Preprocessing
**Data quality report:** row and column counts, duplicates, a per-column table (detected type, missing share, unique values, skew, outlier share, examples, warnings), a missing-values chart and the target distribution. **Leakage warnings** appear here, for columns that would not be known at prediction time, such as a disconnection date in a churn table.

**Preprocessing**
- *Automatic*: a table with the decision and the reason for every column, and dataset-level notes (duplicates dropped, imbalance handling, target transform).
- *Custom*: edit any column: include/exclude, type, imputation, encoding, scaling, transform and outlier handling. Global options cover missing-value indicators, merging spelling variants, rare-category grouping, the high-cardinality threshold, class-imbalance handling (none / class weights / SMOTE / undersampling) and the regression target transform (`log1p`).

"auto" means TabulaML decides per model family: tree models get no scaling and ordinal encoding, while linear, SVM, KNN and neural models get one-hot encoding, skew correction, extreme-outlier clipping and scaling. Data-dependent choices, such as which columns are skewed, are learned on training folds only.

**Pipeline preview:** the exact step list per column for each model family.

**Evaluation settings:** test-set share, number of cross-validation folds and random seed.

## 4. Models
Tick models (hover for a description), or click **Recommended set**. For **Voting** and **Stacking**, choose at least two base models; each base keeps its own preprocessing. Choose the **primary metric** that ranks the leaderboard, then press **Train & Predict**. A progress bar shows each model. If a model fails, its error is reported and the others still finish.

## 5. Results
- **Leaderboard:** cross-validated mean and standard deviation of every metric, held-out test metrics, and fit/predict times. Download it as CSV, or download a JSON run report with the full configuration, decisions, metrics and library versions.
- **Comparison:** test scores by model, and the spread of cross-validation scores (stability). A sentence compares Random Forest with the best gradient booster.
- **Model details:**
  - Classification: confusion matrix, ROC and precision-recall curves.
  - Regression: predicted vs actual, and residuals.
  - **Compute feature importance** shows how much the primary metric drops when each original column is shuffled.

## 6. Predict
Choose a model (the best is preselected).
- **Single record:** a form filled with typical values (median or most common). Change any field and press **Predict**. Classification shows class probabilities. Unknown categories and blank fields are handled.
- **Batch file:** upload a CSV or Excel file with the listed columns. Extra columns are ignored, and missing required columns are reported. Download the predictions.
- **Download model:** a `.joblib` bundle containing the full pipeline and metadata. Only load bundles you created yourself.
