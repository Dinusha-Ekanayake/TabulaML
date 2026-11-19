# TabulaML user guide

Start the app with `streamlit run app.py`. The sidebar shows the six steps and a **Past runs** list. You can always go back, but changing an earlier choice clears the later steps (the app warns you first), so results never mix settings. **Start over** resets everything. The footer shows the app version, the licence and source of the loaded dataset, and a privacy note.

## 1. Problem type
Choose **Classification** (predict a category, for example a tea lot's market catalogue or churn Yes/No) or **Regression** (predict a number, for example a tea auction price or weekly dengue cases).

## 2. Data
- **Upload my dataset:** CSV, TXT, XLSX or XLS (up to 200 MB). The delimiter and text encoding are detected automatically. Rows with more values than headers are rejected with a message instead of being silently misaligned.
- **Use a Sri Lankan sample dataset:** every sample is a genuine, publicly available dataset. Read its card (source, licence, features, limitations), then **Load this sample**.
- **Target column:** checked against the problem type.
- **How should the model be tested?**
  - *Random*: rows are independent.
  - *Grouped*: several rows belong to the same thing, for example one weekly auction sale. A whole group goes to training **or** testing, never both. Choose the group column.
  - *Time-ordered*: rows happen over time. The model trains on the past and is tested on later rows. Choose the date/order column.
  - Samples prefill this from their cards: tea is grouped by `sale_id`, dengue is time-ordered by `week_start`. For uploads, the app suggests grouping when an ID-like column repeats.
- **Number format:** `1,234.5` (decimal point) or `1.234,5` (decimal comma).
- **Positive class** (binary targets only): the outcome you care about. Precision, recall, F1 and the curves are about this class.

## 3. Data quality and preprocessing
First, the app **holds out the test rows**. Every target-based recommendation is made from the training rows only, so the test set stays a fair, unseen check. The blue box states the validation design and the number of rows in each part.

- **Data quality report:** rows, duplicates, a per-column table (type, missing share, unique values, skew, outliers, examples, warnings), missing values, and the target distribution. Problems with the split (for example too few rows of a class for the chosen number of folds) or with number formats appear here.
- **Leakage review:** columns that are almost perfectly related to the target. For each one, answer *"Will this value be known at the moment you make a prediction?"* They stay excluded until you answer, and step 4 is blocked until all are answered. (Example: in the tea data each `category` belongs to exactly one catalogue, so it would give the answer away.)
- **Preprocessing:**
  - *Automatic*: a decision and a reason for every column.
  - *Custom*: edit include/exclude, type, imputation, encoding, scaling, transform and outliers per column. The type you choose is respected by the cleaner too (for example, codes like `00100` stay codes).
  - Global options: missing-value indicators, spelling-variant merging, rare-category grouping, high-cardinality threshold.
  - Class imbalance options: none, class weights, random oversampling, SMOTE (keeps category columns valid), undersampling.
  - Regression target transform: none, `log1p` (offered only when every target value is above -1) or Yeo-Johnson.
- **Pipeline preview:** the exact steps each model family applies to each column.
- **Evaluation settings:** test share, folds and seed. Changing them recreates the test set and re-runs the recommendations.

## 4. Models
Models are shown as **cards** grouped by family. Each card shows the pipeline type, speed and what the model is best for. Toggle cards on, or use **Recommended set** or **All / None** per group. The bar at the top lists your selection.
- **Voting / Stacking:** choose at least two base models.
- **Model settings (hyperparameters):** for each selected model:
  - *Default*: library defaults.
  - *Manual*: set each value yourself, with an explanation under every setting.
  - *Auto-tune*: a random search tries N combinations with the same cross-validation, on the training rows only.
  - Presets: *Fast* (fewer trees), *Balanced* (defaults), *Thorough* (auto-tune).
- Choose the **primary metric**, read the summary, then press **Train & Predict**. A model whose cross-validation fails in any fold is marked *incomplete* and is never chosen as the best.

## 5. Results
The test set is **locked** at first, so you pick a model without peeking at it.
- **Leaderboard (CV):** cross-validated mean and spread for every metric, fit times, the parameters used and the imbalance handling actually applied. Download it as CSV, or download the JSON run report.
- **Comparison:** each dot is one cross-validation fold. Overlapping boxes mean the models are about as good as each other.
- **Model details (CV):** diagnostics from out-of-fold predictions (each row predicted by a model that did not see it).
  - Classification: confusion matrix, ROC and precision-recall curves for the positive class.
  - Binary tasks also get a **decision threshold** slider (with "Best F1") and a calibration plot.
  - Regression: predicted vs actual, and residuals.
  - The settings expander shows the parameters, the auto-tuning history and the imbalance handling.
- **Final test (held out):** press **Reveal final test results** once you have chosen. It shows the test scores, the Random Forest vs gradient boosting sentence, and permutation feature importance on the test set.

## 6. Predict
Choose a model (only fully validated ones are listed). A threshold chosen in step 5 is applied.
- **Single record:** a form filled with typical values. Classification shows class probabilities.
- **Batch file:** upload a CSV or Excel file. Before scoring, a quality check lists values that could not be parsed, values outside the training range and categories never seen in training. Download the predictions.
- **Download model:** a `.joblib` bundle with the pipeline, parameters and threshold. Only load bundles you created yourself.
