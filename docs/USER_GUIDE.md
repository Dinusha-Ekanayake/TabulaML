# TabulaML user guide

Start the app with `streamlit run app.py` and open http://localhost:8501. It works on desktops, tablets and phones.

**Around the app:**
- **Sidebar:** the six steps, **Past runs** and **Start over**. You can always go back. Changing an earlier choice clears the later steps, and the app warns you first.
- **Theme:** open the **⋮** menu at the top right and choose **System**, **Light** or **Dark**.
- **Footer:** the version, the licence and source of the loaded dataset, and a privacy note.

---

## 1. What do you want to predict?
A strip at the top shows the standard workflow the app follows. Pick one of three cards:

| Card | Use it when | Examples |
|---|---|---|
| **Classification** | The target is a category | Tea lot catalogue; churn Yes/No. Several yes/no targets at once = multi-label |
| **Regression** | The target is a number explained by the other columns of the same row | Tea auction price. Several numbers at once = multi-output |
| **Forecasting (time series)** | You want future values of a series from its own past | Next week's dengue cases |

## 2. Data
- **Upload:**
  - Formats: CSV, TXT, Excel (choose the sheet), JSON or Parquet, up to 200 MB.
  - The delimiter and encoding are detected automatically.
  - JSON can be a list of records, JSON lines, `{"column": [values]}` or `{"data": [records]}`.
- **Sample:** a public Sri Lankan dataset with its card (source, licence, features, limitations). Forecasting mode lists only time-series samples.
- **Target column:** checked against the problem type.
- **Predict several columns at once** (classification and regression only): turn it on and add the extra targets. They must be of the same kind. One model copy is trained per target, and the other targets are never used as inputs.
- **How should the model be tested?** (classification and regression)
  - *Random*: rows are independent.
  - *Grouped*: several rows belong to the same thing (for example one auction sale). A whole group goes to training or to testing, never both.
  - *Time-ordered*: train on the past, test on later rows.
- **Number format:** `1,234.5` or `1.234,5`. Spaces and apostrophes as thousands marks (`1 234,5`, `1'234.5`) are understood automatically.
- **Positive class** (single binary target): the outcome you care about. Precision, recall and F1 are about this class.

### Forecast settings (forecasting only)
| Setting | Meaning |
|---|---|
| Date / time column | Puts the rows in time order |
| Series column | Only if the file holds several series (for example one per district) |
| Horizon | How many periods ahead to forecast. Every input is at least this old |
| Lags | How many past values to use (lags *horizon* … *horizon + lags − 1*) |
| Seasonal lag | One season back: 52 for weekly, 12 for monthly, 7 for daily (suggested automatically) |
| Rolling means | Averages of the last N known values |
| Other columns | *Use their past values* (lagged by the horizon, so they are known in time) or *Do not use* |

The app then checks the series:
- **Errors** (Next is blocked): duplicate time points in a series, or a series too short for the settings.
- **Warnings:** irregular gaps, a long horizon, a seasonal lag longer than the first fold.

## 3. Data quality and preprocessing
The app first **holds out the test rows**. Every target-based recommendation uses the training rows only.

- **Evaluation settings** (expander):
  - test share, folds and seed;
  - **Row cap**: train on at most N rows to save time. Time-ordered data keeps the latest rows, grouped data keeps whole groups, otherwise a random sample is used.
  - Changing any of these re-runs the analysis.
- **Data quality report:**
  - rows and duplicates;
  - a per-column table: type, missing values, unique values, skew, outliers and warnings;
  - a missing-values chart and the target distribution;
  - warnings about the split, number formats and small data.
- **Explore data** (before any preprocessing):
  - *One column*: pick any column (the target is listed first). You get stat cards (rows, missing, distinct values, mean, median, spread, skew, outliers or the most common value) and a chart of its distribution: a histogram with a box plot for numbers, the top categories for text, counts over time for dates.
  - A second chart shows the column against the target: boxes per class, class shares per category, a scatter plot, or boxes of the target per category. This uses the **training rows only**, so the test set stays unseen.
  - *Correlations*: a heatmap for the number columns, plus the most strongly related pairs explained in plain words.
- **Leakage review:** columns almost perfectly related to the target. Answer *"Will this value be known when you make a prediction?"* for each one. Step 4 is blocked until every column is answered. In forecasting mode there is nothing to review, because every feature comes from the past.
- **Preprocessing:**
  - *Automatic*: a decision and a reason for every column.
  - *Custom*:
    - **Rows and missing values** (card):
      - *Drop exact duplicate rows* (on by default; the number found is shown).
      - *Rows with missing values*: keep them and fill the gaps (recommended), or drop every row with a missing value. The card shows how many rows that is and warns when more than half would go.
      - *Exclude columns with more missing values than X %*, then **Apply**. The default limit is 60 %, and excluded columns can be re-included in the table.
      - Row options change which rows exist, so they re-create the held-out test set. New data at prediction time is always filled, never dropped.
    - **Missing values (imputation):** for number columns, median, mean, most frequent, constant 0 or KNN; for category columns, an own "missing" category or the most frequent category. Apply a method to all columns of a type, or set each column on its own.
    - **Global options:** missing-value indicators, merging spelling variants, grouping rare categories, the high-cardinality threshold.
    - **Class imbalance:** none, class weights, random oversampling, SMOTE or undersampling. These are off when you predict several targets.
    - **Target transform** (regression): none, `log1p` or Yeo-Johnson.
    - **Per-column table:** include, type, encoding, scaling, transform and outliers.
- **Pipeline preview:** the exact steps each model family applies to each column.

## 4. Models
- **Model cards:**
  - Cards are grouped by family. Each shows its pipeline type, speed and what it is best for.
  - Use **Recommended set**, **All / None** per group, or toggle single cards.
  - Forecasting adds **Naive**, **Seasonal naive** and **Hybrid RF + ARIMA**.
  - Regression includes **Hybrid RF + ANN**.
- **Voting / Stacking:** choose at least two base models.
- **Model settings** (one card per model):
  - *Default*: the model's standard settings.
  - *Manual*: the form starts from the real defaults and explains every setting. Regularisation options include L1 / L2 / elastic net for logistic regression, `reg_alpha`, `reg_lambda` and `gamma` for XGBoost and LightGBM, `l2_leaf_reg` for CatBoost, and cost-complexity pruning for the decision tree.
  - *Random search*: N random combinations.
  - *Grid search*: every combination of a few values per chosen setting, centred on the defaults. The card shows how many combinations and model fits that means (limit 200 combinations).
  - Both searches use the same cross-validation on the training rows only.
  - Presets: Fast, Balanced, Thorough.
- **Estimated training time:**
  - The estimate is a range with a speed level (quick, moderate or long).
  - It warns about slow choices: stacking, hyperparameter search, SVM on large data, KNN imputation, cloud mode.
  - It names the slowest model and suggests ways to speed up.
- **Train & Predict:**
  - Training runs in the background. A progress card shows the model and fold, the elapsed time and the approximate time remaining.
  - **Cancel** stops after the current fold and keeps every model that already finished.
  - If you change settings during training, those results are discarded with a message.
  - A model whose CV fails in a fold is marked *incomplete* and is never chosen as best.

## 5. Results
- **Leaderboard (CV):**
  - mean and spread for every metric, fit times, parameters and the imbalance handling used;
  - CSV and JSON downloads.
- **Comparison:** each dot is one CV fold. Overlapping boxes mean similar models.
- **Random Forest dashboard** (when RF was trained):
  - KPI cards: CV score, spread, trees, average depth, fit time.
  - A fold-by-fold chart against the boosters.
  - A **head-to-head** table: RF wins, losses and ties on the same folds.
  - The forest's built-in feature importance.
- **Model details (CV):** out-of-fold diagnostics, where each row is predicted by a model that never saw it.
  - Classification: confusion matrix, ROC and PR curves, a decision threshold slider (with Best F1), and calibration.
  - Regression: predicted vs actual, and residuals.
  - With several targets, choose the target to inspect.
- **Final test (held out):** press **Reveal final test results** once you have chosen. You then see:
  - the test scores and the RF vs boosting sentence;
  - a per-target table (several targets);
  - permutation importance;
  - in forecasting mode, the test period chart and whether the model beats the naive baseline.

## 6. Predict
Only fully validated models are listed. A threshold chosen in step 5 is applied.
- **Single record:** a form filled with typical values. With several targets, you get one card per target.
- **Batch file:** the file is quality-checked first (unparseable values, values outside the training range, new categories). Then download the predictions, with one `prediction_<target>` column per target.
- **Forecast next periods** (forecasting):
  - The chosen model is refitted on all history, then forecasts the next *horizon* periods for every series.
  - You can use the training data or upload a newer history file.
  - You get a chart and a CSV download.
- **Download model:** **Prepare model files**, then **.joblib** (compressed) or **.pkl** (standard pickle). Load either with `load_bundle(path)`. Only load files you created yourself.

## Tips
- Start with the **Recommended set**. Add stacking or a hyperparameter search only when you need the last bit of accuracy.
- On a slow machine or on Streamlit Cloud, use the row cap and fewer folds.
- In forecasting, always compare with the **naive** baseline. A model that cannot beat "same as last week" has learned nothing useful.
