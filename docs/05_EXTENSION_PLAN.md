# TabulaML extension plan (E1–E4): known limits → features

This plan answers the "known limits" review. Each phase is implemented test-first and gets its own commit. The workflow stays the same six steps, and the new options appear as cards and short explanations in the existing design. Everything stays within the course syllabus: plain supervised learning, honest validation, no deep-learning or advanced statistics.

| Limit | Mitigation | Phase |
|---|---|---|
| Training can be slow, there is no warning and no cancel button | Time estimate with warnings before training; optional row cap; background training with a progress bar and a **Cancel** button that keeps finished models | E1 |
| One target only | **Several targets** of the same problem type: multi-label / multi-output classification and multi-output regression | E2 |
| Time data gets no lag features, so it is not real forecasting | **Forecasting** problem type: lag, rolling-mean and calendar features, a naive baseline, time-ordered validation, and "forecast the next N periods" | E3 |
| Only CSV/Excel; few number formats; small data gives unstable scores | JSON and Parquet upload; `1 234,5` and `1'234.5` number formats; small-dataset warnings | E4 |

---

## E1 · Training time, row cap, background training, cancel

**Core (`tabulaml/training/`)**
- `estimate.py`: `estimate_training(n_rows, n_features, model_keys, cv_folds, tuning, ensemble_bases)` returns a `TrainingEstimate(seconds_low, seconds_high, level, per_model, warnings)`. It uses relative cost factors per model (per 1k rows × fits), plus warnings for:
  - stacking (it refits its base models inside every fold);
  - auto-tuning (N extra CV rounds);
  - KNN imputation on more than 20k rows;
  - SVM near its row limit;
  - cloud mode.

  The level is quick (< 30 s), moderate (< 3 min) or long.
- `PreprocessingConfig.max_rows: int | None` (contract change). `prepare` keeps at most `max_rows` rows *before* the split:
  - random: a seeded random sample;
  - grouped: whole groups are sampled;
  - time-ordered: the **latest** rows are kept.

  A note records the sampling.
- `train_models(..., should_stop=None)` checks `should_stop()` between models and between CV folds. When it stops, finished models are kept and a note says "Cancelled by user after k of n models". If nothing finished, it raises `TrainingCancelled`.
- Progress is reported in fold-level units: `progress(label, done, total)`, where total = models × (folds + 1).
- `jobs.py`: `TrainingJob` runs `train_models` in a daemon thread with a stop `Event` and records the progress label, fraction, elapsed time, result and error. It never touches Streamlit.

**UI (step 4)**
- The summary card shows the time estimate (range), the level chip and the warnings.
- A "Speed options" expander offers "Train on at most N rows" with a sensible suggestion when the data is large.
- **Train & Predict** starts the job. A progress card (`st.fragment(run_every=1s)`) shows the current model and fold, the elapsed and remaining time, and a **Cancel** button.
  - When the job finishes, the app moves to step 5.
  - When cancelled, finished models are shown with a notice.
  - If the configuration changed while training, the result is discarded with a message.
- Navigation and the Train button are disabled while a job is running.

**Tests**
- The estimate is monotonic in rows and models and gives the right warnings.
- `max_rows` works for all three validation schemes and does not mutate the input.
- `should_stop` keeps finished models, and stopping before any model finishes raises.
- `TrainingJob` covers finish, cancel and error.
- UI flow: training runs in the background and results appear.

## E2 · Several targets (multi-output)

- `PreprocessingConfig.extra_targets: tuple[str, ...] = ()` and a `targets` property (contract change).
  - Classification with several targets is multi-label when every target is yes/no, otherwise multi-output.
  - Regression with several targets is multi-output.
- `prepare` returns a 2-D `y`. Each classification target is label-encoded separately (`Prepared.target_labels`). Rows missing **any** target are dropped, and the other targets are never features.
- Pipeline: `cleaner → features → MultiOutputClassifier/Regressor(model)`. For voting and stacking, the whole ensemble is wrapped. Limits that apply with several targets:
  - Class-imbalance samplers and class weights are off and the reason is shown.
  - Target encoding falls back to ordinal or one-hot.
  - Auto-tuning uses the averaged metric.
- Splits: random K-fold (stratification is not possible for 2-D labels). Grouped and time-ordered splits work unchanged.
- Metrics (`metrics.multi_output_metrics`):
  - Classification: subset accuracy (all targets right), Hamming loss, and mean accuracy, F1 (macro) and balanced accuracy across targets.
  - Regression: mean R², RMSE and MAE across targets.
  - Per-target test metrics are stored in `ModelResult.per_target`.
- Results:
  - The leaderboard uses the averaged metrics.
  - A "Per target" table appears in the final test tab.
  - The CV diagnostics show per-target confusion matrices or predicted-vs-actual plots, with a target selector.
  - The ROC and threshold tools are hidden, with an explanation.
- Predict: one `prediction_<target>` column per target. The single-record view shows one card per target. Bundles carry the targets.
- UI step 2: a "Predict several columns at once" toggle with a multiselect for the extra targets. The targets are validated against the problem type, and different types are not allowed.

## E3 · Forecasting (time series)

- Step 1 gets a third card, **Forecasting**. Underneath it is regression (`AppState.forecasting`), with `PreprocessingConfig.forecast: ForecastConfig | None` (contract change). `ForecastConfig` has these fields:
  - `horizon` (h ≥ 1);
  - `n_lags`, giving lags h … h+n_lags−1;
  - `seasonal_lag` (optional, ≥ h);
  - `windows` (rolling means);
  - `series_column` (optional, for several series);
  - `exog` = `"none"` | `"lagged"`.
- `tabulaml/forecasting/features.py`: `add_forecast_features(df, config)`. Steps:
  1. Sort by series and time.
  2. Lags of the target, shifted by at least h. Rolling means over the target shifted by h. Calendar parts of the date (month, week of year, day of week, quarter, year). Optionally, other columns lagged by h.
  3. Drop the first rows of each series whose lag h is missing.

  Only past values are used, and nothing is fitted (no learned parameters), so this is safe before the split. Validation is forced to time-ordered.
- Checks: the order column is required and must parse. Duplicate timestamps in one series are an error, with advice to choose a series column. The series must be long enough (≥ max lag + h + folds × 2). Irregular gaps give a warning. A horizon larger than 25 % of the series gives a warning.
- Models: all regression models, plus **Naive (last known value)** and **Seasonal naive** baselines (`forecasting/baselines.py`). The results show "beats the naive baseline by X % RMSE".
- Future forecasts: `forecast_future(pipeline, history, config)`.
  - Future timestamps come from the inferred frequency (`pd.infer_freq`, falling back to the median step).
  - Future rows are appended with an unknown target and the features are rebuilt. Because every lag is ≥ h, all values the model needs are known.
  - Predictions for steps 1…h are returned for every series.
- The bundle stores the tail of the history needed for the lags. Step 6 in forecasting mode shows **Forecast next h periods**, using the training history or an uploaded newer history, with a chart of history + forecast and a CSV download.

## E4 · Data input add-ons and small-data guidance

- Loader: `.json` (records, columns or JSON-lines) and `.parquet`, with the same limits and errors.
- Number formats: add `1 234,5` (space thousands, decimal comma) and `1'234.5` (apostrophe thousands).
- Warnings:
  - fewer than 100 usable rows: "scores will vary a lot; prefer more folds or more data";
  - fewer than 30 rows: error;
  - free-text columns: a note that they are reduced to simple text statistics.
- Docs: `USER_GUIDE.md` (new options), `README.md` "Capabilities and limits", `02_ARCHITECTURE.md §4` (contract changes), `PROGRESS.md`.

## E5 · RF dashboard, card design, responsive layout, end-to-end checks
- The lecture asks for an **RF dashboard**, an **RF vs XGBoost comparison** on local data, and **hybridization** (RF + ARIMA for time series, RF + ANN for tabular regression):
  - Step 5 gets a **Random Forest dashboard** tab: KPI cards (CV score, spread, trees, average depth, fit time), a fold-by-fold chart against the boosters, a head-to-head table (RF wins, losses and ties on identical folds), and the forest's built-in importance.
  - RF + ANN already exists. RF + ARIMA is added in E3.
- Consistent dashboard cards: the problem types, model cards, estimate card, training card and KPI cards share one style (radius, border, hover, selected state) in light and dark mode.
- Responsive: columns stack on narrow screens, cards drop their fixed minimum heights, the footer wraps, and nothing scrolls sideways at 390 px.
- Playwright walk-through:
  - every problem type end to end (classification, several targets, regression, forecasting);
  - desktop and phone widths, light and dark;
  - no exceptions, no error boxes and no horizontal overflow.

## Definition of done
- Every phase has unit and UI tests. Coverage ≥ 80 % (currently about 95 %). `ruff` is clean. Files stay under 400 lines and functions under 50.
- No dataset-specific code in `tabulaml/`. No synthetic sample datasets.
- One conventional commit per phase.
