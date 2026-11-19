# TabulaML: Review Assessment and Improvement Plan (v2)

Status at commit `e141791`: 6-step app, 18 models plus voting/stacking/hybrid, public Sri Lankan data only, 201 passing tests. This plan covers:
- an assessment of the external review;
- extra defects found in a full self-review;
- a phased plan written for Sonnet, one phase per session, in the same style as `03_IMPLEMENTATION_PLAN.md`.

---

## A. Assessment of the external review

The review was of commit `ac983df`, which still had the synthetic samples. Every technical claim marked "verified" below was reproduced against the current code.

| # | Review claim | Verified? | Verdict | Severity (ours) |
|---|---|---|---|---|
| 1 | Automatic config is computed on the full dataset before the split | Yes (`s3_preprocess._analyse`) | **Partly right.** Unsupervised profiling (types, missingness, cardinality) on all rows does not leak target information. The *target-dependent* choices do use test labels: leakage exclusions, imbalance strategy and `log1p`. The effect is small but real, and it is easy to attack in a viva. | **High** (cheap to fix, important to defend) |
| 1b | "Correlation alone is not leakage; warn, don't auto-exclude" | Yes | **Agree in part.** The thresholds are strict (AUC/\|r\| ≥ 0.98/0.95, plus missingness), so false positives are rare. Still, only the user knows whether a column exists at prediction time. Keep *exclude by default* (the safe default), but require an explicit decision per flagged column. | Medium |
| 2 | The dengue sample needs chronological validation; define the prediction moment | Yes (random split + shuffled K-fold) | **Agree.** Random CV on lagged weekly series is optimistic, and same-week weather is not available for a one-week-ahead forecast. The reviewer missed that **tea auction rows are grouped by weekly sale (`sale_id`)**, so random CV there also leaks across groups. | **High** |
| 3 | Type overrides never reach the cleaner (`"00100"` → 100, `"1,25"` → 125) | Yes, reproduced | **Agree.** The cleaner destroys information before a later "categorical" override can help. Unparseable values silently become NaN. | **High** |
| 4 | Plain SMOTE after ordinal encoding creates invalid category codes; ensembles silently switch strategy | Yes (11 fractional codes in a 300-row test) | **Agree.** Use SMOTENC with the categorical output indices, or RandomOverSampler, and show the *effective* strategy per model. | **High** |
| 5 | The positive class is inconsistent (metrics use the minority class, curves use class 1) | Yes, reproduced | **Agree.** The user must choose the positive class, and it must be used everywhere. | Medium |
| 6 | Failed CV folds (`error_score=nan`) still give status "ok"; class counts are not revalidated after fold/test-share changes | Yes, by code reading; step 2 hard-codes `cv_folds=5` | **Agree.** | Medium |
| 7 | Local data is only partly addressed; keep the synthetic samples as stress tests | Outdated | **Disagree in part.** Since `e141791` the app ships only public data: Colombo Tea Auction (CC BY 4.0) and Colombo dengue (CC0), plus the Kaggle telco data on demand. Re-adding synthetic samples conflicts with the brief and with the user's decision. Its practical advice stands: **prepare the real telco data before the demo.** Generated frames stay inside the tests only. | Medium (telco demo) |
| 8 | Wilcoxon on repeated-CV folds is exploratory; the "best booster" is chosen from the same results | Yes | **Agree.** Folds overlap, so plain tests are anti-conservative, and choosing the best booster post hoc inflates significance. Use the Nadeau-Bengio corrected resampled t-test, compare RF against **each** booster, and apply a Holm correction. | Medium (report) |
| 9 | Hyperparameter controls, prediction validation, resource limits, background training, experiment history, test-set over-use | Yes | **Agree.** Hyperparameters are also explicitly requested by the user. Locking the test set (a holdout lock) is the most valuable item here. | Medium-Low |
| - | Streamlit is a reasonable choice; the architecture and model set are sufficient | - | **Agree.** Don't rewrite the stack; don't add more models. | - |

**Overall:** the review is accurate and constructive. Its main weakness is that its dataset section is out of date, and it argues for synthetic data the brief does not want. It also missed three issues listed below: the `log1p` crash, grouping in the tea data, and test-set reuse in feature importance.

## B. Additional findings from our own full review

| ID | Area | Issue | Severity |
|---|---|---|---|
| F1 | Regression | `log1p` with any target ≤ -1 makes every fold NaN, so the model fails with a confusing message (reproduced). It must be blocked in the UI and recommender, or the target shifted. | High |
| F2 | Validation | Tea rows are grouped by `sale_id` (about 116 rows per weekly sale), so random CV mixes the same week into train and validation. Group CV is needed. | High |
| F3 | Evaluation | Permutation importance and every diagnostic plot use the **test** set. Test data should be used once for final reporting; diagnostics should come from out-of-fold predictions on the dev set. | Medium |
| F4 | Benchmark | `scripts/benchmark.py` also builds the Automatic config on full data (same issue as review #1). | Medium |
| F5 | Cleaner | No locale settings (decimal comma or space thousands separators). Codes with leading zeros, phone numbers and IDs are parsed as numbers. No report of how many values failed to parse in each column. | High (with #3) |
| F6 | Prediction | Batch prediction only checks that columns are present. It does not report parse failures, values outside the training range, or unseen categories per column. | Medium |
| F7 | Binary classification | Fixed 0.5 threshold; no threshold tuning (on out-of-fold data) and no calibration view. | Medium |
| F8 | UI | Training blocks the page with no cancel. A wide dataset (100+ columns) makes the column editor and the prediction form unwieldy. There is no run history. Changing a setting silently clears results with no confirmation. | Medium |
| F9 | Reproducibility | Hyperparameters are not shown in the run report or bundle (only defaults are used today). | Medium (fixed by P15) |
| F10 | Docs | `docs/REPORT.md` and `reports/benchmark_summary.md` still describe the old synthetic results until the public-data benchmark finishes. | High (before submission) |
| F11 | Dengue | The target-defining weather (same-week) is mixed with forecasting features. Two task variants are needed: *nowcast* (same-week weather allowed) and *1-week-ahead forecast* (lagged features only). | Medium |

---

## C. Improvement plan

Order is by value for the viva and correctness. Every phase is self-contained, tests come first, and the gate after each phase is `ruff check`, `pytest -q --cov=tabulaml` (≥ 80 %), then commit. **⚑** marks phases that get an Opus review afterwards.

### Must be kept in every phase (explainability, as asked by the user)
No phase may remove or weaken the existing explanations. New features must explain themselves the same way. The features to keep:
- The `reason` behind every per-column decision, and the dataset-level `explanations` (Automatic mode).
- Leakage warnings with their evidence; the data-quality report.
- The **pipeline preview** for each model family.
- Dataset cards (provenance, licence, limitations).
- Model descriptions and profile badges.
- The "Random Forest vs gradient boosting" sentence; permutation importance; per-model diagnostics.
- The JSON run report.

New features follow the same pattern:
- validation scheme, imbalance strategy, tuned parameters and threshold each get a one-line reason and appear in the run report;
- the tuner shows its search history;
- leakage decisions record the user's answer.

A test (`test_explanations_present`) asserts that every `ColumnConfig` and each new decision carries a non-empty reason.

### Q1 - Footer (quick win; can be done first, independent of everything else)
A persistent footer on every step, rendered by `components.footer(state)` and called once from `app.py` after the page body. It shows:
- **Left:** "TabulaML v{version}" (from package metadata), a one-line purpose, and a link to the GitHub repository.
- **Centre:** the **attribution for the loaded dataset**: title, source link, licence (e.g. "Colombo Tea Auction data, CC BY 4.0, colombo-tea-auction-prices/cta-price-prediction"). CC BY 4.0 *requires* attribution, so this is a licence obligation, not decoration. With no dataset loaded it shows "No dataset loaded".
- **Right:** "Built with Streamlit · scikit-learn · XGBoost · LightGBM · CatBoost"; a privacy note ("Data stays in this session; nothing is uploaded to external services"); links to the User guide and Report on GitHub.

Styling:
- small muted text, a thin top border, three columns that wrap to one column on narrow screens;
- theme tokens, readable in light and dark mode;
- no AI or tool attribution in the footer.

An AppTest asserts that the footer renders on every step, and that the dataset attribution appears after a sample loads. Effort: S.

### Q2 - Ready for Streamlit Community Cloud deployment (quick win)
Target: deploy from GitHub at share.streamlit.io with `app.py` as the entry point and Python 3.12. Files to add or change:
- **`requirements.txt`:** runtime packages only, with **exact pins** copied from `requirements.lock.txt` (keep `scikit-learn==1.5.2`). No `kagglehub`, `matplotlib`, `pytest` or `ruff`; they stay in `requirements-dev.txt`. Check that every pinned version has a Linux wheel for Python 3.12.
- **`packages.txt`:** `libgomp1`, the OpenMP runtime LightGBM needs on Debian.
- **`.streamlit/config.toml`:**
  - `[server] maxUploadSize = 200`, `headless = true`;
  - `[browser] gatherUsageStats = false`;
  - `[theme]` tokens, shared with P18.
- **Imports:** Cloud does not run `pip install -e .`. `app.py` already sits next to `tabulaml/`, so the package imports from the repo root. Add a test that runs `python -c "import app"` from a clean checkout without an editable install.
- **Memory (about 1 GB on Community Cloud):**
  - an env/config switch `TABULAML_CLOUD=1` (or `st.secrets["cloud"]`) that lowers `MAX_CELLS` to 300k;
  - caps `n_estimators`/`iterations` through the "Fast" preset (P15);
  - sets `n_jobs=2`;
  - disables SVM above 5k rows;
  - shows a banner "Cloud mode: limits applied".
- **Caching:**
  - `@st.cache_data` for `load_sample` and profiling;
  - `@st.cache_resource` is not needed;
  - never write to the repo. `artifacts/` writes use a temp dir; downloads are served from memory (already the case for bundles).
- **Secrets:** optional `KAGGLE_USERNAME`/`KAGGLE_KEY` via `st.secrets`, used only by an admin "fetch telco" path. Document a `secrets.toml.example`; never commit real secrets (`.streamlit/secrets.toml` goes in `.gitignore`).
- **Startup check:** `app.py` shows a friendly error if a heavy dependency is missing, instead of a traceback.
- **`docs/DEPLOYMENT.md`:**
  - step by step: fork or push, sign in at share.streamlit.io, "New app", choose the repo, branch `main` and file `app.py`, set Python 3.12 under Advanced settings, deploy;
  - the expected first build time;
  - how to update (push to `main`);
  - limits and troubleshooting (LightGBM `libgomp`, memory, the 200 MB upload size).
- **README:** a "Live demo" badge or link once deployed.
- **Acceptance:**
  - `pip install -r requirements.txt` in a **fresh** venv followed by `streamlit run app.py` works with no dev tools installed.
  - A smoke script `scripts/smoke_deploy.py` boots the server headless, checks `/_stcore/health`, and runs the AppTest flow with the Fast preset in under 2 minutes.

Effort: S-M.

### P11 - Validation strategy and holdout discipline ⚑
**Fixes:** review #1, #2, #6; F2, F3, F4.
- **Types:**
  - `ValidationScheme` enum: `random | grouped | time_ordered`.
  - `PreprocessingConfig` gains `validation: ValidationScheme`, `group_column: str | None` and `order_column: str | None`.
  - Group and order columns are always excluded as features.
- **Splitter module** (`training/splits.py`): `make_holdout(df, config)` returns `(dev_idx, test_idx)`:
  - stratified random split;
  - `GroupShuffleSplit` for grouped data;
  - the last `test_size` share in time order for time-ordered data.
  - `make_cv(config)` returns `StratifiedKFold`, `StratifiedGroupKFold`/`GroupKFold`, or `TimeSeriesSplit`.
  - Write unit tests proving no group overlap and no future rows in training.
- **Holdout first:** step 3 creates the dev/test split *before* any target-dependent decision.
  - `detect_leakage`, the imbalance recommendation and the `log1p` decision run on **dev rows only**.
  - Profiling for display may use all rows, labelled "informational".
  - Store the split indices in the state and the run.
- **Recommendations:**
  - Recommend `time_ordered` when a datetime or year/week-like column exists and rows are sorted by it.
  - Recommend `grouped` when a column repeats with mostly constant other values (for example `sale_id`), and explain why.
- **CV integrity:** if any fold's primary score is NaN, the status becomes `incomplete` (new value) with the failed-fold count. Incomplete results are excluded from `best()`.
- **Class-size check:** revalidate class counts against the *actual* `cv_folds`, `test_size` and scheme whenever those settings change, in step 3 (not only step 2).
- **Diagnostics on dev data:** confusion matrix, ROC/PR, residuals and permutation importance use **out-of-fold dev predictions** (`cross_val_predict` with the same CV). The test set is used once, in a "Final holdout evaluation" panel.
- **Datasets:**
  - Dengue: add a `week_start` date and default the card to `time_ordered`.
  - Tea: default to `grouped` by `sale_id`.
  - Benchmark: use the dev-only config. Run time-ordered CV for dengue and grouped CV for tea.
- **Acceptance:**
  - A spy test proves target-dependent recommendations never see test rows.
  - Grouped and time-ordered split tests pass.
  - An incomplete-CV model is never chosen as best.

> Prompt: "Implement P11 of docs/04_IMPROVEMENT_PLAN.md. Read CLAUDE.md, docs/02_ARCHITECTURE.md §4 and P11 only. Tests first. Update the contract changes log in docs/PROGRESS.md."

### P12 - Cleaner honours user types and locale; parse report ⚑
**Fixes:** review #3; F1, F5.
- `DataCleaner(column_types: dict[str, FeatureType] | None, decimal: str = ".", thousands: str | None = None)`. Explicit types override inference:
  - CATEGORICAL and IDENTIFIER keep the raw string;
  - NUMERIC parses with the locale;
  - BOOLEAN and DATETIME parse as now.
- **Inference safeguards:**
  - Strings with leading zeros (`^0\d+`) or a fixed width of all digits are inferred as **categorical codes**, not numbers.
  - Ambiguous separators (both `,` and `.`, or a comma followed by exactly 2 digits) produce an `ambiguous_number_format` warning that asks for the locale.
- **`parse_report_`:** per column, the count of values that became NaN because they failed to parse (separate from values that were already missing). Shown in step 3 and in batch prediction.
- `build_model_pipeline` passes `config` types into the cleaner. Profiling is re-run after a type change, so the preview matches.
- **Target transform guard:**
  - The recommender only proposes `log1p` when `min(y) ≥ 0`.
  - The UI disables `log1p` when `min(y_dev) ≤ -1` and offers "Yeo-Johnson target" instead (`TransformedTargetRegressor` with `PowerTransformer`).
  - `pipeline_factory` raises a clear `ValueError` otherwise.
- **Acceptance:**
  - `"00100"` stays `"00100"`.
  - `"1,25"` with `decimal=","` becomes `1.25`; without it, a warning is raised.
  - A categorical override changes the cleaner output.
  - A negative target with `log1p` is blocked with a message.

### P13 - Correct class-imbalance handling ⚑
**Fixes:** review #4.
- **Strategies:** `none | class_weight | random_oversample | smote`.
  - `smote` becomes **SMOTENC** when categorical outputs exist. `pipeline_factory` passes `categorical_features` as the ColumnTransformer output indices of ordinal/target/one-hot columns, computed after fitting with a small `SamplerAdapter` that resolves the indices in `fit_resample`. Plain SMOTE is used only when every feature is numeric.
  - In the scale-sensitive profile with one-hot columns, use `random_oversample` unless SMOTENC is explicitly chosen, and explain why.
- **Effective strategy** per model (`ModelResult.effective_imbalance`), shown in the leaderboard and model cards. Ensembles show "class weights (sampler not supported inside ensembles)".
- **Acceptance:**
  - After resampling, ordinal columns contain only integer codes that were seen in training.
  - The effective strategy is recorded.

### P14 - Positive class, threshold and calibration
**Fixes:** review #5; F7.
- `PreprocessingConfig.positive_class: str | None`. The UI selects it in step 2 for binary targets, defaulting to the minority class, with an explanation.
- `metrics.py`, `plots.py` (ROC/PR/probability), the importance scorer and `Predictor` all take the encoded positive index from the config. Remove `minority_label` from the metric paths.
- **Decision threshold (binary):**
  - Slider in results, with an "optimise F1 / balanced accuracy / recall ≥ X" button computed on **out-of-fold dev** probabilities.
  - Stored in the run and the bundle; `Predictor` applies it.
- A calibration (reliability) curve on dev OOF probabilities.
- **Acceptance:** with a minority class of 0, metrics and curves refer to the same class; the threshold changes predictions consistently.

### P15 - Hyperparameters: defaults, manual, auto-tune (user request)
- **`ModelSpec.params`:** a tuple of `ParamSpec(name, kind: int|float|choice|bool, default, low, high, log: bool, choices, help)`, written for each model. Examples:
  - **RF:** `n_estimators` 50-1000, `max_depth` None/2-50, `min_samples_leaf` 1-20, `max_features` sqrt/log2/0.3-1.0, `class_weight`.
  - **XGB:** `n_estimators`, `learning_rate` (log, 0.005-0.3), `max_depth` 2-12, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda` (log).
  - **LGBM:** `num_leaves` 8-256, `learning_rate`, `n_estimators`, `min_child_samples`, `feature_fraction`.
  - **CatBoost:** `iterations`, `learning_rate`, `depth` 3-10, `l2_leaf_reg`.
  - **MLP:** `hidden_layer_sizes` (choices), `alpha` (log), `learning_rate_init`.
  - **SVM:** `C`, `gamma` (log), `kernel`.
  - **KNN:** `n_neighbors`, `weights`.
  - **Linear:** `C` / `alpha` (log).
  - Ensembles expose their base models' params plus the meta-model `C`.
- **Three modes per model** in step 4, inside each model's "Settings" expander:
  1. **Default:** current behaviour.
  2. **Manual:** a widget for each `ParamSpec` (number input, slider or select), validated against its bounds.
  3. **Auto-tune:** `RandomizedSearchCV` (or `HalvingRandomSearchCV` for large data) over the `ParamSpec` ranges, with budget controls: number of candidates (10-100) and a time limit. Optional Optuna backend if installed.
- **Leakage-safe tuning:**
  - Search **only on dev data with the same CV scheme as P11** (grouped and time-ordered respected).
  - Report the tuned model's CV score *and* note that it is optimistically biased.
  - Optional **nested CV** check (outer = the P11 CV) for an unbiased estimate.
  - The test set stays locked.
- **Recorded:** chosen or tuned params go into `ModelResult.params`, the run report, the leaderboard ("params" column, expandable) and the model bundle. Tuning history appears as a table plus a parallel-coordinates chart.
- **Presets:** "Fast" (small `n_estimators`), "Balanced" (defaults), "Thorough" (auto-tune with 30 candidates).
- **Acceptance:**
  - Manual params reach the estimator (asserted via `get_params`).
  - The tuner never sees test indices (spy test).
  - Grouped CV is used inside the tuner.
  - Bundles round-trip params.

### P16 - Leakage decisions by the user
**Fixes:** review #1b.
- Flagged columns stay excluded by default. Step 3 shows a **Leakage review** card per column with the evidence (AUC or r, a missingness plot) and the question *"Will this value be known at the moment you make a prediction?"* (Yes = include / No = keep excluded).
- Step 4 is blocked until every flagged column has an answer.
- Decisions go into the run report.

### P17 - Prediction-time validation
**Fixes:** F6.
- `Predictor.validate_input` returns, per column:
  - parse failures (from P12's `parse_report_`);
  - the share of values outside the training [min, max];
  - unseen categories (count and examples);
  - missing share compared with training.
- The batch UI shows a "data drift / quality" table before scoring, with an option to continue.
- The single-record form shows a warning icon when a value is outside the training range.

### P18 - UI and UX upgrade
- **Theme and layout:**
  - `.streamlit/config.toml` with a light theme, and a dark theme selected via tokens.
  - A top progress bar with step names.
  - Each step opens with a one-line "what this step does" sub-header.
  - Dataset chip in the header (name, rows × columns, task, licence badge).
- **Step 2:** a "column roles" table: target, group column, order column, ignore. P11's suggestions are pre-filled.
- **Step 3:**
  - search and filter in the column editor (by type, warnings or excluded);
  - a per-column details drawer: histogram or top values, missing pattern, parse report;
  - "Reset to Automatic".
- **Step 4: model cards instead of plain checkboxes:**
  - a 3-column card grid per group, with a group icon and an "n of m selected" count plus "Select all / none";
  - each card: icon, name, tags (tree / scaled pipeline / ensemble, speed), a "best for" line, a description, and a select toggle;
  - selected cards get a highlighted border; a summary bar of chips for the selected models;
  - an estimated run time (from rows × columns);
  - P15 settings expanders;
  - preset buttons;
  - an explicit summary before training ("7 models × 5 folds, grouped by sale_id, ~2 min").
- **Training:**
  - run in a background thread (`st.fragment` with polling, or `concurrent.futures`) with a live per-model log and a **Cancel** button;
  - partial results are kept.
- **Step 5:**
  - **holdout lock**: the leaderboard shows CV/dev results, and "Reveal final test results" is a deliberate one-time action recorded in the report;
  - a metric glossary tooltip;
  - a fold-level table;
  - a side-by-side comparison of two models;
  - export of an HTML report (charts plus tables).
- **Step 6:**
  - group the form fields by type, collapsing them when there are more than 20 fields;
  - show the top contributing features for a single prediction (SHAP for tree models if installed, otherwise permutation-based local deltas).
- **Experiment history:**
  - save each run (config, params, metrics, bundle path) to `artifacts/experiments/<timestamp>/`;
  - a sidebar "History" page to reopen or compare runs.
- **Confirmation dialog** (`st.dialog`) before any change that clears trained results.
- **Accessibility:** every chart has a "Show table" toggle; status is never shown by colour alone.
- **Acceptance:** the AppTest flow still passes; a new AppTest covers cancel, holdout reveal and history reopen.

### P19 - Research rigour and report
**Fixes:** review #8; F10, F11.
- **Benchmark:**
  - Nadeau-Bengio corrected resampled t-test.
  - RF against **each** booster, with Holm correction across boosters and datasets.
  - Report effect sizes (mean difference and its 95 % corrected CI), not only p-values.
- **Dengue:** two pre-registered tasks, *nowcast* and *1-week-ahead forecast* (lagged features only), both with forward-chaining CV. Optional RF + ARIMA hybrid (stretch S3).
- **Tea:** grouped CV by `sale_id`.
- **Telco:** fetch it with a Kaggle login and include it as the main lecturer-requested demonstration (RF vs XGBoost/LightGBM/CatBoost); verify the CC0 licence on the page.
- **Update `docs/REPORT.md`:** new results; a validation-design section (random vs grouped vs time); a limitations section on aggregate vs individual patient data; the effect of tuning (P15); the threshold and calibration discussion.
- **Acceptance:** the report contains no synthetic results; every number traces to `reports/benchmark_summary.md`.

### P20 - Optional extras (only if time remains)
- SHAP global and local explanations; learning curves; data-size sampling for fast exploration.
- A memory guard: estimate the one-hot width before training and switch to target encoding above a limit.
- Model card export (Markdown) per trained model.
- Docker file and a one-command launcher (`run.bat`).

---

## D. Order and effort

| Order | Phase | Why first | Effort |
|---|---|---|---|
| 0 | Q1 Footer | Quick win; satisfies the CC BY attribution duty | S |
| 0 | Q2 Streamlit Cloud readiness | A live demo link for the presentation; independent of the other phases | S-M |
| 1 | P11 Validation and holdout ⚑ | Biggest correctness and viva risk; the basis for P14 and P15 | L |
| 2 | P12 Cleaner types and locale ⚑ | Core claim ("generalised preprocessing") | M |
| 3 | P13 Imbalance ⚑ | Methodological correctness | M |
| 4 | P15 Hyperparameters | Explicit user request; depends on P11 CV | L |
| 5 | P14 Positive class, threshold | Metric consistency | M |
| 6 | P16 Leakage decisions | Small, high explanatory value | S |
| 7 | P17 Prediction validation | Robustness | S |
| 8 | P18 UI/UX | Presentation quality | L |
| 9 | P19 Benchmark and report | Final write-up on the corrected method | M |
| 10 | P20 Extras | Optional | - |

**Before the demo regardless:** fetch the real telco data, finish the public-data benchmark, and update `docs/REPORT.md`.
