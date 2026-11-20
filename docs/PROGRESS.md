# Progress

| Phase | Title | Status | Notes |
|---|---|---|---|
| 0 | Scaffold & environment | done | venv at .venv, sklearn 1.5.2, ruff clean, Graphify graph built |
| 1 | Core types, loader, sample datasets | done | synthetic samples built here were later removed (public data only) |
| 2 | Cleaning, profiling, validation | done, reviewed | Review fixed 8 defects (see below) |
| 3 | Preprocessing engine | done | leakage spy test; 6 datasets x 2 profiles generalisation test |
| - | Public Sri Lankan datasets | done | colmozzie dengue bundled (CC0); Kaggle telco builder (needs Kaggle login) |
| 4 | Model registry, ensembles, hybrid | done | R1 gate passed on sklearn 1.5.2; pipeline_factory done early |
| 5 | Training & evaluation engine | done | 171 tests, 96% coverage; real dengue data trains (RF R2 > 0.3) |
| 6 | Inference & persistence | done | Predictor + joblib bundles; save_bundle takes input_spec instead of schema |
| 7 | UI steps 1-3 | done | frozen AppState; editor <-> config helpers tested |
| 8 | UI steps 4-6 | done | AppTest drives full wizard (classification + real dengue regression) |
| 9 | Generalization proof & benchmark | done | 12 datasets incl. 6 random chaos frames; repeated-CV benchmark + Wilcoxon + ablation |
| 10 | Docs, report, polish | done | README, USER_GUIDE, REPORT with benchmark results; 198 tests, 96% coverage |

## Datasets (public only; synthetic samples removed at the user's request)
| Key | Source | Task | Licence | Status |
|---|---|---|---|
| `lk_tea_auction_catalogue` | Colombo Tea Auction (HF colombo-tea-auction-prices/cta-price-prediction) | Classification, 4 classes | CC BY 4.0 | bundled |
| `lk_tea_auction_price` | same file | Regression | CC BY 4.0 | bundled |
| `lk_colombo_dengue` | colmozzie (Epidemiology Unit weekly reports) | Regression | CC0 | bundled |
| `lk_telco_churn` | Kaggle, Lasal Jayawardena, Sri Lankan operator 2023 | Classification | CC0 | run `python scripts/fetch_public_datasets.py telco` after a Kaggle login |

## v2 improvements (docs/04_IMPROVEMENT_PLAN.md)
| Phase | Title | Status | Notes |
|---|---|---|---|
| Q1 | Footer | done | version, repo link, escaped dataset attribution + licence, privacy note |
| Q2 | Streamlit Cloud readiness | done | pinned requirements, packages.txt, config.toml, cloud mode, smoke test, DEPLOYMENT.md |
| P11 | Validation + holdout discipline | done | random/grouped/time splits; holdout before target-based recommendations; incomplete CV status; OOF dev diagnostics |
| P12 | Cleaner types + number format | done | user types honoured, decimal comma, leading-zero codes, parse report, log1p guard, Yeo-Johnson target |
| P13 | Imbalance correctness | done | SMOTE-NC/SMOTE-N, random oversampling, effective strategy per model |
| P14 | Positive class, threshold, calibration | done | positive class used by metrics, curves, importance, predictor; threshold tool; calibration plot |
| P15 | Hyperparameters | done | ParamSpec per model with help text; Default/Manual/Auto-tune (random search on dev rows, same CV); presets |
| P16 | Leakage decisions | done | per-column "known at prediction time?" review blocks step 4 until answered |
| P17 | Prediction validation | done | quality report: unparseable, out-of-range, unseen categories, missing share |
| P18 | UI/UX | done (core) | model cards, holdout lock, history, reset warning, footer; background training/cancel and HTML export not done |
| P19 | Benchmark + report | done | app engine with each dataset's validation design; simple fold-win comparison; ablation; report rewritten |
| P20 | Optional extras | not started | SHAP, learning curves, memory guard, model cards export, Docker |

## Extensions (docs/05_EXTENSION_PLAN.md)
| Phase | Title | Status | Notes |
|---|---|---|---|
| - | Imputation per column | done | type-aware options, bulk apply, validated (no silently ignored choices) |
| E1 | Training time, row cap, background training, cancel | done | estimate range + warnings; fold-level progress; cancel keeps finished models |
| E2 | Several targets | done | MultiOutput wrappers; averaged + exact-match + Hamming metrics; per-target tables and predictions |
| E3 | Forecasting | done | past-only lag/rolling/calendar features; naive + seasonal naive; RF + ARIMA hybrid; future forecasts |
| E4 | Input add-ons | done | JSON / Parquet; apostrophe thousands; small-data warnings |
| E6 | Grid search, regularisation options, real defaults | done | grid of values centred on defaults (max 200); L1/L2/elastic net logreg; XGB/LGBM L1+L2; DT pruning |
| E7 | Row options, Explore data, logo | done | duplicates / incomplete rows / column missing limit; distributions, target relations (training rows), correlations; SVG logo + Figma brand file |
| E8 | Deployment | done | live at https://tabulaml.streamlit.app; package __init__ starts app.py when used as main file; footer layout fixed |
| E5 | RF dashboard, card design, responsive, end-to-end checks | done | head-to-head fold counts; built-in importance; mobile layout; Playwright walk-through |

## Open decisions / blockers
- [ ] Fetch the real telco dataset (needs a free Kaggle account): set `KAGGLE_USERNAME`/`KAGGLE_KEY` or download the zip and pass `--kaggle-dir`.

## Review log
- **Review of Phases 1-2** (regression tests in `tests/unit/test_review_regressions.py`):
  1. Loader: CSV rows wider than the header were shifted into the index (every column misaligned) -> now a `DataLoadError`.
  2. Loader: blank header cells became `"nan"` -> `"unnamed"`.
  3. Cleaner: numeric columns turned `"1,200"` / text numbers into NaN at prediction time -> shared numeric parser.
  4. Cleaner: boolean columns rejected `1`/`0` at prediction time -> accepted.
  5. Cleaner: mixed timezone offsets crashed datetime parsing -> parsed as UTC, then made naive.
  6. Profiler: any sorted unique integer feature was marked IDENTIFIER (dropped by default) -> only consecutive-integer row counters or ID-like names.
  7. Validation: minimum rows per class ignored the test split -> requires `ceil(k / 0.8) + 1` rows.
  8. Leakage: singleton categories counted as "pure", and target case variants were separate classes -> only repeated categories count; shared `clean_target`.
  Also: `TrainingRun.leaderboard()` no longer raises KeyError when every model failed.
- Phase 3: `TargetEncoder` read integer regression targets as multiclass -> `target_type="continuous"` for regression.
- Encoding: a PowerShell 5.1 rewrite corrupted the euro/pound characters in the currency regex (commit 184e2ce) -> now ASCII `\u` escapes plus a test. Do not rewrite text files with PowerShell `Get-Content`/`Set-Content`.

## Contract changes log
- E1-E4: see `docs/02_ARCHITECTURE.md §4.3` (ForecastConfig; `max_rows`, `extra_targets`, `forecast` on PreprocessingConfig; `per_target`, `cancelled`, `target_labels`, `test_order`; `train_models(should_stop=)`; `list_models(forecasting=)`; `Predictor(target_labels=)`).
- v2: `ValidationScheme`; `PreprocessingConfig` gained `validation`, `group_column`, `order_column`, `positive_class`, `decimal`, `thousands`, `model_params`, `tuning`; `ImbalanceStrategy.RANDOM_OVERSAMPLE`; `ModelResult` gained `params`, `n_failed_folds`, `effective_imbalance`, `oof_pred/oof_proba/oof_mask`, `tuning_history`, status `incomplete`; `TrainingRun` gained `y_dev`, `positive_index`.
- v2: `training/splits.py` (`make_holdout`, `make_cv`, `describe`), `training/tuning.py` (`tune`), `training/samplers.py` (`CategoryAwareSMOTE`), `models/params.py` (`ParamSpec`, `validate_params`, `search_space`), `workflow.py` (`Roles`, `auto_configure`, `suggest_validation`).
- v2: `trainer.prepare()` returns `Prepared` (with original row `index`); duplicates are judged on whole rows so the split never depends on feature selection. `compute_metrics/scoring_map/permutation_importance_df` take `positive`. `DataCleaner(column_types, decimal, thousands)` + `parse_report()`. `Predictor(threshold, positive_index)` + `quality_report()`. Bundles store `params`, `threshold`, `positive_index`. `DatasetCard` gained `source_url`, `validation`, `group_column`, `order_column`.
- Phase 1: `SAMPLE_REGISTRY` constant became `_registry()` in `data/samples.py`; `list_samples` hides cards whose CSV is not present; `load_sample` raises `FileNotFoundError` with fetch instructions.
- Phase 1: `TrainingRun.notes` added; `cv_scores` hold display-oriented (positive) values; `LOWER_IS_BETTER` lives in `core/types.py`.
- Review: `clean_target(s, regression)` in `preprocessing/cleaning.py`. **The trainer must use it** to build `y`.
- Phase 3: `build_feature_transformer` outputs **numpy** (XGBoost rejects feature names containing `[`, `]`, `<`). `recommend_config(..., y=None)` takes the raw target. Numeric "auto" choices are resolved per profile by the builder and learned in `fit`. Added `describe_pipeline`, `column_steps`, public `parse_datetime`.
- Phase 4: `build_model_pipeline(model_key, config, y=None, base_keys=None, as_base=False)`; `RECOMMENDED`, `ENSEMBLE_KEYS`, `class_weight_params` in the registry.
