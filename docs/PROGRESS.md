# Progress

| Phase | Title | Model | Status | Notes |
|---|---|---|---|---|
| 0 | Scaffold & environment | Sonnet | done | venv at .venv, sklearn 1.5.2, ruff clean, Graphify graph built |
| 1 | Core types, loader, sample datasets | Sonnet | done | synthetic samples built here were later removed (public data only) |
| 2 | Cleaning, profiling, validation | Sonnet + Opus review | done, reviewed | Opus review fixed 8 defects (see below) |
| 3 | Preprocessing engine | Opus | done | leakage spy test; 6 datasets x 2 profiles generalisation test |
| - | Public Sri Lankan datasets | Opus | done | colmozzie dengue bundled (CC0); Kaggle telco builder (needs Kaggle login) |
| 4 | Model registry, ensembles, hybrid | Opus | done | R1 gate passed on sklearn 1.5.2; pipeline_factory done early |
| 5 | Training & evaluation engine | Opus | done | 171 tests, 96% coverage; real dengue data trains (RF R2 > 0.3) |
| 6 | Inference & persistence | Opus | done | Predictor + joblib bundles; save_bundle takes input_spec instead of schema |
| 7 | UI steps 1-3 | Opus | done | frozen AppState; editor <-> config helpers tested |
| 8 | UI steps 4-6 | Opus | done | AppTest drives full wizard (classification + real dengue regression) |
| 9 | Generalization proof & benchmark | Opus | done | 12 datasets incl. 6 random chaos frames; repeated-CV benchmark + Wilcoxon + ablation |
| 10 | Docs, report, polish | Opus | done | README, USER_GUIDE, REPORT with benchmark results; 198 tests, 96% coverage |

## Datasets (public only; synthetic samples removed at the user's request)
| Key | Source | Task | Licence | Status |
|---|---|---|---|---|
| `lk_tea_auction_catalogue` | Colombo Tea Auction (HF colombo-tea-auction-prices/cta-price-prediction) | Classification, 4 classes | CC BY 4.0 | bundled |
| `lk_tea_auction_price` | same file | Regression | CC BY 4.0 | bundled |
| `lk_colombo_dengue` | colmozzie (Epidemiology Unit weekly reports) | Regression | CC0 | bundled |
| `lk_telco_churn` | Kaggle, Lasal Jayawardena, Sri Lankan operator 2023 | Classification | CC0 | run `python scripts/fetch_public_datasets.py telco` after a Kaggle login |

## Open decisions / blockers
- [ ] Fetch the real telco dataset (needs a free Kaggle account): set `KAGGLE_USERNAME`/`KAGGLE_KEY` or download the zip and pass `--kaggle-dir`.

## Review log
- **Opus review of Phases 1-2** (regression tests in `tests/unit/test_review_regressions.py`):
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
- Phase 1: `SAMPLE_REGISTRY` constant became `_registry()` in `data/samples.py`; `list_samples` hides cards whose CSV is not present; `load_sample` raises `FileNotFoundError` with fetch instructions.
- Phase 1: `TrainingRun.notes` added; `cv_scores` hold display-oriented (positive) values; `LOWER_IS_BETTER` lives in `core/types.py`.
- Review: `clean_target(s, regression)` in `preprocessing/cleaning.py`. **The trainer must use it** to build `y`.
- Phase 3: `build_feature_transformer` outputs **numpy** (XGBoost rejects feature names containing `[`, `]`, `<`). `recommend_config(..., y=None)` takes the raw target. Numeric "auto" choices are resolved per profile by the builder and learned in `fit`. Added `describe_pipeline`, `column_steps`, public `parse_datetime`.
- Phase 4: `build_model_pipeline(model_key, config, y=None, base_keys=None, as_base=False)`; `RECOMMENDED`, `ENSEMBLE_KEYS`, `class_weight_params` in the registry.
