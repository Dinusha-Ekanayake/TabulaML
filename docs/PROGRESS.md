# Progress

| Phase | Title | Model | Status | Notes |
|---|---|---|---|---|
| 0 | Scaffold & environment | Sonnet | â˜‘ | venv at .venv, sklearn 1.5.2, ruff clean, Graphify graph built |
| 1 | Core types, loader, sample datasets | Sonnet | â˜‘ | 27 tests, 95% coverage; churn 4999x24, LOS 2999x22 |
| 2 | Cleaning, profiling, validation | Sonnet + âš‘ Opus review | â˜‘ reviewed | 77 tests, 95% coverage; Opus review fixed 8 defects (see below) |
| 3 | Preprocessing engine | Opus | â˜‘ | 108 tests; leakage spy test; 6 datasets x 2 profiles generalisation test |
| 4 | Model registry, ensembles, hybrid | Sonnet | â˜ | R1 compatibility gate |
| 5 | Training & evaluation engine | Sonnet + âš‘ Opus review | â˜ | |
| 6 | Inference & persistence | Sonnet | â˜ | |
| 7 | UI steps 1â€“3 | Sonnet | â˜ | |
| 8 | UI steps 4â€“6 | Sonnet | â˜ | |
| 9 | Generalization proof & benchmark | Sonnet | â˜ | |
| 10 | Docs, report, polish | Sonnet | â˜ | |

## Open decisions / blockers
- [ ] Confirm with the lecturer whether **synthetic Sri Lankan-context** samples are acceptable, or whether a genuine local dataset is required (see Definition Â§7 and stretch phase S5).

## Review log
- **Opus review of Phases 1â€“2** (regression tests in `tests/unit/test_review_regressions.py`):
  1. Loader: CSV rows wider than the header were shifted into the index (every column misaligned) â†’ now a `DataLoadError`.
  2. Loader: blank header cells became `"nan"` â†’ `"unnamed"`.
  3. Cleaner: numeric columns turned `"1,200"` / text numbers into NaN at prediction time â†’ shared numeric parser.
  4. Cleaner: boolean columns rejected `1`/`0` at prediction time â†’ accepted.
  5. Cleaner: mixed timezone offsets crashed datetime parsing â†’ parsed as UTC, then made naive.
  6. Profiler: any sorted unique integer feature was marked IDENTIFIER (dropped by default) â†’ only consecutive-integer row counters or ID-like names.
  7. Validation: minimum rows per class ignored the test split, so CV folds could miss a class â†’ requires `ceil(k / 0.8) + 1` rows.
  8. Leakage: singleton categories counted as "pure", and target case variants (`Yes`/`yes `) were separate classes â†’ only repeated categories count; shared `clean_target`.
  Also: `TrainingRun.leaderboard()` no longer raises KeyError when every model failed.

## Contract changes log
- Phase 3: `build_feature_transformer` outputs **numpy** (not pandas): XGBoost rejects feature names containing `[`, `]`, `<`, which one-hot names from user data can contain. `ct.get_feature_names_out()` still works.
- Phase 3: `recommend_config(..., y=None)` takes the raw target for imbalance/target-transform decisions. Numeric "auto" choices (skew, clipping, scaling) and low-cardinality encoding stay "auto" and are resolved per profile by the builder, with data-dependent parts learned in `fit`. High-cardinality columns get explicit `encoding="target"`.
- Phase 3: added `describe_pipeline` and `column_steps` (builder); `parse_datetime` is public in cleaning.
- Review: added `clean_target(s, regression) -> pd.Series` to `preprocessing/cleaning.py`. **Phase 5 trainer must use it** to build `y` so validation and training see identical classes.
- Phase 1: `SAMPLE_REGISTRY` constant became the function `_registry()` in `data/samples.py` (discovers `data/samples/*.json`); public API `list_samples`/`load_sample` unchanged.
- Phase 1: `TrainingRun.notes: tuple[str, ...] = ()` added; `ModelResult`/`TrainingRun` runtime fields are excluded from repr/compare. `cv_scores` hold display-oriented (positive) values; `LOWER_IS_BETTER` lives in `core/types.py`.
