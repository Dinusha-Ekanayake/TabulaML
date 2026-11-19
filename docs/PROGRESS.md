# Progress

| Phase | Title | Model | Status | Notes |
|---|---|---|---|---|
| 0 | Scaffold & environment | Sonnet | ☑ | venv at .venv, sklearn 1.5.2, ruff clean, Graphify graph built |
| 1 | Core types, loader, sample datasets | Sonnet | ☑ | 27 tests, 95% coverage; churn 4999x24, LOS 2999x22 |
| 2 | Cleaning, profiling, validation | Sonnet + ⚑ Opus review | ☑ reviewed | 77 tests, 95% coverage; Opus review fixed 8 defects (see below) |
| 3 | Preprocessing engine | Sonnet + ⚑ Opus review | ☐ | |
| 4 | Model registry, ensembles, hybrid | Sonnet | ☐ | R1 compatibility gate |
| 5 | Training & evaluation engine | Sonnet + ⚑ Opus review | ☐ | |
| 6 | Inference & persistence | Sonnet | ☐ | |
| 7 | UI steps 1–3 | Sonnet | ☐ | |
| 8 | UI steps 4–6 | Sonnet | ☐ | |
| 9 | Generalization proof & benchmark | Sonnet | ☐ | |
| 10 | Docs, report, polish | Sonnet | ☐ | |

## Open decisions / blockers
- [ ] Confirm with the lecturer whether **synthetic Sri Lankan-context** samples are acceptable, or whether a genuine local dataset is required (see Definition §7 and stretch phase S5).

## Review log
- **Opus review of Phases 1–2** (regression tests in `tests/unit/test_review_regressions.py`):
  1. Loader: CSV rows wider than the header were shifted into the index (every column misaligned) → now a `DataLoadError`.
  2. Loader: blank header cells became `"nan"` → `"unnamed"`.
  3. Cleaner: numeric columns turned `"1,200"` / text numbers into NaN at prediction time → shared numeric parser.
  4. Cleaner: boolean columns rejected `1`/`0` at prediction time → accepted.
  5. Cleaner: mixed timezone offsets crashed datetime parsing → parsed as UTC, then made naive.
  6. Profiler: any sorted unique integer feature was marked IDENTIFIER (dropped by default) → only consecutive-integer row counters or ID-like names.
  7. Validation: minimum rows per class ignored the test split, so CV folds could miss a class → requires `ceil(k / 0.8) + 1` rows.
  8. Leakage: singleton categories counted as "pure", and target case variants (`Yes`/`yes `) were separate classes → only repeated categories count; shared `clean_target`.
  Also: `TrainingRun.leaderboard()` no longer raises KeyError when every model failed.

## Contract changes log
- Review: added `clean_target(s, regression) -> pd.Series` to `preprocessing/cleaning.py`. **Phase 5 trainer must use it** to build `y` so validation and training see identical classes.
- Phase 1: `SAMPLE_REGISTRY` constant became the function `_registry()` in `data/samples.py` (discovers `data/samples/*.json`); public API `list_samples`/`load_sample` unchanged.
- Phase 1: `TrainingRun.notes: tuple[str, ...] = ()` added; `ModelResult`/`TrainingRun` runtime fields are excluded from repr/compare. `cv_scores` hold display-oriented (positive) values; `LOWER_IS_BETTER` lives in `core/types.py`.
