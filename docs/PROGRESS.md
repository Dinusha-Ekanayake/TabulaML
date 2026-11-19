# Progress

| Phase | Title | Model | Status | Notes |
|---|---|---|---|---|
| 0 | Scaffold & environment | Sonnet | ☑ | venv at .venv, sklearn 1.5.2, ruff clean, Graphify graph built |
| 1 | Core types, loader, sample datasets | Sonnet | ☑ | 27 tests, 95% coverage; churn 4999x24, LOS 2999x22 |
| 2 | Cleaning, profiling, validation | Sonnet + ⚑ Opus review | ☐ | |
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

## Contract changes log
- Phase 1: `SAMPLE_REGISTRY` constant became the function `_registry()` in `data/samples.py` (discovers `data/samples/*.json`); public API `list_samples`/`load_sample` unchanged.
- Phase 1: `TrainingRun.notes: tuple[str, ...] = ()` added; `ModelResult`/`TrainingRun` runtime fields are excluded from repr/compare. `cv_scores` hold display-oriented (positive) values; `LOWER_IS_BETTER` lives in `core/types.py`.
