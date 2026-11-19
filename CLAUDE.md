# TabulaML — instructions for Claude sessions

Interactive ML workbench: problem type → data (upload or Sri Lankan sample) → generalized preprocessing → model selection → results → predict. Streamlit UI over a pure-Python core package `tabulaml/`.

## Read only what you need (token budget)
- Source of truth for interfaces: `docs/02_ARCHITECTURE.md §4` (contracts) and `§5` (preprocessing rules).
- The task for this session: the matching phase in `docs/03_IMPLEMENTATION_PLAN.md`. Read **only that phase**.
- Background (read only if asked): `docs/01_PROJECT_DEFINITION.md`.
- Status: `docs/PROGRESS.md`. Update it at the end of every phase.
- Do not sweep the repo. If `graphify-out/` exists, use `graphify query "..."` / `graphify explain "<symbol>"` before opening files, and open only the line ranges you need. Run `graphify update .` after finishing a phase.

## Commands (Windows, PowerShell)
```
.venv\Scripts\activate
pip install -r requirements-dev.txt; pip install -e .
pytest -q --cov=tabulaml --cov-report=term-missing
ruff check . ; ruff format .
streamlit run app.py
python scripts/generate_samples.py
python scripts/benchmark.py --quick
```

## Hard rules
1. **No dataset-specific code** in `tabulaml/`. Behaviour comes from the schema and config only. Never mention sample column names outside `scripts/`, `data/` and tests.
2. **Leakage safety:** every fitted step lives inside the model pipeline (`DataCleaner → ColumnTransformer → [sampler] → model`). The test split happens before any fit. CV fits the whole pipeline. Never `fit` on the full dataset before splitting.
3. **Immutability:** never mutate an input DataFrame or config. Use frozen dataclasses with `dataclasses.replace` / `with_column`, and use `df.copy()` inside transformers.
4. **UI has no ML logic:** `tabulaml/ui/` only calls functions in §4.2. Core modules never import streamlit.
5. A failing model must not abort a run. Return `ModelResult(status="failed")`.
6. Unseen categories and missing columns at prediction time must never crash.
7. Tests first (pytest). Coverage on `tabulaml/` excluding `ui/` must be at least 80 %. Functions under 50 lines, files under 400 lines. Use type hints. No `print`; use `logging`.
8. Samples: `lk_colombo_dengue` and `lk_telco_churn` are **real public CC0 data** (keep their attribution); `lk_synthetic_*` are **synthetic Sri Lankan-context data** and must always be labelled synthetic. Never invent real company, hospital or person names.
9. Git: do not commit unless the user asks. Commit messages use conventional commits. **Never add Claude/AI co-author or "Generated with" lines.**
10. If the plan conflicts with the contracts, the contracts win. If a contract must change, edit `docs/02_ARCHITECTURE.md §4` in the same change and say so.

## Version pins to respect
scikit-learn `>=1.5,<1.6` (XGBoost/CatBoost tag compatibility inside stacking; see Architecture risk R1). Upgrade only if `tests/unit/test_registry.py` passes.
