# TabulaML

**An interactive machine learning workbench with generalized data preprocessing and ensemble model comparison.**

TabulaML takes you from a raw table to a trained, compared and usable model in six steps:

1. **Problem type:** classification or regression.
2. **Data:** upload a CSV/Excel file, or load a bundled **Sri Lankan sample dataset**, then choose the target.
3. **Preprocessing:** data-quality report, leakage warnings, and an **Automatic** pipeline (every decision explained) or **Custom** per-column overrides, plus a preview of the pipeline each model family gets.
4. **Models:** choose any mix of linear baselines, Decision Tree, Random Forest, Extra Trees, Bagging, AdaBoost, Gradient Boosting, HistGB, **XGBoost, LightGBM, CatBoost**, SVM, KNN, ANN (MLP), **Voting**, **Stacking** and a **Random Forest + ANN hybrid**, then press **Train & Predict**.
5. **Results:** leaderboard (cross-validated mean ± std and held-out test scores), comparison charts, confusion matrix, ROC/PR curves, residual plots and permutation feature importance. Download the leaderboard and a JSON run report.
6. **Predict:** score a single record through an auto-generated form, or a whole CSV, and download the trained model bundle.

The preprocessing engine has no dataset-specific code. It reads the schema of whatever table you load (numeric, categorical, boolean, datetime, identifier, constant, free text), cleans common problems (`"Rs. 1,250"`, `?`/`NA` tokens, `Yes`/`y`/`no` booleans, spelling variants, mixed date formats) and builds a separate pipeline for tree models and for scale-sensitive models. Every learned step is fitted on training folds only.

## Datasets

| Key | What | Task | Source and licence |
|---|---|---|---|
| `lk_colombo_dengue` | Weekly notified dengue cases in Colombo with weather, 2009-2014 (279 weeks) | Regression | **Real, public**: R package [colmozzie](https://github.com/thiyangt/colmozzie) (Thiyanga Talagala), compiled from the Epidemiology Unit's Weekly Epidemiological Reports. CC0 |
| `lk_telco_churn` | First-month usage of 15,000 anonymised customers of a large Sri Lankan operator, 2023 | Classification | **Real, public**: [Kaggle: Real World Customer Churn Dataset in Telco Domain](https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn) (Lasal Jayawardena). CC0. *Fetched on demand, see below* |
| `lk_synthetic_telecom_churn` | Messy synthetic telecom customers | Classification | Synthetic Sri Lankan-context data (`scripts/generate_samples.py`). Stress-tests preprocessing |
| `lk_synthetic_patient_los` | Messy synthetic hospital admissions, length of stay | Regression | Synthetic Sri Lankan-context data. Stress-tests preprocessing |

Every sample has a dataset card (provenance, licence, period, feature dictionary, target definition, limitations), which is shown in the app. The synthetic samples carry a "Synthetic" badge, and each contains one deliberately planted leakage column that the app detects and excludes. The real telco builder keeps one post-outcome column (`active_days_61_90_any_service`) as a real-world leakage example.

### Fetching the real telco dataset
Kaggle requires a free account to download.
```bash
pip install kagglehub
# either: set KAGGLE_USERNAME and KAGGLE_KEY (kaggle.com > Settings > API), then
python scripts/fetch_public_datasets.py telco
# or: download and unzip the dataset manually, then
python scripts/fetch_public_datasets.py telco --kaggle-dir path/to/unzipped/folder
```
The dataset appears in the app automatically once `data/samples/lk_telco_churn.csv` exists.

## Install and run (Windows, Python 3.12)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
streamlit run app.py
```
`requirements.txt` holds the runtime dependencies. `requirements.lock.txt` pins the exact versions that were tested (scikit-learn 1.5.2, see *Notes*).

## Tests and quality
```bash
pytest -q --cov=tabulaml          # unit, integration and headless UI tests; >= 80% coverage gate
ruff check . && ruff format --check .
python scripts/benchmark.py --quick
```
The suite includes:
- a **leakage spy test**: learned steps never see validation or test rows;
- a **generalization test**: one unchanged code path trains and predicts on sklearn datasets, the Sri Lankan samples and seeded random mixed-type tables;
- an **AppTest** that clicks through the whole wizard.

## Research benchmark
`python scripts/benchmark.py` compares Random Forest with XGBoost, LightGBM, CatBoost and Gradient Boosting using paired repeated cross-validation (5 folds × 3 repeats), and tests RF against the best booster with a Wilcoxon signed-rank test. It also runs an ablation that shows the effect of the preprocessing engine and of leakage. The results are written to `reports/benchmark_summary.md` and `reports/figures/`.

## Project layout
```
app.py                      Streamlit entry point
tabulaml/core               immutable data contracts
tabulaml/data               loading, samples, profiling, validation and leakage detection
tabulaml/preprocessing      cleaner, custom transformers, recommender, pipeline builder
tabulaml/models             model registry, voting/stacking, RF+ANN hybrid
tabulaml/training           pipeline factory and trainer
tabulaml/evaluation         metrics, permutation importance, Plotly figures
tabulaml/inference          predictor and model bundles
tabulaml/ui                 Streamlit steps (no ML logic)
scripts/                    sample generator, public dataset fetcher, benchmark
docs/                       definition, architecture, plan, user guide, report
```

## Notes and limitations
- Scope: tabular classification and regression. Images, audio, long free text and time-series forecasting with temporal validation are out of scope.
- scikit-learn is pinned below 1.6 so that XGBoost and CatBoost behave inside stacking. A test checks this before any upgrade.
- Model bundles are pickle files: only load bundles you created yourself.
- Random cross-validation on the weekly dengue series is optimistic compared with forward-in-time evaluation (see the dataset card).
