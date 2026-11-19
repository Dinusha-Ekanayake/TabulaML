# TabulaML

**An interactive machine learning workbench with generalized data preprocessing and ensemble model comparison.**

TabulaML takes you from a raw table to a trained, compared and usable model in six steps:

1. **Problem type:** classification or regression.
2. **Data:** upload a CSV/Excel file, or load a bundled **public Sri Lankan dataset**. Choose the target and how the model should be tested: **random**, **grouped** (rows of the same customer or sale stay together) or **time-ordered** (train on the past, test on later rows). Also choose the number format and, for binary tasks, the positive class.
3. **Preprocessing:** the test rows are **held out first**, and every target-based recommendation uses the training rows only. You get a data-quality report, a **leakage review** where you decide each suspicious column, an **Automatic** pipeline (every decision explained) or **Custom** per-column overrides, and a preview of the pipeline each model family gets.
4. **Models:** pick **model cards**: linear baselines, Decision Tree, Random Forest, Extra Trees, Bagging, AdaBoost, Gradient Boosting, HistGB, **XGBoost, LightGBM, CatBoost**, SVM, KNN, ANN (MLP), **Voting**, **Stacking** and a **Random Forest + ANN hybrid**. Each model can use **default**, **manual** or **auto-tuned** hyperparameters (random search on the training rows only).
5. **Results:** the test set is **locked** until you reveal it. Until then you see a cross-validation leaderboard (mean ± std), fold-by-fold comparison and out-of-fold diagnostics: confusion matrix, ROC/PR curves, decision threshold, calibration and residuals. After the reveal you see the final test scores and feature importance. Download the leaderboard and a JSON run report.
6. **Predict:** score a single record through an auto-generated form, or a whole file after a quality check (unparseable values, values outside the training range, new categories). Download the trained model bundle.

The preprocessing engine has no dataset-specific code. It reads the schema of whatever table you load (numeric, categorical, boolean, datetime, identifier, constant, free text). It cleans common problems: `"Rs. 1,250"`, decimal commas, `?`/`NA` tokens, `Yes`/`y`/`no` booleans, spelling variants, mixed date formats, and codes with leading zeros. It honours the column types you choose, and builds a separate pipeline for tree models and for scale-sensitive models. Every learned step, including SMOTE (kept category-safe with SMOTE-NC), is fitted on training folds only. Every decision comes with a plain-language reason.

## Datasets
Only **publicly available, genuine Sri Lankan datasets** are used. There is no synthetic data.

| Key | What | Task | Source and licence |
|---|---|---|---|
| `lk_tea_auction_catalogue` | 12,233 Colombo Tea Auction price quotes (Nov 2023 - Mar 2026). Predict the market catalogue: Low Grown, High Grown, Dust or Off-Grade | Classification (4 classes) | [Hugging Face: colombo-tea-auction-prices/cta-price-prediction](https://huggingface.co/datasets/colombo-tea-auction-prices/cta-price-prediction), from Forbes & Walker broker reports + Open-Meteo weather. **CC BY 4.0** (attribution required) |
| `lk_tea_auction_price` | Same table. Predict the mid auction price (LKR/kg) | Regression | Same source, CC BY 4.0 |
| `lk_colombo_dengue` | Weekly notified dengue cases in Colombo with weather, 2009-2014 (279 weeks) | Regression | R package [colmozzie](https://github.com/thiyangt/colmozzie) (Thiyanga Talagala), from the Epidemiology Unit's Weekly Epidemiological Reports. **CC0** |
| `lk_telco_churn` | First-month usage of 15,000 anonymised customers of a large Sri Lankan operator, 2023 | Classification | [Kaggle: Real World Customer Churn Dataset in Telco Domain](https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn) (Lasal Jayawardena). **CC0**. *Fetched on demand (needs a free Kaggle login), see below* |

Every sample has a dataset card (provenance, licence, period, feature dictionary, target definition, limitations), which is shown in the app. Two of the datasets contain real leakage columns, which the app detects and excludes:
- In the tea catalogue task, `category`: every tea category is listed in exactly one catalogue.
- In the telco data, `active_days_61_90_any_service`: it is observed after the churn window.

The bundled files can be rebuilt from their pinned public sources with `python scripts/fetch_public_datasets.py public`.

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

## Live deployment
Ready for Streamlit Community Cloud: see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Pinned `requirements.txt`, `packages.txt` and `.streamlit/config.toml` are included; `python scripts/smoke_deploy.py --cloud` checks a headless start.

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
- a **generalization test**: one unchanged code path trains and predicts on sklearn datasets, the Sri Lankan datasets and seeded random mixed-type tables;
- tests proving that grouped and time-ordered splits never mix groups or train on the future, and that test labels never influence the automatic configuration;
- an **AppTest** that clicks through the whole wizard (leakage review, model cards, holdout reveal, prediction).

## Research benchmark
`python scripts/benchmark.py` runs the app's own engine on every dataset with its validation design (tea grouped by sale, dengue in time order). It compares Random Forest with XGBoost, LightGBM, CatBoost, Gradient Boosting and a linear baseline: cross-validation mean ± std, the held-out test score, and *how many folds Random Forest wins* against each booster. An ablation shows what leakage control and grouped validation change. Results go to `reports/benchmark_summary.md` and `reports/figures/`.

## Project layout
```
app.py                      Streamlit entry point
tabulaml/core               immutable data contracts
tabulaml/data               loading, samples, profiling, validation and leakage detection
tabulaml/preprocessing      cleaner, custom transformers, recommender, pipeline builder
tabulaml/models             model registry, voting/stacking, RF+ANN hybrid
tabulaml/training           splits (random/grouped/time), pipeline factory, samplers, tuning, trainer
tabulaml/evaluation         metrics, permutation importance, Plotly figures
tabulaml/inference          predictor and model bundles
tabulaml/workflow.py        holdout-first automatic configuration
tabulaml/ui                 Streamlit steps, model cards, history, footer (no ML logic)
scripts/                    public dataset fetcher, benchmark, deployment smoke test
docs/                       definition, architecture, plan, user guide, report
```

## Notes and limitations
- Scope: tabular classification and regression. Images, audio, long free text and dedicated forecasting models (for example ARIMA) are out of scope; time-ordered validation is supported.
- scikit-learn is pinned below 1.6 so that XGBoost and CatBoost behave inside stacking. A test checks this before any upgrade.
- Model bundles are pickle files: only load bundles you created yourself.
- The dengue data uses same-week weather, so it estimates the current week's cases (a nowcast), not a forecast. A pure one-week-ahead forecast would use only the lagged columns.
