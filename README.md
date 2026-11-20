<p align="center"><img src="tabulaml/ui/assets/logo_wordmark.svg" alt="TabulaML" width="340"></p>

# TabulaML

**An interactive machine learning workbench for tabular data, with generalized preprocessing, Random Forest vs boosting comparison, hybrid models and forecasting.**

TabulaML takes you from a raw table to a trained, compared and usable model in six guided steps. It follows the workflow used in professional practice: define the problem, check the data, hold out a test set **before** any decision, keep every preprocessing step inside the model pipeline, compare models with cross-validation, test once, then predict or export.

![workflow](https://img.shields.io/badge/workflow-6%20steps-2a78d6) ![python](https://img.shields.io/badge/python-3.12-2a78d6) ![tests](https://img.shields.io/badge/tests-pytest%20%2B%20AppTest-1baf7a) ![data](https://img.shields.io/badge/data-public%20Sri%20Lankan-eb6834)

---

## Features at a glance

| Area | What you get |
|---|---|
| **Problem types** | Classification (binary / multi-class), regression, **forecasting (time series)**, and **several targets at once** (multi-label, multi-output) |
| **Data** | CSV, TXT, Excel (sheet picker), JSON, Parquet. Delimiter and encoding detection. Number formats `1,234.5`, `1.234,5`, `1 234,5`, `1'234.5` |
| **Validation** | Random, grouped (rows of one customer or sale stay together) or time-ordered. The test set is held out first; recommendations use training rows only |
| **Explore** | Distribution of every column (histogram, box plot, top categories, counts over time), relationship with the target on training rows only, and a correlation heatmap |
| **Preprocessing** | Automatic, schema-driven, with a reason for every decision; or Custom per-column type, imputation, encoding, scaling, transforms, outliers. Leakage review. Pipeline preview per model family |
| **Models** | Linear baselines, Decision Tree, **Random Forest**, Extra Trees, Bagging, AdaBoost, Gradient Boosting, HistGB, **XGBoost, LightGBM, CatBoost**, SVM, KNN, ANN (MLP), Voting, Stacking |
| **Hybrids** | **RF + ANN** (regression): a neural network learns the forest's residuals. **RF + ARIMA** (forecasting): ARIMA models the time pattern, a forest learns what ARIMA missed |
| **Hyperparameters** | Default, manual (each setting explained) or auto-tuned (random search with the same CV, training rows only) |
| **Training** | Time estimate with warnings, a row cap for speed, background training with live progress and a **Cancel** button |
| **Results** | CV leaderboard (mean ± std), fold-by-fold comparison, **Random Forest dashboard** (RF vs boosting head-to-head, inside the forest), out-of-fold diagnostics, decision threshold, calibration, a test set locked until you reveal it, permutation importance |
| **Use the model** | Single-record form, batch scoring with a data-quality check, **forecast the next periods**, `.joblib` and `.pkl` model downloads |
| **Interface** | Card-based dashboard design, light / dark / system theme, responsive on phones and tablets |

## The six steps
1. **Problem type:** classification, regression or forecasting, shown as cards with examples.
2. **Data:** upload a file or load a public Sri Lankan sample (with its dataset card). Choose the target (or several targets), how the model is tested, the number format and the positive class. In forecasting mode you also set the time column, an optional series column, the horizon, the lags, the seasonal lag and the rolling means. The app checks that the series can be forecast honestly.
3. **Data quality and preprocessing:** the test rows are held out first. Then:
   - a quality report;
   - a leakage review, where you decide for each suspicious column whether its value is known at prediction time;
   - an Automatic or Custom pipeline, including the imputation method per column;
   - a pipeline preview;
   - evaluation settings: test share, folds, seed and row cap.
4. **Models:** pick model cards, or use the **Recommended set**. Set hyperparameters, read the time estimate, press **Train & Predict**, and watch the progress card. You can cancel at any time and keep the finished models.
5. **Results:** the leaderboard, a comparison chart, the RF dashboard and out-of-fold diagnostics. The **final test** stays locked until you reveal it once. Forecasting also shows the test period and whether the model beats the naive baseline.
6. **Predict:** a single record, a batch file or the next periods of a forecast. Download the model.

Full walkthrough: **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)**.

## Quick start (Windows, Python 3.12)
```bash
git clone https://github.com/Dinusha-Ekanayake/TabulaML.git
cd TabulaML
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
streamlit run app.py
```
Then open http://localhost:8501. On macOS or Linux, activate with `source .venv/bin/activate`.

## Datasets
Only **publicly available, genuine Sri Lankan datasets** are bundled. There is no synthetic data.

| Key | What | Task | Source and licence |
|---|---|---|---|
| `lk_tea_auction_catalogue` | 12,233 Colombo Tea Auction quotes (Nov 2023 - Mar 2026). Predict the market catalogue | Classification (4 classes) | [Hugging Face: colombo-tea-auction-prices/cta-price-prediction](https://huggingface.co/datasets/colombo-tea-auction-prices/cta-price-prediction). **CC BY 4.0** |
| `lk_tea_auction_price` | Same table. Predict the mid auction price (LKR/kg) | Regression | Same source, CC BY 4.0 |
| `lk_colombo_dengue` | Weekly dengue cases in Colombo with weather, 2009-2014 (279 weeks) | Regression / **forecasting** | R package [colmozzie](https://github.com/thiyangt/colmozzie) (Thiyanga Talagala), from Epidemiology Unit reports. **CC0** |
| `lk_telco_churn` | First-month usage of 15,000 anonymised customers of a Sri Lankan operator, 2023 | Classification | [Kaggle: Real World Customer Churn Dataset in Telco Domain](https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn). **CC0**. Fetched on demand (needs a free Kaggle login) |

Every sample has a dataset card: provenance, licence, period, feature dictionary, target definition and limitations. Rebuild the bundled files with `python scripts/fetch_public_datasets.py public`. To fetch the telco data, set `KAGGLE_USERNAME` and `KAGGLE_KEY`, then run `python scripts/fetch_public_datasets.py telco`.

## Deploy on Streamlit Community Cloud
1. Push `main` to GitHub.
2. At **share.streamlit.io**, choose **Create app** and pick this repository, branch `main`, file `app.py`.
3. Under **Advanced settings**, choose Python 3.12 and add the secret `TABULAML_CLOUD = "1"`. This lowers the memory-heavy limits.
4. Click **Deploy**.

Details and troubleshooting: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

## Quality
```bash
pytest -q --cov=tabulaml          # unit, integration and headless UI tests (coverage gate 80 %)
ruff check . && ruff format --check .
python scripts/benchmark.py --quick
```
The tests include:
- a leakage spy: learned steps never see validation or test rows;
- a generalization suite: one code path for any table;
- proofs that grouped and time-ordered splits never mix groups or use the future;
- checks that forecast features use only past values;
- cancellation and background-training tests;
- multi-output tests;
- AppTest flows through the whole wizard for every problem type.

## Research benchmark
`python scripts/benchmark.py` runs the app's own engine on every dataset with its validation design. It compares Random Forest with XGBoost, LightGBM, CatBoost and Gradient Boosting, and counts fold wins, losses and ties. An ablation shows what leakage control and grouped validation change. Results: [reports/benchmark_summary.md](reports/benchmark_summary.md).

## Project layout
```
app.py                      Streamlit entry point
tabulaml/core               immutable data contracts
tabulaml/data               loading, samples, profiling, validation, leakage detection
tabulaml/preprocessing      cleaner, transformers, recommender, pipeline builder
tabulaml/models             model registry, voting / stacking, RF + ANN hybrid, hyperparameters
tabulaml/forecasting        forecast features, checks, naive baselines, RF + ARIMA, future forecasts
tabulaml/training           splits, pipeline factory, samplers, tuning, trainer, time estimate, background jobs
tabulaml/evaluation         metrics (incl. multi-output), importance, RF insights, Plotly figures
tabulaml/inference          predictor and model bundles
tabulaml/workflow.py        holdout-first automatic configuration
tabulaml/ui                 Streamlit steps and cards (no ML logic)
scripts/                    dataset fetcher, benchmark, deployment smoke test
docs/                       user guide, architecture, plans, report, deployment
```

## Capabilities and limits
- **Supported:**
  - one table per file;
  - classification, regression, several targets of the same kind, and time-series forecasting, including several series in one file;
  - up to 200 MB and 1 million cells per upload (400k cells in cloud mode).
- **Forecasting** is regression on past-only features, plus naive baselines and RF + ARIMA. The RF + ARIMA hybrid supports a single series. Seasonal ARIMA, Prophet and deep-learning forecasters are out of scope.
- **Not supported:**
  - multiple linked tables;
  - images, audio or video;
  - clustering or anomaly detection;
  - deep NLP. Long free text becomes simple text statistics.
- **Imbalance handling:** class weights and resampling need a single target, so they are off when you predict several targets.
- **scikit-learn** is pinned below 1.6 so XGBoost and CatBoost work inside stacking.
- **Model bundles** are pickle files. Only load bundles you created yourself.

## Documentation
- [User guide](docs/USER_GUIDE.md): every screen and option.
- [Deployment](docs/DEPLOYMENT.md): Streamlit Community Cloud.
- [Architecture](docs/02_ARCHITECTURE.md): design, contracts and leakage-safe flow.
- [Report](docs/REPORT.md): problem, method, experiments and discussion.
- [Changelog](CHANGELOG.md).

## Licence and attribution
The code is released under the MIT licence (see [LICENSE](LICENSE)). The bundled datasets keep their own licences: the tea auction data is CC BY 4.0 (attribution required), and the dengue and telco data are CC0. © 2026 Dinusha Ekanayake.
