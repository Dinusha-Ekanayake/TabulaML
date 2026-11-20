"""Text of the TabulaML guide (rendered to PDF by build_guide.py)."""

from __future__ import annotations

from datetime import date

LIVE_URL = "https://tabulaml.streamlit.app"
REPO_URL = "https://github.com/Dinusha-Ekanayake/TabulaML"


def _fig(images: str, name: str, number: int, caption: str, phone: bool = False) -> str:
    cls = " class='phone'" if phone else ""
    return (
        f"<figure{cls}><img src='{images}/{name}.jpg' alt='{caption}'>"
        f"<figcaption>Figure {number}. {caption}</figcaption></figure>"
    )


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _cover(assets: str, today: date) -> str:
    return f"""
<div class="cover">
  <img src="{assets}/logo_wordmark.svg" alt="TabulaML">
  <h1>TabulaML</h1>
  <p><b>Application overview and user guide</b></p>
  <p>An interactive machine learning workbench for tabular data</p>
  <p>&nbsp;</p>
  <p>Version 0.2.0, {today.strftime("%B %Y")}</p>
  <p>Dinusha Ekanayake</p>
  <p>Live application: {LIVE_URL}</p>
  <p>Source code: {REPO_URL}</p>
</div>"""


def _contents() -> str:
    items = [
        "1. Introduction",
        "2. Requirements and installation",
        "3. Workflow overview",
        "4. Step-by-step guide",
        "5. Forecasting",
        "6. Feature reference",
        "7. Models and hyperparameters",
        "8. Methodology and safeguards",
        "9. Layout on phones and the dark theme",
        "10. Data sources and licences",
        "11. Troubleshooting",
        "12. Limitations",
        "Appendix: project structure and testing",
    ]
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f"<div class='break'></div><h2>Contents</h2><ul class='toc'>{lis}</ul>"


def _intro() -> str:
    return """
<div class="break"></div>
<h2>1. Introduction</h2>
<p>TabulaML takes a user from a raw table to trained, compared and usable predictive
models through six guided steps. It is designed for students and analysts who want to
apply machine learning correctly without writing code, and for anyone who needs to
explain how a result was obtained.</p>
<p>The application supports classification, regression, several targets at once and
time-series forecasting. Its preprocessing engine contains no dataset-specific code: it
reads the structure of whatever table is loaded, detects the type of every column and
proposes a pipeline with a written reason for each decision. Every step that learns from
the data is fitted inside the model pipeline on training rows only, and the test set is
held out before any target-based decision is made, so the reported test scores are an
honest estimate of performance on new data.</p>
<p>The application compares Random Forest with gradient boosting (XGBoost, LightGBM,
CatBoost) and other model families, offers two hybrid models (Random Forest with a
neural network, and Random Forest with ARIMA for forecasting), and lets the user tune
hyperparameters manually, by random search or by grid search.</p>
"""


def _requirements() -> str:
    rows = [
        ["Python", "3.11 or 3.12 (tested on 3.12.2)"],
        [
            "Operating system",
            "Windows 10/11, macOS 12 or later, or Linux. On Linux, LightGBM "
            "needs the OpenMP runtime (<code>libgomp1</code>).",
        ],
        ["Memory", "4 GB RAM minimum; 8 GB recommended for large tables or stacking"],
        ["Disk space", "About 1.5 GB for the Python environment"],
        ["Browser", "A current Chrome, Edge, Firefox or Safari, on a computer, tablet or phone"],
        [
            "Main libraries",
            "scikit-learn 1.5.2, pandas 2.3.3, numpy 2.2.6, XGBoost 3.4.1, "
            "LightGBM 4.7.0, CatBoost 1.2.10, statsmodels 0.15.0, imbalanced-learn 0.12.4, "
            "Streamlit 1.67.0, Plotly 7.1.0 (pinned in requirements.txt)",
        ],
    ]
    return f"""
<h2>2. Requirements and installation</h2>
<h3>2.1 Using the deployed application</h3>
<p>The application is available at <b>{LIVE_URL}</b>. It runs on Streamlit Community
Cloud and needs only a web browser. Uploaded files are processed in the user's session
and are not stored.</p>
<h3>2.2 Running it locally</h3>
{_table(["Item", "Requirement"], rows)}
<p>Installation on Windows (on macOS or Linux, activate the environment with
<code>source .venv/bin/activate</code>):</p>
<pre>git clone {REPO_URL}.git
cd TabulaML
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements-dev.txt
pip install -e .
streamlit run app.py</pre>
<p>The application then opens at http://localhost:8501. No GPU, database or internet
connection is required; the sample datasets are included in the repository.</p>
"""


def _workflow(img: str) -> str:
    rows = [
        ["1. Problem type", "Choose classification, regression or forecasting."],
        [
            "2. Data",
            "Upload a file or load a sample; choose the target(s), how the model is "
            "tested and the number format.",
        ],
        [
            "3. Preprocessing",
            "Hold out the test set, review data quality, explore the data, "
            "answer the leakage review and accept or adjust the pipeline.",
        ],
        ["4. Models", "Select models, set hyperparameters, read the time estimate and train."],
        ["5. Results", "Compare models with cross-validation, then reveal the test set once."],
        [
            "6. Predict",
            "Score single records or files, forecast the next periods and download "
            "the trained model.",
        ],
    ]
    return f"""
<h2>3. Workflow overview</h2>
<p>The six steps follow the order used in professional practice: define the problem,
inspect the data, hold out a test set before any decision, keep preprocessing inside the
pipeline, compare models with cross-validation, test once, then deploy. The sidebar shows
the steps; earlier steps can always be revisited, and changing an earlier choice clears
the results that depended on it.</p>
{_table(["Step", "Purpose"], rows)}
{
        _fig(
            img,
            "01_problem_type",
            1,
            "Step 1, choosing the problem type. The strip above the "
            "cards shows the standard workflow.",
        )
    }
"""


def _step2(img: str) -> str:
    return f"""
<h2>4. Step-by-step guide</h2>
<h3>4.1 Choosing the data</h3>
<p>Files can be uploaded as CSV, TXT, Excel (with a sheet picker), JSON or Parquet, up to
200 MB. The delimiter and text encoding are detected automatically, and rows with more
values than column names are rejected rather than silently shifted. Alternatively, a
public Sri Lankan sample can be loaded; each sample has a card describing its source,
licence, collection period, features and known limitations.</p>
<p>The target column is checked against the problem type. For classification and
regression, the option <i>Predict several columns at once</i> adds further targets of the
same kind (multi-label or multi-output learning).</p>
{_fig(img, "02_data", 2, "Step 2 with a sample dataset loaded and its card shown.")}
<p>The validation design should match how the model will be used. <i>Random</i> suits
independent rows. <i>Grouped</i> keeps all rows of one entity, for example one weekly
auction sale, entirely in training or entirely in testing. <i>Time-ordered</i> trains on
the past and tests on later rows. The number format (decimal point or decimal comma) and,
for a single binary target, the positive class are also set here.</p>
{_fig(img, "03_validation", 3, "Validation design, number format and positive class.")}
"""


def _step3(img: str) -> str:
    return f"""
<h3>4.2 Data quality and preprocessing</h3>
<p>On entering step 3 the application first holds out the test rows. All recommendations
that depend on the target, such as leakage checks, class-imbalance handling and target
transforms, are made from the training rows only. The <i>Evaluation settings</i> panel sets
the test share, the number of cross-validation folds, the random seed and an optional row
cap for faster training.</p>
{
        _fig(
            img,
            "04_quality",
            4,
            "Data quality report: rows, duplicates, per-column profile, "
            "missing values and the target distribution.",
        )
    }
<p>The <b>Explore data</b> tab helps the user understand the data before preparing it.
For any column it shows summary statistics and a distribution chart (a histogram with a
box plot for numbers, the most common values for categories and counts over time for
dates), together with the column's relationship to the target. The second tab shows a
correlation heatmap and lists the most strongly related pairs. Relationships with the
target are computed on the training rows only.</p>
{
        _fig(
            img,
            "05_explore",
            5,
            "Explore data: summary cards, distribution and relationship with the target.",
        )
    }
<p>The <b>leakage review</b> lists columns that are almost perfectly related to the
target. Such a column may be genuine or may only be known after the outcome. For each
one the user answers whether its value will be known at prediction time; the next step is
blocked until every column has been answered.</p>
{
        _fig(
            img,
            "06_leakage",
            6,
            "Leakage review: the user decides whether a suspicious column may be used.",
        )
    }
<p>The <b>Preprocessing</b> tab first shows which features the models will use, which
columns are excluded and why, and how strongly each used feature relates to the target
(mutual information, and correlation for number features). In <i>Automatic</i> mode every
column receives a recommended treatment with a reason. <i>Custom</i> mode adds full
control: duplicate rows can be kept or dropped, rows with missing values can be imputed or
dropped, columns above a missing-value limit can be excluded, the imputation method can be
set per column (median, mean, most frequent, constant or KNN for numbers; a separate
category or the most frequent value for categories), and type, encoding, scaling,
transformation and outlier handling can be changed per column. Class-imbalance handling
(class weights, oversampling, SMOTE, undersampling) and a regression target transform are
also chosen here.</p>
{_fig(img, "07_rows_missing", 7, "Custom preprocessing: rows and missing values.")}
<p>The <b>Pipeline preview</b> lists the exact steps applied to every column for tree
models and for scale-sensitive models.</p>
{_fig(img, "08_pipeline", 8, "Pipeline preview for both model families.")}
"""


def _step4(img: str) -> str:
    return f"""
<h3>4.3 Selecting and training models</h3>
<p>Models are shown as cards grouped by family. Each card states the preprocessing
pipeline it receives, its typical speed and what it is best suited to. The
<i>Recommended set</i> selects a sound starting combination; <i>All</i> and <i>None</i> act
on a whole group. Models that cannot run with the current settings are shown as
unavailable with the reason.</p>
{_fig(img, "09_models", 9, "Model cards with three models selected.")}
<p>Hyperparameters are set per model in the <i>Model settings</i> panel. <i>Default</i>
uses the model's standard values. <i>Manual</i> starts from the real defaults and explains
every setting, including the regularisation options (for example L1, L2 or elastic net
for logistic regression, and L1 and L2 penalties for XGBoost and LightGBM).
<i>Random search</i> tries a chosen number of random combinations. <i>Grid search</i>
tries every combination of a few values per chosen setting, centred on the defaults, and
shows the number of combinations and model fits before training. Both searches use the
same cross-validation on the training rows only.</p>
{_fig(img, "10_grid_search", 10, "Grid search settings for XGBoost.")}
<p>Before training, the application estimates the training time and warns about slow
choices such as stacking, large searches or SVM on large data. Training runs in the
background with a progress indicator and a <i>Cancel</i> button; cancelling keeps every
model that has already finished.</p>
{_fig(img, "11_estimate", 11, "Training-time estimate with its speed level.")}
{
        _fig(
            img,
            "12_training",
            12,
            "Training in progress, with elapsed time, remaining time and the Cancel button.",
        )
    }
"""


def _step5(img: str) -> str:
    return f"""
<h3>4.4 Results</h3>
<p>The best model is chosen by the cross-validated primary metric on the training rows.
The leaderboard lists the mean and spread of every metric for every model, together with
fit times, parameters and the imbalance handling actually applied, and can be downloaded
as CSV or as a JSON run report. Models whose cross-validation failed in any fold are
marked incomplete, with the reason, and are never chosen as best.</p>
{_fig(img, "13_leaderboard", 13, "Results: best model, cross-validation tiles and leaderboard.")}
<p>The <b>Random Forest dashboard</b> appears when Random Forest was trained. It shows key
figures for the forest, a fold-by-fold chart against the boosting models, a head-to-head
table counting the folds Random Forest wins, loses and ties against each rival on
identical folds, and the forest's built-in feature importance.</p>
{
        _fig(
            img,
            "14_rf_dashboard",
            14,
            "Random Forest dashboard: Random Forest compared with gradient boosting.",
        )
    }
<p><b>Model details</b> shows diagnostics from out-of-fold predictions, in which every
training row is predicted by a model that did not see it: a confusion matrix, ROC and
precision-recall curves, a decision-threshold tool and a calibration plot for
classification, and predicted-against-actual and residual plots for regression.</p>
{_fig(img, "15_diagnostics", 15, "Out-of-fold diagnostics for the selected model.")}
<p>The <b>final test</b> is locked until the user reveals it, so that model choices are not
influenced by the test set. Once revealed it shows the test scores, a comparison chart, a
sentence comparing Random Forest with the best booster and, on request, permutation
feature importance on the test set.</p>
{_fig(img, "16_final_test", 16, "Final test results after the reveal.")}
<h3>4.5 Using a model</h3>
<p>Step 6 lists the fully validated models. A single record can be scored through a form
prefilled with typical values; class probabilities are shown for classification. A batch
file is first checked for unparseable values, values outside the training range and
unseen categories, then scored, and the predictions can be downloaded. The trained model
can be downloaded as a <code>.joblib</code> or <code>.pkl</code> bundle containing the
complete pipeline, its settings and the decision threshold.</p>
{_fig(img, "17_predict", 17, "Single-record prediction.")}
"""


def _forecasting(img: str) -> str:
    return f"""
<h2>5. Forecasting</h2>
<p>Forecasting is treated as regression on features built from the series' own past.
For a horizon of <i>h</i> periods, every feature looks at least <i>h</i> periods back: lags
of the target, rolling means, calendar parts of the date and, optionally, other columns
delayed by the horizon. All inputs are therefore known when a forecast is made.
Validation is always time-ordered. The settings are the date column, an optional series
column (for several series in one file), the horizon, the number of lags, a seasonal lag
(suggested from the data frequency) and the rolling-mean windows. The application checks
for duplicate time points, series that are too short and irregular gaps.</p>
{_fig(img, "18_forecast_settings", 18, "Forecast settings and series checks.")}
<p>Two baselines are available: the naive forecast (the last known value) and the
seasonal naive forecast (the value one season earlier). Every model should be compared
with them. The <b>Hybrid RF + ARIMA</b> model fits an ARIMA model to the time pattern of the
series and a Random Forest to ARIMA's errors, using the lag, calendar and other features.
In the forecasting benchmark on the Colombo dengue data
(<code>scripts/forecast_benchmark.py</code>) it was the only model that beat the naive
baseline on the test weeks, by 2.8% at one week ahead and 6.8% at four weeks ahead.
Figure 19 shows a run with the application's default settings.</p>
{
        _fig(
            img,
            "19_forecast_test",
            19,
            "Test period: actual values, the best model and the naive baseline.",
        )
    }
<p>In step 6 the chosen model is refitted on all available history and forecasts the next
<i>h</i> periods for every series. A newer history file can be uploaded instead of the
training data.</p>
{_fig(img, "20_forecast_future", 20, "Forecast of the next period after the end of the history.")}
"""


def _features() -> str:
    rows = [
        [
            "Problem types",
            "Binary and multi-class classification, regression, several targets "
            "(multi-label and multi-output), time-series forecasting including several series",
        ],
        [
            "Data input",
            "CSV, TXT, Excel, JSON, Parquet; delimiter and encoding detection; "
            "number formats 1,234.5, 1.234,5, 1 234,5 and 1'234.5",
        ],
        [
            "Data understanding",
            "Quality report, per-column distributions, relationships with "
            "the target, correlation heatmap, feature relevance",
        ],
        [
            "Validation",
            "Random, grouped and time-ordered designs; test set held out first; "
            "out-of-fold diagnostics",
        ],
        [
            "Preprocessing",
            "Automatic recommendations with reasons; per-column control of type, "
            "imputation, encoding, scaling, transforms and outliers; duplicate and missing-row "
            "options; leakage review",
        ],
        [
            "Imbalance",
            "Class weights, random oversampling, SMOTE (SMOTE-NC for categories), undersampling",
        ],
        [
            "Models",
            "18 models plus voting, stacking, RF + ANN and RF + ARIMA hybrids, and "
            "naive forecasting baselines",
        ],
        ["Hyperparameters", "Default, manual, random search and grid search; presets"],
        ["Training", "Time estimate, row cap, background training with progress and cancel"],
        [
            "Results",
            "Leaderboard, comparison chart, Random Forest dashboard, diagnostics, "
            "decision threshold, calibration, locked test set, permutation importance",
        ],
        [
            "Deployment of models",
            "Single and batch prediction with input checks, forecasts, .joblib and .pkl downloads",
        ],
        [
            "Interface",
            "Card-based layout, light, dark and system themes, phone and tablet "
            "layouts, run history",
        ],
    ]
    return f"<h2>6. Feature reference</h2>{_table(['Area', 'Capability'], rows)}"


def _models() -> str:
    rows = [
        [
            "Linear and logistic regression, Ridge, Lasso, Elastic Net",
            "Linear",
            "C and penalty type (L1, L2, elastic net) for logistic regression; alpha and "
            "l1_ratio for Ridge, Lasso and Elastic Net",
        ],
        ["Decision tree", "Trees", "Depth, minimum leaf size, cost-complexity pruning"],
        [
            "Random Forest, Extra Trees, Bagging",
            "Trees and bagging",
            "Number of trees, depth, minimum leaf size, features per split, sample share",
        ],
        [
            "AdaBoost, Gradient Boosting, HistGB",
            "Boosting",
            "Rounds, learning rate, depth or leaves, subsampling, L2 regularisation (HistGB)",
        ],
        [
            "XGBoost, LightGBM, CatBoost",
            "Boosting",
            "Rounds, learning rate, depth or leaves, sampling, L1 and L2 penalties, minimum "
            "split gain (XGBoost), leaf regularisation (CatBoost)",
        ],
        [
            "SVM, K-nearest neighbours, neural network (MLP)",
            "Other",
            "C, kernel and gamma; number of neighbours and weighting; layers, L2 penalty, "
            "learning rate",
        ],
        ["Voting, Stacking", "Ensembles", "Chosen base models"],
        ["Hybrid RF + ANN", "Hybrid (regression)", "Number of trees"],
        [
            "Naive, seasonal naive, Hybrid RF + ARIMA",
            "Forecasting",
            "ARIMA order (p, d, q) and number of trees for the hybrid",
        ],
    ]
    return f"<h2>7. Models and hyperparameters</h2>{_table(['Models', 'Family', 'Main settings'], rows)}"


def _method() -> str:
    return """
<h2>8. Methodology and safeguards</h2>
<p><b>Holdout first.</b> The test rows are separated before any recommendation that uses
the target, so the test score is not influenced by any choice made in the application.</p>
<p><b>Preprocessing inside the pipeline.</b> Every learned step, including imputation,
outlier limits, skew correction, scaling, target encoding and resampling, is part of the
model pipeline and is refitted on the training part of every cross-validation fold. For
number columns the order is: parse, impute, clip outliers, correct skew, scale. For
category columns it is: impute, then encode. Resampling such as SMOTE is applied only when
fitting, never to validation, test or new rows.</p>
<p><b>Honest validation.</b> Grouped data never shares a group between training and
testing, and time-ordered data always trains on the past. A model that fails in any fold
is reported as incomplete and is not chosen as best.</p>
<p><b>Leakage control.</b> Suspicious columns are excluded until the user confirms that
their values are known at prediction time. Forecast features use only past values.</p>
<p><b>Safe prediction.</b> Unseen categories and missing columns never cause a failure,
and a log-scale target transform keeps predictions within a plausible range of the
training targets.</p>
"""


def _mobile(img: str) -> str:
    return f"""
<h2>9. Layout on phones and the dark theme</h2>
<p>The layout adapts to the screen: cards and columns stack on narrow screens and the
footer wraps. The theme is chosen from the menu in the top-right corner (System, Light or
Dark).</p>
{_fig(img, "21_phone_dark", 21, "Step 1 on a phone with the dark theme.", phone=True)}
"""


def _sources() -> str:
    rows = [
        [
            "Colombo Tea Auction (market catalogue and price)",
            "Hugging Face, colombo-tea-auction-prices/cta-price-prediction",
            "CC BY 4.0 (attribution required)",
        ],
        [
            "Colombo weekly dengue cases and weather",
            "R package colmozzie (Thiyanga Talagala), Epidemiology Unit weekly reports",
            "CC0",
        ],
        [
            "Sri Lankan telecom customer churn",
            "Kaggle, Real World Customer Churn Dataset in Telco Domain (fetched on demand)",
            "CC0",
        ],
    ]
    return f"""
<h2>10. Data sources and licences</h2>
<p>Only publicly available, genuine Sri Lankan datasets are bundled; no synthetic data is
used. Each sample keeps its source and licence, which are also shown in the application
footer.</p>
{_table(["Dataset", "Source", "Licence"], rows)}
"""


def _troubleshooting() -> str:
    rows = [
        [
            "The Next button is disabled in step 3",
            "Answer every column in the leakage review "
            "and make sure at least one feature is included.",
        ],
        [
            "A model is marked incomplete",
            "Read the reason shown in step 5. Lower the number of "
            "folds, set the target transform to none or add a tree model.",
        ],
        [
            "Training is slow",
            "Read the time estimate, deselect slow models, reduce the search "
            "size or folds, or set a row cap in the evaluation settings.",
        ],
        [
            "An upload is rejected",
            "The file must have a header row, at least two columns and "
            "at most one million cells (400,000 on the deployed app).",
        ],
        [
            "Forecasting reports duplicate time points",
            "Choose the series column, or aggregate the data to one row per period.",
        ],
        [
            "The deployed app shows a blank page",
            "The app's main file must be app.py; see docs/DEPLOYMENT.md.",
        ],
    ]
    return f"<h2>11. Troubleshooting</h2>{_table(['Situation', 'What to do'], rows)}"


def _limits() -> str:
    return """
<h2>12. Limitations</h2>
<ul>
<li>One table per file; multiple linked tables, images, audio and long free text are not
modelled.</li>
<li>Clustering and anomaly detection are not included.</li>
<li>The RF + ARIMA hybrid supports one series at a time with a user-set ARIMA order.</li>
<li>Class weights and resampling require a single target.</li>
<li>Model bundles are Python pickle files and should only be opened when they were created
by the user.</li>
</ul>
"""


def _appendix() -> str:
    rows = [
        ["app.py", "Streamlit entry point"],
        ["tabulaml/data", "Loading, samples, profiling, validation and leakage detection"],
        ["tabulaml/preprocessing", "Cleaner, transformers, recommender, pipeline builder"],
        ["tabulaml/models", "Model catalogue, ensembles, hybrid, hyperparameter definitions"],
        [
            "tabulaml/forecasting",
            "Forecast features, checks, baselines, RF + ARIMA, future forecasts",
        ],
        [
            "tabulaml/training",
            "Splits, pipelines, tuning, training, time estimate, background jobs",
        ],
        ["tabulaml/evaluation", "Metrics, importance, exploration, Random Forest insights, charts"],
        ["tabulaml/inference", "Predictor and model bundles"],
        ["tabulaml/ui", "Streamlit screens (no machine learning logic)"],
        ["tests", "Unit, integration and headless interface tests"],
        ["scripts", "Dataset fetcher, benchmarks, browser checks, screenshots, this guide"],
    ]
    return f"""
<h2>Appendix: project structure and testing</h2>
{_table(["Location", "Contents"], rows)}
<p>The automated test suite contains more than 340 tests with about 95% code coverage,
including a check that learned steps never see validation or test rows, a generalization
suite that trains and predicts on many different tables, and headless runs of the whole
interface. A browser test (<code>scripts/e2e_playwright.py</code>) walks through every
problem type on a desktop and a phone screen and fails on any error message or layout
overflow.</p>
"""


def body(images: str, assets: str, today: date) -> str:
    """The complete HTML body of the guide."""
    return "".join(
        [
            _cover(assets, today),
            _contents(),
            _intro(),
            _requirements(),
            _workflow(images),
            _step2(images),
            _step3(images),
            _step4(images),
            _step5(images),
            _forecasting(images),
            _features(),
            _models(),
            _method(),
            _mobile(images),
            _sources(),
            _troubleshooting(),
            _limits(),
            _appendix(),
        ]
    )
