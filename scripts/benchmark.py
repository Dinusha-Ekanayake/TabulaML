"""Research benchmark: Random Forest vs gradient boosting (XGBoost, LightGBM, CatBoost).

Every model is evaluated with the SAME repeated cross-validation folds through the SAME
Automatic-mode TabulaML pipeline, so fold scores are paired and can be compared with a
Wilcoxon signed-rank test. An ablation on the synthetic churn sample shows what the
preprocessing engine and the leakage detector contribute.

Usage:
    python scripts/benchmark.py --quick     # ~3 minutes: fewer datasets, 3 folds x 1 repeat
    python scripts/benchmark.py             # full: 5 folds x 3 repeats
Outputs (reports/): benchmark_results.csv, benchmark_summary.md, figures/*.png
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import get_scorer, roc_auc_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    RepeatedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
)

from tabulaml.core.types import ProblemType  # noqa: E402
from tabulaml.data.profiler import profile_dataset  # noqa: E402
from tabulaml.data.samples import list_samples, load_sample  # noqa: E402
from tabulaml.data.validation import detect_leakage  # noqa: E402
from tabulaml.evaluation.metrics import display_value, scoring_map  # noqa: E402
from tabulaml.preprocessing.cleaning import clean_frame  # noqa: E402
from tabulaml.preprocessing.recommender import recommend_config  # noqa: E402
from tabulaml.training.pipeline_factory import build_model_pipeline  # noqa: E402
from tabulaml.training.trainer import prepare_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
BOOSTERS = ("xgb", "lgbm", "catboost")
PRIMARY = {CLS: "roc_auc", REG: "rmse"}
SECONDARY = {CLS: "f1", REG: "r2"}
NAMES = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "gb": "Gradient Boosting",
    "logreg": "Logistic Regression",
    "ridge": "Ridge",
}
SEED = 42


def _datasets(quick: bool) -> list[tuple[str, pd.DataFrame, str, ProblemType, str]]:
    out = []
    for card in list_samples():
        df, card = load_sample(card.key)
        origin = "synthetic (SL context)" if card.is_synthetic else "real, public (Sri Lanka)"
        out.append((card.key, df, card.target, card.problem_type, origin))
    if not quick:
        for name, loader, ptype in (
            ("breast_cancer", load_breast_cancer, CLS),
            ("wine", load_wine, CLS),
            ("diabetes", load_diabetes, REG),
        ):
            frame = loader(as_frame=True).frame.rename(columns={"target": "y"})
            out.append((name, frame, "y", ptype, "public benchmark (sklearn)"))
    return out


def _config(df: pd.DataFrame, target: str, ptype: ProblemType, folds: int):
    clean = clean_frame(df)
    leaks = detect_leakage(clean, target, ptype)
    cfg = recommend_config(profile_dataset(clean), target, ptype, leaks, y=df[target])
    return replace(cfg, cv_folds=folds, random_state=SEED)


def _metric_names(ptype: ProblemType, y: np.ndarray) -> tuple[str, str]:
    if ptype == CLS and len(np.unique(y)) > 2:
        return "roc_auc", "f1_macro"
    return PRIMARY[ptype], SECONDARY[ptype]


def _evaluate(name, df, target, ptype, origin, folds, repeats) -> list[dict]:
    cfg = _config(df, target, ptype, folds)
    X, y, _, _ = prepare_data(df, cfg)
    scorers = {
        k: get_scorer(v) if isinstance(v, str) else v for k, v in scoring_map(ptype, y).items()
    }
    primary, secondary = _metric_names(ptype, y)
    splitter = (RepeatedStratifiedKFold if ptype == CLS else RepeatedKFold)(
        n_splits=folds, n_repeats=repeats, random_state=SEED
    )
    linear = "logreg" if ptype == CLS else "ridge"
    rows = []
    for key in ("rf", *BOOSTERS, "gb", linear):
        template = build_model_pipeline(key, cfg, y=y)
        for fold, (tr, te) in enumerate(splitter.split(X, y)):
            model = clone(template)
            start = time.perf_counter()
            model.fit(X.iloc[tr], y[tr])
            fit_s = time.perf_counter() - start
            for metric in (primary, secondary):
                raw = scorers[metric](model, X.iloc[te], y[te])
                rows.append(
                    {
                        "dataset": name,
                        "origin": origin,
                        "task": ptype.value,
                        "model": NAMES[key],
                        "key": key,
                        "fold": fold,
                        "metric": metric,
                        "value": float(display_value(metric, raw)),
                        "fit_s": fit_s,
                    }
                )
        print(f"  {name}: {NAMES[key]} done")
    return rows


def _wilcoxon(results: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: RF vs the best booster on paired fold scores of the primary metric."""
    out = []
    for dataset, part in results.groupby("dataset"):
        metric = part["metric"].iloc[0]
        p = part[part["metric"] == metric]
        lower = metric in ("rmse", "mae")
        means = p.groupby("key")["value"].mean()
        boosters = means[[k for k in BOOSTERS if k in means]]
        best = boosters.idxmin() if lower else boosters.idxmax()
        rf = p[p["key"] == "rf"].sort_values("fold")["value"].to_numpy()
        bo = p[p["key"] == best].sort_values("fold")["value"].to_numpy()
        diff = rf - bo
        pvalue = float(wilcoxon(rf, bo).pvalue) if np.any(diff != 0) and len(diff) >= 5 else np.nan
        rf_better = rf.mean() < bo.mean() if lower else rf.mean() > bo.mean()
        out.append(
            {
                "dataset": dataset,
                "metric": metric,
                "rf_mean": rf.mean(),
                "best_booster": NAMES[best],
                "booster_mean": bo.mean(),
                "winner": "tie"
                if np.isclose(rf.mean(), bo.mean(), rtol=0, atol=1e-6)
                else ("Random Forest" if rf_better else NAMES[best]),
                "wilcoxon_p": pvalue,
                "significant_5pct": bool(pvalue < 0.05) if not np.isnan(pvalue) else False,
            }
        )
    return pd.DataFrame(out)


def _ablation(folds: int) -> pd.DataFrame:
    """Synthetic churn: automatic pipeline vs leak kept vs a naive pipeline."""
    df, card = load_sample("lk_synthetic_telecom_churn")
    cfg = _config(df, card.target, card.problem_type, folds)
    X, y, _, _ = prepare_data(df, cfg)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    auto = build_model_pipeline("rf", cfg, y=y_tr).fit(X_tr, y_tr)
    rows = [
        (
            "TabulaML automatic pipeline (leak excluded)",
            roc_auc_score(y_te, auto.predict_proba(X_te)[:, 1]),
        )
    ]
    leaky_cfg = cfg.with_column(
        replace(next(c for c in cfg.columns if c.name == "disconnection_date"), include=True)
    )
    Xl, yl, _, _ = prepare_data(df, leaky_cfg)
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
        Xl, yl, test_size=0.2, random_state=SEED, stratify=yl
    )
    leaky = build_model_pipeline("rf", leaky_cfg, y=yl_tr).fit(Xl_tr, yl_tr)
    rows.append(
        (
            "Same pipeline but leakage column kept (inflated, invalid)",
            roc_auc_score(yl_te, leaky.predict_proba(Xl_te)[:, 1]),
        )
    )
    naive = df.drop(columns=["customer_id", "disconnection_date"]).dropna()
    yn = (naive.pop(card.target) == "Yes").astype(int)
    Xn = pd.get_dummies(naive.astype(str))  # no type detection: numbers become categories
    Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
        Xn, yn, test_size=0.2, random_state=SEED, stratify=yn
    )
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(Xn_tr, yn_tr)
    rows.append(
        (
            f"Naive: drop NaN rows ({len(df) - len(naive)} lost), one-hot everything as text",
            roc_auc_score(yn_te, rf.predict_proba(Xn_te)[:, 1]),
        )
    )
    return pd.DataFrame(rows, columns=["setup", "test_roc_auc"])


def _figure(summary: pd.DataFrame, path: Path) -> None:
    """Dot + std whiskers on a zoomed axis (bars from zero would hide 0.99 vs 0.995).
    Best model on top in every panel; Random Forest highlighted."""
    datasets = summary["dataset"].unique()
    fig, axes = plt.subplots(2, (len(datasets) + 1) // 2, figsize=(13, 6.4), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        part = summary[summary["dataset"] == dataset]
        metric = part["metric"].iloc[0]
        lower = metric in ("rmse", "mae")
        part = part.sort_values("mean", ascending=not lower)  # worst first = bottom, best on top
        for i, row in enumerate(part.itertuples()):
            is_rf = row.model == "Random Forest"
            ax.errorbar(
                row.mean,
                i,
                xerr=row.std,
                fmt="o",
                markersize=7 if is_rf else 6,
                color="#184f95" if is_rf else "#2a78d6",
                ecolor="#8c8b86",
                elinewidth=1.5,
                capsize=3,
            )
        ax.set_yticks(range(len(part)), part["model"])
        for label in ax.get_yticklabels():
            label.set_fontweight("bold" if label.get_text() == "Random Forest" else "normal")
        arrow = "lower is better" if lower else "higher is better"
        ax.set_title(f"{dataset}\n{metric} ({arrow})", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(datasets) :]:
        ax.set_visible(False)
    fig.suptitle("Mean +/- std over 15 paired CV folds (Random Forest in bold)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    def cell(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _markdown(summary, tests, ablation, folds, repeats) -> str:
    lines = [
        "# Benchmark: Random Forest vs gradient boosting",
        "",
        f"Repeated cross-validation ({folds} folds x {repeats} repeats, seed {SEED}); identical "
        "folds for every model; TabulaML Automatic-mode preprocessing for every dataset. "
        "Error metrics (RMSE) are lower-is-better; ROC-AUC, F1 and R2 are higher-is-better.",
        "",
    ]
    for dataset, part in summary.groupby("dataset", sort=False):
        lines += [
            f"## {dataset} ({part['origin'].iloc[0]})",
            "",
            "| Model | Metric | Mean | Std | Fit time (s) |",
            "|---|---|---|---|---|",
        ]
        for _, r in part.iterrows():
            lines.append(
                f"| {r['model']} | {r['metric']} | {r['mean']:.4f} | {r['std']:.4f} | {r['fit_s']:.2f} |"
            )
        lines.append("")
    lines += [
        "## Random Forest vs best gradient booster (Wilcoxon signed-rank on paired folds)",
        "",
        _md_table(tests),
        "",
        "## Ablation: what preprocessing and leakage control contribute (synthetic churn, RF, test ROC-AUC)",
        "",
        _md_table(ablation),
        "",
    ]
    return "\n".join(lines)


def _write_outputs(results: pd.DataFrame, folds: int, repeats: int) -> None:
    primary = results.groupby("dataset")["metric"].transform("first")
    summary = (
        results[results["metric"] == primary]
        .groupby(["dataset", "origin", "model", "metric"], sort=False)
        .agg(mean=("value", "mean"), std=("value", "std"), fit_s=("fit_s", "mean"))
        .reset_index()
    )
    tests = _wilcoxon(results[results["metric"] == primary])
    ablation = _ablation(folds)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    _figure(summary, REPORTS / "figures" / "benchmark_primary_metric.png")
    (REPORTS / "benchmark_summary.md").write_text(
        _markdown(summary, tests, ablation, folds, repeats), encoding="utf-8"
    )
    print(tests.to_string(index=False))
    print(ablation.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="rebuild summary and figure from reports/benchmark_results.csv",
    )
    args = parser.parse_args()
    folds, repeats = (3, 1) if args.quick else (5, 3)
    warnings.filterwarnings("ignore")
    if args.replot:
        _write_outputs(pd.read_csv(REPORTS / "benchmark_results.csv"), folds, repeats)
        return
    rows = []
    for name, df, target, ptype, origin in _datasets(args.quick):
        print(f"Benchmarking {name} ...")
        rows += _evaluate(name, df, target, ptype, origin, folds, repeats)
    results = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "benchmark_results.csv", index=False)
    _write_outputs(results, folds, repeats)


if __name__ == "__main__":
    main()
