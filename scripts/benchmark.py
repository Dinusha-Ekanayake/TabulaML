"""Benchmark: Random Forest vs gradient boosting (XGBoost, LightGBM, CatBoost).

It uses EXACTLY the app's engine, so the benchmark follows the same rules as the UI:
- each dataset keeps its own validation design (tea: grouped by weekly sale; dengue: time
  order; sklearn sets: random);
- the automatic configuration is built from the training (dev) rows only;
- models are compared with cross-validation on the dev rows, then scored once on the
  held-out test rows.

The comparison is kept simple and readable: mean +/- standard deviation over folds, and
"RF better in k of n folds". When the folds disagree (k close to n/2) the models are
practically tied on that dataset.

Usage:
    python scripts/benchmark.py --quick   # sample datasets only, 3 folds
    python scripts/benchmark.py           # all datasets, 5 folds
Outputs (reports/): benchmark_results.csv, benchmark_summary.md, figures/*.png
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine  # noqa: E402

from tabulaml.core.types import LOWER_IS_BETTER, ProblemType, ValidationScheme  # noqa: E402
from tabulaml.data.samples import list_samples, load_sample  # noqa: E402
from tabulaml.training.trainer import train_models  # noqa: E402
from tabulaml.workflow import Roles, auto_configure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
BOOSTERS = ("xgb", "lgbm", "catboost")
SEED = 42


def _datasets(quick: bool) -> list[tuple[str, pd.DataFrame, Roles, str]]:
    out = []
    for card in list_samples():
        df, card = load_sample(card.key)
        roles = Roles(
            card.target,
            card.problem_type,
            ValidationScheme(card.validation),
            card.group_column,
            card.order_column,
        )
        out.append((card.key, df, roles, "real, public (Sri Lanka)"))
    if not quick:
        for name, loader, ptype in (
            ("breast_cancer", load_breast_cancer, CLS),
            ("wine", load_wine, CLS),
            ("diabetes", load_diabetes, REG),
        ):
            frame = loader(as_frame=True).frame.rename(columns={"target": "y"})
            out.append((name, frame, Roles("y", ptype), "public benchmark (sklearn)"))
    return out


def _rows(name: str, origin: str, run) -> list[dict]:  # noqa: ANN001 - TrainingRun
    rows = []
    for r in run.results:
        scores = r.cv_scores.get(run.primary_metric, ())
        for fold, value in enumerate(scores):
            rows.append(
                {
                    "dataset": name,
                    "origin": origin,
                    "model": r.name,
                    "key": r.key,
                    "status": r.status,
                    "metric": run.primary_metric,
                    "fold": fold,
                    "cv_value": value,
                    "test_value": r.test_metrics.get(run.primary_metric, np.nan),
                    "fit_s": r.fit_seconds,
                    "validation": run.config.validation.value,
                }
            )
    return rows


def _evaluate(name: str, df: pd.DataFrame, roles: Roles, origin: str, folds: int) -> list[dict]:
    roles = replace(roles, cv_folds=folds, random_state=SEED)
    config = auto_configure(df, roles).config  # dev-only recommendations, leaks excluded
    linear = "logreg" if roles.problem_type == CLS else "ridge"
    run = train_models(df, config, ["rf", *BOOSTERS, "gb", linear])
    print(f"  {name}: {', '.join(f'{r.key}={r.status}' for r in run.results)}")
    return _rows(name, origin, run)


def _comparisons(results: pd.DataFrame) -> pd.DataFrame:
    """RF vs each booster, fold by fold: wins, losses and ties (equal scores)."""
    out = []
    for dataset, part in results.groupby("dataset", sort=False):
        metric = part["metric"].iloc[0]
        lower = metric in LOWER_IS_BETTER
        rf = part[part["key"] == "rf"].sort_values("fold")["cv_value"].to_numpy()
        for booster in BOOSTERS:
            bo = part[part["key"] == booster].sort_values("fold")["cv_value"].to_numpy()
            if len(bo) != len(rf) or not len(rf):
                continue
            tie = np.isclose(rf, bo, rtol=0, atol=1e-9)
            rf_wins = int(((rf < bo) if lower else (rf > bo))[~tie].sum())
            bo_wins = int(((bo < rf) if lower else (bo > rf))[~tie].sum())
            n = len(rf)
            verdict = (
                "RF better"
                if rf_wins >= 0.8 * n
                else "booster better"
                if bo_wins >= 0.8 * n
                else "about the same"
            )
            out.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "booster": booster,
                    "rf_cv_mean": rf.mean(),
                    "booster_cv_mean": bo.mean(),
                    "folds_rf_won": rf_wins,
                    "folds_booster_won": bo_wins,
                    "folds_tied": int(tie.sum()),
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(out)


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["dataset", "origin", "validation", "model", "metric"], sort=False)
        .agg(
            cv_mean=("cv_value", "mean"),
            cv_std=("cv_value", "std"),
            test=("test_value", "first"),
            fit_s=("fit_s", "first"),
        )
        .reset_index()
    )


def _ablation(folds: int) -> pd.DataFrame:
    """Tea catalogue: automatic pipeline vs the leaky 'category' kept vs ignoring groups."""
    df, card = load_sample("lk_tea_auction_catalogue")
    roles = Roles(
        card.target,
        card.problem_type,
        ValidationScheme.GROUPED,
        "sale_id",
        cv_folds=folds,
        random_state=SEED,
    )
    auto = auto_configure(df, roles).config
    leak = next(c for c in auto.columns if c.name == "category")
    leaky = auto.with_column(replace(leak, include=True))
    random_cfg = auto_configure(
        df, replace(roles, validation=ValidationScheme.RANDOM, group_column=None)
    ).config
    rows = []
    for label, cfg in (
        ("TabulaML: grouped validation, leaky 'category' excluded", auto),
        ("Same, but 'category' kept (it encodes the answer)", leaky),
        ("Random split that ignores the weekly-sale groups", random_cfg),
    ):
        run = train_models(df, cfg, ["rf"])
        r = run.result_for("rf")
        rows.append(
            {
                "setup": label,
                "cv_f1_macro": float(np.nanmean(r.cv_scores["f1_macro"])),
                "test_f1_macro": r.test_metrics.get("f1_macro", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _figure(summary: pd.DataFrame, path: Path) -> None:
    """Dots with +/- std whiskers on a zoomed axis; best model on top; RF in bold."""
    datasets = summary["dataset"].unique()
    ncols = (len(datasets) + 1) // 2 if len(datasets) > 2 else len(datasets)
    nrows = 2 if len(datasets) > 2 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.2 * nrows), squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        part = summary[summary["dataset"] == dataset]
        metric = part["metric"].iloc[0]
        lower = metric in LOWER_IS_BETTER
        part = part.sort_values("cv_mean", ascending=not lower)  # worst at bottom, best on top
        for i, row in enumerate(part.itertuples()):
            rf = row.model == "Random Forest"
            ax.errorbar(
                row.cv_mean,
                i,
                xerr=row.cv_std,
                fmt="o",
                markersize=7 if rf else 6,
                color="#184f95" if rf else "#2a78d6",
                ecolor="#8c8b86",
                capsize=3,
            )
        ax.set_yticks(range(len(part)), part["model"])
        for label in ax.get_yticklabels():
            label.set_fontweight("bold" if label.get_text() == "Random Forest" else "normal")
        ax.set_title(
            f"{dataset}\n{metric} ({'lower' if lower else 'higher'} is better)", fontsize=9
        )
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(datasets) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Cross-validation mean +/- std (training rows only; Random Forest in bold)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    def cell(v):  # noqa: ANN001, ANN202
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _markdown(
    summary: pd.DataFrame, comparisons: pd.DataFrame, ablation: pd.DataFrame, folds: int
) -> str:
    lines = [
        "# Benchmark: Random Forest vs gradient boosting",
        "",
        f"Same engine as the app: dev-only automatic configuration, {folds}-fold "
        "cross-validation on the training rows with each dataset's validation design, "
        "then one score on the held-out test rows. RMSE: lower is better; ROC-AUC, F1, "
        "R2: higher is better.",
        "",
    ]
    for dataset, part in summary.groupby("dataset", sort=False):
        lines += [
            f"## {dataset} ({part['origin'].iloc[0]}, {part['validation'].iloc[0]} validation)",
            "",
            _md_table(part[["model", "metric", "cv_mean", "cv_std", "test", "fit_s"]]),
            "",
        ]
    lines += [
        "## Random Forest vs each booster (cross-validation folds)",
        "",
        "'RF better in k of n folds' counts the folds where Random Forest scored better. "
        "Verdict: RF/booster better when it wins at least 80% of folds, otherwise "
        "'about the same'.",
        "",
        _md_table(comparisons),
        "",
        "## Ablation on the tea catalogue task (Random Forest, macro-F1)",
        "",
        _md_table(ablation),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="rebuild tables/figure from reports/benchmark_results.csv",
    )
    args = parser.parse_args()
    folds = 3 if args.quick else 5
    warnings.filterwarnings("ignore")
    if args.replot:
        _write(pd.read_csv(REPORTS / "benchmark_results.csv"), folds)
        return
    rows = []
    for name, df, roles, origin in _datasets(args.quick):
        print(f"Benchmarking {name} ...")
        rows += _evaluate(name, df, roles, origin, folds)
    results = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "benchmark_results.csv", index=False)
    _write(results, folds)


def _write(results: pd.DataFrame, folds: int) -> None:
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    summary = _summary(results)
    comparisons = _comparisons(results)
    ablation = _ablation(folds)
    _figure(summary, REPORTS / "figures" / "benchmark_primary_metric.png")
    (REPORTS / "benchmark_summary.md").write_text(
        _markdown(summary, comparisons, ablation, folds), encoding="utf-8"
    )
    print(comparisons.to_string(index=False))
    print(ablation.to_string(index=False))


if __name__ == "__main__":
    main()
