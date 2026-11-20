"""Pure (Streamlit-free) helpers used by the UI, kept here so they are unit-testable."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from enum import Enum
from typing import Any

import pandas as pd

from tabulaml.core.types import ColumnConfig, FeatureType, PreprocessingConfig, TrainingRun
from tabulaml.evaluation.metrics import METRIC_LABELS

EDITOR_COLUMNS = (
    "column",
    "include",
    "type",
    "impute",
    "encoding",
    "scaling",
    "transform",
    "outliers",
    "reason",
)
OPTIONS = {
    "type": [t.value for t in FeatureType],
    "impute": ["auto", "median", "mean", "most_frequent", "constant", "knn"],
    "encoding": ["auto", "onehot", "ordinal", "target"],
    "scaling": ["auto", "standard", "robust", "minmax", "none"],
    "transform": ["auto", "yeo_johnson", "none"],
    "outliers": ["auto", "iqr_clip", "none"],
}
BOOSTERS = ("xgb", "lgbm", "catboost")


def config_to_editor(config: PreprocessingConfig) -> pd.DataFrame:
    rows = [
        {
            "column": c.name,
            "include": c.include,
            "type": c.feature_type.value,
            "impute": c.impute,
            "encoding": c.encoding,
            "scaling": c.scaling,
            "transform": c.transform,
            "outliers": c.outliers,
            "reason": c.reason,
        }
        for c in config.columns
    ]
    return pd.DataFrame(rows, columns=list(EDITOR_COLUMNS))


def editor_to_config(config: PreprocessingConfig, edited: pd.DataFrame) -> PreprocessingConfig:
    """Return a NEW config reflecting the per-column table edited by the user."""
    new = config
    for row in edited.to_dict("records"):
        new = new.with_column(
            ColumnConfig(
                name=row["column"],
                feature_type=FeatureType(row["type"]),
                include=bool(row["include"]),
                impute=row["impute"],
                encoding=row["encoding"],
                scaling=row["scaling"],
                transform=row["transform"],
                outliers=row["outliers"],
                reason=row.get("reason") or "",
            )
        )
    return replace(new, mode="custom") if new != config else new


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


METRIC_HELP = {
    "r2": "Share of the target's variation the model explains: 1 is perfect, 0 is no better "
    "than always predicting the average, and BELOW 0 means worse than the average.",
    "rmse": "Typical size of an error, in the target's units; large errors count extra. "
    "Lower is better.",
    "mae": "Average absolute error, in the target's units. Lower is better.",
    "mape": "Average error as a percentage of the true value. Lower is better.",
    "accuracy": "Share of rows predicted correctly.",
    "f1": "Balance of precision and recall for the positive class (0 to 1).",
    "f1_macro": "F1 averaged over classes, so small classes count as much as big ones.",
    "roc_auc": "How well the model ranks positives above negatives: 0.5 is guessing, 1 is perfect.",
    "subset_accuracy": "Share of rows where every target is right.",
}


def format_metric(value: float) -> str:
    """Readable metric values: 4 decimals normally, compact notation for huge numbers."""
    if value is None or not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1e6:
        return f"{value:.3g}"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4f}"


_FIXED_HEADERS = {
    "model": "Model",
    "status": "Status",
    "fit_s": "Fit time (s)",
    "predict_s": "Predict time (s)",
    "imbalance": "Imbalance handling",
    "params": "Parameters",
    "failed_folds": "Failed folds",
    "error": "Message",
}


def _header(column: str) -> str:
    if column in _FIXED_HEADERS:
        return _FIXED_HEADERS[column]
    for prefix, label in (("cv_", "CV"), ("test_", "Test")):
        if column.startswith(prefix):
            metric = column[len(prefix) :]
            suffix = ""
            for end, name in (("_mean", ""), ("_std", " (std)")):
                if metric.endswith(end):
                    metric, suffix = metric[: -len(end)], name
            return f"{label} {metric_label(metric)}{suffix}"
    return column


def display_board(board: pd.DataFrame) -> pd.DataFrame:
    """Leaderboard with readable headers and rounded numbers (display only)."""
    shown = board.rename(columns={c: _header(c) for c in board.columns})
    return shown.round(4)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def run_report(run: TrainingRun, dataset_name: str) -> dict[str, Any]:
    """JSON-serialisable summary of a run (config, decisions, metrics, versions)."""
    board = run.leaderboard()
    return {
        "dataset": dataset_name,
        "problem_type": run.problem_type.value,
        "primary_metric": run.primary_metric,
        "config": _plain(asdict(run.config)),
        "notes": list(run.notes),
        "leaderboard": json.loads(board.to_json(orient="records")),
        "versions": run.versions,
    }


def rf_vs_boosting(run: TrainingRun) -> str | None:
    """One-sentence comparison of Random Forest with the best gradient booster, if both ran."""
    ok = {r.key: r for r in run.results if r.status == "ok"}
    boosters = [ok[k] for k in BOOSTERS if k in ok]
    if "rf" not in ok or not boosters:
        return None
    metric = run.primary_metric
    lower = metric in {"rmse", "mae", "medae", "mape", "log_loss"}
    rf = ok["rf"].test_metrics.get(metric)
    best = sorted(
        boosters, key=lambda r: r.test_metrics.get(metric, float("nan")), reverse=not lower
    )[0]
    other = best.test_metrics.get(metric)
    if rf is None or other is None:
        return None
    rf_wins = rf < other if lower else rf > other
    winner, loser = ("Random Forest", best.name) if rf_wins else (best.name, "Random Forest")
    return (
        f"On the held-out test set, {winner} beat {loser} on {metric_label(metric)} "
        f"({rf:.4f} for Random Forest vs {other:.4f} for {best.name})."
    )
