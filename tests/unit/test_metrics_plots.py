import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from tabulaml.core.types import ProblemType
from tabulaml.evaluation import plots
from tabulaml.evaluation.metrics import (
    compute_metrics,
    default_primary_metric,
    display_value,
    minority_label,
    scoring_map,
)

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION


def test_binary_metrics_use_minority_as_positive():
    y_train = np.array([0, 0, 0, 1])
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    m = compute_metrics(CLS, y_true, y_pred, y_train=y_train)
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5
    y_train_flip = np.array([1, 1, 1, 0])  # now class 0 is the minority / positive class
    m0 = compute_metrics(CLS, y_true, y_pred, y_train=y_train_flip)
    assert m0["recall"] == 0.5 and m0["f1"] == 0.5


def test_probability_metrics():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    m = compute_metrics(CLS, y, proba.argmax(1), proba)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["log_loss"] > 0


def test_multiclass_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.eye(3)[y] * 0.8 + 0.2 / 3
    m = compute_metrics(CLS, y, y, proba)
    assert m["f1_macro"] == 1.0 and m["roc_auc"] == 1.0 and "f1" not in m


def test_regression_metrics_and_mape_guard():
    m = compute_metrics(REG, [1.0, 2.0, 4.0], [1.0, 3.0, 4.0])
    assert m["mae"] == pytest.approx(1 / 3) and m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert "mape" in m
    assert "mape" not in compute_metrics(REG, [0.0, 2.0], [1.0, 2.0])


def test_defaults_and_helpers():
    assert default_primary_metric(REG, [1, 2]) == "rmse"
    assert default_primary_metric(CLS, [0] * 9 + [1]) == "f1"
    assert default_primary_metric(CLS, [0, 1] * 5) == "roc_auc"
    assert default_primary_metric(CLS, [0, 1, 2]) == "f1_macro"
    assert minority_label([1, 1, 0]) == 0
    assert display_value("rmse", -2.0) == 2.0 and display_value("r2", 0.5) == 0.5
    assert "pr_auc" in scoring_map(CLS, [0, 1, 1]) and "roc_auc" in scoring_map(CLS, [0, 1, 2])
    assert "mape" not in scoring_map(REG, [0.0, 1.0])


def test_every_plot_builder_returns_a_figure():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    proba = np.column_stack([1 - rng.random(60), rng.random(60)])
    proba = proba / proba.sum(1, keepdims=True)
    board = pd.DataFrame({"model": ["A", "B"], "status": ["ok", "ok"], "test_f1": [0.5, 0.7]})
    imp = pd.DataFrame(
        {"feature": ["a", "b"], "importance_mean": [0.1, 0.2], "importance_std": [0.01, 0.02]}
    )
    y3 = rng.integers(0, 3, 60)
    p3 = rng.dirichlet([1, 1, 1], 60)
    figs = [
        plots.leaderboard_bar(board, "f1", "F1", False),
        plots.cv_boxplot({"A": [0.1, 0.2], "B": [0.3, 0.4]}, "F1"),
        plots.confusion_matrix_fig(y, proba.argmax(1), ["no", "yes"]),
        plots.roc_fig(y, proba, ["no", "yes"]),
        plots.roc_fig(y3, p3, ["a", "b", "c"]),
        plots.pr_fig(y, proba, ["no", "yes"]),
        plots.pred_vs_actual_fig(rng.random(20), rng.random(20)),
        plots.residuals_fig(rng.random(20), rng.random(20)),
        plots.importance_fig(imp, "F1"),
        plots.class_distribution_fig(pd.Series(y)),
        plots.target_histogram_fig(pd.Series(rng.random(20), name="y")),
        plots.missingness_fig(pd.Series({"a": 0.1, "b": 0.0})),
        plots.probability_fig({"no": 0.3, "yes": 0.7}),
    ]
    assert all(isinstance(f, go.Figure) and len(f.data) >= 1 for f in figs)
