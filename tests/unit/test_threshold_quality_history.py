"""P14 threshold/calibration/positive class, P17 prediction quality report, history."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tabulaml.core.types import ColumnConfig, FeatureType, PreprocessingConfig, ProblemType
from tabulaml.evaluation import plots
from tabulaml.evaluation.metrics import best_threshold, compute_metrics, threshold_curve
from tabulaml.inference.persistence import bundle_bytes, load_bundle
from tabulaml.inference.predictor import Predictor, make_input_spec
from tabulaml.training.trainer import train_models
from tabulaml.ui import history

CLS = ProblemType.CLASSIFICATION


def _binary_run(**kw):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300), "c": rng.choice(["a", "b", "c"], 300)})
    df["y"] = np.where(df["x"] + rng.normal(scale=0.7, size=300) > 0.8, "churn", "stay")
    cols = (ColumnConfig("x", FeatureType.NUMERIC), ColumnConfig("c", FeatureType.CATEGORICAL))
    cfg = PreprocessingConfig(CLS, "y", "auto", cols, cv_folds=3, **kw)
    return df, cfg, train_models(df, cfg, ["logreg"])


def test_positive_class_is_used_everywhere():
    _, _, run = _binary_run(positive_class="stay")
    assert run.class_labels[run.positive_index] == "stay"
    r = run.result_for("logreg")
    fig = plots.pr_fig(run.y_test, r.y_proba, list(run.class_labels), run.positive_index)
    assert fig.data[0].name == "stay"
    y = np.array([0, 0, 1, 1])
    m0 = compute_metrics(CLS, y, np.array([0, 1, 1, 1]), positive=0)
    m1 = compute_metrics(CLS, y, np.array([0, 1, 1, 1]), positive=1)
    assert m0["recall"] == 0.5 and m1["recall"] == 1.0


def test_threshold_tools_and_calibration_figure():
    y = np.array([0, 0, 0, 1, 1, 1] * 20)
    p = np.clip(y * 0.6 + np.linspace(0, 0.39, 120), 0, 1)
    curve = threshold_curve(y, p, 1)
    assert {"threshold", "precision", "recall", "f1"} <= set(curve.columns)
    assert 0.05 <= best_threshold(y, p, 1) <= 0.95
    assert isinstance(plots.calibration_fig(y, p, 1, "pos"), go.Figure)
    assert isinstance(plots.threshold_fig(curve, 0.5), go.Figure)


def test_predictor_threshold_changes_predictions_and_round_trips():
    df, cfg, run = _binary_run()
    r = run.result_for("logreg")
    spec = make_input_spec(df[list(run.feature_columns)], cfg)
    low = Predictor(
        r.pipeline, cfg, run.class_labels, spec, threshold=0.05, positive_index=run.positive_index
    ).predict_batch(df.head(50))
    high = Predictor(
        r.pipeline, cfg, run.class_labels, spec, threshold=0.95, positive_index=run.positive_index
    ).predict_batch(df.head(50))
    pos = run.class_labels[run.positive_index]
    assert (low["prediction"] == pos).sum() > (high["prediction"] == pos).sum()
    loaded = load_bundle(bundle_bytes(r, run, spec, threshold=0.05))
    assert loaded.threshold == 0.05 and loaded.positive_index == run.positive_index


def test_quality_report_flags_unparseable_range_and_new_categories():
    df, cfg, run = _binary_run()
    r = run.result_for("logreg")
    spec = make_input_spec(df[list(run.feature_columns)], cfg)
    predictor = Predictor(r.pipeline, cfg, run.class_labels, spec)
    batch = pd.DataFrame({"x": ["abc", "999", "0.1"], "c": ["a", "zzz", "b"]})
    issues = set(predictor.quality_report(batch)["issue"])
    assert "could not be parsed" in issues
    assert any(i.startswith("outside training range") for i in issues)
    assert "categories not seen in training" in issues
    assert predictor.quality_report(df.head(20).drop(columns="y")).empty


def test_history_saves_and_lists_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("TABULAML_ARTIFACTS", str(tmp_path))
    _, _, run = _binary_run()
    assert history.save_run(run, "demo") is not None
    runs = history.list_runs()
    assert runs and runs[0]["dataset"] == "demo" and runs[0]["leaderboard"]
