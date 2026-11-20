"""End-to-end training on the bundled public Sri Lankan samples."""

import pandas as pd
import pytest

from tabulaml.core.types import ImbalanceStrategy, ProblemType
from tabulaml.data.profiler import profile_dataset
from tabulaml.data.samples import load_sample
from tabulaml.data.validation import detect_leakage
from tabulaml.evaluation.importance import permutation_importance_df
from tabulaml.preprocessing.cleaning import clean_frame
from tabulaml.preprocessing.recommender import recommend_config
from tabulaml.training.trainer import prepare_data, train_models

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION


pytestmark = pytest.mark.usefixtures("fast_models")


def _auto_config(key, **overrides):
    df, card = load_sample(key)
    clean = clean_frame(df)
    leaks = detect_leakage(clean, card.target, card.problem_type)
    cfg = recommend_config(
        profile_dataset(clean), card.target, card.problem_type, leaks, y=df[card.target]
    )
    return df, cfg.__class__(**{**cfg.__dict__, "cv_folds": 3, **overrides})


def test_multiclass_run_on_tea_catalogue():
    df, cfg = _auto_config("lk_tea_auction_catalogue")
    run = train_models(df, cfg, ["logreg", "rf", "xgb", "lgbm", "catboost"])
    board = run.leaderboard()
    assert (board["status"] == "ok").all(), board.get("error")
    assert run.primary_metric == "f1_macro"
    assert run.result_for("rf").test_metrics["accuracy"] > 0.75
    primary = board[f"cv_{run.primary_metric}_mean"].tolist()
    assert primary == sorted(primary, reverse=True)
    assert set(run.class_labels) == {"dust", "high_grown", "low_grown", "off_grade"}
    assert "category" not in run.feature_columns  # structural leak excluded automatically


def test_regression_with_log_target_on_tea_price():
    df, cfg = _auto_config("lk_tea_auction_price", target_transform="log1p")
    run = train_models(df, cfg, ["ridge", "rf"])
    assert all(r.status == "ok" for r in run.results), [r.error for r in run.results]
    assert run.result_for("rf").test_metrics["r2"] > 0.8
    assert any("missing target" in n for n in run.notes)  # 3.6% of prices are blank


def test_real_dengue_data_trains():
    df, cfg = _auto_config("lk_colombo_dengue")
    run = train_models(
        df,
        cfg,
        ["rf", "lgbm", "stacking", "hybrid_rf_ann"],
        ensemble_bases={"stacking": ["rf", "ridge"]},
    )
    assert all(r.status == "ok" for r in run.results), [r.error for r in run.results]
    assert run.result_for("rf").test_metrics["r2"] > 0.3


def test_failing_model_does_not_abort_run(monkeypatch):
    df, cfg = _auto_config("lk_tea_auction_catalogue")
    import tabulaml.training.trainer as trainer

    real = trainer.build_model_pipeline

    def broken(key, *a, **k):
        if key == "dt":
            raise RuntimeError("boom")
        return real(key, *a, **k)

    monkeypatch.setattr(trainer, "build_model_pipeline", broken)
    seen = []
    run = train_models(
        df, cfg, ["dt", "logreg"], progress=lambda label, i, n: seen.append((label, i, n))
    )
    status = {r.key: r.status for r in run.results}
    assert status == {"dt": "failed", "logreg": "ok"}
    assert "boom" in run.result_for("dt").error
    total = 2 * (cfg.cv_folds + 1)
    assert all(n == total for _, _, n in seen)
    assert [i for _, i, _ in seen] == sorted(i for _, i, _ in seen)  # never goes backwards
    assert seen[-1] == ("done", total, total)
    assert any(label.startswith("Logistic Regression: fold") for label, _, _ in seen)


def test_smote_and_voting_run():
    df, cfg = _auto_config("lk_tea_auction_catalogue", imbalance=ImbalanceStrategy.SMOTE)
    run = train_models(df, cfg, ["logreg", "voting"], ensemble_bases={"voting": ["logreg", "rf"]})
    assert all(r.status == "ok" for r in run.results), [r.error for r in run.results]


def test_permutation_importance_on_raw_columns():
    df, cfg = _auto_config("lk_tea_auction_catalogue")
    run = train_models(df, cfg, ["rf"])
    r = run.result_for("rf")
    imp = permutation_importance_df(r.pipeline, run.X_test, run.y_test, CLS, "roc_auc", n_repeats=2)
    assert set(imp["feature"]) == set(run.feature_columns)
    assert imp["importance_mean"].iloc[0] > 0


def test_prepare_data_errors_and_encoding():
    df = pd.DataFrame({"x": range(40), "y": ["b", "a"] * 20})
    _, cfg = _auto_config("lk_tea_auction_catalogue")
    with pytest.raises(ValueError, match="not found"):
        prepare_data(df, cfg)
    with pytest.raises(ValueError, match="at least one model"):
        train_models(df, cfg, [])
