import json

import numpy as np

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ModelResult,
    PreprocessingConfig,
    ProblemType,
    TrainingRun,
)
from tabulaml.ui.helpers import config_to_editor, editor_to_config, rf_vs_boosting, run_report
from tabulaml.ui.state import AppState, cleared_fields

CLS = ProblemType.CLASSIFICATION


def _cfg():
    cols = (
        ColumnConfig("a", FeatureType.NUMERIC, reason="r"),
        ColumnConfig("b", FeatureType.CATEGORICAL),
    )
    return PreprocessingConfig(CLS, "y", "auto", cols)


def test_editor_roundtrip_is_identity():
    cfg = _cfg()
    assert editor_to_config(cfg, config_to_editor(cfg)) == cfg


def test_editor_change_switches_to_custom_without_mutating():
    cfg = _cfg()
    table = config_to_editor(cfg)
    table.loc[1, "include"] = False
    table.loc[0, "scaling"] = "robust"
    new = editor_to_config(cfg, table)
    assert new.mode == "custom" and cfg.mode == "auto"
    assert not new.columns[1].include and new.columns[0].scaling == "robust"


def _run(rf, xgb):
    res = (
        ModelResult("rf", "Random Forest", "ok", {"f1": (rf,)}, {"f1": rf}, 1, 0.1),
        ModelResult("xgb", "XGBoost", "ok", {"f1": (xgb,)}, {"f1": xgb}, 1, 0.1),
    )
    return TrainingRun(CLS, _cfg(), "f1", res, ("No", "Yes"), None, np.array([0, 1]))


def test_rf_vs_boosting_sentence():
    assert "XGBoost beat Random Forest" in rf_vs_boosting(_run(0.5, 0.6))
    assert "Random Forest beat XGBoost" in rf_vs_boosting(_run(0.7, 0.6))


def test_run_report_is_json_serialisable():
    report = run_report(_run(0.5, 0.6), "demo")
    text = json.dumps(report, default=str)
    assert '"problem_type": "classification"' in text and report["leaderboard"]


def test_cleared_fields_cover_later_steps_only():
    cleared = cleared_fields(4)
    assert "run" in cleared and "config" not in cleared
    assert cleared["ensemble_bases"] == {} and AppState().step == 1


def test_display_board_headers():
    import pandas as pd

    from tabulaml.ui.helpers import display_board

    board = pd.DataFrame(
        {
            "model": ["RF"],
            "cv_f1_macro_mean": [0.81234],
            "cv_rmse_std": [1.0],
            "test_r2": [0.5],
            "fit_s": [1.23456],
        }
    )
    shown = display_board(board)
    assert list(shown.columns) == [
        "Model",
        "CV F1 (macro)",
        "CV RMSE (std)",
        "Test R²",
        "Fit time (s)",
    ]
    assert shown.iloc[0, 1] == 0.8123
