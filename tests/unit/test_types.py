import dataclasses

import pytest

from tabulaml.core.types import (
    ColumnConfig,
    ColumnProfile,
    DatasetSchema,
    FeatureType,
    ModelResult,
    PreprocessingConfig,
    ProblemType,
    TrainingRun,
)


def _cfg() -> PreprocessingConfig:
    cols = (
        ColumnConfig("a", FeatureType.NUMERIC),
        ColumnConfig("b", FeatureType.CATEGORICAL, include=False),
    )
    return PreprocessingConfig(ProblemType.CLASSIFICATION, "y", "auto", cols)


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _cfg().target = "z"  # type: ignore[misc]


def test_with_column_returns_new_object_and_leaves_original():
    cfg = _cfg()
    new = cfg.with_column(ColumnConfig("b", FeatureType.CATEGORICAL, include=True))
    assert new is not cfg
    assert [c.include for c in cfg.columns] == [True, False]
    assert [c.include for c in new.columns] == [True, True]


def test_with_column_appends_unknown_column():
    new = _cfg().with_column(ColumnConfig("c", FeatureType.BOOLEAN))
    assert [c.name for c in new.columns] == ["a", "b", "c"]


def test_included_filters_excluded_columns():
    assert [c.name for c in _cfg().included()] == ["a"]


def test_schema_names_of_and_get():
    cp = ColumnProfile("a", "float64", FeatureType.NUMERIC, 0.0, 5, 0.5, ("1",))
    schema = DatasetSchema((cp,), 10, 0)
    assert schema.names_of(FeatureType.NUMERIC) == ["a"]
    assert schema.names_of(FeatureType.TEXT) == []
    assert schema.get("a") is cp
    with pytest.raises(KeyError):
        schema.get("zzz")


def _result(key: str, cv: float, status: str = "ok") -> ModelResult:
    return ModelResult(key, key.upper(), status, {"f1": (cv, cv)}, {"f1": cv}, 1.0, 0.1)


def test_leaderboard_sorted_best_first_and_failed_last():
    run = TrainingRun(
        ProblemType.CLASSIFICATION,
        _cfg(),
        "f1",
        (_result("a", 0.5), _result("b", 0.9), _result("c", 0.99, "failed")),
        ("0", "1"),
        None,
        None,
    )
    board = run.leaderboard()
    assert list(board["key"]) == ["b", "a", "c"]
    assert run.best().key == "b"


def test_leaderboard_lower_is_better_for_rmse():
    results = (
        ModelResult("a", "A", "ok", {"rmse": (3.0,)}, {"rmse": 3.0}, 1, 0.1),
        ModelResult("b", "B", "ok", {"rmse": (1.0,)}, {"rmse": 1.0}, 1, 0.1),
    )
    run = TrainingRun(ProblemType.REGRESSION, _cfg(), "rmse", results, None, None, None)
    assert run.best().key == "b"


def test_best_raises_when_all_failed():
    run = TrainingRun(
        ProblemType.CLASSIFICATION, _cfg(), "f1", (_result("a", 0.1, "failed"),), None, None, None
    )
    with pytest.raises(ValueError):
        run.best()
