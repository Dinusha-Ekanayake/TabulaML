import pytest

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ModelProfile,
    PreprocessingConfig,
    ProblemType,
)
from tabulaml.preprocessing.builder import describe_pipeline
from tabulaml.ui.imputation import (
    imputable_columns,
    impute_label,
    impute_options,
    with_imputation,
)


def _cfg():
    cols = (
        ColumnConfig("num", FeatureType.NUMERIC),
        ColumnConfig("cat", FeatureType.CATEGORICAL),
        ColumnConfig("flag", FeatureType.BOOLEAN),
        ColumnConfig("dropped", FeatureType.NUMERIC, include=False),
    )
    return PreprocessingConfig(ProblemType.CLASSIFICATION, "y", "auto", cols)


def test_options_depend_on_column_type():
    assert "knn" in impute_options(FeatureType.NUMERIC)
    assert impute_options(FeatureType.CATEGORICAL) == ("auto", "most_frequent", "constant")
    assert impute_options(FeatureType.BOOLEAN) == ()


def test_labels_explain_auto_per_type():
    assert "median" in impute_label("auto", FeatureType.NUMERIC).lower()
    assert "missing" in impute_label("auto", FeatureType.CATEGORICAL).lower()
    assert "missing" in impute_label("constant", FeatureType.CATEGORICAL).lower()
    assert "0" in impute_label("constant", FeatureType.NUMERIC)


def test_only_included_numeric_and_categorical_are_imputable():
    assert [c.name for c in imputable_columns(_cfg())] == ["num", "cat"]


def test_with_imputation_returns_new_custom_config_without_mutating():
    cfg = _cfg()
    new = with_imputation(cfg, {"num": "knn", "cat": "most_frequent"})
    assert cfg.mode == "auto" and cfg.columns[0].impute == "auto"
    assert new.mode == "custom"
    assert new.columns[0].impute == "knn" and new.columns[1].impute == "most_frequent"


def test_with_imputation_unchanged_returns_same_config():
    cfg = _cfg()
    assert with_imputation(cfg, {"num": "auto"}) is cfg


@pytest.mark.parametrize(
    "methods", [{"cat": "median"}, {"flag": "mean"}, {"nope": "median"}, {"num": "bogus"}]
)
def test_with_imputation_rejects_invalid_choices(methods):
    with pytest.raises(ValueError):
        with_imputation(_cfg(), methods)


def test_chosen_method_reaches_the_pipeline():
    new = with_imputation(_cfg(), {"num": "mean", "cat": "most_frequent"})
    steps = str(describe_pipeline(new, ModelProfile.TREE))
    assert "impute:mean" in steps and "impute:most_frequent" in steps
