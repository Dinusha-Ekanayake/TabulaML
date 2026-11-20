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


def test_exclude_mostly_missing_keeps_targets_and_returns_names():
    from tabulaml.ui.imputation import exclude_mostly_missing

    cfg = _cfg()
    new, names = exclude_mostly_missing(cfg, {"num": 0.8, "cat": 0.1, "y": 0.9}, 0.5)
    assert names == ["num"] and new.mode == "custom"
    assert not next(c for c in new.columns if c.name == "num").include
    assert next(c for c in cfg.columns if c.name == "num").include  # input untouched
    same, none = exclude_mostly_missing(cfg, {"num": 0.1}, 0.5)
    assert same is cfg and none == []


def test_row_options_duplicates_and_missing_rows():
    from dataclasses import replace

    import numpy as np
    import pandas as pd

    from tabulaml.training.prepare import incomplete_rows, prepare

    df = pd.DataFrame(
        {"num": [1.0, 1.0, np.nan, 4.0, 5.0, "?"], "cat": list("aabcde"), "y": [0, 0, 1, 1, 0, 1]}
    )
    base = replace(_cfg(), target="y", columns=_cfg().columns[:2])
    assert len(prepare(df, base).X) == 5  # one exact duplicate dropped by default
    kept = prepare(df, replace(base, drop_duplicates=False))
    assert len(kept.X) == 6 and any("kept" in n for n in kept.notes)
    assert incomplete_rows(df, base).tolist() == [False, False, True, False, False, True]
    dropped = prepare(df, replace(base, missing_rows="drop"))
    assert len(dropped.X) == 3 and any("missing feature value" in n for n in dropped.notes)


def test_dropping_every_row_is_refused():
    from dataclasses import replace

    import numpy as np
    import pandas as pd
    import pytest

    from tabulaml.training.prepare import prepare

    df = pd.DataFrame({"num": [np.nan] * 4, "cat": list("abcd"), "y": [0, 1, 0, 1]})
    cfg = replace(_cfg(), target="y", columns=_cfg().columns[:2], missing_rows="drop")
    with pytest.raises(ValueError, match="nothing"):
        prepare(df, cfg)
