"""Regression tests for defects found in the Phase 1-2 review."""

import io

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ModelResult,
    PreprocessingConfig,
    ProblemType,
    TrainingRun,
)
from tabulaml.data.loader import DataLoadError, read_tabular_file
from tabulaml.data.profiler import infer_feature_type
from tabulaml.data.validation import detect_leakage, validate_target
from tabulaml.preprocessing.cleaning import DataCleaner, clean_target

CLS = ProblemType.CLASSIFICATION


def test_ragged_csv_rows_raise_instead_of_shifting_columns():
    with pytest.raises(DataLoadError, match="more values"):
        read_tabular_file(io.BytesIO(b"a,b\n1,2,3\n4,5,6\n"), "d.csv")


def test_blank_header_cell_becomes_unnamed():
    out = read_tabular_file(io.BytesIO(b"a,,c\n1,2,3\n4,5,6\n"), "d.csv")
    assert list(out.columns) == ["a", "unnamed", "c"]


def test_numeric_column_accepts_text_numbers_at_predict_time():
    cleaner = DataCleaner().fit(pd.DataFrame({"n": [1.0, 2.0, 3.0]}))
    assert cleaner.transform(pd.DataFrame({"n": ["1,200"]}))["n"].iloc[0] == 1200.0


def test_euro_and_pound_prefixes_parse():
    df = pd.DataFrame({"p": ["€1,000", "£250", "Rs. 5"] * 10})
    assert DataCleaner().fit_transform(df)["p"].iloc[:3].tolist() == [1000.0, 250.0, 5.0]


def test_boolean_column_accepts_one_zero_at_predict_time():
    cleaner = DataCleaner().fit(pd.DataFrame({"b": ["yes", "no", "yes"]}))
    out = cleaner.transform(pd.DataFrame({"b": ["1", 0, "No"]}))["b"]
    assert out.tolist() == [1.0, 0.0, 0.0]


def test_timezone_aware_strings_become_naive_datetimes():
    df = pd.DataFrame({"t": ["2024-01-01T10:00:00+05:30", "2024-01-02T10:00:00+00:00"] * 5})
    out = DataCleaner().fit_transform(df)["t"]
    assert str(out.dtype) == "datetime64[ns]" and out.notna().all()


def test_sorted_unique_integer_feature_is_not_an_identifier():
    s = pd.Series(np.arange(1000, 1100) * 37, name="income")
    assert infer_feature_type(s, 100) == FeatureType.NUMERIC


def test_shuffled_row_counter_is_an_identifier():
    s = pd.Series(np.random.default_rng(0).permutation(np.arange(1, 101)), name="row")
    assert infer_feature_type(s, 100) == FeatureType.IDENTIFIER


def test_class_needs_enough_rows_for_split_plus_cv():
    df = pd.DataFrame({"y": ["a"] * 50 + ["b"] * 6, "x": range(56)})
    assert "class_too_small" in {i.code for i in validate_target(df, "y", CLS, 5)}


def test_target_case_variants_are_one_class():
    y = clean_target(pd.Series(["Yes", "yes ", "YES", "No", "?"]), regression=False)
    assert y.dropna().nunique() == 2 and y.isna().sum() == 1


def test_high_cardinality_singletons_are_not_flagged_as_leakage():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"name": [f"person {i}" for i in range(400)], "y": rng.choice(["a", "b"], 400)}
    )
    assert detect_leakage(df, "y", CLS) == []


def test_datetime_with_missing_values_in_leakage_scan():
    rng = np.random.default_rng(0)
    d = pd.Series(pd.date_range("2020-01-01", periods=300)).where(rng.random(300) > 0.3)
    df = pd.DataFrame({"d": d, "y": rng.choice(["a", "b"], 300)})
    assert detect_leakage(df, "y", CLS) == []


def test_leaderboard_when_every_model_failed():
    cfg = PreprocessingConfig(CLS, "y", "auto", (ColumnConfig("a", FeatureType.NUMERIC),))
    failed = ModelResult("rf", "RF", "failed", {}, {}, 0.0, 0.0, error="boom")
    run = TrainingRun(CLS, cfg, "f1", (failed,), None, None, None)
    assert run.leaderboard()["status"].tolist() == ["failed"]
    with pytest.raises(ValueError):
        run.best()
