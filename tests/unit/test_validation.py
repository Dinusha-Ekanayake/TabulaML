import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import ProblemType
from tabulaml.data.profiler import profile_dataset
from tabulaml.data.samples import load_sample
from tabulaml.data.validation import (
    check_supported,
    detect_leakage,
    suggest_problem_type,
    validate_target,
)
from tabulaml.preprocessing.cleaning import clean_frame

CLS = ProblemType.CLASSIFICATION
REG = ProblemType.REGRESSION


def _codes(issues):
    return {i.code for i in issues}


def test_regression_on_string_target_is_error():
    df = pd.DataFrame({"y": ["a", "b", "c"] * 10, "x": range(30)})
    assert "target_not_numeric" in _codes(validate_target(df, "y", REG, 5))


def test_classification_small_class_is_error():
    df = pd.DataFrame({"y": ["a"] * 30 + ["b"] * 2, "x": range(32)})
    assert "class_too_small" in _codes(validate_target(df, "y", CLS, 5))


def test_single_class_is_error():
    df = pd.DataFrame({"y": ["a"] * 30, "x": range(30)})
    assert "single_class" in _codes(validate_target(df, "y", CLS, 5))


def test_missing_target_rows_reported_and_column_missing_error():
    df = pd.DataFrame({"y": ["a", "b", None, "a", "b"] * 6, "x": range(30)})
    assert "target_missing_rows" in _codes(validate_target(df, "y", CLS, 5))
    assert "target_missing" in _codes(validate_target(df, "zzz", CLS, 5))


def test_valid_targets_have_no_errors(messy_df, regression_df):
    assert not [i for i in validate_target(messy_df, "target", CLS, 5) if i.level == "error"]
    assert not [i for i in validate_target(regression_df, "y", REG, 5) if i.level == "error"]


def test_many_classes_warns():
    df = pd.DataFrame({"y": [f"c{i % 60}" for i in range(600)], "x": range(600)})
    assert "many_classes" in _codes(validate_target(df, "y", CLS, 5))


def test_regression_with_few_values_warns():
    df = pd.DataFrame({"y": [1, 2, 3] * 20, "x": range(60)})
    assert "target_looks_categorical" in _codes(validate_target(df, "y", REG, 5))


def test_suggest_problem_type(regression_df, messy_df):
    assert suggest_problem_type(regression_df["y"]) == REG
    assert suggest_problem_type(messy_df["target"]) == CLS
    assert suggest_problem_type(pd.Series(["a", "b"] * 20)) == CLS


@pytest.mark.parametrize(
    ("key", "leaks", "safe"),
    [
        # each tea 'category' is listed in exactly one catalogue: a genuine structural leak
        ("lk_tea_auction_catalogue", {"category"}, ["price_mid_lkr", "fx_usd", "region", "tier"]),
        ("lk_tea_auction_price", set(), ["catalogue", "grade", "fx_usd"]),
        ("lk_colombo_dengue", set(), ["cases_last_week", "rainfall_mm", "week"]),
    ],
)
def test_leakage_on_public_samples(key, leaks, safe):
    df, card = load_sample(key)
    flagged = {i.column for i in detect_leakage(clean_frame(df), card.target, card.problem_type)}
    assert leaks <= flagged
    assert not flagged & set(safe)


def test_post_outcome_missingness_leak_is_flagged():
    """A column filled only after the outcome (e.g. a disconnection date) leaks via missingness."""
    rng = np.random.default_rng(3)
    y = np.where(rng.random(400) < 0.25, "Yes", "No")
    dates = pd.Series(pd.date_range("2024-01-01", periods=400)).astype(str).where(y == "Yes")
    df = pd.DataFrame({"tenure": rng.integers(0, 60, 400), "closed_on": dates, "y": y})
    flagged = {i.column for i in detect_leakage(clean_frame(df), "y", CLS)}
    assert flagged == {"closed_on"}


def test_identical_and_categorical_mapping_leaks():
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b"], 300)
    df = pd.DataFrame(
        {
            "copy": y,
            "mapped": np.where(y == "a", "left", "right"),
            "noise": rng.normal(size=300),
            "y": y,
        }
    )
    flagged = {i.column for i in detect_leakage(df, "y", CLS)}
    assert {"copy", "mapped"} <= flagged and "noise" not in flagged


def test_regression_leak_by_correlation():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    df = pd.DataFrame({"twin": y * 2 + 1, "noise": rng.normal(size=300), "y": y})
    flagged = {i.column for i in detect_leakage(df, "y", REG)}
    assert flagged == {"twin"}


def test_check_supported_errors(messy_df):
    tiny = messy_df.head(10)
    schema = profile_dataset(clean_frame(tiny))
    assert "too_few_rows" in _codes(check_supported(tiny, schema, "target"))
    only_ids = pd.DataFrame({"row_id": [f"ID{i}" for i in range(40)], "y": [0, 1] * 20})
    schema = profile_dataset(clean_frame(only_ids))
    assert "no_usable_features" in _codes(check_supported(only_ids, schema, "y"))


def test_check_supported_ok(messy_df):
    schema = profile_dataset(clean_frame(messy_df))
    assert check_supported(messy_df, schema, "target") == []
