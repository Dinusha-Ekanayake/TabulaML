import numpy as np
import pandas as pd

from tabulaml.core.types import FeatureType
from tabulaml.data.profiler import infer_feature_type, profile_dataset
from tabulaml.preprocessing.cleaning import clean_frame


def _types(df):
    schema = profile_dataset(clean_frame(df))
    return {c.name: c.inferred_type for c in schema.columns}


def test_messy_frame_types(messy_df):
    t = _types(messy_df)
    assert t["row_id"] == FeatureType.IDENTIFIER
    assert t["income"] == FeatureType.NUMERIC
    assert t["charge"] == FeatureType.NUMERIC
    assert t["city"] == FeatureType.CATEGORICAL
    assert t["flag"] == FeatureType.BOOLEAN
    assert t["start_date"] == FeatureType.DATETIME
    assert t["constant_col"] == FeatureType.CONSTANT
    assert t["target"] == FeatureType.BOOLEAN


def test_free_text_is_text():
    s = pd.Series(
        [f"this is a fairly long free text comment number {i} about service" for i in range(50)],
        name="note",
    )
    assert infer_feature_type(s, 50) == FeatureType.TEXT


def test_integer_column_named_id_is_identifier():
    s = pd.Series(range(100), name="customer_id")
    assert infer_feature_type(s, 100) == FeatureType.IDENTIFIER


def test_continuous_float_is_never_identifier():
    s = pd.Series(np.random.default_rng(0).normal(size=100), name="reading")
    assert infer_feature_type(s, 100) == FeatureType.NUMERIC


def test_all_nan_is_constant():
    assert infer_feature_type(pd.Series([np.nan] * 5, name="x"), 5) == FeatureType.CONSTANT


def test_profile_statistics(messy_df):
    schema = profile_dataset(clean_frame(messy_df))
    income = schema.get("income")
    assert 0.04 < income.missing_ratio < 0.06
    assert income.skewness is not None and income.skewness > 1
    assert income.outlier_ratio is not None and income.outlier_ratio > 0
    assert schema.n_rows == 200
    assert "identifier" in schema.get("row_id").warnings
    assert "constant" in schema.get("constant_col").warnings


def test_low_cardinality_numeric_warns():
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5] * 20, "y": [0, 1] * 50})
    assert "low_card_numeric" in profile_dataset(df).get("rating").warnings


def test_duplicate_rows_counted():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert profile_dataset(df).n_duplicate_rows == 1
