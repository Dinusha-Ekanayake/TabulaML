import numpy as np
import pandas as pd

from tabulaml.preprocessing.cleaning import DataCleaner, clean_frame, coerce_numeric


def test_currency_strings_become_numbers(messy_df):
    out = clean_frame(messy_df)
    assert out["charge"].dtype == "float64"
    assert out["charge"].between(500, 5000).all()


def test_rs_1250_exact():
    df = pd.DataFrame({"p": ["Rs. 1,250", "LKR 2,000", "300", "?"] * 10})
    assert list(clean_frame(df)["p"].iloc[:4].fillna(-1)) == [1250.0, 2000.0, 300.0, -1.0]


def test_boolean_variants_map_to_zero_one(messy_df):
    out = clean_frame(messy_df)
    assert set(out["flag"].dropna().unique()) == {0.0, 1.0}


def test_placeholder_tokens_become_nan():
    df = pd.DataFrame({"c": ["a", "?", "NA", " null ", "b", "-"] * 5})
    assert clean_frame(df)["c"].isna().sum() == 20


def test_category_case_and_whitespace_normalised(messy_df):
    out = clean_frame(messy_df)
    assert out["city"].nunique() == 4
    assert not out["city"].str.contains(r"^\s|\s$").any()


def test_dates_are_parsed(messy_df):
    out = clean_frame(messy_df)
    assert str(out["start_date"].dtype).startswith("datetime64")


def test_input_frame_is_not_mutated(messy_df):
    before = messy_df.copy(deep=True)
    clean_frame(messy_df)
    pd.testing.assert_frame_equal(messy_df, before)


def test_unseen_column_order_and_extra_columns_handled(messy_df):
    cleaner = DataCleaner().fit(messy_df.drop(columns="target"))
    shuffled = messy_df.drop(columns="target").iloc[:5][::-1].copy()
    shuffled = shuffled[sorted(shuffled.columns)]
    shuffled["brand_new"] = 1
    out = cleaner.transform(shuffled)
    assert list(out.columns) == list(cleaner.feature_names_in_)
    assert "brand_new" not in out.columns


def test_missing_column_becomes_nan(messy_df):
    cleaner = DataCleaner().fit(messy_df)
    out = cleaner.transform(messy_df.drop(columns=["income"]).head(3))
    assert out["income"].isna().all()


def test_unseen_category_is_kept_without_error(messy_df):
    cleaner = DataCleaner().fit(messy_df)
    record = messy_df.head(1).copy()
    record["city"] = "Atlantis"
    assert cleaner.transform(record)["city"].iloc[0] == "Atlantis"


def test_single_record_transform_matches_batch(messy_df):
    cleaner = DataCleaner().fit(messy_df)
    batch = cleaner.transform(messy_df)
    one = cleaner.transform(messy_df.iloc[[7]])
    pd.testing.assert_series_equal(batch.iloc[7], one.iloc[0], check_names=False)


def test_native_dtypes_pass_through():
    df = pd.DataFrame(
        {"n": [1, 2, 3], "b": [True, False, True], "d": pd.date_range("2024-01-01", periods=3)}
    )
    out = DataCleaner().fit_transform(df)
    assert out["n"].dtype == "float64" and out["b"].dtype == "float64"
    assert str(out["d"].dtype).startswith("datetime64")


def test_all_missing_column_is_handled():
    df = pd.DataFrame({"a": [np.nan] * 5, "b": [1, 2, 3, 4, 5]})
    assert DataCleaner().fit_transform(df)["a"].isna().all()


def test_coerce_numeric_helper():
    out = coerce_numeric(pd.Series(["1,000", "x", "5%"]))
    assert out.iloc[0] == 1000 and np.isnan(out.iloc[1]) and out.iloc[2] == 5


def test_unfitted_transform_raises():
    import pytest

    with pytest.raises(ValueError):
        DataCleaner().transform(pd.DataFrame({"a": [1]}))


def test_case_normalisation_can_be_disabled():
    df = pd.DataFrame({"c": ["Aa", "aa", "AA", "bb"] * 5})
    assert DataCleaner(normalize_category_case=False).fit_transform(df)["c"].nunique() == 4
