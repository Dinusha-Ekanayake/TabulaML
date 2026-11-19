import numpy as np
import pandas as pd

from tabulaml.preprocessing.transformers import (
    DatetimeFeatures,
    OutlierClipper,
    SkewCorrector,
    TextStats,
    ToCategory,
    ToNumeric,
)


def test_to_numeric_coerces_text():
    out = ToNumeric().fit_transform(pd.DataFrame({"a": ["1,000", "x", 3]}))
    assert out[0, 0] == 1000 and np.isnan(out[1, 0]) and out[2, 0] == 3


def test_to_category_unifies_float_and_int_spelling():
    enc = ToCategory().fit(pd.DataFrame({"a": [1.0, 2.0, np.nan]}))
    assert enc.transform(pd.DataFrame({"a": [1]}))[0, 0] == "1"
    assert enc.transform(pd.DataFrame({"a": [1.0, None]})).tolist() == [["1"], [np.nan]]


def test_datetime_features_reference_learned_in_fit():
    train = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-11"])})
    dtf = DatetimeFeatures().fit(train)
    out = dtf.transform(pd.DataFrame({"d": ["2024-01-21", None]}))
    assert out.shape == (2, 5)
    assert out[0, 4] == 20 and np.isnan(out[1]).all()
    assert list(dtf.get_feature_names_out())[0] == "d_year"


def test_text_stats():
    out = TextStats().fit_transform(pd.DataFrame({"t": ["hello big world", None]}))
    assert out[0].tolist() == [15, 3] and np.isnan(out[1]).all()


def test_outlier_clipper_learns_bounds_and_skips_zero_iqr():
    train = np.array([[1.0, 0], [2, 0], [3, 0], [4, 0], [5, 1]])
    clip = OutlierClipper(1.5).fit(train)
    out = clip.transform(np.array([[1000.0, 1.0]]))
    assert out[0, 0] == clip.upper_[0] < 1000
    assert out[0, 1] == 1.0  # indicator column with IQR 0 untouched


def test_skew_corrector_only_touches_skewed_columns():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.lognormal(0, 1, 500), rng.normal(size=500)])
    sc = SkewCorrector(1.0).fit(X)
    out = sc.transform(X)
    assert list(sc.columns_) == [0]
    np.testing.assert_allclose(out[:, 1], X[:, 1])
    assert abs(pd.Series(out[:, 0]).skew()) < 1


def test_skew_corrector_noop_when_nothing_skewed():
    X = np.random.default_rng(0).normal(size=(100, 2))
    np.testing.assert_allclose(SkewCorrector().fit(X).transform(X), X)
