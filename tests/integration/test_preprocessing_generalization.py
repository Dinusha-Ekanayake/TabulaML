"""The same code must preprocess very different datasets, leakage-safely."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from tabulaml.core.types import ModelProfile, ProblemType
from tabulaml.data.profiler import profile_dataset
from tabulaml.data.samples import load_sample
from tabulaml.data.validation import detect_leakage
from tabulaml.preprocessing.builder import build_feature_transformer
from tabulaml.preprocessing.cleaning import DataCleaner, clean_frame, clean_target
from tabulaml.preprocessing.recommender import recommend_config

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION


def _sklearn(loader):
    data = loader(as_frame=True)
    return data.frame.rename(columns={"target": "y"})


def _injected():
    X, y = make_classification(n_samples=400, n_features=6, random_state=0)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["colour"] = rng.choice(["red", "Red ", "blue", "?"], 400)
    df["code"] = rng.choice([f"c{i}" for i in range(60)], 400)
    df["when"] = pd.date_range("2020-01-01", periods=400).astype(str)
    df.loc[rng.random(400) < 0.1, "f0"] = np.nan
    df["y"] = np.where(y == 1, "yes", "no")
    return df


def _datasets():
    out = [
        ("breast_cancer", _sklearn(load_breast_cancer), "y", CLS),
        ("wine_multiclass", _sklearn(load_wine), "y", CLS),
        ("diabetes", _sklearn(load_diabetes), "y", REG),
        ("injected", _injected(), "y", CLS),
    ]
    for key in ("lk_tea_auction_catalogue", "lk_tea_auction_price", "lk_colombo_dengue"):
        df, card = load_sample(key)
        out.append((key, df, card.target, card.problem_type))
    return out


@pytest.mark.parametrize("profile", [ModelProfile.TREE, ModelProfile.SCALE_SENSITIVE])
@pytest.mark.parametrize(("name", "df", "target", "ptype"), _datasets())
def test_recommend_build_transform_on_any_dataset(name, df, target, ptype, profile):
    clean = clean_frame(df)
    cfg = recommend_config(
        profile_dataset(clean), target, ptype, detect_leakage(clean, target, ptype), y=df[target]
    )
    y = clean_target(df[target], regression=ptype == REG)
    X = df.loc[y.notna()].drop(columns=target)
    y = y[y.notna()]
    if ptype == CLS:
        y = pd.Series(pd.factorize(y)[0], index=y.index)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=0)
    cleaner = DataCleaner().fit(X_tr)
    ct = build_feature_transformer(cfg, profile)
    train = ct.fit_transform(cleaner.transform(X_tr), y_tr)
    test = ct.transform(cleaner.transform(X_te))
    assert train.shape[1] == test.shape[1] > 0
    assert np.isfinite(test.astype(float)).all(), name


class SpyTransformer(BaseEstimator, TransformerMixin):
    """Records which row labels each fit call saw."""

    seen: list = []

    def fit(self, X, y=None):
        SpyTransformer.seen.append(set(X.index))
        return self

    def transform(self, X):
        return X


def test_learned_steps_never_see_test_or_validation_rows(messy_df):
    clean = clean_frame(messy_df)
    cfg = recommend_config(profile_dataset(clean), "target", CLS, y=messy_df["target"])
    X, y = messy_df.drop(columns="target"), messy_df["target"]
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    pipe = Pipeline(
        [
            ("spy", SpyTransformer()),
            ("cleaner", DataCleaner()),
            ("features", build_feature_transformer(cfg, ModelProfile.SCALE_SENSITIVE)),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    SpyTransformer.seen = []
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cross_validate(pipe, X_tr, y_tr, cv=cv)
    assert len(SpyTransformer.seen) == 5
    test_rows = set(X_te.index)
    for seen in SpyTransformer.seen:
        assert not seen & test_rows
        assert len(seen) <= int(np.ceil(len(X_tr) * 0.8))
