import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ImbalanceStrategy,
    ModelProfile,
    ProblemType,
)
from tabulaml.data.profiler import profile_dataset
from tabulaml.data.validation import detect_leakage
from tabulaml.preprocessing.builder import build_feature_transformer, describe_pipeline
from tabulaml.preprocessing.cleaning import DataCleaner, clean_frame, clean_target
from tabulaml.preprocessing.recommender import recommend_config

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
PROFILES = [ModelProfile.TREE, ModelProfile.SCALE_SENSITIVE]


def _auto(df, target, ptype):
    clean = clean_frame(df)
    schema = profile_dataset(clean)
    leaks = detect_leakage(clean, target, ptype)
    return clean, recommend_config(schema, target, ptype, leaks, y=df[target])


def test_recommender_on_messy_frame(messy_df):
    _, cfg = _auto(messy_df, "target", CLS)
    by = {c.name: c for c in cfg.columns}
    assert "target" not in by
    assert not by["row_id"].include and not by["constant_col"].include
    assert by["income"].include and "Yeo-Johnson" in by["income"].reason
    assert by["start_date"].feature_type == FeatureType.DATETIME
    assert all(c.reason for c in cfg.columns)
    assert cfg.explanations and cfg.mode == "auto"


def test_high_cardinality_gets_target_encoding():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"shop": rng.choice([f"s{i}" for i in range(40)], 400), "y": rng.random(400)})
    _, cfg = _auto(df, "y", REG)
    assert cfg.columns[0].encoding == "target"


def test_imbalance_and_target_transform_recommendations():
    rng = np.random.default_rng(0)
    y = pd.Series(np.where(rng.random(500) < 0.1, "yes", "no"))
    df = pd.DataFrame({"x": rng.normal(size=500), "y": y})
    assert _auto(df, "y", CLS)[1].imbalance == ImbalanceStrategy.CLASS_WEIGHT
    df_r = pd.DataFrame({"x": rng.normal(size=500), "y": rng.lognormal(2, 1, 500)})
    assert _auto(df_r, "y", REG)[1].target_transform == "log1p"


def _fit_and_score_shapes(df, target, ptype, profile):
    clean, cfg = _auto(df, target, ptype)
    y = clean_target(df[target], regression=ptype == REG)
    keep = y.notna()
    X, y = df.loc[keep].drop(columns=target), y[keep]
    if ptype == CLS:
        y = pd.Series(pd.factorize(y)[0], index=y.index)
    half = len(X) // 2
    cleaner = DataCleaner().fit(X.iloc[:half])
    ct = build_feature_transformer(cfg, profile)
    train = ct.fit_transform(cleaner.transform(X.iloc[:half]), y.iloc[:half])
    test = ct.transform(cleaner.transform(X.iloc[half:]))
    return ct, train, test


@pytest.mark.parametrize("profile", PROFILES)
def test_builder_handles_unseen_missing_and_shuffled_columns(messy_df, profile):
    clean, cfg = _auto(messy_df, "target", CLS)
    X = messy_df.drop(columns="target")
    y = messy_df["target"]
    cleaner = DataCleaner().fit(X)
    ct = build_feature_transformer(cfg, profile).fit(cleaner.transform(X), y)
    nasty = X.head(3).copy()
    nasty["city"] = ["Atlantis", None, "?"]
    nasty.loc[nasty.index[1], :] = np.nan
    nasty = nasty[list(reversed(nasty.columns))]
    out = ct.transform(cleaner.transform(nasty))
    assert out.shape[1] == ct.transform(cleaner.transform(X.head(1))).shape[1]
    assert np.isfinite(out.astype(float)).all()


def test_profiles_differ_in_scaling_and_encoding(messy_df):
    _, cfg = _auto(messy_df, "target", CLS)
    tree = {r["column"]: r["steps"] for r in describe_pipeline(cfg, ModelProfile.TREE)}
    scaled = {r["column"]: r["steps"] for r in describe_pipeline(cfg, ModelProfile.SCALE_SENSITIVE)}
    assert "scale" not in tree["income"] and "scale:standard" in scaled["income"]
    assert "ordinal" in tree["city"] and "onehot" in scaled["city"]


def test_explicit_override_wins_in_tree_profile(messy_df):
    _, cfg = _auto(messy_df, "target", CLS)
    by = {c.name: c for c in cfg.columns}
    custom = cfg.with_column(
        ColumnConfig("income", FeatureType.NUMERIC, scaling="robust", transform="none")
    )
    steps = {r["column"]: r["steps"] for r in describe_pipeline(custom, ModelProfile.TREE)}
    assert "scale:robust" in steps["income"]
    assert by["income"].scaling == "auto"


def test_type_override_numeric_to_categorical_works(regression_df):
    df = regression_df.assign(rating=np.arange(len(regression_df)) % 5)
    _, cfg = _auto(df, "y", REG)
    cfg = cfg.with_column(ColumnConfig("rating", FeatureType.CATEGORICAL))
    ct = build_feature_transformer(cfg, ModelProfile.SCALE_SENSITIVE)
    out = ct.fit_transform(DataCleaner().fit_transform(df.drop(columns="y")), df["y"])
    assert np.isfinite(out).all()


def test_no_features_raises(regression_df):
    _, cfg = _auto(regression_df, "y", REG)
    for c in cfg.columns:
        cfg = cfg.with_column(ColumnConfig(c.name, c.feature_type, include=False))
    with pytest.raises(ValueError, match="No usable features"):
        build_feature_transformer(cfg, ModelProfile.TREE)


@pytest.mark.parametrize("profile", PROFILES)
def test_feature_names_available(messy_df, profile):
    ct, train, _ = _fit_and_score_shapes(messy_df, "target", CLS, profile)
    assert len(ct.get_feature_names_out()) == train.shape[1]


def test_preprocessing_steps_follow_the_standard_order():
    """Numbers: parse -> impute -> clip -> transform -> scale. Categories: impute -> encode.
    Every step lives inside the model pipeline, so it is refitted on each training fold."""
    from tabulaml.core.types import (
        ColumnConfig,
        FeatureType,
        ModelProfile,
        PreprocessingConfig,
        ProblemType,
    )
    from tabulaml.preprocessing.builder import column_steps

    cfg = PreprocessingConfig(
        ProblemType.REGRESSION,
        "y",
        "custom",
        (
            ColumnConfig("n", FeatureType.NUMERIC, outliers="iqr_clip", transform="yeo_johnson"),
            ColumnConfig("c", FeatureType.CATEGORICAL),
        ),
    )
    num = [label for label, _ in column_steps(cfg.columns[0], cfg, ModelProfile.SCALE_SENSITIVE)]
    order = ["to_numeric", "impute", "clip", "yeo_johnson", "scale"]
    positions = [next(i for i, step in enumerate(num) if step.startswith(k)) for k in order]
    assert positions == sorted(positions), num
    cat = [label for label, _ in column_steps(cfg.columns[1], cfg, ModelProfile.TREE)]
    assert cat[0] == "to_category" and cat[1].startswith("impute") and "ordinal" in cat[2]
