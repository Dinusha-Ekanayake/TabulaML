"""Build a model-aware ColumnTransformer from a PreprocessingConfig (P6-P10).

"auto" settings are resolved per ModelProfile here, and the data-dependent parts of "auto"
(which columns are skewed, where the outlier bounds lie, which categories are rare) are
decided inside ``fit`` -- i.e. on training folds only. Explicit user choices win in both
profiles. Columns with an identical recipe share one sub-pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from tabulaml.config import SKEW_THRESHOLD
from tabulaml.core.types import (
    ColumnConfig,
    FeatureType,
    ModelProfile,
    PreprocessingConfig,
    ProblemType,
)
from tabulaml.preprocessing.transformers import (
    DatetimeFeatures,
    OutlierClipper,
    SkewCorrector,
    TextStats,
    ToCategory,
    ToNumeric,
)

MISSING_CATEGORY = "__missing__"
AUTO_CLIP_FACTOR = 3.0  # "auto" only trims extreme outliers; explicit iqr_clip uses 1.5
Step = tuple[str, Callable[[], Any]]


def _scaler_step(name: str) -> Step | None:
    factories = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    if name not in factories:
        return None
    return (f"scale:{name}", factories[name])


def _resolve(value: str, auto: str) -> str:
    return auto if value == "auto" else value


def _numeric_imputer(cfg: ColumnConfig, indicator: bool) -> Step:
    strategy = _resolve(cfg.impute, "median")
    if strategy == "knn":
        return ("impute:knn", lambda: KNNImputer(add_indicator=indicator))
    if strategy == "constant":
        return ("impute:constant0", lambda: SimpleImputer(strategy="constant", fill_value=0.0))
    if strategy not in ("median", "mean", "most_frequent"):
        strategy = "median"
    return (
        f"impute:{strategy}{'+indicator' if indicator else ''}",
        lambda: SimpleImputer(strategy=strategy, add_indicator=indicator),
    )


def _numeric_steps(cfg: ColumnConfig, conf: PreprocessingConfig, scaled: bool) -> list[Step]:
    """impute -> clip outliers -> skew transform -> scale (the standard order).

    Outliers are clipped BEFORE the Yeo-Johnson transform, because extreme values distort
    the transform's estimated shape; scaling comes last so it sees the final values."""
    steps: list[Step] = [("to_numeric", ToNumeric), _numeric_imputer(cfg, conf.missing_indicator)]
    outliers = _resolve(cfg.outliers, "clip_extreme" if scaled else "none")
    if outliers == "clip_extreme":
        steps.append(("clip:3xIQR", lambda: OutlierClipper(factor=AUTO_CLIP_FACTOR)))
    elif outliers == "iqr_clip":
        steps.append(("clip:1.5xIQR", lambda: OutlierClipper(factor=1.5)))
    transform = _resolve(cfg.transform, "yeo_johnson_auto" if scaled else "none")
    if transform == "yeo_johnson_auto":
        steps.append(("yeo_johnson:if_skewed", lambda: SkewCorrector(threshold=SKEW_THRESHOLD)))
    elif transform == "yeo_johnson":
        steps.append(("yeo_johnson", lambda: SkewCorrector(threshold=0.0)))
    scaler = _scaler_step(_resolve(cfg.scaling, "standard" if scaled else "none"))
    return steps + ([scaler] if scaler else [])


def _categorical_steps(cfg: ColumnConfig, conf: PreprocessingConfig, scaled: bool) -> list[Step]:
    if cfg.impute == "most_frequent":
        imputer: Step = ("impute:most_frequent", lambda: SimpleImputer(strategy="most_frequent"))
    else:
        imputer = (
            "impute:'__missing__'",
            lambda: SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
        )
    encoding = _resolve(cfg.encoding, "onehot" if scaled else "ordinal")
    if encoding == "target" and conf.is_multi_output:  # needs ONE target to encode against
        encoding = "onehot" if scaled else "ordinal"
    rare = conf.rare_category_min_frequency or None
    seed = conf.random_state
    # Integer regression targets (e.g. counts, days) would otherwise be read as multiclass.
    target_type = "continuous" if conf.problem_type == ProblemType.REGRESSION else "auto"
    encoders: dict[str, Step] = {
        "onehot": (
            "onehot(rare grouped)",
            lambda: OneHotEncoder(
                handle_unknown="infrequent_if_exist", min_frequency=rare, sparse_output=False
            ),
        ),
        "ordinal": (
            "ordinal(unknown=-1)",
            lambda: OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=rare
            ),
        ),
        "target": (
            "target_encoding(cross-fitted)",
            lambda: TargetEncoder(target_type=target_type, random_state=seed),
        ),
    }
    steps = [("to_category", ToCategory), imputer, encoders.get(encoding, encoders["ordinal"])]
    if scaled and encoding != "onehot":
        steps.append(("scale:standard", StandardScaler))
    return steps


def _boolean_steps(cfg: ColumnConfig, conf: PreprocessingConfig, scaled: bool) -> list[Step]:
    return [
        ("to_numeric", ToNumeric),
        ("impute:most_frequent", lambda: SimpleImputer(strategy="most_frequent")),
    ]


def _expanded_steps(expander: Step, scaled: bool, cfg: ColumnConfig) -> list[Step]:
    steps: list[Step] = [expander, ("impute:median", lambda: SimpleImputer(strategy="median"))]
    scaler = _scaler_step(_resolve(cfg.scaling, "standard" if scaled else "none"))
    return steps + ([scaler] if scaler else [])


def _datetime_steps(cfg: ColumnConfig, conf: PreprocessingConfig, scaled: bool) -> list[Step]:
    return _expanded_steps(("datetime_parts", DatetimeFeatures), scaled, cfg)


def _text_steps(cfg: ColumnConfig, conf: PreprocessingConfig, scaled: bool) -> list[Step]:
    return _expanded_steps(("text_length_words", TextStats), scaled, cfg)


_STEP_BUILDERS = {
    FeatureType.NUMERIC: _numeric_steps,
    FeatureType.CATEGORICAL: _categorical_steps,
    FeatureType.BOOLEAN: _boolean_steps,
    FeatureType.DATETIME: _datetime_steps,
    FeatureType.TEXT: _text_steps,
}


def _effective_type(cfg: ColumnConfig) -> FeatureType:
    """Identifier/constant columns the user chose to keep are treated as categorical."""
    if cfg.feature_type in (FeatureType.IDENTIFIER, FeatureType.CONSTANT):
        return FeatureType.CATEGORICAL
    return cfg.feature_type


def column_steps(cfg: ColumnConfig, conf: PreprocessingConfig, profile: ModelProfile) -> list[Step]:
    """The ordered (label, factory) steps applied to one column under ``profile``."""
    scaled = profile == ModelProfile.SCALE_SENSITIVE
    if cfg.feature_type == FeatureType.IDENTIFIER and cfg.encoding == "auto":
        cfg = ColumnConfig(cfg.name, FeatureType.CATEGORICAL, encoding="target")
    return _STEP_BUILDERS[_effective_type(cfg)](cfg, conf, scaled)


def build_feature_transformer(
    config: PreprocessingConfig, profile: ModelProfile
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the included columns (numpy output)."""
    included = [c for c in config.included() if c.name != config.target]
    if not included:
        raise ValueError("No usable features: include at least one column besides the target.")
    groups: dict[tuple[str, ...], tuple[list[Step], list[str]]] = {}
    for cfg in included:
        steps = column_steps(cfg, config, profile)
        key = (_effective_type(cfg).value, *(label for label, _ in steps))
        groups.setdefault(key, (steps, []))[1].append(cfg.name)
    transformers = []
    for i, (key, (steps, columns)) in enumerate(groups.items()):
        pipe = Pipeline([(f"s{j}", factory()) for j, (_, factory) in enumerate(steps)])
        transformers.append((f"{key[0]}_{i}", pipe, columns))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def describe_pipeline(config: PreprocessingConfig, profile: ModelProfile) -> list[dict[str, str]]:
    """Human-readable per-column recipe for the UI's pipeline preview."""
    rows = []
    for cfg in config.included():
        if cfg.name == config.target:
            continue
        labels = [label for label, _ in column_steps(cfg, config, profile)]
        rows.append(
            {
                "column": cfg.name,
                "type": _effective_type(cfg).value,
                "steps": " -> ".join(labels),
            }
        )
    return rows
