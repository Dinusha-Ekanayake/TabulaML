"""Use a trained pipeline for single-record and batch predictions (P13).

The exact fitted pipeline (cleaner included) is reused, so prediction inputs are cleaned the
same way as training data; unseen categories and missing values never crash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from tabulaml.core.types import FeatureType, PreprocessingConfig, ProblemType, ValidationIssue
from tabulaml.preprocessing.cleaning import DataCleaner

MAX_OPTIONS = 50


def _spec_for(name: str, ftype: FeatureType, s: pd.Series) -> dict[str, Any]:
    spec: dict[str, Any] = {"name": name, "type": ftype.value, "default": None, "options": None}
    present = s.dropna()
    if ftype == FeatureType.NUMERIC and len(present):
        spec |= {
            "default": float(present.median()),
            "min": float(present.min()),
            "max": float(present.max()),
        }
    elif ftype == FeatureType.BOOLEAN and len(present):
        spec["default"] = bool(present.mode().iloc[0] >= 0.5)
    elif ftype == FeatureType.DATETIME and len(present):
        spec["default"] = present.sort_values().iloc[len(present) // 2].date()
    elif len(present):
        counts = present.astype(str).value_counts()
        spec |= {"default": counts.index[0], "options": list(counts.index[:MAX_OPTIONS])}
    return spec


def make_input_spec(X_raw: pd.DataFrame, config: PreprocessingConfig) -> list[dict[str, Any]]:
    """Form fields for each model input: type, default (median / mode) and options."""
    cleaned = DataCleaner(normalize_category_case=config.normalize_category_case).fit_transform(
        X_raw
    )
    specs = []
    for col in config.included():
        if col.name == config.target or col.name not in cleaned.columns:
            continue
        specs.append(_spec_for(col.name, col.feature_type, cleaned[col.name]))
    return specs


class Predictor:
    def __init__(
        self,
        pipeline: Any,
        config: PreprocessingConfig,
        class_labels: Sequence[str] | None,
        input_spec: Sequence[dict[str, Any]],
        threshold: float | None = None,
        positive_index: int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.class_labels = tuple(class_labels) if class_labels else None
        self._input_spec = list(input_spec)
        roles = {config.target, config.group_column, config.order_column}
        self.feature_columns = [c.name for c in config.included() if c.name not in roles]
        self.threshold = threshold  # binary only: P(positive) >= threshold -> positive
        self.positive_index = positive_index

    @property
    def is_classifier(self) -> bool:
        return self.config.problem_type == ProblemType.CLASSIFICATION

    def input_spec(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._input_spec]

    def validate_input(self, df: pd.DataFrame) -> list[ValidationIssue]:
        if df.empty:
            return [ValidationIssue("error", "empty_input", "The file has no rows.")]
        issues = []
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            msg = f"Missing required columns: {', '.join(missing)}."
            issues.append(ValidationIssue("error", "missing_columns", msg))
        extra = [c for c in df.columns if c not in self.feature_columns and c != self.config.target]
        if extra:
            msg = f"Ignoring columns the model does not use: {', '.join(map(str, extra[:10]))}."
            issues.append(ValidationIssue("warning", "extra_columns", msg))
        present = [c for c in self.feature_columns if c in df.columns]
        empty_rows = int(df[present].isna().all(axis=1).sum()) if present else 0
        if empty_rows:
            msg = f"{empty_rows} rows have no values; their predictions rely only on defaults."
            issues.append(ValidationIssue("warning", "empty_rows", msg))
        return issues

    def _decode(self, y_pred: np.ndarray) -> np.ndarray:
        if not self.is_classifier or self.class_labels is None:
            return y_pred
        return np.asarray(self.class_labels, dtype=object)[np.asarray(y_pred, dtype=int)]

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a NEW frame: the input plus ``prediction`` (and ``proba_<class>`` columns)."""
        errors = [i.message for i in self.validate_input(df) if i.level == "error"]
        if errors:
            raise ValueError(" ".join(errors))
        X = df[self.feature_columns]
        out = df.copy()
        pred = np.asarray(self.pipeline.predict(X)).ravel()
        if self.is_classifier and hasattr(self.pipeline, "predict_proba"):
            proba = np.asarray(self.pipeline.predict_proba(X), dtype=float)
            pred = self._apply_threshold(pred, proba)
            for i, label in enumerate(self.class_labels or range(proba.shape[1])):
                out[f"proba_{label}"] = proba[:, i]
        out["prediction"] = self._decode(pred)
        return out

    def _apply_threshold(self, pred: np.ndarray, proba: np.ndarray) -> np.ndarray:
        """Binary tasks with a chosen threshold: positive when P(positive) >= threshold."""
        if self.threshold is None or self.positive_index is None or proba.shape[1] != 2:
            return pred
        pos = self.positive_index
        return np.where(proba[:, pos] >= self.threshold, pos, 1 - pos)

    def quality_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per column: unparseable values, values outside the training range, unseen
        categories and missing share. Shown before batch scoring (nothing is changed)."""
        rows: list[dict[str, Any]] = []
        cleaner = _find_cleaner(self.pipeline)
        parse = cleaner.parse_report(df) if cleaner is not None else {}
        for spec in self._input_spec:
            name = spec["name"]
            if name not in df.columns:
                continue
            col = df[name]
            if name in parse:
                rows.append(
                    _issue_row(
                        name, "could not be parsed", parse[name]["failed"], parse[name]["examples"]
                    )
                )
            rows.extend(_range_rows(name, spec, col, cleaner))
            missing = float(col.isna().mean())
            if missing > 0.2:
                rows.append(
                    _issue_row(
                        name,
                        "many missing values",
                        int(col.isna().sum()),
                        [f"{missing:.0%} missing"],
                    )
                )
        return pd.DataFrame(rows, columns=["column", "issue", "rows", "examples"])

    def predict_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = pd.DataFrame([dict(record)]).reindex(columns=self.feature_columns)
        result = self.predict_batch(row).iloc[0]
        probabilities = None
        if self.is_classifier and self.class_labels:
            probabilities = {
                lab: float(result[f"proba_{lab}"])
                for lab in self.class_labels
                if f"proba_{lab}" in result
            } or None
        prediction = result["prediction"]
        if not self.is_classifier:
            prediction = float(prediction)
        return {"prediction": prediction, "probabilities": probabilities}


def _find_cleaner(estimator: Any) -> Any:
    """The fitted DataCleaner inside a (target-wrapped / ensemble) pipeline, if any."""
    est = (
        getattr(estimator, "regressor_", None) or getattr(estimator, "regressor", None) or estimator
    )
    steps = getattr(est, "named_steps", None)
    if steps and "cleaner" in steps:
        return steps["cleaner"]
    for _, member in (
        getattr(est, "estimators_", []) or getattr(est, "named_estimators_", {}).items()
    ):
        found = _find_cleaner(member)
        if found is not None:
            return found
    return None


def _issue_row(column: str, issue: str, rows: int, examples: Sequence[Any]) -> dict[str, Any]:
    return {
        "column": column,
        "issue": issue,
        "rows": int(rows),
        "examples": ", ".join(map(str, list(examples)[:3])),
    }


def _range_rows(name: str, spec: dict[str, Any], col: pd.Series, cleaner: Any) -> list[dict]:
    if spec["type"] == "numeric" and spec.get("min") is not None:
        from tabulaml.preprocessing.cleaning import coerce_numeric

        values = coerce_numeric(col)
        outside = (values < spec["min"]) | (values > spec["max"])
        if outside.any():
            return [
                _issue_row(
                    name,
                    f"outside training range [{spec['min']:g}, {spec['max']:g}]",
                    int(outside.sum()),
                    values[outside].unique(),
                )
            ]
    if spec["type"] == "categorical" and cleaner is not None:
        seen = set(cleaner.category_maps_.get(name, {}))
        if seen:
            text = col.dropna().astype(str).str.strip().str.lower()
            unseen = text[~text.isin(seen)]
            if len(unseen):
                return [
                    _issue_row(
                        name, "categories not seen in training", len(unseen), unseen.unique()
                    )
                ]
    return []
