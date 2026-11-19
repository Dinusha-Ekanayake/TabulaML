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
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.class_labels = tuple(class_labels) if class_labels else None
        self._input_spec = list(input_spec)
        self.feature_columns = [c.name for c in config.included() if c.name != config.target]

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
        out["prediction"] = self._decode(self.pipeline.predict(X))
        if self.is_classifier and hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)
            for i, label in enumerate(self.class_labels or range(proba.shape[1])):
                out[f"proba_{label}"] = proba[:, i]
        return out

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
