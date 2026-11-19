"""Class-imbalance samplers that keep encoded categories valid.

Plain SMOTE draws new rows *between* two neighbours, which turns category codes 0 and 2
into a meaningless 0.7. ``CategoryAwareSMOTE`` looks at the already-encoded training fold:
columns holding only a few whole numbers (ordinal codes, one-hot 0/1 flags, counts) are
treated as categories and copied from a neighbour instead (SMOTE-NC); continuous columns
are interpolated as usual.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
from sklearn.base import BaseEstimator

MAX_CATEGORY_VALUES = 50


def categorical_columns(X: np.ndarray, max_values: int = MAX_CATEGORY_VALUES) -> list[int]:
    """Indices of columns that contain only a few whole-number values."""
    cols = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.allclose(col, np.round(col)) and len(np.unique(col)) <= max_values:
            cols.append(j)
    return cols


class CategoryAwareSMOTE(BaseEstimator):
    """SMOTE for numeric data, SMOTE-NC for mixed data, SMOTE-N for all-categorical data."""

    def __init__(self, k_neighbors: int = 5, random_state: int | None = None) -> None:
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        cats = categorical_columns(X)
        self.categorical_features_ = cats
        if not cats:
            sampler = SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state)
            self.method_ = "SMOTE"
        elif len(cats) == X.shape[1]:
            sampler = SMOTEN(k_neighbors=self.k_neighbors, random_state=self.random_state)
            self.method_ = "SMOTE-N"
        else:
            sampler = SMOTENC(
                categorical_features=cats,
                k_neighbors=self.k_neighbors,
                random_state=self.random_state,
            )
            self.method_ = "SMOTE-NC"
        return sampler.fit_resample(X, y)
