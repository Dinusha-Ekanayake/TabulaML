"""Model specification record shared by the registry and composite model specs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tabulaml.core.types import ModelProfile, ProblemType

CLS, REG = ProblemType.CLASSIFICATION, ProblemType.REGRESSION
BOTH = frozenset({CLS, REG})
TREE, SCALED = ModelProfile.TREE, ModelProfile.SCALE_SENSITIVE

Factory = Callable[[ProblemType, int], Any]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    group: str
    profile: ModelProfile
    tasks: frozenset[ProblemType]
    factory: Factory
    supports_class_weight: bool
    description: str
    forecast_only: bool = False  # offered only in forecasting mode


def _pick(cls_factory: Callable[[int], Any], reg_factory: Callable[[int], Any]) -> Factory:
    def factory(problem_type: ProblemType, random_state: int) -> Any:
        maker = cls_factory if problem_type == CLS else reg_factory
        return maker(random_state)

    return factory


def _only(task: ProblemType, maker: Callable[[int], Any]) -> Factory:
    def factory(problem_type: ProblemType, random_state: int) -> Any:
        if problem_type != task:
            raise ValueError(f"Model only supports {task.value}.")
        return maker(random_state)

    return factory


def _ensemble_placeholder(problem_type: ProblemType, random_state: int) -> Any:
    raise ValueError("Ensembles are built by tabulaml.models.ensembles from base learners.")
