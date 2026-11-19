"""Bundled sample datasets and their dataset cards.

Samples are discovered from ``data/samples/*.json`` so adding a real dataset only needs a CSV
and a card; no code changes. Loaded samples go through the same pipeline as uploads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tabulaml.core.types import ProblemType

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"


@dataclass(frozen=True)
class DatasetCard:
    key: str
    title: str
    problem_type: ProblemType
    target: str
    provenance: str
    is_synthetic: bool
    description: str
    context: str
    license: str
    collection_period: str
    features: dict[str, str]
    target_definition: str
    known_limitations: tuple[str, ...]
    anonymisation: str
    recommended_exclusions: tuple[str, ...]
    source_url: str = ""


def _card_from_json(path: Path) -> DatasetCard:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DatasetCard(
        key=raw["key"],
        title=raw["title"],
        problem_type=ProblemType(raw["problem_type"]),
        target=raw["target"],
        provenance=raw["provenance"],
        is_synthetic=bool(raw["is_synthetic"]),
        description=raw["description"],
        context=raw.get("context", ""),
        license=raw["license"],
        collection_period=raw.get("collection_period", ""),
        features=dict(raw.get("features", {})),
        target_definition=raw["target_definition"],
        known_limitations=tuple(raw.get("known_limitations", ())),
        anonymisation=raw.get("anonymisation", ""),
        recommended_exclusions=tuple(raw.get("recommended_exclusions", ())),
        source_url=raw.get("source_url", ""),
    )


def _registry(samples_dir: Path = SAMPLES_DIR) -> dict[str, Path]:
    return {p.stem: p for p in sorted(samples_dir.glob("*.json"))}


def _csv_path(card_path: Path) -> Path:
    """A card may name a shared data file (``"file"``), e.g. two tasks on one table."""
    raw = json.loads(card_path.read_text(encoding="utf-8"))
    return card_path.parent / raw.get("file", f"{card_path.stem}.csv")


def list_samples(problem_type: ProblemType | None = None) -> list[DatasetCard]:
    """Cards of samples whose CSV is present (fetch-on-demand datasets appear once fetched)."""
    cards = [_card_from_json(p) for p in _registry().values() if _csv_path(p).exists()]
    if problem_type is None:
        return cards
    return [c for c in cards if c.problem_type == problem_type]


def load_sample(key: str) -> tuple[pd.DataFrame, DatasetCard]:
    registry = _registry()
    if key not in registry:
        raise KeyError(f"Unknown sample dataset '{key}'. Available: {sorted(registry)}")
    card = _card_from_json(registry[key])
    csv_path = _csv_path(registry[key])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sample '{key}' has not been downloaded yet. "
            f"Run: python scripts/fetch_public_datasets.py (see the dataset card)."
        )
    return pd.read_csv(csv_path), card
