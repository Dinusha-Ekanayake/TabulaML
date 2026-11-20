"""Save and load trained model bundles (joblib).

SECURITY: joblib/pickle files can execute code when loaded. Only load bundles that this
application produced; never load a bundle from an untrusted source.
"""

from __future__ import annotations

import io
import logging
import pickle
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib

from tabulaml.core.types import ModelResult, TrainingRun
from tabulaml.inference.predictor import Predictor

logger = logging.getLogger(__name__)
BUNDLE_FORMAT = "tabulaml-bundle-v1"
_REQUIRED = ("format", "pipeline", "config", "class_labels", "input_spec")


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def make_bundle(
    result: ModelResult,
    run: TrainingRun,
    input_spec: Sequence[dict],
    threshold: float | None = None,
) -> dict[str, Any]:
    if result.status != "ok" or result.pipeline is None:
        raise ValueError(f"Model '{result.name}' did not train successfully; nothing to save.")
    return {
        "format": BUNDLE_FORMAT,
        "model_key": result.key,
        "model_name": result.name,
        "pipeline": result.pipeline,
        "config": run.config,
        "class_labels": run.class_labels,
        "input_spec": list(input_spec),
        "test_metrics": dict(result.test_metrics),
        "params": dict(result.params),
        "threshold": threshold,
        "positive_index": run.positive_index,
        "target_labels": dict(run.target_labels),
        "versions": dict(run.versions),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tabulaml_version": _package_version("tabulaml"),
    }


def bundle_bytes(
    result: ModelResult,
    run: TrainingRun,
    input_spec: Sequence[dict],
    threshold: float | None = None,
    fmt: str = "joblib",
) -> bytes:
    """Serialised bundle for a download button: ``fmt`` is "joblib" (compressed, the
    scikit-learn convention) or "pickle" (standard library, no extra package to load)."""
    bundle = make_bundle(result, run, input_spec, threshold)
    if fmt == "pickle":
        return pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    if fmt != "joblib":
        raise ValueError(f"Unknown bundle format '{fmt}'. Use 'joblib' or 'pickle'.")
    buffer = io.BytesIO()
    joblib.dump(bundle, buffer, compress=3)
    return buffer.getvalue()


def save_bundle(
    directory: str | Path, result: ModelResult, run: TrainingRun, input_spec: Sequence[dict]
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = Path(directory) / f"{result.key}_{stamp}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(make_bundle(result, run, input_spec), path, compress=3)
    return path


def load_bundle(source: str | Path | bytes) -> Predictor:
    """Load a bundle produced by TabulaML (trusted sources only, see module docstring)."""
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    bundle = _deserialise(raw)
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError("Not a TabulaML model bundle.")
    missing = [k for k in _REQUIRED if k not in bundle]
    if missing:
        raise ValueError(f"Bundle is missing fields: {missing}")
    saved = bundle.get("versions", {}).get("scikit-learn")
    current = _package_version("scikit-learn")
    if saved and saved != current:
        logger.warning("Bundle saved with scikit-learn %s, running %s.", saved, current)
    return Predictor(
        bundle["pipeline"],
        bundle["config"],
        bundle["class_labels"],
        bundle["input_spec"],
        threshold=bundle.get("threshold"),
        positive_index=bundle.get("positive_index"),
        target_labels=bundle.get("target_labels"),
    )


def _deserialise(raw: bytes) -> Any:
    """joblib (compressed or not) first; plain pickle as a fallback (trusted files only)."""
    try:
        return joblib.load(io.BytesIO(raw))
    except Exception:  # noqa: BLE001 - try the standard-library format next
        return pickle.loads(raw)  # noqa: S301 - TabulaML bundles only, see module docstring
