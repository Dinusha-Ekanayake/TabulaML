"""Public (real) Sri Lankan datasets: bundled dengue data and the Kaggle telco builder."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabulaml.core.types import ProblemType
from tabulaml.data.samples import load_sample
from tabulaml.data.validation import detect_leakage
from tabulaml.preprocessing.cleaning import clean_frame

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "fetch", ROOT / "scripts" / "fetch_public_datasets.py"
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def test_dengue_sample_is_real_public_data():
    df, card = load_sample("lk_colombo_dengue")
    assert not card.is_synthetic and card.license == "CC0-1.0"
    assert len(df) == 279 and card.target == "dengue_cases"
    assert set(card.features) == set(df.columns)


def test_dengue_lags_only_use_past_weeks():
    raw = pd.DataFrame(
        {"Cases": [10, 20, 30, 40], "Year": [2010] * 4, "Week": [1, 2, 3, 4], "PP": [1, 2, 3, 4]}
        | {c: [0.0] * 4 for c in ["TEM", "TMAX", "Tm", "SLP", "H", "VV", "V", "VM"]}
    )
    out = fetch.build_dengue(raw)
    assert out["cases_last_week"].tolist()[1:] == [10, 20, 30]
    assert np.isnan(out["cases_2_weeks_ago"].iloc[1])
    assert out["rainfall_prev_4_weeks_mm"].tolist()[1:] == [1, 3, 6]
    assert out.columns[-1] == "dengue_cases"


def _fake_kaggle_folder(tmp_path: Path, n: int = 120) -> Path:
    rng = np.random.default_rng(0)
    usage_types = ["usage_app_whatsapp_daily", "usage_voice_onnet_daily"]
    ids = [f"{i:064x}" for i in range(n)]
    churn = (rng.random(n) < 0.3).astype(int)
    rows = []
    for cid, c in zip(ids, churn, strict=True):
        for usage in usage_types:
            days = rng.gamma(2, 5, 90)
            if c:
                days[45:] = 0  # churners stop using the service later in the window
            rows.append([cid, usage, *days])
    cols = ["customer_id", "usage_type", *[f"Day_{i}" for i in range(1, 91)]]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "main.csv", index=False)
    labels = pd.DataFrame({"customer_id": ids, "churn": churn})
    labels.iloc[:90].to_csv(tmp_path / "train_cxid.csv", index=False)
    labels.iloc[90:].to_csv(tmp_path / "test_cxid.csv", index=False)
    return tmp_path


def test_telco_builder_produces_features_and_flags_its_leak(tmp_path):
    df = fetch.build_telco(_fake_kaggle_folder(tmp_path))
    assert len(df) == 120 and set(df["churn"]) == {"Yes", "No"}
    assert "app_whatsapp_m1_total" in df.columns
    assert "voice_onnet_m1_trend" in df.columns
    flagged = {
        i.column for i in detect_leakage(clean_frame(df), "churn", ProblemType.CLASSIFICATION)
    }
    assert "active_days_61_90_any_service" in flagged
    assert not any(c.endswith("_m1_total") for c in flagged)


def test_missing_fetched_sample_gives_helpful_error(tmp_path, monkeypatch):
    import tabulaml.data.samples as samples

    (tmp_path / "ghost.json").write_text((ROOT / "data/samples/lk_colombo_dengue.json").read_text())
    monkeypatch.setattr(samples, "SAMPLES_DIR", tmp_path)
    monkeypatch.setattr(samples, "_registry", lambda d=tmp_path: {"ghost": tmp_path / "ghost.json"})
    assert samples.list_samples() == []
    with pytest.raises(FileNotFoundError, match="fetch_public_datasets"):
        samples.load_sample("ghost")
