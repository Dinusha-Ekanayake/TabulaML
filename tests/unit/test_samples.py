import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import skew

from tabulaml.core.types import ProblemType
from tabulaml.data.samples import list_samples, load_sample

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("key", ["lk_telecom_churn", "lk_patient_los"])
def test_sample_loads_and_matches_card(key):
    df, card = load_sample(key)
    assert card.target in df.columns
    assert card.is_synthetic and "Synthetic" in card.provenance
    assert set(card.recommended_exclusions) <= set(df.columns)
    assert set(card.features) == set(df.columns)


def test_list_samples_filters_by_problem_type():
    assert [c.key for c in list_samples(ProblemType.CLASSIFICATION)] == ["lk_telecom_churn"]
    assert [c.key for c in list_samples(ProblemType.REGRESSION)] == ["lk_patient_los"]
    assert len(list_samples()) == 2


def test_unknown_sample_raises():
    with pytest.raises(KeyError):
        load_sample("nope")


def test_churn_positive_rate_is_realistic():
    df, _ = load_sample("lk_telecom_churn")
    rate = (df["churned"] == "Yes").mean()
    assert 0.15 <= rate <= 0.30


def test_los_target_is_right_skewed_positive_integers():
    df, _ = load_sample("lk_patient_los")
    y = df["length_of_stay_days"]
    assert y.min() >= 1 and skew(y) > 1


def test_samples_contain_planted_messiness():
    churn, _ = load_sample("lk_telecom_churn")
    assert churn["monthly_charge_lkr"].astype(str).str.startswith("Rs.").any()
    assert churn.duplicated().any()
    assert churn["disconnection_date"].notna().sum() == (churn["churned"] == "Yes").sum()
    assert churn.isna().any().any()


def test_generator_is_reproducible(tmp_path):
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_samples.py"), "--out", str(out)],
            check=True,
            capture_output=True,
        )
        outs.append(pd.read_csv(out / "lk_telecom_churn.csv"))
    pd.testing.assert_frame_equal(outs[0], outs[1])
