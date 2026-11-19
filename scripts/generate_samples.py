"""Generate the bundled synthetic Sri Lankan-context sample datasets and their cards.

Usage: python scripts/generate_samples.py [--seed 42] [--out data/samples]

The data is SYNTHETIC. It is deliberately messy (missing values, placeholder tokens,
currency strings, inconsistent case, duplicates, outliers) and each dataset contains one
planted leakage column so the preprocessing engine has something to prove.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DISTRICT_PROVINCE = {
    "Colombo": "Western",
    "Gampaha": "Western",
    "Kalutara": "Western",
    "Kandy": "Central",
    "Matale": "Central",
    "Nuwara Eliya": "Central",
    "Galle": "Southern",
    "Matara": "Southern",
    "Hambantota": "Southern",
    "Jaffna": "Northern",
    "Kilinochchi": "Northern",
    "Mannar": "Northern",
    "Vavuniya": "Northern",
    "Mullaitivu": "Northern",
    "Batticaloa": "Eastern",
    "Ampara": "Eastern",
    "Trincomalee": "Eastern",
    "Kurunegala": "North Western",
    "Puttalam": "North Western",
    "Anuradhapura": "North Central",
    "Polonnaruwa": "North Central",
    "Badulla": "Uva",
    "Monaragala": "Uva",
    "Ratnapura": "Sabaragamuwa",
    "Kegalle": "Sabaragamuwa",
}
DISTRICTS = list(DISTRICT_PROVINCE)
DISTRICT_WEIGHTS = np.array([8 if d in ("Colombo", "Gampaha", "Kandy") else 1.5 for d in DISTRICTS])
DISTRICT_WEIGHTS = DISTRICT_WEIGHTS / DISTRICT_WEIGHTS.sum()
TAMIL_PROVINCES = {"Northern", "Eastern"}
REFERENCE_DATE = pd.Timestamp("2024-12-31")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(score: np.ndarray, target_rate: float) -> float:
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(score + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _languages(provinces: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = []
    for prov in provinces:
        probs = [0.15, 0.8, 0.05] if prov in TAMIL_PROVINCES else [0.8, 0.07, 0.13]
        out.append(rng.choice(["Sinhala", "Tamil", "English"], p=probs))
    return np.array(out)


def _inject_nan(df: pd.DataFrame, cols: list[str], frac: float, rng: np.random.Generator) -> None:
    for col in cols:
        mask = rng.random(len(df)) < frac
        df[col] = df[col].astype("object").where(~mask, np.nan)


def _inject_tokens(
    df: pd.DataFrame, cols: list[str], frac: float, rng: np.random.Generator
) -> None:
    for col in cols:
        mask = rng.random(len(df)) < frac
        df[col] = df[col].astype("object").where(~mask, "?")


def _messy_text(series: pd.Series, rng: np.random.Generator) -> pd.Series:
    def tweak(value: str) -> str:
        roll = rng.random()
        if roll < 0.04:
            return f" {value.lower()}"
        if roll < 0.08:
            return value.upper()
        if roll < 0.10:
            return f"{value} "
        return value

    return series.map(tweak)


def _add_duplicates(df: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    dupes = df.sample(int(len(df) * frac), random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, dupes], ignore_index=True).sample(frac=1.0, random_state=1)


def _yes_no_messy(flag: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yes = rng.choice(["Yes", "yes", "Y"], len(flag), p=[0.7, 0.2, 0.1])
    no = rng.choice(["No", "no", "N"], len(flag), p=[0.7, 0.2, 0.1])
    return np.where(flag, yes, no)


def generate_telecom_churn(seed: int, n: int = 4950) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    district = rng.choice(DISTRICTS, n, p=DISTRICT_WEIGHTS)
    province = np.array([DISTRICT_PROVINCE[d] for d in district])
    connection = rng.choice(
        ["Mobile Prepaid", "Mobile Postpaid", "Fibre", "4G Home Broadband", "ADSL"],
        n,
        p=[0.45, 0.2, 0.12, 0.15, 0.08],
    )
    contract = rng.choice(["No contract", "12-month", "24-month"], n, p=[0.5, 0.3, 0.2])
    tenure = np.clip(rng.gamma(2.2, 14, n), 0, 120).round().astype(int)
    base_charge = (
        pd.Series(connection)
        .map(
            {
                "Mobile Prepaid": 900,
                "Mobile Postpaid": 2200,
                "Fibre": 3800,
                "4G Home Broadband": 3000,
                "ADSL": 2400,
            }
        )
        .to_numpy()
    )
    charge = (base_charge * rng.lognormal(0, 0.25, n)).round(-1)
    complaints = rng.poisson(0.8, n)
    signal = np.clip(rng.normal(3.4, 0.9, n), 1, 5).round(1)
    late = rng.poisson(0.6, n)
    recharge_gap = np.clip(rng.gamma(2, 6, n), 0, 120).round().astype(int)
    score = (
        0.35 * complaints
        - 0.55 * (signal - 3.4)
        - 0.02 * tenure
        + 0.3 * late
        + 0.012 * recharge_gap
        + 0.0001 * (charge - 2000)
        + pd.Series(contract)
        .map({"No contract": 0.8, "12-month": 0.0, "24-month": -0.7})
        .to_numpy()
        + rng.normal(0, 0.6, n)
    )
    churn = rng.random(n) < _sigmoid(score + _calibrate_intercept(score, 0.22))
    reg = REFERENCE_DATE - pd.to_timedelta(tenure * 30 + rng.integers(0, 28, n), unit="D")
    gap_days = rng.integers(5, 120, n)
    disconnect = pd.Series(reg + pd.to_timedelta(tenure * 30 + gap_days, unit="D"))
    disconnect = disconnect.dt.strftime("%Y-%m-%d").where(churn)
    df = pd.DataFrame(
        {
            "customer_id": [f"LKC{100000 + i}" for i in range(n)],
            "district": district,
            "province": province,
            "age": np.clip(rng.normal(37, 13, n), 18, 80).round().astype(int),
            "gender": rng.choice(["Male", "Female"], n),
            "language_preference": _languages(province, rng),
            "connection_type": connection,
            "contract_type": contract,
            "tenure_months": tenure,
            "monthly_charge_lkr": charge,
            "data_usage_gb": np.round(
                rng.gamma(2, 6, n) * np.where(connection == "Mobile Prepaid", 0.5, 1.0), 1
            ),
            "voice_minutes": rng.gamma(2, 120, n).round().astype(int),
            "sms_count": rng.poisson(40, n),
            "num_complaints_6m": complaints,
            "network_issues_reported": rng.poisson(1.2, n),
            "avg_signal_quality": signal,
            "payment_method": rng.choice(
                ["Cash", "Card", "Mobile wallet", "Bank transfer", "Reload card"],
                n,
                p=[0.25, 0.15, 0.25, 0.1, 0.25],
            ),
            "late_payments_12m": late,
            "has_idd_pack": _yes_no_messy(rng.random(n) < 0.15, rng),
            "has_streaming_bundle": _yes_no_messy(rng.random(n) < 0.25, rng),
            "registration_date": pd.Series(reg).dt.strftime("%Y-%m-%d"),
            "days_since_last_recharge": recharge_gap,
            "disconnection_date": disconnect,
            "churned": np.where(churn, "Yes", "No"),
        }
    )
    return _make_messy_telecom(df, rng)


def _make_messy_telecom(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["district"] = _messy_text(df["district"], rng)
    outliers = rng.random(len(df)) < 0.005
    df.loc[outliers, "data_usage_gb"] = df["data_usage_gb"].max() * rng.uniform(8, 20)
    charge = df["monthly_charge_lkr"].astype(float)
    text_mask = rng.random(len(df)) < 0.15
    df["monthly_charge_lkr"] = charge.astype("object")
    df.loc[text_mask, "monthly_charge_lkr"] = charge[text_mask].map(lambda v: f"Rs. {int(v):,}")
    _inject_nan(df, ["age", "avg_signal_quality", "data_usage_gb", "payment_method"], 0.05, rng)
    _inject_tokens(df, ["voice_minutes", "gender"], 0.01, rng)
    return _add_duplicates(df, 0.01, rng)


def generate_patient_los(seed: int, n: int = 2970) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    diag_effect = {
        "Dengue": 0.55,
        "Diabetes complications": 0.9,
        "Cardiac": 1.0,
        "Respiratory": 0.8,
        "CKD/CKDu": 1.1,
        "Trauma": 0.85,
        "Maternal": 0.3,
        "Other": 0.4,
    }
    diag = rng.choice(list(diag_effect), n, p=[0.2, 0.14, 0.14, 0.13, 0.1, 0.1, 0.1, 0.09])
    age = np.clip(rng.normal(48, 19, n), 1, 95).round().astype(int)
    comorb = np.clip(rng.poisson(1.1, n) + (age > 60), 0, 6)
    admission = rng.choice(["Emergency", "Elective", "Referral"], n, p=[0.5, 0.25, 0.25])
    platelets = np.where(diag == "Dengue", rng.normal(90, 45, n), rng.normal(260, 70, n)).clip(
        10, 600
    )
    creat = np.where(diag == "CKD/CKDu", rng.normal(3.2, 1.2, n), rng.normal(1.0, 0.3, n)).clip(
        0.3, 9
    )
    mu = (
        pd.Series(diag).map(diag_effect).to_numpy()
        + 0.012 * age
        + 0.12 * comorb
        + pd.Series(admission)
        .map({"Emergency": 0.25, "Elective": -0.1, "Referral": 0.15})
        .to_numpy()
        + 0.25 * (platelets < 80)
        + 0.1 * creat
    )
    los = np.clip(np.round(np.exp(mu + rng.normal(0, 0.4, n)) * 0.9), 1, 60).astype(int)
    ward = rng.choice(["General", "Semi-private", "Private"], n, p=[0.75, 0.15, 0.1])
    mult = pd.Series(ward).map({"General": 1.0, "Semi-private": 1.1, "Private": 1.25}).to_numpy()
    admit = REFERENCE_DATE - pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    bmi = np.clip(rng.normal(24.5, 4.5, n), 13, 50).round(1)
    ckd = np.where(diag == "CKD/CKDu", rng.integers(1, 6, n), np.nan)
    df = pd.DataFrame(
        {
            "patient_id": [f"LKP{200000 + i}" for i in range(n)],
            "hospital_type": rng.choice(
                [
                    "Teaching",
                    "Provincial General",
                    "Base",
                    "District General",
                    "Divisional",
                    "Private",
                ],
                n,
                p=[0.2, 0.18, 0.18, 0.2, 0.12, 0.12],
            ),
            "district": rng.choice(DISTRICTS, n, p=DISTRICT_WEIGHTS),
            "age": age,
            "sex": rng.choice(["Male", "Female"], n),
            "bmi": bmi,
            "admission_type": admission,
            "diagnosis_group": diag,
            "comorbidity_count": comorb,
            "diabetes": _yes_no_messy(rng.random(n) < (0.12 + 0.002 * age), rng),
            "hypertension": _yes_no_messy(rng.random(n) < (0.1 + 0.004 * age), rng),
            "ckd_stage": ckd,
            "systolic_bp": np.clip(rng.normal(125, 18, n), 70, 220).round(),
            "fasting_blood_sugar_mgdl": np.clip(rng.normal(110, 35, n), 55, 400).round(),
            "haemoglobin_gdl": np.clip(rng.normal(12.2, 2, n), 5, 18).round(1),
            "platelet_count_k": platelets.round(),
            "wbc_k": np.clip(rng.normal(8, 3, n), 1.5, 35).round(1),
            "serum_creatinine_mgdl": creat.round(2),
            "admission_date": pd.Series(admit).dt.strftime("%Y-%m-%d"),
            "ward_type": ward,
            "total_bill_lkr": (8500 * los * mult * rng.normal(1, 0.03, n)).round(0),
            "length_of_stay_days": los,
        }
    )
    return _make_messy_patient(df, rng)


def _make_messy_patient(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["district"] = _messy_text(df["district"], rng)
    df["hospital_type"] = _messy_text(df["hospital_type"], rng)
    outliers = rng.random(len(df)) < 0.004
    df.loc[outliers, "systolic_bp"] = 300
    bill = df["total_bill_lkr"].astype("object")
    mask = rng.random(len(df)) < 0.12
    bill[mask] = df.loc[mask, "total_bill_lkr"].map(lambda v: f"Rs. {int(v):,}")
    df["total_bill_lkr"] = bill
    _inject_nan(df, ["bmi", "haemoglobin_gdl", "fasting_blood_sugar_mgdl", "sex"], 0.06, rng)
    _inject_tokens(df, ["wbc_k", "hospital_type"], 0.01, rng)
    return _add_duplicates(df, 0.01, rng)


CHURN_CARD = {
    "key": "lk_telecom_churn",
    "title": "Sri Lankan telecom customer churn (synthetic)",
    "problem_type": "classification",
    "target": "churned",
    "provenance": "Synthetic Sri Lankan-context data generated by scripts/generate_samples.py (seed 42)",
    "is_synthetic": True,
    "description": "One row per customer of a fictional Sri Lankan telecom operator, with usage, "
    "billing, complaint and contract information, labelled by whether the customer churned.",
    "context": "Uses the 25 districts and 9 provinces of Sri Lanka, LKR charges, and Sinhala/"
    "Tamil/English language preference. No real operator or customer is represented.",
    "license": "CC0-1.0 (synthetic)",
    "collection_period": "Synthetic; observation date 2024-12-31",
    "features": {
        "customer_id": "Unique customer identifier (should be excluded)",
        "district": "District of registration (25 districts, messy case)",
        "province": "Province derived from district",
        "age": "Customer age in years",
        "gender": "Male / Female",
        "language_preference": "Sinhala, Tamil or English",
        "connection_type": "Product type (Mobile Prepaid, Mobile Postpaid, Fibre, 4G Home "
        "Broadband, ADSL)",
        "contract_type": "No contract, 12-month or 24-month",
        "tenure_months": "Months since registration",
        "monthly_charge_lkr": "Monthly charge in LKR (some values stored as 'Rs. 1,250')",
        "data_usage_gb": "Monthly data usage in GB",
        "voice_minutes": "Monthly voice minutes",
        "sms_count": "Monthly SMS count",
        "num_complaints_6m": "Complaints in last 6 months",
        "network_issues_reported": "Network issues reported",
        "avg_signal_quality": "Average signal quality, 1 (poor) to 5 (excellent)",
        "payment_method": "Cash, Card, Mobile wallet, Bank transfer or Reload card",
        "late_payments_12m": "Late payments in the last 12 months",
        "has_idd_pack": "International calling pack (Yes/No, inconsistent spelling)",
        "has_streaming_bundle": "Streaming bundle (Yes/No, inconsistent spelling)",
        "registration_date": "Date the customer registered",
        "days_since_last_recharge": "Days since last recharge or payment",
        "disconnection_date": "Date of disconnection; only present for churners (LEAKAGE)",
        "churned": "Target: Yes if the customer churned",
    },
    "target_definition": "churned = 'Yes' if the customer left the operator within the "
    "observation window.",
    "known_limitations": [
        "Synthetic data: relationships are simulated, not measured.",
        "disconnection_date is a planted leakage column and must be excluded.",
        "Results do not describe any real Sri Lankan operator.",
    ],
    "anonymisation": "Not applicable; no real individuals are represented.",
    "recommended_exclusions": ["customer_id", "disconnection_date"],
}

LOS_CARD = {
    "key": "lk_patient_los",
    "title": "Sri Lankan hospital length of stay (synthetic)",
    "problem_type": "regression",
    "target": "length_of_stay_days",
    "provenance": "Synthetic Sri Lankan-context data generated by scripts/generate_samples.py (seed 42)",
    "is_synthetic": True,
    "description": "One row per hospital admission with demographics, diagnosis group and "
    "admission-time lab values, with the length of stay in days as the target.",
    "context": "Uses Sri Lankan districts, hospital tiers (Teaching, Provincial General, Base, "
    "District General, Divisional, Private), LKR billing and diagnoses common locally such as "
    "dengue and CKD/CKDu. No real hospital or patient is represented.",
    "license": "CC0-1.0 (synthetic)",
    "collection_period": "Synthetic; admissions 2023-01 to 2024-12",
    "features": {
        "patient_id": "Unique admission identifier (should be excluded)",
        "hospital_type": "Hospital tier",
        "district": "District of the hospital",
        "age": "Age in years",
        "sex": "Male / Female",
        "bmi": "Body mass index (kg/m2)",
        "admission_type": "Emergency, Elective or Referral",
        "diagnosis_group": "Primary diagnosis group",
        "comorbidity_count": "Number of comorbid conditions",
        "diabetes": "Diabetes (Yes/No, inconsistent spelling)",
        "hypertension": "Hypertension (Yes/No, inconsistent spelling)",
        "ckd_stage": "CKD stage 1-5; missing when not CKD",
        "systolic_bp": "Systolic blood pressure (mmHg)",
        "fasting_blood_sugar_mgdl": "Fasting blood sugar (mg/dL)",
        "haemoglobin_gdl": "Haemoglobin (g/dL)",
        "platelet_count_k": "Platelet count (thousand per microlitre)",
        "wbc_k": "White cell count (thousand per microlitre)",
        "serum_creatinine_mgdl": "Serum creatinine (mg/dL)",
        "admission_date": "Date of admission",
        "ward_type": "General, Semi-private or Private",
        "total_bill_lkr": "Total bill in LKR, known only after discharge (LEAKAGE)",
        "length_of_stay_days": "Target: days in hospital",
    },
    "target_definition": "length_of_stay_days = whole days between admission and discharge.",
    "known_limitations": [
        "Synthetic data: relationships are simulated, not clinical findings.",
        "total_bill_lkr is a planted leakage column and must be excluded.",
        "Not suitable for any clinical decision making.",
    ],
    "anonymisation": "Not applicable; no real patients are represented.",
    "recommended_exclusions": ["patient_id", "total_bill_lkr"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/samples"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (generate_telecom_churn(args.seed), CHURN_CARD),
        (generate_patient_los(args.seed), LOS_CARD),
    ]
    for df, card in jobs:
        df.to_csv(args.out / f"{card['key']}.csv", index=False)
        (args.out / f"{card['key']}.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {card['key']}: {df.shape}")


if __name__ == "__main__":
    main()
