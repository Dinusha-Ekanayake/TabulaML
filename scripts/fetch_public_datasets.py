"""Fetch and prepare the PUBLIC, genuine Sri Lankan sample datasets.

Usage:
    python scripts/fetch_public_datasets.py public   # dengue + tea (no login needed)
    python scripts/fetch_public_datasets.py dengue
    python scripts/fetch_public_datasets.py tea
    python scripts/fetch_public_datasets.py telco --kaggle-dir PATH   # folder with main.csv,
                                                                       # train_cxid.csv, test_cxid.csv
    python scripts/fetch_public_datasets.py telco                     # downloads via kagglehub
                                                                       # (needs Kaggle credentials)

Datasets
--------
lk_colombo_dengue  (regression, CC0)
    Weekly notified dengue cases in Colombo district with weather variables,
    2008-W52 to 2014-W21 (279 weeks). Source: R package 'colmozzie' by Thiyanga Talagala,
    https://github.com/thiyangt/colmozzie (CC0), compiled from the Weekly Epidemiological
    Reports of the Epidemiology Unit, Ministry of Health, Sri Lanka.

lk_tea_auction_catalogue (classification) and lk_tea_auction_price (regression), CC BY 4.0
    12,233 Colombo Tea Auction price records from 105 weekly Forbes & Walker broker reports
    (Nov 2023 - Mar 2026) with lagged Open-Meteo weather. Source: Hugging Face dataset
    colombo-tea-auction-prices/cta-price-prediction (paper: "Forecasting Weather-Driven Price
    Dynamics Across Sri Lankan Tea Market Catalogues", MERCon 2026). Both tasks share one file.

lk_telco_churn  (classification, CC0)
    "Real World Customer Churn Dataset in Telco Domain" by Lasal Jayawardena,
    https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn (CC0): anonymised daily
    usage of 65,005 customers of a large Sri Lankan operator, Jan-Mar 2023. Kaggle requires a
    (free) account to download, so this dataset is built locally from the downloaded files.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
COLMOZZIE_SHA = "c964aa4b6ac79c7536dbde16c9b9a3a8d34f6909"
COLMOZZIE_URL = (
    f"https://raw.githubusercontent.com/thiyangt/colmozzie/{COLMOZZIE_SHA}/data-raw/dengColombo.csv"
)
TEA_REVISION = "a671701409e385808823cc7d3b183b12893ea401"
TEA_URL = (
    "https://huggingface.co/datasets/colombo-tea-auction-prices/cta-price-prediction/"
    f"resolve/{TEA_REVISION}/colombo_tea_auction_dataset.csv"
)
TEA_FILE = "lk_tea_auction.csv"
TELCO_SAMPLE_ROWS = 15_000  # stratified sample keeps the file small and under the cell limit
SEED = 42


def _write(df: pd.DataFrame, card: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{card['key']}.csv", index=False)
    (OUT / f"{card['key']}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"wrote {card['key']}: {df.shape}")


def _download_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed, pinned URLs
        return pd.read_csv(io.BytesIO(resp.read()))


# --------------------------------------------------------------------------- tea auction

TEA_FEATURES = {
    "sale_id": "Weekly auction sale identifier (e.g. SALE_2023_42)",
    "catalogue": "Market catalogue: low_grown, high_grown, dust or off_grade",
    "category": "Tea category within the catalogue (e.g. bop, fbop, pekoe)",
    "elevation": "Elevation group (low_grown, high_grown, low, medium, high)",
    "segment": "Market segment (e.g. best_western, best_uva; often 'unknown')",
    "grade": "Leaf grade (e.g. bop, bopf, op; often 'unknown')",
    "tier": "Quality tier (select_best, best, below_best, others; often 'unknown')",
    "price_mid_lkr": "Mid auction price in LKR per kg ((low + high) / 2)",
    "region": "Growing region used for the weather join",
    "text_crop_change": "Crop intake direction parsed from the broker report text",
    "text_keywords": "Weather keywords parsed from the broker report text",
    "temperature_2m_mean_mean": "Mean temperature in the auction week (C)",
    "precipitation_sum_total_lag1": "Total rainfall 1 week before the auction (mm)",
    "sunshine_duration_total_lag1": "Total sunshine 1 week before (seconds)",
    "temperature_2m_mean_mean_lag1": "Mean temperature 1 week before (C)",
    "precipitation_sum_total_lag2": "Total rainfall 2 weeks before (mm)",
    "sunshine_duration_total_lag2": "Total sunshine 2 weeks before (seconds)",
    "temperature_2m_mean_mean_lag2": "Mean temperature 2 weeks before (C)",
    "precipitation_sum_total_lag3": "Total rainfall 3 weeks before (mm)",
    "sunshine_duration_total_lag3": "Total sunshine 3 weeks before (seconds)",
    "temperature_2m_mean_mean_lag3": "Mean temperature 3 weeks before (C)",
    "sale_number": "Sale number within the year",
    "sale_year": "Sale year",
    "sale_date_raw": "Auction date as printed in the report (free text)",
    "avg_weather_severity": "Composite weather severity score from the report text",
    "fx_usd": "USD/LKR exchange rate",
}
TEA_COMMON = {
    "provenance": "Public dataset: Hugging Face 'colombo-tea-auction-prices/cta-price-prediction' "
    f"(revision {TEA_REVISION[:7]}), extracted from 105 weekly Forbes & Walker Tea Brokers "
    "reports, with Open-Meteo weather. Paper: 'Forecasting Weather-Driven Price Dynamics "
    "Across Sri Lankan Tea Market Catalogues' (MERCon 2026).",
    "is_synthetic": False,
    "file": TEA_FILE,
    "context": "Genuine Sri Lankan market data: the Colombo Tea Auction, one of the world's "
    "largest tea auctions, with weather for the growing regions.",
    "license": "CC-BY-4.0 (attribution required)",
    "collection_period": "November 2023 to March 2026 (105 weekly sales)",
    "features": TEA_FEATURES,
    "anonymisation": "Market prices only; no personal data.",
}
TEA_CATALOGUE_CARD = {
    "key": "lk_tea_auction_catalogue",
    "title": "Colombo Tea Auction: market catalogue (real, public)",
    "problem_type": "classification",
    "target": "catalogue",
    "description": "One row per auction price quote. Predict which market catalogue (Low Grown, "
    "High Grown, Dust or Off-Grade) a lot belongs to from its price, grade, tier, region, "
    "exchange rate and weather.",
    "target_definition": "catalogue = the broker report's market catalogue for the lot.",
    "known_limitations": [
        "'category' determines the catalogue almost exactly (each category is listed in one "
        "catalogue), so TabulaML flags and excludes it as leakage.",
        "Many 'unknown' values in grade, tier and segment where the reports do not state them.",
        "Rows from the same weekly sale are correlated; random cross-validation is optimistic.",
    ],
    "recommended_exclusions": ["category"],
    **TEA_COMMON,
}
TEA_PRICE_CARD = {
    "key": "lk_tea_auction_price",
    "title": "Colombo Tea Auction: price in LKR (real, public)",
    "problem_type": "regression",
    "target": "price_mid_lkr",
    "description": "One row per auction price quote. Predict the mid auction price (LKR/kg) "
    "from catalogue, grade, tier, region, exchange rate and lagged weather.",
    "target_definition": "price_mid_lkr = (low price + high price) / 2 for the quote, in LKR "
    "per kg; 3.6% of rows have no price and are dropped.",
    "known_limitations": [
        "Rows from the same weekly sale are correlated; random cross-validation is optimistic "
        "compared with forecasting the next week.",
        "Many 'unknown' values in grade, tier and segment.",
    ],
    "recommended_exclusions": [],
    **TEA_COMMON,
}


def fetch_tea() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = _download_csv(TEA_URL)
    df.to_csv(OUT / TEA_FILE, index=False)
    for card in (TEA_CATALOGUE_CARD, TEA_PRICE_CARD):
        (OUT / f"{card['key']}.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"wrote {TEA_FILE}: {df.shape} (+ 2 dataset cards)")


# --------------------------------------------------------------------------- dengue


def build_dengue(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(
        columns={
            "Cases": "dengue_cases",
            "Year": "year",
            "Week": "week",
            "TEM": "avg_temp_c",
            "TMAX": "max_temp_c",
            "Tm": "min_temp_c",
            "SLP": "sea_level_pressure_hpa",
            "H": "relative_humidity_pct",
            "PP": "rainfall_mm",
            "VV": "visibility_km",
            "V": "wind_speed_kmh",
            "VM": "max_wind_kmh",
        }
    ).sort_values(["year", "week"], kind="stable")
    # Only PAST information is added, so every feature is known at prediction time.
    df["cases_last_week"] = df["dengue_cases"].shift(1)
    df["cases_2_weeks_ago"] = df["dengue_cases"].shift(2)
    df["rainfall_prev_4_weeks_mm"] = df["rainfall_mm"].shift(1).rolling(4, min_periods=1).sum()
    target = df.pop("dengue_cases")
    df["dengue_cases"] = target
    return df.reset_index(drop=True)


DENGUE_CARD = {
    "key": "lk_colombo_dengue",
    "title": "Colombo weekly dengue cases and weather (real, public)",
    "problem_type": "regression",
    "target": "dengue_cases",
    "provenance": "Public dataset: R package 'colmozzie' (Thiyanga Talagala), "
    "https://github.com/thiyangt/colmozzie, commit " + COLMOZZIE_SHA[:7] + ". Compiled from the "
    "Weekly Epidemiological Reports of the Epidemiology Unit, Ministry of Health, Sri Lanka.",
    "is_synthetic": False,
    "description": "One row per week for Colombo district: notified dengue cases with that "
    "week's weather. Lagged case counts and recent rainfall were added by "
    "scripts/fetch_public_datasets.py using past weeks only.",
    "context": "Genuine Sri Lankan public-health surveillance data (Colombo district).",
    "license": "CC0-1.0",
    "collection_period": "2008 week 52 to 2014 week 21 (279 weeks)",
    "features": {
        "year": "Calendar year",
        "week": "Week number within the year",
        "avg_temp_c": "Average temperature of the week (C)",
        "max_temp_c": "Maximum temperature of the week (C)",
        "min_temp_c": "Minimum temperature of the week (C)",
        "sea_level_pressure_hpa": "Sea level pressure",
        "relative_humidity_pct": "Relative humidity (%)",
        "rainfall_mm": "Precipitation amount (mm)",
        "visibility_km": "Mean visibility (km)",
        "wind_speed_kmh": "Mean wind speed (km/h)",
        "max_wind_kmh": "Maximum sustained wind speed (km/h)",
        "cases_last_week": "Notified cases in the previous week (derived)",
        "cases_2_weeks_ago": "Notified cases two weeks earlier (derived)",
        "rainfall_prev_4_weeks_mm": "Rainfall over the previous four weeks (derived)",
        "dengue_cases": "Target: notified dengue cases in the week",
    },
    "target_definition": "dengue_cases = total notified dengue cases in Colombo district "
    "for the week.",
    "known_limitations": [
        "Small (279 rows), single district.",
        "Weekly time series: random cross-validation is optimistic compared with a strict "
        "forward-in-time evaluation.",
        "The first two weeks have missing lag values (handled by imputation).",
    ],
    "anonymisation": "Aggregate counts only; no individual patient data.",
    "recommended_exclusions": [],
}


def fetch_dengue() -> None:
    _write(build_dengue(_download_csv(COLMOZZIE_URL)), DENGUE_CARD)


# --------------------------------------------------------------------------- telco


def _kaggle_dir(path: str | None) -> Path:
    if path:
        return Path(path)
    import kagglehub  # optional dependency, only needed for automatic download

    return Path(kagglehub.dataset_download("lasaljaywardena/real-world-churn"))


def _find(folder: Path, name: str) -> Path:
    matches = list(folder.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {folder}")
    return matches[0]


def _usage_features(main_csv: Path) -> pd.DataFrame:
    """Per customer and usage type: first-month totals/activity/trend + a leakage column."""
    days = [f"Day_{i}" for i in range(1, 91)]
    parts = []
    for chunk in pd.read_csv(
        main_csv, chunksize=50_000, usecols=["customer_id", "usage_type", *days]
    ):
        x = chunk[days].to_numpy(dtype="float64")
        parts.append(
            pd.DataFrame(
                {
                    "customer_id": chunk["customer_id"].to_numpy(),
                    "usage": chunk["usage_type"]
                    .str.replace("usage_", "")
                    .str.replace("_daily", ""),
                    "m1_total": x[:, :30].sum(axis=1),
                    "m1_active_days": (x[:, :30] > 0).sum(axis=1),
                    "m1_trend": x[:, 20:30].mean(axis=1) - x[:, :10].mean(axis=1),
                    "late_active": (x[:, 60:90] > 0).sum(axis=1),
                }
            )
        )
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(
        index="customer_id", columns="usage", values=["m1_total", "m1_active_days", "m1_trend"]
    )
    wide.columns = [f"{usage}_{stat}" for stat, usage in wide.columns]
    wide = wide[sorted(wide.columns)]
    # Days 61-90 happen AFTER the churn decision window: kept on purpose as a real-world
    # leakage example that TabulaML's leakage detector must flag.
    wide["active_days_61_90_any_service"] = long.groupby("customer_id")["late_active"].max()
    return wide.reset_index()


def build_telco(folder: Path) -> pd.DataFrame:
    labels = pd.concat(
        [pd.read_csv(_find(folder, "train_cxid.csv")), pd.read_csv(_find(folder, "test_cxid.csv"))]
    ).drop_duplicates("customer_id")
    features = _usage_features(_find(folder, "main.csv"))
    df = features.merge(labels, on="customer_id", how="inner")
    df["churn"] = np.where(df["churn"].astype(int) == 1, "Yes", "No")
    if len(df) > TELCO_SAMPLE_ROWS:
        df = df.groupby("churn", group_keys=False).sample(
            frac=TELCO_SAMPLE_ROWS / len(df), random_state=SEED
        )
    df["customer_id"] = df["customer_id"].str[:12]  # shorter, still unique, still anonymous
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


TELCO_CARD = {
    "key": "lk_telco_churn",
    "title": "Sri Lankan telco customer churn (real, public)",
    "problem_type": "classification",
    "target": "churn",
    "provenance": "Public dataset: 'Real World Customer Churn Dataset in Telco Domain' by Lasal "
    "Jayawardena, https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn. Features "
    "derived by scripts/fetch_public_datasets.py; stratified random sample of 15,000 customers.",
    "is_synthetic": False,
    "description": "One row per anonymised customer: first-month (days 1-30) usage totals, "
    "active days and trend for each of 12 services (apps, packs, call types), and whether "
    "the customer churned.",
    "context": "Genuine anonymised data from one of the largest telecom operators in Sri Lanka "
    "(January-March 2023), including Sri Lankan apps such as Helakuru.",
    "license": "CC0-1.0",
    "collection_period": "1 January to 31 March 2023",
    "features": {
        "customer_id": "Anonymised hashed customer ID (identifier, excluded)",
        "<service>_m1_total": "Total usage in days 1-30 (MB for apps, LKR for packs, minutes "
        "for calls)",
        "<service>_m1_active_days": "Days with any use in days 1-30",
        "<service>_m1_trend": "Average daily use in days 21-30 minus days 1-10",
        "active_days_61_90_any_service": "Days with any use in days 61-90 (LEAKAGE: observed "
        "after the churn window)",
        "churn": "Target: Yes if the customer left the operator",
    },
    "target_definition": "churn = 'Yes' if the customer churned during the observation period "
    "(author's label).",
    "known_limitations": [
        "Usage data only (no plan, price, complaints or demographics).",
        "One operator and one quarter.",
        "active_days_61_90_any_service is kept deliberately as a leakage example and must be "
        "excluded.",
    ],
    "anonymisation": "Hashed customer IDs only; no names, numbers, locations or demographics.",
    "recommended_exclusions": ["customer_id", "active_days_61_90_any_service"],
}


def fetch_telco(kaggle_dir: str | None) -> None:
    folder = _kaggle_dir(kaggle_dir or os.environ.get("TABULAML_KAGGLE_DIR"))
    _write(build_telco(folder), TELCO_CARD)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("dataset", choices=["dengue", "tea", "telco", "public", "all"])
    parser.add_argument("--kaggle-dir", help="folder containing the extracted Kaggle files")
    args = parser.parse_args()
    if args.dataset in ("tea", "public", "all"):
        fetch_tea()
    if args.dataset in ("dengue", "public", "all"):
        fetch_dengue()
    if args.dataset in ("telco", "all"):
        fetch_telco(args.kaggle_dir)


if __name__ == "__main__":
    main()
