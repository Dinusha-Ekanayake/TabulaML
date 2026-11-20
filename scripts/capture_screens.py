"""Capture the screenshots used in docs/TabulaML_Guide.pdf (the app must be running).

python scripts/capture_screens.py --url http://localhost:8501 --out docs/images
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_playwright as e  # noqa: E402 - shared helpers for driving the app

WIDTH, HEIGHT = 1440, 900


TOP = "() => document.querySelector('[data-testid=stMain]')?.scrollTo(0, 0)"


def shot(page: Page, out: Path, name: str, top: bool = True) -> None:
    """Screenshot the viewport; by default from the top of the step (the page keeps its
    scroll position after navigation)."""
    if top:
        page.evaluate(TOP)
        time.sleep(0.4)
    page.screenshot(path=str(out / f"{name}.jpg"), type="jpeg", quality=88)


def show(page: Page, text: str) -> None:
    target = page.locator(f"text={text} >> visible=true").first
    target.evaluate("el => el.scrollIntoView({block: 'start'})")  # section at the top
    time.sleep(0.6)


def tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).first.click()
    e.settle(page)


def tabular_tour(page: Page, out: Path) -> None:
    shot(page, out, "01_problem_type")
    e.load_sample(page, "classification", "Tea Auction")
    shot(page, out, "02_data")
    show(page, "How should the model be tested?")
    shot(page, out, "03_validation", top=False)
    e.go_next(page, "3. Data quality and preprocessing")
    shot(page, out, "04_quality")
    tab(page, "Explore data")
    e.choose(page, "Column to explore", "price_mid_lkr")
    shot(page, out, "05_explore")
    e.answer_leaks(page)
    shot(page, out, "06_leakage")
    tab(page, "Preprocessing")
    page.get_by_text("Custom", exact=True).first.click()
    e.settle(page)
    show(page, "Rows and missing values")
    shot(page, out, "07_rows_missing", top=False)
    tab(page, "Pipeline preview")
    shot(page, out, "08_pipeline")
    e.go_next(page, "4. Choose models")
    for key in ("dt", "rf", "xgb"):
        e.toggle_model(page, key)
    shot(page, out, "09_models")
    page.get_by_text("Model settings (hyperparameters)").click()
    e.settle(page)
    page.get_by_role("tab", name="XGBoost").click()
    e.settle(page)
    page.locator("label >> text=Grid search >> visible=true").first.click()
    e.settle(page)
    show(page, "Settings to search")
    shot(page, out, "10_grid_search", top=False)
    page.locator("label >> text=Default >> visible=true").first.click()
    e.settle(page)
    show(page, "Estimated training time")
    shot(page, out, "11_estimate", top=False)
    e.click(page, "Train & Predict")
    time.sleep(2)
    shot(page, out, "12_training")
    page.get_by_role("heading", name="5. Results").wait_for(timeout=e.TRAIN_TIMEOUT_MS)
    e.settle(page, 1)
    shot(page, out, "13_leaderboard")
    tab(page, "Random Forest dashboard")
    shot(page, out, "14_rf_dashboard")
    tab(page, "Model details (CV)")
    shot(page, out, "15_diagnostics")
    tab(page, "Final test")
    e.click(page, "Reveal final test results")
    tab(page, "Final test")  # the reveal re-runs the page, which re-opens the first tab
    shot(page, out, "16_final_test")
    e.click(page, "Use a model")
    page.get_by_role("heading", name="6. Use a model").wait_for(timeout=e.STEP_TIMEOUT_MS)
    page.get_by_role("button", name="Predict", exact=True).first.click()
    e.settle(page, 1)
    shot(page, out, "17_predict")


def forecasting_tour(page: Page, out: Path) -> None:
    e.load_sample(page, "forecasting", None)
    show(page, "Forecast settings")
    shot(page, out, "18_forecast_settings", top=False)
    e.go_next(page, "3. Data quality and preprocessing")
    e.go_next(page, "4. Choose models")
    for key in ("naive", "rf", "rf_arima"):
        e.toggle_model(page, key)
    e.click(page, "Train & Predict")
    page.get_by_role("heading", name="5. Results").wait_for(timeout=e.TRAIN_TIMEOUT_MS)
    e.settle(page, 1)
    tab(page, "Final test")
    e.click(page, "Reveal final test results")
    tab(page, "Final test")
    show(page, "/Held-out test period/")
    shot(page, out, "19_forecast_test", top=False)
    e.click(page, "Use a model")
    page.get_by_role("heading", name="6. Use a model").wait_for(timeout=e.STEP_TIMEOUT_MS)
    e.click(page, "Forecast the next")
    show(page, "History and forecast")
    shot(page, out, "20_forecast_future", top=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default="docs/images", type=Path)
    parser.add_argument("--chromium", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = (
            p.chromium.launch(executable_path=args.chromium)
            if args.chromium
            else p.chromium.launch()
        )
        for tour in (tabular_tour, forecasting_tour):
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            page.goto(args.url)
            page.get_by_role("heading", name="1. What do you want to predict?").wait_for(
                timeout=e.STEP_TIMEOUT_MS
            )
            tour(page, args.out)
            page.close()
        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        phone.goto(args.url)
        phone.get_by_role("heading", name="1. What do you want to predict?").wait_for(
            timeout=e.STEP_TIMEOUT_MS
        )
        e.set_theme(phone, "Dark")
        shot(phone, args.out, "21_phone_dark")
        browser.close()


if __name__ == "__main__":
    main()
