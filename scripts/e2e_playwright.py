"""End-to-end browser check of the running app (Playwright, Chromium).

Usage (the app must be running, e.g. `streamlit run app.py`):

    python scripts/e2e_playwright.py --url http://localhost:8501 --out reports/e2e

It walks through every problem type (classification, several targets, regression,
forecasting) at desktop and phone widths in light and dark theme, and after every step
checks that there is no exception, no "Something went wrong" box and no horizontal
overflow. Screenshots go to ``--out``. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

logger = logging.getLogger("e2e")
TRAIN_TIMEOUT_MS = 600_000
STEP_TIMEOUT_MS = 120_000
VIEWPORTS = {"desktop": (1400, 900), "phone": (390, 844)}


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    checks: int = 0

    def check(self, ok: bool, what: str) -> None:
        self.checks += 1
        if not ok:
            self.failures.append(what)
            logger.error("FAIL: %s", what)


_STATE = "document.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')"


def settle(page: Page, seconds: float = 0.2) -> None:
    """Wait for Streamlit's re-run to START (if it does) and then to FINISH."""
    time.sleep(seconds)
    try:
        page.wait_for_function(f"() => {_STATE} === 'running'", timeout=1500)
    except PlaywrightTimeout:
        pass  # the run was too quick to observe, or nothing re-ran
    page.wait_for_function(f"() => {_STATE} === 'notRunning'", timeout=TRAIN_TIMEOUT_MS)
    time.sleep(0.3)


def go_next(page: Page, heading: str) -> None:
    """Press Next and wait until the next step's heading is shown."""
    for _ in range(3):  # a click during a re-run can be dropped by Streamlit: retry
        click(page, "Next →")
        try:
            page.get_by_role("heading", name=heading).wait_for(timeout=15_000)
            return
        except PlaywrightTimeout:
            continue
    raise RuntimeError(f"Could not reach '{heading}'.")


def audit(page: Page, report: Report, label: str, out: Path) -> None:
    exceptions = page.locator('[data-testid="stException"]').count()
    broken = page.get_by_text("Something went wrong").count()
    overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
    report.check(exceptions == 0, f"{label}: {exceptions} exception box(es)")
    report.check(broken == 0, f"{label}: 'Something went wrong' shown")
    report.check(overflow <= 1, f"{label}: page scrolls sideways by {overflow}px")
    page.screenshot(path=str(out / f"{label}.png"), full_page=True)


def click(page: Page, name: str, exact: bool = False) -> None:
    """Click the first button whose label contains ``name``."""
    page.get_by_role("button", name=name, exact=exact).first.click()
    settle(page)


def toggle_model(page: Page, key: str) -> None:
    page.locator(f".st-key-card_{key} label").first.click()
    settle(page, 0.8)


def set_theme(page: Page, theme: str) -> None:
    page.locator('[data-testid="stMainMenu"]').first.click()
    time.sleep(0.6)
    page.locator(f'[data-testid="stMainMenuItem-theme-{theme}"]').click()
    time.sleep(0.6)
    page.keyboard.press("Escape")
    settle(page, 0.5)


def answer_leaks(page: Page) -> None:
    page.get_by_role("tab", name="Data quality report").wait_for(timeout=STEP_TIMEOUT_MS)
    settle(page)
    tab = page.get_by_role("tab", name="Leakage review")
    if tab.count():
        tab.first.click()
        settle(page, 0.8)
        for option in page.get_by_text("No: keep it excluded").all():
            option.click()
            settle(page, 0.8)


def choose(page: Page, label: str, option: str) -> None:
    """Pick ``option`` (a substring) in the Streamlit selectbox / multiselect called ``label``."""
    box = page.locator('[data-testid="stSelectbox"], [data-testid="stMultiSelect"]')
    box.filter(has_text=label).first.locator("input").first.click()
    page.get_by_role("option").filter(has_text=option).first.click()
    settle(page, 0.8)


def load_sample(page: Page, problem: str, title_part: str | None) -> None:
    click(page, f"Choose {problem}")
    page.get_by_role("tab", name="Use a Sri Lankan sample dataset").click()
    settle(page, 0.8)
    if title_part:
        choose(page, "Sample dataset", title_part)
    click(page, "Load this sample")


def train_and_finish(page: Page, report: Report, label: str, out: Path) -> None:
    audit(page, report, f"{label}-4-models", out)
    click(page, "Train & Predict")
    page.get_by_role("heading", name="5. Results").wait_for(timeout=TRAIN_TIMEOUT_MS)
    settle(page, 2)
    audit(page, report, f"{label}-5-results", out)
    for tab in ("Random Forest dashboard", "Model details (CV)", "Final test"):
        found = page.get_by_role("tab", name=tab)
        if found.count():
            found.first.click()
            settle(page, 1)
    reveal = page.get_by_role("button", name="Reveal final test results")
    if reveal.count():
        reveal.first.click()
        settle(page, 2)
    audit(page, report, f"{label}-5-test", out)
    click(page, "Use a model")
    page.get_by_role("heading", name="6. Use a model").wait_for(timeout=STEP_TIMEOUT_MS)
    audit(page, report, f"{label}-6-predict", out)


def classification(page: Page, report: Report, label: str, out: Path) -> None:
    load_sample(page, "classification", "Tea Auction")
    audit(page, report, f"{label}-2-data", out)
    go_next(page, "3. Data quality and preprocessing")
    answer_leaks(page)
    audit(page, report, f"{label}-3-preprocessing", out)
    go_next(page, "4. Choose models")
    for key in ("dt", "rf"):
        toggle_model(page, key)
    train_and_finish(page, report, label, out)
    page.get_by_role("button", name="Predict", exact=True).first.click()
    settle(page, 2)
    audit(page, report, f"{label}-6-single", out)


def several_targets(page: Page, report: Report, label: str, out: Path) -> None:
    load_sample(page, "classification", "Tea Auction")
    page.get_by_text("Predict several columns at once").click()
    settle(page, 1)
    choose(page, "Additional target columns", "tier")
    page.keyboard.press("Escape")
    settle(page, 1.5)
    audit(page, report, f"{label}-2-data", out)
    go_next(page, "3. Data quality and preprocessing")
    answer_leaks(page)
    go_next(page, "4. Choose models")
    toggle_model(page, "dt")
    train_and_finish(page, report, label, out)


def regression(page: Page, report: Report, label: str, out: Path) -> None:
    load_sample(page, "regression", "dengue")
    go_next(page, "3. Data quality and preprocessing")
    audit(page, report, f"{label}-3-preprocessing", out)
    go_next(page, "4. Choose models")
    for key in ("ridge", "rf", "xgb"):
        toggle_model(page, key)
    train_and_finish(page, report, label, out)


def forecasting(page: Page, report: Report, label: str, out: Path) -> None:
    load_sample(page, "forecasting", None)
    audit(page, report, f"{label}-2-data", out)
    go_next(page, "3. Data quality and preprocessing")
    go_next(page, "4. Choose models")
    for key in ("naive", "ridge", "rf_arima"):
        toggle_model(page, key)
    train_and_finish(page, report, label, out)
    click(page, "Forecast the next")
    audit(page, report, f"{label}-6-forecast", out)


SCENARIOS: dict[str, Callable[[Page, Report, str, Path], None]] = {
    "classification": classification,
    "several-targets": several_targets,
    "regression": regression,
    "forecasting": forecasting,
}


def run(url: str, out: Path, exe: str | None, scenarios: list[str]) -> Report:
    report = Report()
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        for name in scenarios:
            for view, (width, height) in VIEWPORTS.items():
                theme = "Dark" if view == "phone" else "Light"
                label = f"{name}-{view}-{theme.lower()}"
                logger.info("Scenario %s", label)
                page = browser.new_page(viewport={"width": width, "height": height})
                try:
                    page.goto(url)
                    page.get_by_text("What do you want to predict?").wait_for(
                        timeout=STEP_TIMEOUT_MS
                    )
                    set_theme(page, theme)
                    audit(page, report, f"{label}-1-problem", out)
                    SCENARIOS[name](page, report, label, out)
                except Exception as exc:  # noqa: BLE001 - record and continue with the next scenario
                    report.check(False, f"{label}: {type(exc).__name__}: {str(exc)[:200]}")
                    page.screenshot(path=str(out / f"{label}-ERROR.png"), full_page=True)
                finally:
                    page.close()
        browser.close()
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default="reports/e2e", type=Path)
    parser.add_argument("--chromium", default=None, help="path to a Chromium/Chrome executable")
    parser.add_argument("--only", nargs="*", default=list(SCENARIOS), choices=list(SCENARIOS))
    args = parser.parse_args()
    report = run(args.url, args.out, args.chromium, args.only)
    logger.info("%d checks, %d failures", report.checks, len(report.failures))
    for failure in report.failures:
        logger.info("  - %s", failure)
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
