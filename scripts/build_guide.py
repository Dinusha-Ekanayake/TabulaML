"""Build docs/TabulaML_Guide.pdf (application overview and user guide) from HTML.

    python scripts/capture_screens.py      # refresh docs/images (app must be running)
    python scripts/build_guide.py          # writes docs/TabulaML_Guide.pdf

Plain, print-friendly layout: Arial, black text, simple table borders, figures with
captions. Rendered to A4 with Chromium (Playwright).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guide_content import body  # noqa: E402 - sits next to this script

CSS = """
@page { size: A4; margin: 22mm 20mm 22mm 20mm; }
* { box-sizing: border-box; }
body { font-family: Arial, Helvetica, sans-serif; color: #000; font-size: 10.5pt;
       line-height: 1.45; margin: 0; }
h1 { font-size: 22pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; margin: 22pt 0 8pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 16pt 0 6pt; page-break-after: avoid; }
p { margin: 0 0 8pt; text-align: left; }
ul, ol { margin: 0 0 8pt 18pt; padding: 0; }
li { margin-bottom: 3pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 12pt; font-size: 9.5pt;
        page-break-inside: auto; }
th, td { border: 0.6pt solid #000; padding: 4pt 6pt; vertical-align: top; text-align: left; }
th { font-weight: bold; background: none; }
tr { page-break-inside: avoid; }
figure { margin: 6pt 0 14pt; page-break-inside: avoid; }
figure img { width: 100%; border: 0.6pt solid #000; }
figure.phone img { width: 38%; }
figcaption { font-size: 9pt; margin-top: 4pt; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; }
pre { font-family: Consolas, "Courier New", monospace; font-size: 9pt; margin: 0 0 10pt;
      padding: 6pt 8pt; border: 0.6pt solid #000; white-space: pre-wrap; }
.cover { height: 240mm; display: flex; flex-direction: column; justify-content: center; }
.cover img { width: 70mm; margin-bottom: 16pt; }
.cover p { font-size: 11pt; margin: 2pt 0; }
.break { page-break-before: always; }
.toc li { list-style: none; margin-left: 0; }
.toc { margin-left: 0; }
"""


def build(out: Path, images: Path) -> Path:
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>TabulaML Guide</title>"
        f"<style>{CSS}</style></head><body>"
        + body(images.as_uri(), (ROOT / "tabulaml/ui/assets").as_uri(), date.today())
        + "</body></html>"
    )
    page_file = out.with_suffix(".html")
    page_file.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        browser = (
            p.chromium.launch(executable_path=_chromium()) if _chromium() else p.chromium.launch()
        )
        page = browser.new_page()
        page.goto(page_file.as_uri())
        page.wait_for_load_state("networkidle")
        page.pdf(
            path=str(out),
            format="A4",
            print_background=False,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=(
                "<div style='font-family:Arial;font-size:8pt;width:100%;text-align:center;"
                "color:#000'>TabulaML Guide, page <span class='pageNumber'></span> of "
                "<span class='totalPages'></span></div>"
            ),
            margin={"top": "20mm", "bottom": "20mm", "left": "18mm", "right": "18mm"},
        )
        browser.close()
    page_file.unlink()
    return out


def _chromium() -> str | None:
    local = Path.home() / "AppData/Local/ms-playwright/chromium-1243/chrome-win64/chrome.exe"
    return str(local) if local.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "TabulaML_Guide.pdf")
    parser.add_argument("--images", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args()
    print(build(args.out, args.images.resolve()))  # noqa: T201 - command-line output


if __name__ == "__main__":
    main()
