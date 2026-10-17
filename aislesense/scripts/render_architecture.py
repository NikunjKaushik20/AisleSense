"""Render docs/architecture.html -> docs/architecture.png and docs/architecture_transparent.png (2x, README and
slides; the transparent one drops the page background so it sits on any slide).

  python scripts/render_architecture.py

Needs Playwright and Microsoft Edge (or set AISLE_BROWSER=chromium after `playwright install chromium`).
The page loads IBM Plex from Google Fonts; offline it falls back to Segoe UI."""
from __future__ import annotations

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    channel = os.environ.get("AISLE_BROWSER", "msedge")
    with sync_playwright() as p:
        b = p.chromium.launch(channel=None if channel == "chromium" else channel, headless=True)
        pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=2)
        pg.goto((DOCS / "architecture.html").as_uri())
        pg.wait_for_function("window.drawn === true", timeout=30000)
        out = DOCS / "architecture.png"
        pg.locator("#canvas").screenshot(path=str(out))
        clear = DOCS / "architecture_transparent.png"
        pg.evaluate("document.body.style.background = 'transparent'")
        pg.locator("#canvas").screenshot(path=str(clear), omit_background=True)
        b.close()
    print(out)
    print(clear)


if __name__ == "__main__":
    main()
