"""用无头 Chromium（Playwright）把 HTML 课程海报渲染成 PNG。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=ROOT / "assets/poster/routeep-poster.html")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/poster/routeep-poster.png")
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    with sync_playwright() as p:
        # Playwright 自带版本缺失时，允许使用预装的 Chromium。
        browser = p.chromium.launch(executable_path=os.environ.get("POSTER_CHROMIUM") or None)
        page = browser.new_page(viewport={"width": 1080, "height": 1620},
                                device_scale_factor=args.scale)
        page.goto(args.html.resolve().as_uri())
        page.evaluate("document.fonts.ready")
        overflow = page.evaluate(
            "document.querySelector('.page').scrollHeight - window.innerHeight")
        if overflow > 0:
            raise SystemExit(f"poster content overflows the canvas by {overflow}px")
        page.screenshot(path=str(args.output), full_page=False)
        browser.close()
    print(args.output)


if __name__ == "__main__":
    main()
