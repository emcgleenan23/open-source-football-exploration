"""
Plain-Selenium FBref pull — the most robust backend.

Uses vanilla `selenium.webdriver.Chrome`, NOT seleniumbase/undetected-chromedriver.
Selenium Manager (built into selenium >=4.6) auto-downloads the chromedriver that
matches your installed Chrome, which is why this avoids the uc-mode launch/version
problems entirely. A real browser also executes Cloudflare's JS challenge, so this
often succeeds where the curl_cffi backend gets a 403.

Approach adapted from working StackOverflow answers:
  https://stackoverflow.com/a/79757833  (grab #stats_<type> table -> read_html)
  https://stackoverflow.com/a/79757990  (dismiss the osano cookie overlay first)

One browser session is reused across all requests to keep Cloudflare cookies.

    python3 -m src.pull_data_selenium
then:
    python -m src.run
"""
from __future__ import annotations

import io
import random
import sys
import time
from pathlib import Path

import pandas as pd

from .config import Config
from .pull_data_http import (
    COMPS, RATE_SECONDS, STAT_TYPES, _flatten_and_clean, _league_url,
    merge_stat_frames,
)


def _make_driver(headless: bool):
    from selenium import webdriver

    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")  # modern headless; less detectable
    opts.add_argument("--window-size=1400,1000")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    # Selenium Manager resolves the matching chromedriver automatically.
    return webdriver.Chrome(options=opts)


def _dismiss_cookies(driver) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "button.osano-cm-button--type_denyAll")
            )
        ).click()
        print("  [cookies] dismissed consent banner")
    except Exception:
        pass  # banner may not appear


def _read_table(driver, keyword: str, timeout: int = 12) -> pd.DataFrame | None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        el = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, f"stats_{keyword}"))
        )
    except Exception:
        return None
    html = el.get_attribute("outerHTML")
    try:
        df = pd.read_html(io.StringIO(html))[0]
    except ValueError:
        return None
    return _flatten_and_clean(df)


def pull_league(driver, league: str, season: str) -> pd.DataFrame:
    if league not in COMPS:
        print(f"  [skip] no FBref id mapped for {league}", file=sys.stderr)
        return pd.DataFrame()
    comp_id, slug = COMPS[league]

    frames = []
    for path, keyword in STAT_TYPES:
        url = _league_url(comp_id, season, slug, path)
        try:
            driver.get(url)
            time.sleep(RATE_SECONDS + random.random())
            tbl = _read_table(driver, keyword)
            if tbl is not None and len(tbl):
                frames.append(tbl)
            else:
                print(f"  [{league}] {path}: table not found", file=sys.stderr)
        except Exception as exc:
            print(f"  [{league}] {path} failed: {exc}", file=sys.stderr)

    return merge_stat_frames(frames, league)


def main() -> None:
    cfg = Config.load()
    raw_dir = Path(cfg.out_path).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    driver = _make_driver(cfg.fbref_headless)
    parts, summary = [], []
    try:
        # Warm up + cookie consent on the homepage first.
        driver.get("https://fbref.com/en/")
        time.sleep(RATE_SECONDS)
        _dismiss_cookies(driver)

        for leagues, season in cfg.league_groups():
            for lg in leagues:
                print(f"Fetching {lg} @ {season} ...")
                df = pull_league(driver, lg, season)
                summary.append((lg, len(df)))
                if len(df):
                    parts.append(df)
    finally:
        driver.quit()

    if not parts:
        raise RuntimeError(
            "No FBref data pulled. If every table was 'not found', you may still "
            "be Cloudflare-flagged - try a VPN/other network, or run "
            "`python -m src.diagnose_fetch`."
        )

    out = pd.concat(parts, ignore_index=True)
    from .paths import stamped_raw
    out.to_parquet(stamped_raw(cfg.season), index=False)
    print(f"\nsaved -> {stamped_raw(cfg.season).name}")

    print("=== per-league row counts ===")
    for lg, n in summary:
        print(f"  {lg:24s} {n:>5}")
    print(f"\nTotal: {len(out):,} player rows, {out.shape[1]} columns")
    print("Next: python -m src.run")


if __name__ == "__main__":
    main()
