"""
Browserless FBref pull — the reliable path.

FBref sits behind Cloudflare, which blocks plain `requests` at the TLS layer.
`curl_cffi` impersonates a real Chrome TLS fingerprint, so it gets through
WITHOUT Selenium/undetected-chromedriver and all their version pain.

FBref hides the per-player tables inside HTML comments; we strip the comment
markers, then parse with pandas. URL pattern confirmed from FBref itself, e.g.
    https://fbref.com/en/comps/13/2024-2025/stats/2024-2025-Ligue-1-Stats

Usage (drop-in alternative to pull_data.py):
    python -m src.pull_data_http
then continue with:
    python -m src.run
"""
from __future__ import annotations

import io
import random
import sys
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Comment

from .config import Config

FBREF = "https://fbref.com"

# league -> (FBref competition id, URL slug). IDs are stable on FBref.
COMPS: dict[str, tuple[int, str]] = {
    "ENG-Premier League": (9, "Premier-League"),
    "ITA-Serie A": (11, "Serie-A"),
    "ESP-La Liga": (12, "La-Liga"),
    "GER-Bundesliga": (20, "Bundesliga"),
    "FRA-Ligue 1": (13, "Ligue-1"),
    "ENG-Championship": (10, "Championship"),
    "POR-Primeira Liga": (32, "Primeira-Liga"),
    "NED-Eredivisie": (23, "Eredivisie"),
    "BRA-Serie A": (24, "Serie-A"),
    "ARG-Primera Division": (21, "Primera-Division"),
}

# (URL path segment, player-table id keyword). They differ for standard/playing time.
STAT_TYPES: list[tuple[str, str]] = [
    ("stats", "standard"),
    ("shooting", "shooting"),
    ("passing", "passing"),
    ("passing_types", "passing_types"),
    ("gca", "gca"),
    ("defense", "defense"),
    ("possession", "possession"),
    ("playingtime", "playing_time"),
    ("misc", "misc"),
]

RATE_SECONDS = 4.0  # polite crawl delay between requests


# --------------------------------------------------------------------------- #
# Pure parsing helpers (unit-tested offline in tests/test_http_parse.py)
# --------------------------------------------------------------------------- #
def uncomment_tables(html: str) -> BeautifulSoup:
    """Return soup with FBref's comment-hidden tables promoted to real nodes."""
    soup = BeautifulSoup(html, "lxml")
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            comment.replace_with(BeautifulSoup(comment, "lxml"))
    return soup


def extract_player_table(soup: BeautifulSoup, table_keyword: str) -> pd.DataFrame | None:
    """Pull the per-player stats table (id == 'stats_<keyword>') and tidy it."""
    table = soup.find("table", id=f"stats_{table_keyword}")
    if table is None:
        return None
    df = pd.read_html(io.StringIO(str(table)))[0]
    df = _flatten_and_clean(df)
    return df


def _flatten_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten the two-row MultiIndex header FBref uses.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(p) for p in col if p and not str(p).startswith("Unnamed"))
            .strip("_")
            for col in df.columns
        ]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    # Drop repeated mid-table header rows and the trailing 'matches' link column.
    if "player" in df.columns:
        df = df[df["player"] != "Player"]
    df = df[[c for c in df.columns if c != "matches"]]
    if "rk" in df.columns:
        df = df.drop(columns=["rk"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _session():
    """A Chrome-impersonating session, warmed up so Cloudflare hands us a
    clearance cookie before we touch any stat pages (this is what usually
    fixes FBref 403s)."""
    from curl_cffi import requests as cffi

    s = cffi.Session(impersonate="chrome")
    s.headers.update({
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fbref.com/",
    })
    try:
        r = s.get(FBREF + "/en/", timeout=30)
        print(f"  [warmup] fbref.com/en -> HTTP {r.status_code}")
    except Exception as exc:
        raise RuntimeError(f"Could not reach FBref at all: {exc}") from exc

    if r.status_code == 403:
        raise RuntimeError(
            "FBref returned 403 on warmup - this machine's IP is still flagged. "
            "Don't retry in a loop (that resets the timer). Wait ~30-60 min, then "
            "run `python -m src.diagnose_fetch` ONCE; only run the pull when that "
            "shows FBref = 200. A VPN / different network clears it immediately."
        )
    if r.status_code != 200:
        raise RuntimeError(f"FBref warmup returned HTTP {r.status_code}; aborting.")

    time.sleep(RATE_SECONDS)
    return s


def _fetch(session, url: str) -> str:
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    time.sleep(RATE_SECONDS + random.random())
    return r.text


def _league_url(comp_id: int, season: str, slug: str, path: str) -> str:
    return f"{FBREF}/en/comps/{comp_id}/{season}/{path}/{season}-{slug}-Stats"


def pull_league(session, league: str, season: str) -> pd.DataFrame:
    """All stat blocks for one league-season, merged on player identity."""
    if league not in COMPS:
        print(f"  [skip] no FBref id mapped for {league}", file=sys.stderr)
        return pd.DataFrame()
    comp_id, slug = COMPS[league]

    frames = []
    for path, keyword in STAT_TYPES:
        url = _league_url(comp_id, season, slug, path)
        try:
            soup = uncomment_tables(_fetch(session, url))
            tbl = extract_player_table(soup, keyword)
            if tbl is not None and len(tbl):
                frames.append(tbl)
        except Exception as exc:
            print(f"  [{league}] {path} failed: {exc}", file=sys.stderr)

    return merge_stat_frames(frames, league)


def merge_stat_frames(frames: list[pd.DataFrame], league: str) -> pd.DataFrame:
    """Join the per-stat-type tables for one league on player identity.
    Shared by the curl_cffi and Selenium backends."""
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    key = [c for c in ("player", "squad", "nation", "pos") if c in merged.columns]
    for df in frames[1:]:
        new = [c for c in df.columns if c not in merged.columns or c in key]
        merged = merged.merge(df[new], on=key, how="outer", suffixes=("", "_dup"))
    merged["league"] = league
    return merged


def main() -> None:
    cfg = Config.load()
    raw_dir = Path(cfg.out_path).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = _session()
    parts, summary = [], []
    for leagues, season in cfg.league_groups():
        for lg in leagues:
            print(f"Fetching {lg} @ {season} ...")
            df = pull_league(session, lg, season)
            summary.append((lg, len(df)))
            if len(df):
                parts.append(df)

    if not parts:
        raise RuntimeError("No FBref data pulled - see errors above.")

    out = pd.concat(parts, ignore_index=True)
    from .paths import stamped_raw
    out.to_parquet(stamped_raw(cfg.season), index=False)
    print(f"\nsaved -> {stamped_raw(cfg.season).name}")

    print("=== per-league row counts ===")
    for lg, n in summary:
        print(f"  {lg:24s} {n:>5}")
    print(f"\nTotal: {len(out):,} player rows, {out.shape[1]} columns")
    print("Next: python -m src.run   (Transfermarkt value/age is optional - "
          "set max_market_value_m: null in config to skip)")


if __name__ == "__main__":
    main()
