"""
Pull the raw scouting data.

Two sources, both open and both wrapped by `soccerdata`:
  * FBref (StatsBomb-powered)  -> advanced per-90 performance stats
  * Transfermarkt              -> market value, age, contract, minutes

IMPORTANT: soccerdata drives a headless browser and needs open outbound
network. Run this on your own machine (VS Code), NOT in a locked sandbox.
Data is cached under ~/soccerdata by default, so re-runs are cheap.

Usage:
    python -m src.pull_data                 # uses config.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import Config

# The FBref advanced stat blocks we want. soccerdata concatenates them for us.
FBREF_STAT_TYPES = [
    "standard",
    "shooting",
    "passing",
    "passing_types",
    "goal_shot_creation",
    "defense",
    "possession",
    "misc",
    "playing_time",
]


def _pull_fbref_group(leagues: list[str], season: str,
                      headless: bool = False) -> pd.DataFrame:
    """Advanced stats for one set of leagues sharing a season format.

    headless=False opens a real Chrome window, which is far more reliable at
    getting past FBref's Cloudflare challenge than headless mode.
    """
    import soccerdata as sd

    fb = sd.FBref(leagues=leagues, seasons=season, headless=headless)

    # soccerdata catches a failed Chrome launch and leaves _driver unset, which
    # otherwise surfaces later as a cryptic AttributeError. Fail fast instead.
    if not hasattr(fb, "_driver"):
        raise RuntimeError(
            "FBref's Chrome driver failed to launch (soccerdata swallowed the "
            "real error). Run `python -m src.check_browser` to see it and get "
            "the fix — usually a stale chromedriver vs your installed Chrome."
        )
    frames = []
    for stat in FBREF_STAT_TYPES:
        try:
            frames.append(fb.read_player_season_stats(stat_type=stat))
        except Exception as exc:  # a league may lack a given block
            print(f"  [fbref] {season} skipped {stat}: {exc}", file=sys.stderr)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for df in frames[1:]:
        new_cols = [c for c in df.columns if c not in merged.columns]
        merged = merged.join(df[new_cols], how="outer")
    return merged.reset_index()


def pull_fbref(cfg: Config) -> pd.DataFrame:
    """Pull every configured league group and stack into one table.

    European and South American leagues run on different calendars, so they are
    pulled with different season strings (see Config.league_groups).
    """
    parts = []
    for leagues, season in cfg.league_groups():
        print(f"  fetching {len(leagues)} leagues @ season {season} "
              f"(headless={cfg.fbref_headless}) ...")
        part = _pull_fbref_group(leagues, season, headless=cfg.fbref_headless)
        if len(part):
            parts.append(part)

    if not parts:
        raise RuntimeError("FBref returned no data - check network / league names.")

    out = pd.concat(parts, ignore_index=True)
    out.columns = _flatten_cols(out.columns)
    return out


def pull_transfermarkt(cfg: Config) -> pd.DataFrame | None:
    """Market value / age / contract. Returns None if the merge is disabled."""
    if cfg.filters.get("max_market_value_m") is None:
        print("  [tm] market-value filter disabled -> skipping Transfermarkt")
        return None
    import soccerdata as sd

    parts = []
    for leagues, season in cfg.league_groups():
        try:
            tm = sd.Transfermarkt(leagues=leagues, seasons=season)
            parts.append(tm.read_players().reset_index())
        except Exception as exc:
            print(f"  [tm] {season} failed: {exc}", file=sys.stderr)
    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df.columns = _flatten_cols(df.columns)
    keep = [c for c in df.columns if c in {
        "player", "team", "league", "age", "position",
        "market_value", "max_market_value", "contract_expires", "nationality",
    }]
    return df[keep] if keep else df


def _flatten_cols(cols) -> list[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if p not in ("", None)]
            out.append("_".join(parts).strip("_").lower().replace(" ", "_"))
        else:
            out.append(str(c).lower().replace(" ", "_"))
    return out


def main() -> None:
    cfg = Config.load()
    raw_dir = Path(cfg.out_path).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_leagues = len(cfg.leagues) + len(cfg.leagues_calendar)
    print(f"Pulling FBref for {n_leagues} leagues ...")
    fbref = pull_fbref(cfg)
    fbref.to_parquet(raw_dir / "fbref_players.parquet", index=False)
    print(f"  saved {len(fbref):,} player rows, {fbref.shape[1]} columns")

    tm = pull_transfermarkt(cfg)
    if tm is not None:
        tm.to_parquet(raw_dir / "transfermarkt_players.parquet", index=False)
        print(f"  saved {len(tm):,} Transfermarkt rows")

    print("Done. Next: python -m src.run")


if __name__ == "__main__":
    main()
