"""Season-aware locations for the raw FBref pulls.

Each season is stored as data/raw/fbref_players_{season}.parquet so the 2024/25
(backtest) and 2025/26 (forward) pulls coexist and never overwrite each other.
Falls back to the legacy unstamped file so existing pulls keep working.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def stamped_raw(season: str) -> Path:
    return RAW / f"fbref_players_{season}.parquet"


def resolve_raw(season: str) -> Path:
    """Prefer the season-stamped file; fall back to the legacy unstamped one."""
    s = stamped_raw(season)
    if s.exists():
        return s
    legacy = RAW / "fbref_players.parquet"
    return legacy
