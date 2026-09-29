"""
End-to-end pipeline entry point.

    python -m src.pull_data     # 1) fetch raw data (needs network) - run once
    python -m src.run           # 2) build table, engineer, score, export

Reads cached parquet from data/raw, writes a ranked shortlist to
data/processed/shortlist.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .build_base_table import build
from .config import Config
from .features import add_all_features
from .merge_market_value import maybe_attach
from .signals import add_minutes_share, apply_filters, score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SHORTLIST_COLS = [
    "player", "team", "league", "position", "position_group", "sub_position",
    "age", "minutes", "minutes_share", "market_value_m", "contract_expires",
    "scout_score", "sig_performance", "sig_intensity",
    "sig_underused", "sig_misprofiling",
]


def load_raw(season: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    from .paths import resolve_raw
    fbref = pd.read_parquet(resolve_raw(season))
    tm_path = RAW / "transfermarkt_players.parquet"
    tm = pd.read_parquet(tm_path) if tm_path.exists() else None
    return fbref, tm


def run(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config.load()
    fbref, tm = load_raw(cfg.season)

    df = build(fbref, tm, cfg)
    df = add_minutes_share(df)

    # Attach market value from a user-provided CSV (see merge_market_value.py).
    mv_csv = cfg.raw.get("market_value_csv")
    df = maybe_attach(df, (ROOT / mv_csv) if mv_csv else None)

    # Normalise any pre-existing market_value column into millions.
    if "market_value" in df.columns and "market_value_m" not in df.columns:
        df["market_value_m"] = pd.to_numeric(df["market_value"], errors="coerce") / 1e6

    df = add_all_features(df)
    df = apply_filters(df, cfg.filters)
    ranked = score(df, cfg.weights)

    cols = [c for c in SHORTLIST_COLS if c in ranked.columns]
    shortlist = ranked[cols]
    out = cfg.out_path / "shortlist.csv"
    shortlist.to_csv(out, index=False)
    print(f"Wrote {len(shortlist):,} eligible players -> {out}")
    print(shortlist.head(20).to_string(index=False))
    return ranked


if __name__ == "__main__":
    run()
