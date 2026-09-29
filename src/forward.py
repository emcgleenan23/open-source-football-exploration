"""
Phase 3 - the forward run.

Applies the validated model to the LATEST season (2025/26) to name players who
fit the profile of the ones who broke out last year. Same point-in-time scoring
as the backtest, but valued as of today and with no outcome labels (the future
isn't known yet).

Prereqs:
  1. Pull 2025/26 FBref data first:
       - set  season: "2025-2026"  in config.yaml
       - run  python -m src.pull_data_selenium   (writes fbref_players_2025-2026.parquet)
  2. Kaggle CSVs in data/ (players.csv, player_valuations.csv, clubs.csv, appearances.csv).

    python -m src.forward
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import rank_pointintime
from .config import Config
from .paths import resolve_raw

ROOT = Path(__file__).resolve().parents[1]

SHORTLIST_COLS = [
    "player", "team", "league", "position_group", "sub_position", "age",
    "minutes", "minutes_share", "market_value_m", "scout_score",
    "sig_performance", "sig_intensity", "sig_underused", "sig_misprofiling",
]


def run_forward(cfg: Config | None = None, data_dir: str | Path | None = None,
                as_of: str = "2026-06-30", top: int = 50) -> pd.DataFrame:
    cfg = cfg or Config.load()
    data_dir = Path(data_dir or (ROOT / cfg.raw.get("kaggle_data_dir", "data")))

    fbref = pd.read_parquet(resolve_raw(cfg.season))
    ranked = rank_pointintime(cfg, data_dir, fbref, as_of)

    cols = [c for c in SHORTLIST_COLS if c in ranked.columns]
    out = ROOT / cfg.output_dir / f"forward_{cfg.season}.csv"
    ranked[cols].to_csv(out, index=False)

    print(f"Forward shortlist for {cfg.season} (valued as of {as_of})")
    print(f"Eligible: {len(ranked):,}  ->  {out}")
    print(f"\nTop {top}:")
    print(ranked[cols].head(top).to_string(index=False))
    return ranked


if __name__ == "__main__":
    run_forward()
