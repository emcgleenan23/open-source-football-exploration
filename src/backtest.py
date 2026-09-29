"""
Point-in-time backtest: does the 2024/25 scout_score predict who broke out?

Pipeline:
  1. Rank 2024/25 players using ONLY info available by `as_of` (June 2025):
     FBref 24/25 stats + market value as-of June 2025 + granular position.
  2. Attach forward outcomes measured strictly after the cutoff:
     transfer fee (2025/26 windows) and market-value change to `later`.
  3. A "hit" = fee >= hit_fee_m OR value up >= hit_value_pct.
  4. Report whether higher scout_score => higher hit rate (lift, precision@k).

    python -m src.backtest            # uses config + data/ CSVs

No lookahead: value_asof and the ranking use only <= as_of; outcomes use > as_of.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .build_base_table import build
from .config import Config
from .features import add_all_features
from .identity import resolve_player_ids
from .merge_market_value import SUBPOSITION_GROUP
from .signals import add_minutes_share, apply_filters, score
from .valuations import attach_outcomes, attach_pointintime_value, load_valuations

ROOT = Path(__file__).resolve().parents[1]


def _refine_positions(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Use players.csv sub_position (via player_id) to fix the peer groups."""
    players = pd.read_csv(data_dir / "players.csv", usecols=["player_id", "sub_position"])
    sub = players.set_index("player_id")["sub_position"]
    df["sub_position"] = df["player_id"].map(sub)
    refined = df["sub_position"].map(SUBPOSITION_GROUP)
    df["position_group"] = refined.fillna(df["position_group"])
    return df


def rank_pointintime(cfg: Config, data_dir: Path, fbref: pd.DataFrame,
                     as_of: str) -> pd.DataFrame:
    """Score a season using ONLY info available at `as_of`: FBref stats +
    market value as-of that date + granular position. Shared by backtest and
    the forward run so both rank identically."""
    df = build(fbref, None, cfg)
    df = add_minutes_share(df)
    df["player_id"] = resolve_player_ids(df, data_dir)
    df = _refine_positions(df, data_dir)
    pv = load_valuations(data_dir)
    df = attach_pointintime_value(df, pv, as_of)
    df = add_all_features(df)
    elig = apply_filters(df, cfg.filters)
    return score(elig, cfg.weights)


def run_backtest(cfg: Config | None = None, data_dir: str | Path | None = None,
                 season="2024-2025", as_of="2025-06-30", later="2026-06-30",
                 hit_fee_m=10.0, hit_value_pct=50.0) -> pd.DataFrame:
    cfg = cfg or Config.load()
    data_dir = Path(data_dir or (ROOT / cfg.raw.get("kaggle_data_dir", "data")))
    from .paths import resolve_raw

    fbref = pd.read_parquet(resolve_raw(season))
    ranked = rank_pointintime(cfg, data_dir, fbref, as_of)

    # Forward outcomes (strictly after the cutoff).
    ranked = attach_outcomes(ranked, data_dir, as_of, later, hit_fee_m, hit_value_pct)

    _report(ranked, hit_fee_m, hit_value_pct)
    out = ROOT / cfg.output_dir / "backtest_2024_25.csv"
    cols = ["player", "team", "league", "position_group", "sub_position", "age",
            "minutes", "market_value_m", "scout_score", "sig_performance",
            "sig_intensity", "sig_underused", "sig_misprofiling",
            "value_asof_m", "value_later_m", "value_change_pct", "fee_eur_m",
            "to_club", "hit"]
    ranked[[c for c in cols if c in ranked.columns]].to_csv(out, index=False)
    print(f"\nFull backtest board -> {out}")
    return ranked


def _report(ranked: pd.DataFrame, hit_fee_m, hit_value_pct) -> None:
    lab = ranked.dropna(subset=["hit"]).copy()
    lab["hit"] = lab["hit"].astype(bool)
    n, base = len(lab), lab["hit"].mean()
    print("=" * 66)
    print(f"BACKTEST  (hit = sold >= €{hit_fee_m:g}m  OR  value up >= {hit_value_pct:g}%)")
    print("=" * 66)
    print(f"Eligible players with a known outcome: {n:,}")
    print(f"Base hit rate (all of them):           {base:.1%}")

    lab = lab.sort_values("scout_score", ascending=False).reset_index(drop=True)
    print("\n  top-K by scout_score   hit-rate    lift vs base")
    for k in (20, 50, 100, 200):
        if k <= n:
            hr = lab.head(k)["hit"].mean()
            print(f"    top {k:<4d}             {hr:6.1%}      {hr/base:4.2f}x")

    # Rank-based AUC (probability a random hit outranks a random miss).
    order = lab["scout_score"].rank()
    pos, neg = lab["hit"].sum(), (~lab["hit"]).sum()
    auc = (order[lab["hit"]].sum() - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else float("nan")
    print(f"\n  ROC-AUC (score vs hit): {auc:.3f}   (0.5 = no skill)")
    print("\n  Highest-scoring hits (model's best calls that paid off):")
    top_hits = lab[lab["hit"]].head(10)
    for _, r in top_hits.iterrows():
        fee = f"€{r['fee_eur_m']:.0f}m fee" if pd.notna(r.get("fee_eur_m")) else ""
        chg = f"+{r['value_change_pct']:.0f}% value" if pd.notna(r.get("value_change_pct")) else ""
        print(f"    {r['player'][:22]:22s} {str(r['team'])[:16]:16s} "
              f"score {r['scout_score']:.3f}  {fee} {chg}")


if __name__ == "__main__":
    run_backtest()
