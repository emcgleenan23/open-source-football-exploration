"""
Point-in-time market value + forward outcomes, keyed by Transfermarkt player_id.

  * value_asof(as_of)   - latest valuation on or before a date (no lookahead)
  * value change        - value_asof(later) / value_asof(as_of)
  * transfer outcome    - max real fee in transfers after the cutoff

Used both to give the 2024/25 ranking a season-accurate value (Phase 1) and to
build the outcome labels for the backtest (Phase 2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_valuations(data_dir: str | Path) -> pd.DataFrame:
    pv = pd.read_csv(Path(data_dir) / "player_valuations.csv",
                     usecols=["player_id", "date", "market_value_in_eur"])
    pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
    return pv.dropna(subset=["date"]).sort_values("date")


def value_asof(pv: pd.DataFrame, as_of: str) -> pd.Series:
    """player_id -> latest market value on/before `as_of`."""
    cut = pv[pv["date"] <= pd.Timestamp(as_of)]
    latest = cut.groupby("player_id")["market_value_in_eur"].last()
    return latest


def value_between(pv: pd.DataFrame, as_of: str, later: str) -> pd.DataFrame:
    """player_id -> value at as_of, value at later, pct change."""
    v0 = value_asof(pv, as_of).rename("value_asof")
    v1 = value_asof(pv, later).rename("value_later")
    out = pd.concat([v0, v1], axis=1)
    out["value_change_pct"] = (out["value_later"] / out["value_asof"] - 1.0) * 100
    return out


def transfer_outcomes(data_dir: str | Path, after: str) -> pd.DataFrame:
    """player_id -> max real transfer fee (EUR) after `after`, + destination."""
    tr = pd.read_csv(Path(data_dir) / "transfers.csv",
                     usecols=["player_id", "transfer_date", "transfer_fee", "to_club_name"])
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    tr["transfer_fee"] = pd.to_numeric(tr["transfer_fee"], errors="coerce")
    # Real, dated moves after the cutoff; ignore future-dated contract rows.
    tr = tr[(tr["transfer_date"] > pd.Timestamp(after))
            & (tr["transfer_date"] <= pd.Timestamp("2026-09-01"))
            & (tr["transfer_fee"] > 0)]
    if not len(tr):
        return pd.DataFrame(columns=["fee_eur", "to_club"]).rename_axis("player_id")
    idx = tr.groupby("player_id")["transfer_fee"].idxmax()
    best = tr.loc[idx].set_index("player_id")
    return best.rename(columns={"transfer_fee": "fee_eur", "to_club_name": "to_club"})[
        ["fee_eur", "to_club"]]


def attach_pointintime_value(df: pd.DataFrame, pv: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Set market_value_m from the as-of valuation (needs a player_id column)."""
    out = df.copy()
    if "player_id" not in out.columns:
        return out
    v = value_asof(pv, as_of)
    out["market_value_eur"] = out["player_id"].map(v)
    out["market_value_m"] = pd.to_numeric(out["market_value_eur"], errors="coerce") / 1e6
    return out


def attach_outcomes(df: pd.DataFrame, data_dir: str | Path,
                    as_of: str, later: str,
                    hit_fee_m: float = 10.0, hit_value_pct: float = 50.0) -> pd.DataFrame:
    """Add forward outcomes + a boolean `hit` (fee>=X OR value up>=Y%)."""
    out = df.copy()
    pv = load_valuations(data_dir)
    vb = value_between(pv, as_of, later)
    fees = transfer_outcomes(data_dir, as_of)

    out["value_asof_m"] = (out["player_id"].map(vb["value_asof"]) / 1e6)
    out["value_later_m"] = (out["player_id"].map(vb["value_later"]) / 1e6)
    out["value_change_pct"] = out["player_id"].map(vb["value_change_pct"])
    out["fee_eur_m"] = out["player_id"].map(fees["fee_eur"]) / 1e6
    out["to_club"] = out["player_id"].map(fees["to_club"])

    sold = out["fee_eur_m"].fillna(0) >= hit_fee_m
    rose = out["value_change_pct"].fillna(-999) >= hit_value_pct
    out["hit"] = (sold | rose).astype("boolean")
    # A player we couldn't resolve to an id has unknown outcome -> mark NA.
    out.loc[out["player_id"].isna(), "hit"] = pd.NA
    return out
