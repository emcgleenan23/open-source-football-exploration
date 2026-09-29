"""
Assemble the analysis base table: one clean row per player.

Merges FBref advanced stats with Transfermarkt value/age, normalises a few
column names, and attaches the positional peer group each player is scored in.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .config import Config

# FBref's flattened column names (over-header + stat) -> our canonical names.
# Verified against a real 10-league pull. Only columns that exist get renamed;
# xG-family names are included so they map automatically if a pull includes them.
FBREF_RENAME = {
    # identity / playing time
    "pos": "position",
    "squad": "team",
    "playing_time_min": "minutes",
    "playing_time_90s": "nineties",
    # output
    "performance_gls": "goals",
    "performance_ast": "assists",
    "standard_sh": "shots",
    "standard_sot": "shots_on_target",
    "kp": "key_passes",
    "sca_sca": "sca",
    "gca_gca": "gca",
    # xG family (present only for advanced-data leagues; mapped if there)
    "expected_xg": "xg",
    "expected_npxg": "npxg",
    "expected_xag": "xa",
    # defense / intensity
    "tackles_tkl": "tackles",
    "int": "interceptions",
    "blocks_blocks": "blocks",
    "tkl+int": "tackles+interceptions",
    "clr": "clearances",
    "rec": "recoveries",
    # possession / progression (distance columns proxy for the count columns,
    # which FBref sometimes drops; fine after within-group standardisation)
    "carries_prgdist": "progressive_carries",
    "total_prgdist": "progressive_passes",
    "carries_1/3": "carries_final_third",
    "take-ons_att": "take_ons",
    "take-ons_succ": "take_ons_succ",
    "touches_touches": "touches",
    "touches_att_pen": "touches_att_pen",
    "touches_def_pen": "touches_def_pen",
    "pass_types_crs": "crosses",
}


def harmonise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename FBref's raw flattened columns to canonical names, skipping any
    rename that would collide with an existing column."""
    mapping = {k: v for k, v in FBREF_RENAME.items()
               if k in df.columns and v not in df.columns}
    return df.rename(columns=mapping)


def _num(s: pd.Series) -> pd.Series:
    """Coerce a scraped column to numeric, stripping thousands separators."""
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _age_years(s: pd.Series) -> pd.Series:
    """FBref ages look like '27-123' (years-days). Keep the years part."""
    return pd.to_numeric(s.astype(str).str.split("-").str[0], errors="coerce")


def _norm_name(s: pd.Series) -> pd.Series:
    """Accent-fold + lowercase for a robust join key."""
    def one(x: str) -> str:
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z ]", "", x.lower()).strip()
    return s.map(one)


def assign_position_group(pos: str, groups: dict[str, list[str]]) -> str:
    """Map an FBref position string to one of our peer groups."""
    if not isinstance(pos, str):
        return "UNK"
    primary = pos.split(",")[0].strip().upper()
    for group, members in groups.items():
        members_up = [m.upper() for m in members]
        if primary in members_up or pos.upper() in members_up:
            return group
    # Fallbacks by common FBref codes
    fallback = {"DF": "CB", "MF": "MID", "FW": "FWD", "GK": "GK"}
    return fallback.get(primary, "UNK")


def build(fbref: pd.DataFrame, tm: pd.DataFrame | None, cfg: Config) -> pd.DataFrame:
    df = fbref.copy()

    # Map FBref's raw flattened names -> canonical names used downstream.
    df = harmonise_columns(df)

    # Coerce the core numeric columns (they arrive as strings, and FBref uses
    # '27-123' for age and '1,710' for minutes).
    if "minutes" in df.columns:
        df["minutes"] = _num(df["minutes"])
    if "nineties" in df.columns:
        df["nineties"] = _num(df["nineties"])
    if "age" in df.columns:
        df["age"] = _age_years(df["age"])

    df["_key"] = _norm_name(df["player"]) + "|" + _norm_name(df["team"])
    df["position_group"] = df["position"].map(
        lambda p: assign_position_group(p, cfg.position_groups)
    )

    if tm is not None and len(tm):
        tm = tm.copy()
        tm["_key"] = _norm_name(tm["player"]) + "|" + _norm_name(tm.get("team", ""))
        tm_cols = [c for c in ["age", "market_value", "contract_expires", "nationality"]
                   if c in tm.columns]
        df = df.merge(tm[["_key", *tm_cols]], on="_key", how="left",
                      suffixes=("", "_tm"))

    return df.drop(columns=["_key"]).reset_index(drop=True)
