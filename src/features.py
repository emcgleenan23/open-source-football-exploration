"""
Feature engineering: turn raw counts into comparable, normalised features.

All normalisation is done *within a player's positional peer group* so a
full-back is never compared to a striker. This mirrors the scouting literature's
role-based / multidisciplinary profiling (Bergkamp et al. 2022; Barraclough
et al. 2025) and the valuation literature's emphasis on position as a
first-order determinant (Poli et al. 2022; Franceschi et al. 2023).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Metric families used by the signals. Names follow FBref's flattened columns;
# missing ones are ignored gracefully so the pipeline survives schema drift.
# Canonical names produced by build_base_table.harmonise_columns(). Any that are
# absent in a given pull (e.g. xG for non-advanced leagues) are skipped safely.
PERFORMANCE_METRICS = [
    "goals", "assists", "shots", "shots_on_target", "key_passes", "sca", "gca",
    "xg", "npxg", "xa",  # used if the pull includes the Expected columns
]
INTENSITY_METRICS = [
    "tackles", "interceptions", "blocks", "tackles+interceptions",
    "clearances", "recoveries", "progressive_carries", "carries_final_third",
    "take_ons",
]
PROFILE_METRICS = [  # used to detect misprofiling via role clustering
    "sca", "gca", "key_passes", "progressive_passes", "progressive_carries",
    "carries_final_third", "tackles", "interceptions", "touches_att_pen",
    "touches_def_pen", "crosses", "xa",
]


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def to_per90(df: pd.DataFrame, metrics: list[str], minutes_col: str = "nineties") -> pd.DataFrame:
    """Divide raw season totals by 90s played to get per-90 rates.

    Every metric in PERFORMANCE/INTENSITY/PROFILE is a counting total (goals,
    xG, tackles, carries, ...), never a pre-computed rate or percentage, so we
    always divide. This is deterministic and source-agnostic (works for both
    FBref totals and Understat totals). Values are coerced to numeric first
    because scraped tables arrive as strings.
    """
    def _num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )

    out = df.copy()
    nn = _num(out[minutes_col]).replace(0, np.nan)
    for m in _present(df, metrics):
        vals = _num(out[m])
        out[m] = vals
        out[f"{m}_p90"] = vals / nn
    return out


def zscore_within_group(df: pd.DataFrame, cols: list[str],
                        group_col: str = "position_group") -> pd.DataFrame:
    """Standardise each metric within positional peer group (mean 0, sd 1)."""
    out = df.copy()
    for c in _present(df, cols):
        g = out.groupby(group_col)[c]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        out[f"z_{c}"] = ((out[c] - mu) / sd).fillna(0.0)
    return out


def percentile_within_group(df: pd.DataFrame, cols: list[str],
                            group_col: str = "position_group") -> pd.DataFrame:
    out = df.copy()
    for c in _present(df, cols):
        out[f"pct_{c}"] = out.groupby(group_col)[c].rank(pct=True)
    return out


def zscore_global(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Standardise across the WHOLE pool (mean 0, sd 1), ignoring position.

    Used for the role/misprofiling model: a player's fingerprint must be
    absolute, not relative to the group they are (possibly wrongly) listed in.
    """
    out = df.copy()
    for c in _present(df, cols):
        mu, sd = out[c].mean(), out[c].std()
        out[f"zg_{c}"] = ((out[c] - mu) / (sd if sd else np.nan)).fillna(0.0)
    return out


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pass: per-90s + within-group z-scores for every family."""
    all_metrics = sorted(set(PERFORMANCE_METRICS + INTENSITY_METRICS + PROFILE_METRICS))
    out = to_per90(df, all_metrics)
    p90_cols = [f"{m}_p90" for m in all_metrics if f"{m}_p90" in out.columns]
    out = zscore_within_group(out, p90_cols)
    out = percentile_within_group(out, p90_cols)
    # Global (position-agnostic) z-scores for the profile/role model.
    profile_p90 = [f"{m}_p90" for m in PROFILE_METRICS if f"{m}_p90" in out.columns]
    out = zscore_global(out, profile_p90)
    return out
