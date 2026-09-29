"""
The four scouting signals, plus the eligibility gate and composite score.

Design notes (mapped to the uploaded literature):

* performance   - within-position standardised output. Poli et al. (2022) and
                  the Franceschi (2023) review both rank on-pitch performance and
                  position as first-order value determinants.
* intensity     - pressing / duel / carrying volume per 90, an open-data proxy
                  for the physical "can handle intense games" requirement (true
                  GPS/high-speed-running data is mostly proprietary).
* misprofiling  - a player whose statistical fingerprint clusters with a role
                  other than the one they are listed in. Directly motivated by
                  the role-based / multidisciplinary profiling in Bergkamp et al.
                  (2022) and Barraclough et al. (2025): listed position often
                  mis-describes actual function.
* underused     - strong per-90 output on a low share of team minutes. The
                  value literature treats minutes/experience as a determinant,
                  so productive players with low minutes are systematically
                  under-priced.

Every function is pure pandas so it is unit-testable without a live pull.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import INTENSITY_METRICS, PERFORMANCE_METRICS, PROFILE_METRICS


# --------------------------------------------------------------------------- #
# Eligibility gate
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Keep only players that pass the age / minutes / value gate."""
    out = df.copy()
    mask = pd.Series(True, index=out.index)

    # min_minutes is a hard gate: a row with unknown minutes is unreliable, drop it.
    if "minutes" in out.columns and filters.get("min_minutes"):
        mask &= out["minutes"] >= filters["min_minutes"]

    # The remaining filters are NaN-lenient: an unknown value should not silently
    # eliminate a player (that was the bug that emptied the whole shortlist).
    if "minutes_share" in out.columns and filters.get("max_minutes_share"):
        s = out["minutes_share"]
        mask &= (s <= filters["max_minutes_share"]) | s.isna()

    if "age" in out.columns and filters.get("max_age"):
        a = out["age"]
        mask &= (a <= filters["max_age"]) | a.isna()

    if "market_value_m" in out.columns and filters.get("max_market_value_m") is not None:
        v = out["market_value_m"]
        mask &= (v <= filters["max_market_value_m"]) | v.isna()

    return out[mask].reset_index(drop=True)


def add_minutes_share(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the maximum minutes played in the player's own team.
    A cheap proxy for 'nailed-on starter vs rotation'."""
    out = df.copy()
    if "minutes" not in out.columns or "team" not in out.columns:
        out["minutes_share"] = np.nan
        return out
    team_max = out.groupby("team")["minutes"].transform("max").replace(0, np.nan)
    out["minutes_share"] = out["minutes"] / team_max
    return out


# --------------------------------------------------------------------------- #
# Signals -> each returns a 0..1 score column
# --------------------------------------------------------------------------- #
def _mean_z(df: pd.DataFrame, metrics: list[str]) -> pd.Series:
    cols = [f"z_{m}_p90" for m in metrics if f"z_{m}_p90" in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].mean(axis=1)


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-x))


def performance_signal(df: pd.DataFrame) -> pd.Series:
    return _sigmoid(_mean_z(df, PERFORMANCE_METRICS))


def intensity_signal(df: pd.DataFrame) -> pd.Series:
    return _sigmoid(_mean_z(df, INTENSITY_METRICS))


def underused_signal(df: pd.DataFrame) -> pd.Series:
    """High per-90 output * low minutes share. Both terms in 0..1."""
    perf = performance_signal(df)
    share = df.get("minutes_share")
    if share is None:
        return pd.Series(0.0, index=df.index)
    scarcity = (1.0 - share.clip(0, 1)).fillna(0.0)
    return perf * scarcity


def misprofiling_signal(df: pd.DataFrame, position_groups: list[str] | None = None) -> pd.Series:
    """
    Fit a light role model on the PROFILE features and flag players whose
    profile best matches a group other than their listed one.

    Returns the (soft) probability mass assigned to the *best non-listed*
    role - high => the player plays like something other than their label.
    Falls back to 0 if scikit-learn or the profile columns are unavailable.
    """
    # Use GLOBAL (position-agnostic) z-scores so a player's fingerprint is
    # absolute, not relative to the group they are listed in.
    prof_cols = [f"zg_{m}_p90" for m in PROFILE_METRICS if f"zg_{m}_p90" in df.columns]
    if len(prof_cols) < 3:  # fall back to within-group if global not present
        prof_cols = [f"z_{m}_p90" for m in PROFILE_METRICS if f"z_{m}_p90" in df.columns]
    if len(prof_cols) < 3 or "position_group" not in df.columns:
        return pd.Series(0.0, index=df.index)

    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except Exception:
        return pd.Series(0.0, index=df.index)

    work = df[df["position_group"].isin(
        df["position_group"].value_counts().loc[lambda s: s >= 5].index
    )].copy()
    if work["position_group"].nunique() < 2:
        return pd.Series(0.0, index=df.index)

    X = work[prof_cols].fillna(0.0).values
    y = work["position_group"].values
    try:
        lda = LinearDiscriminantAnalysis().fit(X, y)
        proba = pd.DataFrame(lda.predict_proba(X), columns=lda.classes_, index=work.index)
    except Exception:
        return pd.Series(0.0, index=df.index)

    listed = work["position_group"]
    own = proba.to_numpy()[np.arange(len(work)),
                           [list(lda.classes_).index(g) for g in listed]]
    best_other = proba.copy()
    for i, g in zip(work.index, listed):
        if g in best_other.columns:
            best_other.loc[i, g] = 0.0
    score = best_other.max(axis=1)  # confidence in an alternative role
    return score.reindex(df.index).fillna(0.0)


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def score(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["sig_performance"] = performance_signal(out)
    out["sig_intensity"] = intensity_signal(out)
    out["sig_underused"] = underused_signal(out)
    out["sig_misprofiling"] = misprofiling_signal(out)

    w = weights
    out["scout_score"] = (
        w.get("performance", 0) * out["sig_performance"]
        + w.get("intensity", 0) * out["sig_intensity"]
        + w.get("underused", 0) * out["sig_underused"]
        + w.get("misprofiling", 0) * out["sig_misprofiling"]
    )
    return out.sort_values("scout_score", ascending=False).reset_index(drop=True)
