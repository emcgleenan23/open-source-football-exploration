"""
Smoke + logic tests on a synthetic FBref-shaped frame.
Runs WITHOUT any network, so it works in CI or a locked sandbox.

    python -m pytest tests/ -q      (or just: python tests/test_pipeline.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_base_table import assign_position_group  # noqa: E402
from src.features import add_all_features  # noqa: E402
from src.signals import (  # noqa: E402
    add_minutes_share, apply_filters, misprofiling_signal, score, underused_signal,
)

RNG = np.random.default_rng(42)


def _synthetic(n_per_group: int = 40) -> pd.DataFrame:
    """Four role archetypes with distinct stat fingerprints + noise."""
    rows = []
    archetypes = {
        "FWD": dict(npxg=0.55, xa=0.15, sca=3.0, tackles=0.6, interceptions=0.4,
                    aerials_won=2.0, progressive_carries=2.5, key_passes=1.0),
        "MID": dict(npxg=0.12, xa=0.25, sca=3.5, tackles=2.2, interceptions=1.6,
                    aerials_won=0.8, progressive_carries=1.8, key_passes=1.8),
        "CB":  dict(npxg=0.05, xa=0.03, sca=0.6, tackles=1.6, interceptions=2.2,
                    aerials_won=3.5, progressive_carries=0.7, key_passes=0.3),
        "FB":  dict(npxg=0.08, xa=0.18, sca=2.0, tackles=2.4, interceptions=1.9,
                    aerials_won=1.2, progressive_carries=2.2, key_passes=1.2),
    }
    teams = [f"Team{i}" for i in range(10)]
    for grp, means in archetypes.items():
        for j in range(n_per_group):
            minutes = float(RNG.integers(300, 3200))
            row = dict(
                player=f"{grp}_{j}", team=RNG.choice(teams), league="TEST",
                position=grp, minutes=minutes, nineties=minutes / 90.0,
                age=float(RNG.integers(18, 33)),
                market_value=float(RNG.integers(1, 60)) * 1e6,
            )
            for stat, mu in means.items():
                row[stat] = max(0.0, RNG.normal(mu, mu * 0.35 + 0.05)) * (minutes / 90.0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_position_group_mapping():
    groups = {"CB": ["DF", "CB"], "MID": ["MF"], "FWD": ["FW"], "FB": ["FB"]}
    assert assign_position_group("DF", groups) == "CB"
    assert assign_position_group("FW,MF", groups) == "FWD"
    assert assign_position_group("MF", groups) == "MID"


def test_pipeline_runs_and_ranks():
    df = _synthetic()
    df["position_group"] = df["position"]
    df["market_value_m"] = df["market_value"] / 1e6
    df = add_minutes_share(df)
    df = add_all_features(df)

    # z-scores must exist and be ~centred within group
    assert "z_npxg_p90" in df.columns
    grp_means = df.groupby("position_group")["z_npxg_p90"].mean().abs()
    assert (grp_means < 1e-6).all(), "within-group z-score not centred"

    filtered = apply_filters(df, dict(min_minutes=450, max_age=24,
                                      max_market_value_m=20))
    assert len(filtered) > 0 and (filtered["age"] <= 24).all()

    weights = dict(performance=0.35, intensity=0.25, misprofiling=0.20, underused=0.20)
    ranked = score(filtered, weights)
    assert ranked["scout_score"].is_monotonic_decreasing
    assert ranked["scout_score"].between(0, 1).all()


def test_underused_prefers_low_minutes():
    df = _synthetic()
    df["position_group"] = df["position"]
    df = add_minutes_share(df)
    df = add_all_features(df)
    u = underused_signal(df)
    # among high performers, lower minute share => higher underused score
    corr = np.corrcoef(df["minutes_share"].fillna(1), u)[0, 1]
    assert corr < 0, f"underused signal should fall with minutes share (corr={corr:.2f})"


def test_misprofiling_flags_disguised_player():
    df = _synthetic()
    df["position_group"] = df["position"]
    # Relabel a clear MID-profile player as a CB -> should score high.
    mid_like = df[df["position"] == "MID"].index[0]
    df.loc[mid_like, "position_group"] = "CB"
    df = add_all_features(df)
    mis = misprofiling_signal(df)
    assert mis.loc[mid_like] > 0.4, f"disguised player not flagged ({mis.loc[mid_like]:.2f})"


if __name__ == "__main__":
    test_position_group_mapping()
    test_pipeline_runs_and_ranks()
    test_underused_prefers_low_minutes()
    test_misprofiling_flags_disguised_player()
    print("All tests passed.")
