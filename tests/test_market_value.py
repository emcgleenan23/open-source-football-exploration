"""Offline tests for the market-value merge (src/merge_market_value.py)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.merge_market_value import attach, load_market_values  # noqa: E402
from src.signals import apply_filters  # noqa: E402


def _write_csv(rows) -> str:
    # Mimic the Kaggle players.csv schema (name / current_club_name / market_value_in_eur).
    fd = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
    pd.DataFrame(rows).to_csv(fd.name, index=False)
    return fd.name


def test_load_and_match_name_and_accents():
    csv = _write_csv([
        {"name": "Nuno Tavares", "current_club_name": "Lazio",
         "market_value_in_eur": 18_000_000},
        {"name": "Vinícius Júnior", "current_club_name": "Real Madrid",
         "market_value_in_eur": 200_000_000},
        {"name": "Some Veteran", "current_club_name": "Old Club",
         "market_value_in_eur": 2_000_000},
    ])
    mv = load_market_values(csv)

    base = pd.DataFrame({
        "player": ["Nuno Tavares", "Vinicius Junior", "Unknown Kid"],  # accent-stripped
        "team": ["Lazio", "Real Madrid", "Small Team"],
    })
    out = attach(base, mv)

    assert out.loc[out["player"] == "Nuno Tavares", "market_value_m"].iloc[0] == 18.0
    # accent-folded name still matches
    assert out.loc[out["player"] == "Vinicius Junior", "market_value_m"].iloc[0] == 200.0
    # unmatched player -> NaN, not an error
    assert pd.isna(out.loc[out["player"] == "Unknown Kid", "market_value_m"].iloc[0])


def test_value_filter_applies_and_is_nan_lenient():
    csv = _write_csv([
        {"name": "Cheap Youngster", "market_value_in_eur": 5_000_000},
        {"name": "Expensive Star", "market_value_in_eur": 80_000_000},
    ])
    mv = load_market_values(csv)
    base = pd.DataFrame({
        "player": ["Cheap Youngster", "Expensive Star", "No Value Data"],
        "team": ["A", "B", "C"], "minutes": [900, 900, 900],
    })
    out = attach(base, mv)
    filt = apply_filters(out, {"min_minutes": 450, "max_market_value_m": 20})

    names = set(filt["player"])
    assert "Cheap Youngster" in names          # under ceiling -> kept
    assert "Expensive Star" not in names       # over ceiling -> dropped
    assert "No Value Data" in names            # unknown value -> NOT auto-dropped


if __name__ == "__main__":
    test_load_and_match_name_and_accents()
    test_value_filter_applies_and_is_nan_lenient()
    print("Market-value tests passed.")
