"""
Multi-season determinant model: what predicts a big market-value rise?

Builds a player-season panel from the Kaggle Transfermarkt dump (all seasons
where appearances exist, ~2012-2025), then trains a gradient-boosting classifier
to predict whether a player's value rises >= `rise_pct` over the next year, using
only information available at the end of that season (no leakage).

Reports:
  * out-of-time validation (train on early seasons, test on recent ones)
  * feature importances = the determinants of value growth
  * a `riser_prob` that can rank current players

Phase A = basic stats only (minutes/goals/assists) + age + position + value.
Advanced FBref metrics (xG, progression, defensive) are a later enrichment.

    python3 -m src.value_model
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RISE_PCT = 50.0        # "riser" = forward 1yr value change >= this
TRAIN_MAX_YEAR = 2022  # temporal split: train <= this, test after


def build_panel(data_dir: Path) -> pd.DataFrame:
    # --- seasonal value + forward change ---
    pv = pd.read_csv(data_dir / "player_valuations.csv",
                     usecols=["player_id", "date", "market_value_in_eur"])
    pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
    pv = pv.dropna(subset=["date"])
    pv["year"] = pv["date"].dt.year
    yr = (pv.sort_values("date").groupby(["player_id", "year"])["market_value_in_eur"]
          .last().reset_index().sort_values(["player_id", "year"]))
    yr["prev"] = yr.groupby("player_id")["market_value_in_eur"].shift(1)
    yr["prev_year"] = yr.groupby("player_id")["year"].shift(1)
    yr["next"] = yr.groupby("player_id")["market_value_in_eur"].shift(-1)
    yr["next_year"] = yr.groupby("player_id")["year"].shift(-1)
    yr = yr[yr["next_year"] == yr["year"] + 1].copy()
    yr["fwd_change_pct"] = (yr["next"] / yr["market_value_in_eur"] - 1) * 100
    yr["riser"] = (yr["fwd_change_pct"] >= RISE_PCT).astype(int)
    yr["log_value"] = np.log1p(yr["market_value_in_eur"])
    yr["value_momentum"] = np.where(yr["prev_year"] == yr["year"] - 1,
                                    (yr["market_value_in_eur"] / yr["prev"] - 1) * 100, np.nan)

    # --- seasonal basic performance ---
    ap = pd.read_csv(data_dir / "appearances.csv",
                     usecols=["player_id", "date", "minutes_played", "goals", "assists"])
    ap["date"] = pd.to_datetime(ap["date"], errors="coerce")
    ap["year"] = ap["date"].dt.year
    apy = ap.groupby(["player_id", "year"]).agg(
        minutes=("minutes_played", "sum"), goals=("goals", "sum"),
        assists=("assists", "sum"), apps=("minutes_played", "size")).reset_index()
    apy["nineties"] = apy["minutes"] / 90.0
    apy["goals_p90"] = apy["goals"] / apy["nineties"].replace(0, np.nan)
    apy["assists_p90"] = apy["assists"] / apy["nineties"].replace(0, np.nan)

    # --- static player attributes ---
    pl = pd.read_csv(data_dir / "players.csv",
                     usecols=["player_id", "date_of_birth", "sub_position",
                              "position", "height_in_cm", "foot"])
    pl["dob"] = pd.to_datetime(pl["date_of_birth"], errors="coerce")

    panel = yr.merge(apy, on=["player_id", "year"], how="inner").merge(pl, on="player_id", how="left")
    panel["age"] = panel["year"] - panel["dob"].dt.year
    panel["age2"] = panel["age"] ** 2
    panel = panel[(panel["age"].between(15, 40)) & (panel["minutes"] >= 450)]
    return panel


FEATURES = ["age", "age2", "minutes", "apps", "goals_p90", "assists_p90",
            "height_in_cm", "log_value", "value_momentum"]
CAT = ["sub_position", "position", "foot"]


def _prep(panel: pd.DataFrame):
    X = panel[FEATURES].copy()
    for c in CAT:
        X[c] = panel[c].astype("category")
    y = panel["riser"]
    return X, y


def train(panel: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    tr = panel[panel["year"] <= TRAIN_MAX_YEAR]
    te = panel[panel["year"] > TRAIN_MAX_YEAR]
    Xtr, ytr = _prep(tr)
    Xte, yte = _prep(te)

    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=400,
        categorical_features=[Xtr.columns.get_loc(c) for c in CAT],
        l2_regularization=1.0, random_state=0)
    model.fit(Xtr, ytr)

    p = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    base = yte.mean()

    # lift at the top decile of predicted risers
    te2 = te.assign(p=p).sort_values("p", ascending=False)
    k = max(1, len(te2) // 10)
    top_rate = te2.head(k)["riser"].mean()

    print("=" * 64)
    print(f"VALUE-RISE MODEL  (riser = value up >= {RISE_PCT:g}% next year)")
    print("=" * 64)
    print(f"Panel: {len(panel):,} player-seasons  ({panel.year.min()}-{panel.year.max()})")
    print(f"Train <= {TRAIN_MAX_YEAR}: {len(tr):,}   Test > {TRAIN_MAX_YEAR}: {len(te):,}")
    print(f"Base riser rate (test): {base:.1%}")
    print(f"Out-of-time ROC-AUC:    {auc:.3f}   (0.5 = no skill)")
    print(f"Top-decile riser rate:  {top_rate:.1%}   ({top_rate/base:.2f}x base)")

    imp = permutation_importance(model, Xte, yte, n_repeats=5, random_state=0,
                                 scoring="roc_auc")
    order = np.argsort(imp.importances_mean)[::-1]
    print("\nDeterminants of value growth (permutation importance, AUC drop):")
    for i in order:
        print(f"    {Xtr.columns[i]:16s} {imp.importances_mean[i]:+.4f}")
    return model


def main() -> None:
    data_dir = ROOT / "data"
    panel = build_panel(data_dir)
    out = ROOT / "data" / "processed" / "value_panel.parquet"
    panel.to_parquet(out, index=False)
    print(f"panel -> {out}\n")
    train(panel)


if __name__ == "__main__":
    main()
