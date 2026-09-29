"""
Attach market value + granular position + contract from a USER-PROVIDED
Transfermarkt-derived CSV (we do NOT scrape Transfermarkt; ToS + blocklisted).

Designed for the Kaggle `davidcariboo/player-scores` `players.csv`, which has:
  name, current_club_name, market_value_in_eur, sub_position, date_of_birth,
  contract_expiration_date, current_club_domestic_competition_id
but degrades gracefully to any CSV that at least has a name + value column.

Beyond the value filter, `sub_position` fixes FBref's coarse positions (FBref
only gives "DF"/"MF"/"FW", so full-backs were being scored against centre-backs).

Note: this dataset covers 8 of the 10 leagues - it lacks the English
Championship (GB2) and Argentine Primera (AR1N). Players there keep their FBref
position group and get no value (kept, since the filters are NaN-lenient).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

NAME_COLS = ["name", "player", "player_name", "pretty_name"]
VALUE_COLS = ["market_value_in_eur", "market_value_eur", "market_value", "value"]
TEAM_COLS = ["current_club_name", "club", "team", "squad", "current_club"]
SUBPOS_COLS = ["sub_position", "subposition"]
DOB_COLS = ["date_of_birth", "dob", "birth_date"]
CONTRACT_COLS = ["contract_expiration_date", "contract_expires", "contract_expiry"]
COMP_COLS = ["current_club_domestic_competition_id", "competition_id", "domestic_competition_id"]

# TM competition codes for our leagues (8 present in the Kaggle dump).
LEAGUE_CODES = {"GB1", "GB2", "IT1", "ES1", "L1", "FR1", "PO1", "NL1", "BRA1", "AR1N"}

# Granular TM sub_position -> our peer groups (config.yaml position_groups keys).
SUBPOSITION_GROUP = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Right-Back": "FB", "Left-Back": "FB",
    "Defensive Midfield": "MID", "Central Midfield": "MID", "Attacking Midfield": "MID",
    "Right Midfield": "WIDE", "Left Midfield": "WIDE",
    "Right Winger": "WIDE", "Left Winger": "WIDE",
    "Centre-Forward": "FWD", "Second Striker": "FWD",
}

AGE_REF = pd.Timestamp("2025-01-01")  # mid-season reference for a 2024-25 pull


def _norm(s: pd.Series) -> pd.Series:
    def one(x: str) -> str:
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z ]", "", x.lower()).strip()
    return s.map(one)


def _norm_team(s: pd.Series) -> pd.Series:
    """Like _norm but turns punctuation into spaces so 'Botafogo-RJ' -> two
    tokens, not one merged blob."""
    def one(x: str) -> str:
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
    return s.map(one)


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def load_market_values(path: str | Path, league_filter: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    name_c = _pick(raw, NAME_COLS)
    val_c = _pick(raw, VALUE_COLS)
    if not name_c or not val_c:
        raise ValueError(
            f"CSV needs a name column ({NAME_COLS}) and a value column ({VALUE_COLS}). "
            f"Found: {list(raw.columns)}"
        )
    team_c, sub_c = _pick(raw, TEAM_COLS), _pick(raw, SUBPOS_COLS)
    dob_c, con_c = _pick(raw, DOB_COLS), _pick(raw, CONTRACT_COLS)
    comp_c = _pick(raw, COMP_COLS)

    if league_filter and comp_c:
        keep = raw[comp_c].isin(LEAGUE_CODES)
        if keep.any():
            raw = raw[keep]

    out = pd.DataFrame({
        "_name": _norm(raw[name_c]),
        "_team": _norm_team(raw[team_c]) if team_c else "",
        "market_value_eur": pd.to_numeric(raw[val_c], errors="coerce"),
    })
    out["sub_position"] = raw[sub_c].values if sub_c else pd.NA
    if dob_c:
        dob = pd.to_datetime(raw[dob_c], errors="coerce")
        out["tm_age"] = ((AGE_REF - dob).dt.days / 365.25).round(1)
    if con_c:
        out["contract_expires"] = pd.to_datetime(raw[con_c], errors="coerce")

    out = out.dropna(subset=["market_value_eur"])
    # Keep the highest-valued row per name/team (usually the senior player).
    return out.sort_values("market_value_eur", ascending=False)


_TEAM_STOP = {"fc", "cf", "sc", "ac", "afc", "club", "de", "the", "and", "football",
              "calcio", "ssd", "spa", "aps", "cd", "ud", "rc", "sad", "association",
              "футбол", "fussball", "futebol", "sporting", "real", "united", "city"}


def _broad_cat_fbref(pos: str) -> str:
    """FBref position string -> GK/DF/MF/FW (broad category)."""
    if not isinstance(pos, str) or not pos:
        return ""
    first = pos.split(",")[0].strip().upper()
    return first if first in {"GK", "DF", "MF", "FW"} else ""


_SUBPOS_BROAD = {
    "GK": "GK", "CB": "DF", "FB": "DF", "MID": "MF", "WIDE": "FW", "FWD": "FW",
}


def _broad_cat_tm(sub_position: str) -> str:
    grp = SUBPOSITION_GROUP.get(sub_position, "")
    return _SUBPOS_BROAD.get(grp, "")


# Incompatible category pairs: a match spanning either of these is rejected.
def _compatible(fb: str, tm: str) -> bool:
    if not fb or not tm:
        return True  # unknown on one side -> don't block
    if (fb == "GK") != (tm == "GK"):
        return False  # never match a keeper to an outfielder
    if {fb, tm} == {"DF", "FW"}:
        return False  # never match a defender to a forward
    return True


def _team_tokens(t: str) -> set[str]:
    return {w for w in str(t).split() if w and w not in _TEAM_STOP}


def attach(base: pd.DataFrame, mv: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df = df.reset_index(drop=True)
    df["_bidx"] = range(len(df))
    df["_name"] = _norm(df["player"])
    df["_team"] = _norm_team(df["team"]) if "team" in df.columns else ""
    df["_fbcat"] = df.get("position", pd.Series([""] * len(df))).map(_broad_cat_fbref)

    m = mv.copy()
    m["_tmcat"] = m["sub_position"].map(_broad_cat_tm) if "sub_position" in m.columns else ""

    # Candidate matches on name, then score by team overlap + category compatibility.
    cand = df[["_bidx", "_name", "_team", "_fbcat"]].merge(m, on="_name", how="inner")
    if len(cand):
        cand["_ok"] = [_compatible(f, t) for f, t in zip(cand["_fbcat"], cand["_tmcat"])]
        cand = cand[cand["_ok"]]
        base_tokens = df.set_index("_bidx")["_team"].map(_team_tokens)
        tm_team_col = "_team_y" if "_team_y" in cand else "_team"
        cand["_overlap"] = [
            len(_team_tokens(tt) & base_tokens[b])
            for tt, b in zip(cand[tm_team_col], cand["_bidx"])
        ]
        # If the source carries team info, REQUIRE team-name overlap: a name-only
        # match on a mononym is too risky (grabs a same-named player elsewhere).
        # If it has no team data at all, fall back to best-value name match.
        source_has_team = m["_team"].astype(str).str.len().gt(0).any() if "_team" in m else False
        if source_has_team:
            cand = cand[cand["_overlap"] > 0]
        cand = cand.sort_values(["_overlap", "market_value_eur"], ascending=False)
        best = cand.drop_duplicates(subset="_bidx", keep="first").set_index("_bidx")
    else:
        best = pd.DataFrame().reindex(columns=m.columns)

    for col in ["market_value_eur", "sub_position", "tm_age", "contract_expires"]:
        if col in m.columns:
            df[col] = df["_bidx"].map(best[col]) if col in best.columns else pd.NA

    matched = df["market_value_eur"].notna().sum()
    print(f"  [transfermarkt] matched {matched:,}/{len(df):,} players "
          f"({matched/len(df):.0%}) for value/position")

    df["market_value_m"] = pd.to_numeric(df["market_value_eur"], errors="coerce") / 1e6

    # Refine positional peer group from granular sub_position (now category-safe).
    if "sub_position" in df.columns and "position_group" in df.columns:
        refined = df["sub_position"].map(SUBPOSITION_GROUP)
        df["position_group"] = refined.fillna(df["position_group"])

    if "tm_age" in df.columns and "age" in df.columns:
        df["age"] = df["age"].fillna(df["tm_age"])

    return df.drop(columns=["_bidx", "_name", "_team", "_fbcat"], errors="ignore")


def maybe_attach(base: pd.DataFrame, csv_path: str | Path | None) -> pd.DataFrame:
    if not csv_path:
        return base
    p = Path(csv_path)
    if not p.exists():
        print(f"  [transfermarkt] no CSV at {p} - skipping value/position enrichment")
        return base
    return attach(base, load_market_values(p))
