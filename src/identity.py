"""
Resolve FBref players -> Transfermarkt player_id.

The hard part is that a player's *current* club (players.csv) differs from their
2024/25 club once they transfer. We fix this by pooling every club a player_id
has been associated with, from three sources:
  * players.csv            - current club
  * appearances.csv        - the club they actually played for in 2024/25
  * player_valuations.csv  - the club on each dated valuation near the season
A FBref (name, team) row then resolves to a player_id if the name matches and the
team shares a token with ANY of that id's known clubs.

Reaches ~77% of the 10-league pull; the rest are leagues absent from the dataset
(Championship, Argentina) or spelling gaps.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

_STOP = {"fc", "cf", "sc", "ac", "afc", "club", "de", "the", "and", "football",
         "calcio", "ssd", "spa", "aps", "cd", "ud", "rc", "sad", "association",
         "fussball", "futebol", "sporting", "real", "united", "city", "sa", "fk",
         "ss", "as"}


def _nn(x) -> str:
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", x.lower()).strip()


def _toks(x) -> set[str]:
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return {w for w in re.sub(r"[^a-z0-9]+", " ", x.lower()).split()
            if w and w not in _STOP}


def build_alias_index(data_dir: str | Path,
                      season_start="2024-07-01", season_end="2025-06-30"):
    """name(normalised) -> list of (player_id, club_token_set)."""
    d = Path(data_dir)
    players = pd.read_csv(d / "players.csv",
                          usecols=["player_id", "name", "current_club_name"])
    id2nm = players.set_index("player_id")["name"].map(_nn)

    frames = [pd.DataFrame({"pid": players.player_id,
                            "nm": players.name.map(_nn),
                            "club": players.current_club_name})]

    ap_path, cl_path = d / "appearances.csv", d / "clubs.csv"
    if ap_path.exists() and cl_path.exists():
        cl = pd.read_csv(cl_path, usecols=["club_id", "name"]).set_index("club_id")["name"]
        ap = pd.read_csv(ap_path, usecols=["player_id", "player_name", "player_club_id", "date"])
        ap["date"] = pd.to_datetime(ap["date"], errors="coerce")
        ap = ap[(ap.date >= season_start) & (ap.date <= season_end)]
        frames.append(pd.DataFrame({"pid": ap.player_id, "nm": ap.player_name.map(_nn),
                                    "club": ap.player_club_id.map(cl)}))

    pv_path = d / "player_valuations.csv"
    if pv_path.exists():
        pv = pd.read_csv(pv_path, usecols=["player_id", "date", "current_club_name"])
        pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
        pv = pv[(pv.date >= "2024-01-01") & (pv.date <= "2025-12-31")]
        frames.append(pd.DataFrame({"pid": pv.player_id,
                                    "nm": pv.player_id.map(id2nm).values,
                                    "club": pv.current_club_name}))

    al = pd.concat(frames, ignore_index=True).dropna(subset=["nm"])
    al = al[al.nm.str.len() > 0]
    index: dict[str, list] = defaultdict(list)
    seen = set()
    for pid, nm, club in zip(al.pid, al.nm, al.club):
        key = (nm, pid)
        ct = _toks(club)
        # merge club tokens per (name,pid)
        index[nm].append((pid, ct))
    return index


def resolve_player_ids(fbref_df: pd.DataFrame, data_dir: str | Path,
                       team_col: str | None = None) -> pd.Series:
    """Return a player_id Series aligned to fbref_df (NaN where unresolved)."""
    team_col = team_col or ("team" if "team" in fbref_df.columns else "squad")
    index = build_alias_index(data_dir)

    out = []
    for nm, team in zip(fbref_df["player"].map(_nn), fbref_df[team_col]):
        tt = _toks(team)
        hits = []
        for pid, ct in index.get(nm, []):
            ov = len(ct & tt)
            if ov:
                hits.append((ov, pid))
        if hits:
            hits.sort(reverse=True)
            out.append(hits[0][1])
        else:
            out.append(pd.NA)
    return pd.Series(out, index=fbref_df.index, dtype="Int64")
