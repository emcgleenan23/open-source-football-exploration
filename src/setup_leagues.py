"""
Register the non-default leagues with soccerdata.

soccerdata only ships the Big 5 + a few internationals. The Championship,
Primeira Liga, Eredivisie, Brazil and Argentina exist on FBref but must be added
to soccerdata's custom league dict, which lives OUTSIDE this project at
    ~/soccerdata/config/league_dict.json   (or $SOCCERDATA_DIR/config/...)

The "FBref" value must match FBref's competition name EXACTLY (accents included) —
verified against https://fbref.com/en/comps/ .

Run once before the first pull:
    python -m src.setup_leagues
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Exact FBref competition names (from fbref.com/en/comps/).
EXTRA_LEAGUES = {
    # European, August–May seasons (same format as the Big 5)
    "ENG-Championship": {
        "FBref": "Championship",
        "season_start": "Aug", "season_end": "May",
    },
    "POR-Primeira Liga": {
        "FBref": "Primeira Liga",
        "season_start": "Aug", "season_end": "May",
    },
    "NED-Eredivisie": {
        "FBref": "Eredivisie",
        "season_start": "Aug", "season_end": "May",
    },
    # South American, single calendar-year seasons
    "BRA-Serie A": {
        "FBref": "Campeonato Brasileiro Série A",
        "season_code": "single-year",
    },
    "ARG-Primera Division": {
        "FBref": "Argentine Primera División",
        "season_code": "single-year",
    },
}


def config_dir() -> Path:
    base = os.environ.get("SOCCERDATA_DIR")
    return (Path(base) if base else Path.home() / "soccerdata") / "config"


def main() -> None:
    cfg_dir = config_dir()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    target = cfg_dir / "league_dict.json"

    existing = {}
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))

    merged = {**existing, **EXTRA_LEAGUES}
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registered {len(EXTRA_LEAGUES)} leagues -> {target}")
    for k in EXTRA_LEAGUES:
        print(f"  + {k}")
    print("\nNow run:  python -m src.pull_data")


if __name__ == "__main__":
    main()
