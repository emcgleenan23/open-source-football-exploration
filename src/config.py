"""Load and validate config.yaml into a simple typed object."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    season: str
    leagues: list[str]
    filters: dict[str, Any]
    weights: dict[str, float]
    position_groups: dict[str, list[str]]
    output_dir: str
    season_calendar: str | None = None
    leagues_calendar: list[str] = field(default_factory=list)
    fbref_headless: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = ROOT / "config.yaml") -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(
            season=data["season"],
            leagues=data["leagues"],
            filters=data["filters"],
            weights=data["weights"],
            position_groups=data["position_groups"],
            output_dir=data.get("output_dir", "data/processed"),
            season_calendar=data.get("season_calendar"),
            leagues_calendar=data.get("leagues_calendar", []) or [],
            fbref_headless=data.get("fbref_headless", False),
            raw=data,
        )

    def league_groups(self) -> list[tuple[list[str], str]]:
        """(leagues, season) pairs to pull, skipping empty groups."""
        groups = [(self.leagues, self.season)]
        if self.leagues_calendar and self.season_calendar:
            groups.append((self.leagues_calendar, self.season_calendar))
        return [(lg, ss) for lg, ss in groups if lg]

    @property
    def out_path(self) -> Path:
        p = ROOT / self.output_dir
        p.mkdir(parents=True, exist_ok=True)
        return p
