"""
Offline tests for the browserless FBref parser (src/pull_data_http.py).
Uses a synthetic FBref-shaped page: a two-row header, a repeated mid-table
header row, and the table hidden inside an HTML comment (as FBref does).

    python tests/test_http_parse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pull_data_http import extract_player_table, uncomment_tables  # noqa: E402

# A miniature FBref "standard" player table, wrapped in a comment like the real site.
FIXTURE_HTML = """
<html><body>
<div id="all_stats_standard">
<!--
<div>
<table class="stats_table" id="stats_standard">
  <thead>
    <tr class="over_header">
      <th></th><th></th><th></th><th></th>
      <th colspan="2">Performance</th><th colspan="2">Expected</th>
    </tr>
    <tr>
      <th>Rk</th><th>Player</th><th>Pos</th><th>Squad</th>
      <th>Gls</th><th>Ast</th><th>xG</th><th>npxG</th><th>Matches</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>Alpha Striker</td><td>FW</td><td>Team A</td>
        <td>12</td><td>4</td><td>10.1</td><td>9.2</td><td>Matches</td></tr>
    <tr class="thead"><td>Rk</td><td>Player</td><td>Pos</td><td>Squad</td>
        <td>Gls</td><td>Ast</td><td>xG</td><td>npxG</td><td>Matches</td></tr>
    <tr><td>2</td><td>Beta Mid</td><td>MF</td><td>Team B</td>
        <td>3</td><td>7</td><td>2.8</td><td>2.8</td><td>Matches</td></tr>
  </tbody>
</table>
</div>
-->
</div>
</body></html>
"""


def test_uncomment_promotes_hidden_table():
    soup_raw = uncomment_tables("<div><!-- <table id='stats_standard'></table> --></div>")
    assert soup_raw.find("table", id="stats_standard") is not None


def test_extract_and_clean():
    soup = uncomment_tables(FIXTURE_HTML)
    df = extract_player_table(soup, "standard")
    assert df is not None, "table not found after un-commenting"

    # header flattened + lowercased
    assert "player" in df.columns
    assert any(c.endswith("xg") for c in df.columns), df.columns.tolist()

    # repeated mid-table header row dropped -> exactly 2 real players
    assert len(df) == 2, f"expected 2 players, got {len(df)}: {df['player'].tolist()}"
    assert set(df["player"]) == {"Alpha Striker", "Beta Mid"}

    # 'matches' link column and 'rk' dropped
    assert "matches" not in df.columns
    assert "rk" not in df.columns

    # numeric stat survived
    xg_col = [c for c in df.columns if c.endswith("xg") and "np" not in c][0]
    assert float(df.loc[df["player"] == "Alpha Striker", xg_col].iloc[0]) == 10.1


if __name__ == "__main__":
    test_uncomment_promotes_hidden_table()
    test_extract_and_clean()
    print("HTTP parser tests passed.")
