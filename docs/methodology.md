# Methodology

This pipeline surfaces players who are (a) capable of high-intensity football,
(b) young with a good age profile, (c) low market value, and (d) either
misprofiled by position or playing limited minutes. Below is how each idea maps
to the code and to the literature you supplied.

## 1. Scope

Leagues are the current Opta Power Rankings top 10 (Premier League, Serie A,
La Liga, Bundesliga, Ligue 1, Championship, Primeira Liga, Eredivisie, Brazil
Série A, Argentina Primera). Opta's ratings are proprietary and are used only to
choose *which* leagues to scout — all player data is open (FBref/StatsBomb,
Transfermarkt). All 10 are available on FBref (verified names in
`src/setup_leagues.py`), but Brazil and Argentina run calendar-year seasons
(handled via a separate `season_calendar`) and their advanced stats can be
thinner than the Big 5.

## 2. Data

* **FBref (StatsBomb-powered)** — ~200 advanced per-90 columns per player-season
  (shooting/xG, passing, pass types, shot- and goal-creating actions, defense,
  possession/carries, playing time, misc). This is the performance backbone.
* **Transfermarkt** — market value, age, contract expiry. Supplies the value and
  age filters.

Both are pulled via `soccerdata`, which parses FBref's comment-embedded tables
for us. The pull must run in an open-network environment (your machine).

## 3. Normalisation — always within position

Every performance metric is converted to per-90 and standardised **within the
player's positional peer group** (z-scores and percentiles). A full-back is
never compared to a striker. This reflects two threads in the literature:

* Position is a first-order determinant of value (Poli, Besson & Ravenel 2022;
  Franceschi et al. 2023, systematic review), so cross-position comparison would
  bias the value read.
* Scouting practice is role-based and multidisciplinary — players are assessed
  against the demands of their function, not a global average (Bergkamp,
  Frencken, Niessen, Meijer & den Hartigh 2022; Barraclough, Till, Kerr &
  Emmonds 2025).

## 4. The four signals

Each returns 0–1 and is combined with configurable weights (`config.yaml`).

### performance (weight 0.35)
Mean within-group z-score across output metrics (npxG, xA, SCA, GCA, goals,
assists, key passes, shots on target), squashed with a sigmoid. Grounds the
shortlist in on-pitch productivity — the dominant value determinant across the
reviewed studies.

### intensity (weight 0.25)
Mean within-group z-score across pressing/duel/carrying volume (tackles,
interceptions, blocks, pressures, progressive carries, aerials, recoveries).
This is an **open-data proxy** for the physical "can handle intense games"
requirement, because true GPS/high-speed-running data is largely proprietary.
Where you can source real physical data, it slots in here.

### misprofiling (weight 0.20)
A Linear Discriminant model learns each role's statistical fingerprint from
**globally** standardised profile features (so a player's fingerprint is
absolute, not relative to the group they are listed in). For each player we take
the probability mass assigned to the *best role other than the one they are
listed in*. A high value means "plays like something other than their label" —
e.g. a listed winger whose profile clusters with progressive full-backs. This
operationalises the literature's core point that listed position frequently
mis-describes actual function.

### underused (weight 0.20)
`performance × (1 − minutes_share)`, where `minutes_share` is minutes relative to
the team's most-used player. Rewards strong per-90 output on a low share of
minutes. Since minutes/experience is itself a value determinant, productive
players with low minutes tend to be systematically under-priced.

## 5. Eligibility gate

Applied before scoring (`config.yaml → filters`): minimum minutes for a stable
per-90 sample (default 450), maximum age (default 24), a maximum minutes-share to
exclude nailed-on starters, and a market-value ceiling (default €20m).

## 6. Output

A ranked `shortlist.csv` with each player's composite `scout_score` and the four
component signals, so any name can be audited back to *why* it surfaced.

## 7. Known limitations

* Intensity is a proxy; without tracking data it cannot capture true
  high-speed-running capacity.
* The misprofiling model needs enough players per role; thin leagues/positions
  degrade it gracefully to 0.
* Name-based joins between FBref and Transfermarkt are accent-folded but not
  fuzzy — a small fraction of players may fail to merge and lose their value/age.
* Market values are a media/crowd proxy for price, not actual transfer fees
  (Poli et al. 2022 discuss the divergence, especially for out-of-contract
  players).

## References (supplied)

* Bergkamp, T.L.G., Frencken, W.G.P., Niessen, A.S.M., Meijer, R.R., den
  Hartigh, R.J.R. (2022). *How soccer scouts identify talented players.*
  European Journal of Sport Science, 22(7), 994–1004.
* Barraclough, S., Till, K., Kerr, A., Emmonds, S. (2025). *Challenges and
  solutions to talent (de)selection and development in a youth soccer academy.*
  Frontiers in Psychology.
* Poli, R., Besson, R., Ravenel, L. (2022). *Econometric Approach to Assessing
  the Transfer Fees and Values of Professional Football Players.* Economies, 10, 4.
* Franceschi, M., Brocard, J.-F., Follert, F., Gouguet, J.-J. (2023).
  *Determinants of football players' valuation: A systematic review.* Journal of
  Economic Surveys.
* Franceschi, M., et al. (2023). *Football players in light of economic value
  theory.* Managerial and Decision Economics.
* (mf-04-2020-0213) Emerald — player valuation study.
