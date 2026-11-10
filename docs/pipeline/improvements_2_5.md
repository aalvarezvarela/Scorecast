# 2_5 feature improvements backlog

Status: **Plan** (opened 2026-11-10) — nothing here is implemented. These are
weaknesses of the `2_5` feature set found in the 2026-11-10 review of the
closing pipeline that are not bugs to fix in place. Numbers come from
`closing_line_data_2_5_20261016.parquet` (11,543 games, 1,670 columns).

A new column goes in a new schema layer. Changing or removing an existing
column needs a new base (`3_0`) or a decision to rebuild `2_5` in place, as in
`defect_fixes_2_5.md`. Either way the change is adopted only after a campaign
that clears the seed-noise gate of the `experiments` skill.

| # | Topic | Kind |
|---|---|---|
| 1 | Back-to-back and schedule density | New columns (layer) |
| 2 | Head-to-head totals are stale raw points | Change or drop |
| 3 | Near-constant price columns | Drop |
| 4 | Consensus pick percentages of 2023-24 | Data backfill |

## 1. Back-to-back and schedule density

`BACK_TO_BACK_BEFORE` (`add_features_after_merging.py`) is true only when
**both** teams played the day before (`REST_DAYS_BEFORE_MATCH <= 1` on each
side): 4.9% of games. One team alone is on a back-to-back far more often (home
15.9%, away 18.8%), and the model can only rebuild that from the two rest-day
columns. The column is also stored as a bool. Nothing describes the schedule
beyond the last rest gap.

Proposed columns, per team and as home-minus-away differences:

- `IS_BACK_TO_BACK_TEAM_{HOME,AWAY}_BEFORE` (0/1) and a rest-advantage flag
  (one side on a back-to-back, the other with two or more rest days);
- games in the last 4, 6 and 10 days (3-in-4, 4-in-6, 5-in-7);
- away games among those, and consecutive away games (road-trip length);
- minutes played by the current top-8 rotation in the last 3 days, which
  separates a back-to-back after a blowout from one after overtime.

Rest and travel were found fully priced at the close (broad review
2026-09-17), so expect value, if any, at T-720 and earlier.

## 2. Head-to-head totals are stale raw points

`last_5_matchups_for_frame()` (`historic_games_statistics.py`) writes six
columns: `LAST_k_GAMES_TOTAL_POINTS_BEFORE` for `k` = 1 to 5, the total points
of the k-th most recent earlier meeting **with the same home and away teams**,
and `LAST_5_GAMES_TOTAL_POINTS_BEFORE_MEAN`, the mean of those available.

- **Age.** A pairing meets once or twice a season in each arena. The previous
  meeting (`k=1`) is a median of 284 days old; the fifth (`k=5`) a median of
  1,094 days, and over 1,866 days for one game in ten. There is no limit on
  how far back the lookup goes.
- **Scale.** The values are raw totals while the league average total moved
  from 212.7 (2017-18) to 231.2 (2025-26). A 215 from 2018 and a 215 from 2025
  mean opposite things, so the columns mostly encode the era and each team's
  long-run scoring level.
- **Rosters.** Three-year-old meetings were played by different rosters and
  coaches.
- **Coverage.** `k=5` is missing for 8.1% of games from 2021-22 on, and every
  value depends on how many past seasons the build loaded (the scope mismatch
  between training and the two-season prediction build).

Options, cheapest first:

1. Drop the six columns and measure (they did not appear among the features
   correlated with line error).
2. Keep only meetings from the current and previous season, both
   orientations, and store each as a residual instead of raw points: total
   minus that game's closing line, or total minus the league average of its
   season.
3. Replace them with one shrunk pairing effect: the mean residual of the
   pairing's recent meetings, weighted by `n / (n + k)`.

## 3. Near-constant price columns

About 70 columns hold one value in more than 99.5% of regular-season rows from
2021-22 on. The cause is `normalize_total_lines`: each book's total is moved
to its 50/50 line, so both prices become 1.91. bet365's total prices are
1.91/1.91 on practically every row and the other books on 89–99.7%.

Affected families:

- per-book total prices, vig, no-vig probabilities and price log-ratios;
- the rolling histories built from them by the dynamic column discovery in
  `team/rolling.py`, for totals and for spread and moneyline prices (63 + 63
  rolling price columns);
- `LEAGUE_GAMES_LAST_*G`, `REF_CREW_UNKNOWN_COUNT`, `INJURY_REPORT_COVERED`.

They cost build time, memory and columns for the correlation pruning to
remove, and add nothing. Proposal: stop generating the rolling price families
and the per-book vig features of normalised markets. Related: the
normalisation overwrites the quoted line and price, so the dataset cannot tell
how far the target line is from what the book showed; keeping the quoted line
and the shift as two columns would let that be audited.

## 4. Consensus pick percentages of 2023-24

`ODDS_total_consensus_pct_*` and `ODDS_spread_consensus_pct_*` (SBR's
"consensus", which is the Covers Consensus pick share) are broken for one
season:

- missing in 55–77% of games from October 2023 to March 2024 (about 60% of the
  season; 0–3% in every other season), so the NaN pattern marks the season;
- where present, the values are not the usual ones: standard deviation 18.8
  against 6.7–7.5 in the other seasons, 14% of them exactly 0 or 100 (none in
  other seasons), 44% on coarse fractions such as 25, 50, 75. They look like
  shares of a handful of picks.

Covers still serves the real counts. A probe of four dates (2023-11-12,
2023-12-26, 2024-01-12, 2024-03-10) returned pick counts for 39 of 39 games,
several hundred picks each, and they disagree with the stored values (stored
75, 100, 50, 60 against 68, 70, 58, 66 for the same day). The Covers store
holds only 14 games of that season (`lh_consensus`).

Proposal: backfill `lh_consensus` for 2023-24 (one matchups page per date to
get the Covers ids, one consensus request per game, about 1,300 requests) and
write the four percentage columns of that season from it. `fill_from_covers`
only fills empty cells and never replaces an SBR value, so the 2023-24 values
need an explicit overwrite for the season, not the standing fill. The rolling
and league-level columns derived from the percentages change with it. Whether
the percentages carry signal at all is a separate question (their correlation
with line error is about -0.03).
