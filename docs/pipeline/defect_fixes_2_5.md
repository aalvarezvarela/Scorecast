# 2_5 defect fixes

Defects found in the `2_5` training and prediction pipelines (consolidated
2026-10-16 from `perf/pipeline-speedup` and `fix/todays_prediction_data`),
fixed on `fix/training-data-defects`.

Status: **Implemented** (checked 2026-10-22) — the fixes are merged, the `2_5`
data was rebuilt on 2026-10-16, and all 51 production slots were retrained on it
and promoted on 2026-10-20. Item A is accepted and not fixed.

| # | Problem | Status |
|---|---|---|
| 1 | Cross-book total std computed over rolling columns, not books | Fixed (code) |
| 2 | Early-2021 BetMGM totals hold half-game-sized values | Fixed (repair rule) |
| 3 | Injury-report spans use stale game tip-offs | Fixed (store reloaded, cache refreshed) |
| 4 | Injured and questionable players lose their last game | Fixed (code) |
| 5 | Roster membership lags transactions by one game | Fixed (game-day roster) |
| 6 | `engineer_odds_features` counts derived columns as books | Fixed (code) |
| 7 | Player EWMA weighting | Changed by decision |
| 8 | Veterans without current-season history vanish from player groups | Fixed (per-member earlier-history fallback) |
| 9 | Available players without a target box-score row lose their last game | Fixed (date-based `_AFTER` selection) |
| A | Intermediate `GLOBAL_*` can include results after the snapshot | **Accepted, not fixed** |
| 10 | Closing repair takes a book's stale opener as its close (2026-27 Fanatics) | Fixed (repair rule, 2026-10-31) |

## Decisions

- **Schema version: `2_5` is rebuilt in place.** The usual rule
  (`nba_ou.config.dataset_versions`, CLAUDE.md) is that changing an existing
  column's values needs a new base (`3_0`). These are bug fixes to `2_5`, so
  the base is regenerated under the same label instead. Old `2_5` files and
  models built on them are not comparable with the new ones; the `2_6`
  manifests reference their parent's checksum, so `2_6` must be rebuilt from
  the new `2_5` too.
- **BetMGM:** a code rule in the closing-line repair (10-point threshold for
  unpriced totals), not a source-data edit.
- **Roster membership:** the game-day roster definition in item 5.
- **EWMA:** recent games weigh more, moderately (item 7).

## Remaining steps

1. **Rebuild** the closing and intermediate `2_5` datasets, then the `2_6`
   layer. The history memo keys cover all `nba_ou` source files, so the code
   changes invalidate old entries automatically.
2. **Check the rebuild:** Appendix A (cross-book std median about 0.3, ranges
   above 5 points rare), full-vs-tonight-only parity against a reference built
   at the fixed commit (old references carry the old values).
3. **Retrain with Optuna** and compare with the current production models
   using the `experiments` protocol, then promote.
4. **Meta-learner:** regenerate its dataset afterwards. Note that
   `meta_learner_training_data.py` still reads the retired registry layout.

---

## 1. Cross-book total std computed over rolling columns

**Problem.** `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE` was the std over every
column starting with `ODDS_TOTAL_LINE_`, including each book's rolling means,
season averages and trends for both teams: 212 columns against 6 books.
Stored mean 81.6 (median 104.1); over the real books the mean is 0.37 (median
0.32), and the two correlate at -0.007. In practice the column was an era
marker plus a line-level proxy. Enabled models read
`GLOBAL_CROSSBOOK_TOTAL_STD_AVG_15G/75G_BEFORE` (closing and intermediate).

**Root cause.** `extract_total_line_books` accepted any suffix as a book
(`^ODDS_TOTAL_LINE_(.+)$`).

**Fix.**
- `config/odds_columns.py`: `BOOK_SLUG_PATTERN = r"[a-z0-9_]+"`. Raw book
  columns have a lower-case slug and derived columns an upper-case suffix, so
  `extract_total_line_books` and `extract_spread_line_home_books` use a
  `fullmatch` on that pattern. This also fixes the fallback in
  `resolve_main_total_line_col`.
- `global_market_features.py`: the opener is excluded
  (`CROSSBOOK_EXCLUDED_BOOKS`), and a game with fewer than two books gets NaN
  instead of std 0 (`CROSSBOOK_MIN_BOOKS`).
- Tests: `tests/test_crossbook_total_std.py`.

**Expected after rebuild.** Std median about 0.3; `GLOBAL_CROSSBOOK_*` well
under a point. Every model reading these columns must be retrained.

## 2. Early-2021 BetMGM totals

**Problem.** A few 2020-21 BetMGM closing totals sat far from every other book:
114.5 and 119.5 (first-half lines stored as full-game totals) and some 190s,
for example `0022000437` (bet365 234.5, BetMGM 119.5, FanDuel 235.5).

**Why the repair missed them.** The other books were verified, so the
cross-book check could run. The BetMGM cells were *unpriced* (a line without a
price), and the outlier check compared priced levels only, so these cells were
never tested.

**Fix** (`data_processing/odds/closing_line_repair.py`, step B):
- An unpriced totals cell is compared on its raw line against the verified
  median of the other books, with a 10-point threshold
  (`UNPRICED_OUTLIER_THRESHOLDS`).
- An unverified total outside `TOTAL_LINE_BOUNDS` (150–300) is an outlier even
  without other books to compare against.
- The anchor book is median-filled only when the other books' median exists.

On 2019–2026 the rule nulls exactly 7 more BetMGM 2020-21 cells (21 nulled in
total, 14 before). A 219 against a 225 median is left alone. Tests: the
unpriced-total cases in `tests/test_closing_line_repair.py`.

## 3. Stale tip-offs in the injury-report store

**Problem.** The Aiven `injury_report` store fixes each game's tip-off
(`ir_game.tipoff_utc`) when a season is loaded, and every status and filing
span is clamped at it. 2025-26 and two 2021-22 playoff games were loaded before
later tip-off corrections: 57 games differ, by up to 6 hours. In 37 the stored
tip-off is later than the real one, so the "at tip" closing features can read
a report published after the game started (a small in-game leak). In 20 it is
earlier, so the last pre-game reports are missed. Example: `0022500645` stored
at 23:00 UTC, real tip-off 17:00. All 57 corrected tip-offs agree with the
line-history store.

**Fix.** Reload both seasons from the archived PDFs (S3, or a local copy of
them with `--local-root`), then refresh the S3 input cache, which the builders
read for finished seasons:

```bash
python scripts/injury_reports/load_injury_reports_to_aiven.py --season 2025-26 --replace-season
python scripts/injury_reports/load_injury_reports_to_aiven.py --season 2021-22 --replace-season
python scripts/refresh_input_cache.py --seasons 2021 2025
```

`--replace-season` is required; without it inserts are `ON CONFLICT DO
NOTHING` and the old spans stay. The old rows are deleted only after the whole
season has been parsed.

**Reload (2026-10-16).** Both seasons were reloaded from a local copy of the
S3 PDFs; the dry runs and the real loads gave identical summaries:

| season | reports | observations | name resolution | tip-offs from reports | games dropped (no tip-off) |
|---|---|---|---|---|---|
| 2021-22 | 6,015 | 345,592 | 99.98% | 0 | 0 |
| 2025-26 | 18,951 | 1,122,017 | 99.96% | 131 | 0 |

The 131 are the 2025-26 games missing from the schedule feed (playoffs,
play-in, NBA Cup), as expected.

Rows replaced: 2025-26 had 20,065 status spans and now has 20,090; 2021-22
keeps 16,485. The filing-span and report counts are unchanged.

**Verification (passed 2026-10-16).**
- Appendix B: 0 changed games in every season. The 131 report-derived
  tip-offs, which are not in the schedule feed, all agree with line history.
- `scripts/perf/check_injury_window_ingest.py --first 2026-04-09 --last
  2026-04-10`: `RESULT: EQUAL` (429 spans and 84 filing spans).
- `refresh_input_cache.py --seasons 2021 2025` uploaded the changed queries:
  `last_status_before_tip` and `listed_status_events` for 2025, and
  `report_age_at_tip` for 2021 and 2025. A second run reports everything
  unchanged.

## 4. Injured and questionable players lose their last game

**Problem.** `precompute_cumulative_avg_stat` gives each row the EWMA of the
player's *earlier* games (`shift(1)`), correct for a game he plays. A player
Out or Questionable tonight has no row, so the injured groups read his last
row before the game, and that row excludes the game itself. His most recent
game, already played, was always left out.

**Fix** (`players/players_statistics.py`, `players/attach_player_features.py`).
Each stat also gets an unshifted `<stat>_AFTER` average: the value a row after
that game would carry. The injured and questionable groups (`TOP*_INJURED_*`,
`AVG_INJURED_*`, `TOTAL_INJURED_*`, the `QUESTIONABLE` twins and the minutes
threshold that orders them) read `_AFTER`. Training and today's prediction use
the same code. A player with no previous game now gets NaN averages in
training, as he already did live (previously 0).

## 5. Roster membership lags transactions by one game

**Problem.** Training decided team membership from the player's last box-score
team *before* the game, plus report-listed players. Every trade, signing,
release and debut was recognised one game late, and season openers used last
season's rosters. Live already used the official roster, so training and live
differed exactly in those cases.

**Fix: game-day roster.** The possible players of a team-game are:
- **Historical:** the game's box-score players plus everyone the injury report
  lists for that team. A player listed injured but missing from the box score
  is assumed to belong to the team. The injury reports remain the source of
  truth for who is injured; the box score only defines who could play.
- **Today:** the NBA API roster (`CommonTeamRoster`) plus the report-listed
  players. If the official roster is missing for a team-game, the fallback is
  the game-day roster of the team's previous game.

Code: `PlayerLookupIndex.game_roster` / `lookup_for(game_day_listed=…)` in
`past_injuries/past_injuries.py`; `game_day_rosters` and
`previous_game_rosters` in `players/live_roster.py`. The closing builder, the
intermediate injuries and today's prediction all pass these rosters to the
player groups, All-Star vote shares and roster continuity.

## 6. `engineer_odds_features` counts derived columns as books

**Problem.** Found in review after item 1: `engineer_odds_features` detected
books with its own `^ODDS_TOTAL_LINE_(.+)$`. In the `2_5` data that produced
211 `ODDS_book_total_line_mid_*` columns, of which only 6 are books, and the
`ODDS_total_line_books_*` and `ODDS_n_books_*` aggregates were computed over
them.

**Fix.** `merged_home_away_data/odds_feature_engeneer.py` reuses
`extract_total_line_books` and `BOOK_SLUG_PATTERN` for the totals, prices,
spread and moneyline families. The rebuilt datasets lose the 205 spurious
per-book columns. Tests: `tests/test_odds_feature_books.py`.

## 7. Player EWMA weighting

**Decision.** The player averages used `adjust=False` with a 10-game
half-life, which gave the latest game a lot of weight (10 then 30 points read
11.34 once a previous-season fill is included). They now use `adjust=True`
with a 15-game half-life (`EWMA_HALFLIFE_GAMES`): recent games still weigh
more, but less sharply.

---

## 8. Veterans without current-season history

**Problem.** A game-day roster could include an injured or questionable veteran
whose last row was in an earlier season, but the statistic lookup searched only
the current season. If teammates had history, the whole-roster fallback never
ran. The veteran disappeared from the ranked groups and their counts. Recorded
inputs included 595 such report rows in 460 games; Dallas in `0022500133` counted
one injured player instead of four through the live roster path.

**Fix.** `PlayerLookupIndex.prior_history_members` detects missing earlier rows
per member. Those historical game-day rosters use `live_roster_history`, exactly
as an official live roster does, retaining earlier seasons and constructing
safe pre-game states without changing membership. Rookies still have no invented
statistics. Regression coverage includes Out, Questionable and available
veterans, partial current-season rosters, DNP rows, and both array/frame paths.

## 9. Available players without a target box-score row

**Problem.** Out/Questionable groups included the last known game via `_AFTER`,
but the available group always read the shifted `_CUM_AVG`. A player considered
available by the report but absent from the target box score lost their last
game: prior scores 10 and 30 read 10 historically versus 20.23 through serving.

**Fix.** Both the frame and array selectors choose `_AFTER` when the selected
row is strictly earlier than the target date, including the minutes threshold
used for ranking. Same-day rows retain `_CUM_AVG` so target-game outcomes stay
excluded. Regression tests compare historical and official-roster paths.

## 10. Stale per-book closes from an opener-only history

Found 2026-10-31. **Problem.** In 2026-27 the SBR/Covers feed keeps only
Fanatics' opener for about a third of the games: its last totals tick is over
24 hours before tip in 34% of them (0.1% in 2025-26). Step A of the closing
repair takes each book's last pre-tip quote however old, so it replaced SBR's
correct close with that opener. Example `0022600020`: SBR 237.0, every other
book 237-237.5, repaired Fanatics 242.0 (the opener of 10-21). After the
repair 24 of 80 Fanatics closes sat 2.5+ points from the other books' median
(6 in SBR's raw value). Step B's 6-point threshold, built for closes, let
these through. They fed every `*_fanatics_sportsbook_*` rolling feature and
the cross-book aggregates, in training and in tonight's rows alike.

**Fix** (`closing_line_repair.py`, step A2): a history cell whose last quote
is over `STALE_BOOK_LAG_MINUTES` (720) older than the freshest verified book
of its game and market is emptied (`stale_book_nulled`; the anchor book is
filled with the median, `stale_book_median_filled`) when at least 3 fresh
books agree within `OTHERS_MAX_RANGE` and it sits `STALE_BOOK_THRESHOLDS` (2
points; 0.05 moneyline win probability) or more from them. Measured on
2023-24 to 2026-27 history closes: up to a 12-hour lag 1.5% of totals sit 2.5+
points from the other books, beyond it 13-39%. SBR's value is no fallback: on
the lagged cells it was off as often (11% vs 17% of totals, 5% vs 3% of
spreads). A stale quote still in line with the market (a book that did not
move all day) is kept.

The rule empties 22 cells of 2026-27 (21 Fanatics totals) and 2-26 per market
in earlier seasons. Applied with `repair_sportsbook_closes.py --season-year 2026
--redo` and the 2026 closing and intermediate partitions rebuilt on 2026-10-31:
Fanatics closes 2.5+ points off the median went from 24 to 5 of 80, and only
Fanatics, BetMGM and cross-book columns changed. Earlier seasons keep their
memoised repair (`closing_line_repair` history memo, frozen key) until they are
rebuilt. Tests: the stale-book cases in `tests/features/test_closing_line_repair.py`.

## Accepted in 2_5 (not fixed)

### A. Intermediate `GLOBAL_*` can include results after the snapshot

**What happens.** `add_global_market_features` computes the league-wide
indicators (`GLOBAL_*`, `LEAGUE_GAMES_*`) once per game from all games on
earlier dates, and the intermediate builder copies them to every snapshot of
the game. A long-horizon snapshot of an early tip-off falls on the previous
evening, before some of that day's games have finished, yet its features
already include their results.

Example: `0021900036` tips at 19:30 UTC, so its T-1080 snapshot is at 01:30
UTC. Its `GLOBAL_MARKET_BIAS_15G_BEFORE` already includes `0021900035`
(tip-off 02:00 UTC, 252 points).

**Size, intermediate `2_5`, 146,815 rows:**
- A prior-date game not yet started at the snapshot: 279 rows (229 games),
  T-960 and T-1080 only.
- Not yet finished (tip-off + 2.5 h): 961 rows (618 games), T-720 to T-1080.
- Team features are not affected: no team's own previous game was unfinished
  at any snapshot.

**Why it is accepted.** It affects about 0.65% of rows at long horizons, and
the extra information is one or two games inside 15–150-game windows. Today's
prediction does the same thing (yesterday's results are used for a T-1080
snapshot taken yesterday evening), so training and serving agree. It only
flatters the long-horizon backtest slightly.

**If it is fixed later.** After the snapshot join in
`create_intermediate_line_df`, give each row the `GLOBAL_*` values of the
latest game date whose earlier games had all finished by the snapshot
(tip-off + 3 h). `base` already holds every date's values, and `lh_games` has
the tip-offs. The same code covers training and live. Comparing a full build
with a date-truncated one does not detect this intraday case.

---

## Appendix A: cross-book numbers from a closing build (items 1 and 2)

Point it at any full closing dataset (Parquet) built with `create_df_to_predict`.
Before the fix `stored` equals the std over all `ODDS_TOTAL_LINE_*` columns;
after it, `stored` should equal `correct`.

```python
import re
import pandas as pd
import pyarrow.parquet as pq

PATH = "closing_line_data_2_5.parquet"  # any full closing build
BOOKS = ["bet365", "betmgm", "fanduel", "draftkings", "fanatics_sportsbook", "betrivers"]

names = pq.read_schema(PATH).names
total_cols = [c for c in names if re.match(r"^ODDS_TOTAL_LINE_(.+)$", c)]
book_cols = [f"ODDS_TOTAL_LINE_{b}" for b in BOOKS if f"ODDS_TOTAL_LINE_{b}" in names]
d = pd.read_parquet(
    PATH,
    columns=sorted({*total_cols, "SEASON_YEAR", "GAME_ID", "GAME_DATE",
                    "THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE"}),
)

stored = d["THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE"]
over_all = d[total_cols].apply(pd.to_numeric, errors="coerce").std(axis=1, ddof=0)
print(len(total_cols), "ODDS_TOTAL_LINE_* columns")
print("stored == std over all of them:", float((stored - over_all).abs().max()))

books = d[book_cols]
n = books.notna().sum(axis=1)
correct = books.std(axis=1, ddof=0).where(n >= 2)
spread = (books.max(axis=1) - books.min(axis=1)).where(n >= 2)
print("stored == correct:", float((stored - correct).abs().max()))
print("stored :", stored.describe().round(3).to_dict())
print("correct:", correct.describe().round(3).to_dict())
print(d.assign(correct=correct).groupby("SEASON_YEAR")[
    ["THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE", "correct"]].mean().round(2))
for t in (2, 3, 5, 10):
    print(f"range > {t}:", int((spread > t).sum()))
print(d.loc[spread > 10, ["GAME_ID", "GAME_DATE", *book_cols]])
```

## Appendix B: stored injury-report tip-offs against today's (item 3)

Read-only. It needs the same DB credentials as the builders: Aiven for the
injury-report and line-history stores, and Supabase as the ingest's game
source.

```python
import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.injury_report_aiven import resolve as R
from nba_ou.postgre_db.injury_report_aiven.schema import SCHEMA
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_games

with connect_nba_db("aiven") as conn:
    stored = pd.read_sql_query(
        f"SELECT game_id, tipoff_utc, tipoff_source FROM {SCHEMA}.ir_game", conn
    )
stored["game_id"] = stored["game_id"].astype(str)
stored["season"] = 2000 + stored["game_id"].str.zfill(10).str[3:5].astype(int)
seasons = sorted(stored["season"].unique().tolist())

with connect_nba_db(env="supabase") as source:
    today = R.load_game_dimension(seasons, source)  # what a reload would use
today["game_id"] = today["game_id"].astype(str)
lh = fetch_games(seasons)
lh["game_id"] = lh["game_id"].astype(str)

g = (
    stored.merge(today[["game_id", "tipoff_utc"]].rename(columns={"tipoff_utc": "today"}),
                 on="game_id", how="left")
    .merge(lh[["game_id", "tipoff_utc"]].rename(columns={"tipoff_utc": "line_history"}),
           on="game_id", how="left")
)
for c in ("tipoff_utc", "today", "line_history"):
    g[c] = pd.to_datetime(g[c], utc=True)
g["delta_min"] = (g["tipoff_utc"] - g["today"]).dt.total_seconds() / 60
changed = g["delta_min"].fillna(0).ne(0)

print("not in today's game dimension:", int(g["today"].isna().sum()))
print(g.assign(changed=changed, stored_later=g["delta_min"] > 0)
       .groupby("season")[["changed", "stored_later"]].sum())
print("changed games agreeing with line history:",
      int(g.loc[changed, "line_history"].eq(g.loc[changed, "today"]).sum()),
      "of", int(changed.sum()))
print(g.loc[changed].sort_values("delta_min")[
    ["game_id", "tipoff_utc", "today", "line_history", "delta_min"]])
```
