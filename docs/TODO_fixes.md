# Fixes to schedule

Known defects found in the `2_5` training and prediction pipelines, consolidated
from `perf/pipeline-speedup` and `fix/todays_prediction_data` on 2026-10-16.
Each entry preserves its evidence, impact, proposed fix and validation steps.
The measurements below are the original findings, not a fresh audit of the
current databases.

**Status: all five fixes are pending.** Consolidating this document does not
change code, source data, datasets or models. Live prediction retains the
current statistical behaviour, with one deliberate exception: it already uses
official roster membership, while training still infers membership (item 5).

## Pending fixes found in 2_5

Both closing and intermediate datasets are affected.

| # | Problem | Kind of fix |
|---|---|---|
| 1 | Cross-book total std uses rolling columns as well as raw books | Code |
| 2 | Early-2021 BetMGM totals contain half-game-sized values | Source-data repair or code rule; decision pending |
| 3 | Injury-report spans use stale game tip-offs | Source-data reload and S3 input-cache refresh |
| 4 | Injured and questionable player statistics omit their last game | Shared training/live code |
| 5 | Historical roster membership recognises transactions one game late | Reproducible training/live membership definition |

## Shared rebuild and validation plan

- **Schema version.** These are defects *found in* `2_5`, not permission to
  overwrite the frozen `2_5` base. Under `nba_ou.config.dataset_versions` and
  CLAUDE.md ("Schema versions are layers"), changing an existing column's
  values or meaning requires a new base (`3_0`), not an additive layer or an
  in-place `2_5` rebuild. Confirm the version label before implementation.
- **Coordinate the work.** Apply the agreed fixes together, then rebuild both
  closing and intermediate datasets and retrain the affected models. Compare
  with the current models using the same evaluation protocol. The BetMGM
  repair rule and the historical roster definition still need decisions.
- **Caches and references.** Refresh the S3 input cache after source-data
  repairs. The history memo keys cover package code, pandas/numpy versions and
  chained input digests; verify the corrected inputs invalidate old entries.
  Rebuild full reference datasets at the corrected commit before testing the
  tonight-only path against them.
- **Validation.** Run the affected unit tests, full-vs-tonight-only parity,
  point-in-time checks and live replay with identical roster inputs. The live
  official-roster difference in item 5 is intentional until training adopts
  the agreed game-day definition.
- **Feature documentation.** Keep `docs/README_Training Data Processing.md`
  in sync, including `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE` and the changed
  player/roster semantics.
- **After retraining.** Regenerate the meta-learner dataset with
  `scripts/build_meta_learner_training_data.py`.
- **Reproduce the evidence.** The original measurements used gitignored local
  builds under `artifacts/`. Appendix A reproduces the market diagnostics for
  items 1 and 2; Appendix B compares stored tip-offs for item 3. Items 4 and 5
  retain their player-statistic example and trade scenario below.

---

## 1. The cross-book total std is computed over rolling columns, not books

### Impact

`THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE` is meant to be the spread of a game's
closing total across sportsbooks, typically a fraction of a point. It is
actually the std over **every** column whose name starts with
`ODDS_TOTAL_LINE_`. When it is computed, those include each book's rolling
means, WMAs, season averages and trend columns for both teams: 212 columns in
the all-season closing build, against 6 books.

All affected columns come from
`src/nba_ou/data_processing/merged_home_away_data/global_market_features.py`
(section 8, "Cross-book disagreement features"):

| column | read by enabled models (2026-10-16) |
|---|---|
| `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE` | no (excluded from intermediate as leaky) |
| `THIS_GAME_CROSSBOOK_TOTAL_RANGE_BEFORE` | no (excluded from intermediate as leaky) |
| `GLOBAL_CROSSBOOK_TOTAL_STD_AVG_15G_BEFORE` | **yes**, closing and intermediate |
| `GLOBAL_CROSSBOOK_TOTAL_STD_AVG_30G_BEFORE` | no |
| `GLOBAL_CROSSBOOK_TOTAL_STD_AVG_75G_BEFORE` | **yes**, closing and intermediate |

Not affected: the intermediate snapshot columns
`ODDS_SNAP_{TOT,SPR,ML}_CONSENSUS_CROSSBOOK_{STD,RANGE}`. They come from
`data_processing/line_history/cross_book.py`, are computed over book ticks,
and are sub-point as expected.

What the models actually learned from it: the stored value is ~0 in seasons
without book lines and then tracks the league's average total line. In effect
it is an era marker and a line-level proxy, not book disagreement. Expect the
retrained models to lean on it differently.

| season | stored std (mean) | bet365 total line (mean) |
|---|---|---|
| 2017-18 | 0.0 | (no lines) |
| 2018-19 | 2.4 | 216 |
| 2019-20 | 102.7 | 222 |
| 2022-23 | 106.4 | 228 |
| 2025-26 | 107.6 | 230 |

### Evidence

All-season closing build (11,543 games), commit 2944ee5:

- The stored `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE` equals the std over all 212
  `ODDS_TOTAL_LINE_*` columns of the final frame (max difference 2.8e-14).
- Stored: mean 81.6, median 104.1, max 116.4.
- Over the 6 closing books instead: mean 0.37, median 0.32, 90th percentile
  0.55. The range has median 1.0 and 90th percentile 1.5.
- Correlation between stored and correct values: -0.007.
- The value depends on which derived columns a build happens to hold. The
  intermediate 2025-26 build averages 107.8 for the same `GLOBAL_*` columns.

### Root cause

`global_market_features.py`, around line 315:

```python
total_line_books = extract_total_line_books(out)
book_lines = pd.DataFrame(
    {b: pd.to_numeric(out[f"ODDS_TOTAL_LINE_{b}"], errors="coerce")
     for b in total_line_books if f"ODDS_TOTAL_LINE_{b}" in out.columns}
)
crossbook_std = book_lines.std(axis=1, skipna=True, ddof=0)
```

`extract_total_line_books` (`src/nba_ou/config/odds_columns.py`, line 260)
accepts any suffix as a book name:

```python
m = re.match(r"^ODDS_TOTAL_LINE_(.+)$", col)
```

so `bet365_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME` counts as a book.

### Proposed fix

1. **Accept lower-case slugs only.** Raw book columns are
   `ODDS_TOTAL_LINE_<slug>` with a lower-case slug: `bet365`, `betmgm`,
   `fanduel`, `draftkings`, `fanatics_sportsbook`, `betrivers`,
   `consensus_opener`. Every derived column carries an upper-case suffix. On
   the all-season closing build this pattern finds exactly those 7.

   ```python
   def extract_total_line_books(df: pd.DataFrame) -> list[str]:
       """Books with a raw total column ``ODDS_TOTAL_LINE_<book>``.

       Derived columns (rolling means, trends, season averages:
       ``ODDS_TOTAL_LINE_<book>_LAST_..._BEFORE_TEAM_HOME``) carry an
       upper-case suffix and are not books.
       """
       books = set()
       for col in df.columns:
           m = re.fullmatch(r"ODDS_TOTAL_LINE_([a-z0-9_]+)", col)
           if m:
               books.add(m.group(1))
       return sorted(books)
   ```

   This also fixes the function's other caller, `resolve_main_total_line_col`
   (`odds_columns.py`, line 293). Its fallback runs when the configured book's
   column is missing, and today it could pick a derived column as the main
   line.

2. **Leave the opener out, and require two books.** In
   `global_market_features.py`:

   ```python
   #: An opening consensus, not a book's close: it would measure line movement.
   CROSSBOOK_EXCLUDED_BOOKS = ("consensus_opener",)

   total_line_books = [
       b for b in extract_total_line_books(out) if b not in CROSSBOOK_EXCLUDED_BOOKS
   ]
   if len(total_line_books) > 1:
       book_lines = pd.DataFrame(
           {b: pd.to_numeric(out[f"ODDS_TOTAL_LINE_{b}"], errors="coerce")
            for b in total_line_books}
       )
       n_books = book_lines.notna().sum(axis=1)
       crossbook_std = book_lines.std(axis=1, ddof=0).where(n_books >= 2)
       crossbook_range = (
           book_lines.max(axis=1) - book_lines.min(axis=1)
       ).where(n_books >= 2)
   ```

   Today, with `ddof=0`, a game with a single book gets std 0, which reads as
   "the books agree". With the fix it gets NaN (14 games in the closing
   build). `_rolling_game_agg` already skips NaN, so the `GLOBAL_*_AVG_*G`
   rollups average over the games that have a value.

3. **Test.** For example in `tests/test_global_market_speedups.py`, next to
   the existing global-market tests. Use a frame with two books plus derived
   `ODDS_TOTAL_LINE_<book>_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME` columns.
   Assert that the std uses only the two books, and test
   `extract_total_line_books` on the same column names.

### Verification

- After rebuilding, `THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE` should have a
  median of about 0.3, and `GLOBAL_CROSSBOOK_TOTAL_STD_AVG_*` should be well
  under a point, except near the bad rows of item 2. Appendix A computes both.
- No stored model matches the new values. Retrain every slot that reads
  `GLOBAL_CROSSBOOK_TOTAL_STD_AVG_15G/75G`.
- The tonight-only parity checks (`scripts/perf/run_tonight_acceptance.sh` on
  `perf/pipeline-speedup`) compare against a full build. Their old reference
  carries the old values, so rebuild a full reference at the fixed commit
  first.
- Side note for the tonight-only builds: today, target mode must compute every
  row's total-line trends, because this std reads them
  (`data_processing/team/rolling.py`). After the fix nothing reads them
  through this path. Confirm with a parity run before relying on that. It
  would only be a speed-up, not a correctness change.

---

## 2. Early-2021 BetMGM totals hold half-game-sized numbers

### Impact

A few 2020-21 BetMGM closing totals sit far from every other book. They feed
the `ODDS_TOTAL_LINE_betmgm*` features directly, and they inflate item 1's
corrected cross-book std and range for those games.

### Evidence

With item 1's corrected range over the 6 books, the all-season closing build
has 7 games above 10 points, 27 above 5, 103 above 3 and 265 above 2. The
largest:

| GAME_ID | date | bet365 | betmgm | fanduel |
|---|---|---|---|---|
| 0022000437 | 2021-02-17 | 234.5 | 119.5 | 235.5 |
| 0022000446 | 2021-02-18 | 230.0 | 114.5 | 230.0 |
| 0022000454 | 2021-02-19 | 225.5 | 114.5 | 225.5 |
| 0022000455 | 2021-02-19 | 231.0 | 114.5 | 231.5 |
| 0022000206 | 2021-01-18 | 215.5 | 190.5 | 216.0 |
| 0022000355 | 2021-02-06 | 227.0 | 189.5 | 227.5 |

114.5 and 119.5 look like first-half lines stored as the full-game total. The
190s are unexplained. All of them survived the closing-line repair's outlier
nulling (`Closing-line repair (cells): ... outlier_nulled=14` in the build
log).

### Proposed fix (decision needed)

Choose one of these:

- Add a rule to the closing-line repair (`postgre_db/odds/merge_odds_data.py`,
  `_repair_from_line_history`): null a book's close that sits more than N
  points from the median of the other books. Choose N from the range
  distribution above.
- Or correct the stored rows in `odds_sportsbook` at the source.

### Verification

Rerun the range count in Appendix A. Ranges above 5 points should be rare,
and each should have an explanation.

---

## 3. Stale tip-offs in the injury-report store

### Impact

The Aiven `injury_report` store keeps each game's tip-off in
`injury_report.ir_game.tipoff_utc`, fixed when the season was loaded. Every
status and filing span is clamped at it (`valid_to <= tipoff_utc`). The
2025-26 season, and two 2021-22 playoff games, were loaded before later
tip-off corrections, so their stored tip-offs are wrong:

- **Stored tip-off later than the real one** (37 games): spans run past the
  real tip-off. The closing "at tip" injury features
  (`last_status_before_tip`, `filing_at_tip`, `report_age_at_tip` and the
  status features built on them) can therefore read a report published after
  the game started. That is a small in-game leak.
- **Stored tip-off earlier** (20 games): spans stop early, and the last
  pre-game reports are missing from the closing features. In the intermediate
  data, snapshots between the stale and the real tip-off see no listed
  players. The snapshot times themselves are correct: they come from the line
  history.

### Evidence

Measured 2026-10-16 with the read-only comparison in Appendix B. It compares
the stored tip-offs with the ones a reload would use today
(`injury_report_aiven/resolve.py: load_game_dimension`: the schedule feed plus
the game-time index, with the stale-row correction):

| season | games with a different tip-off | of which stored later than the real tip |
|---|---|---|
| 2018-19 to 2020-21 | 0 | 0 |
| 2021-22 | 2 (playoffs, 30 and 45 min early) | 0 |
| 2022-23 to 2024-25 | 0 | 0 |
| 2025-26 | 55 | 37 |

- The differences go up to 6 hours. 21 games are stored at least an hour
  late and 10 at least an hour early. Example: `0022500645` is stored at
  2026-01-24 23:00 UTC; the real tip-off is 17:00.
- All 57 corrected tip-offs agree with the line-history store's
  `lh_game.tipoff_utc`, and none of the stored ones do.
- Effect on the spans themselves:
  `scripts/perf/check_injury_window_ingest.py --first 2026-04-09 --last
  2026-04-10` (read-only, on `perf/pipeline-speedup`) rebuilds those dates'
  spans and compares them with the store. Three games differ only in where
  their spans end: `0022501173` ends at 23:30 instead of 23:00, `0022501178`
  at 00:00 instead of 01:30, and `0022501182` at 02:00 instead of 01:00.

### Proposed fix

Reload both seasons into the store, then refresh the S3 input cache:

```bash
python scripts/injury_reports/load_injury_reports_to_aiven.py --season 2025-26 --replace-season
python scripts/injury_reports/load_injury_reports_to_aiven.py --season 2021-22 --replace-season
python scripts/refresh_input_cache.py
```

- `--replace-season` is required. Without it, inserts are `ON CONFLICT DO
  NOTHING`, and the old spans stay.
- 131 stored 2025-26 games (playoffs, play-in, NBA Cup) are missing from
  today's schedule feed. They have `tipoff_source = 'injury_report'`, and the
  ingest derives their tip-offs from the reports again (`tipoffs_from_reports`
  in `ingest.py`). Check that the reload summary reports about 131 tip-offs
  taken from reports and 0 games dropped for want of a tip-off.
- The cache refresh is needed because the builders read finished seasons'
  `injury_report.*` queries from S3. 2025-26 is finished, so builds keep the
  old values until the refresher sees the changed fingerprints and
  re-uploads them.
- The tonight-only report-state memo is keyed on the store's fingerprints and
  invalidates itself.

### Verification

- The Appendix B comparison reports 0 changed games.
- `scripts/perf/check_injury_window_ingest.py --first 2026-04-09 --last
  2026-04-10` reports `RESULT: EQUAL`.

---

## 4. Injured and questionable players lose their last game

**Found:** 2026-10-16, while checking live-roster parity on
`fix/todays_prediction_data`.

**What is wrong.** `precompute_cumulative_avg_stat` gives every row the
EWMA of the player's *earlier* games (`shift(1)`), which is right for the row
of the game being predicted. A player who is Out/Doubtful (or Questionable) for
the target game has no row in it, so `latest_player_states(..., injured=True)`
and `_latest_states_on_roster(..., injured=True)` take his last row strictly
before the game. That row's value was computed for predicting *that* game, so
it excludes it. His most recent game is therefore always left out, although it
was played and including it leaks nothing.

Example: 10 points, then 30 points, then Out tonight. Training reads 10.0; an
average that includes the 30-point game reads 11.34 (EWMA with the
previous-season fill).

**Affected columns:** `TOP*_INJURED_*`, `AVG_INJURED_*`,
`TOTAL_INJURED_*`, the `QUESTIONABLE` twins, and the minutes threshold
(`MEETS_MIN_THRESHOLD`) that orders those rankings. Closing and intermediate
builds both read them.

**Why it was left.** Every trained model learned the current behaviour, and
live prediction must match training, not be more correct on one side only.
The live-roster path was aligned with the current rule on purpose.

**How to fix.**
1. For the injured and questionable groups, use the average updated with the
   player's last appearance: the value a row after that game would carry. The
   live roster path's synthetic same-day state already computes this; reuse it
   for both training and live.
2. Rebuild the closing and intermediate datasets with the other fixes in this
   document.
3. Verify that the player history memo is rebuilt for the corrected code and
   inputs. Its key covers every `nba_ou` source file, pandas/numpy versions
   and chained input digests, so these changes invalidate old entries
   automatically; do not reuse old reference datasets for parity checks.
4. Retrain and compare with the current models as a normal experiment. The
   expected effect is small (a one-game lag, and injury information is mostly
   priced at close), so judge it against the noise floor.
5. Extend the live-roster parity tests with varying statistics. With constant
   statistics both rules give the same value and hide the difference.

## 5. Roster membership lags transactions by one game

**Found:** 2026-10-16, checking whether live official rosters
(`CommonTeamRoster`) match training on `fix/todays_prediction_data`.

**What is wrong.** Training decides who belongs to a team-game in
`PlayerLookupIndex.lookup_for` (`past_injuries.py`): the player's last
box-score team strictly before the game, plus anyone the game's report lists
Out, Doubtful or Questionable. A season opener falls back to last season's
assignments. Box scores are the only transaction signal, so every roster move
is recognised one game late. Verified with the production functions on a
constructed trade (X traded from A to B after A's game 3):

| Game | Training | Official roster |
|---|---|---|
| A's next game, X gone | X still counted for A | not counted |
| B's first game with X, X unlisted or Available | X not counted for B | counted |
| Same, X listed Out/Doubtful/Questionable for B | counted (report) | counted |
| B's second game with X | counted | counted |

The same lag hits debuting rookies (not members until their first box score)
and released players (members of their old team until they play elsewhere).
The worst case is a season opener, where training uses last season's rosters.

**Affected columns:** everything built on membership: the available, injured
and questionable groups (`TOP*_PLAYER_*`, `TOP*_INJURED_*`,
`*_QUESTIONABLE_*`, `AVG_*`, `TOTAL_*`, `N_*` counts), availability effects,
All-Star vote shares and roster continuity.

**Why it was left, and the accepted mismatch.** Models learned the lagged
membership. Live prediction nevertheless uses the official roster (decided
2026-10-16), so tonight's predictions reflect trades, signings and releases.
Until training changes, live rows deliberately differ from training rows in
exactly the cases in the table above.

**How to fix.**
1. Pick a game-day membership definition that training can reproduce for
   every past game and live can reproduce before tip. Candidate: the game's
   box-score players plus its inactive list. That is close to the official
   roster on game day and is roster information, not outcome information; it
   must not depend on minutes played (see `nba_ou.config.leakage`).
2. Live uses `CommonTeamRoster` (archived per fetch under
   `train_data/live_inputs/v1/rosters/`) plus report-listed players. Measure
   how often it differs from the training definition on archived days before
   trusting the match.
3. Season openers then need no prior-season fallback.
4. Rebuild, verify memo invalidation and retrain together with the other entries
   in this document.

---

## Appendix A: cross-book numbers from a closing build (items 1 and 2)

Point it at any full closing dataset (Parquet) built with `create_df_to_predict`.

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
print("stored :", stored.describe().round(3).to_dict())
print("correct:", correct.describe().round(3).to_dict())
print("corr   :", round(stored.corr(correct), 3))
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
