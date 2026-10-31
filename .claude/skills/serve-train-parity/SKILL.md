---
name: serve-train-parity
description: How to check that the rows a prediction read equal the rows training builds for the same games afterwards, and how to trace a disagreement to its cause. Use when live or past-snapshot predictions look worse than expected, after a fix to the prediction or dataset builders, when asked whether "yesterday's prediction rows match today's training rows", or before trusting a historical backfill.
---

# Serve/train parity: today's prediction rows vs the training build

A model is only evaluated on what it was trained on if the row it predicts
from is built like its training rows. Two code paths build them: the
prediction builders (tonight-only, live stores, `prediction/`) and the
dataset builders behind the season partitions (target mode, after the game,
`create_training_data/season_partitions.py`). Every disagreement found so far
was silent: no error, plausible values, a slightly different feature.

## The check, in order

1. **Pick a played date D** whose next-day partition update has run (the daily
   refit at ~07:00 UTC on D+1). Check the partition's `written_at` and
   `git_commit` in the manifest: a partition written by older code compares
   old code, not today's.
2. **Run the comparison**:

   ```bash
   poetry run python scripts/perf/serve_train_parity.py --date D --impact --out DIR
   ```

   It reads the run archive (`prediction_snapshots/v2/game_date=D/`): the
   intermediate rows of D's last past-snapshot (`own`) run, every passed
   snapshot, and each game's closing row from the last daily run before its
   tip; the training rows are the current `2_5` season partition. Rows match
   on `GAME_ID` (+ `TIME_TO_MATCH_MIN`). `--impact` serves every enabled slot
   on the prediction rows and on the same rows with the training values of the
   differing columns, so each mismatch gets a size in points and side flips.
3. **Sort the differing columns into families** (below) before reading any
   one of them. A family that moves together has one cause.
4. **Rank by impact, not by row count.** A column most models do not read can
   differ on every row and move nothing; read `--impact` first. Check the
   specs' features (`resolve_config_spec(...).features.feature_names`).
5. **Find the cause in the data, then in the code.** Recompute the feature
   from the store as it is now (if that equals training, the store changed
   after the prediction read it), inspect the raw source (report PDFs, ticks),
   and only then the builder. Run the reproduction for several dates: a cause
   that shows up every day is structural.

## Expected differences (not bugs)

| Columns | Why |
|---|---|
| `TOTAL_POINTS`, `LINE_ERROR`, `SPREAD_ERROR`, `HOME_MARGIN`, `PTS_*`, `IS_OVERTIME`, `MATCHUP_*` | outcomes, missing before the game |
| `SNAPSHOT_TS_UTC`, `SNAPSHOT_PASSED`, `GAME_TIME`, `LINE_SOURCE`, `LATEST_INPUT` | prediction-only bookkeeping |
| closing: per-book lines and prices, `ODDS_*books*`, `THIS_GAME_CROSSBOOK_*`, `ODDS_implied_*` | the last live run is minutes before tip; the training close is the last pre-tip quote |
| closing: `ODDS_*_pct_*`, `*_consensus_pct_*` | public percentages at run time vs final |
| closing: `LAST_STATUS_REPORT_AGE_MIN_*` | report age at run time vs at tip |
| `TEAM_NAME_*` | display names from two feeds ("LA Clippers") |

The intermediate `own` rows are the clean test: same snapshot instant, same
history cutoff. Any non-outcome difference there needs a cause.

## Causes found so far (check these first)

- **Live vs batch injury-report store.** Before tip the predictor reads spans
  written incrementally (`fetch_data/injury_reports/sync.py` ->
  `injury_report_aiven/live_ingest.py`); the daily job rebuilds finished
  games with the batch ingest (`daily_injury_reports.py` -> `ingest_season`)
  and training reads that. Symptom: `INJ_SNAP_*NEWS*` and the
  `ODDS_SNAP_NEWS_*` columns built on them differ while the report-state
  columns match. Compare the store's spans for tonight's games with
  `live.build_live_injury_reports` (the batch code, dry run) at the same
  cutoff. 2026-10-30: a player_id int/text split made players flicker
  Out/not-listed/Out (fixed in PR 102); spans the old code wrote survive
  incremental rebuilds, because `read_existing_observations` replays them.
- **Closing-line repair.** Per-book closes in `odds_sportsbook` are rewritten
  from the line history (`data_processing/odds/closing_line_repair.py`), and
  the same repair runs again in every build (`merge_odds_data`). A feed that
  stops recording one book makes its stale quote the "close"; it then enters
  every rolling `*_<book>_*` feature, in training and in tonight's history
  alike. Check each book's distance from the others' median per season and
  `odds_sportsbook_sbr_raw` against the repaired value.
- **Build scope.** Season-level and expanding features computed over the
  history a build loads (`*_SEASON_BEFORE`, head-to-head, availability
  effects) differ when the tonight-only build loads less history.
- **Day-page values the training row has and the snapshot does not**
  (the opener, public percentages): fixed for the intermediate opener in
  PR 102; the closing backfill fills the consensus opener from the stored row
  (`prediction/historical_inputs.py`).
- **Games missing from training.** A game whose line history stops early is
  held back for 3 days by the daily SBR and Covers line-history updates; it is predicted
  but absent from the intermediate partition meanwhile. Not a mismatch.

## Traps

- **Stores move under you.** The store a prediction read at 05:00 UTC can be
  rewritten at 07:00 UTC. Reproduce with the current store first; if that
  equals training, the difference came from the store's earlier state.
- **History memos.** Finished seasons' repairs and features are memoised with
  a frozen code key (`create_training_data/history_memo.py`): a code fix
  reaches past seasons only when they are rebuilt with the memo bypassed.
- **Partitions are versions.** Rebuilding a season changes training rows the
  older prediction archives cannot match; compare dates whose archive and
  partition were produced by the same code, or say which changed.
- **One date is a sample.** Run three or more dates; a cause that is real
  usually shows on all of them.
- **Do not refit to test parity.** Serve the production builds on both
  frames; a refit adds model noise that hides the input effect.

## Related tools

- `scripts/perf/live_replay_parity.py`, `point_in_time_parity.py`,
  `target_parity.py`: builder-level equality of tonight-only vs full builds.
- `scripts/perf/check_live_injury_reports.py`: report states, live vs stored.
- `scripts/create_train_data/compare_season_partition.py`: a season rebuilt in
  memory against its stored partition, before paying for a rebuild.
- `scripts/create_train_data/build_season_partitions.py rebuild`: rewrite a
  season after a fix (closing a few minutes, intermediate up to ~40).
