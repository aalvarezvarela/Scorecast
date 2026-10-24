# Live intermediate-line predictions

Status: **Implemented** (checked 2026-10-22) — the 48 intermediate slots are
served by `scripts/predict_nba_games.py`; the open items are under "To take care
of".

Serve the intermediate slots on game days, from tonight's data, without ever
storing a partial game. The plan started with the 16 `2_5/line_error`
horizons (T-30 to T-1080); `ENABLED_MODELS` now holds those 16 horizons for each
of the three targets (`line_error`, `total_points`, `spread_error`), plus the
three closing slots.

## The rule

**Box scores and complete line history are stored only once a game is finished.**
Published injury reports are now an explicit exception: they are archived in S3
and ingested into Aiven before prediction. The database is their prediction
source at every time cutoff. Referee assignments use a separate S3 cache.
See [live injury reports](../injuries/live_reports.md).

Live roster membership uses the latest same-day official roster observed by
the run, as documented in [live rosters](../injuries/live_rosters.md). Report publication
cutoffs remain strict per horizon; an intraday roster transaction may therefore
affect a horizon that passed before the roster was fetched. Historical replay
matches training when given the same roster inputs; it cannot reconstruct
official membership before roster observations were archived.

## The design: exact rows for every horizon already passed

A snapshot at horizon `h` for a game tipping at `T` is built only from data at
or before `T - h`. So once the clock passes `T - h`, the T-`h` row of tonight's
game can be computed *exactly* as in training -- no approximation of "now" to
the nearest grid point.

A prediction run at time `now` therefore:

1. builds tonight's rows for every horizon with `T - h <= now`;
2. lets each slot predict its own row (`TIME_TO_MATCH_MIN == h`);
3. flags as **active** the smallest such `h` -- the most recent snapshot, the
   model to bet with;
4. records slot, horizon, `as_of`, the line it priced against and the
   prediction, so each slot can be settled later against its own line.

A slot's prediction never changes once its horizon has passed, so the schedule
only needs one run after each horizon, not a run at each exact minute.

**Policy since 2026-10-20:** every enabled slot predicts on every run, whatever
the time to tip. A slot whose snapshot is still ahead predicts from the row
built from the ticks so far, marked `SNAPSHOT_PASSED = False`; the list above
describes the rows that are exact.

**Parity test.** Replay a past date "as of" chosen times (live readers with
`as_of`, builder with the replayed data substituted for the stored rows) and
require the rows to equal the training dataset's for those games.

## Latest input: a model fed the run's own moment (2026-10-20)

The user bets at the time of the run, so a slot can instead predict from each
game's **latest snapshot**, whatever its own horizon: a T-360 model run at
T-47 reads the T-47 row as if it were its T-360 row. The two modes, per run:

- `latest` (`[PredictionModels] INTERMEDIATE_INPUT_MODE`, the default during
  the day): the slots of `LATEST_INPUT_SLOTS` (`all`, or `ENABLED_MODELS`
  rows) read the latest row; the others their own. The user runs it at the
  grid times by cron.
- `own` (`predict_nba_games.py --intermediate-input own`): every slot reads its
  own snapshot. Run once after the last tip-off, every snapshot is the past,
  so each slot predicts exactly its training row. That run is
  `scripts/predict_past_snapshots.py` (workflow "NBA Past-Snapshot
  Predictions", on dispatch): the intermediate slots only, on a schedule that
  keeps started and finished games (the live one drops them), snapshot rows
  still ahead left out. Its default date is the Eastern date 12 hours ago, so
  it can run from the last game's tip-off until noon ET the next day; after
  that, pass the date. Dry run on 2026-10-20 (3 games): 48/48 slots, ~13 min.

How the latest row is built (`prediction.intermediate_frame`,
`create_intermediate_line_df(extra_target_snapshots=...)`):

- Its horizon is `ceil(tip - as_of)` in minutes, 0 once the game has tipped,
  so it never falls after the run. It is added to tonight's grid only; the
  stored history of the other games stays on the grid, so no training build
  can change.
- Every value is computed as of that moment exactly as a grid row's is: the
  market as of then, the movement windows anchored on it (`move_last_W` is the
  last W minutes before it), the Ridge with its real horizon, the injury
  reports published before it, the referees once released.
- The one exception is the injury stage's calibration on earlier games (the
  players' availability and the availability-effect fit), which a full build
  only has at grid horizons: an off-grid horizon reads it at the nearest grid
  horizon (`intermediate_injuries._alias_horizons`), tonight's own report state
  still at the run's moment. The memos stay keyed on the grid.

What the model may read of it (`prediction.input_limits`), applied to the
latest input only:

- **Referees**: masked as training masks them, before 09:00 ET on game day, but
  at the model's nominal snapshot `tip - H`. A T-720 model of a 19:00 ET game
  reads no crew; of a 22:00 ET game it does.
- **Injury news** (`INJ_SNAP_*`, `ODDS_SNAP_NEWS_*`): a value becomes 0 (the
  no-news reading) where under 5% of the 2_5 training rows at H carry one --
  mostly T-480 to T-960, when no report is published overnight
  (`input_support_table`, regenerated with
  `scripts/build_input_support_table.py`). The injury status itself is in
  ~95% of rows at every horizon and is not limited.

Each prediction carries `INPUT_MODE` (`latest`/`own`), the input row's
`TIME_TO_MATCH_MIN` and `INPUT_LIMITED_COUNT` in its S3 snapshot; the
Postgres table keeps its columns until the prediction store is decided.
When SBR's feed is frozen or empty, tonight's line history comes from Covers
instead, the same feed (`prediction/line_source.py`;
`docs/odds/covers_line_history.md`, "Live fallback for predictions"). Each
row carries `LINE_SOURCE` (`sbr`/`covers`), stored as
`prediction.inputs.line_source`.

Features whose meaning grows with time elapsed (`MINUTES_SINCE_OPEN`,
`N_MOVES_SO_FAR`, ...) are left as they are: the model reads the real values,
outside its training range at a much later moment.

## Status

### Done (2026-09-27)

- **Store guard** -- `line_history_aiven.ingest.is_finished`: `ingest_scraped_games`
  drops any game SBR does not list as `Final`, for every write path (daily
  refresh, gap fill, backfill). The daily update reports them as
  `held back N unfinished game(s)`. Before this, a daytime refresh would store a
  game's pre-game ticks and, insert-only, keep that partial minute forever;
  today's games were kept out only because `nba_games` did not know them yet.
- **Live line history** -- `line_history_aiven.live.fetch_live_line_history`:
  scrapes a date into memory, runs the ingest's own `build_frames` (resolution,
  repairs, encodings) and the fetch's own `_decode_ticks`, skipping only the
  insert. Replayed on 2026-04-10 (15 games) against the store: identical tick
  keys, timing, opener flags, dtypes and game rows; <1% of line/price values
  differ, all SBR revising a minute's quote between scrapes, not the transform.
- **Builder hook** -- `create_intermediate_line_df(live_line_history=...)`
  appends the live ticks/games (live replacing stored rows per GAME_ID, which is
  what makes replays possible) and admits a season the store does not hold yet.
- Tests: `tests/test_live_line_history.py`.

### Done (2026-09-28)

- **`LiveGameDay`** (`create_training_data/live_game_day.py`) bundles tonight:
  schedule, line history as of `as_of`, referee crews if released.
  `create_intermediate_line_df(live=...)` takes it; history then stops the day
  before (`recent_limit_to_include` defaults to it).
- **Base features for scheduled games** (`create_base_game_features(scheduled_games=...)`):
  team rows appended before the rolling statistics, placeholder player rows
  for the all-star stage, tonight in the game context so the lagged roster
  report reaches it. Roster continuity deliberately does **not** see the
  placeholders -- it treats them as known on game day, which a stored game's
  own box score never is (it counts from the next day). Replay of 2026-04-10:
  every one of the 1,303 base columns that reach the intermediate dataset
  matches the training rows except the outcomes (NaN live, by design) and two
  text columns no model uses (`MATCHUP_*`, one team-name spelling). With the
  placeholders, 12 `ROSTER_*` model features had differed by up to 0.21.
- **Referees**: tonight's games join the legacy estimator as rows with no
  outcome, with their released crew; no crew means NaN (the user's rule --
  pre-09:00-ET snapshots are NaN in training too). The tendency features take
  the crews through their existing `df_referees_scheduled` path. Tested in
  `tests/test_intermediate_referees.py` (a live game gets exactly the features
  it would get once stored).
- **Whole-builder replay, 2026-04-10 as of 16:00 ET** (same code, training
  mode vs live; 171 rows for the 13 horizons already passed, 3,193 columns;
  ~25 min and 4.2 GB per build):
  - all non-player, non-line columns match exactly;
  - `ODDS_SNAP_*` (264 columns, a few rows each): SBR's own revisions -- the
    as-of cut is exact (identical 6,059 tick keys and opener flags), 49 ticks
    carry a revised value;
  - player features. Injury *status* comes only from the injury report. The
    *roster* -- who belongs to the team -- comes from box scores
    (`create_player_lookup`): within the game's season bucket (regular season
    for a regular-season game), every player whose last team strictly before
    the game is this one, plus anyone the game's report lists for the team.
    Only if that bucket is still empty (opening night, a first play-in game)
    does it fall back to the season year -- i.e. the preseason roster, camp
    cuts included -- and then to the previous season's last assignments.
    Tonight has no box score, so placeholders are added for it, but
    `standardize_and_merge_scheduled_games_to_players_data` builds them from
    every player's latest row across all loaded seasons: 9-16 extra players per
    team (former players, camp cuts), and on opening night it also blocks the
    fallback. `_same_bucket_roster` keeps only players with a row in tonight's
    bucket; checked with the real lookup it returns the training roster for
    4/4 team-games on the 2025-10-21 opener, 24/24 on 2025-10-22, 24/24 on
    2025-10-24 and 30/30 on 2026-04-10. (Tried and wrong: last 1/10 games --
    drops in-season players; season year -- keeps ~7 camp cuts by day 3.)
  - traded players: in both training and live, a player is on his old team
    until his first box score for the new one -- e.g. 12/12 traded players in
    2025-26 were absent from the new team's roster on their debut, some playing
    24-26 minutes -- unless that game's injury report lists him for the new
    team. Consistent between training and live; a shared blind spot a
    transactions feed could close.
  - The claimed training-side look-ahead (roster from the game's own box
    score) was mistaken and is withdrawn.
- **Replay harness**: `create_training_data/live_replay.replay_scheduled_games`
  turns a played date into `get_schedule_games` rows (the live fetcher drops
  them, since a finished game's status text is "Final", not a tip time).

### To do, in order

1. ~~Base features for scheduled games.~~ Done above.
2. ~~Referees.~~ Done above.
3. ~~Injuries per horizon~~ -- done: ingestion implemented. The scheduled-data path
   synchronizes yesterday's and today's published PDFs into S3 and Aiven.
   `add_snapshot_injury_features` reads that timeline at each horizon's strict
   publication cutoff; it does not need an in-memory PDF overlay.
   `fetch_live_game_day` also synchronizes current runs. Explicit `as_of` reads
   stored state without refreshing. The archived-PDF overlay remains available
   to the historical replay harness, separate from current prediction reads.
   The daily finished-day ingest from `perf` remains in the database workflow.
4. ~~Market dynamics and ridge~~ -- done (d233c06): fitted over the stored
   history rows merged with tonight's in the full build's order.
5. ~~Leakage gate and targets~~ -- tonight's rows pass (NaN outcomes kept).
6. ~~Prediction job~~ -- done (4de9d8e): `prediction.intermediate_frame`
   builds tonight's rows tonight-only and keeps the passed snapshots; an
   intermediate slot predicts the rows of its own horizon
   (`load_registry_model_and_predict(frames=...)`). Runtime: the tonight-only
   mode (`create_intermediate_line_df(target_game_ids=...)`) computes only
   tonight's rows; every earlier game comes from per-season stored history
   (`intermediate_history`, `PersistentRowCache`), see
   `docs/pipeline/speedup.md`.
7. ~~Parity test~~ -- `point_in_time_parity.py --kind intermediate --targets`
   (tonight-only build == full build rows, dataset and scoring) and
   `live_replay_parity.py` (live tonight-only == live full, replayed date).
   These harnesses are retained; their previous acceptance results do not
   certify this integration until rerun on the merged commit.
8. ~~Daily intermediate refit~~ -- done: one-horizon intermediate slots are
   refitted daily from the season partitions (`docs/serving/model_registry.md`
   §7 items 10-11). The injury store also holds published
   reports for upcoming games, read under each snapshot's strict time cutoff.

The 2026-27 regular season starts around 2026-10-20; until then everything is
tested by replaying past dates.

## To take care of (deferred, not forgotten)

- [x] **Game-day injury-report ingest.** The prediction scheduled-data path
      archives new immutable PDFs in S3 and atomically updates affected Aiven
      game timelines, including empty/unchanged reports and retry receipts.
- [x] **Live injury-report source.** Aiven supplies current prediction states;
      archived-PDF overlays are retained only for replay diagnostics.
- [x] **Daily finished-day injury ingest.** The database workflow retains
      `scripts/injury_reports/daily_injury_reports.py` to rebuild finished days.
- [ ] **Referees on the prediction side** can be missing (assignments are
      released ~9am ET). Missing means NaN, exactly as training masks them
      before release -- never a fallback value.
- [ ] **Prediction schedule**: the predictor workflow runs every 30-60 min on
      game days (crons are currently commented out in
      `.github/workflows/nba_predictor_daily.yml`).
- [x] **Enable the slots** in `ENABLED_MODELS` -- done: 51 rows, the closing
      slot and 16 horizons for each of the three targets.
- [ ] **Archive the old CSVs** in `data/train_data/` to S3, then delete them.
      Builders now write Parquet only, and a config reads exactly the file it
      names, so the CSVs serve only configs still pinned to them. The two 2.5
      Parquet copies predate the identifier-dtype fix and fail cleaning; rebuild
      instead of reusing them.
- [x] **Commit** the parquet, naming, daily-job and live line-history changes
      -- done (Parquet in 322a91a).
- [x] Pre-existing lint in `line_history_aiven/tipoff_corrections.py` (import
      order) -- clean now.
- [x] **Closing prediction path: same roster inflation** (fixed 2026-10-02).
      The shared `standardize_and_merge_scheduled_games_to_players_data` now
      builds a placeholder only for players with a row in tonight's season
      bucket, so both prediction paths get training's roster. Prediction-only:
      training builds never call it. Checked on the closing path's data with the
      real lookup: rosters identical 4/4 (2025-10-21 opener, 0 placeholders ->
      same fallback as training), 24/24 (2025-10-24), 30/30 (2026-04-10);
      roster continuity identical on the first two, 7/30 team-games off by at
      most 0.0005 on 2026-04-10 (was up to 0.21). Before: e.g. Charlotte 37
      players vs 19, Lakers 30 vs 18. Tests in tests/test_scheduled_player_rows.py.
- [x] **Residual player-feature gap: verified, a small training-side dependence
      on tonight's box score** (2026-10-02). Fixed in training as item 9 of
      `docs/pipeline/defect_fixes_2_5.md` (date-based `_AFTER` selection) and in
      the 2026-10-16 rebuild. With identical rosters, a player's
      stats come from his latest row on or before game day
      (`latest_player_states`, available branch). In training that is tonight's
      box-score row if he appeared (stats already shifted, so pre-game
      averages), otherwise his previous game's row -- whose average stops before
      that game, one game stale. Live, every roster player has tonight's
      placeholder. On 2026-04-10, 589 roster players: all 400 who appeared in
      the box score match exactly; 117 of the 189 who did not differ (all of them
      taken from an earlier row in training); the other 72 had nothing new in
      their last game. In the pipeline the out/questionable players are removed
      first, so the affected ones are those who neither dressed nor were listed
      (two-way, G League, not with team). Effect on features: active-weighted
      ratings ~0.11 vs a spread of 3.8, top-player stats mostly tiny, one team's
      `TOP3_AVAILABILITY_EFFECT_*` ~1.8. Same mechanism in the closing dataset
      (shared `add_player_history_features`). Live cannot reproduce it -- it
      would need to know who will be in tonight's box score. The fix belongs in
      training: give every roster player a state as of game day regardless of
      whether he appeared. That changes the datasets, so it needs a rebuild and
      retraining; best folded into the next planned dataset rebuild.
- [ ] **SBR revises quotes after the fact**: ~0.8% of a day's ticks (49 of
      6,059 on 2026-04-10) have a different line/price when re-scraped the next
      day. Live features see the then-current view; the stored history (and so
      training) the revised one. Inherent to reading live; small, but it is the
      floor on how exactly live rows can match training rows.
