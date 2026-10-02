# Live intermediate-line predictions

Serve the 16 promoted `2_5/line_error/t0030..t1080` slots on game days, from
tonight's data, without ever storing a partial game.

## The rule

**A game is stored only once it is finished.** Tonight's line history, injury
reports and referee assignments are read into memory for the prediction and
written the day after, whole. This is the rule the closing-odds tables already
follow (`get_all_info_for_scheduled_games` scrapes today's odds into
`scheduled_data`; nothing is inserted).

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

**Parity test.** Replay a past date "as of" chosen times (live readers with
`as_of`, builder with the replayed data substituted for the stored rows) and
require the rows to equal the training dataset's for those games.

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
3. **Injuries per horizon.** `add_snapshot_injury_features` reads the report
   timeline from the Aiven injury store, which is loaded season-at-a-time from
   archived PDFs by a manual script -- nothing updates it daily. Needs (a) a
   daily ingest of finished days' reports (same finished-only rule) and (b) a
   live, in-memory parse of today's reports published so far.
4. **Market dynamics and ridge move-to-close** over history plus tonight; the
   ridge stays fitted on completed games only.
5. **Leakage gate and targets**: tonight's rows carry no outcome and must
   survive the target step without being dropped.
6. **Prediction job**: build the intermediate frame for passed horizons, select
   each slot's row, add `"intermediate_line"` to
   `nba_ou.prediction.prediction.SERVABLE_DATASET_TYPES`, add the active flag
   and the recorded fields above; run every 30-60 minutes on game days.
   - **Keep only passed horizons** (`TIPOFF_UTC - TIME_TO_MATCH_MIN <= as_of`).
     The builder also emits rows for horizons still ahead; those are built from
     the latest ticks so far, not from the real snapshot, and must never be
     predicted from or recorded.
   - **Runtime.** A live build recomputes the whole season's history (base
     features, prior-game line dynamics, the walk-forward ridge) to produce a
     handful of rows -- too slow to repeat every 30 minutes. Cache the
     history-only parts once a day (they cannot change until tonight's games
     are stored) and rebuild only tonight's rows per run; the replay test
     guards that the cached path still matches.
7. **Parity test** end to end on replayed dates.
8. **Daily intermediate refit** (`docs/model_registry_reorg_plan.md` section 5):
   rebuild from the store, which now only ever holds finished games.

The 2026-27 regular season starts around 2026-10-20; until then everything is
tested by replaying past dates.

## To take care of (deferred, not forgotten)

- [ ] **Daily NBA injury-report ingest.** The Aiven injury store is loaded
      season-at-a-time from archived PDFs by
      `scripts/injury_reports/load_injury_reports_to_aiven.py`; nothing updates
      it daily, so in-season it goes stale and the per-horizon injury features
      (`INJ_SNAP_*`, market dynamics) lose their recent reports. Add a daily job
      that stores *finished* days' reports (same rule as line history) --
      mind that spans are built from contiguous runs of reports per season.
- [ ] **Live injury reports** for tonight's games: parse the reports published
      so far into spans in memory, never stored (stage 3 above).
- [ ] **Referees on the prediction side** can be missing (assignments are
      released ~9am ET). Missing means NaN, exactly as training masks them
      before release -- never a fallback value.
- [ ] **Prediction schedule**: the predictor workflow runs every 30-60 min on
      game days (crons are currently commented out in
      `.github/workflows/nba_predictor_daily.yml`).
- [ ] **Enable the 16 slots** in `ENABLED_MODELS` once live serving works.
- [ ] **Archive the old CSVs** in `data/train_data/` to S3, then delete them
      (verified Parquet copies exist for the two 2.5 datasets).
- [ ] **Commit** the parquet, naming, daily-job and live line-history changes.
- [ ] Pre-existing lint in `line_history_aiven/tipoff_corrections.py` (import
      order) -- unrelated, left alone.
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
- [ ] **Residual player-feature gap: verified, a small training-side dependence
      on tonight's box score** (2026-10-02). With identical rosters, a player's
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
