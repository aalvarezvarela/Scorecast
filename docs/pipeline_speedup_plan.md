# Speeding up the 2_5 data pipelines

Scope: the closing-line builder (`create_df_to_predict`) and the
intermediate-line builder (`create_intermediate_line_df`) at schema `2_5`, for
three uses: historical training builds, the daily refit, and same-day
predictions. Branch `perf/pipeline-speedup`, to be merged back into
`feat/pregame-rotation-features`.

Later schemas are layers over this base (`create_training_data/schema_layers`:
2_6 is a 2_5 frame plus added columns), so every 2_5 speed-up carries over to
them. Two layer-specific points are listed under phases 1 and 4.

**Ground rule: no optimization may change a value.** Every step below is
accepted only when `compare_builds` (work item 0.1) shows the new build equal
to the reference build on every row and column. Rows may change only where a
step is *meant* to fix train/serve parity (work item 4), and then only for the
columns that step names.

## Measurements (2026-10-14)

Profiled on this machine (4 cores, 15 GB) against the live databases (Supabase
for games/players/odds, Aiven for line history and injury reports).

**Closing, 2 seasons to 2026-04-10** (2,537 games, the scope today's
prediction uses): 9:20 wall under cProfile, 1.4 GB peak. cProfile inflates the
CPU-bound stages, so treat the split as relative.

| Stage | Time | Kind |
|---|---:|---|
| Odds load + closing-line repair (`fetch_closing_ticks`, ~10 s/season server-side, done twice) | 143 s | remote DB (111 s waiting) |
| Player history loop (`add_player_history_features`) | 126 s | Python, per team-game |
| Referee history inputs (6 more seasons of games + odds, repaired again) | 94 s | mostly remote DB |
| Legacy `REF_AVG/STD/SUM` (`compute_referee_features`) | 60 s | Python, per game |
| Rolling team stats (of which trend slope 32 s) | 50 s | Python `rolling.apply` |
| Availability effects / all-star / box history / games+players load | 26 / 24 / 19 / 22 s | mixed |
| 23 database connections opened | 13 s | connect overhead |

**Intermediate, season 2025-26 only** (20,645 rows, 17 horizons): 25:20 wall,
4.1 GB peak, py-spy sampling.

| Stage | Time |
|---|---:|
| Snapshot injury features, per horizon (`add_snapshot_injury_features`) | 13.3 min |
| &nbsp;&nbsp;player history 5.2 · availability effects 3.5 · all-star 2.9 · report features 1.5 | |
| Referee features (reloads 6 seasons of games/odds/refs) | 4.2 min |
| Base game features (3 seasons incl. lookback) | 3.9 min |
| Closing-tick fetch for the repair (inside base + referee) | 3.3 min |
| Pre-game tick fetch | 1.3 min |
| Snapshot panel, movement, pivots, market dynamics, ridge, audit | ~2.5 min |

A full intermediate build (store seasons 2021-2025 + lookback) was not timed.
The snapshot side scales with seasons, so expect well over an hour.

### Train/serve finding: the 2-season prediction scope changes features

The 2-season build above was compared with the stored all-season file
`closing_line_data_2_5_20261003.parquet` on the 1,216 games of 2025-26 they
share. 1,847 of 1,874 columns are identical. The 27 that differ depend on how
much history is loaded, not on the code:

- `TOP3_AVAILABILITY_EFFECT_*`, `TOP3_INJURED_*`, `TOP2_QUESTIONABLE_*`: the
  empirical-Bayes shrinkage is fitted on every earlier row *loaded*. Values
  differ on 95-100% of rows. `*_DIFF_FROM_LINE` correlates 0.96 with the
  full build. `*_SPREAD_ERROR` is constant in the full build (no
  between-player variance, so every effect shrinks to zero) but has a
  median |value| of about 0.5 in the 2-season build.
- `LAST_N_GAMES_TOTAL_POINTS_*` (head-to-head): NaN when the earlier
  meetings are outside the window. The real prediction path partly covers this
  with `extra_game_ids`; this training-mode replay does not.
- `BENCH_*`: under 1% of rows.

So the closing models are served features from a different history scope than
they were trained on. That is the same problem already recorded for the
intermediate dataset (`*_SEASON_BEFORE`, effect shrinkage). **Point 2 is
therefore a correctness fix as well as a speed-up**: the fast path must load
the training build's full history, which is only affordable once the work below
is done.

## Where the time goes, in one paragraph

About half the wall time is waiting on remote databases. The same tables are
read several times per build: games three times in intermediate; injuries,
box history and all-star voting twice; odds repaired once for the output
seasons and again for the referee history. The rest is per-row Python loops:
player history, all-star, availability effects, legacy referees and the
rolling slope. Each one does pandas boolean-mask filtering inside the loop.
The intermediate builder then repeats the player, all-star and effect stages
once per horizon. And same-day prediction runs all of this over full seasons
to keep about ten rows.

## Plan

Ordered by value per effort. Phases 1-3 are pure speed-ups and change no
value. Phase 4 is the prediction fast path and the parity fix. Phase 5 makes
daily datasets incremental.

### Phase 0 — Safety net (do first)

**0.1 `scripts/perf/compare_builds.py --exact`** is the acceptance test.
Rows are matched on `GAME_ID[, TIME_TO_MATCH_MIN]`, and the two builds must
have:
- the same columns, in the same order;
- the same rows, in the same order;
- the same dtypes;
- every value equal bit for bit (NaN equals NaN; -0.0 and 0.0 are different);
- the same frame digest.

Without `--exact` it reports differences by column, for diagnosis only.

**0.2 Reference builds with frozen inputs** (`scripts/perf/`). The databases
change daily, so a reference is the output *and* the inputs it was built from:
- `db_tape.py` patches `psycopg.Cursor` and records the rows of every query a
  build runs to a tape. Replaying the tape answers each query without
  reaching the server, so a replayed build sees exactly the reference's
  inputs. Replay is strict: a query not on the tape raises a
  `BaseException`, because several loaders catch `Exception` and would
  otherwise return `None` without anyone noticing.
- `reference_build.py` builds closing or intermediate in record, replay or
  live mode. It writes the dataset with the pipeline's own
  `write_training_dataset`, plus a manifest (commit, arguments, timings,
  digest). It pins `PYTHONHASHSEED=0` and takes `--code-root`, so the code
  under test comes from a worktree pinned to one commit, not the shared
  checkout.
- `run_reference_suite.sh` builds the references at 7bd96a9
  (`artifacts/perf/ref_7bd96a9/`):

  | Reference | Self-checks |
  |---|---|
  | `closing_2s`: closing, 2 seasons, limit 2026-04-10 | replay == record; second live build == record |
  | `inter_2025`: intermediate 2025-26, limit 2026-04-10 | replay == record (dataset and scoring) |
  | `closing_full`: closing, all seasons, limit 2026-10-03 | == stored `closing_line_data_2_5_20261003.parquet`; replay == record |
  | `inter_full`: intermediate, all store seasons, limit 2026-10-03 | == stored `intermediate_line_data_2_5_20261003*.parquet`; replay == record |

**How a speed-up PR is accepted.** Run its build in replay mode from the
matching tape, then run `compare_builds --exact` against the reference. The
fast pair (closing 2s, intermediate 2025) gates every PR. The full pair gates
merging back into `feat/pregame-rotation-features`. Changes that move or
merge queries (phase 1) can't replay the old tape key for key. For those,
the new input layer is first checked to return frames equal to the
tape's, and then the build is compared as above.

**Development loop: `stage_harness.py`.** A full replayed intermediate build
takes 20+ minutes. `stage_harness.py capture` saves one stage's inputs and
output from a replayed reference build (`artifacts/perf/stage/inj_2025`, the
snapshot injury stage of `inter_2025`). `stage_harness.py run` then reruns only
that stage with the code under test, answers its queries from the same tape,
and compares the result `--exact`. It is a development check only.
`run_candidate_suite.sh <worktree> <ref dir> <out>` is the acceptance: it
replays every reference tape with a pinned candidate commit and compares each
output with its reference.

**0.3 Stage timer for the closing builder**, like the intermediate
`_stage_logger`. It prints `[mm:ss]` per stage so every PR can state its
before/after numbers.

**0.4 Append-invariance test** (needed by phases 4-5). Build to `D` and to
`D+7` with the same history start, and require every row dated `<= D` to be
identical. If a stage fails it, it is not walk-forward. It must be fixed or
listed as a known exception before rows are built incrementally.

### Phase 1 — Read each input once, from a local cache (largest, simplest win)

**1.1 In-process memoization.** Add a per-build loader layer so
`load_games_from_db`, `load_players_from_db`, `get_injury_data_from_db`,
`get_refs_data_from_db`, `load_player_box_history`,
`load_all_star_voting_from_db`, the injury-report state and
`load_and_merge_odds_yahoo_sportsbookreview` are fetched once per
`(table, season)` and sliced in memory. Reuse one connection per database
instead of 23. This removes the duplicate reads with no storage decisions.

**1.2 Parquet cache of the raw inputs, per season.** Store it under
`data/cache/inputs/<table>/season=YYYY.parquet`, with a manifest recording the
row count, the max game date and a content hash per partition.
- Finished seasons are immutable and cached until invalidated. The DB-repair
  scripts (closing-line repair, spread repair, tipoff fixes) bump the
  manifest, or the cache is rebuilt with `--refresh`.
- The current season is refreshed incrementally: only games after the
  cached max date.
- **Cache the repaired closing odds**, not only the raw tables. The repair
  (`_repair_from_line_history`) is the single most expensive read: about
  10 s of server time per season, repeated for the referee history. Its result
  for a finished game never changes unless the store is corrected.
- The pre-game line-history ticks (`fetch_pregame_ticks`) for finished
  seasons go in the cache too.

Expected: removes most of the 143 + 94 s DB time from the 2-season closing
build, and 8-10 min from the one-season intermediate (estimate, verify with
0.3).

**1.3 Faster reads for what still hits the DB.** Use `COPY (...) TO STDOUT`
or ADBC/connectorx instead of `pd.read_sql_query` on a raw psycopg
connection, which converts row by row in Python. Run the independent loads
concurrently in threads, since they are I/O-bound.

**1.4 S3 mirror of the cache** (optional, small once 1.2 exists).
`s3://<bucket>/data_cache/<table>/season=YYYY.parquet`, using the same
manifest. It matters for the retrain job: it runs on `ubuntu-latest` and
currently rebuilds the full closing frame from Supabase. It also lets a second
machine share the cache. The local cache stays primary on the self-hosted
runner.

**1.5 Layer inputs go through the same loaders.** `schema_layers.inputs` and
`LayerContext` load each layer's inputs (lineups, stints and so on) once per
run. Those loads should use the 1.1/1.2 layer too, so a 2_6 build does not
re-query what the 2_5 base already read.

### Phase 2 — CPU hot spots, same values

Each item is its own PR with a unit test that checks it against the old
implementation on random data, plus a 0.1 comparison against the matching
reference build.

1. **Trend slope** (`rolling.compute_trend_slope`, 32 s closing / 38 s one
   intermediate season). It calls a Python `calculate_slope` per window;
   replace that with an array implementation. Keep the semantics: drop NaN
   inside the window, re-index x as 1..n, return 0 below 2 points. Either a
   numpy sliding-window version or numba. Numba is not a dependency yet;
   numpy first.
2. **Legacy referee features** (`compute_referee_features`, 56 s closing;
   `_legacy_referee_features`, 83 s intermediate). The per-game loop
   re-filters the season frame by date and referee. Replace it with per-(season,
   referee) cumulative sums and counts ordered by date: "with" is the prefix
   sum, and "without" is the season prefix total minus "with". The previous-season
   fallback reads the full previous season, which is a constant per (season,
   referee).
3. **Player history loop** (126 s closing, 309 s one intermediate season).
   `player_lookup` (`past_injuries._lookup_by_key`), `latest_player_states`
   and `rank_player_states` build pandas frames per team-game. First move
   their internals to numpy, with per-team sorted arrays and `searchsorted` for
   "before date", keeping the existing tie-breaking order. Then, if still
   heavy, split rows into chunks across processes: rows are independent given
   the lookups, and a fork shares the lookups copy-on-write.
4. **All-star loop** (24 s closing, 174 s intermediate because it reruns per
   horizon). Same treatment, same per-row independence.
5. **Availability effects** (26 s closing, 208 s intermediate). The
   `lru_cache` on `_cached_effect` is rebuilt on every call. Hoist it so the
   three families in a build share it, and so do the 17 horizons in
   intermediate: the raw effect depends on (team, season, date, player) and the
   availability map, not on the horizon. The per-horizon part is only the
   empirical-Bayes weighting.
6. **`groupby(...).transform(lambda ...)` in rolling stats** (53 s of
   `_transform_general` in the closing profile). Use the vectorised
   `groupby().shift()` + `groupby().rolling()` forms where the lambda is a
   plain shifted mean/std.

**Iteration 2 (36e707b, 21cb9a9, 8c4acb7; 2026-10-14): 2.1-2.3 done, all exact.**
- 2.3 player loop: the roster lookup returns row positions in its season
  bucket (columns cached per bucket), and the split, latest-state and ranking
  steps run on arrays with the same stable sort keys and the same Python
  scalars. The frame path is kept as the test reference (`ARRAY_ROSTER_STEPS`).
- 2.2 legacy referees: the participation mask is computed once per referee
  and date for all metrics, not once per metric. It is not prefix sums: pandas
  means would sum in a different order and change the last bits.
- 2.1 trend slope: `_linregress_slope` repeats `np.cov`'s own steps (same
  `dot(X, X.T.conj())`, so the same BLAS path), 59 -> 16 us per window,
  0 mismatches on 160k random windows.

Acceptance (`run_candidate_suite.sh` at 8c4acb7): all four builds EQUAL,
2,280 tests pass.

| Build | Reference replay | Iteration 1 | Iteration 2 |
|---|---:|---:|---:|
| closing, 2 seasons | 280 s | 291 s | 159 s |
| intermediate 2025-26 | 1,174 s | 745 s | 506 s |
| closing, all seasons | 1,060 s | 904 s | 443 s |
| intermediate, all store seasons | 4,295 s | 2,592 s | 1,818 s |

What remains in a replayed closing build is mostly input handling: the
closing-line repair fetch converting rows and connecting, the odds merge and
the referee-history reload. In a live build that is database wait, so Phase 1
is next. Decided 2026-10-14: the persistent cache stays local, and every build
validates each cached (table, season) with a read-only server-side
fingerprint (row count + content hash, 1.4-2.3 s for a line-history season
against 15-29 s for the repair fetch).

### Phase 3 — Intermediate: stop recomputing per horizon

The per-horizon injury stage is 53% of the intermediate build.
`RowCache` already skips player rows whose report did not change between
horizons. Extend the same idea:

- all-star rows and the per-player raw availability effects keyed the same way
  (2.5 above), so a horizon only recomputes the team-games whose report
  changed since the previous horizon;
- the referee stage reads its six seasons from the 1.2 cache instead of
  reloading;
- if still dominant, run horizons in a small process pool. Memory limits it
  to about 2 workers on this machine for a full build, which is why
  de-duplication comes first.

**Iteration 1 (4d5b496, 2026-10-14): done for the per-horizon inputs.**
The report-independent inputs are now prepared once instead of 17 times:
- the stat players, their sort, the roster lookup index, the bench and
  prior-roster lookups (`PreparedPlayerHistory`, `PlayerLookupIndex`);
- the All-Star players, votes and indexes (`PreparedAllStarInputs`);
- the box history and player form (`PreparedBoxHistory`);
- the raw availability effects, shared by the three families and all horizons
  through a `RowCache` bound to a digest of the game history.

The same commit also makes four exact per-call changes, used by closing as well:
- the availability loop reads only its ~12 columns of the ~2,000-column frame;
- team-history windows and player memberships are memoised;
- the All-Star roster lookup returns ids without building frames, keeping the
  set's insertion order, because float sums over it depend on that order;
- player timelines are binary-searched.

Acceptance (`run_candidate_suite.sh`, every reference tape replayed with
4d5b496): all four builds are EQUAL, scoring files included.

| Build | Reference replay | Iteration 1 replay |
|---|---:|---:|
| closing, 2 seasons | 280 s | 291 s |
| intermediate 2025-26 | 1,174 s | 745 s |
| closing, all seasons | 1,060 s | 904 s |
| intermediate, all store seasons | 4,295 s | 2,592 s |

Harness result, season 2025, under py-spy with another build running:
885 s -> 374 s, bit-identical (frame digest equal). What remains is mostly
the first horizon's player loop, where every row is a RowCache miss. That
is Phase 2.3 (per-row pandas in `latest_player_states`/`rank_player_states`),
not per-horizon work. Still open from this phase: the referee stage reading
from the 1.2 cache, and `build_status_events`, which differs per horizon only
through the drift's covered games.

### Phase 4 — Same-day predictions: full history, compute only tonight

The prediction path today builds two full seasons to predict about ten games.
Phase 1 makes full-history loads cheap. This phase makes the *per-row* work
proportional to the games being predicted.

**4.1 `target_game_ids` through both builders.** `None` is today's behaviour
(every row). With a set, the cheap vectorised stages (rolling stats,
odds features, travel, global market, referee tendencies) still run over all
loaded rows, because their outputs feed later rows. The per-row emitters
(player history, all-star, roster continuity, injury-report features,
availability effects, legacy referees) produce rows only for the targets.

Two dependencies have to be split rather than skipped:
- **Availability map.** The effects read every earlier game's
  `{available, injured}` sets, which come out of the player loop. Split
  `add_player_history_features` into a membership pass (roster minus realised
  injured, all rows, cheap once 2.3 is done) and a feature pass (target rows
  only).
- **Shrinkage fit.** The empirical-Bayes weight for a row is fitted on the raw
  effects of every earlier row. Option A recomputes the raw effects for all
  rows; it is cheap after 2.5. Option B persists them beside the dataset.
  Start with A and measure.

**4.2 Prediction loads the training build's history span**
(`older_season_limit=None`, the same start as the training build). Drop the
`extra_game_ids` head-to-head workaround once full history is loaded. This
fixes the scope finding above. It moves the 27 listed closing columns, and only
those, which 0.1 must show.

**4.3 Parity test for both pipelines.** For a past date `D`, a target build
for D's games must equal the full build's rows for `D`. The intermediate side
already has the replay harness (`live_replay`, `LiveGameDay`). The closing side
replays with `todays_prediction=True` from stored scheduled data.

**Layers in target mode.** `apply_layers` takes a 2_5 frame. In target mode
it receives only tonight's rows, so every layer must either compute from its
own history inputs (via `LayerContext`) or declare that it needs the base's
history rows. The layer contract should state which, and the 4.3 parity
test runs per schema version.

**4.4 Baselines and TabPFN.** The baselines read only `TOTAL_POINTS` and
`GAME_DATE` from the history rows, so give them those from the cached games
table. The TabPFN client trains on history *features*. It is disabled in the
daily workflow (`--no-tabpfn`). If it is re-enabled it should read the stored
training dataset, not the prediction frame.

Expected: the closing prediction frame goes from about 10 min to the cost of
loading the cache plus the vectorised stages, likely 1-2 min after phases 1-2
(estimate). Intermediate live predictions become feasible at full scope, which
today's code cannot do in a usable time.

### Phase 5 — Incremental daily datasets (both pipelines)

With 0.4 passing, a game's row depends only on data before it. Then:

- store each dataset partitioned by season in Parquet, under
  `train_data/2_5/` (keys in `train_data_store`), locally and on S3;
- the daily job builds only the newly finished games
  (`target_game_ids` = games since the last build) and appends them;
- a full rebuild happens only when the code version (recorded in the manifest)
  or a DB correction invalidates it.

This is what the daily intermediate refit is blocked on
(`modeling/refit.py`, `docs/model_registry_reorg_plan.md` section 5), and it
removes the full closing rebuild from the retrain job.

**Checksum caveat.** Campaign configs pin `data.expected_checksum` to a file.
An appended dataset is a new file and gets a new checksum. That is the intended
behaviour, but daily builds should be named by limit date as they are now,
never overwritten in place.

## Order of work

| # | Item | Changes values? | Expected effect |
|---|---|---|---|
| 0.1-0.4 | compare script, references, stage timer, append test | no | enables everything else |
| 1.1 | in-process memoization, one connection per DB | no | removes duplicate reads |
| 1.2 | Parquet input cache incl. repaired closing odds | no | about half of remote-DB time |
| 2.1-2.2 | slope, legacy referees | no | ~1.5 min closing, ~2 min intermediate |
| 3 + 2.5 | intermediate horizon de-dup, shared effect cache | no | largest intermediate win |
| 2.3-2.4 | player / all-star loops | no | ~2 min closing, ~8 min intermediate |
| 4.1-4.3 | target rows + full-history prediction | yes, the 27 scope columns only | prediction minutes, train/serve parity |
| 5 | incremental datasets, S3 | no | daily refit without full rebuilds |
| 1.3-1.4 | COPY/ADBC reads, S3 cache mirror | no | GitHub-hosted runner, remaining DB time |

The time figures are estimates from the profiles above. Each PR replaces its
own estimate with a measurement taken with 0.3.

## Open questions

- Retrain closing models after 4.2? The production closing slots were trained
  on all-season features and are served 2-season ones. 4.2 removes that gap
  without a retrain, so serving moves *toward* the training distribution.
- Cache invalidation after DB repairs: a manifest bump from the repair scripts
  (automatic), or a documented `--refresh` (manual)? Automatic is safer, given
  how many repair scripts already exist.
- numba as a dependency, or numpy-only? Decide at 2.1, by whether the numpy
  version is fast enough.
