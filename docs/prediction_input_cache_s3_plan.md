# Past seasons from S3, current season from the database

Goal: make today's prediction fast. Past seasons are read from S3, and only the
current season is queried from Supabase and Aiven. It extends the line-history
cache of `docs/pipeline_speedup_plan.md` (Phase 1) to every input table, and
moves the store from local disk to S3, so the GitHub runner that predicts can
use it.

Status: plan only (2026-10-15), on branch `perf/pipeline-speedup`.

## What it saves

Live query time per build, from the recorded reference builds (7bd96a9):

| Build | Supabase | Aiven | of which `nba_players` |
|---|---:|---:|---:|
| closing, 2 seasons (today's prediction scope) | 53 s | 101 s | 29 s |
| closing, all seasons (Phase 4 scope) | 83 s | 105 s | 51 s |

The current season is about half of a 2-season load and an eighth of an
all-season load. Expected, from row shares (to be measured):

- 2-season prediction: Supabase 53 s -> ~25 s. Aiven is already cached
  locally for past seasons; on S3 that also reaches the runner.
- all-season prediction (after Phase 4): Supabase 83 s -> ~12 s, Aiven
  ~105 s -> ~25 s.

The 27 feature columns that change between a 2-season and an all-season build
are not touched here. Loading all seasons cheaply is what makes fixing them
(Phase 4.2) affordable.

## Design

### Two jobs, one store

- **Refresher**: a step in the daily database job, which runs after games are
  final (`update_finished_matches_daily`, self-hosted). For every table and
  past season it computes the server fingerprint and compares it with the
  stored one. If it differs, or the season is missing, it refetches and
  uploads. Repair scripts and backfills call it for the seasons they changed:
  `scripts/refresh_input_cache.py --tables ... --seasons ...`.
- **Consumer**: the prediction job. It downloads the manifest and the past
  seasons it needs from S3 and queries the current season from the database.
  It does not run fingerprints, so it pays only download time.

Why the consumer trusts the refresher: the fingerprint is cheap for most
tables (0.1-1 s per season) but costs 2-3.6 s for `nba_players`. Validating
every past season in each prediction run would take back 10-25 s of the
gain. Past seasons change only through repairs and backfills you run
yourself, and those call the refresher.

Safety net: the manifest records when each season was last validated, and the
consumer falls back to the database for any season whose validation is older
than a limit (default 36 h). So a refresher that stops running can make
predictions slower, never stale.

### Store layout

```
s3://<bucket>/input_cache/v1/<loader>/<key>/
    manifest.json            # per season: fingerprint, validated_at, rows, sha256, format
    season=2019.parquet
    ...
```

- `<key>` hashes the loader's source code, its arguments and the
  pandas/pyarrow versions, like `input_cache.code_key`. Changing a query or
  upgrading pandas starts a new key, so an old entry is never misread.
- Format: Parquet when it round-trips exactly, otherwise pickle. On upload the
  frame is written, read back and compared (dtypes, index, digest); a frame
  that doesn't round-trip exactly (mixed object columns, tz-aware datetimes) is
  stored as pickle. The runner uses `requirements.txt`, so library versions
  match the refresher's or the key differs.
- **S3 is the store; local disk is optional.** Decided 2026-10-15: nothing
  may depend on a machine's disk, because runners may move to cloud services.
  The refresher writes to S3. The consumer reads S3, and a local copy is only
  a speed-up that may be absent, as on a fresh cloud runner. The Phase 1 local
  line-history cache (`data/cache/inputs`) moves into the same S3 layer; its
  local directory stays as the optional speed-up. Credentials: the existing
  `S3` settings and `s3_models` client; the prediction workflow already
  assumes an AWS role.

### Loaders that get the cache

| Loader | Table | Ordered? | Notes |
|---|---|---|---|
| `load_all_nba_data_from_db` (players) | `nba_players` | **no** | biggest: 29-51 s |
| `load_player_box_history` | `nba_players` join `nba_games` | **no** | queries all seasons (`season_years=None`) |
| `load_games_from_db` | `nba_games` | yes (date, id) | total order |
| `load_odds_sportsbook_from_db`, `load_odds_yahoo_from_db` | `odds_*` | date only | same-day games in arbitrary order |
| `get_injury_data_from_db` | `nba_injuries` | **no** | |
| `get_refs_data_from_db` | `nba_refs` | **no** | |
| `load_all_star_voting_from_db` | `nba_all_star_voting` | yes | small |
| `fetch_closing_ticks`, `fetch_pregame_ticks` | Aiven line history | yes | cached locally today; move to S3 |
| injury-report state | Aiven `injury_report` | mixed | later; ~5 s |

`extra_game_ids` (the head-to-head workaround) adds a few older games outside
the loaded seasons. They come from the past-season entries, filtered by id, so
nothing changes for the caller.

### The row-order problem

Five loaders have no `ORDER BY` (or a partial one), so Postgres returns rows
in whatever order its plan produces. One multi-season query and per-season
pieces joined together can come back in different orders. Wherever the
pipeline breaks ties by row position (stable sorts, `drop_duplicates`,
`groupby().first()`, set iteration), the last bits of some features could
change, as happened with the roster hash seed.

Handled in two steps:

1. **Measure first** (test T0 below). If the build doesn't depend on row order,
   nothing has to change.
2. **If it does**, give each loader a deterministic order: a total `ORDER BY`
   in SQL, with the same sort applied to the assembled per-season frames. That
   changes the no-cache path too, so it is its own commit, gated by a new
   reference build. It may only change rounding-level bits, never a value
   beyond floating-point noise, and the comparison must show that.

### Done: T0 and step 3 (2026-10-15)

- **T0** (`scripts/perf/row_order.py`, `reference_build.py --shuffle-seed`):
  replays with every result reordered as its ORDER BY permits. The builds
  did depend on the server's row order, at rounding level only: 23-30
  closing and 46-49 intermediate columns per seed, up to 1.4e-14
  (injury-status questionable/probable effects, form and expected sums,
  intermediate `*_SNAP_NEWS_*`). The scoring file never changed.
- **Step 3** (e0f8bb0): `nba_ou/postgre_db/row_order.read_sql_canonical` in
  every Supabase and injury-report loader. It keeps the runs the ORDER BY
  fixes, exactly as the server sent them, and sorts the rest by the other
  columns (text by `str`, never by collation).
- **Acceptance** (`artifacts/perf/order_accept_e0f8bb0`): all four builds
  EQUAL at `--rtol 1e-12` against `ref_7bd96a9`; the exact differences are only
  those columns (max 2.8e-14); the scoring files are bit-identical; shuffled
  replays are bit-identical to the unshuffled one.
- **New references**: `artifacts/perf/ref_e0f8bb0/` (the 7bd96a9 tapes, outputs
  replayed at e0f8bb0). Every later change is gated on these with `--exact`.

## Tests

None of them need tonight's games.

| # | Test | How | Pass condition |
|---|---|---|---|
| T0 | Row-order sensitivity | Replay the closing 2-season and intermediate 2025 tapes with each unordered loader's rows shuffled (5 seeds) | `compare_builds --exact` EQUAL, or a list of what changes |
| T1 | Store mechanics | Unit tests with `tests/fake_s3.py`: hit, miss, changed fingerprint, corrupted object, stale `validated_at`, S3 unreachable, Parquet round-trip fallback | as specified |
| T2 | Per-season assembly | For each loader, past + current season frames vs the single multi-season query, live, read-only | identical frames (or identical after the T0 sort) |
| T3 | Training builds | Live A/B/C as in Phase 1: no cache / cold / warm, closing 2-season and intermediate 2025, same day | B and C bit-identical to A |
| T4 | Prediction mode on past dates | Rebuild a past date's `scheduled_data` (schedule from `live_replay.replay_scheduled_games`, odds, referees and injuries from the stored rows), run `create_df_to_predict(todays_prediction=True)` with and without the cache, for 5 dates across the season | bit-identical with and without the cache |
| T5 | Freshness | Change a row in a fake source (or the fingerprint of a fake season), run the refresher, then the consumer | the consumer gets the new data; untouched seasons aren't re-uploaded |
| T6 | Failure modes | S3 down, a season missing, manifest stale | falls back to the database, output identical, a warning printed |
| T7 | Runner timing | `workflow_dispatch` of the prediction workflow on a past date (T4's date replay), before and after | time per stage in the log |

T4 is how this gets tested without today's games: a date that has already
been played is fed to the prediction path as if it were tonight. The data are
known, so the cache can be checked against the database bit for bit. T4 needs
one new helper: a past-date `scheduled_data` builder (closing side). It is
also the harness Phase 4.3 needs for the train/serve parity test, so it is
built once for both.

## Order of work

1. ~~T0: measure row-order sensitivity.~~ Done: it was sensitive at rounding level.
2. S3 store, read-through layer and refresher script, plus T1.
3. ~~Deterministic loader order, new references.~~ Done (e0f8bb0, `ref_e0f8bb0`).
4. Wire the loaders, one commit each, with T2 per loader.
5. T3, then the past-date `scheduled_data` builder and T4.
6. The refresher in the daily job; the consumer in the prediction workflow; T5,
   T6 and T7.

## Open decisions

- **Consumer trust**: trust the refresher with a 36 h staleness fallback
  (recommended, fastest), or validate fingerprints in the prediction job too
  (+10-25 s, protects against a repair that forgot to refresh).
- **Bucket and prefix**: the existing models bucket under `input_cache/`, or a
  separate one.
- **Workflow placement**: `nba_predictor_daily.yml` currently says
  `runs-on: self-hosted`. If predictions move to GitHub-hosted runners, the
  refresher should stay on the self-hosted daily database job, and only the
  consumer runs on GitHub.
