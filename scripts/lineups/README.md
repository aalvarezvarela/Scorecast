# Lineup data ingestion

The lineup projection in `docs/lineup_projection_plan.md` starts with a raw
archive and validated five-on-five stints. Run these commands from the repo root
with the Poetry environment (or `.venv/bin/python`).

```bash
python scripts/lineups/import_pbp_archive.py --min-season 2018 --max-season 2025
python scripts/lineups/backfill_lineup_raw.py --endpoints gamerotation
python scripts/lineups/build_lineup_stints.py --season 2018
python scripts/lineups/report_lineup_coverage.py --with-db
```

## Import the play-by-play instead of fetching it

`shufinskiy/nba_data` republishes the same `PlayByPlayV3` payloads season by
season, so a whole season costs one ~8 MB download rather than 1,312
rate-limited calls. `import_pbp_archive.py` rewrites them into the ordinary raw
archive and marks them `ok` in the manifest, which halves the backfill: only
`gamerotation` is left to fetch, and `--endpoints gamerotation` stops the
fetcher from asking for play-by-play it already has.

**This was verified, not assumed.** For the 440 games we had already fetched
from the API, the rebuilt payloads produce **byte-identical stints in all 437
that pass rotation validation**. Three details the importer has to handle:

- The archive omits `shotValue`. It is rebuilt from the `3PT` marker in the
  description, which matched the API exactly on all 78,050 field goals in
  2018-19. Non-shot rows (blocks) get 0; the stint parser never reads them.
- Corrected actions share an `actionNumber` with the row they amend, so the
  sort must be stable or a turnover and its steal swap teams.
- The archive strips diacritics from `playerName` and pads some text columns.
  Players are identified by `personId`, so this does not reach the stints.

The archive covers 1996–2025 including playoffs, but has **no `gamerotation`**,
which is why the API is still needed for the other half.

Run the remaining rotation backfill in chunks rather than in one long process:

```bash
python scripts/lineups/backfill_lineup_raw.py --endpoints gamerotation \
    --min-season 2018 --max-season 2018
```

`--limit N` caps the API calls per run and `--timeout` sets the per-request read
timeout. Everything is resumable: the manifest skips whatever is already `ok`,
so re-running after any interruption costs no extra API calls.

Both the fetcher and the stint builder show a `tqdm` bar. The fetcher resolves
what is already archived **before** starting it, so the total, rate and ETA
describe only the calls still owed:

```
fetch (560 archived): 67% 2/3 [00:03<00:02, 5.04s/call, ok=2, empty=0, failed=0, skipped=560]
```

The bar disables itself when stdout is not a terminal, so a `nohup` run writes a
periodic progress line to its log instead of thousands of redraws.

**Measured throughput (2026-09-20):** the pacer waits at least 3 s between
request *starts*, but `stats.nba.com` response times swing between 0.5 s and a
30 s hang, so the real cost was **5 s per call over 46 consecutive games** and
much worse during a bad stretch. Budget well over the 3 s/call floor when
planning a backfill. Importing the play-by-play removes half of the ~20.4k
calls, leaving ~10.2k rotation calls, or roughly 14 h rather than 29 h.

The raw fetcher stores gzip-compressed original NBA responses under
`data/nba_api_raw/{gamerotation,playbyplayv3}/{season_year}/{game_id}.json.gz`.
Its per-season manifest is in `data/nba_api_raw/manifest/`. Rerunning skips
only `ok` entries whose object still exists. Empty and failed responses remain
eligible for a retry. The fetcher makes one request at a time, at least three
seconds apart, and exits after four consecutive blocks. To archive in the
configured S3 bucket, add `--s3`; the manifest remains local and is mirrored
to S3 at the end of the run or when the circuit breaker opens.

The stint builder writes one Parquet per validated game and a per-season status
file under `data/lineup_stints/`. It rejects games whose lineups, total seconds,
points or player minutes do not reconcile. `--force` rebuilds validated games
after parser changes. The report prints pass rates and failure reasons; use
`--with-db` to calculate raw coverage against all finished games.

## Where the data lives

**The local Parquet store is the default destination.** `data/lineup_stints/`
holds one file per validated game plus a per-season `game_status.parquet`, and
each stint file carries its `game_date`, so ratings can be fitted with no
database at all. Read it with
`nba_ou.data_processing.lineups.stint_store.read_stints()`.

Postgres is **opt-in** and stays that way until the lineup features earn a
place in the training pipeline:

```bash
python scripts/lineups/load_lineup_stints.py --season 2018      # manual load
python scripts/lineups/update_lineups.py --local-root data --s3 --load-db
```

`load_lineup_stints.py` creates the `lineups` schema in the configured default
database, skips unchanged game statuses and is idempotent per game; use
`--force` after a parser/storage migration.

For daily operation, the combined command fetches missing finished games and
validates new archives into the Parquet store:

```bash
python scripts/lineups/update_lineups.py --local-root data --s3
```

Without `--load-db` it neither reads nor writes the `lineups` schema: it
compares against the local `game_status.parquet` files, so keep `data/` on a
persistent disk (the finished-match workflow runs on a self-hosted runner, so
it does). A game recorded as `failed` is retried on the next run, because its
raw archive is already paid for. An interprocess lock prevents this command and
a historical backfill from writing the same local manifests concurrently.

The raw archive is the source of truth and permits a parser fix without another
NBA API backfill.

The walk-forward ridge code can be smoke-tested with explicit regularization
values:

```bash
python scripts/lineups/fit_player_ratings.py --last-season 2025 \
  --lambda-offdef 100 --lambda-pace 1000 \
  --output data/lineup_ratings/pilot.parquet
```

Every ratings command reads the Parquet store by default; pass `--source db` to
read `lineups.lu_stint` instead.

Those lambda values are an example, **not calibrated production values**.
Phase C requires walk-forward tuning and the 2021–2025 go/no-go comparison
before these ratings become model features.

Tune explicit penalty grids only after the stint coverage gate passes:

```bash
python scripts/lineups/tune_player_ratings.py --last-season 2025 \
  --validation-from 2021-10-01 --validation-to 2025-06-30 \
  --output-dir data/lineup_ratings/cv
```

After selecting and recording production penalties, materialize the reusable
walk-forward cache with explicit values:

```bash
python scripts/lineups/build_player_ratings.py --last-season 2025 \
  --as-of-from 2019-10-01 --as-of-to 2025-06-30 \
  --lambda-offdef <chosen-value> --lambda-pace <chosen-value>
```

Then run the required Phase C comparison on identical games:

```bash
python scripts/lineups/evaluate_player_ratings.py --last-season 2025 \
  --validation-from 2021-10-01 --validation-to 2025-06-30 \
  --ratings data/lineup_ratings/player_ratings.parquet \
  --output-dir data/lineup_ratings/gate
```

The baseline sums each team's mean points scored over its prior five games.
Both methods are scored only where both are available.

Pace uses `FGA + 0.44*FTA - OREB + TOV` per segment. V3 player rebound rows
carry cumulative offensive/defensive counts in `description`; those counts are
decoded by the builder. Team rebounds without a player ID are currently not
classified, so this is an estimated possession count, not an official count.

## Evaluate the game projection against the line

`evaluate_game_projection.py` regenerates the phase-F / phase-G numbers in
`docs/lineup_projection_plan.md` from committed code. It computes the
projection exactly as the `LU_*` features do (`data_processing/lineups/features.py`),
then reports its MAE, the slope of `LINE_ERROR` on the absence counterfactual
and on `proj - line` per season with date-clustered bootstrap intervals, and
the directional accuracy of "OVER when impact > 0". 2025-26 is reported
separately as the season no tuning has touched.

```bash
python scripts/lineups/build_player_ratings.py --last-season 2025 \
  --as-of-from 2021-10-01 --as-of-to 2026-06-30 \
  --lambda-offdef 1000 --lambda-pace 30000
python scripts/lineups/evaluate_game_projection.py \
  --lines data/train_data/training_data_2_3_20260909.csv \
  --output-dir data/lineup_ratings/eval_projection
```

The default cache now covers 2021-10-19 to 2026-06-30 (1,038 rating dates).
The same cache feeds the opt-in `--lineup-features` build of the training data.
Before the phase-F results, the evaluation drivers were run ad hoc and not
committed; a rebuild from the modules matches their slopes but not every game,
so treat this script's output as the reference.
