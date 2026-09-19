# Lineup data ingestion

The lineup projection in `docs/lineup_projection_plan.md` starts with a raw
archive and validated five-on-five stints. Run these commands from the repo root
with the Poetry environment (or `.venv/bin/python`).

```bash
python scripts/lineups/backfill_lineup_raw.py --limit 100
python scripts/lineups/build_lineup_stints.py --season 2018
python scripts/lineups/report_lineup_coverage.py --with-db
```

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

After reviewing the report and choosing the database, load validated games:

```bash
python scripts/lineups/load_lineup_stints.py --season 2018
```

This creates the `lineups` schema in the configured default database. The
loader skips unchanged game statuses and is idempotent per game; use `--force`
after a parser/storage migration. The raw archive is the source of truth and
permits a parser fix without another NBA API backfill.

For daily operation, the combined command fetches missing finished games,
validates new archives and loads only changed statuses:

```bash
python scripts/lineups/update_lineups.py --local-root data --s3
```

The daily command compares against `lineups.lu_game_status`, so a cleaned
runner workspace only downloads and builds games not already stored. An
interprocess lock prevents it and a historical backfill from writing the same
local manifests concurrently. The finished-match GitHub workflow uses the S3
raw archive and runs this command after updating the game database.

The walk-forward ridge code can be smoke-tested with explicit regularization
values:

```bash
python scripts/lineups/fit_player_ratings.py --last-season 2025 \
  --lambda-offdef 100 --lambda-pace 1000 \
  --output data/lineup_ratings/pilot.parquet
```

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

Pace uses `FGA + 0.44*FTA - OREB + TOV` per segment. V3 player rebound rows
carry cumulative offensive/defensive counts in `description`; those counts are
decoded by the builder. Team rebounds without a player ID are currently not
classified, so this is an estimated possession count, not an official count.
