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
loader is idempotent per game. The raw archive is the source of truth and
permits a parser fix without another NBA API backfill.

Pace uses `FGA + 0.44*FTA - OREB + TOV` per segment. V3 player rebound rows
carry cumulative offensive/defensive counts in `description`; those counts are
decoded by the builder. Team rebounds without a player ID are currently not
classified, so this is an estimated possession count, not an official count.
