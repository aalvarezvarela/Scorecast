# Keeping the line-history store up to date

Two entry points. `nba_games` (Supabase) is refreshed daily and is the
reference for what *should* exist in the Aiven `line_history` store.

## Daily update (the one the scheduled job runs)

`scripts/update_databases/update_line_history_database.py` — the counterpart to
`update_all_databases.py`. Seasons default to the current one, so it needs no
yearly edit.

```bash
# what the daily job runs: current season
python scripts/update_databases/update_line_history_database.py

# see the plan without issuing a single request
python scripts/update_databases/update_line_history_database.py --dry-run

# past seasons
python scripts/update_databases/update_line_history_database.py --start 2021 --end 2024
```

It must run **after** the games update, since line history is keyed on
`nba_games`. Each run does three things:

1. **Re-fetches the last `--refresh-days` (default 3) of dates unconditionally.**
   Lines move right up to tipoff, so a game fetched the morning it is played is
   *present* but not *final* — and presence alone would never bring it back.
2. **Fills games `nba_games` has that the store has never seen.**
3. **Tops up games missing a book that most games on their own date carry.**

## Backfill / ad-hoc

`scripts/line_history/update_line_history.py` works in date ranges rather than
seasons — use it for large historical fills.

```bash
# what is missing, no fetching and no writes
python scripts/line_history/update_line_history.py --report-only

# fill every gap
python scripts/line_history/update_line_history.py

# a specific window
python scripts/line_history/update_line_history.py \
    --start-date 2026-04-13 --end-date 2026-06-13

# re-fetch dates already stored, to top up a book or market
python scripts/line_history/update_line_history.py --dates 2026-04-14 2026-04-15
```

Note the flag names differ deliberately: `--start`/`--end` are **season years**
on the daily updater and `--start-date`/`--end-date` are **dates** here.

## The timezone problem is gone

The Phase 0 findings (`line_history_phase0_findings.md`) calibrate the CSV-era
scrape to `Europe/Madrid`. That calibration was necessary because the old
scraper read the *rendered* line-history table, which SBR draws client-side in
the browser's local timezone with no offset attached — so the stored timestamps
silently depended on the machine the scraper ran from.

Confirmed directly: loading the same page (event 316538) under three browser
timezones yields three different tables for the same tick.

| browser timezone | rendered time | actual instant |
|---|---|---|
| `Europe/Madrid` | `04/05 1:05 am` | 2025-04-04 23:05 UTC |
| `UTC` | `04/04 11:05 pm` | 2025-04-04 23:05 UTC |
| `America/New_York` | `04/04 7:05 pm` | 2025-04-04 23:05 UTC |

The page also ships a `__NEXT_DATA__` payload whose `oddsDate` values carry an
explicit UTC offset, and which is identical under every timezone. The current
scraper reads that instead, so **the result no longer depends on where it runs**
and a GitHub Actions runner is as correct as a laptop in Madrid.

Three further consequences:

- The payload is server-rendered, so a plain HTTP GET is enough. No Playwright,
  no cookie banner, no clicking through each book and market.
- One request per game returns **every** sportsbook across totals, spread and
  moneyline, rather than one request per book/market combination.
- The payload carries the game's own `startDate`, so `mins_to_tip` is computed
  against the same response the ticks came from. The NBA schedule feed is still
  consulted, but only as a cross-check: a disagreement beyond
  `TIPOFF_TOLERANCE_MINUTES` is reported, and the page value is the one used.

## Conventions the loader preserves

- **Timestamps are truncated to the minute.** SBR polls the books on a fixed
  cadence and stamps every tick with the same seconds value (`:20`), so the
  seconds carry no information. Dropping them makes a re-scrape reproduce the
  keys already in the store, which is what keeps loads idempotent — re-ingesting
  a game already loaded from the CSV era inserts exactly 0 rows.
- **Left is away/OVER, right is home/UNDER**, matching the existing columns:

  | market | `left_line` | `left_price` | `right_line` | `right_price` |
  |---|---|---|---|---|
  | `totals` | total | over odds | total | under odds |
  | `point_spread` | away spread | away odds | home spread | home odds |
  | `money_line` | NULL | away odds | NULL | home odds |

- **Writes are insert-only** on both `lh_line` and `lh_game`. `lh_game` is
  deliberately not upserted: `mins_to_tip` on already-stored ticks was computed
  against the stored `tipoff_utc`, so moving it would desynchronise rows the run
  is not touching.
- **Books are matched by slugified display name** (`Fanatics Sportsbook` ->
  `fanatics_sportsbook`), so new rows land on the books already in `lh_book`.

## Caesars is no longer available

SBR served six books historically but now exposes five: `bet365`, `betmgm`,
`draftkings`, `fanduel`, `fanatics_sportsbook`. **Caesars was dropped from the
site**, so the 270k historical Caesars rows cannot be refetched. This is the
concrete reason the loader must never replace a game's rows wholesale — a
refresh of an old game would silently destroy data the source can no longer
provide.

Caesars is listed in `update.DISCONTINUED_BOOKS` and excluded from the
completeness check on both sides, so **a game counts as complete when the other
books are there**. It is never re-fetched on Caesars' account, which would mean
chasing data the source no longer has, forever.

## What counts as "incomplete"

A book is *expected* on a date only once it priced at least
`--min-book-share` (default 0.5) of that date's games. The threshold is not
cosmetic: Fanatics launched on 2025-11-05 covering 1 game out of 11, so a
naive "match the best-covered game" rule marked the other ten partial and would
have re-fetched them on every run forever.

The check cannot distinguish "this book never priced this game" from "the
scrape missed it" without fetching once. If a specific game keeps reappearing,
that is the former — raise `--min-book-share` or pass
`--skip-incomplete-check`.

## Adding a book the store never held (BetRivers)

The incomplete check above cannot add a new book: no stored game carries it, so
no date "expects" it. `--backfill-books` re-fetches every stored game with no
tick for the named books instead. SBR's BetRivers history starts with 2021-22,
so start there — earlier seasons would be re-requested for nothing:

```bash
python scripts/update_databases/update_line_history_database.py \
    --start 2021 --end 2025 --backfill-books betrivers --refresh-days 0 \
    --skip-incomplete-check
```

Writes are flushed every `--flush-every-dates` dates (default 7), so a failure
late in a multi-season run keeps what was already fetched; re-running resumes,
because stored games no longer lack the book. Inserts stay insert-only, so the
re-fetch also adds any tick of the other books that the first scrape missed,
and never changes one.

The closing table gets the same book through
`update_sportsbook_database.py --backfill-books betrivers`, which fills only
BetRivers' NULL columns on stored games.

That scraper now reads the same `__NEXT_DATA__` payload (`--engine json`, the
default) instead of driving a browser: ~1.2 s per date instead of ~20 s, so a
season takes minutes. It returns the same raw frames the browser engine did —
checked on 33 dates from 2019 to scheduled 2026-27 games with zero differences —
except for games with no odds at all, where the browser engine copied a
neighbouring game's teams and odds (rows that never matched a game). Writes land
every `--write-every-dates` dates (default 14), so an interrupted run resumes.
`--engine browser` keeps the old path as a fallback.

`--all-seasons` checks every season at once, and `--dry-run` shows the dates it
would scrape per season. It skips missing games dated before the first stored
odds (SBR has nothing earlier) and backfill games dated before the backfilled
book's first stored value, so reruns do not re-request dates that can never
fill:

```bash
python -m nba_ou.postgre_db.odds_sportsbook.update_sportsbook.update_sportsbook_database \
    --all-seasons --backfill-books betrivers
```

**Stored is not a feature.** `nba_ou.config.odds_columns.HISTORY_ONLY_BOOKS`
lists books both dataset builders drop at read time (the intermediate builder's
tick fetch, the closing loader, and the same-day scrape). BetRivers' coverage
starts two seasons after the others, so its presence alone encodes the season;
admitting it is a deliberate, ablated change.

## Leakage still has to be filtered

Nothing here changes the Phase 0 finding that SBR records in-play ticks with the
same shape as pre-game ones. `is_pregame` and `mins_to_tip` remain the only
separation, and feature queries should keep using
`fetch.DEFAULT_MIN_MINUTES_BEFORE_TIP`.

## Monthly backup

`.github/workflows/monthly_line_history_backup.yml` runs on the 1st of each
month and calls `scripts/backup_line_history_to_s3.py`, which writes the whole
`line_history` schema to
`s3://<BUCKET>/backups/db/<YYYY-MM-DD>/line_history/<table>.parquet` — the same
layout the Supabase backup uses, so a restore reads the same way for both.

```bash
python scripts/backup_line_history_to_s3.py            # back it up
python scripts/backup_line_history_to_s3.py --dry-run  # list, upload nothing
python scripts/backup_line_history_to_s3.py --list     # existing backup dates
```

Each run writes under its own date tag, so runs never overwrite one another.

**`lh_line` is exported one partition per file, and the partitioned parent is
skipped.** This is not cosmetic: `information_schema.tables` reports the parent
*and* all five partitions as `BASE TABLE`, so the obvious table list would write
all 1.8M rows twice. Discovery goes through `pg_class.relkind` instead. Per
partition also bounds peak memory to one season and makes a single season
restorable on its own.

Nine files per run, roughly 15 MB total — the 236 MB database compresses hard
because the fact table is almost entirely small integers. At that size a year of
monthly backups is under 200 MB, so there is no retention policy; add one only
if that changes.
