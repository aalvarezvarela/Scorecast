# OddsLogic line history (the odds on WagerTalk)

Status: **Implemented** (2026-10-23). This is the second line-history source
beside SBR, stored in the Aiven schema `line_history_oddslogic` from the
2026-27 season on. Nothing reads it yet: the fallback to it when SBR is missing
is a later phase.

## Why

SBR's lines froze on 2026-10-20. The odds on wagertalk.com/odds are an iframe
of OddsLogic's free widget, which publishes each book's full line history:
every change, timed in UTC, for spread, total and moneyline, archived back to
2020-12-01. It carries every book SBR has except bet365. Parity with SBR on
past games is in [Validation](#validation).

## Endpoints

The base is `https://freeodds.oddslogic.com/ol-lite-stage/sources/php/`.
Requests need no auth; only a browser user agent and the widget's `Referer`
are sent.

| Request | Answer |
|---|---|
| `get.php?free_odds_lite=1&data=all.txt&date=YYYY-MM-DD` | Every league's schedule for that date: rotation number, event id, teams, tip (UTC, about 10 min after the scheduled tip). |
| `line_history_free_odds.php?t=SPREAD&d=<date>&n=<away rotation>&s=<book id>&p=0&z=0000&c=1` | One book's SPREADS and TOTALS history for one game, newest first. `z=0000` gives UTC. |
| same with `t=ML` | MONEYLINES and TOTALS. |
| `archive_available.php[?date=]` | The archive's date range. |

`s` takes a single id; lists, `0` and `all` answer "No line history found".
A game's three markets therefore cost 2 requests per book, 24 per game with
the 12 active books. The pause between requests is 0.8–1.2 s and each answer takes
1–3 s, so a game takes about 1–1.5 minutes.

## Books

Code: `fetch_data/odds_oddslogic/books.py`. `lh_book.book_id` is the OddsLogic
id, so ids never depend on load order.

| id | slug | note |
|---|---|---|
| 109 | `draftkings` | |
| 112 | `fanduel` | |
| 106 | `betmgm` | |
| 107 | `caesars` | |
| 111 | `fanatics_sportsbook` | |
| 113 | `hard_rock_bet` | |
| 60 | `betrivers` | not in the widget's list; see below. **Inactive**: no 2026-27 lines |
| 110 | `espn_bet` | not on SBR |
| 108 | `circa` | not on SBR |
| 117 | `south_point` | not on SBR |
| 139 | `kalshi` | not on SBR |
| 115 | `oddslogic_consensus` | OddsLogic's consensus, not a book |
| 116 | `consensus_opener` | the "Open" column |

Books SBR also carries reuse SBR's slug, so a later merge lands on the same
book. A new slug becomes a model feature only if something reads this store
into the pipeline. The fallback must therefore keep to SBR's books, or extend
`HISTORY_ONLY_BOOKS` for the rest.

**Identifying the unlisted ids (2026-10-23).**
- Ids 1–160 were probed; about 19 answer on the history endpoint without
  being in the list.
- On 40 games of 2025-26, every id was matched against every SBR book on
  three measures:
  - line and price equal on a 10-minute grid from T-720 to T-10;
  - line alone equal on the same grid;
  - SBR line moves repeated within 15 minutes.
- The listed books land on their own SBR book (line 0.76–0.97).
- **id 60 is BetRivers:** line 0.88, moves 0.87; no other book reaches more
  than 0.10 line+price against it.
- **No id is bet365.** The best candidate, id 44, follows 45% of bet365's
  moves. Caesars (107) tracks SBR's bet365 better than any unlisted id does.
- **BetRivers has no 2026-27 lines.** On the three 2026-10-20 games id 60
  holds only August–September quotes, and nothing on later games (checked
  2026-10-23). It is in `books.INACTIVE_BOOKS`: registered in `lh_book`, not
  requested by the daily update, and fetchable with `--books betrivers`.
- The other unlisted ids follow the market (44, 59, 104, 117, 97/98, 126) or
  copy DraftKings/BetMGM (64) or Fanatics (63). They are not stored.

## Parsing rules

Code: `fetch_data/odds_oddslogic/parse.py`. The output is SBR's `ScrapedGame`,
so `line_history_aiven.ingest.build_frames` does the rest unchanged: game_id
resolution, the spread price-bleed and implausible-line repairs, the doubled
SMALLINT lines and the minute dedup.

- **Stale timestamps.** The widget records later polls under the time of the
  last change it saw. A run of consecutive rows sharing a timestamp therefore
  keeps only its oldest row.
  - The newer rows of such a run often hold in-play quotes. Example: FanDuel,
    PHI @ NY on 2026-10-20, shows -1.5 -132 stamped 15:35 UTC, while every
    other book was at -4.5 and tip was at 23:00.
  - The filter runs before the per-minute collapse. Otherwise those rows would
    win their minute.
- **Collapse and opener.** Seconds are truncated and the latest quote in a
  minute wins. The earliest tick per (book, market) is the opener. This is
  SBR's rule.
- **Sides.** Left/right are OVER/UNDER on totals, taken from the `o`/`u`
  letter rather than the column order. On spread and moneyline they are
  AWAY/HOME. Moneyline rows are prices only; `pk` is 0 and `ev` is +100.
- **Year.** Dates have no year. Each gets the latest year that puts it no
  more than 30 days after the game.
- **Tip-off.** Taken from the NBA schedule feed (`fetch_schedules`, with
  `tipoff_corrections`), not from the widget, whose tips are about 10 minutes
  late.
  - Ticks after the feed's tip are stored with `is_pregame = false`, as SBR's
    in-play rows are, and readers drop them.
- **TOTALS appears in both answers.** The longer copy is used.

## Store

Code: `postgre_db/line_history_oddslogic/`.

- **Tables.** They are the SBR store's: `lh_book`, `lh_market`, `lh_game`,
  `lh_line` (list-partitioned by season) and `lh_load_meta`. The DDL is
  `line_history_aiven.schema.DDL_STATEMENTS`, used verbatim.
- **Contents.** `lh_game.event_id` is the OddsLogic event id and the team
  columns hold tricodes. `lh_load_meta` records `UTC`/`high` with the last
  run's counts.
- **Writes are insert-only.** COPY goes to an UNLOGGED staging table in this
  schema, followed by one `INSERT ... ON CONFLICT DO NOTHING` per season.
- **Reads.** `store.fetch_pregame_ticks` and `store.fetch_games` return the
  frames the SBR readers return.
- **The SBR modules are not changed for a second schema on purpose.**
  `line_history_aiven/{schema,load,ingest,fetch}.py` are in the import closure
  that keys the training-data caches (`history_memo.code_key`,
  `input_cache.code_key`). An edit there invalidates every memo, so this
  package reuses their constants and repeats their few SQL statements instead.
- **Public betting split.** Each history row also carries one (`"NY 57%,
  40%"`), and the widget has "Tickets" (25) and "Money" (26) series. The
  format is not decoded: the series' cells mix markets and sides, and it is
  unclear which number is tickets and which money. A planned `lh_public` table
  waits for that.
  - The two series are **not fetched** (decided 2026-10-23). They are the
    largest answers, 200–450 KB per game, and nothing uses them.
  - The split embedded in each history row stays in the archived answers.

## Raw archive

Code: `fetch_data/odds_oddslogic/archive.py`. The layout is
`s3://<bucket>/train_data/odds_sources/oddslogic/raw/v1/<date>/`:
- `schedule/<fetched_at>.json.gz`
- `<event_id>/<fetched_at>.json.gz`, every answer of one fetch, `null` where it
  stayed empty.

Each fetch writes a new key, so nothing is overwritten. A failed archive write
is reported and does not stop the store.

## Daily update

`scripts/update_databases/update_oddslogic_line_history.py` runs as the step
"Update OddsLogic line history" of `update_database_daily.yml`, after the SBR
update. It has `continue-on-error`, because nothing later in the job reads it.

The plan is per game and book (`update.plan_requests`):
- A game in `nba_games` but not in the store gets every book.
- A stored game of the last `--retry-days` (3) gets the books that do not
  have all three markets yet.
- Older games get nothing. The archive is final once a game is over.
- Games are fetched once they are 4 hours past tip, and only from
  2026-10-20 on.

**Empty answers.** "No line history found" is often transient: the same
request succeeds minutes later. So each request is retried twice (5 s and
10 s apart), and each date's empty answers are asked once more after the rest
of the date. A book that genuinely skips a game is retried for 3 days, then
left.

```bash
# first run
python scripts/update_databases/update_oddslogic_line_history.py --create-schema
# fetch and parse only, no Aiven or S3 writes; any season with --dates
python scripts/update_databases/update_oddslogic_line_history.py --dry-run \
    --dates 2026-01-15 --save-dir /tmp/oddslogic
# coverage, freshness and parity against SBR
python scripts/line_history/oddslogic/compare_oddslogic_vs_sbr.py --seasons 2026
python scripts/line_history/oddslogic/compare_oddslogic_vs_sbr.py \
    --from-dir /tmp/oddslogic --seasons 2025
```

## Validation

**Dry run on 2026-10-20 to 2026-10-22** (16 games, 13 books, 2026-10-23):
- 31,239 ticks were prepared. 565 of them (1.8%) fall after the feed's
  tip-off and are stored with `is_pregame = false`.
- 21 spread rows went through the price-bleed repair. These are the source's
  own glitches, e.g. `-1.5-112 / 2.5-115` in early October.
- **Coverage.** Games with ticks per market, out of 16:

  | Book | ML | Spread | Totals |
  |---|---|---|---|
  | FanDuel, Fanatics, Circa | 16 | 16 | 16 |
  | DraftKings, BetMGM, Caesars, Hard Rock, ESPN BET, South Point | 14–15 | 14–16 | 15–16 |
  | Kalshi, OddsLogic consensus, Open | 11–16 | 13–15 | 13–16 |
  | BetRivers | 3 | 3 | 3 |

  BetRivers' 3 games hold only August–September lines (see Books).
- **Agreement with the probe's raw download of the same days.** Totals agree
  0.98–1.00 at T-720, T-60 and T-30. Spread agrees 0.98 at T-720 and
  0.93 at T-60 and T-30. The spread gaps near tip are the stale-timestamp
  rows: the probe kept the newest row of each run, the parser keeps the oldest.
- **Cost.** 416 planned requests became 1,119 with the empty-answer retries:
  a genuinely empty answer costs up to 6 requests. In the afternoon (US
  daytime) the server answered in 5–10 s per request, against 1–3 s in the
  European morning. The run took about 1 h 50 min.

**Parity with SBR on past games.** In the probe, two games of 2026-01-15 with
6 shared books gave the same as-of line in 107 of 120 cells. The other 13
differ by 0.5–1 point, during a line move. In the 40-game identification run
the listed books matched their own SBR book at line 0.76–0.97 on a 10-minute
grid.

**Season sample, 2025-26** (dry run on 2025-12-10, 2026-01-15 and 2026-03-05;
20 games; 6 books, of which SBR carries 5;
`compare_oddslogic_vs_sbr.py --from-dir`). Share of (game, book) with the same
as-of line, or price for moneyline:

| Market | T-720 | T-240 | T-60 | T-5 |
|---|---|---|---|---|
| totals | 0.86 | 0.86 | 0.86 | 0.81 |
| spread | 0.95 | 0.95 | 0.86 | 0.96 |
| moneyline (price) | 0.75 | 0.64 | 0.60 | 0.59 |

- Lines agree as well as OddsTrader's did. Moneyline prices differ by a few
  cents around moves, as with OddsTrader.
- **FanDuel moneyline agrees only 0.30–0.45 at every horizon.** That looks
  systematic rather than a matter of timing, and is not explained yet.
- OddsLogic's last pre-game totals tick is a median 15–45 minutes before tip,
  against 8–23 for SBR. Part of the gap is the stale-row filter, which drops
  the last polls.

**First load (2026-10-24).** 2026-10-20 to 2026-10-22: 16 games and 32,663
ticks, every game with all three markets. 21 answers stayed empty and are
retried by the next runs. 957 requests in 1 h 38 min, at night in Europe. The
schema takes 7.3 MB (the database 454 MB of 1 GB). The raw archive holds 19
objects, 0.5 MB.

## Books for prediction (note for the fallback phase)

The daily update keeps fetching **every** book in `OL_BOOKS`, including the
ones that add little: Kalshi, South Point, Circa and the OddsLogic consensus.
History is cheap to keep and cannot be fetched later if the widget closes.

**When this source feeds prediction (live or same-day reads), request only the
books the models use.** A request is one book of one game, so each unused book
costs 2 requests per game at prediction time, for data no model reads. Take
the list from the serving slots' feature books, not from `OL_BOOKS`.

## Limits

- **No bet365**, the main book (`MAIN_SPORTSBOOK`).
- **The endpoint is undocumented.** It is the free widget's backend and can
  change or close without notice. The raw archive keeps what was fetched.
- **No live intraday reads.** The store is filled the morning after.
