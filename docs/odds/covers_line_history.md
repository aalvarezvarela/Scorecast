# Covers line history

Status: **Implemented** (2026-10-24). A line-history source beside SBR, stored
in the Aiven schema `line_history_covers` from the 2026-27 season on (plus the
backfilled past games, [SBR gaps in past seasons](#sbr-gaps-in-past-seasons)),
with the SBR store's tables, plus each game's Covers Consensus pick counts
(`lh_consensus`, from 2026-10-25). It is SBR's per-game fallback in the
prediction, read live ([Live fallback](#live-fallback-for-predictions)), and
in the stored line-history reads that training uses
([Fallback in the stored reads](#fallback-in-the-stored-reads)).

## Why

SBR's line history froze on 2026-10-21 (and `odds_sportsbook` is incomplete
from 2026-10-23). covers.com's "Line Movement" modal serves the same feed as
SBR for the nine US books. It is the only source found so far that carries
bet365 together with all of SBR's other books. On past games it reproduces
SBR tick for tick (see [Validation](#validation)).

## Endpoints

Plain GETs, no auth, server-rendered HTML. Only a browser user agent is sent.

| Request | Answer |
|---|---|
| `/sports/nba/matchups?selectedDate=YYYY-MM-DD` | The date's games. Each is an `<article id="nba-<covers id>">` whose attributes name both teams in full (`data-away-team-fullname=memphis grizzlies`, unquoted). |
| `/sport/MatchupOdds/LineHistoryBrick?gameId=<id>&location=odds&countryCode=us&stateProvince=nj&oddsFormat=american&betType=total` | One game's history of moneyline, spread and total, for every book: about 1.8 MB. The lower-case path answers with a 302 to this one. |
| `contests.covers.com/consensus/matchupconsensusdetails?gameId=<id>` | One game's Covers Consensus pick counts ([below](#consensus-pick-percentages)). |

So one game costs two requests (lines and pick counts), and a date one more.
The client pauses 1.2–2 s between requests.

**Region.** The books depend on `countryCode`/`stateProvince`, not on where
the request comes from. Every US state gives the same nine books. Ontario
(`ca`/`on`) gives 13, including Pinnacle; GB and ES give four. Only the US
view is fetched.

**History depth** (probed 2026-10-24, not used: the store starts in 2026-27):
bet365, DraftKings, FanDuel and Caesars go back to 2019-04, and nothing exists
before 2018-19. BetMGM is there from 2019-10 and BetRivers from 2021-22. Some
single games come back empty on otherwise covered days.

## Books

Code: `fetch_data/odds_covers/books.py`. `lh_book.book_id` is the Covers id
(`data-book-id`).

| id | slug | note |
|---|---|---|
| 1 | `bet365` | |
| 26 | `betmgm` | |
| 18 | `draftkings` | |
| 19 | `fanduel` | |
| 142 | `fanatics_sportsbook` | |
| 29 | `betrivers` | |
| 573 | `caesars` | |
| 6 | `hard_rock_bet` | |
| 44 | `thescore` | not on SBR |
| 1000 | `consensus_opener` | synthetic: BetMGM's opener, see below |

Books SBR also carries reuse SBR's slug. A book id the map does not know is
skipped and reported as a warning.

## Parsing rules

Code: `fetch_data/odds_covers/parse.py`. The output is SBR's `ScrapedGame`, so
`line_history_aiven.ingest.build_frames` does the rest unchanged: game_id
resolution, the spread price-bleed and implausible-line repairs, the doubled
SMALLINT lines and the minute dedup.

- **Tables.** Each market tab lists two tables per book: a one-row `OPEN`
  table, which is skipped (it repeats the oldest history row with only a clock
  time), and the history, newest change first.
- **Visible spans only.** Each side is written three times: American,
  decimal and fractional odds. Only the `display: inline-flex` one is read.
- **Time.** Times are Eastern, to the minute. The date label is `Oct 23`, or
  `Today`/`Yesterday` relative to the fetch's Eastern date.
  - The label has no year. Each gets the latest year that puts it no more than
    30 days after the game (the OddsLogic rule).
  - Pre-season openers (August for an October game) land in the right year.
- **Collapse and opener.** The latest quote in a minute wins, and the
  earliest tick per (book, market) is the opener. This is SBR's rule.
- **Sides.** OVER/UNDER on totals, taken from the `o`/`u` letter rather than
  the column order. AWAY/HOME on spread (per-team handicaps, the home
  favourite negative, as in SBR) and moneyline (prices only). `PK` is a zero
  line and `EV` a +100 price.
- **Teams.** The full names on the matchups page go through
  `string.capwords` (`title()` would turn `76ers` into `76Ers`). They resolve
  through SBR's team-name standardisation.
- **Tip-off.** Taken from the NBA schedule feed (`fetch_schedules`, with
  `tipoff_corrections`), because Covers shows no tip time for finished games.
  - A game without a feed tipoff is not stored and is retried on the next
    run. Without a tipoff, in-play ticks cannot be told apart.
  - Ticks after the tip are stored with `is_pregame = false`, as SBR's in-play
    rows are, and readers drop them. Some books keep quoting during the game.

## consensus_opener

SBR's opener column, as stored in `odds_sportsbook`, is **BetMGM's first
quote, line and prices as quoted**. Checked against the SBR line history,
2021-22 to 2025-26:

| Market | Match |
|---|---|
| totals: line and both prices | 99.8–100% per season |
| spread: line | 100% |
| moneyline: prices | 100% in 2021-22, 89–92% from 2023-24 (a few cents off on one side) |

The store therefore copies each market's BetMGM opener under `consensus_opener`
(`parse.consensus_opener_ticks`). A game BetMGM does not price gets none.

Two things easily mistaken for the rule:

- **The training data's `ODDS_TOTAL_LINE_consensus_opener` and
  `ODDS_SPREAD_LINE_HOME_consensus_opener` sit at -110/-110 equivalents.**
  That is the pipeline's own centering (`data_processing/odds/normalize_total_lines.py`
  and `normalize_spread_lines.py`, applied to every book in
  `postgre_db/odds/merge_odds_data.py`). It is not SBR's, and must not be
  applied twice.
- **SBR's stored spread opener prices are wrong at two-digit lines.** At
  |line| ≥ 10 the stored price is the line's integer part (-13.5 is stored
  with a "price" of -13). This accounts for the 76–91% price match on spread;
  Covers has the real prices.

## Store

Code: `postgre_db/line_history_covers/`. The layout is the OddsLogic store's.

- **Tables.** `lh_book`, `lh_market`, `lh_game`, `lh_line` (list-partitioned
  by season) and `lh_load_meta`, from `line_history_aiven.schema.DDL_STATEMENTS`
  verbatim. `lh_game.event_id` is the Covers game id.
- **`lh_consensus`** is this store's own table (`schema.COVERS_DDL_STATEMENTS`):
  one row per stored game with its Covers Consensus pick counts, the counts
  SBR's `*_consensus_pct_*` shares are taken from.

  | Column | Type | |
  |---|---|---|
  | `game_id` | TEXT | primary key, references `lh_game` |
  | `season_year` | SMALLINT | |
  | `spread_away_picks`, `spread_home_picks` | INTEGER | ATS "Total Picks" |
  | `total_over_picks`, `total_under_picks` | INTEGER | O/U "Total Picks" |
  | `fetched_at` | TIMESTAMPTZ | when the page was read |

  A count of 0 is a count. The counts are read once, at least 4 hours after
  tip-off, so they are the final ones. `PickConsensus.sbr_consensus` turns a
  row into SBR's shares.
- **Writes are insert-only.** COPY goes to an UNLOGGED staging table, followed
  by one `INSERT ... ON CONFLICT DO NOTHING` per season. Fetching a stored game
  again only adds what was missing.
- **A game and its ticks go in one transaction.** The season partitions are
  created first, since they are DDL. A game enters `lh_game` only with its
  ticks. A write that fails rolls the game back, so it is never taken for a
  stored one.
- **Reads.** `store.fetch_pregame_ticks` and `store.fetch_games` return the
  frames the SBR readers return. `store.fetch_consensus(conn, season_years,
  game_ids)` returns the stored pick counts. No reader uses them yet.
- **The SBR modules are not changed for another schema.** They are in the code
  closure that keys the training-data caches, so this package repeats their
  few SQL statements instead.
- **Backup.** The Aiven backup discovers the schema on its own.

## Raw archive

Code: `fetch_data/odds_covers/archive.py`. SBR keeps no raw answers; this
source does, so the parser can be redone without the site.

    train_data/odds_sources/covers/raw/v1/<YYYY-MM-DD>/
      matchups/<fetched_at>.html.gz
      <covers_id>/<fetched_at>.html.gz
      <covers_id>/consensus/<fetched_at>.html.gz

The HTML is stored as served, gzipped. Each fetch writes a new key, so the
archive is insert-only. A failed archive write is reported and does not stop
the store.

## Live fallback for predictions

When SBR's feed is frozen or empty, tonight's lines come from Covers, read
live and never stored. The code is `prediction/line_source.py` (the
decision), `postgre_db/line_history_covers/live.py` (the fetch and the line
history) and `fetch_data/odds_covers/day_odds.py` (the closing frame's odds).
Each stored prediction records the source of its input row
(`prediction.inputs.line_source`: 1 SBR, 2 Covers,
`docs/serving/prediction_storage.md`).

- **When.** The decision is made once per run, before anything reads a line.
  - SBR's feed check (`feed_freshness`) finds it fresh → SBR.
  - SBR frozen, or its check failing → Covers is fetched (one request for the
    date, one per game, ~25 s for 8 games) and judged by the same rule.
    Fresh → Covers for every game. Frozen too → the run stops as before
    (`--allow-stale-lines` predicts on SBR anyway). With the check failing
    and no usable Covers, the run carries on with SBR, unchecked.
  - SBR chosen, but its day page has no odds at all (2026-10-24: eight games
    listed, every cell empty) or its live line history no tick → the run
    switches to Covers.
  - SBR chosen and serving most games → a game it does not price, or gives no
    tick, is filled from Covers.
  - The end-of-day past-snapshot run has no game left to tip, so the check
    cannot tell. A game whose SBR history stops more than
    `TRUNCATED_HISTORY_MINUTES` before tip moves to Covers, if Covers goes
    on later.
- **Closing frame, same columns.** `covers_day_odds` builds the
  `oddsTableModel` SBR's JSON engine parses, from Covers ticks, and runs it
  through the engine's own parsers and day merge. The columns therefore come
  out with SBR's names, order and rounding:
  - The book list is SBR's (`SBR_DAY_PAGE_BOOKS`). theScore gets no column,
    and Hard Rock Bet, the eighth book, is dropped by the parsers as on SBR.
  - Per book, the current line is the latest pre-tip tick at the fetch; the
    opener is `consensus_opener`.
  - The consensus pick percentages (`*_consensus_pct_*`, tonight's values are
    features of every production slot) come from the Covers Consensus page,
    one more request per game (below). A game whose page fails keeps them
    empty.
- **Intermediate frame.** Covers' games go through SBR's
  `live_frames_from_scraped`, without `thescore` and `consensus_opener`
  (absent from the SBR store), so the books are the eight the models trained
  on. Covers is read again at that step, as SBR's live history is.
- **Force it.** `predict_nba_games.py --line-source covers` reads Covers
  without checking either feed (`--no-persist` for a dry run).

### Consensus pick percentages

SBR's `consensus` (`overPickPercent`, `homeSpreadPickPercent`, ...) is the
**Covers Consensus**: the share of Covers users' picks per side. SBR's game
ids are Covers'. One game's counts:

    https://contests.covers.com/consensus/matchupconsensusdetails?gameId=<covers id>

- **Parsing.** `parse.parse_consensus` reads the public section
  (`consensus_analysis_content`), not the experts' section after it. It takes
  its two "Total Picks" rows (`finalWagersleft`/`finalWagersRight`): ATS
  away/home, then O/U over/under.
- **Converting.** `PickConsensus.sbr_consensus` turns them into SBR's shares.
  The moneyline is 0, as SBR serves it. SBR's day-page parser then rounds them
  as it rounds its own.
- **Agreement with SBR** (30 games of 2025-11-12, 2026-01-15 and 2026-03-05,
  read 2026-10-24):

  | Share | Same rounded value | Median gap | Max gap | Correlation |
  |---|---|---|---|---|
  | Home spread | 0.73 | 0.16 pt | 0.65 pt | 0.9998 |
  | Over | 0.63 | 0.34 pt | 1.39 pt | 0.998 |

  These are the same counts read at another moment: SBR's 56.98% on game
  388572 is exactly 396/695 picks, while Covers now shows 410/722.

## Daily update

Script: `scripts/update_databases/update_covers_line_history.py`. Workflow
step: "Update Covers line history" in `update_database_daily.yml`, after
SBR, with `continue-on-error`: nothing later in the job reads it. The
OddsLogic and MoviOdds steps are commented out; those integrations are unused
and unmaintained.

- **What is fetched.** Every game of the season in `nba_games` that the store
  does not hold, once it is at least 4 hours past tip-off.
  - **A game whose answer has no line history is not stored.** It stays
    missing and is fetched again, with a warning, on every run.
  - **A truncated history is held back**, by the SBR ingest's rule
    (`ingest.is_held_back`). Truncated means the freshest pre-tip tick is more
    than 60 minutes (`TRUNCATED_HISTORY_MINUTES`) before tip, the signature of
    a feed freeze. Such a game is not stored for 3 days after tip-off
    (`TRUNCATED_HOLD_DAYS`) and is fetched again on every run. After that it
    is stored as it is.
- **Pick counts.** Every game stored in `lh_game` without an `lh_consensus`
  row, once it is at least 4 hours past tip-off, gets its Covers Consensus
  page: one request per game, archived raw. This covers the stored games
  without counts and the games the run has just stored. The counts are
  fetched in the date's group, after the date's lines are written.
  - A page without counts (the section is there, the "Total Picks" rows are
    not) is not stored. It is fetched again, with a warning, on the next run.
  - A failed request or parse is a failure like a line-history one (exit 1).
    It stops neither the date's lines nor the other games.
  - `--consensus-only` fetches no lines and only fills the counts of stored
    games, for a backfill. `--dates` limits it to those dates. With
    `--dry-run` the pages are parsed and nothing is written.
    `--save-dir` writes the rows to `lh_consensus.parquet`.
- **Retries.** A stored game from the last 3 days (`--retry-days`) that misses
  one of the ten books is fetched again.
  - Hard Rock and theScore skip some games, so those games are re-requested
    for three days and then left alone. That costs one request a day each.
- **Window.** The store starts on 2026-10-20 (`update.FIRST_DATE`). Earlier
  dates are refused unless `--dry-run`, which parses any season and can write
  the rows to Parquet (`--save-dir`) for checks, or the backfill
  (`--games-file`, see [SBR gaps in past seasons](#sbr-gaps-in-past-seasons)).
- **Failures.** A failed request or an answer that does not parse is reported
  per game and does not stop the run. The script then exits 1, so a source
  outage shows on the workflow step while the job carries on. Games left for
  the next run (not finished, no feed tipoff, held back, empty) do not change
  the exit code.
- **Log.** Each date is a collapsible group in the workflow log
  (`utils.ci.log_group`), with one line per game, flushed as it happens:

  ```
  ::group::2026-10-22: 2 game(s) planned
    matchups: 2 listed, 2 matched to nba_games
    + Cleveland Cavaliers @ Philadelphia 76ers (0022600006): 10 book(s), 2395 tick(s), last pre-tip tick 1 min before tip [7.2 MB, 6.3 s]
    - Denver Nuggets @ Oklahoma City Thunder (0022600007): not finished (tip +1.2 h ago), next run
    stored 1 new game(s) and 2395 new tick(s) (of 2395 prepared)
    + CLE @ PHI (0022600006): consensus ATS 296 away / 517 home, O/U 213 over / 317 under
    stored the consensus of 1 new game(s) (of 1 prepared)
  ::endgroup::
  ```

  `+` is stored, `-` is left for the next run (with why), `!` is a failure and
  `?` a game the matchups page does not list.

```
# first run
python scripts/update_databases/update_covers_line_history.py --create-schema
# what the job does
python scripts/update_databases/update_covers_line_history.py
# a past season, nothing written
python scripts/update_databases/update_covers_line_history.py \
    --dry-run --dates 2026-01-15 --save-dir /tmp/covers
# only the pick counts of stored games that have none
python scripts/update_databases/update_covers_line_history.py --consensus-only
```

## Validation

**Dry run on 2025-03-12, 2025-11-12, 2026-01-15 and 2026-03-05** (39 games, 7
books shared with SBR, `compare_covers_vs_sbr.py --from-dir`; 2026-10-24):

- **Coverage.** Same games per book as SBR (Fanatics 18/39 in both).
- **Freshness.** Same median minutes from the last pre-game tick to tip for
  every book.
- **Totals and spread lines.** Same as-of line in 0.99–1.00 of
  (game, book) at T-5, T-60, T-240 and T-720.
- **Moneyline prices.** 0.92–0.95 agreement. BetMGM and BetRivers are the
  lowest; the gaps are a few cents around moves.
- **Openers.** The same in 0.996 (totals, spread) and 0.964 (moneyline).
- **consensus_opener against `odds_sportsbook`.** Totals 39/39, spread line
  39/39, moneyline 38/39.

An earlier probe of 62 games on 8 dates of 2024-26 gave the same picture. It
found the same median ticks per game as SBR, 1.3% in-play rows and 6.8% exact
duplicate rows.

## SBR gaps in past seasons

Script: `scripts/line_history/covers/audit_sbr_gaps.py` (read-only). It lists
the games of `nba_games` from Covers' first history date (2019-04) to the last
finished season whose SBR line history is missing or incomplete, with the
reasons in its docstring. Its `--out` file is the input of the backfill:

```
python scripts/line_history/covers/audit_sbr_gaps.py --out /tmp/sbr_gaps.csv
python scripts/update_databases/update_covers_line_history.py \
    --games-file /tmp/sbr_gaps.csv \
    --reasons no_game no_ticks no_main_book truncated missing_books
```

- **Same store.** Backfilled games go into `line_history_covers` like the
  daily ones (archive, insert-only write, one transaction per date). A season
  before 2026-27 therefore holds **only the backfilled games**: it is a
  per-game supplement to SBR, not a season to compare with SBR as a whole.
- **No retries.** A stored game is never fetched again (`update.plan_backfill`),
  so running the backfill again picks up only the games still missing (empty
  answer, failed request). The daily job never plans a date before
  `FIRST_DATE`, so it does not touch them.
- **Truncated games are stored as Covers has them.** The 3-day hold has long
  passed.

**Dry run of the 437 games of those reasons (2026-10-24): Covers keeps SBR's
gaps.** It is the same feed, so a gap in the stored history is in both:

- **What Covers adds: 175 games.**
  - The 161 games of 2019-04-01..06-13: 4 books, bet365 in all.
  - The 2024-03-29 slate (12, six books each; `odds_sportsbook` has no book
    line for them).
  - The two 2024-03-28 games: last tick 6 min before tip, where SBR stops 11 h
    before.
- **What it does not:**
  - 168 of the 170 truncated games end at the same minute in Covers.
  - 0 of the 92 missing books are there.
  - 0022100316 has no bet365 there either.
  - 0021900945, 0022300576 and 0022300589 come back empty, and 0022000539 is
    not on its date's matchups page.
- **Against `odds_sportsbook`** (bet365 total, last tick ≥5 min before tip):
  280/291 equal and 289 within a point. The differences are 2019 playoff
  closes, which SBR scraped once and could not be repaired from history.

So historically Covers fills only what SBR never collected. Its use as a
fallback is live: on 2026-10-20 SBR's stored history ended up complete (as in
Covers), and on 2026-10-23 SBR has no ticks while Covers has the whole slate.

Run 2026-10-24 (9,080 games, 2019-04-01 to 2026-06-13; 539 flagged):

| Reason | Games | Where |
|---|---|---|
| no SBR history at all | 161 | 2018-19 from April on (SBR starts 2019-10-22); `odds_sportsbook` has a close for 32 (from 2019-05-03) |
| `no_game` / `no_ticks` | 3 / 13 | the whole 2024-03-29 slate (12), 0021900945, 0022000539, 0022300576, 0022300589 |
| `no_main_book` | 1 | 0022100316 (no bet365) |
| `truncated` | 170 | 120 of them in 2019-20 and 2020-21, mostly 60-120 min: fewer books, so quieter markets, not freezes. Clock-shared freezes: 24 games. The long ones are 2024-03-28 and 2025-26 (11 games, 3-25 h) |
| `late_start` | 106 | the history starts after T-1080 |
| `missing_books` (book on ≥90% of the season) | 92 | BetMGM + BetRivers on 2021-10-19..25 (46), Caesars 2023-12-16..29 (18), BetRivers 2025-02-11..12 (19) |

Books absent from whole seasons (not flagged per game): BetMGM in 2019-20
(38%) and 2020-21 (none), Caesars in 2019-20 (72%), BetRivers before 2021-22.
Covers has BetMGM from 2019-10 and no BetRivers before 2021-22.

## Fallback in the stored reads

Both sources serve one feed, so where SBR has a game the two agree; Covers
only fills what SBR lacks. A game is always read whole from one source, and
the books only Covers carries (`thescore`, `consensus_opener`) are dropped, so
it has the same books whichever it comes from.

The training side
(`postgre_db/line_history_covers/fallback.py`, called by
`line_history_aiven.fetch`'s `fetch_pregame_ticks`, `fetch_closing_ticks` and
`fetch_games`; `covers_fallback=False` turns it off):

- **Which games come from Covers.** A game SBR has no tick for, or whose last
  Covers tick is more than 30 min (`MARGIN_MINUTES`) later than SBR's.
  - Checked 2026-10-24: the 161 games of 2019-04..06, the 14 of
    2024-03-28/29 and the 12 of 2026-10-23.
  - No 2025-26 game: its freezes are the same in Covers.
- **After the caches.** The fallback runs after the S3 and local tick caches,
  which keep holding SBR's reads only.
- **Fingerprints.** The season fingerprints that key the odds-repair and
  intermediate memos (`season_fingerprints`, `pregame_tick_fingerprints`) get
  the Covers season's appended, only for a season Covers holds games of. Other
  seasons keep their keys.
- **The daily close repair** (`repair_sportsbook_closes.py`) reads closing
  ticks through the same function. It runs after the Covers step, so it fills
  `odds_sportsbook` rows SBR left without lines. The 2026-10-23 dry run filled
  219 cells and replaced 31 of the 12 games.
- **No `consensus_opener` from Covers.** Openers are not in the SBR line
  history, so the repair does not touch them.
- **A failed Covers read** leaves SBR's frame as it is, with a `!` line in the
  log.

## Limits

- **Undocumented site endpoints.** The markup is parsed with regular
  expressions. A redesign shows up as games without line history or as
  unmapped books in the step's warnings.
- **Only the US view.** Ontario's Pinnacle would need a second request per
  game and a new book map.
- **No money split.** The consensus page counts picks, as SBR's consensus
  does; there is no share of bets or money (that is Yahoo's,
  `odds_yahoo/`).
