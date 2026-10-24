# Odds source alternatives to Sportsbook Review

Status: **Partial** (checked 2026-10-22) — research notes from 2026-10-21. Only
the OddsTrader archive is implemented (2026-10-22, "S3 archive" below): a copy
in S3, not a model input. Using any source as a backup or as features remains.

The question was
whether another source could back up Sportsbook Review (SBR) when the site is
down, or add more books to the odds features. **Adding** books is the
direction worth keeping. **Replacing** SBR is not possible today without
losing books the models use.

Prices and coverage change often. Re-check them before acting on anything
below.

## What the pipeline needs from an odds source

- **Books admitted as features:** bet365 (the main book, `MAIN_SPORTSBOOK`),
  DraftKings, FanDuel, BetMGM, BetRivers and Fanatics, plus SBR's own
  `consensus_opener`. Hard Rock Bet is stored but history-only
  (`HISTORY_ONLY_BOOKS` in `src/nba_ou/config/odds_columns.py`).
- **Markets:** total, spread and moneyline, each with line and American price.
- **Line history:** every pre-game tick with a minute timestamp. The
  intermediate snapshots at 30–1080 minutes before tip, and the movement
  features, are built from these ticks.
- **Depth:** closing lines from 2019-05 (BetRivers from 2021-10).

The history already lives in Supabase (`odds_sportsbook`) and Aiven
(`line_history`). A short SBR outage does not need another source:
`scripts/line_history/update_line_history.py` fills every game present in
`nba_games` and missing from `lh_game` once SBR is back, because SBR's
line-history page keeps all ticks of past games. Another source matters only
in three cases: SBR is down at prediction time, SBR is gone for good, or we
want books SBR does not carry.

## Commercial APIs

Prices are from the vendors' own pages on 2026-10-21 unless marked otherwise.

| Provider | Price / month | bet365 US | DK / FD / MGM | BetRivers | Fanatics | Hard Rock | Line history | From |
|---|---|---|---|---|---|---|---|---|
| The Odds API | $30 (20K credits), $59 (100K), $119 (5M), $249 (15M) | no (only `bet365_au`) | yes | yes | yes (paid plans only) | yes (`us2`) | Snapshots every 10 min, every 5 min since 2022-09 | 2020-06-06 |
| Odds-API.io | £49 (2 books), £99 (5), £179 (10), £229 (15) | "bet365 NJ" | yes | yes | yes | yes | Closing lines only; `/odds/movements` only for bet365, MGM, DK, FD, depth not stated | 2025-12 |
| OddsPapi | Free tier 250 req/month; paid tiers not published | listed among UK books, so possibly not the US feed | yes | yes | yes | yes | Change log per fixture (`changedAt` in ms), 1–3 books per call; historical calls do not count against quota | 2026-01 |
| SportsGameOdds | Pro $299 (another of its pages says $499) | Pro plan | yes | unknown | Pro plan | unknown | Opening and closing line per book only | 2024-02 |
| BALLDONTLIE (GOAT) | $39.99 | per docs, not verified | yes | yes | yes | no | Current and opening line only | 2025 season |
| TheRundown | $49 to $2,499 | no | yes | no | no | yes | Every change with `updated_at` | history window depends on plan |
| SportsDataIO | Contract | not listed | yes | yes | yes | ? | Line movement endpoints | not stated |
| OpticOdds / Unabated | About $5k / $3k (third-party estimates) | yes | yes | yes | yes | yes | Yes | several years |

Notes:

- **Two history models.**
  - *Snapshot APIs* (The Odds API) return the market state at a requested
    timestamp, plus `previous_timestamp` and `next_timestamp`. Movement is
    rebuilt by walking snapshots: 10 credits × markets × regions per call,
    about 8K credits per NBA day at 5-minute resolution.
  - *Change-log APIs* (TheRundown, OddsPapi, SportsDataIO, OpticOdds) return
    every tick of one game in one call, the same shape as SBR's line-history
    page.
- **TheRundown's licence** requires a supplemental agreement for "backtesting,
  training … machine-learning or statistical models" and limits storage to
  24 h. That rules it out regardless of coverage.
- **The best fits:**
  - for live backup, Odds-API.io, the only cheap one listing a US bet365;
  - for cheap recent change logs, OddsPapi;
  - for the most mature product, The Odds API, at the cost of losing bet365.
- **Before trusting any of them,** run it in parallel with SBR for a few weeks
  and compare per game, book and horizon. The models learned SBR's values,
  and a systematic half-point difference is larger than the edge being chased.

### TheRundown: existing RapidAPI account tested (2026-10-24)

The existing credentials work for NBA 2026-27: live per-book quotes, openers,
closes and all three core market histories were retrieved. DraftKings,
FanDuel, BetMGM and Caesars overlap with SBR; bet365, BetRivers and Fanatics
are absent from the authenticated catalog and official book reference.
This is a possible **partial** technical backup. V1 closing timestamps need
comparison against reconciled tip-offs, and the applicable RapidAPI storage
and model-use rights remain unverified. See the
[bounded probe and documentation review](therundown_backup_probe.md) for
coverage counts, API-version distinctions, limitations and the repeatable CLI.

## OddsTrader

`www.oddstrader.com` advertises closing and line-history data from 2020.

### Access

- **The website is not scrapeable.** Every path, including `robots.txt`, sits
  behind a Cloudflare managed challenge (Turnstile). A plain Playwright
  Chromium, headless or headful, loops on the challenge. No anti-detection
  tooling was tried.
- **The data API is open.** The site renders from a GraphQL service on a
  separate host, `https://ms.virginia.us-east-1.oddstrader.com/odds-v2/odds-v2-service`,
  served by CloudFront with no Cloudflare and no auth.
  - Requests go as GET with `?query=<url-encoded GraphQL>`.
  - Schema introspection is enabled.
  - About 300 requests at 1/s caused no blocking.
- **It is the legacy SBR odds platform:** the same schema the old `pysbr`
  library used, and the token embedded in the frontend is issued by "SBR".
  It is a separate stack from today's sportsbookreview.com (different event
  ids and host), but probably the same lineage. Do not count it as an
  independent backup of SBR.
- **Quirks:**
  - field aliases break its proxy ("Undefined response from proxy");
  - URLs over about 8 KB return 414;
  - `sportsbooks(...)` fails because the proxy injects bad defaults, so book
    names come from `sportsbook(sbid: X) { paid nam }` one at a time. `sbid`
    is not `paid`: several sbids point at one `paid`, and the names are
    messy, so the `paid` → book map is curated in
    `src/nba_ou/fetch_data/odds_oddstrader/books.py`;
  - lines are returned only when `paid` is passed explicitly.
- **Batching works.** A POST whose JSON body is a list of `{"query": ...}`
  objects returns a list of results. `lineHistory` takes one `paid` and one
  `mtid`, so a game is ~80 queries (about 27 books × 3 markets); batched, that
  is one request of ~1 s and ~70 KB gzipped. POST also avoids the 414 limit.

### Ids and queries (NBA)

- **League:** `lid` 5 (also `spid` 5).
- **Markets (`mtid`):** 402 total, 401 spread, 83 moneyline, all including
  overtime. Totals use the Over/Under participants `partid` 15143 / 15144.
- **Day's games:** `eventsByDateNew(lid: [5], startDate: <epoch ms>, hoursRange: 24, mtid: [402], showEmptyEvents: true) { events { eid dt des es } }`.
  `dt` is the tip-off in epoch ms and `des` is `"Away@Home"`, with
  inconsistent names ("San Antonio", "L.A. Lakers"). The reliable teams are
  `participants { partid ih source { ... on Team { nam abbr } } }`: `ih`
  marks the home team, `abbr` is the NBA tricode and the team `partid` is
  stable across seasons. `scores { partid val pn }` gives points per period.
- **Books present on a game:**
  `currentLines(eid: [<eid>, ...], mtid: [402, 401, 83], paid: [1, ..., 400])`.
  `eid` takes a list, so one request covers a whole slate. Add
  `needSbid: true` to get `sbid`. A book with no current line on a game has
  no history on it either: none of the 600 absent (market, `paid`) pairs of a
  2024 game had any.
- **Ticks:** `lineHistory(eid: <eid>, mtid: 402, paid: <paid>, partid: [15143, 15144]) { tim lines }`.
  - `partid` is required; without it the result is empty.
  - Each line carries `tim` (epoch ms of the change), `adj` (the line), `ap`
    (American price), `pri` (decimal price), `partid` and `lineid`.
  - Spread and moneyline use the same query with `mtid` 401 / 83 and the two
    team `partid`s (tested 2026-10-22). A spread `adj` is the team's handicap,
    so a home favourite is negative, the same sign as SBR's `homeSpread`. On
    the moneyline `adj` is 0.
  - The minute buckets (`tim`) are not one quote each: a bucket can hold
    several sub-minute updates of a side, and a line's own `tim` (ms) can fall
    in another minute than its bucket's. The parser replays lines by their own
    `tim`.
- **`paid` book ids:**

  | Id | Book | Id | Book |
  |---|---|---|---|
  | 106 | bet365 US | 5 | bet365 (international feed) |
  | 99 | DraftKings | 98 | FanDuel |
  | 109 | BetMGM | 113 | BetRivers |
  | 101 | Caesars | 102 | SugarHouse |
  | 107 | Unibet | 122 | SBK |
  | 20 | Pinnacle | 8 | BetOnline |
  | 9 | Bovada | 10 | Bookmaker |
  | 44 | Heritage | 82 | MyBookie |

  There is no Fanatics and no Hard Rock. The US books also have per-state
  `sbid`s (3150–3185); the frontend's name map is in its `main.js`.

### Coverage over time

Measured on one or two games per sampled date:

- **History starts with the 2006-07 season.** 2006-01-16 has no events;
  2007-01-15 has lines from nine books with ticks. Through 2018-19 they are
  offshore and international books (5Dimes, BetOnline, Bovada, Bookmaker,
  Pinnacle, bet365 international, William Hill, ...), 20 to 35 per game.

| Period | US regulated books present |
|---|---|
| 2019-20 | bet365 US, SugarHouse, Unibet, plus offshore books and Pinnacle |
| 2020-21 | adds FanDuel, BetMGM, BetRivers, Caesars, SBK |
| 2021-22 to 2024-25 | adds DraftKings: all five model books plus Pinnacle |
| 2025-26, Oct to mid-Dec | DraftKings and Pinnacle gone; FanDuel and bet365 often stale for hours before tip |
| from 2025-12-15 to 2026-01-20 onward (still true on 2026-10-21) | **none**: only offshore books (BetOnline, Bovada, Bookmaker, Heritage, MyBookie, Bet105, SportsBetting, JustBet, Everygame, BetAnything) |

### Parity with SBR (totals)

The check covered 20 games over 2022-01-15, 2023-02-10, 2024-03-05 and
2025-01-20, four or five per date. For each game and book, the line in force
at tip and at 60, 240 and 720 minutes before was taken from both sources,
with SBR ticks from our own scraper.

| Book | Same line at tip | Same at 240 min | Pre-game ticks, median (SBR / OddsTrader) |
|---|---|---|---|
| BetMGM | 20/20 | 20/20 | 12.5 / 10.5 |
| bet365 | 17/20 | 17/20 | 11 / 11 |
| DraftKings | 14/16 | 14/16 | 20.5 / 13.5 |
| FanDuel | 15/20 | 15/20 | 12 / 11 |
| BetRivers | 16/20 | 19/20 | 25.5 / 49.5 |

- Through 2024-25 every mismatch is ±0.5 and comes from timing: a
  last-minutes move one source caught and the other did not.
- On the 2025-11-15 games the gaps grow to 1–3 points, with last ticks 5–19 h
  before tip.
- **All three markets** were compared on 2026-10-22 from the archive's own
  tables, over the 14 games of 2024-01-15/16 and the six shared books
  (Caesars included). Same line in force: totals 88–95% and spread 87–99%
  across T-5/60/240/720. Same moneyline price: 71–86%. Line gaps are ±0.5–1
  and moneyline gaps a few cents (−180 vs −179), so the cause is timing
  again. Caesars' moneyline agrees least (25–43%), with one-cent differences.

### Assessment

- **As a backup for 2021–2025 history of the five model books:** usable,
  nearly the same feed as SBR.
- **As a live source:** useless for the current feature books.
- **To add books:** it offers a consistent offshore panel from 2019 to now:
  BetOnline, Bovada, Bookmaker, Heritage and MyBookie, plus Pinnacle until
  2024-25. That is the one additive use worth remembering. Those books would
  enter through a new schema layer and be tested like any other feature
  family.
- **Risk:** an undocumented internal API with no terms of use. It can close at
  any time, as SBR's previous public GraphQL did.

### S3 archive

Everything the API holds for the NBA is copied to S3, shaped like the SBR tick
store so that loading it into Postgres later is a plain copy. Nothing is
written to Postgres, there is no daily job, and no feature reads it.

```
s3://adrian-nba-model-registry-eu-west-1/train_data/odds_sources/oddstrader/
  raw/v1/<YYYY-MM-DD>/events.json.gz   the slate (eventsByDateNew) + currentLines of its games
  raw/v1/<YYYY-MM-DD>/<eid>.json.gz    one game: every lineHistory answer, as received
  line_history/                        same layout as backups/db/<date>/line_history/
    lh_book, lh_market, lh_game, lh_line_<season_year>, lh_load_meta (.parquet), manifest.json
```

- **Under `train_data/`:** the local IAM user cannot write a new top-level
  prefix. The NBA injury-report PDFs, the other raw source archive, also live
  under `train_data/`.
- **`raw/` is the source of truth and insert-only.**
  - An object that exists is never overwritten.
  - A game is stored only when all its queries answered cleanly, and only 8 h
    after tip-off.
  - So an interrupted run is resumed by rerunning it, and the tables can be
    rebuilt without the API.
- **`line_history/` is derived and rebuilt whole.** It goes through the SBR
  ingest path (`ingest.build_frames`), so it has the Aiven columns, types and
  encodings:
  - doubled half-point SMALLINT lines, with the price sentinel nulled;
  - NOT NULL `mins_to_tip` and `is_pregame`;
  - `is_opener` on the first tick per book and market;
  - `left`/`right` = OVER/UNDER on totals, AWAY/HOME on spread and moneyline.

  Beyond the Aiven columns, it differs in four ways:
  - `book_id` is the OddsTrader `paid`, also kept as `lh_book.source_book_id`;
  - `lh_game.event_id` is the OddsTrader `eid`, and its teams are tricodes;
  - `lh_load_meta` records `UTC` / `high`, because the timestamps are epoch ms;
  - a relisted game keeps the event with more ticks.
- **Book slugs match SBR's** for bet365 (US feed, `paid` 106), DraftKings,
  FanDuel, BetMGM, BetRivers and Caesars. A future merge therefore lands them
  on the same `lh_book` rows. Books without a confident name are stored as
  `oddstrader_<paid>`.
- **Seasons: 2014-15 onward**, in `raw/` and in the tables. The source goes
  back to 2006-07, with 23 to 32 books per game through 2014-17 as well, but
  `nba_games` starts with 2014-15, so older games could not get a `game_id`
  and were not fetched (`FIRST_SEASON` in `backfill.py`).
- **Reading:** `nba_ou.fetch_data.odds_oddstrader.store.load_ticks(seasons)`
  returns the frame `fetch_pregame_ticks` returns for the Aiven store.

Archive contents (built 2026-10-23, all 2014-15 to 2026-10-21):

- **Raw:** 4,403 dates and 16,976 events, all fetched with no gaps.
- **Tables:**
  - 15,474 games matched to `nba_games`, about 1,310–1,323 per full season;
    the rest are preseason and cancelled listings;
  - 21.4M ticks and 72 books, 166 MB of Parquet;
  - 20 books are unnamed (`oddstrader_<paid>`). Some have long coverage:
    `paid` 15 2014–25, 123 from 2021.
- **Orientation, every season:** the std of (home margin − market home margin)
  at the close is 11.8–14.0 as stored against 17.1–21.4 flipped. The
  devigged home moneyline probability is 0.54–0.59.
- **Parity with SBR**: share of (game, book) with the same line, or the same
  price for moneyline, from T-5 to T-720:

  | Season | Totals | Spread | Moneyline |
  |---|---|---|---|
  | 2021-22 to 2024-25 | 0.85–0.97 | 0.88–0.97 | 0.68–0.80 |
  | 2025-26 (Oct–Dec only) | 0.82–0.89 | 0.88–0.95 | 0.60–0.73 |
  | 2019-20 (bet365, FanDuel) | 0.46–0.83 | 0.72–0.90 | 0.89–0.95 |

  - The moneyline gaps are mostly Caesars (0.20–0.36 in every season) and
    BetMGM in 2022-23 (0.27–0.31). These are cent-level price differences.
  - In 2019-20, totals agree only 46% at T-5 but 83% at T-720. That is the
    pattern of a timestamp offset. 2019-20 is one of the two SBR seasons whose
    timezone could not be pinned (`lh_load_meta` confidence "low"), while
    OddsTrader's times are explicit UTC. This archive could therefore settle
    SBR's 2019-20 and 2020-21 timezone. Not done yet.
- **Coverage notes:**
  - 5Dimes ends with 2022-23 and Pinnacle with 2024-25.
  - MyBookie's last pre-game tick is a median 4–6 h before tip from 2022-23
    on, so it is stale at the close.

Commands, under `scripts/line_history/oddstrader/`:

- `backfill_oddstrader_raw.py`:
  - fetches the raw archive, season by season in priority order (2021-24
    first);
  - is resumable, and `--dry-run` reports what is left;
  - takes about 2.5 s per game, roughly 12 h for 2014-15 to today.
- `build_oddstrader_line_history.py` turns `raw/` into `line_history/`.
- `compare_oddstrader_vs_sbr.py` reports coverage, freshness, the spread and
  moneyline orientation against realised margins, and parity with SBR.

## OddsLogic (WagerTalk)

The odds on wagertalk.com come from OddsLogic's free widget. It answers with
each book's full line history: spread, total and moneyline, timed in UTC,
archived from 2020-12-01. It has DraftKings, FanDuel, BetMGM, Caesars,
Fanatics, Hard Rock and BetRivers (unlisted id 60), plus ESPN BET, Circa,
South Point and Kalshi. **It has no bet365.**

Since 2026-10-23 it is stored daily in the Aiven schema
`line_history_oddslogic`, from the 2026-27 season on. Endpoints, book
identification, parsing rules and parity with SBR:
[oddslogic_line_history.md](oddslogic_line_history.md).

## What sits behind sportsbookreview.com today

- **Front end:** Next.js under `/sbr-odds/`, served through CloudFront
  (`x-powered-by: Next.js`, `via: … cloudfront.net`).
- **No odds API is exposed.** Pages are rendered on the server with the data
  embedded in `__NEXT_DATA__`, which is what our scraper reads. The client
  bundles call no odds API, and the old `/ms-odds-v2/odds-v2-service` path on
  sportsbookreview.com redirects (301) instead of answering.
- **The upstream provider is not visible from outside.** `go.metabet.io`
  appears only as a stylesheet and in affiliate-link tracking
  (`sportsbookreview.go.metabet.io`), not as a data call.
- **SBR's book ids** (`sportsbookId`: bet365 2, DraftKings 42, FanDuel 34,
  Caesars 435) do not match OddsTrader's, apart from bet365.
- **Ownership:** Wikipedia says SBR "was included in a $1.2 billion
  acquisition by Genius Sports in February 2026". That deal is Genius
  Sports' purchase of Legend (Covers.com, Casino.org), which closed on
  2026-05-01. The primary sources read do not name SBR, and OddsTrader's
  ownership was not established.
  - If SBR is part of Genius Sports, its odds pages could change or move
    behind a paid product.
  - The timing next to OddsTrader losing the US books in 2025-12/2026-01 is
    a coincidence we cannot explain.
