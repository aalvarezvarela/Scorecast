# Odds source alternatives to Sportsbook Review

Status: **Research** (checked 2026-10-22) — notes from 2026-10-21; nothing here
is implemented.

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
    names come from `sportsbook(sbid: X)` one at a time;
  - lines are returned only when `paid` is passed explicitly.

### Ids and queries (NBA)

- **League:** `lid` 5 (also `spid` 5).
- **Markets (`mtid`):** 402 total, 401 spread, 83 moneyline, all including
  overtime. Totals use the Over/Under participants `partid` 15143 / 15144.
- **Day's games:** `eventsByDateNew(lid: [5], startDate: <epoch ms>, hoursRange: 24, mtid: [402], showEmptyEvents: true) { events { eid dt des es } }`.
  `dt` is the tip-off in epoch ms and `des` is `"Away@Home"`.
- **Books present on a game:**
  `currentLines(eid: [<eid>], mtid: [402], paid: [1, ..., 400])`. Add
  `needSbid: true` to get `sbid`.
- **Ticks:** `lineHistory(eid: <eid>, mtid: 402, paid: <paid>, partid: [15143, 15144]) { tim lines }`.
  - `partid` is required; without it the result is empty.
  - Each line carries `tim` (epoch ms of the change), `adj` (the line), `ap`
    (American price), `pri` (decimal price), `partid` and `lineid`.
  - Spread and moneyline use the same query with `mtid` 401 / 83 and team
    `partid`s. This was not tested.
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
