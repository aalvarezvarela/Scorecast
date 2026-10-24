# TheRundown / RapidAPI backup probe

Checked **2026-10-24**, using the project's existing RapidAPI credentials and
`https://therundown-therundown-v1.p.rapidapi.com`. Scope: NBA 2026-27 only.

**Result: technically usable as a partial backup, without bet365.** The
authenticated catalog contains 25 sources. bet365, BetRivers and Fanatics are
absent both from that catalog and from the official
[sportsbook reference](https://docs.therundown.io/reference/sportsbooks).
`consensus_opener` is SBR's own source and is not supplied here either.
No source equivalence or agreement of prices with SBR has been established.

## Observed coverage

All 14 bounded live requests returned HTTP 200. No subscription changes,
database updates, S3 uploads or production ingestion changes were made.

The first sample queried UTC dates October 22, 23 and 24: 7 + 7 + 9 events,
all labelled `Regular Season`, `season_year=2027` (the season ending in 2027).
Twenty were final and three scheduled at the time of that sample.

A separate request used `offset=240` for the **October 24 Eastern-time
jornada**, plus `affiliate_ids=19,22,23,35`: eight scheduled games, including
games starting after midnight UTC on October 25. Counts below require both
sides and valid American prices; an affiliate object alone does not count.

| SBR book | TheRundown ID | Valid totals / 8 | Valid spreads / 8 | Valid moneylines / 8 |
|---|---:|---:|---:|---:|
| DraftKings | 19 | 8 | 8 | 8 |
| BetMGM | 22 | 8 | 8 | 8 |
| FanDuel | 23 | 8 | 8 | 8 |
| Caesars | 35 | 5 | 8 | 8 |
| bet365 | absent | 0 | 0 | 0 |
| BetRivers | absent | 0 | 0 | 0 |
| Fanatics | absent | 0 | 0 | 0 |

HardRock (28) is also published, but its catalog health was `unhealthy` at
the time of the probe. It is already history-only in this project. Catalog
health is useful metadata, not proof that every NBA market is populated.

## Openers, closes and movement

For UTC October 22, both `/sports/4/openers/2026-10-22` and
`/sports/4/closing/2026-10-22` returned valid totals, spreads and moneylines
on **7/7 games for each of DraftKings, BetMGM, FanDuel, Caesars and HardRock**.
The ordinary `/events/` snapshot had mostly off-board prices on those final
games, so it is not a substitute for the dedicated endpoints.

Historical movement was also accessible for a completed game: Indiana at
New Orleans, `event_id=56e05551fe710711708a47be3520ec7e`, DraftKings
`line_id=18830527`:

| Endpoint | Rows returned | Rows with valid two-sided prices |
|---|---:|---:|
| `/lines/18830527/total` | 74 | 73 |
| `/lines/18830527/spread` | 92 | 91 |
| `/lines/18830527/moneyline` | 41 | 40 |

Each history included a final off-board row. A second total-history probe,
for scheduled Miami at Charlotte (`line_id=19574377`), returned four valid
updates. These are observed rows, not a guarantee of complete tick capture.

## Documentation findings that affect an integration

- **Correct API version and credentials.** This project calls the RapidAPI
  V1 gateway with `X-RapidAPI-Key` and `X-RapidAPI-Host`. Official direct-API
  examples use `/api/v1` or `/api/v2` and `X-TheRundown-Key`. Do not send the
  RapidAPI key to the direct host or assume V2 access. The
  [migration guide](https://docs.therundown.io/guides/v1-to-v2-migration)
  describes separate V2 subscription requirements and recommends
  `teams_normalized` for stable V1 team identifiers.
- **Book filter and date boundary.** `affiliate_ids` selects published numeric
  IDs; it does not add books outside the catalog. Default `offset=0` uses UTC.
  For an Eastern NBA date, derive the offset from `America/New_York`: 240
  minutes during daylight saving time and 300 during standard time. The
  [V1 events reference](https://docs.therundown.io/api-reference/generated/v1-events/get-events-with-lines-for-a-sport-and-date)
  also documents status filters and `include=all_periods`; full-game data
  does not require the latter. Both book filtering and offset were tested.
- **Missing prices and deltas.** The documented
  [`0.0001` sentinel](https://docs.therundown.io/reference/sentinel-values)
  means off-board. Treat it as missing, including in historical rows.
  In the observed responses, fields ending in `_delta` represent changes,
  not absolute closing prices: for example, `total_over=230.5` with
  `total_over_delta=-1`. The legacy `process_odds_date` reads `_delta`
  fields for past games; do not reuse that processing for this backup.
- **Historical line identity and limits.** Use the `line_id` of the requested
  book, not the event ID. The current
  [V1 moneyline-history reference](https://docs.therundown.io/api-reference/generated/v1-lines/get-moneylines-for-an-event)
  describes default pregame full-game history, period grouping with
  `include=all_periods`, and a newest-1,000-row cap per period with no
  pagination. Treat responses reaching that size as potentially incomplete.
- **Closing cutoff needs validation.** The
  [V2 closing documentation](https://docs.therundown.io/api-reference/generated/v2-events/get-closing-lines-for-a-sport-and-date)
  defines the cutoff as scheduled start. Our V1 total closes had
  `date_updated > event_date` in 3/7 DraftKings games, 7/7 FanDuel games and
  5/7 Caesars games; BetMGM had 0/7. That may reflect actual tip-off versus
  published start or different V1 semantics. Do not assume those closes meet
  this project's pre-tip cutoff; reconstruct from history against the
  reconciled tip-off and compare to SBR first.
- **V2 history is another interface.** The
  [history guide](https://docs.therundown.io/guides/historical-odds)
  documents an event history endpoint requiring 1–3 affiliates per call,
  `from`/`to`, `main_line`, a default 1,000-row limit and maximum 5,000,
  with no cursor. Narrow time windows if the cap is reached. Direct-plan
  history windows range from 7 days to unlimited; opening/closing snapshots
  start at Pro. Those advertised tiers do not establish this RapidAPI
  account's entitlement. Its observed access includes recent completed games.
- **Polling and quota.** This RapidAPI account reported a request allowance
  of 1,000. The direct API's
  [data-point billing](https://docs.therundown.io/rate-limits) is a separate
  published model; no price or renewal period was inferred from our headers.
  Do not bootstrap V2 deltas from this V1 response: its cursor is UUID-like,
  while the [current delta guide](https://docs.therundown.io/guides/efficient-polling)
  requires a positive numeric cursor and confirmed zero-delay entitlement.
- **Storage and model use.** The published
  [licensing summary](https://docs.therundown.io/licensing-and-attribution)
  requires a supplemental agreement for retention beyond 24 hours. The
  history guide requires written permission for backtesting or model
  validation. The applicable RapidAPI contract has not been verified; the
  technical probe does not establish archival or training rights.

The [RapidAPI listing](https://rapidapi.com/therundown/api/therundown) and
its pricing/playground pages were checked, but their JavaScript-rendered
specification was not readable through the web tool. The linked official
V1/V2 references and live RapidAPI responses support the findings above.

## Repeat the probe

The read-only CLI is `scripts/line_history/probe_therundown.py`. It loads the
existing config and secrets, optionally accepts `ODDS_API_KEY` from the
environment, redacts the key, stops on errors and accepts at most three
explicit season dates. It writes a diagnostic summary, not a raw archive.

```bash
.venv/bin/python scripts/line_history/probe_therundown.py \
  --dates 2026-10-24 --offset 240 --affiliate-ids 19,22,23,35 \
  --check-open-close --history-line-id 18830527 \
  --history-markets total spread moneyline \
  --output /tmp/therundown_probe.json
```

Verification: live HTTP responses, offline checks for unavailable prices,
NaN and valid zero-point spreads, plus Ruff and Black. No full-season
download, SBR price-parity study or automatic fallback has been implemented.
