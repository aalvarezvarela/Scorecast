# MoviOdds bet365 line history

Status: **Implemented** (2026-10-24). This is the bet365 line-history source
beside SBR, stored in the Aiven schema `line_history_moviodds` from the 2026-27
season on, independently of SBR and WagerTalk/OddsLogic. This phase provides
ingestion, raw archives, readers and comparison tools. Training and prediction
do not consume it yet.

## Storage contract

`schema.DDL_STATEMENTS` is the unchanged object imported from the SBR schema.
No source tables or shared definitions are altered. The same five tables and
season partitions are used; the general Aiven backup discovers the schema.

- `lh_book`: `(1, 'bet365', 'bet365')`. The schema identifies the source. The
  API has not established whether these prices correspond to US or world.
- Markets: `1=totals`, `2=point_spread`, `3=money_line`.
- `lh_game`: NBA id, official game date, season, corrected NBA UTC tipoff,
  NBA tricodes, and the matched MoviOdds fixture id in `event_id`.
- `lh_line`: UTC `TIMESTAMPTZ`; half-point lines multiplied by two in
  `SMALLINT`; decimal prices converted to rounded American odds. Left is
  OVER/away, right is UNDER/home. Negative `mins_to_tip` means pregame.
  The primary key remains `(game_id, market_id, book_id, line_ts, season_year)`.
- `lh_load_meta`: per-season raw/inserted counts and discard reasons, UTC and
  high **clock** confidence. This makes no assertion about price equivalence.

Writers use COPY into staging and `ON CONFLICT DO NOTHING`. A game's dimension
and ticks commit together. Re-fetching cannot replace existing prices, null
suspension states, or a stored tipoff. A conflicting official tipoff postpones
that game with a warning; resolving existing rows needs a separate decision.

## Discovery and timelines

See the provider's [API documentation](https://moviodds.com/doc) and
[history documentation](https://moviodds.com/bet365-historical-odds-api/).

The default run considers every played NBA game since **2026-10-20**, even if
SBR has data. It queries `/results?sport=basketball&date=...` for the official
game day and the following UTC day: results dates refer to **finishing**, not
starting. Teams, played `nba_games` rows and the NBA schedule must agree.
Only finished matches with an unambiguous identity are accepted. The corrected
NBA schedule supplies the tipoff; a missing/ambiguous clock postpones ingestion.
MoviOdds' approximately ten-minute-later kickoff is never substituted.

`/history?fi=...&group=1453&limit=50000` supplies the main pregame group:
`1454` total, `1453` spread, `960` moneyline. Props, alternative markets and
live groups are excluded. Linked pregame/live fixture ids are accepted together;
ticks from the main group after NBA tipoff remain marked as postgame/in-play.
The provider keeps the newest capped points. Truncated downloads page backwards
with an inclusive `to` boundary and exact overlap deduplication. Missing flags,
invalid timestamps, ignored bounds or non-progressing pages reject ingestion
of that game. Empty histories remain eligible for retry.

Selection states are reconstructed by precise timestamp, applying simultaneous
updates together before keeping the final state of each minute. Stable selection
identity can resolve old missing/swapped metadata. Historical prices and
handicaps always come from that historical point. No current handicap is used
as a substitute.

The provider occasionally relabels a few snapshots of a selection with the
other team's name while its handicap and price stay with the selection id (all
three such games of the first load: 1-6% of points, both selections swapped at
the same instant, handicap sign unchanged). A selection whose labels disagree
keeps its usual identity when that identity holds at least 90% of its labelled
points (`MIN_DOMINANT_SHARE`) and every disagreeing point, with a complete
quote, continues the selection's own numbers but not the opposite selection's.
"Continues" means line within 2 points and implied probability within 0.10 of
the nearest complete point under that label; points without numbers are never
references. Numbers that came from the other side flip the handicap or the
favourite and fail. In a near-even market (e.g. -0.5/+0.5 at 1.90/2.00) both
sides' numbers fit either selection, so a relabelling there stays ambiguous and
the market is invalidated. Accepted points are counted as
`relabelled_selection` repairs; anything else is ambiguous.

Suspensions, incomplete pairs and mismatched handicaps emit **all four quote
fields null**. Readers retain those rows; snapshot selection happens before
availability checks, so an older price cannot cross a suspension. Ambiguous
selection identities invalidate the affected market for that download, with
explicit discard counters. No timestamps or lines are shifted to resemble SBR.
`is_opener` is the first valid available minute in the download, not necessarily
bet365's original opening price.

A game is complete only if its fully downloaded history contains a valid
pregame pair in all three markets. Missing games are discovered regardless of
age; stored incomplete games retry for three days. Explicit dates re-fetch
older and complete games while retaining insert-only writes. Requests start at
least one second apart, with three attempts for transient errors and respect
for `Retry-After`. One game's failure does not stop the remaining games.

## Commands

Initial schema creation and the daily update (these **write remote data**):

```bash
python scripts/update_databases/update_moviodds_line_history.py --create-schema
python scripts/update_databases/update_moviodds_line_history.py
```

Read, parse and inspect locally, with **no PostgreSQL or S3 writes**, even if
`--create-schema` is also supplied:

```bash
python scripts/update_databases/update_moviodds_line_history.py \
  --dry-run --start-date 2026-10-20 --end-date 2026-10-22 \
  --save-dir /tmp/moviodds
```

Other controls: `--dates YYYY-MM-DD ...`, `--retry-days N`, `--no-archive`.
The local save directory contains encoded `lh_line.parquet`, `lh_game.parquet`
and `run.json` (warnings, failures and discard counts). Request/game failures
produce a nonzero exit; postponements and archive failures produce warnings.

Normal ingestion archives each original response and request plus NBA matching
metadata in immutable gzip JSON bundles below
`train_data/odds_sources/moviodds/raw/v1/`. Each fetch gets a unique key,
including retries and failed/incomplete downloads. Archive failures warn and do
not prevent ingestion. `--no-archive` disables this explicitly.

The daily workflow runs this after WagerTalk, with `continue-on-error` and a
45-minute step timeout. The schema must be initialized once before deployment;
no remote initialization was performed during development.

Readers `store.fetch_games` and `store.fetch_pregame_ticks` return the shared
`GAME_COLUMNS` and `TICK_COLUMNS`. The tick reader decodes doubled lines,
returns positive minutes before tip, and uses the usual five-minute margin.
It does not filter null suspension states.

Compare against SBR in Aiven, or supply decoded local SBR `TICK_COLUMNS` Parquet
with `--sbr-ticks`:

```bash
python scripts/line_history/moviodds/compare_moviodds_vs_sbr.py \
  --seasons 2026 --from-dir /tmp/moviodds --save-dir /tmp/moviodds/comparison
```

The report shows availability, line/price equality and tick age by market at
T-0, 30, 60, 240, 720 and 1080. T-0 means the last state at least five minutes
before tip, consistently with existing readers. Moneyline has no line-equality
measure. Ages measure last price change, not confirmed polling freshness.
Discrepancies are informational and do not prevent storage.

## Validation on 2026-10-24

181 focused tests passed across MoviOdds, WagerTalk, SBR, NBA schedules and the
shared snapshot builder. New cases cover UTC midnight and both DST transitions,
missing/contradictory NBA clocks, swapped metadata, disordered selections,
handicap changes, suspension/reopening, unavailable snapshots, ambiguity,
pagination overlaps and stalls, rate limits/retries, empty/partial failures,
insert-only re-execution and dry-run write isolation.

A live CLI dry-run of the initial **16 games (2026-10-20 through 22)** used 20
requests and prepared **2,835 ticks**; all 16 have all three markets. The first
run (2,429 ticks) had rejected spread and moneyline in three games whose
selection ids carried the other team's name in a few snapshots:

| NBA game | Relabelled points | Recovered markets |
|---|---:|---|
| `0022600002` PHI @ NYK | 37 | spread, moneyline |
| `0022600088` CHI @ TOR | 8 | spread, moneyline |
| `0022600006` CLE @ PHI | 20 | spread, moneyline |

In each, the handicap sign and the favourite stayed with the selection id
through the relabelled snapshots, and the recovered quotes match SBR's bet365
before its freeze (below). There were also 40 conflicting simultaneous
selection updates, represented as unavailable states, and 9 null pregame
ticks. Some snapshots are therefore deliberately unavailable.

An isolated PostgreSQL 16 instance under `/tmp`, using only a local Unix socket,
verified (on the first run's 2,429 ticks) exact column/constraint parity with
SBR, real COPY of all 16 games, retention of null pregame ticks, zero inserts
on re-execution, no overwrites after changing submitted prices, and rollback of
a game's dimension when its COPY/merge fails. No remote database or S3 writes
were made.

Comparison used SBR responses captured on 2026-10-23 for those same 16 games
(2,576 decoded bet365 ticks), with the same authoritative NBA clocks. SBR's
bet365 history freezes about 1,830–1,900 minutes before tip in these games, so
the snapshots before that point are the like-for-like test:

| Snapshot | Spread: same quote | Moneyline: same price | Totals: same line / price |
|---|---:|---:|---:|
| T-4320 | 14/16 | 16/16 | 12/16 / 11/16 |
| T-2880 | 15/16 | 16/16 | 15/16 / 11/16 |
| T-2000 | 13/16 | 14/16 | 15/16 / 11/16 |
| T-0 (SBR frozen) | 3/16 | 3/16 | 7/16 / 8/16 |

While both sources were live they carry the same bet365 numbers; totals price
differences are a few cents and line differences half a point. At T-0 the
median last-change age is 10 minutes for MoviOdds moneyline, 11 for spread and
59 for totals, against about 1,830–1,895 for SBR, whose frozen tail explains
the low T-0 agreement. `/results` returns no 2025-26 games, so the provider
cannot be compared with SBR on earlier seasons.
