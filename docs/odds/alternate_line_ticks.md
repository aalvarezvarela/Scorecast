# Alternate-line ticks in the SBR/Covers line history

Status: **Implemented** (2026-10-27). Reader-side filter for the 2026-27 season
on: `src/nba_ou/postgre_db/line_history_aiven/alt_lines.py`.

## What is wrong

A book offers one main total and spread, priced near -110/-110, plus a ladder
of alternate lines priced away from it. The line history SBR and Covers share
should carry the main line only. At times it also carries the alternates'
updates as if they were the main line's. For example, bet365's total for
WAS-MIL on 2026-10-21 at 14:05Z is recorded as 231.5 at -245/+172, while
the main line was 238.5 at -110/-110 according to MoviOdds.

Audit of 2026-10-27, comparing each tick with the median of the other books
at that moment:

| Ticks captured | Totals/spread ticks that are alternate lines |
| --- | --- |
| 2019 - March 2026 | <= 0.05% |
| April - June 2026 (2025-26 playoffs) | 1.3 - 3.9% |
| 2026-10-21 12:00Z - 2026-10-22 ~20:30Z | 50 - 84%, every book |
| From 2026-10-23 | ~0.1% |

- **Moneyline is clean.** It has no alternate lines.
- **Not news:** while a book's spread or total jumped ±9 points, its
  moneyline moved by a median of 0.003 in probability.
- **Priced like a ladder:** the prices follow the offset, with an implied
  sigma of 20.7 points for totals and 15.6 for spreads. About 10% pair the
  prices of two different lines (+108/+137).
- **Upstream, not our parser:** the raw SBR payload holds these entries
  with no field telling them apart, Covers holds the same, and MoviOdds
  (bet365) and OddsLogic show the main line unmoved.

What it broke: every prior-game line-dynamics feature (`*_ABS_MOVE_OPEN_TO_CLOSE_*`,
`N_MOVES`, `LINE_STD`, `N_PRICE_ONLY_TICKS`, team and league windows and the
`*_SEASON_BEFORE` means) of every game from 2026-10-23, and the snapshot lines
of the games of 2026-10-21 and 2026-10-22. The intermediate spread slots then
picked AWAY on every game. The closing lines were not affected: the day-page
odds and `closing_line_repair` kept them within 1.5 points of MoviOdds.

## The filter

Centering an alternate to -110 recovers only about 75% of its offset, and the
flips would still read as moves. So these ticks are dropped instead: the
book's previous tick stays its line for that moment. A totals or spread tick
with valid prices (|American| >= 100) is dropped when it is:

- **unbalanced:** the devigged probability of either side is outside
  0.40-0.60. No main-line tick of 2025-26 was;
- **impossible:** the overround is outside 0.02-0.10. The 0.1st-99.9th
  percentile of 2025-26 main lines is 0.036-0.089;
- **a near alternate:** 3 or more points from the median of the other books
  (at least two) at that moment, and priced for that offset, with a lean of
  0.03 or more towards the favoured side. A book leading a real move is off
  the others too, but priced near -110, so it stays. The median reads only
  the ticks not dropped yet, and this rule is repeated until a pass drops
  nothing more. Otherwise several books' alternates quoted at once hold the
  median up between them (61 more ticks of 2026-27 than with one pass).

The filter runs at the end of `fetch_pregame_ticks`, after the Covers
fallback, and on tonight's live line history (`fetch_live_game_day`).
`fetch_closing_ticks` normally keeps only the last ticks of each book
(`last_n`, cut in SQL). For the filtered seasons it reads each book's whole
history instead, filters it, and cuts the tail afterwards. Cutting first lost
the close of 113 (game, market, book) cells of 2026-27, whose last five ticks
were all alternates. Reading the 2026-27 season whole takes ~10 s. Training and prediction therefore read the same
ticks, and so do the closing-line repair and the intermediate builder.

Measured on the stored ticks:

| | Dropped |
| --- | --- |
| 2025-26 regular season (not filtered; what the rule would drop) | 0.08% |
| 2025-26 playoffs (not filtered) | 4.2% |
| 2026-10-21 / 2026-10-22 | 80% / 56% |
| Other days of 2026-27 | 1-3% |

The mean prior-game `ABS_MOVE_OPEN_TO_CLOSE` per game goes from 113 points
(raw 2026-27) to 9-12, against 8-12 in 2025-26.

## Scope and caches

- Only seasons from `FIRST_FILTERED_SEASON` (2026) are filtered. Earlier
  seasons feed the frozen 2_5 base. The 2025-26 playoffs are left as they are
  until a base rebuild decides otherwise.
- The filtered seasons' fingerprints carry `FILTER_VERSION`
  (`pregame_tick_fingerprints`, and the closing ticks'
  `attrs["season_fingerprints"]`). The intermediate snapshot-history memo
  is keyed on these fingerprints, so the 2026 entry is rebuilt rather than
  served unfiltered. Change `FILTER_VERSION` whenever the rule changes.
- The raw-tick input caches hold the store's rows as stored. The filter runs
  after them.
- After a change to the filter, rebuild the filtered seasons' partitions
  (`scripts/create_train_data/build_season_partitions.py rebuild`).

Not covered: DraftKings' MIA-MIN (0022600004) totals flip between 235.5 and
242.5, both at -110, from August to September 2026 (8.4k ticks). Both are
priced like main lines, so the rule cannot tell them apart. The per-game
median over books keeps that game's dynamics in range.
