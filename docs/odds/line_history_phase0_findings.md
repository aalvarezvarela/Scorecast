# Line history -> Aiven: Phase 0 findings

Status: **Research** (checked 2026-10-22) — the 2026-08 calibration of the
line-history store; the store it led to is described in
[line_history_updating.md](line_history_updating.md).

Reproduce with `python scripts/line_history/phase0_calibration.py` (read-only).

## Summary

| Question | Answer |
|---|---|
| Tipoff source | **Not in Supabase.** `nba_game_time_index` was coded but never created in any DB. Replaced by the NBA season-schedule feed. |
| Timestamp timezone | **`Europe/Madrid`** (CET/CEST), confirmed structurally for 2021-22 onward. 2019-20 / 2020-21 low confidence. |
| game_id match rate | 96.0-100% per season. **All misses are preseason games**, which are absent from `nba_games`. |
| Tipoff coverage | 100% for every season except 2025-26 (96.3%; 31 NBA-Cup-period games missing from the feed). |
| In-play rows | **2.7-3.7% of recent-season rows land at/after tipoff.** Real leakage hazard. |

## (0) Tipoff source changed

`nba_game_time_index` does not exist on Supabase or locally, and `nba_games` has
only `game_date` — no tipoff. The `sync_game_time_index.py` script fetches one
`cdn.nba.com` boxscore per game (8,154 requests) and was returning HTTP 403.

The 403 was only a missing `Referer`/`Origin` header. But the better source is
the per-season schedule feed — **one request per season instead of ~1,400**:

```
https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{season_year}/league/00_full_schedule.json
```

It exposes `gid` (matches `nba_games.game_id`), `gdtutc` + `utctm` (tipoff UTC),
`etm` (tipoff ET), arena, and team tricodes. 7 requests returned **9,332 games
with 100% tipoff coverage**. Implemented as
`src/nba_ou/fetch_data/nba_schedule/fetch_nba_schedule.py`.

## (a) Timezone: `Europe/Madrid`

The scraped `timestamp` is naive. Two independent measurements pin it.

**1. Which DST calendar does the clock follow?** Treating the naive value as
UTC, the closing-line offset was measured inside vs outside each of the EU and
US daylight-saving windows. The two calendars flip on different dates, so they
are separable.

| Season | EU step | US step | winter offset | summer offset |
|---|---|---|---|---|
| 2019-20 | +2.79 | +0.87 | +0.52 | +3.31 |
| 2020-21 | +0.50 | +0.28 | +0.08 | +0.58 |
| 2021-22 | +0.92 | +0.40 | +0.87 | +1.78 |
| 2022-23 | +1.03 | +0.50 | +0.70 | +1.73 |
| 2023-24 | +0.90 | +0.17 | +0.92 | +1.82 |
| 2024-25 | +1.08 | +0.25 | +0.90 | +1.98 |
| 2025-26 | +0.90 | +0.05 | +0.85 | +1.75 |

A clean ~+1h step on the **EU** calendar and ~0 on the US calendar. Offsets of
**+0.85h winter / +1.81h summer** are CET/CEST.

**2. Residual after conversion.** Converting from each candidate zone, the
median closing line should land at tipoff:

| Zone | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|
| **Europe/Madrid** | **-0.27** | **-0.10** | **-0.08** | **-0.17** |
| UTC | +0.82 | +0.98 | +0.98 | +0.88 |
| Europe/London | +0.72 | +0.90 | +0.92 | +0.83 |
| America/New_York | +5.10 | +5.12 | +5.22 | +5.32 |

Madrid centres on zero; every other candidate is ~1h or ~5h late.

**Independent confirmation.** Game `0022100005` (2021-10-20, tipoff 23:30 UTC)
carries a 5-minute-cadence tick block. Under Madrid it begins 15 min before
tipoff and runs to +190 min — i.e. it brackets the game. Under UTC it would
start mid-third-quarter.

### Caveats

- **2019-20 and 2020-21 are low confidence** (12% of the dataset). Both are
  COVID-affected, sparse, and show no clean DST step; 2020-21 actually fits
  UTC/London better. These two seasons likely came from a different scrape
  environment. Calibrate them individually or load them last.
- **DST-ambiguous timestamps**: 48-911 rows per season fall in the repeated
  hour when EU clocks go back. Needs an explicit policy (`ambiguous=True` to
  take the DST reading, or drop) rather than a silent `NaT`.
- Do not carry the closing-line *level* alone as evidence; it moves with data
  sparsity. The DST step is the structural signal.

## (b) game_id match rate: preseason is the whole gap

| Season | rows | dedup | games | match | unmatched games |
|---|---|---|---|---|---|
| 2019-20 | 119,239 | 108,105 | 883 | 99.89% | 1 |
| 2020-21 | 134,621 | 123,081 | 902 | 100.00% | 0 |
| 2021-22 | 652,615 | 282,229 | 1286 | 99.94% | 2 |
| 2022-23 | 225,434 | 207,153 | 1234 | 97.18% | 63 |
| 2023-24 | 302,567 | 271,456 | 1263 | 97.23% | 65 |
| 2024-25 | 365,168 | 339,408 | 1316 | 96.03% | 68 |
| 2025-26 | 675,151 | 592,624 | 1270 | 98.66% | 43 |

Every unmatched game is **preseason** — 2024-25 misses cluster on Oct 4-7 2024
(season opened Oct 22); 2025-26 on Oct 10-11 2025 (opened Oct 21). `nba_games`
holds only regular season and playoffs.

**Decision: drop preseason.** It should not feed a totals model anyway. This
turns the "unmatched" gate green rather than requiring a name-mapping fix.

## (c) Tipoff coverage

100% everywhere except **2025-26: 96.29%** — 31 games, all dated Dec 9-16 2025
(NBA Cup knockout period), with `002`/`006` game-id prefixes absent from the
static schedule feed.

**Fix:** fall back to the per-game `cdn.nba.com` boxscore endpoint for the
missing ids (31 requests, and it works with the corrected headers).

## In-play rows: the significant new finding

SBR line history includes **in-play** ticks, not just pre-game movement. They
are labelled `row_kind = "history"` exactly like pre-game rows — `mins_to_tip`
is the only way to separate them.

| Season | rows at/after tipoff |
|---|---|
| 2019-20 | 24.28% |
| 2020-21 | 7.41% |
| 2021-22 | 8.56% |
| 2022-23 | 3.74% |
| 2023-24 | 3.09% |
| 2024-25 | 2.74% |
| 2025-26 | 2.79% |

An in-play totals line is conditioned on points already scored, so using one as
a pre-game feature is direct target leakage. This also explains the row-count
outliers: 2021-22 (652k) and 2025-26 (675k) captured dense in-play ticking that
other seasons did not.

**Implications for the plan:**

1. `mins_to_tip` is not a convenience column — it is the leakage filter. It must
   be `NOT NULL` for any row used in training.
2. Feature queries must filter `mins_to_tip < 0` (recommend `<= -5` for safety).
3. Consider a stored `is_pregame` boolean so the filter cannot be forgotten.
4. In-play rows are worth keeping — they are a genuine asset for post-hoc
   analysis — but they must never default into a feature set.

## Post-load data quality (found by verifying the loaded store)

Two scraper defects, both caught by structural tests rather than by eyeballing
ranges. A valid spread is mirrored (`left_line = -right_line`); a valid total
quotes the same number on both sides (`left_line = right_line`).

**Spread price-bleed -- 1,054 rows (0.208% of spreads), 0 totals.** On a pick'em
the SBR cell holds only a price and no spread number, and the scraper's
`([+-]\d+(?:\.\d+)?)` pattern matched that price as the line. Confirmed against
`left_value_raw`, which reads `"-110"` on these rows versus `"-5 -110"` on valid
ones. The give-away is complementary *price* pairs (-110/-110, -115/-105).
Repaired by relabelling the value as the price it demonstrably is; the spread is
left NULL rather than inferred as 0, since the source never said so.

**Impossible pre-game lines -- 12 rows.** 10 totals (including one `2285`, i.e.
`228.5` with the decimal lost) and 2 spreads. Cleared, keeping prices and the row.

**In-play spreads are not a defect.** 119 of the 121 spreads beyond 30 points are
in-play rows -- live lines during a rout -- so the plausibility guard applies to
pre-game rows only.

Both are guarded in `transform.py` for future loads and repaired in place by
`scripts/line_history/repair_spread_price_bleed.py` (idempotent).

Post-repair pre-game ranges: totals 159.5-264.5, spread -27.5 to +25.0.

## Plan revisions

- **New dependency:** the schedule fetcher becomes part of the load pipeline;
  `lh_game.tipoff_utc` is populated from it, not from `game_time_index`.
- **Add** `is_pregame BOOLEAN NOT NULL` to `lh_line`.
- **Add** a per-season timezone map (Madrid for 2021-22+; 2019-20 / 2020-21
  flagged) rather than one hardcoded zone.
- **Add** an explicit DST-ambiguity policy.
- **Drop preseason** at load time, by construction.
- Size estimate is unchanged (~240 MB); dropping preseason removes ~2-4%.
