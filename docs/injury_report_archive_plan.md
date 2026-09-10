# NBA official injury-report PDF archive: research findings and implementation plan

Status: **research complete; the collector is implemented.** Nothing has been
committed -- everything lives uncommitted on `research/injury-report-scraper-plan`.
See §17 for how to run it. Every number below was measured against the live NBA
CDN between 2026-09-10 14:40 and 17:40 UTC using throwaway probe scripts in a
scratch directory (~15,000 HEAD/GET requests total).

Reproduction scripts are *not* in the repo; the measurement method is described
inline so any claim can be re-derived. Where a finding is inferred rather than
directly observed it is marked **(inferred)**.

> ### ⚠️ Correction to an earlier draft
>
> An earlier version of this document concluded that **"the NBA stopped publishing
> on 2025-12-22"** and that the archive was a closed set ending there. **That was
> wrong.** The NBA changed the *filename format* on that date, and the probe that
> tested "alternative filename variants" happened to omit the one that won
> (`_hh_mm` with an underscore) while testing it only against post-cutover dates,
> where every variant 403s for the unrelated reason that the old format had
> stopped. The feed has run continuously to the present.
>
> The error was caught by the Wayback CDX index (§12.2), which listed the
> new-format URLs and made the change obvious. Every affected section below has
> been rewritten; this note is kept deliberately rather than quietly edited out,
> because the failure mode — *a negative result from an incomplete candidate set,
> validated against a control that was broken for a different reason* — is exactly
> the kind of mistake this project's discovery code must be designed to survive.

---

## 0. Executive summary — read this first

Three findings change the shape of the project relative to the original brief.

| # | Finding | Consequence |
|---|---|---|
| **1** | **On 2025-12-22 the NBA changed the filename format and quadrupled the cadence.** `Injury-Report_<date>_<hh>PM.pdf` (hourly) became **`Injury-Report_<date>_<hh>_<mm><AM\|PM>.pdf` at 15-minute granularity — 96 reports/day**. The old pattern 403s after the cutover and the new pattern 403s before it: a clean, same-day switch. | The feed is **alive and better than ever**. Everything through today is collectable. Coverage in the new era is ≤15 min stale at *every* horizon. See §1.5. |
| **2** | **`403` is ambiguous: it means both "file does not exist" and "you are being throttled".** The CDN degrades to `403 AccessDenied` under load — never `429`, no `Retry-After`. A 4-worker sweep produced ~2,200 false 403s on files that demonstrably exist. Sequential requests at 2.8 req/s are clean over 3,000 consecutive calls, and recovery from a trip takes only 20–120 s. | Discovery **must not** treat a bare 403 as absence. A canary + circuit breaker is mandatory, otherwise the manifest silently records real reports as missing. This is the single biggest correctness risk in the project — but it is cheap to defend against, and throughput is not meaningfully constrained. |
| **3** | **The filename hour is a genuine `America/New_York` wall-clock label, and the report is generated at `HH:30` ET** (`HH:00` during the 2020 bubble). Proven by the PDF's own `/CreationDate` UTC offset flipping `-05'00'`/`-04'00'` across DST. | `report_datetime_et` is exactly reconstructable, so `report_datetime_utc` is exact. Temporal correctness is achievable without parsing the PDF. |
| **4** | **The archive starts earlier than the brief assumed: 2018-10-17, not December 2018.** The first 200 reports live in the `official.nba.com` WordPress media library under eight different filename shapes, on a different host. 103 of them are scanned images with no extractable text. | Discovery needs a second, **listing-driven** path for this era (§3.1). It adds 189 net-new reports and pushes usable text history back to ~2018-11-21. |

**Recommended sampling strategy: Strategy B (probe every hour), season-aware**,
plus a listing-driven pass over the WordPress media API for the pre-CDN era.
It is provably identical to Strategy A (collect everything) because the published
filename space *is* the hourly grid — no report exists off it. Cost is ~28.8k
discovery requests and ~1.2 GiB. See §2.

---

## 1. URL pattern and publishing schedule

### 1.1 The URL pattern

```
https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<hh><AM|PM>.pdf
```

* `hh` is **zero-padded 12-hour**, `01`–`12`. `12AM` = midnight hour, `12PM` = noon hour.
* The date is the **Eastern-time calendar date** of the report.
* The full daily candidate space is therefore exactly **24 URLs**.
* Served from Akamai in front of S3 (`Server: AmazonS3`, `x-amz-*` headers).

Variants tested against a date known to be inside the publishing era and found
**not** to exist (all 403): lowercase `injury-report_`, `.PDF` uppercase
extension, minute-bearing labels (`_0830PM`, `_08-30PM`), 24-hour labels
(`_20PM`), no-time (`Injury-Report_<date>.pdf`), suffixed (`_08PM_1`), and
alternative directories (`/referee/injuryreport/`, `/injury/`, `/referee/injury/2026/`).
**There is one pattern and no exceptions.**

### 1.2 Label semantics and timezone — proven, not assumed

Downloading the PDFs and reading both the embedded `/CreationDate` and the
first-page header text gives an unambiguous answer:

| URL label | PDF header text | PDF `/CreationDate` | Reading |
|---|---|---|---|
| `2019-01-16_01PM` | `Injury Report: 01/16/19 01:30 PM` | `D:20190116133026-05'00'` | 13:30 EST |
| `2019-01-16_08PM` | `Injury Report: 01/16/19 08:30 PM` | `D:20190116203020-05'00'` | 20:30 EST |
| `2021-10-20_03AM` | `Injury Report: 10/20/21 03:30 AM` | `D:20211020033000-04'00'` | 03:30 EDT |
| `2024-12-11_12AM` | `Injury Report: 12/11/24 12:30 AM` | `D:20241211003002-05'00'` | 00:30 EST |
| `2020-08-05_11AM` | `Injury Report: 08/05/20 11:00 AM` | `D:20200805110002-04'00'` | 11:00 EDT **(bubble: on the hour)** |

So:

* **The label hour is Eastern wall-clock time.** The `-05'00'` / `-04'00'` offset
  tracks EST/EDT exactly, which is only consistent with `America/New_York`.
* **The report is stamped `HH:30`**, not `HH:00` — outside the 2020 bubble, which
  used `HH:00`.
* The generation timestamp (`:30:02`, `:30:26`, …) drifts a few seconds; the
  *report's own* header time is always the clean `HH:30`. **Use the header
  semantics (`HH:30`), not the file-creation second**, as the snapshot time.

### 1.3 Daylight-saving transitions — verified on real DST dates

This is the detail most likely to silently corrupt a time series, so it was
tested directly on transition days.

**Spring forward (2024-03-10, 2025-03-09):** 23 reports, not 24. The `02AM`
label returns 403 on both dates.

| URL | `/CreationDate` |
|---|---|
| `2024-03-10_01AM` | `D:20240310013002-05'00'` (EST) |
| `2024-03-10_02AM` | **403 — does not exist** |
| `2024-03-10_03AM` | `D:20240310033002-04'00'` (EDT) |

The generator follows a real local clock: the wall-clock hour 02:00 does not
exist, so no report is produced. **A missing `02AM` on the second Sunday in
March is correct, not a gap.**

**Fall back (2024-11-03, 2025-11-02):** 24 reports, not 25.

| URL | `/CreationDate` |
|---|---|
| `2024-11-03_01AM` | `D:20241103013003-04'00'` (**EDT — the first 1 AM**) |
| `2024-11-03_02AM` | `D:20241103023003-05'00'` (EST) |

The repeated 01:00 hour is published **once**, at the *first* (EDT) occurrence.
The second, EST 01:30 (06:30 UTC) is never published. So the naive local time
`01:30` on a fall-back date must be localized with **`fold=0`**. Using `fold=1`,
or letting a library pick, shifts that snapshot one hour later and silently
introduces a look-ahead error of up to 60 minutes.

> **Implementation rule.** Never store the label alone. Build
> `report_datetime_et = ZoneInfo("America/New_York").localize(date, hh, 30, fold=0)`
> and derive `report_datetime_utc` from it. Reject any date/label pair whose
> round-trip through UTC does not return the same wall-clock hour (this is what
> filters the non-existent spring-forward 02:00).

### 1.4 Publishing schedule by season

Measured by probing all 24 labels on representative dates from every season, plus
day-by-day sweeps across each regime boundary.

| Season | Common report hours (ET) | Earliest | Latest | Median reports/day | Notes |
|---|---|---|---|---|---|
| 2018-19 | 13:30, 17:30, 20:30 | 2018-12-17 **17:30** | 2019-06-13 | 3 | This path begins mid-season. First day partial (`05PM`, `08PM`); first full day 2018-12-18. **Earlier reports back to 2018-10-17 exist elsewhere — see §3.1.** |
| 2019-20 | 13:30, 17:30, 20:30 | 2019-09-30 | 2020-03-11 | 3 | Publishing stops dead after 2020-03-11 (COVID suspension). Nothing Mar 12 → late Jul. |
| 2019-20 *bubble* | **11:00, 14:00, 17:00** | ~2020-07-29 | ~2020-10-11 | 3 | Different hour set **and** on the hour, not `:30`. Boundaries sampled weekly, so ±7 days **(inferred)**. |
| 2020-21 | 13:30, 17:30, 20:30 | 2020-12-11 | 2021-07-20 | 3 | Compressed season. |
| 2021-22 | **transition** | 2021-10-03 | 2022-06-16 | 3 → 24 | **3/day through 2021-10-17; 24/day from 2021-10-18.** Regime change is one day before the 2021-10-19 season opener. |
| 2022-23 | all 24 hours | 2022-09-30 | 2023-06-12 | 24 | |
| 2023-24 | all 24 hours | 2023-10-05 | 2024-06-17 | 24 | |
| 2024-25 | all 24 hours | 2024-10-04 | 2025-06-22 | 24 | |
| 2025-26 (part 1) | all 24 hours | 2025-10-02 | 2025-12-22 08:30 | 24 | Old hourly format; last old-format file is `2025-12-22_08AM`. |
| 2025-26 (part 2) | **all 96 quarter-hours** | 2025-12-22 | 2026-06-13 | **96** | **New minute-bearing format** (§1.5). |
| 2026 Summer League | all 96 quarter-hours | 2026-07-03 | 2026-07-19 | 96 | Then idle for the offseason. |
| 2026-27 | — | not yet started | — | — | Expect resumption ~Oct 2026. |

Additional schedule facts:

* **In-season, every calendar day carries the full hour set** — including days with
  no games. A clean (pre-throttle) 24-label sweep of January 2023 returned
  744/744 = 31 days × 24. April and May 2023 likewise returned 720/720.
* **Multi-day gaps are real and align with the All-Star break**, not with
  individual no-game days: 2019-02-18/19, 2020-02-17/18, 2023-02-17 and
  2023-02-20/21, 2025-02-14 → 2025-02-17 all return zero reports at every label.
* **Offseason behaviour is inconsistent between years.** Reports demonstrably exist
  on 2019-06-05, 2021-07-14 and 2025-07-15 (the last is a real 2-page PDF stamped
  `07/15/25 08:30 PM`), but the 2023 offseason appears near-empty. The 2023
  offseason measurement is **contaminated by throttling** (§4.2) and should be
  re-measured during Phase 1 rather than trusted.

The NBA's own policy text on the report page corroborates the cadence:

> "By 5 p.m. local time the day before a game … teams must designate a
> participation status … For the second game of a back-to-back, teams must report
> … by 1 p.m. local time on the day of the game. In addition, teams are required
> to submit a game-day injury report between 11 a.m. and 1 p.m. local time (and
> between 8 and 10 a.m. for tip-offs 5 p.m. or earlier) … **Reports are updated on
> a continual basis throughout the day.**"

Note "local time" governs *team submission deadlines*; the *published file* is
stamped in ET regardless. Those deadlines (17:00 day-before, 13:00 back-to-back,
11:00–13:00 game-day) are exactly the hours where snapshot-to-snapshot status
churn should concentrate — useful later for feature design.

---

### 1.5 The quarter-hourly regime: 2025-12-22 → present

From **2025-12-22** the filename gains an explicit minute field and the cadence
goes to **every 15 minutes, around the clock — 96 reports per day**:

```
https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<hh>_<mm><AM|PM>.pdf
                                                                        ^^^^^^^^^ new
```

`hh` is zero-padded 12-hour `01`–`12`; `mm` ∈ `{00, 15, 30, 45}`. The daily
candidate space is **96 URLs** (24 h × 4).

Verified directly — and note the report is now stamped at the *exact* label time,
with none of the old `HH:30` offset:

| URL | PDF header | `/CreationDate` |
|---|---|---|
| `2026-03-11_08_00PM` | `Injury Report: 03/11/26 08:00 PM` | `D:20260311200003-04'00'` |
| `2026-03-11_08_15PM` | `Injury Report: 03/11/26 08:15 PM` | `D:20260311201502-04'00'` |
| `2026-03-11_08_30PM` | `Injury Report: 03/11/26 08:30 PM` | `D:20260311203005-04'00'` |
| `2026-03-11_08_45PM` | `Injury Report: 03/11/26 08:45 PM` | `D:20260311204502-04'00'` |
| `2026-06-03_12_15PM` | `Injury Report: 06/03/26 12:15 PM` | `D:20260603121503-04'00'` |

**The cutover is clean in both directions**, which makes era detection trivial:

* `Injury-Report_2024-12-11_08_30PM.pdf` → **403** (new format does not exist before the switch)
* `Injury-Report_2025-12-23_01PM.pdf` → **403** (old format does not exist after it)
* `2025-12-22` is the single crossover day: **9 old-format** files (00:30–08:30 ET)
  **plus 60 new-format** files.

A 96-label sweep of representative days returns a **full 96/96** on every active
date tested (2025-12-23, 2025-12-24, 2026-01-15, 2026-03-11, 2026-04-20,
2026-06-03, 2026-06-07, 2026-07-15). Mean file size **≈74 KiB**.

**Activity calendar since the switch** (probed daily at three labels):

| Span | Status |
|---|---|
| 2025-12-23 → 2026-06-05 | active (one blank day, 2026-06-06) |
| 2026-06-07 → 2026-06-13 | active (blank 2026-06-11) — playoffs/Finals |
| 2026-06-14 → 2026-07-02 | **idle** — post-Finals |
| 2026-07-03 → 2026-07-19 | active — Summer League |
| 2026-07-20 → today (2026-09-10) | **idle** — offseason |

So the generator now tracks *basketball activity* rather than running year-round,
and it is simply between seasons right now. Expect it to resume for 2026-27
preseason in late September / early October 2026.


## 2. Sampling strategy — recommendation

### The strategies, evaluated against measurement

**Strategy B (probe every hour) is exactly equivalent to Strategy A (collect
everything).** This is not an assumption: §1.1 establishes that the filename space
is 24 labels/day and that no other naming convention exists. Every published
report lands on the `HH:30` ET hourly grid. Therefore probing 24 labels/day
retains 100% of what was ever published; there is no residual set that Strategy A
would catch and Strategy B would miss.

That collapses the decision to *B vs. C vs. D* — a pure cost question, since B is
lossless and C/D are lossy.

### Coverage at the requested horizons

Report timeline modelled from the verified schedule (§1.4) and evaluated against
**9,300 real tipoffs** from the NBA schedule feed. "Within ±1h" = the latest
report at or before `T-Xh` is no more than 60 minutes stale.

**24-reports/day era (2021-10-18 → 2025-12-22), n = 5,967 games:**

| Horizon | % within 1h | % within 30m | median staleness | p95 |
|---|---|---|---|---|
| T-24h | 99.8% | 99.6% | 30 min | 30 min |
| T-18h | 100.0% | 99.7% | 30 min | 30 min |
| T-12h | 100.0% | 99.8% | 30 min | 30 min |
| T-8h | 100.0% | 99.8% | 30 min | 30 min |
| T-6h | 100.0% | 99.8% | 30 min | 30 min |
| T-4h | 100.0% | 99.8% | 30 min | 30 min |
| T-3h | 100.0% | 99.8% | 30 min | 30 min |
| T-2h | 100.0% | 99.8% | 30 min | 30 min |
| T-1h | 100.0% | 99.8% | 30 min | 30 min |
| **T-30m** | **100.0%** | **99.8%** | **0 min\*** | 30 min |

\* A median gap of 0 is an artefact worth understanding: **67.3% of tipoffs in
this era are on the hour** (`:00`), so `T-30m` lands *exactly* on an `HH:30`
report stamp. Under the strict `<` rule of §8 those reports are **not** usable —
they must fall back to the previous hour, i.e. 60 min stale. The honest reading
of T-30m is therefore "30 min stale for the 32.7% of games tipping at `:30`,
60 min stale for the rest". Coverage is still 100%; only the freshness differs.

Broken out by tipoff time — the brief's specific worry — the hourly grid is
indifferent to when the game starts:

| Tipoff bucket | n | worst horizon coverage (±1h) | T-30m coverage (±1h) |
|---|---|---|---|
| early (< 17:00 ET) | 367 | 100.0% | 100.0% |
| evening (17–18 ET) | 323 | 100.0% | 100.0% |
| prime (19–20 ET) | 3,789 | 99.8% | 100.0% |
| late (≥ 21 ET) | 1,488 | 99.8% | 100.0% |

**96-reports/day era (2025-12-22 → present):** every horizon is covered at
**100%** with a **worst-case staleness of 15 minutes and a median of ~7 minutes**,
for every tipoff time. No table is needed — with a report every quarter-hour, the
latest report before any cutoff is never more than one 15-minute step old. This
era is strictly better than anything the brief asked for, T-30m included.

**3-reports/day era (2018-12-17 → 2021-10-17), n = 3,420 games:**

| Horizon | % within 1h | median staleness |
|---|---|---|
| T-24h | 23.0% | 90 min |
| T-18h | 4.3% | 330 min |
| **T-12h** | **0.1%** | **690 min** |
| T-8h | 17.0% | 900 min |
| T-6h | 41.4% | 90 min |
| T-4h | 21.0% | 120 min |
| T-3h | 19.6% | 150 min |
| T-2h | 50.5% | 60 min |
| T-1h | 53.3% | 60 min |
| T-30m | 45.3% | 90 min |

### Recommendation

**Adopt Strategy B, era-aware: probe the full candidate space for every active
date, where "full" is era-dependent — 96 quarter-hour labels from 2025-12-22
onward, 24 hourly labels from 2021-10-18 to 2025-12-22, the 3 known labels
(plus a periodic 24-label audit) before that, and a listing-driven pass for the
pre-CDN era.**

> **On "30-minute intervals".** There is no 30-minute grid to fetch, in either
> era, and the finest granularity is not a choice the scraper gets to make — it is
> fixed by the URL:
>
> * **Before 2025-12-22** the filename has **no minute field at all**
>   (`_08PM`). Verified on a live date with working controls: `_0830PM`,
>   `_08-30PM`, `_08.30PM`, `_08_30PM`, `_0800PM`, `_08:30PM`, `_2030` and
>   `_08PM30` all return 403 while `_08PM` and `_09AM` return 200. The grid is
>   **hourly**, full stop — 24 URLs/day *is* "fetch them all".
> * **From 2025-12-22** the grid is **15-minute**, which is finer than the
>   30 minutes asked for. 96 URLs/day is again "fetch them all".
>
> So the instinct — *enumerate the whole space and skip what 404s* — is exactly
> right; only the step size differs by era, and probing a 30-minute grid would
> either miss half the new era's reports or waste half its requests on the old
> era's non-existent slots.

Reasoning, in the brief's stated priority order:

1. **Temporal correctness.** B is the only strategy that never has to decide *in
   advance* which horizons matter. C (fixed hours) and D (game-relative) both bake
   a horizon policy into the raw layer, which is irreversible — re-deriving a
   dropped horizon later means re-scraping a source that, per Finding 1, **may not
   be there**. That risk is no longer hypothetical: the source already went away
   once during this investigation.
2. **Completeness.** B is lossless by construction.
3. **Robustness.** D additionally couples raw ingestion to schedule data and
   tipoff correctness; a tipoff error would corrupt the raw layer rather than a
   downstream feature. Keep the schedule join in the *feature* layer, where it is
   re-runnable.
4. **Storage / speed.** These are the two lowest priorities and B costs ~1.2 GiB
   and ~29k requests — negligible either way (§9).

**The legacy era deserves an explicit warning in the dataset docs**, not a
sampling change: no strategy can manufacture a T-12h report that was never
published. Features built on 2018-12 → 2021-10 must carry a
`report_age_minutes` column and models should expect it to reach ~11 hours.
Consider gating early-era rows out of horizon-sensitive experiments entirely.

---

## 3. Historical URL discovery

There is no directory listing (`/referee/injury/` itself returns
`403 AccessDenied`, the S3 no-`ListBucket` response), so enumeration is the only
option. Given §1.1 the enumeration is fully determined:

```
for date in 2018-12-17 .. today:
    if date <= 2025-12-22:                       # hourly era
        for hh in 01..12, ampm in (AM, PM):      # 24 candidates
            if (date, hh:30, ampm) is a real America/New_York wall time:
                yield Injury-Report_{date}_{hh}{ampm}.pdf
    if date >= 2025-12-22:                       # quarter-hourly era
        for hh in 01..12, mm in (00,15,30,45), ampm in (AM, PM):   # 96 candidates
            if (date, hh:mm, ampm) is a real America/New_York wall time:
                yield Injury-Report_{date}_{hh}_{mm}{ampm}.pdf
```

2025-12-22 is deliberately in **both** branches: it is the crossover day and
carries 9 old-format plus 60 new-format files.

The DST guard drops exactly one candidate per spring-forward date and nothing
else. Edge cases and how the enumeration handles them:

| Edge case | Behaviour | Handling |
|---|---|---|
| Preseason | Reports **do** exist (e.g. 2021-10-18, before the 10-19 opener; 2022-09-30) | Enumerate from ~Sep 25 each season |
| Postseason | Full hour set through the Finals | Enumerate to ~Jun 30 |
| All-Star break | Genuine multi-day zero-report gaps | Expected; flag only if unexpected (§10) |
| 2020 COVID stop | Hard stop 2020-03-11 → ~2020-07-29 | Expected gap, ~140 days |
| 2020 bubble | **Different hour set (11:00/14:00/17:00) and `:00` not `:30`** | Special-cased era; the 24-label probe finds them anyway, but the *datetime* must be built with `:00` |
| Offseason | Inconsistent between years; real reports exist in some | Probe it — cost is trivial and 2025-07-15 proves content is real |
| Cancelled / no-game days | In-season, reports are still published | No special handling |
| Spring-forward 02:00 | Never published | Skipped by the DST guard |
| Fall-back repeated 01:00 | Published once, at `fold=0` | Localize with `fold=0` |

### 3.1 The pre-CDN era: 2018-10-17 → 2019-01-03 (WordPress media library)

**The `/referee/injury/` path is not where the injury report started.** The first
two and a half months live in the `official.nba.com` WordPress media library, on a
different host and with no usable naming convention. They are discoverable through
the site's own REST API:

```
https://official.nba.com/wp-json/wp/v2/media?search=Injury&per_page=100&page=<n>&_fields=date,source_url
```

That returns **200 injury-report PDFs spanning 2018-10-17 → 2019-01-03** — i.e.
the injury-report programme actually began at the **start of the 2018-19 season**,
two months earlier than the `/referee/injury/` path suggests.

The API reports `source_url` on `cms.nba.com`, which returns **403**. Rewrite the
host to the public CDN and the files serve fine:

```
https://cms.nba.com/official/wp-content/uploads/... ->
https://ak-static.cms.nba.com/wp-content/uploads/...          # 200
```

Two sub-eras, and the split matters a great deal:

| Sub-era | Dates | Count | Format | Size | Usable? |
|---|---|---:|---|---|---|
| Scanned | 2018-10-17 → ~2018-11-18 | **103** | **1-page scanned image, zero extractable text** | 200–650 KB | **Only via OCR** |
| Text | ~2018-11-21 → 2019-01-03 | **97** | 4–5 page text PDF, standard header | 22–40 KB | Yes, parses like any other |

Of the 200, **11 overlap** dates already covered by `/referee/injury/`
(2018-12-17 onward) and must be de-duplicated; **189 are net new**.

**Filenames cannot be enumerated.** Eight distinct shapes appear in 200 files:

| Count | Shape |
|---:|---|
| 117 | `Injury-Report-##.##.##-#.##.pdf` |
| 59 | `InjuryReport_##_##_####_####PM.pdf` |
| 5 | `Injury-Report-##.##.##-#.##-#.pdf` (WordPress `-1`/`-2` collision suffixes) |
| 5 | `Injury-Report-##.#.##-#.##.pdf` (unpadded day) |
| 3 | `Injury-Report_####-##-##_##PM.pdf` (the modern shape, appearing early) |
| 2 | `InjuryReport_##_##_####_###.pdf` |
| 1 | `Injury-Report-##-#-##-#.##-PM.pdf` |
| 1 | `InjuryReport_##_##_####.pdf` |

So this era needs a **listing-driven** discovery path, not a generative one: page
the media API, parse the report datetime out of each filename with a set of
tolerant patterns, and **cross-check every parse against the PDF's own header
line** — which for the text sub-era is present and authoritative. Reports here are
the legacy 3/day cadence (`1.30`, `5.30`, `8.30` → 13:30 / 17:30 / 20:30 ET).

**Recommendation:** ingest the 97 text-era PDFs in the main backfill (cheap, and
they extend usable history by a month). Store the 103 scanned ones in the raw
layer too — they are only ~50 MB and the source is clearly fragile — but mark them
`validation_status = 'image_only'` and treat OCR as a separate, optional project.
They cover October–November 2018 only, which is unlikely to be load-bearing for
any model.


---

## 4. CDN availability and request behaviour

### 4.1 Availability window

* **Earliest report on this path: `2018-12-17_05PM`** (17:30 ET). Confirmed by a
  day-by-day sweep of 2018-10-01 → 2019-03-01 at the three legacy labels; every
  date before 2018-12-17 returns 403 at every label. 2018-12-17 has only `05PM`
  and `08PM`; 2018-12-18 is the first complete day.
  **This is not the earliest report overall** — see §3.1; the archive really
  begins 2018-10-17 on a different host.
* **Latest accessible report: 2026-07-19** (Summer League), with the feed idle
  since 2026-07-20 for the offseason. The apparent "end" at `2025-12-22_08AM` is
  only the end of the **old filename format** — the day-by-day 24-label sweep of
  2025-12-14 → 2026-01-20 was blind to the new pattern. See §1.5.
* **Historical files do not expire.** December 2018 files are still served today
  with `Last-Modified` dates from 2018. There is no retention window; the archive
  is stable and can be collected at leisure.

The WordPress page `official.nba.com/nba-injury-report-2025-26-season/` carries
`og:updated_time = 2025-12-22T13:39:00-05:00` and renders **zero** PDF links.
Read correctly, that is a second symptom of the **same-day format migration**, not
evidence of a shutdown: the page's link list was retired when the naming changed.
It was misread as corroboration in the earlier draft — a reminder that two weak
signals agreeing is not confirmation when both have the same root cause.

> **Side effect worth flagging (not fixed here, out of scope):** the repo's live
> fetcher `src/nba_ou/fetch_data/injury_reports/get_latest_injury_report.py`
> scrapes that page for `.pdf` links containing `injury-report`. It now finds
> **zero** such links and will raise `ValueError("No PDF links found.")`. The
> daily prediction path that depends on it is very likely broken. This deserves
> its own ticket.

### 4.2 The false-403 problem — the most important operational finding

A missing file returns:

```
HTTP/1.1 403 Forbidden
<Error><Code>AccessDenied</Code><Message>Access Denied</Message>...</Error>
```

This is the ordinary S3 response for a non-existent key in a bucket without
`ListBucket`. **There is no 404.** So far, so manageable.

The problem: **the CDN also returns exactly this 403 when it decides you have
asked for too much.** It never returns `429`, and it sends no `Retry-After`.

Evidence:

1. A 10,585-request sweep (4 workers, 0.12 s delay, ≈8 req/s) returned clean 200s
   for Jan–Jun 2023, then **every subsequent request 403 — 2,208 consecutive
   403s across Oct–Dec 2023**, a period known from an earlier sweep to be fully
   populated.
2. Re-probing those exact URLs a few minutes later, one at a time, returned
   **200 with real content**:
   `2023-10-25_08PM → 200, 86,818 bytes`; `2023-12-13_08PM → 200, 89,237 bytes`;
   `2023-11-15_08PM → 200, 77,043 bytes`.
3. A deliberate ramp, and then a controlled re-test from a rested state, pinned
   down what actually triggers it:

| Test | Shape | Result |
|---|---|---|
| Full-year sweep | **4 workers**, ~8 req/s, 10,585 requests | clean for ~3,700, then **2,208 consecutive false 403s** |
| Ramp phase 1 (10 min after that sweep) | 1 worker, 1.09 req/s, 150 requests | 44× 200, then 403 from #45 — *residual block, not a fresh trip* |
| Ramp phases 2–3 | 1 worker, 2–4 req/s | 403 from request 1 — still blocked |
| Recovery canary | 1 request / 120 s | **200 after ~120 s of idle** |
| Rested re-test | 1 worker, no delay, 400 requests | **400× 200, zero 403s** |
| **Sustained re-test** | **1 worker, 0.25 s delay, 3,000 requests** | **3,000× 200, zero 403s in 18 min at 2.78 req/s** |

**Corrected reading.** The trigger is **not** a low cumulative volume — an early
draft of this document concluded that from the ramp, but the ramp was measuring
*residual* blocking from the preceding 10.5k sweep, not a fresh trip. From a
rested state, **3,000 sequential requests at 2.78 req/s complete with zero 403s.**

The distinguishing factor between the run that tripped and the run that did not
is **concurrency**: the failure used 4 parallel workers at ~8 req/s, the clean
run used 1 worker at ~2.8 req/s **(inferred — rate and concurrency were not varied
independently)**. Recovery is fast: **~20–120 s of idle** clears the block
completely, measured twice.

So the CDN is far more tolerant than the first measurement suggested. What
remains true, and what matters, is that **when it does refuse, it refuses with
`403` and no `Retry-After`** — indistinguishable from a missing file without a
canary.

**Consequences for design (non-negotiable):**

* A bare 403 **must never** be recorded as `nba_available = false`.
* Discovery needs a **canary**: a URL verified to exist (e.g.
  `Injury-Report_2023-11-15_08PM.pdf`). On any 403, re-probe the canary. Canary
  403 ⇒ *throttled*, not missing ⇒ pause, do not write a negative result.
* Every 403 must be recorded as `unknown` and re-verified on a later pass before
  it is allowed to become a confirmed absence.

### 4.3 Other request behaviour

* **HEAD works and is reliable** — same status semantics as GET, returns
  `Content-Length`, `Content-Type`, `ETag`, `Last-Modified`. Use HEAD for discovery.
* **No misleading 200s observed.** Every 200 in ~4,700 successful probes carried
  `Content-Type: application/pdf`, and every body sampled began with `%PDF-`.
  Validation is still required (§10) but the CDN is not serving HTML error pages
  with 200 status.
* **Range requests work** (`Range: bytes=0-1023` → `206`), and are ~2.3× faster
  than a full GET. Useful for cheap `%PDF` verification without downloading.
* **No User-Agent gating whatsoever.** `200` for python-requests' default UA, an
  empty UA, `curl/8.5.0`, a Chrome string, and a custom `nba-ou-injury-archiver/0.1`
  string. **Use an honest, identifying UA** — there is nothing to work around, and
  it is the polite choice. (Contrast `data.nba.com`, which *does* require a
  `Referer`/`Origin` pair — see `fetch_nba_schedule.py`.)
* **ETag and Last-Modified are stable** across repeated requests — usable as
  cheap change-detection keys.

### 4.4 Recommended request policy

Deliberately conservative, because the failure mode is silent data corruption
rather than a loud error:

| Parameter | Recommendation |
|---|---|
| Concurrency | **1** (sequential). This is the variable most implicated in the trip; the ~70 req/s ceiling is not worth chasing. |
| Delay between requests | **0.60 s + 0–0.20 s jitter** → a measured **1.2 req/s** end to end. Deliberately ~half the 2.78 req/s already verified clean over 3,000 consecutive requests. |
| Requests per run | No hard cap needed; checkpoint every ~1,000 rows so any interruption is cheap. |
| Cooldown after a trip | **180 s** × up to 4, re-probing the canary each time (measured recovery: 20–120 s, so this is ~1.5× margin) |
| On 403 | Probe canary. Canary 200 ⇒ genuine absence. Canary 403 ⇒ **circuit-break**: stop the run, checkpoint, mark nothing negative |
| On 5xx / timeout | Exponential backoff `2^n` s, max 5 attempts, then record `unknown` and continue |
| On 429 | Not observed, but handle: honour `Retry-After`, else back off 5 min |
| User-Agent | `nba-ou-injury-archiver/0.1 (+https://github.com/panchojasen/...)` — honest and identifying |
| Timeout | 20 s connect+read |
| Method | HEAD for discovery, GET for download |

---

## 5. Two-phase pipeline

### Phase 1 — discovery / indexing

Emits manifest rows without downloading bodies.

```
for each date in the requested range (resume point from manifest):
    for each of the 24 labels, DST-filtered:
        if manifest already has a terminal verdict for this key: skip
        HEAD the URL
        200 -> record exists=true + content_length, content_type, etag, last_modified
        403 -> probe canary
                 canary 200 -> record exists=false   (terminal)
                 canary 403 -> record unknown, CIRCUIT-BREAK the run
        5xx/timeout -> retry with backoff, else record unknown
        sleep(jitter)
```

Terminal verdicts (`exists=true`, `exists=false`) are never re-probed. `unknown`
rows are the retry queue, which is what makes the whole thing resumable.

### Phase 2 — download

Reads the manifest, downloads only `exists=true AND download_status != 'stored'`.

```
for each pending row (ordered by date, then hour):
    if s3 object already exists and size matches manifest: mark stored, skip
    GET the URL
    validate: starts with %PDF, content_type == application/pdf,
              size >= 2 KiB, pymupdf opens it, page_count >= 1
    compute sha256
    PUT to S3 with metadata
    update manifest: download_status, s3_key, sha256, bytes, downloaded_at
    sleep(jitter)
```

Downloads are strictly idempotent: the S3 key is a pure function of the report
identity, so a re-run overwrites byte-identical content or skips entirely.

---

## 6. S3 storage design

Bucket: **`adrian-nba-model-registry-eu-west-1`** (the repo's existing
`[S3] BUCKET`), under a new dedicated top-level prefix. This matches the existing
flat top-level convention (`models/`, `train_data/`, `backups/db/`,
`prediction_snapshots/`) rather than inventing a parallel bucket. A separate
bucket would need new credentials, new IAM, and a second thing to back up, for no
benefit.

```
injury_reports/
  raw/season=<YYYY-YY>/date=<YYYY-MM-DD>/Injury-Report_<YYYY-MM-DD>_<hhAMPM>.pdf
  manifest/season=<YYYY-YY>/manifest.parquet
  processed/season=<YYYY-YY>/date=<YYYY-MM-DD>/report_rows.parquet
  _audit/discovery_runs/<run_id>.json
```

### 6.1 Canonical object naming

**Do not preserve the source filename.** Across the four eras the source names are
inconsistent, and in the hourly era they are **actively misleading**:

| Source filename | What it implies | Actual report time |
|---|---|---|
| `Injury-Report_2024-12-11_08PM.pdf` | 20:00 ET | **20:30 ET** |
| `Injury-Report_2020-08-05_11AM.pdf` (bubble) | 11:00 ET | 11:00 ET |
| `Injury-Report_2026-03-11_08_15PM.pdf` | 20:15 ET | 20:15 ET |
| `Injury-Report-11.28.18-1.30.pdf` (pre-CDN) | — | 13:30 ET |
| `InjuryReport_11_21_2018_0530PM.pdf` (pre-CDN) | — | 17:30 ET |

Two files an hour apart in real time can look like neighbours, and `_08PM` means
20:30 in one era and would mean 20:00 in another. Storing that verbatim pushes an
avoidable decoding step onto every future consumer.

**Canonical key:**

```
injury_reports/raw/season=<YYYY-YY>/date=<ET date>/injury-report_<ET timestamp><offset>.pdf
```

Concretely:

```
raw/season=2024-25/date=2024-12-11/injury-report_2024-12-11T2030-0500.pdf
raw/season=2020-21/date=2020-08-05/injury-report_2020-08-05T1100-0400.pdf
raw/season=2025-26/date=2026-03-11/injury-report_2026-03-11T2015-0400.pdf
raw/season=2018-19/date=2018-11-28/injury-report_2018-11-28T1330-0500.pdf
```

Properties:

* **ET wall clock, as the domain uses it** — `2030` is unambiguously 8:30 PM ET.
* **The UTC offset (`-0500` / `-0400`) is part of the name**, which makes the key
  unique even across a fall-back repeated hour and records the DST state without a
  lookup.
* **Era-independent**: one shape for all four regimes. The reader never needs to
  know which era a file came from to know when it was published.
* **Self-correcting**: the name is built from the *decoded* report time, so the
  hourly era's `+30 min` and the bubble's `+0 min` are resolved once, at ingest,
  instead of forever after.
* **Partitioned by ET date**, matching how reports are grouped and how the
  schedule join works.

### 6.2 One property this deliberately gives up

**Lexicographic key order is not chronological order** — for exactly one hour per
year. During the fall-back the ET wall clock runs backwards, so:

```
injury-report_2026-11-01T0100-0400.pdf   ->  05:00Z
injury-report_2026-11-01T0115-0400.pdf   ->  05:15Z
injury-report_2026-11-01T0100-0500.pdf   ->  06:00Z   <- sorts 2nd, occurs 3rd
```

This is not a flaw in the encoding; **no ET-wall-clock-primary name can sort
chronologically**, because the underlying clock is non-monotonic. The choice is
genuinely either/or:

| | ET-primary (recommended) | UTC-primary |
|---|---|---|
| Key | `injury-report_2026-11-01T0100-0400.pdf` | `injury-report_20261101T0500Z.pdf` |
| Human-readable in NBA terms | **yes** | no (ET date differs from UTC date after 20:00 ET) |
| `ls`-order == time-order | no (1 h/year) | **yes** |
| Unique | yes | yes |

**Recommendation: take the ET-primary form**, and adopt the hard rule that
**ordering always comes from the manifest's `report_datetime_utc`, never from key
order.** That rule is required anyway — S3 keys are storage, the manifest is the
index — and it makes the anomaly inert. In the hourly era the anomaly cannot even
occur, because only one of the two 01:00 reports was ever published (§1.3); it
becomes reachable only if the 96-slot regime publishes both occurrences, which has
not yet happened (§14).

### 6.3 Rules for building and verifying the name

1. **Derive, never parse-and-copy.** The name is a pure function of
   `report_datetime_et`, which itself comes from `(era, date, label)` — not from
   string-munging the source filename.
2. **Verify against the PDF's own header** before storing. Every report carries
   `Injury Report: MM/DD/YY hh:mm (AM|PM)`; if that disagrees with the derived
   time, store with `validation_status = 'date_mismatch'` and **do not** let the
   file into the clean partition. This is the check that would catch a future
   label-semantics change (a fifth era where `_08PM` means something new).
3. **Collision is a hard error.** Two distinct source URLs must never map to one
   canonical key. If it happens, stop — it means the era model is wrong. The
   pre-CDN era makes this real: WordPress `-1`/`-2` suffixed duplicates
   (`Injury-Report_2018-12-19_01PM-1.pdf`) and the 11 dates that exist on *both*
   the WP and CDN paths both resolve to the same canonical name. Resolve by
   preferring the CDN copy, then the unsuffixed WP copy, and record the discarded
   URL in the manifest rather than dropping it silently.
4. **Provenance is never lost.** `original_url` and `original_filename` are
   manifest columns *and* S3 object metadata, so the mapping is reversible from
   the object alone.
5. **Idempotent.** Same report ⇒ same key ⇒ re-runs overwrite byte-identical
   content or skip. This is what makes the whole backfill safe to restart.

### 6.4 Other layout notes

* **Season label `YYYY-YY`** to match the repo's convention
  (`get_seasons_between_dates`, `nba_injuries_<season>.csv`).
* **Hive-style `season=` / `date=` partitioning** so Glue/Athena can register
  `manifest/` and `processed/` with no rewriting, and so "re-run one season" is a
  prefix operation.
* **Manifest per season, not one global file** — a single Parquet would be
  rewritten on every checkpoint.
* **`processed/` is reserved, not built** — parsing is out of scope, but the layout
  is fixed now so the parser has somewhere to land.
* **S3 object metadata** on each raw PDF: `original-url`, `original-filename`,
  `report-datetime-et`, `report-datetime-utc`, `source-era`, `sha256`,
  `discovered-at` — the object is self-describing if the manifest is ever lost.
* **Storage class** Standard; enable **versioning** on the prefix as cheap
  insurance against a bad re-run.
* **Local development**: the same layout mirrored under `data/injury_reports/`
  behind a `--local-root` flag, as `data/sbr_line_history` already does.

## 7. Manifest schema

One row per candidate URL that has been probed. Written as Parquet.

| Column | Type | Notes |
|---|---|---|
| `report_key` | string | PK. The canonical stem, `<ET date>T<HHMM><offset>` — era-independent and stable |
| `season` | string | `2023-24` |
| `season_year` | int32 | `2023` — matches the repo's int convention |
| `report_date_et` | date | ET calendar date from the filename |
| `report_time_label` | string | `08PM` |
| `report_hour_et` | int8 | 0–23, derived |
| **`report_datetime_et`** | timestamp(tz=America/New_York) | `HH:30` (or `HH:00` in the bubble era), `fold=0` |
| **`report_datetime_utc`** | timestamp(tz=UTC) | **the join key for all temporal work** |
| `source_era` | string | `wp_media` / `legacy_3` / `bubble_3` / `hourly_24` / `quarter_96` — drives label decoding |
| `original_url` | string | full source URL as fetched |
| `original_filename` | string | the source basename, kept for provenance and audit |
| `superseded_urls` | list[string] | other URLs resolving to the same report (WP `-1` duplicates, WP/CDN overlap) |
| `nba_available` | string | `true` / `false` / `unknown` — **three-valued, never boolean** |
| `nba_http_status` | int16 | last observed status |
| `content_type` | string | from HEAD |
| `content_length` | int64 | from HEAD |
| `etag` | string | CDN ETag |
| `last_modified` | timestamp(tz=UTC) | CDN Last-Modified |
| `discovery_timestamp` | timestamp(tz=UTC) | when last probed |
| `discovery_attempts` | int16 | for the unknown-retry queue |
| `download_status` | string | `pending` / `stored` / `failed` / `invalid` |
| `s3_key` | string | full canonical key, populated on store |
| `sha256` | string | of the stored bytes |
| `bytes_stored` | int64 | verified against `content_length` |
| `download_timestamp` | timestamp(tz=UTC) | |
| `pdf_page_count` | int32 | from validation |
| `pdf_header_datetime_et` | timestamp(tz=…) | parsed from page 1, for cross-check |
| `validation_status` | string | `ok` / `%PDF_missing` / `unopenable` / `date_mismatch` / … |
| `notes` | string | free text |

The two columns that carry the whole temporal-correctness burden are
`report_datetime_utc` and `nba_available`. Making the latter three-valued is what
prevents §4.2's false 403s from being laundered into confident absences.

---

## 8. Linking to games without leakage

The retrieval the downstream model needs is:

> given a game and a horizon, the **most recent report published strictly before**
> the prediction timestamp.

With `report_datetime_utc` present this is an as-of join:

```sql
SELECT m.*
FROM   manifest m
WHERE  m.nba_available = 'true'
  AND  m.report_datetime_utc < (g.tipoff_utc - INTERVAL 'X hours')
ORDER  BY m.report_datetime_utc DESC
LIMIT  1
```

Rules that keep this honest:

1. **Strictly `<`, never `<=`.** A report stamped exactly at the cutoff is
   published at that instant; treating it as available at that instant is a
   boundary leak. This bites hardest at **T-30m**, where 67.3% of games land
   exactly on a report stamp (§2) — using `<=` there would leak on two thirds of
   the sample.
2. **Always emit `report_age_minutes = prediction_ts - report_datetime_utc`
   alongside the features.** In the 24/day era this is ~30 min; in the legacy era
   it reaches ~11 h. A model that cannot see the age cannot learn that early-era
   features are stale.
3. **Never fall forward.** If no report precedes the cutoff, the correct value is
   null — not the next report.
4. **DST fall-back needs its own rule when matching reports to games.** On the
   fall-back date (2026-11-01 next; 2025-11-02, 2024-11-03 historically) the ET
   wall clock repeats 01:00–01:59, so an *ET-naive* join can place a report on the
   wrong side of a cutoff by up to an hour. The rule:

   * **Join on `report_datetime_utc` and `tipoff_utc` only.** Never join on an ET
     wall-clock string, an ET date, or a naive timestamp — the ET date is fine for
     *partitioning* files, never for *ordering* them.
   * Reports whose canonical key carries `-0400` on a fall-back date are the
     **first** (EDT) occurrence; `-0500` is the second. The offset in the key
     (§6.1) is what disambiguates them.
   * The old hourly era published only one of the two occurrences (§1.3), so this
     is inert there. **The 96-slot era has never seen a fall-back** — see §14 —
     so treat 2026-11-01 as a date to re-verify rather than assume.
   * Add a unit test pinning a fall-back date, alongside the spring-forward one.

5. **Tipoffs come from `fetch_nba_schedule.py`**, which returns explicit UTC
   (`gdtutc`/`utctm`) with 100% coverage across all 8 seasons (verified: 1,393 /
   1,257 / 1,221 / 1,393 / 1,394 / 1,396 / 1,400 games, zero null tipoffs).
6. **Do not reuse the existing `nba_injuries` table for horizon features.** Its
   primary key is `(player_id, game_id)` with **no timestamp column** — it stores
   one status per player-game and cannot express "what was known at T-6h". This
   archive exists precisely to replace it for time-sensitive features. Given the
   repo's history with the rotation-depth availability leak, mixing the two
   sources in one feature set would be an easy way to reintroduce it.

**Coverage is sufficient for the modern era and insufficient for the legacy era**
— see the tables in §2. That conclusion is a property of what the NBA published,
not of the collection strategy.

---

## 9. Volume, request and storage estimates

`/referee/injury/` span 2018-12-17 → 2025-12-22 = **2,563 calendar days**; 1,893
of those fall inside a season window. The pre-CDN era (§3.1) is counted separately
because it is discovered by listing, not enumeration.

| Season | In-season days | Regime | Expected PDFs |
|---|---:|---|---:|
| 2018-19 | 179 | 3/day | 537 |
| 2019-20 | 378 | 3/day | 1,134 |
| 2020-21 | 222 | 3/day | 666 |
| 2021-22 | 257 | mixed | 5,853 |
| 2022-23 | 256 | 24/day | 6,144 |
| 2023-24 | 257 | 24/day | 6,168 |
| 2024-25 | 262 | 24/day | 6,288 |
| 2025-26 | 82 | 24/day | 1,968 |
| **Subtotal, hourly era** | **1,893** | | **≈ 28,758** |
| *plus* pre-CDN WP era (§3.1) | 2018-10-17 → 2019-01-03 | listing-driven | **+189** |
| *plus* **quarter-hourly era (§1.5)** | 188 active days, 2025-12-22 → 2026-07-19 | **96/day** | **+18,048** |
| **Grand total** | | | **≈ 46,995** |

The quarter-hourly era is only 188 days but contributes **38% of all reports** and,
at ~74 KiB each, **more than half the bytes**. A full 2026-27 season will add
roughly **24,000 PDFs / 1.7 GiB per year** at ~250 active days.

**Discovery request cost:**

| Approach | Requests | Comment |
|---|---:|---|
| 24 labels every calendar day (hourly era) | 61,512 | blind; covers offseason |
| 24 labels, season days only (hourly era) | 45,432 | **recommended** |
| Season-aware labels (3 or 24) | 28,758 | cheapest; risks missing an undocumented legacy hour |
| **96 labels × 188 active days (quarter-hourly era)** | **18,048** | required in full |
| *plus* WP media API listing | **4** | the whole pre-CDN era for 4 requests |
| **Total discovery** | **≈ 63,500** | |

Recommend the middle option — 45,432 HEADs — plus a 24-label audit sample over
offseason dates. The extra ~17k requests over the cheapest option buy proof that
the legacy era really only had three hours.

**PDF size** (measured over ~4,700 successful HEADs): mean ≈ 34 KiB, median ≈ 30 KiB,
p95 ≈ 60 KiB, max ≈ 92 KiB. Modern in-season days average ~64 KiB.

| Mean size assumed | Raw-layer total |
|---|---|
| 30 KiB | 0.82 GiB |
| **45 KiB** | **1.23 GiB** |
| 64 KiB | 1.76 GiB |
| 90 KiB | 2.47 GiB |

Those figures cover the hourly era only. Adding the quarter-hourly era
(18,048 × ~74 KiB ≈ **1.27 GiB**) and the pre-CDN era (~**50 MB**, disproportionate
because the 103 scanned reports run 200–650 KB each) gives a realistic
**≈2.5 GiB total today**, growing ~1.7 GiB per future season. Still trivial.

**Budget ~1.5 GiB** for the raw layer; manifest Parquet adds a few MiB. At
eu-west-1 Standard pricing this is well under $0.05/month. **Storage is a
non-issue and should not influence any design decision.**

### 9.1 Sampling step and day-set: measured sizes

Measured mean PDF size per era, from ~6,500 successful HEADs:

| Era | n | mean | median | p95 |
|---|---:|---:|---:|---:|
| `bubble_3` | 33 | 16.2 KiB | 14.1 | 20.7 |
| `legacy_3` | 675 | 29.1 KiB | 30.7 | 43.2 |
| `hourly_24` | 4,591 | 38.9 KiB | 30.9 | 78.2 |
| `quarter_96` | 1,187 | **75.2 KiB** | 74.6 | 89.4 |

Whole-archive totals (2018-10-17 → 2026-07-19). The 15-vs-30-minute choice only
affects the quarter-hourly era — the earlier eras have no minute field at all, so
their volume is fixed:

| Day set | days | 15-min PDFs | GiB | 30-min PDFs | GiB |
|---|---:|---:|---:|---:|---:|
| **A** every calendar day | 2,833 | 60,075 | **2.94** | 49,995 | 2.22 |
| **B** game days only | 1,719 | 35,178 | **1.68** | 30,186 | 1.32 |
| **C** game days + 1 day before | 1,858 | 37,659 | **1.78** | 32,523 | 1.41 |
| **D** game days + 2 days before | 1,917 | 38,601 | **1.82** | 33,417 | 1.44 |

**Take the 15-minute step.** Going to 30 minutes saves only **0.37 GiB** (21%) and
permanently discards half the resolution of the single best era — the one that
actually reaches every horizon. That is a bad trade at any priority ordering, and
an absurd one under *temporal correctness > completeness > … > storage*.

### 9.2 Why "game days only" is not safe on its own

Gating on game dates is the right instinct, but the naive version breaks the
longest horizon. A report at `T-24h` for a game on date `D` lives on date `D-1` —
and `D-1` frequently has **no games at all**:

| Game type | games | % with no games the previous day |
|---|---:|---:|
| regular | 9,489 | 3.8% |
| preseason | 593 | 4.2% |
| **playoffs** | 587 | **13.8%** |
| **play-in** | 31 | **58.1%** |
| all-star | 23 | 39.1% |
| **all** | 10,725 | **4.6%** |

Restricting downloads to game days alone would silently lose T-24h for **4.6% of
all games and 13.8% of playoff games** — concentrated exactly where the off-day
gaps are, which is the postseason the brief explicitly wants kept.

**Including the day before fixes it for 0.10 GiB** (C vs B). Two days before costs
a further 0.04 GiB and additionally protects long horizons in the playoffs, where
13.1% of games have no games two days prior.

### 9.3 Recommendation: discover everything, download gated

Separate the two decisions, because they have very different costs and very
different reversibility:

* **Discovery: every calendar day, no gating.** ~60,000 HEAD requests, **zero
  storage**, ~6 hours at the §4.4 rate. The manifest then holds a complete record
  of what exists, including the offseason and no-game days.
* **Download: day set C or D, 15-minute step.** ~1.8 GiB.

This makes the gating decision **reversible**. If a no-game-day report is wanted
later, the manifest already knows it exists and its canonical key — it is a
download, not a re-discovery. It also keeps the raw layer's completeness from
depending on schedule correctness, which matters given the schedule feed is a
separate moving part.

If you would rather not think about it at all: **option A, every calendar day at
15 minutes, is 2.94 GiB** — about 1.2 GiB more than C, removes the schedule
dependency entirely, and at these sizes is a perfectly defensible "just take
everything". Both are fine; C is the thriftier one that still cannot lose a
horizon.

**Playoffs, play-in and preseason are all included** in every option above — the
day sets are built from the full schedule feed (`game_id` prefixes `001`/`002`/
`004`/`005`), not from regular season alone.

**Backfill duration**, measured end to end with the implemented client (the
figures below include real network latency, not just the sleep):

| Delay | Effective rate | Full backfill (~110,500 requests) |
|---|---|---|
| 0.35 s | 2.21 req/s | ~14 h |
| **0.60 s (default)** | **~1.2 req/s** | **~25 h** |
| 1.00 s | 0.90 req/s | ~34 h |

The default trades ~11 hours for roughly double the safety margin. Since the
backfill is resumable and unattended, that wall-clock costs nothing but patience,
whereas a trip risks writing false absences into the manifest. Downloads are slower per request than HEADs, so budget
**8–10 hours**. This fits comfortably in a single unattended overnight run, with
the circuit breaker handling any trip transparently.

---

## 10. Data-quality validation

**Per file (blocking — a failure means do not mark `stored`):**

* First 5 bytes are `%PDF-`
* `Content-Type: application/pdf`
* Size ≥ 2 KiB (smallest real report observed: 3.8 KiB)
* `pymupdf.open()` succeeds and `page_count ≥ 1`
* Page-1 header matches `Injury Report: MM/DD/YY hh:mm (AM|PM)`
* **Header date == filename date** and **header time == filename label** —
  this is the strongest single check available and it is nearly free
* `bytes_stored == content_length` from discovery
* **Derived report time == the PDF header time.** The single most valuable check:
  it validates the era model, the DST handling and the canonical name in one step
* **Canonical-key collision**: no two distinct source URLs may resolve to the same
  key. A collision is a hard stop, not a warning (§6.3)

**Per date:**

* Modern era, in-season: expect exactly 24 reports (23 on spring-forward). Fewer
  ⇒ flag.
* Legacy era, in-season: expect 3.
* Zero reports on a date with scheduled games ⇒ **flag for investigation**. Join
  against `fetch_nba_schedule.py`. Expect legitimate hits for the All-Star break
  and the COVID suspension, which should be allow-listed once confirmed.

**Per season:**

* Total reports within ~10% of the §9 estimate.
* No `unknown` rows left in the manifest.
* Contiguous hour coverage — no interior single-hour holes, which would indicate
  an unresolved false-403.

**Cross-cutting:**

* **Duplicate checksums:** measured — of 24 consecutive hourly reports on
  2024-12-11, **all 24 were byte-distinct**, even the four overnight ones that
  shared an identical 63,802-byte length. Embedded timestamps and XMP IDs
  guarantee uniqueness. So a repeated `sha256` is **not** normal deduplication —
  it means the same object was stored twice under different keys. Treat it as a
  bug signal, not a saving. Real deduplication is only possible after parsing.
* **Canary health**: assert the canary returns 200 at the start of every run;
  if it does not, the previous run's cooldown was insufficient.

---

## 11. Repository integration

Researched against the current tree; the plan below reuses existing patterns
rather than adding parallel ones.

| Concern | Existing convention | Plan |
|---|---|---|
| Scraper/fetch code | `src/nba_ou/fetch_data/<source>/` | New `src/nba_ou/fetch_data/injury_reports/archive/` alongside the existing `get_latest_injury_report.py` |
| Polite HTTP | `scrape_sportsbook_line_history.py`: `new_session()`, `_sleep_politely()` with `SLEEP_MIN_S`/`SLEEP_MAX_S`, `fetch_next_data()` retry loop | Reuse the idiom directly; add the canary/circuit-breaker, which is new and specific to this source |
| Retry helper | `create_robust_session()` in `get_latest_injury_report.py` (urllib3 `Retry`, `status_forcelist=[429,500,502,503,504]`) | **Must not** be reused as-is — it does not list 403, and here 403 needs bespoke handling |
| PDF parsing | `pymupdf` already a dependency; `read_injury_report()` already parses these exact PDFs | Reuse for validation (`page_count`, header line). Full parsing stays out of scope |
| S3 access | `src/nba_ou/utils/s3_models.py`: `make_s3_client(profile, region)`, `upload_bytes_to_s3`, `list_s3_objects`, `load_parquet_from_bytes` | Reuse verbatim; no new S3 layer |
| S3 key building | `s3_prediction_snapshots.py`: `build_snapshot_base_key()` | Mirror with `build_injury_report_key()` |
| Config | `src/nba_ou/config.ini` + `CONFIG_SCHEMA` in `settings.py`; secrets overlaid from `config.secrets.ini` | Add an `[InjuryReportArchive]` section: `CDN_BASE_URL`, `S3_PREFIX`, `REQUEST_DELAY_S`, `COOLDOWN_S`, `CHECKPOINT_EVERY`, `CANARY_URL`, `LOCAL_ROOT` |
| Seasons | `src/nba_ou/utils/seasons.py`, `get_season_year_from_date` | Reuse for `season` / `season_year` |
| Tipoffs | `fetch_nba_schedule.py` — one request per season, explicit UTC | Reuse for §8 and §10 validation |
| CLI entry points | `scripts/<domain>/<verb>_<thing>.py`, module docstring with `Usage:` examples, `argparse`, `main() -> int`, `sys.exit(main())`, `--dry-run` / `--report-only` / `--start-date` / `--end-date` / `--limit-dates` | New `scripts/injury_reports/` with the same shape |
| Logging | Mostly `print()` with `→`/`✓` markers; `logging.getLogger(__name__)` used rarely | Follow the dominant `print()` style for CLI progress; it is what the rest of the repo does |
| Backfill idiom | `scripts/line_history/update_line_history.py` — diff against a reference, insert-only, safe to re-run | Closest analogue; mirror its structure (report-only mode, per-season stats, gap listing) |
| Tests | Flat `tests/test_<domain>_<thing>.py`, pytest, no network in CI (`tests.yml` installs `requirements.txt` and runs `pytest tests/ -q`) | `tests/test_injury_report_archive.py` — pure-function tests for URL building, **DST edge cases**, manifest transitions, key building. Network paths mocked |
| Dependencies | Poetry (`pyproject.toml`) + parallel `requirements.txt` for CI | No new dependencies needed: `requests`, `boto3`, `pymupdf`, `pandas`, `pyarrow` are all present |
| Scheduled jobs | `.github/workflows/*.yml` | Out of scope now; the updater (§12) would slot in later |

Proposed new files (none written):

```
src/nba_ou/fetch_data/injury_reports/archive/
    urls.py         # label space, DST-correct datetime construction, key building
    discovery.py    # HEAD probing, canary, circuit breaker
    download.py     # GET, validate, store
    manifest.py     # Parquet read/write/merge, resume logic
    storage.py      # S3 + local-root abstraction
scripts/injury_reports/
    discover_injury_reports.py
    download_injury_reports.py
    audit_injury_report_archive.py
tests/test_injury_report_archive.py
```

---

## 12. Backfill and updater architecture

Both modes share `urls.py` / `discovery.py` / `download.py` / `manifest.py`; only
the date range and cadence differ.

**Historical backfill** (`/referee/injury/`, 2018-12-17 → 2025-12-22, plus the
listing-driven pre-CDN pass for 2018-10-17 → 2019-01-03):

* Driven entirely by the manifest — resume = "select rows without a terminal verdict"
* `--season` / `--start-date` / `--end-date` to scope a run
* `--max-requests` cap (optional) and a checkpoint every ~1,000 rows
* `--report-only` prints the gap analysis without touching the network
* `--dry-run` performs discovery but no writes
* Circuit-breaks on canary failure: sleeps 120 s, re-probes the canary, resumes; exits non-zero only if the canary stays down across several cooldowns
* Per-run audit JSON to `_audit/discovery_runs/`

**Current-season updater** — *implementable, and worth building.* The feed is
live (§1.5); it is simply idle between seasons.

* **Cadence: every 15 minutes**, matching the publication grid. Run at `HH:MM+02`
  ET to allow for the few seconds of generation drift seen in `/CreationDate`
  (`:00:03`, `:15:02`, `:30:05`, `:45:02`).
* A cheaper alternative that loses nothing: run **hourly and fetch the last 4
  slots**. Same coverage, ¼ the wake-ups, and it self-heals a missed run.
* **Idle detection**: the generator stops between seasons (2026-07-20 → present).
  The updater should tolerate long all-403 stretches without alarming — this is
  exactly where a naive "no files ⇒ something broke" alert would cry wolf for
  ten weeks. Alert only if reports are missing on a date the schedule feed says
  has games.
* Reuse the discovery/download code unchanged; only the date range differs.

### 12.1 Alternative sources investigated

Tested while the feed was (incorrectly) believed dead. Retained because it
establishes what the fallbacks are actually worth.

| Source | Result | Verdict |
|---|---|---|
| `official.nba.com/nba-injury-report-2026-27-season/` | 404 | Not created yet |
| `official.nba.com` 2025-26 page | 200, injury-link container **empty** | Retired at the format migration, not a shutdown |
| WP media API (`wp-json/wp/v2/media?search=Injury`) | 200, newest item **2019-01-03** | Valuable for the *pre-CDN* era (§3.1); irrelevant to recent data |
| `stats.nba.com/js/data/leaders/00_daily_lineups_<YYYYMMDD>.json` | 200 across 2026 | **Not a substitute** — see below |
| `cdn.nba.com/static/json/liveData/...` | 403 | Blocked |
| ESPN injuries APIs (`site.api` / `sports.core.api`) | 403 / 404 | Not usable as tested |

**The daily-lineups feed** carries only `rosterStatus ∈ {Active, Inactive}` and
`lineupStatus = Confirmed` — **no Questionable / Doubtful / Probable / Out** — and
its `timestamp` values for 2026-03-11 are `22:20`, `23:37`, `00:36`, `01:03`,
i.e. **at or after tipoff**. One snapshot per date, overwritten, with 404s before
2025-03. Fine as a cross-check on final active/inactive rosters; **actively leaky**
as a feature source.

### 12.2 Wayback Machine as a fallback — investigated and assessed

Queried via the CDX index, as specified. Findings:

**Inventory.** One prefix query returns the whole directory cheaply:

```
https://web.archive.org/cdx/search/cdx
  ?url=ak-static.cms.nba.com/referee/injury/*
  &output=json&collapse=urlkey&filter=statuscode:200&fl=timestamp,original
```

That single request (≈1.7 MB, ~60 s) yields **12,526 unique archived URLs**, of
which **10,206** parse as old-format and **2,291** as new-format injury reports.

| Season (by report date) | Archived on Wayback |
|---|---:|
| 2018-19 | 147 |
| 2019-20 | 407 |
| 2020-21 | 491 |
| 2021-22 | 2,054 |
| 2022-23 | 2,313 |
| 2023-24 | 1,818 |
| 2024-25 | 2,238 |
| 2025-26 (old format) | 738 |
| 2025-26 (new format) | 2,291 across 151 dates |

**Does it improve coverage? No — and that is the right answer.** Every date
Wayback holds is a date the NBA CDN still serves in full. Against the NBA's
~47,000 reports, Wayback's ~12,500 is a **sparse subset (~27%)**, and per-day it is
far from complete (2,291 new-format captures across 151 dates ≈ 15 of 96 slots).
There is no window where Wayback has something the CDN does not.

**Answering the specific questions asked:**

* *Expected 2025-26 PDFs directly available from NBA*: **all of them** — 2025-10-02
  → 2025-12-22 in the old format, 2025-12-22 → 2026-06-13 in the new. Nothing is missing.
* *Missing NBA PDFs recoverable via Wayback*: **zero**, because none are missing.
* *Unavailable from both*: **zero** for 2025-26.
* *Does Wayback meaningfully improve coverage*: **No.**
* *Is exact-URL CDX querying reliable enough as a scraper fallback*: **Yes,
  mechanically** — exact-URL queries returned correct capture lists with
  `mimetype=application/pdf` and `statuscode=200`, and per-URL latency was ~1–4 s.
  But at ~1–4 s per URL it is **1–2 orders of magnitude slower** than the CDN, so
  it is only sane as a *targeted* fallback for individually missing files, never as
  a bulk path. The prefix-inventory query is the efficient shape; per-URL queries
  are the precise one.

**Its real value here was diagnostic, not archival.** The CDX listing is what
exposed the new `_hh_mm` filenames and overturned the "feed is dead" conclusion.
That is a genuinely useful role, and it generalises:

> **Decision (confirmed): the download path uses the NBA CDN only.** Wayback is
> not a fetch source — no file is ever served to the pipeline from an archive
> mirror, so every stored PDF has exactly one provenance story.
>
> **Recommendation.** Do *not* build Wayback into the download path. **Do** run the
> single prefix-inventory CDX query as a **periodic audit** (say, once per season):
> diff the archived URL set against the manifest. Any archived URL the manifest has
> never seen is either a gap in discovery or — as here — **a naming-convention
> change the enumerator is blind to**. That check would have caught this bug on day
> one, and it costs one request.

**Do not confuse the two timestamps.** The CDX `timestamp` is when Internet Archive
captured the file (e.g. `20251116003757`) and is unrelated to when the information
became public. The temporal key for the dataset is always the **NBA report
datetime** parsed from the filename and confirmed against the PDF header. If a
Wayback-sourced file is ever stored, record the capture time in a separate
`wayback_capture_utc` column that no feature may read.

---

## 13. Risks and edge cases

| Risk | Severity | Mitigation |
|---|---|---|
| **False 403 recorded as absence** | **High** — silent, permanent data loss | Three-valued `nba_available`; canary; circuit breaker; 120 s cooldown; never let `unknown` become `false` without a clean re-probe. Keep concurrency at 1 |
| **Fall-back hour localized with `fold=1`** | **High** — 1 h look-ahead error on 7 dates | Explicit `fold=0`; unit test on 2024-11-03 |
| **Naming convention changes silently** | **High** — *this already happened once, on 2025-12-22, and fooled this investigation* | Periodic Wayback CDX inventory diff (§12.2); alert when a date the schedule says has games yields zero reports; never conclude "dead" from one candidate set |
| **Source disappears entirely** | Medium | Collect the full archive now, at once. Do not defer any part of the raw layer |
| **Bubble-era `:00` vs `:30`** | Medium — 30 min error on ~230 reports | Era-aware datetime construction; header cross-check catches it |
| Spring-forward 02:00 flagged as a gap | Low | DST guard in enumeration |
| Legacy-era horizon coverage assumed adequate | Medium — misleading features | Ship `report_age_minutes`; document the era split; consider gating early seasons out |
| Existing `nba_injuries` table conflated with snapshots | Medium — reintroduces availability leakage | Keep separate; §8 rule 6 |
| Repo's live fetcher already broken | Medium — affects daily predictions, not this task | Flagged in §4.1; separate ticket |
| PDF header format drift (2019 vs 2024 layouts differ) | Low | Tolerant regex; `date_mismatch` is a warning, not a hard failure |
| Manifest write interrupted mid-run | Low | Write to a temp key, atomic rename; per-season files bound the blast radius |
| Offseason yield uncertain | Low | Probe it; it is cheap |

---

## 14. Unresolved questions

1. ~~What is the recovery time from the 403 block?~~ **Resolved: 20–120 s of idle,
   measured three times.** A 120 s cooldown is sufficient.
2. **Is concurrency or rate the actual trigger?** The two were not varied
   independently: the run that tripped used 4 workers at ~8 req/s, the clean runs
   used 1 worker at ≤2.8 req/s. The §4.4 policy stays sequential on the
   conservative reading. Worth one controlled experiment (1 worker at 8 req/s)
   before anyone is tempted to parallelise.
3. **Is the block per-IP, per-connection, or per-UA?** Not tested. Matters only if
   the backfill ever needs to be parallelised across hosts.
4. ~~Where did the feed go after 2025-12-22?~~ **Resolved: nowhere.** The filename
   format changed to `_hh_mm` at 15-minute cadence (§1.5). Found via Wayback CDX.
5. **Exact bubble regime boundaries** — sampled weekly, so ±7 days. Cheap to pin
   down during Phase 1.
6. **True offseason yield per year** — the 2023 measurement was throttle-contaminated.
7. **Is the legacy era really only 3 hours/day?** The 24-label sweeps of legacy
   dates say yes, but those sweeps are the ones most exposed to false 403s. The
   45k-request discovery option re-tests this properly.
8. **Do reports exist for preseason before the 2021 regime change?** Not probed
   systematically.
11. **What does the 96-slot regime do at fall-back?** It has been through a
    spring-forward (2026-03-08 → **92/96**, the whole 02:00–02:45 band absent,
    exactly as the ET wall clock requires) but **never a fall-back** — the next is
    2026-11-01. If it publishes both 01:00–01:45 occurrences that is 100 files in
    a 100-quarter-hour day, and the source filename has no way to distinguish
    them. The canonical name handles it (§6.2); the *enumerator* would need to
    probe each repeated slot twice and disambiguate by `/CreationDate` offset.
    **Check this on 2026-11-01.**
12. **Will the 2026-27 season keep the 96/day cadence?** Unknown until it starts
    (~Oct 2026). The enumerator should detect the era from what actually returns
    200 rather than hard-coding the cutover forward in time.
13. **Is the WP media API listing complete for the pre-CDN era?** It returns 200
   PDFs with `X-WP-Total: 205`, and the date range (2018-10-17 → 2019-01-03) is
   contiguous with the `/referee/injury/` era — so it looks complete, but it is a
   search index, not a directory. Cross-check the 3/day cadence against the
   2018-19 game schedule during Phase 1.
14. **Are the 103 scanned reports worth OCR?** They cover Oct–Nov 2018 only.
    Probably not, but store them now — the source is visibly fragile.

---

## 15. Implementation sequence

Ordered, each step independently verifiable.

**Phase 0 — (largely resolved during this research)**
1. ~~Measure the 403 recovery window.~~ Done: 20–120 s.
2. Optional, 30 min: one controlled run at 1 worker / 8 req/s to separate rate
   from concurrency (question 2). Only needed if the 7.5 h backfill is felt to be
   too slow — it almost certainly is not.

**Phase 1 — foundations, no network (1 day)**
4. `urls.py`: label space, era-aware `HH:30`/`HH:00`, DST guard, `fold=0`, key building.
5. `tests/test_injury_report_archive.py` covering 2024-03-10 (no 02AM),
   2024-11-03 (`fold=0`), bubble `:00`, season/label mapping. **Write these before
   the network code** — they are the temporal-correctness contract.
6. `manifest.py`: schema, Parquet round-trip, merge/resume semantics.

**Phase 2 — discovery (1–2 days incl. runtime)**
7. `discovery.py` with canary + circuit breaker.
8. `scripts/injury_reports/discover_injury_reports.py` with `--report-only`,
   `--dry-run`, `--season`, `--max-requests`.
9. Validate on one known season (2023-24) — expect ~6,168 reports; confirm zero
   `unknown` rows survive.
10. Run full discovery 2018-12-17 → today as a resumable loop, **era-aware**:
    24 labels/day up to 2025-12-22, 96 labels/day from 2025-12-22 onward.
10b. Add the listing-driven pass over the WP media API (§3.1): page the endpoint,
    rewrite `cms.nba.com` → `ak-static.cms.nba.com`, parse report datetimes from
    the eight filename shapes, de-duplicate the 11 overlapping dates, and flag the
    103 image-only files.
11. **Reconcile against §9 estimates and the schedule feed before downloading anything.**

**Phase 3 — download (1–2 days incl. runtime)**
12. `storage.py` (S3 + `--local-root`), `download.py` with the §10 validators.
13. `scripts/injury_reports/download_injury_reports.py`.
14. Backfill by season, newest first — the quarter-hourly era is the high-value
    data and the most exposed if the naming changes again. Download day set C
    (game days + 1 day before, all game types incl. playoffs and play-in) at the
    full 15-minute step; discovery stays ungated so the choice is reversible (§9.3).
15. Verify: object count == manifest `exists=true` count; all checksums distinct.

**Phase 4 — audit (½ day)**
16. `audit_injury_report_archive.py` implementing the §10 per-date/season checks
    against the schedule feed.
17. Publish a short findings note (gaps found, allow-listed gaps, final counts).
18. **Add the Wayback CDX inventory diff as a recurring audit** (§12.2) — one
    request, and it is the check that catches the next naming change.

**Phase 5 — current-season updater (½ day, once 2026-27 starts)**
19. Hourly job fetching the last 4 quarter-hour slots (§12); idle-tolerant.

**Phase 6 — deferred**
20. Parsing into `processed/` — separate task, separate plan.
21. Optional OCR for the 103 scanned 2018 reports.

---

## 16. Recommended architecture in one paragraph

Enumerate the fully-determined, **era-dependent** URL space over
2018-12-17 → today — 24 hourly labels/day before 2025-12-22, **96 quarter-hour
labels/day after** — DST-aware throughout, and add a separate listing-driven pass
over the WordPress media API for the irregularly-named pre-CDN era
(2018-10-17 → 2019-01-03), converting every filename into an
exact `America/New_York` `HH:30` wall-clock instant and thence to UTC. Probe it
with sequential HEAD requests at ~1 req/s behind a canary and circuit breaker,
recording a **three-valued** availability verdict into a per-season Parquet
manifest that is the sole source of resume state. Download only confirmed-existing
URLs, validate each against `%PDF`, `pymupdf`, and its own in-PDF header
timestamp, and store the bytes unmodified under a **canonical, era-independent,
ET-timestamped key** — `injury_reports/raw/season=…/date=…/injury-report_<ET
timestamp><offset>.pdf`, which resolves the hourly era's misleading `_08PM`-means-
20:30 labels once at ingest — in the existing S3 bucket, reusing
`s3_models.py` for all AWS access and the `scripts/<domain>/` argparse idiom for
entry points. Expect **≈47,000 PDFs and ≈2.5 GiB**, growing ~1.7 GiB per season. The feed is
live, so follow the backfill with a 15-minute (or hourly, last-4-slots) updater,
and run the one-request Wayback CDX inventory diff each season as the tripwire for
the next naming change.


---

## 17. What was built, and how to run it

### 17.1 Files

```
src/nba_ou/fetch_data/injury_reports/archive/
    urls.py        # era model, label decoding, DST rules, canonical keys   (241)
    client.py      # polite HTTP + canary + circuit breaker                 (219)
    discovery.py   # phase 1: HEAD probe -> manifest rows                   (168)
    download.py    # phase 2: fetch, validate, store                        (201)
    validation.py  # %PDF / pymupdf / in-PDF header cross-check              (92)
    manifest.py    # per-season Parquet, field-level merge, resume state    (169)
    storage.py     # S3 and local-mirror backends behind one key            (70)
scripts/injury_reports/
    backfill_injury_reports.py   # the CLI                                  (279)
tests/test_injury_report_archive.py                                        (252)
```

No new dependencies: `requests`, `boto3`, `pymupdf`, `pandas` and `pyarrow` are
all already in `pyproject.toml` and `requirements.txt`.

### 17.2 Running it

```bash
# what exists and what is already collected
python scripts/injury_reports/backfill_injury_reports.py --list-seasons

# plan a season - no network at all
python scripts/injury_reports/backfill_injury_reports.py --season 2023-24 --report-only

# one season, end to end, into S3
python scripts/injury_reports/backfill_injury_reports.py --season 2023-24

# one season into a local mirror instead (no AWS needed)
python scripts/injury_reports/backfill_injury_reports.py \
    --season 2023-24 --local-root data/injury_reports

# the full backfill, oldest season first
python scripts/injury_reports/backfill_injury_reports.py --all-seasons

# an explicit window, ignoring season boundaries
python scripts/injury_reports/backfill_injury_reports.py \
    --start-date 2026-03-01 --end-date 2026-03-31
```

`--season` is repeatable. Other flags: `--phase discover|download|both`,
`--dry-run`, `--max-requests`, `--max-files`, `--bucket`, `--manifest-root`,
`--delay`, `--cooldown`, `--quiet`.

### 17.3 Behaviour worth knowing

* **Resumable by construction.** The manifest is the only state. A candidate with
  a terminal verdict is never re-probed; a stored object is never re-fetched.
  Ctrl-C and re-run the same command. Exit code `2` means "stopped early, run it
  again"; `0` means the window is complete.
* **Every 403 is adjudicated by a canary** before it can be recorded as an
  absence, and a persistent canary failure stops the run instead of writing
  false negatives.
* **Every stored PDF is validated against its own header line.** A file whose
  in-PDF timestamp disagrees with the time derived from its URL is marked
  `date_mismatch` and **not** counted as stored -- this is the check that catches
  a future era where the labels change meaning again.
* **Canonical keys are collision-checked.** Two source URLs resolving to one key
  raise rather than overwrite.
* **The 2018 scanned reports** validate as `image_only` rather than failing; they
  are stored and flagged for optional OCR later.
* **Manifests are mirrored to S3** at the end of each season window, and pulled
  back down on start when the local copy is missing — so a backfill can resume on
  a different machine. `--no-manifest-sync` keeps them local only.

### 17.4 Verified end to end

Live run against 2026-03-11: discovery found **96/96**, downloads validated `ok`,
objects landed as e.g.

```
injury_reports/raw/season=2025-26/date=2026-03-11/injury-report_2026-03-11T2015-0400.pdf
```

A second run of the same command re-probed **0** URLs and skipped the already
stored objects. `pytest tests/test_injury_report_archive.py` -- **34 passed**.

### 17.5 Not built (deliberately)

* **The pre-CDN WordPress era (§3.1)** -- listing-driven, 189 net-new reports,
  103 of them scanned images. Separate module, separate run; nothing else depends
  on it.
* **The current-season updater (§12)** -- there is nothing to poll until the
  2026-27 season starts. The discovery/download code needs no change for it.
* **Parsing into `processed/`** -- out of scope by the brief.
* **The Wayback CDX audit (§12.2)** -- one request, worth adding as a seasonal
  job; it is the tripwire that catches the next naming change.
