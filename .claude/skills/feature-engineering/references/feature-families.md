# Feature families in detail

Companion to `../SKILL.md`, which carries the temporal rules and the philosophy.
Read that first. This file is the per-family reference.

Each family answers six questions:

1. **Captures** — what information it encodes.
2. **Computed** — how, concretely.
3. **Predictive because** — the mechanism.
4. **Temporal correctness** — why it cannot see the future.
5. **Sport-agnostic core** — what to keep.
6. **MLB** — what changes.

---

## 1. Rolling team performance

**Captures.** Recent form, season-long level, volatility, direction of travel,
and home/away tendency — for every volume, efficiency and market statistic.

**Computed.** On team-game rows, one row per team per game. For each source
statistic, several variants are produced:

| Variant | Column shape | Window | Notes |
|---|---|---|---|
| Rolling mean | `{P}_LAST_ALL_{w}_MATCHES_BEFORE` | 1, 2, 3, 5, 10 | `shift(1).rolling(w, min_periods=1).mean()`, grouped by team **across seasons** |
| Home/away split | `{P}_LAST_HOME_AWAY_{w}_MATCHES_BEFORE` | same | stored as **(home/away mean − overall mean)**, not the raw split |
| Window contrast | `{P}_LAST_5_MINUS_LAST_10_MATCHES_BEFORE` | 5 vs 10 | "recent vs less recent" without the model differencing two columns |
| Weighted mean | `{P}_LAST_{w}_WMA_BEFORE` | 5 | linear weights `1..n`, so recency is graded rather than a cliff |
| Season mean | `{P}_SEASON_BEFORE_AVG` | expanding | grouped by (team, season, home/away) |
| Season std | `{P}_SEASON_BEFORE_STD` | expanding | `ddof=0`; the volatility/consistency measure |
| Trend slope | `{P}_TREND_SLOPE_LAST_{w}_GAMES_BEFORE` | 5, 10 | OLS slope over the last w values, `min_periods=2` |
| Slope home/away | `..._HOME_AWAY_GAMES_BEFORE` | 5 | again a **delta** vs the overall slope |

Which statistics get which variants is deliberate: everything gets a 5-game mean
and a season mean/std; a short list (points, points-per-40, the market line, the
market's error) additionally gets 1/2/3/10-game windows, a WMA and trends. Odds
and consensus-percentage columns are discovered *dynamically* by name pattern and
rolled too — so a new book automatically gets its full rolling family.

**Three design decisions worth copying:**

- **Splits are stored as deltas, not levels.** `HOME_AWAY` columns hold
  *(same-venue mean − overall mean)*. The level is already in the overall column;
  the delta is the marginal information, and it is far less collinear.
- **Rolling means cross the offseason (`group_by_season=False`), trend slopes do
  not.** A 5-game window at a season's start would otherwise be empty; a *slope*
  spanning an offseason is not a trend.
- **Only source columns discovered at function entry are trended.** Rediscovering
  from the frame mid-function would match already-derived columns and recursively
  build trends of rolling averages.

**Predictive because.** Team scoring level is persistent; deviation from it
mean-reverts. A totals market prices a consensus estimate, and short-window form
versus the season baseline is where a model can disagree with it. Volatility
matters independently: two teams with the same mean but different variance
produce different total distributions, and the payoff is a threshold on that
distribution.

**Temporal correctness.** `shift(1)` before every window. Season means are
expanding-with-shift. The season-opener gap is closed by the previous-regular-
season fallback chain (see SKILL.md rule 3), not by widening the window.

**Sport-agnostic core.** Multi-window means with an explicit recency-weighted
variant; expanding season mean **and** std; a slope for direction; splits stored
as deltas; dynamic discovery so new source columns are picked up automatically.

**MLB.** Direct, with three adjustments:
- **Windows should be longer.** Baseball's game-to-game outcome noise is far
  higher relative to team skill; a 5-game team run-scoring mean is nearly pure
  noise. Use 10/20/30 and lean on the season expanding mean.
- **Roll the peripherals, not the results.** wOBA, wRC+, K%, BB%, hard-hit rate
  stabilise far faster than runs scored or ERA. Roll the stabilising quantity and
  let the model map it to runs.
- **Handedness splits** are the MLB analogue of home/away splits, and are more
  important: vs-LHP and vs-RHP team offence differ materially. Store them the
  same way — as deltas from the overall.

---

## 2. Team-vs-team relationships

**Captures.** What happens when *these two* teams meet, rather than each team's
independent quality.

**Computed.** Four sub-groups, all from `_BEFORE` inputs only:

**(a) Mechanical differences.** `_DIFF_BEFORE` columns generated
automatically: every column matching a betting-related pattern *and* a
rolling-statistic pattern gets `home − away`. Idempotent — existing diff columns
are overwritten in place.

**(b) Offence-vs-defence crossings.**
```
OFFDEF_MISMATCH_HOME_OFF_MINUS_AWAY_DEF = home offensive rating − away defensive rating
OFFDEF_MISMATCH_AWAY_OFF_MINUS_HOME_DEF = away offensive rating − home defensive rating
```
Plus `DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE` on 5-game windows rather than season
means.

**(c) Expected tempo, then expected points.**
```
EXPECTED_POSS_FROM_PACE      = (home season pace + away season pace) / 2
EXPECTED_PTS_HOME_FROM_OFFR_PACE = EXPECTED_POSS × home offensive rating / 100
POSS_X_TSPCT_HOME            = home season possessions × home true-shooting
```
This is the volume × efficiency decomposition made explicit.

**(d) Style-rate crossing.** The most interesting construction here. Each team's
completed games produce five *offensive* rate observations (three-point-attempt
rate, free-throw rate, turnover rate, offensive-rebound rate, shots per
possession) and, by taking **the opponent's same-game rate**, five matching
*allowed* observations. Those ten are rolled into `_SEASON_BEFORE_AVG` history.
Then, at merge time:

```
expected_FG3A_rate_home = mean(home 3PA-rate history, away 3PA-rate-ALLOWED history)
expected_total_FG3A     = expected_possessions ×
                          (home shots/poss × expected_FG3A_rate_home +
                           away shots/poss × expected_FG3A_rate_away)
```

The raw style history columns are then **dropped**, leaving only the crossed
interactions — the sources were an intermediate, not a feature family.

**(e) Head-to-head.** Last-5 prior meetings excluding the current game, and
points-conceded differentials in this matchup.

**Predictive because.** A high-pace offence against a team that allows a high
pace produces more possessions than either team's average suggests. Totals are
the market where matchup interactions matter most, because the two teams' effects
*add* rather than compete.

**Temporal correctness.** Every input carries `_BEFORE`. The style crossing has
no access to same-game attempts, turnovers or possessions — enforced by a check
that the configured history suffix contains `BEFORE`. Critically, the
opponent-value helper returns a value **only for valid two-team games with two
non-null values**, so a scheduled game's blank box score stays blank and cannot
enter its own history via the opponent's row.

**Sport-agnostic core.** (i) Cross each team's offensive rate with the opponent's
allowed rate; (ii) estimate the event count from both teams' tempo; (iii)
multiply. Keep the crossed interaction, discard the raw style history.

**MLB.** The framework transfers cleanly and is arguably *more* natural, because
baseball's platoon structure gives real interactions:
- offence-vs-defence → **team wOBA vs starting-pitcher FIP/xFIP**, then vs
  bullpen quality for the later innings.
- expected tempo → **expected plate appearances**, driven by on-base rate and
  expected innings (a rally extends the inning; a strikeout pitcher shortens it).
- style crossing → **batted-ball mix vs pitcher batted-ball profile** (a fly-ball
  offence against a fly-ball pitcher in a small park is the canonical MLB total
  interaction), and **team K% vs pitcher K%**.
- **Handedness is a genuine MLB-only interaction** with no NBA analogue: the
  lineup's platoon split against the starter's throwing hand. Build it.
- Head-to-head is weaker in MLB (more games, more roster churn) but
  batter-vs-pitcher history exists — treat it with heavy shrinkage; sample sizes
  are tiny and it is a well-known overfitting trap.

---

## 3. Schedule, rest and travel

**Captures.** Accumulated fatigue and disruption from the calendar and geography.

**Computed.** Rest first, on team-game rows:
- `REST_DAYS_BEFORE_MATCH` = day difference from the team's previous game
  **within the same season**, defaulting to 7 for a season's first game.
- `BACK_TO_BACK_BEFORE` — a game-level flag set when **both** teams are on ≤1
  day's rest.
- `REST_DAYS_DIFF_HOME_MINUS_AWAY` — the mismatch, which is what actually moves a
  line.

Travel is computed from a **team-centric game log** where each game contributes
two rows and the `CITY` is *where the game is played* — so the away team's city is
the **home team's** city. That one line is the whole trick: home stands and road
trips fall out of it automatically.

- `TRAVEL_KM` = great-circle distance from the previous game's city, with three
  zero rules: first game, missing city, or **both current and previous games at
  home**.
- `KM_LAST_{1,2,5,7,14}_DAYS` = calendar-window rolling sums, `closed="both"`,
  **including the trip to the current game** because that travel has already
  happened before tip-off.
- `JETLAG_HOURS_FROM_LAST_GAME` = absolute UTC-offset difference between the
  previous game's city and this one, computed on the game date so DST is handled,
  and **zeroed if the previous game was more than 4 days ago** — you have adapted.
- `TOTAL_KM_*` are `log1p`-transformed. Distance effects are not linear;
  the difference between 0 and 500 km matters more than 3,000 vs 3,500.
- `TRAVEL_RECENCY_RATIO_{SIDE}_2D_OVER_14D` = compression: the same 3,000 km in
  two days is not the same as over two weeks.

**Predictive because.** Fatigue lowers shooting efficiency and pace; the market
prices the obvious back-to-back but tends to underprice cumulative and
compressed travel.

**Temporal correctness.** All values derive from *previous* game dates and
locations, both known. The current game's location is known from the schedule.
Nothing here touches an outcome. Rest resets per season so the offseason is not
counted as rest.

**Sport-agnostic core.** A team-centric game log keyed on *where the game is
played*; distance from the previous location; multiple calendar-window rolling
sums; timezone change with an adaptation cutoff; log-scaling; a compression ratio
between a short and long window; the **rest differential**, not just the levels.

**MLB.** The concepts transfer but the framing must change, and this is the
family that needs the most rethinking:
- **Baseball travels in series, not games.** Three or four consecutive games at
  one venue means per-game travel distance is zero for most rows and the feature
  is nearly constant. Compute travel at the **series** level and attach it to
  every game in the series, plus a `game_number_in_series` and
  `is_series_opener` / `is_getaway_day`.
- **Rest days are mostly 0.** The MLB schedule is near-daily, so `REST_DAYS` has
  almost no variance. The informative version is **consecutive games played**
  (a 17-in-17 stretch), and **days since the last off-day**.
- **The real MLB fatigue variable is the bullpen, and it has no NBA analogue.**
  Relief innings thrown in the last 1/2/3 days, back-to-back appearances by the
  closer, whether a bullpen game is scheduled, whether last night went extra
  innings. This is probably the single highest-value MLB-specific feature family
  and should be built as its own family, not squeezed into "travel".
- **Getaway-day and day-after-night games** are the classic scheduling effects.
- Timezone change still matters (coast-to-coast series).

---

## 4. Player availability and injuries

The highest-value family and the one with the most leakage risk. Read
SKILL.md's leakage audit alongside this.

### Where the information comes from

Two sources, one shape.

**Historical:** the game's own **inactive list** from the box-score summary
endpoint, stored as `nba_injuries(player_id, game_id, team_id, game_date,
season_id, season_year)`. Binary: on the list or not. Augmented by any player
whose box-score `COMMENT` matches `injur|injry`.

**Same-day:** the league's official **injury-report PDF**, scraped, parsed with
PyMuPDF, filtered to `OUT_STATUSES = {"Out", "Doubtful"}`. Games whose report
reads `NOT YET SUBMITTED` are **excluded from prediction entirely** rather than
predicted with unknown availability.

Both become `{game_id: {team_id: [player_id, ...]}}` and feed one code path.

### Resolving player identity

Report names arrive `"Last, First"`, are reversed, looked up in the provider's
static player index, and disambiguated by querying each candidate's current team
when several share a name. Unmatched players are enumerated with team and
date — **but the raise is commented out**, so they are logged and skipped. Fix
that in a new build.

### Aligning an injury to a historical game

Historical: the inactive list is *already keyed by* `game_id`, so no alignment is
needed. This is the reason the settled source is used — and the reason for the
train/serve skew.

Same-day: the report's `Game Date` + `Matchup` is concatenated into a merge key
matched against the schedule's game code.

### Roster context — who *would* have played

An absence is meaningless without the counterfactual roster.
`create_player_lookup` precomputes, per season:
- every player's chronological `(date, team_id)` timeline,
- the set of players who ever appeared for each team,

and answers "is player P on team T as of date D" by finding P's **last
appearance strictly before D** and checking it was for T. Trades are handled by
that rule alone. A cross-check drops a player who appears on another team's
injury list for the same game.

Season openers have no earlier same-season game, so the roster falls back to the
game's own box-score rows — see the leakage audit.

### From absences to team-level features

Three tiers, in increasing sophistication.

**Tier 1 — who and how good.** For each team-game, players are split into
available and injured, each sorted by a **recency-weighted (EWMA, halflife 10
games) season-to-date average** of each of six statistics (points, pace,
defensive rating, offensive rating, true shooting, minutes), with a minimum
average-minutes threshold applied as a *sort key* rather than a hard filter (so a
short list still fills). Emitted:

- top-6 available and top-4 injured player values per statistic,
- `TOTAL_INJURED_PLAYER_{stat}` / `TOTAL_NON_INJURED_PLAYER_{stat}` — sums over
  **all** players, not just the top N,
- `AVG_INJURED_{stat}`, `N_INJURED_PLAYERS`, `N_ACTIVE_PLAYERS`,
- `TOP{i}_INJURED_STREAK_PTS` — consecutive **team games** the player has been
  out (see below),
- bench depth: among available players whose 5-game average minutes fall in
  **7–21**, the average and max points-per-minute and pace, plus a count.

Player **ids and names** are carried through as bookkeeping so a later stage can
locate the top players, then **dropped at the very end**. An id is worse than
useless as a numeric feature — player 1610612747 is not "greater than" 201939 —
so every split on it is arbitrary.

**Tier 2 — importance weights.** Rather than assuming a weight, importance is
whatever the sort statistic says: scoring average for `PTS`, minutes for `MIN`,
and so on. A separate external proxy (all-star vote share, family 8) covers star
power the box score misses. Explicitly **not** used: starts, usage rate, or
salary.

**Tier 3 — empirical player-availability effect.** The most transferable idea
here. For each of a team's leading players, over that team's games in the
**current and previous season, strictly before this game's date**:

```
raw_effect = mean(outcome | player present) − mean(outcome | player injured)
```

computed for two outcomes: total points, and the market's error
(`total − closing line`). Then shrunk empirical-Bayes style:

```
n_eff     = min(n_games_injured, n_games_present)
shrunk    = raw_effect × n_eff / (n_eff + k)        # k = 10
```

Aggregated to `MEAN` and `MAX_ABS` per side, plus the total sample size. Two
passes run: one over the top available players, one over the top injured players.
`n_eff = 0` gives 0, which is also the fill value for "no evidence" — so the
feature is continuous at exactly the point shrinkage was designed to handle.

**Why the market's error as an outcome.** "This player's absence adds 4 points"
is interesting; "this player's absence adds 4 points *that the market did not
price*" is the actual edge. Measuring the effect against the line rather than
against the raw total is the sharper construction.

**Predictive because.** Absences are the largest single-game perturbation to a
team's scoring distribution, they are known before tip, and the market prices
star absences well but role-player and depth absences less well.

**Temporal correctness.**
- Player averages are EWMA over *prior valid appearances* with an explicit
  `shift(1)`.
- Availability effects filter `GAME_DATE < before_date` and are restricted to two
  seasons.
- Injury streaks walk the team's game list backwards from the current game,
  capped at two seasons.
- **The unresolved issues** are in SKILL.md's audit: the settled-vs-forecast
  source skew, the post-game `COMMENT` field, backdated IL stints in MLB, and
  the season-opener roster fallback.

**Sport-agnostic core.**
1. Absences → the counterfactual roster → per-player importance → a team-level
   aggregate.
2. Importance from a **recency-weighted prior-appearance average**, not a static
   depth chart.
3. The empirical **present-minus-absent** effect, **shrunk by the smaller of the
   two sample sizes**, measured **against the market line** as well as the raw
   outcome.
4. Absence **duration in team games**, not days.
5. Both a top-N view and an all-players sum.
6. Refuse to predict when availability is unknown.

**MLB.** The single biggest structural difference in the whole translation:

- **The starting pitcher dominates.** A pitcher's presence/absence is not a
  perturbation, it is close to a different game. Do not fold pitchers into a
  generic "injured players" aggregate — model the **announced starter** as a
  first-class feature (his own rolling history, family 1) and treat a *late
  scratch* as its own event.
- **Availability is near-binary and published.** The daily **lineup card**
  (posted 2–4 hours before first pitch) gives the exact nine hitters and the
  starter. That is a far better source than an injury report — archive it.
- **The IL is the injury source, and it is dangerous.** Stints are **backdated**;
  querying "was player X on the IL on date D" after the fact returns a *yes* for
  dates when nobody knew. Use the **transaction feed timestamped at
  publication**, not the stint's official start date.
- **Rest days for regulars** are routine, not injuries. A catcher's scheduled day
  off is a planned absence with a known, small effect — worth distinguishing from
  an injury absence, because their effects differ and pooling them dilutes both.
- **The bullpen** is a rolling availability state, not a binary — see family 3.
- The empirical present-minus-absent estimator transfers directly and is
  *better* suited to MLB, where the 162-game season gives much larger `n_eff`.

---

## 5. Roster continuity

**Captures.** How much of the team is the same team as before — over a long
horizon (since last March) and a short one (the last two months).

**Computed.** Six columns, all `_BEFORE`:
`ROSTER_MINUTES_CONTINUITY_PCT`, its 2-month variant,
`ROSTER_NEW_PLAYER_MINUTES_PCT` and its 2-month variant, and the two `NET`
columns (incoming minus lost, equivalently incoming + continuity − 1).

Mechanics:
- The window for a target season opens **March 15 of the preceding season** and
  closes at the target game. A player enters the window when a box score *or* an
  injury report assigns them to a team.
- A player counts as **lost** only when their latest assignment inside the window
  is to a *different* team.
- Every candidate is weighted by **minutes played for the target team per team
  game**, normalised by a full team's minutes; the current season's rate is used,
  falling back to the previous season's.
- **Incoming** players are those whose latest assignment is this team and whose
  preceding distinct assignment in the window was another team; their value is
  their average minutes for that previous team.
- The short-horizon window is two calendar months, **extended back to March 1 if
  that start would fall in June–September** — so an offseason start still spans to
  a usable pre-summer baseline.

**One event-timing rule is the whole leakage story.** Each roster event carries
both an actual date and a **known date**: a box score becomes known **the next
day**, while an injury report is legitimate **same-day, pre-tip** information.
Scheduled-game placeholder rows (identified by an explicit scheduled id plus a
null-minutes contract) are known same-day.

**Predictive because.** Continuity proxies for chemistry, scheme familiarity and
role stability, none of which appear in per-player averages. It is also the
correct discount on early-season rolling features: a team that returns 90% of its
minutes is well described by last season; one that returns 40% is not.

**Temporal correctness.** The known-date rule above, plus regular-season and
postseason games only.

**Sport-agnostic core.** Weight roster membership by **playing time**, not
headcount. Track **both** directions (retained and incoming) and their net. Use
**two horizons** — offseason turnover and recent churn are different phenomena.
Give every membership event an explicit *known-at* timestamp distinct from its
*occurred-at*.

**MLB.** Directly applicable with a different weight:
- Weight by **plate appearances** for hitters and **innings pitched** for
  pitchers, computed separately — a lineup that returns 90% of its PAs with a
  rebuilt rotation is a different team from the reverse.
- **The trade deadline** is the sharp short-horizon event, and **September
  call-ups** / roster expansion are a second one with no NBA analogue.
- Minor-league churn means far more players cross the boundary; consider
  restricting candidates to those above a playing-time floor, or continuity will
  be dominated by one-game call-ups.

---

## 6. Market and line movement

Full construction is in the `odds-data-architecture` skill (snapshots,
normalisation, canonical level, cross-book aggregation). This section covers
what reaches the model and the rules on using it.

### Closing-line dataset (one row per game)

- Per-book totals, spreads and moneylines; consensus percentages; the consensus
  **opening** line.
- Derived: implied team totals `(total ± spread) / 2`; per-book over/under price
  skew as `log(price_over / price_under)`; devigged probability difference; the
  vig itself; cross-book **median, std, IQR, MAD and range** of the line;
  coverage counts (how many books quoted at all).
- Rolling histories of the line and of the market's error: the market's error on
  a team's *previous* games is a legitimate pre-game feature and is rolled over
  1/2/3/5/10-game windows plus a season std and a trend.
- **Move-from-opener features are deliberately excluded** from this dataset, to
  avoid a train/inference mismatch where the opener is not reliably available at
  serving time.

### Snapshot dataset (one row per game × horizon)

Per book, per market, at each of ten horizons: raw line, both prices,
`has_quote`, `line_age_minutes`, normalised line, `norm_minus_raw`, devigged
probabilities, overround; movement since the opener; movement, absolute movement,
velocity and probability-movement over each trailing window with a `has_window`
flag; move counts, price-only re-price counts, distinct levels, reversals, the
realised min/max/std of the path, `position_in_range`, moves per hour, and
whether the recent direction opposes the *first* move.

Across books, per market and horizon: median consensus, cross-book std and range,
book count, median line age, steam count and fraction, and each book's signed and
z-scored deviation from consensus with an outlier flag.

### What is legal at each horizon

| Quantity | Legal at horizon T? |
|---|---|
| Any tick with `minutes_before_tip >= T` | **yes** |
| The opening line | **yes** — openers land a median ~25h before tip |
| Closing line of **this** game | **no** — physically separated into a scoring file |
| Closing line of **previous, finished** games | **yes** — family 11 |
| Any tick at or after tipoff | **no** — `is_pregame` |
| Movement over a window ending *after* T | **no** — the builder raises |

**Predictive because.** The line is the single strongest feature in any sports
model — it aggregates everything the market knows. The residual against it is the
only quantity worth modelling. Movement, dispersion and staleness are the visible
traces of *how confident* that aggregate is.

**Sport-agnostic core.** Model the **residual against the line**, not the raw
outcome. Carry both the raw and price-normalised line. Make availability and
staleness explicit features. Report ROI against the raw, executable number.

**MLB.** Same structure. Two additions: the **run line is not a spread** (it is a
fixed ±1.5 with moving prices, so the *price* carries the information the line
carries in NBA — treat the devigged probability as the level), and totals are
conditioned on the announced starters, so a snapshot grid should know whether
they were announced.

---

## 7. Officials

**Captures.** The individual arbiters' measurable effect on scoring, fouls and
the market's error.

**Computed.** Referee names are canonicalised (jersey suffixes stripped, periods
removed, whitespace collapsed), sorted and de-duplicated into deterministic
slots — the computation is order-invariant, so the scheduled and historical
sources must normalise identically or they will not match.

For each of three metrics (total points, market error, total fouls) and each
official on the current crew:

```
delta = mean(metric | games with this official) − mean(metric | games without)
```

over **the same season before this date**, falling back to **the whole previous
season** when the current-season sample is one-sided. The per-official deltas are
then aggregated to `REF_AVG_`, `REF_STD_` and `REF_SUM_`. An optional exact-trio
variant exists and is **disabled by default** — exact trios are too rare.

**Predictive because.** Officiating tendency shifts free-throw volume and pace,
and the market prices crews weakly or not at all.

**Temporal correctness.** Same-season games are filtered `GAME_DATE < current`.
The previous-season fallback is **not** date-filtered, relying on that season
being complete — true for NBA, worth re-checking elsewhere.

**Sport-agnostic core.** Per-official **with-minus-without** delta, aggregated
order-invariantly, with a previous-season fallback, computed against the market's
error as well as the raw outcome. Assignments must be normalised identically on
both the historical and scheduled paths.

**MLB.** **Stronger than the NBA version, and one of the best MLB-specific
features available.** The home-plate umpire's strike zone directly moves K rate,
walk rate and run scoring; umpire assignments are published pre-game; and there
is public per-umpire zone data (called-strike rate, zone size). Use
with-minus-without on runs, on K rate, on walks, and on the market's error. Only
the plate umpire matters much — do not aggregate the whole crew.

---

## 8. External importance signals

**Captures.** Star quality and public prominence, sourced **independently of the
team's own box scores**.

**Computed.** All-Star fan voting is scraped per season and mapped to players.
For each team-game: the team's share of league-wide fan votes among available
players, the max vote share among *injured* players, raw vote totals, and a
candidate count. Player-to-team assignment as of the game date uses the same
last-appearance-before-date timeline as the injury features, with the injury
report taking priority when it names a team. The build **raises** if voting data
is missing for a required season rather than emitting zeros.

**Predictive because.** Star absence is not linear in minutes or points, and the
market reacts to *name recognition* as much as to production. An
externally-sourced measure also works at the start of a season when rolling
averages are empty or stale.

**Temporal correctness.** The vote season is derived from the game date such that
a game before March uses the *previous* cycle's voting — the results are public
before the games they are used on.

**Sport-agnostic core.** Have **at least one importance signal that is not a
function of your own rolling statistics.** Snap it to the correct point-in-time
cycle. Fail loudly when it is missing.

**MLB.** All-Star voting exists but is weak. Better substitutes, in order:
**preseason projection systems** (Steamer / ZiPS / PECOTA projected WAR, published
before the season and therefore trivially point-in-time correct), **contract
value / AAV**, and prospect rankings for rookies. Projected WAR is the natural
analogue and is a better importance weight than anything the NBA version has.

---

## 9. League-wide market regime

**Captures.** How well the market as a whole has been pricing recently —
independent of the two teams involved.

**Computed.** Over rolling windows of **15/30/75/150 games**, **3/7/14 days**, and
EWM spans of 15/30/75 games, all strictly from earlier dates:

- signed bias (mean of `actual − closing line`) and robust median bias,
- MAE and its median,
- error volatility (std),
- tail-miss rates at |error| > 10, 15, 20,
- over / under / push rates,
- league mean closing total and mean actual total, and the gap,
- short-window ÷ long-window ratios (regime shift), and their differences
  (acceleration / correction speed),
- game counts per window (league activity),
- opening-line features: open-to-close move, direction, whether the close beat the
  open, error against the open,
- cross-book dispersion, per game and rolled.

**One implementation detail is the whole leakage story.** The rolling aggregation
is done **by calendar-date group**, and every row on the same date receives the
value computed from strictly *earlier* dates. Same-day games therefore cannot
influence each other. A naive `rolling()` over a game-ordered frame would let the
first game of a slate leak into the last.

**Predictive because.** Scoring environments drift within a season (rule
emphases, injuries, fatigue), and books adapt with a lag. A persistent league-wide
bias is a real, if small and non-stationary, edge — and a model that only
rediscovers it should be measured against a "line + its historical drift" null
rather than against the line alone.

**Temporal correctness.** Date-group aggregation from strictly earlier dates.
Uses this game's own closing line only as the *reference point* for other games'
errors, never its own.

**Sport-agnostic core.** Roll the market's realised error at the **league** level
over multiple game- and calendar-windows; include bias, dispersion and tail rates;
aggregate **by date group**; add short-over-long ratios for regime shift.
Benchmark any model against a drift-adjusted line, not the bare line.

**MLB.** Direct, with more meaningful sub-regimes: split by park, by
day/night, and by month (weather-driven scoring drifts hard across April→July).
The 162-game season gives much better-sampled windows.

---

## 10. Calendar and context

**Captures.** Cheap contextual signals with no extra data source.

**Computed.** `IS_WEEKEND_BEFORE`, `MONTH_BEFORE`, `IS_US_HOLIDAY_BEFORE` (via a
federal-holiday calendar computed over the observed date range);
`SAME_CONFERENCE_BEFORE`, `SAME_DIVISION_BEFORE`, per-side conference flags;
`PLAYOFF_GAMES_LAST_SEASON` (playoff game count from the *previous* season,
shifted by one season); `IS_PLAYOFF_GAME_BEFORE`; overtime history
(`IS_OVERTIME_LAST_GAME_BEFORE`, 5-game and season overtime frequency, all built
by an explicit forward walk so a row without an outcome receives history without
entering it); pre-game win/loss record and win rate.

**Predictive because.** Rivalry and divisional games differ in intensity; national
TV windows change officiating and pace; last season's playoff depth is a
persistence prior at a season's start when nothing else has data.

**Sport-agnostic core.** Cheap calendar flags; a *structural* rivalry flag from a
static map; a previous-season achievement prior for early-season rows; a
prior-outcome-frequency history built without letting the current row enter it.

**MLB.** Add **day game after night game**, **series position**, **getaway day**,
**interleague**, and **DH-rule regime** for historical seasons. `MONTH` is much
more informative than in the NBA because it proxies weather.

---

## 11. Prior-game line dynamics

**Captures.** How much the market **re-priced** this team's previous games — a
measure of market *uncertainty* about the team, not of direction.

**Computed.** For each finished game, the complete open-to-close tick history is
summarised per book and aggregated across books with a **median** (so one
densely-ticking book does not dominate): number of level moves, absolute total
movement, signed open-to-close move, count of price-only re-prices, and the
path's standard deviation. Those per-game figures are then rolled forward over
each team's last 5/10/20 games, for **all three markets separately** — how a
team's spread gets re-priced is different information from how its totals do.
Emitted with `_TEAM_HOME` / `_TEAM_AWAY` and `_DIFF_BEFORE`.

**Predictive because.** A team whose lines get re-priced repeatedly is one the
market keeps changing its mind about. That is information about *uncertainty*,
which is exactly what a threshold-on-a-distribution bet cares about — and it is
orthogonal to the size or direction of any single move.

**Temporal correctness.** These games **finished before the snapshot**, so their
complete history *including their closing line* was fully known at prediction
time. This is the one place a closing line may legitimately be used, and saying so
explicitly is what keeps it from being mistaken for a leak.

**Sport-agnostic core.** Summarise *finished* games' full market paths and roll
them forward per team, per market. Median across books. Use the count of
re-pricings as an uncertainty proxy, distinct from movement magnitude.

**MLB.** Direct. Additionally informative because MLB lines move sharply on
lineup and pitcher news — a team whose totals re-price heavily is often one with
an unsettled rotation or lineup, which is itself predictive.
