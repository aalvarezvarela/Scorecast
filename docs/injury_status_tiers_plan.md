# Injury report statuses in the closing-line dataset

Branch: `lab/injury-status-tiers` (from `lab/injury-report-intermediate-times`).
Status (2026-09-16): **implemented**, and the database issues in §7 are fixed.
Schema 2_4 (status block) is built and measured in §8; schema 2_5 (three
availability groups, §3.1) is coded and unit-tested but **not yet built** -- no
dataset on disk carries it. The experiments in §5 Phase 4 have not been run.

**Scope.** Feature engineering for the closing-line training dataset only.
Production parity and intermediate (multi-horizon) features come later.

---

## 1. Rules

1. **One snapshot per game: the last report published before tipoff.** No
   horizon parameter. The read is `valid_from < tipoff_utc` and
   `valid_to >= tipoff_utc`, so it stays strictly before tip.
2. **Three availability groups, not two** (revised 2026-09-16): **injured** =
   Out ∪ Doubtful, **questionable** = Questionable, **available** = Probable,
   Available and anyone unlisted. They partition a covered roster. Questionable
   was folded into the out set until 2_4; it plays 62% of the time at the last
   report, so subtracting it as an absence was wrong in both directions.
3. **Counters per report category.**
4. **No report timeline → exactly the current (2_3) behaviour.** This applies
   when the database has no report for that game (2017-18, 2018-19, gaps), or
   the team had not filed before tip. For that team-game:
   - the out set is the existing inactive list ∪ comment regex, so every
     existing family comes out identical to 2_3;
   - only the out and available counters are filled;
   - the category counters, the questionable-group columns and every
     status-block feature are NaN;
   - `INJURY_REPORT_COVERED = 0`.

   A test pins this: rebuilding a legacy-only window must reproduce the 2_3
   values column for column.
5. **Per-status history features** for Questionable and Probable, from history
   only (strictly earlier dates). Doubtful was added and then removed on
   2026-09-15: it plays ~2% of the time, so its columns were noise. It keeps
   its counter.
   - the player's chance of playing under that status;
   - the league-wide chance over the current + previous season;
   - the player's expected change in minutes, points and pace when they play
     under that status.
6. **Two injury dicts, not one.**
   - `pregame_out_dict`: rules 1-4, for the game being built.
   - `realized_absence_dict`: the existing inactive list ∪ comment regex, for
     *past* games (streaks, the availability-effect history, the
     available/injured history map, lagged roster continuity).

   Rule 6 is required by rule 2. If the effect history used the pre-game out
   set, every game a Questionable player actually played would be recorded as a
   game they missed.

---

## 2. What the data says about these rules

Aiven `injury_report` 2019-20..2025-26, regular season + play-in, box scores
from Supabase. "Played" = minutes > 0.

### 2.1 Folding Questionable into out

At the last report before tip, rotation players (last-10 >= 20 min):

| Status | Listings | Played |
|---|---:|---:|
| Out | 31,058 | 0.0% |
| Doubtful | 268 | 3.0% |
| Questionable | 1,984 | 61.9% |
| Probable | 949 | 96.1% |

Rotation Questionables appear in 9% of team-games at the last report (4% for
Probable). Doubtful → out is clearly right. Questionable → out puts a group that
**mostly plays** into the out-set features, which is why it is no longer done:
as of schema 2_5 Questionable is its own availability group (rule 2, §3.1), and
E3 -- the experiment that was going to test folding it in -- is settled by
construction. The status block (§3.3) stays, and answers the separate question
of what the status implies for that player.

### 2.2 Chance of playing when Questionable

Test set: players Questionable on the last report, 2023-25 (625 listings,
56.6% played). Every estimate uses only earlier dates.

| Estimator | Log-loss | AUC |
|---|---:|---:|
| constant 0.5 | 0.693 | 0.50 |
| league rate, last-report Questionables, current + previous season | 0.686 | 0.47 |
| player rate from their **last-report** Questionable history, shrunk (k=5) | 0.623 | 0.68 |
| player rate from **every game they were Questionable at any point that day**, shrunk toward the league rate (k=2-5) | **0.608** | **0.71** |
| logistic: league rate + player rates + player's last-10 minutes | 0.567 | 0.76 |

History available for a last-report Questionable (2021-25):

| Player history source | ≥1 prior listing | ≥5 prior | Median |
|---|---:|---:|---:|
| last-report Questionable only, cur+prev season | 75% | 30% | 2 |
| any-time-that-day Questionable, cur+prev season | 96% | 74% | 9 |
| any-time-that-day Questionable, all seasons | 98% | 83% | 17 |

- **The player's own history carries the signal.** The league rate is a good
  prior but does not separate players (AUC below 0.5 is just drift around a
  near-constant).
- **Build the player history from every day the player was Questionable**, not
  only last-report listings. It has 3-8x the samples and predicts better, even
  though its base rate is lower (46.8% vs 54.8%). The shrinkage prior (the
  league last-report rate) corrects that level.
- Adding the player's minutes role helps a lot (log-loss 0.567). Big-minute
  players play through Questionable more often.

### 2.3 Effect when a Questionable plays

Players who played (last-10 >= 10 min). Each value is the change versus the
player's own last-10 average, minus the same change for unlisted players:

| Sample | n | Minutes | Points | Pace |
|---|---:|---:|---:|---:|
| Questionable any time that day, played | 6,441 | -1.42 ± 0.16 | -0.69 ± 0.16 | +0.24 ± 0.14 |
| Questionable at last report, played | 1,428 | -1.43 ± 0.34 | -0.79 ± 0.36 | +0.08 ± 0.28 |

Same "any time that day" sample, split by the player's last-10 minutes:

| Last-10 min | n | Minutes | Points | Pace |
|---|---:|---:|---:|---:|
| 10-20 | 1,183 | -0.34 | -0.21 | +0.29 |
| 20-28 | 1,982 | -1.51 | -0.43 | +0.26 |
| 28-34 | 2,110 | -1.75 | -0.74 | +0.13 |
| 34+ | 1,166 | -1.77 | -1.53 | +0.34 |

**Is the effect player-specific?** Walk-forward test: regress each Questionable
game's change on the player's own earlier Questionable average, shrunk by a
between-player variance estimate.

| Stat | Player variance / game noise | Implied k | Slope on shrunk prior | t |
|---|---:|---:|---:|---:|
| Minutes | 1.40 / 44.3 | ~32 | +0.66 | **+4.6** |
| Points | 0.73 / 45.2 | ~62 | +0.51 | +1.5 |
| Pace | 0.26 / 32.3 | ~123 | +0.92 | +2.8 (raw mean: +0.8) |

- **Minutes:** real player differences, but a player needs ~30 Questionable
  games before their own history counts as much as the prior.
- **Points:** mostly role (who the player is), little player-specific signal.
- **Pace:** essentially none. Still built as asked, since it is cheap, but
  expect it to be flat.
- The prior should be the **role-bucket league effect**, not a single
  league-wide number.

### 2.4 What to expect

- The closing line already moves on Questionable resolutions (t -12 to -34 for
  the line move T-60 → close). What happens to the Questionable player does not
  predict the error against the close.
- The leftover correlation between missing scoring and the closing-line error is
  small (t ≈ 2.5-3).
- One 609-game holdout has a ±4-point confidence interval on win rate, so a
  0.5% gain cannot be seen there. §5 says how to measure it.

---

## 3. Features

Per team row, before the home/away merge. Names follow the existing
`_BEFORE` convention (`_BEFORE_TEAM_HOME`/`_AWAY` after merging).

### 3.1 The three availability groups

The existing out-player families keep their names and their code. Only their
input changes to `pregame_out_dict`, now Out ∪ Doubtful:

- `TOP{1..4}_INJURED_PLAYER_{stat}`, `AVG_INJURED_*`, `TOTAL_INJURED_PLAYER_*`,
  `N_INJURED_PLAYERS`, `ALL_STAR_*_INJURED_*`, `INJURY_PTS_SHARE`
- the available side (`TOP{i}_PLAYER_*`, bench)
- `TOP{i}_INJURED_STREAK_PTS`: the current game from `pregame_out_dict`,
  earlier games from `realized_absence_dict`
- `TOP3_*AVAILABILITY_EFFECT_*`: which players, from `pregame_out_dict`; their
  present/absent history, from `realized_absence_dict`

**The questionable group gets the same families, computed independently**
(schema 2_5). Nothing about the estimators changes -- only which players they
are pointed at, because every value involved is a property of the player, not
of the group he is in tonight:

| Injured family | Questionable twin | Slots |
|---|---|---|
| `TOP{i}_INJURED_PLAYER_<stat>` | `TOP{i}_QUESTIONABLE_PLAYER_<stat>` | 2 (`N_TOP_PLAYERS_QUESTIONABLE`) |
| `AVG_INJURED_<stat>` | `AVG_QUESTIONABLE_<stat>` | top N |
| `TOTAL_INJURED_PLAYER_<stat>` | `TOTAL_QUESTIONABLE_PLAYER_<stat>` | all members |
| `N_INJURED_PLAYERS` | `N_QUESTIONABLE_PLAYERS` | -- |
| `TOP{i}_INJURED_STREAK_PTS` | `TOP{i}_QUESTIONABLE_STREAK_PTS` | 2 |
| `ALL_STAR_{MAX_INJURED_FAN_VOTE_SHARE,MIN_INJURED_SCORE}` | `ALL_STAR_{MAX_QUESTIONABLE_FAN_VOTE_SHARE,MIN_QUESTIONABLE_SCORE}` | -- |
| `TOP3_INJURED_AVAILABILITY_EFFECT_*` | `TOP2_QUESTIONABLE_AVAILABILITY_EFFECT_*` | 2 PTS + 1 MIN ids |

Three details that are not mechanical:

- **Group membership is read with the `injured=True` branch** of
  `get_top_n_averages_with_names`, i.e. each player's last game *strictly
  before* this one. The available branch would take a same-day row, which
  exists only if the player ended up playing -- the rotation-depth leak
  recorded in `config/leakage.py`.
- **The streak needs its own lookup.** `create_injury_streak_lookup` walks back
  from the current game and stops at the first game the player is not in the
  set; a Questionable player is not in tonight's out set, so the injured streak
  would read 0 for him always. Building a second lookup with the questionable
  set as the current game makes the column mean what its injured twin means:
  consecutive team games, this one included, of being unavailable or at risk.
- **Uncovered team-games have no questionable group**, so every column above is
  NaN there (rule 4), never 0. The G League filter is *not* applied to group
  membership, only to the counters, because the legacy inactive list carries
  those players too.
- **The availability-effect builder is coverage-blind** and fills a missing
  aggregate with 0 — the correct limit for an estimator that shrinks toward
  zero, but a lie on a team-game where nobody knows who is at risk. So
  `mask_uncovered_group_columns` NaNs the whole
  `TOP2_QUESTIONABLE_AVAILABILITY_EFFECT_{SIDE}_*` block per side after the call.
  The injured block is never masked: it predates the report and the 2_3 control
  has to reproduce it.
- **`ALL_STAR_MIN_QUESTIONABLE_SCORE` needs a neutral value, not NaN.** The
  score is rank-like (1 = biggest star), so a minimum over an empty group is
  really +infinity; `NO_CANDIDATE_IN_GROUP_SCORE = 1000` stands in for it, above
  every score the voting table produces (max observed 178.8). A per-season
  maximum would be wrong: one season's "nobody" would rank as a bigger star
  than another season's real player. Without this the column would be NaN on the
  ~87% of team-games with nobody Questionable and the cleaner would drop it at
  the 5% threshold. Its injured twin is left alone, NaN and all (6.2% missing
  today), because the 2_3 control reproduces it.

### 3.2 Counters

| Column | Report covers the team | No report timeline (rule 4, legacy) |
|---|---|---|
| `N_INJURED_PLAYERS` (existing) = out set on roster | O+D+Q | inactive list ∪ comment regex, as in 2_3 |
| `N_AVAILABLE_ROSTER_PLAYERS` | ✓ | ✓ |
| `N_REPORT_OUT_PLAYERS` | ✓ | NaN |
| `N_REPORT_DOUBTFUL_PLAYERS` | ✓ | NaN |
| `N_REPORT_QUESTIONABLE_PLAYERS` | ✓ | NaN |
| `N_REPORT_PROBABLE_PLAYERS` | ✓ | NaN |
| `INJURY_REPORT_COVERED` | 1 | 0 |
| `LAST_STATUS_REPORT_AGE_MIN` (§7 finding 2) | ✓ | NaN |

- **`N_REPORT_*`** count the report's listings for the team, **excluding the
  `g_league` category**. Two-way and assignment players are roster mechanics
  (30% of Out rows). They stay in the out set, as they already are in the
  inactive list.
- **`N_AVAILABLE_ROSTER_PLAYERS`** counts players on the available side of the
  split who logged minutes in the 30 days before the game. Without the recency
  window, waived players would stay "on the roster" all season.
  - It never reads the target game's box score.
  - Its name must not start with `N_ACTIVE_PLAYERS`, the rotation-leak guard in
    `config/leakage.py`.
- **Report-derived names avoid `injury_`/`injured_`.** The missing-data policy
  zero-fills those, which would turn "no report" into "fresh report, nobody
  listed". `INJURY_REPORT_COVERED` is never NaN, so its name is safe.

### 3.3 Listed-status features (Questionable, Probable, Doubtful)

Everything below applies per status, with `QUESTIONABLE` replaced by `PROBABLE`
or `DOUBTFUL`. Default per-player slots are Questionable 2, Probable 1 and
Doubtful 1 (`--n-top-questionable/--n-top-probable/--n-top-doubtful`). Each
status uses only its own history for `P_PLAY` and for the player's effect. The
effect prior falls back from that status and role, to that status, to both
statuses and role, to both.

**Doubtful gets the form columns and nothing estimated.** Since 2021 only 3
players Doubtful at the last report played, so a `P_PLAY` or an effect fitted
for it would be noise — but *who* is Doubtful is a fact known before tip, and
`TOP1_DOUBTFUL_FORM_{MIN,PTS,PACE_PER40}` says how big that player is. It stays
in the out set and in `N_REPORT_DOUBTFUL_PLAYERS`. Read the form columns as the
size of the loss and the counter as whether there is one at all: an empty slot
and a player with no form both read 0.

The listed players on the team are ranked by **minutes form for Questionable
and Doubtful** and by points form for Probable (`RANK_STAT`), **excluding G
League reason categories** — the same filter the counters use (§3.2), so a slot is
filled exactly when `N_REPORT_<STATUS>_PLAYERS > 0`. (A two-way or assignment
listing stays in the out set, as the legacy inactive list carries it, but it is
not a player at risk.) The top `N_TOP_<STATUS>` get per-player columns. One
ranking per status fills the slot, and every column of that slot then describes
that one player — unlike `TOP{i}_INJURED_PLAYER_<stat>`, which re-ranks per
statistic and can name a different player in each.

Why minutes for the out-set statuses (decided 2026-09-16): the slot exists to
say how much of the rotation is at risk, which minutes measure directly, while
points confound role with scoring rate. The key only decides anything when a
team lists more players of a status than it has slots — 17.8% of Doubtful
team-games and 8.5% of Questionable ones — and there the two keys name a
different player in 18% (Doubtful) and 24% (Questionable) of cases, the
minutes-ranked pick averaging +4 form minutes (max +15). The model is not blind
to scoring either way: `FORM_PTS` of whoever fills the slot is published, and
the counters and `SUM_*` aggregates cover every listed player. Questionable defaults to **2**; 1 is
also acceptable, and experiment E2-top1 (§5) compares them. Two covers almost
every row, since a third Questionable rotation player is rare.

| Column | Meaning |
|---|---|
| `TOP{i}_QUESTIONABLE_FORM_{MIN,PTS,PACE_PER40}` | the player's own pre-game form (§4.3) — the only block Doubtful gets |
| `TOP{i}_QUESTIONABLE_P_PLAY` | player's shrunk chance of playing (§4.1) |
| `TOP{i}_QUESTIONABLE_N_HISTORY` | player's prior Questionable days |
| `TOP{i}_QUESTIONABLE_EFFECT_{MIN,PTS,PACE_PER40}` | expected change if they play (§4.2) |
| `SUM_QUESTIONABLE_EXP_PLAYERS` | Σ p |
| `SUM_QUESTIONABLE_EXP_{MIN,PTS}` | Σ p·(form + effect), the expected production recovered from the out set |
| `MEAN_QUESTIONABLE_P_PLAY` | mean p |
| `LEAGUE_QUESTIONABLE_P_PLAY` | league rate, current + previous season (per game, emitted once, not per side) |

The sums and the mean cover **all** of the team's Questionable players, not
only the top N.

- **Report covers the team, empty slot or no Questionables:** sums 0. Empty
  per-player slots take the neutral "nobody at risk" values: `P_PLAY = 1`,
  `N_HISTORY = 0`, effects and forms 0; `MEAN_QUESTIONABLE_P_PLAY = 1`.
  - They must not be NaN. About 90% of team-games have no Questionable player,
    and the training cleaner drops columns above `cleaning.nan_threshold` (5%
    by default). The first 2_4 build left them NaN, and 11 of the 12
    Questionable columns were dropped before training.
- **No report timeline:** everything NaN (rule 4).

New columns per side: 7 counters/flags, then per status with history 8·N_TOP
(5 estimates + 3 forms) + 4 aggregates, and 3·N_TOP for Doubtful. The league
rate and report age are the same for both sides, and the home/away merge
de-duplicates such columns.

---

## 4. Computations (all strictly earlier dates)

### 4.1 `P_PLAY`

Event table: every (game, player) with a pre-tip Questionable span on any
report, and whether the player played. Regular season, play-in and playoffs,
from the first loaded report (2018-12-17). It is independent of the dataset's
own season window.

- `league_rate(date)` = played / listings over **last-report** Questionables in
  the current and previous season, before `date`. With fewer than 200 listings
  in that window it falls back to all earlier last-report listings, then to 0.5.
- `player_rate(date)` = `(plays + k·league_rate) / (listings + k)` over the
  player's **any-time-that-day** Questionable events, all seasons, before
  `date`. Start with k = 3; choose k on 2019-2022 log-loss only.
- A calibrated logistic over league rate, player rate and last-10 minutes (§2.2)
  is a later option. Adopt it only if it beats the shrunk rate on the next
  season.

### 4.2 `QUESTIONABLE_EFFECT_{MIN,PTS,PACE}`

For every event where the player **played**: `d = stat − form − drift`.

- **Form:** EWMA (half-life 10 games, as in the TOP-N features) per
  player-season, over played games strictly before the date. With no game yet
  this season, it is the previous regular-season mean.
- **Drift:** the expanding mean of `stat − form` for unlisted players with form
  ≥ 10 minutes, over report-covered games only.
- **Pace:** `PACE_PER40`, from games with ≥ 8 minutes.

Shrinkage:

- `prior(role, date)` = expanding mean of `d` over earlier Questionable-played
  events in the same role.
  - Roles are form minutes <10 / 10-20 / 20-28 / 28-34 / 34+.
  - A role with fewer than 30 events uses all roles; with no events at all the
    prior is 0.
- `effect(player, date)` = `(Σd + k·prior) / (n + k)`.
  - k = σ²/τ² per stat, refitted per season on earlier seasons only, from
    players with ≥ 5 events, clipped to [5, 500].
  - With fewer than 30 such players the defaults apply: 30 minutes, 60 points,
    120 pace.

### 4.3 `<STATUS>_FORM_{MIN,PTS,PACE_PER40}`

The same form the effect is measured against in §4.2, published per slot: the
player's EWMA (half-life 10 games) over played games strictly before the date,
falling back to their previous regular-season mean, then to 0. Pace uses games
of ≥ 8 minutes only.

It is the one block every listed status gets, Doubtful included, because it
needs no history of the status itself — only the player's own box scores. For
Questionable and Probable it separates the two things `SUM_<STATUS>_EXP_MIN`
multiplies together: how likely the player is to play, and how much he is worth.

### 4.4 Artifact

Not built. The events and estimates are recomputed from Aiven and Supabase on
every dataset build.

---

## 5. Implementation

### Phase 0 — after the database fixes
- Port the §2 analysis to `scripts/injury_reports/analyze_questionable.py` and
  rerun it on the fixed database. Update §2 if anything moved, especially:
  - P(play) by status;
  - the P_PLAY log-loss table;
  - the effect sizes and fitted k;
  - how many team-games fall back to the legacy rule.

### Phase 1 — data layer (implemented)
- `postgre_db/injury_report_aiven/fetch.py`: `last_status_before_tip`,
  `filing_at_tip`, `report_age_at_tip`, `questionable_events`, `listed_pairs`.
- `data_processing/injury_status/report_state.py`:
  - `InjuryReportState` and `load_injury_report_state`;
  - `report_out_overrides` (rules 2-4);
  - `apply_report_out_overrides`;
  - `report_counter_features`.
- `data_processing/injury_status/questionable.py`: form, events, `P_PLAY`
  (§4.1), effects (§4.2), `add_questionable_features`.
- `data_processing/injury_status/features.py`: `add_injury_report_features`.
- `get_injured_players_dict` stays the realized-absence builder. Its G League
  comment is corrected. The regex gaps are left alone, to keep legacy rows
  identical to 2_3.

### Phase 2 — pipeline wiring (implemented)
- `create_df_to_predict`: build both dicts. Pass `pregame_out_dict` to
  `add_player_history_features` and `add_all_star_voting_features`, and
  `realized_absence_dict` to `add_roster_continuity_feature`.
- `add_player_history_features`:
  - the roster split uses `pregame_out_dict`;
  - it returns the available/injured history map built from
    `realized_absence_dict`, which feeds `availability_dict` in the effect
    functions;
  - it emits the §3.2 counters.
- `create_injury_streak_lookup`: takes both dicts (current game vs. history).
- New `add_questionable_features(df_team, pregame_status, questionable_history)`
  right after player features, for §3.3.
- Check that the new columns survive `select_training_columns` and
  `clean_df_for_training`.
- `config/dataset_versions.py`: `2_4` entry.

### Phase 3 — tests (implemented: `tests/test_injury_report_status_features.py`)
- **Truncation:** delete all spans with `valid_from >= tipoff` and all events
  dated on/after the game. Rebuild. Identical output.
- **Expanding fits:** changing a future game's outcome leaves earlier rows'
  `P_PLAY` and effects unchanged.
- **Legacy fallback:** no report timeline → category counters and Questionable
  features NaN, `INJURY_REPORT_COVERED=0`. Every pre-existing column equals the
  2_3 build for those team-games. Check it on 2017-18 (no PDFs at all) against
  `training_data_2_3_20260909.csv`.
- **History split:** a Questionable who played is not an absent game in the
  effect or streak history.
- **Sign symmetry** (as for the rotation leak): home and away Questionable
  features must relate to the spread residual with opposite signs.

### Phase 4 — experiments

| Cell | Dataset |
|---|---|
| E0 | `training_data_2_3_<date>_rebuild.csv` (`--no-injury-report-features`): the 2_3 columns and out sets, verified against the original 2_3 file (§8) |
| E1 | out set from the report (O+D+Q) + counters, no Questionable features |
| E2 | E1 + Questionable features, top-2 (the proposal) |
| E2-top1 | E2 with `N_TOP_QUESTIONABLE = 1` |
| E3 | superseded: Questionable is now its own availability group, not a member of either side. The comparison worth running instead is **E4** |
| E4 | schema 2_5: three availability groups, the questionable group carrying the injured group's whole family (§3.1). Compare against E2 (same status block, Questionable folded into out) to price the split itself |

- Models: line_error 6-fold×100-game, total_points, spread cells.
- ≥5 seeds each, same holdout, edge threshold fixed beforehand.
- A 0.5% gain is below what one holdout can show. Judge on:
  - paired per-fold CV MAE and win rate, E2 vs E1, across all seeds and folds;
  - feature-level checks on E2: permutation importance of the Questionable
    block, and the partial dependence of the error on `SUM_QUESTIONABLE_EXP_PTS`.

---

## 6. Decisions

Decided (2026-09-15):
- Last report before tipoff, whatever its age.
- Out set = Out ∪ Doubtful ∪ Questionable.
- No report timeline for a team-game → the 2_3 behaviour.
- Top-2 Questionable players per side (top-1 acceptable; E2-top1 compares).
- Closing-line dataset only; production later.

Decided (2026-09-16):
- **Three availability groups**: injured = Out ∪ Doubtful, questionable,
  available. The questionable group receives the injured group's whole family of
  columns (§3.1), computed independently. Schema 2_5.
- The status block (probabilities and effects) and the availability-group
  families are kept separate; a Questionable player carries both.
- No `P_PLAY` and no effect for Doubtful — measured as noise (§ results).
- Every listed status, Doubtful included, still publishes the slot player's
  own form: `TOP{i}_<STATUS>_FORM_{MIN,PTS,PACE_PER40}`, top-1 for Doubtful.
  The point is to say how big the player at risk is, which needs no history of
  the status.

Default unless changed:
- Player P_PLAY and effect histories count every game the player was
  Questionable at any point that day (§2.2-2.3), not only last-report
  listings.

---

## 7. Database audit (2026-09-15, after `a024a24`)

Aiven `injury_report`, 276 MB:
- 44,633 reports, all `parse_ok`;
- 10,847 games;
- 119,031 status spans;
- 39,289 filing spans.

### Passed

| Check | Result |
|---|---|
| Status spans overlapping for the same game + player | 0 |
| Gaps between consecutive spans | 0 |
| Last span not ending exactly at tipoff / ending after it | 0 / 0 |
| `valid_from` ≠ first report's `observed_at`; `mins_to_tip` inconsistent | 0 / 0 |
| Span or filing on a team not in the game | 0 |
| Filing spans overlapping, or past tip | 0 |
| Status span on a team whose filing says "not yet submitted" | 0 |
| `ir_game` date vs box-score date (10,844 games) | 0 disagreements |
| Tipoff vs odds-snapshot tipoff, 2021-22..2024-25 | 100% exact |
| Regular-season/play-in/playoff games in `nba_games` missing from `ir_game` | 0 |
| Listed player's team ≠ box-score team | 0 |

The 16 tipoffs before noon ET are preseason games abroad. The 19 at 23:00 ET are
real 2025-26 late tips that the odds data confirms. Four 2025-26 games disagree
with the odds data; in each, the box-score date matches `ir_game`, so the odds
rows are the stale ones.

### Coverage at tip (regular season + play-in team-games filed)

| Season | Filed at tip | Legacy fallback (rule 4) |
|---|---:|---:|
| 2017-18 | not in DB | all |
| 2018-19 | 64.1% (reports start 2018-12-17) | 884 team-games |
| 2019-20 | 100% | 0 |
| 2020-21 | 99.6% | 9 (report says not yet submitted) |
| 2021-22 .. 2025-26 | 100% (2023-24: 99.9%) | 2 in 2023-24 |

### Status vs box score (last report before tip, regular season + play-in)

P(played) per season, 2018-19 .. 2025-26:

| Status | Range |
|---|---|
| Out | 0.000-0.003 |
| Doubtful | 0-5% |
| Questionable | 47-61% |
| Probable | 86-97% |

The parsed 2018-19 legacy reports look like the other seasons.

Out-but-played: 14-24 per season in the 3/day era, 0-5 later. Nearly all are
G League players listed the previous day and recalled.

### Findings and what was done

1. **Unresolved names.** 39 name/team/season rows, 1,377 report rows (~0.05%).
   Most are players traded or waived onto a team they never played for, so
   there is nothing to resolve. Four are real players on the correct roster and
   need aliases:

   | Report name | Team | Season | Rows | Box score |
   |---|---|---|---:|---|
   | Carrington, Carlton | WAS | 2024-25 | 204 | B. Carrington 1642267 |
   | Green, Jalen | HOU | 2024-25 | 96 | J. Green 1630224 (Jeff Green is also "J. Green" on HOU) |
   | Kanter, Enes | BOS / NYK / POR | 2018-19, 2019-20 | 113 | Enes Kanter Freedom |
   | Louzada Silva, Marcos | NOP | 2020-21 | 16 | Didi Louzada 1629712 |

   **Done.** `resolve.CURATED_PLAYER_ALIASES`, including David Jones García
   (1642357, reported as "Jones, David", UTA 2024-25), is applied last and only
   when the id is on that team's roster that season.
   - 2018-19, 2019-20, 2020-21 and 2024-25 were reloaded with
     `--replace-season`.
   - 477 report rows now resolve through the aliases. Unresolved rows dropped
     from 1,377 to about 900, all traded or waived players.
   - Integrity checks re-run after the reload: still 0 overlaps, 0 gaps, every
     span ends at tip.
   - **Local manifests were stale.** They predated the restamp fix, and a
     reload from them would have dropped the 164 recovered reports. They were
     replaced with the S3 copies before reloading; the old copies are backed
     up outside the repo.
2. **Report age depends on era.** Age of the last report at tip:
   - 2018-2020 (3/day): 50-54% of games are 1-2 h old, ~30% are 2-4 h, and
     up to 2.2% are more than 12 h (no report that day before tip);
   - hourly era: ≤ 60 min, except 8 games on 2021-11-08 and 2024-11-13 where
     an hourly report is missing;
   - 2025-26: 90% ≤ 30 min.

   As a result, the status mix at the last report shifts by era. Questionable
   listings per season: 510-951 in 2018-2020, 236-374 in 2021-2024, 71 in
   2025-26. Doubtful: 115-166, then 7-44. **Proposal:** add
   `INJURY_REPORT_AGE_MIN` (or the report era) as one column so the model can
   tell a stale 3/day-era Questionable from a 15-minute one. That is one extra
   column; it does not change rule 1. **Done:** `LAST_STATUS_REPORT_AGE_MIN`.
3. **Near-duplicate reports.** Three pairs in 2026 are 1-6 minutes apart:
   - 2026-03-08 06:14 / 06:15;
   - 2026-05-05 15:43 / 15:45 and 17:39 / 17:45.

   **Not a bug.** These are late publications: the file labelled 01:00 AM has a
   01:14 header, 11:30 AM has 11:43, 1:30 PM has 13:39. The loader correctly
   places each at its header time.
4. **One real trade mid-report.** Game 0022500732 (2026-02-04): Harden and
   Garland are listed Out on both teams on the same day. The spans hand over
   correctly, and at tip each player is on his new team. No action.

### Change vs the 2026-09-14 snapshot

- 7,899 games are in both, and 50 status rows at tip changed (37 in 2019-20,
  13 in 2025-26):
  - mostly Questionable/Probable → Available or Out;
  - 6 newly Out.
- 1,212 games are new: 788 in 2018-19 and 408 in Oct-Dec 2019 (legacy parser),
  16 on 2025-12-20/21 (crossover fix).
- §2's numbers should barely move, apart from the extra 2018-19 data. Phase 0
  confirms it.

---

## 8. Build and validation (2026-09-15)

### Features on real data (all seasons, 20,456 team-games, ~5 s)

**`P_PLAY` for players Questionable at the last report:**

| Sample | n | Observed | Log-loss | AUC |
|---|---:|---:|---:|---:|
| 2023-25 | 668 | 0.569 | 0.609 | 0.70 |
| 2023-25, constant | 668 | 0.569 | 0.684 | |

This matches §2.2 (0.608 / 0.71).

Calibration by quintile:

| Predicted | Observed |
|---:|---:|
| 0.28 | 0.39 |
| 0.41 | 0.48 |
| 0.49 | 0.50 |
| 0.57 | 0.58 |
| 0.72 | 0.70 |

The bottom quintile is somewhat pessimistic.

**Effect columns vs the realized change** (last-report Questionable players who
played, 2021+):

| Column | n | Corr | Slope |
|---|---:|---:|---:|
| `EFFECT_MIN` | 845 | +0.14 | 0.70 |
| `EFFECT_PTS` | 845 | +0.02 | |
| `EFFECT_PACE_PER40` | 802 | +0.13 | |

Minutes is calibrated; points and pace are close to flat, as §2.3 predicted.

Fitted k per season (2019→2025):

| Stat | k |
|---|---|
| Minutes | 30, 8, 12, 28, 32, 20, 22 |
| Points | 60, 32, 23, 56, 39, 54, 41 |
| Pace | 120, 500, 27, 36, 103, 296, 56 |

### 2_4 vs 2_3 (`training_data_2_3_20260909.csv`)

- 11,543 games in both, 44 columns added, none removed.
- 9,780 games covered on both sides, 1,756 legacy on both sides, 7 mixed.
- **Legacy games:** identical on every shared column except two games
  (2021-05-16, 2023-10-25), and only in availability-effect columns. Those
  games are dated after covered games, and the effect shrinkage is fitted on
  all earlier effects.
- **Covered games:**
  - `N_INJURED_PLAYERS` changed in ~32% of team rows; mean 3.51 → 3.26
    (home), 3.64 → 3.46 (away).
  - The general availability effects change slightly through the same
    shrinkage refit (median shift 0.7% of a standard deviation, correlation
    0.97).
- **Sign symmetry / |margin|:** no new column correlates with |HOME_MARGIN|
  above 0.025. The leaky rotation family was +0.55.
- `N_BENCH_PLAYERS` and `N_AVAILABLE_ROSTER_PLAYERS` are 100% NaN in 2017-18.
  This predates 2_4: it is also NaN in the 2_3 file.

### What survives training cleaning

Probed with the 2_3 line_error follow-up config pointed at the 2_4 file (2019
floor, `nan_threshold` 5%, correlation pruning 0.95). The first build kept 16
of 44 new columns; the per-player Questionable block was lost to the NaN
threshold, which the neutral fill in §3.3 fixes.

After the rebuild (checksum `sha256:ad75125273bd99bf`), **36 of 44 new columns
reach the feature matrix** (1,221 features in total). The 8 dropped are
exact home/away duplicates (`LAST_STATUS_REPORT_AGE_MIN`,
`LEAGUE_QUESTIONABLE_P_PLAY`) or pruned as correlated at 0.95:
`TOP1_QUESTIONABLE_P_PLAY` with `MEAN_QUESTIONABLE_P_PLAY`, and
`SUM_QUESTIONABLE_EXP_{PLAYERS,PTS}` with `SUM_QUESTIONABLE_EXP_MIN`.

### Doubtful and Probable (added after the first build)

Listings on the last report of covered team-games, estimates from earlier
dates only:

| Status | Listings | 2023-25 n | Observed | Predicted | Log-loss (constant) | AUC |
|---|---:|---:|---:|---:|---|---:|
| Questionable | 3,792 | 668 | 0.569 | 0.522 | 0.609 (0.684) | 0.70 |
| Probable | 1,941 | 251 | 0.884 | 0.903 | 0.321 (0.358) | 0.72 |
| Doubtful | 639 | 65 | 0.015 | 0.038 | 0.047 (0.080) | n/a (one player played) |

Effect columns vs the realized change, last-report listings who played, 2021+:

| Status | n | Minutes corr | Points corr | Pace corr |
|---|---:|---:|---:|---:|
| Questionable | 845 | +0.14 | +0.02 | +0.13 |
| Probable | 479 | +0.29 | +0.08 | +0.09 |
| Doubtful | 3 | -- | -- | -- |

- Probable carries real information: chance of playing, and the minutes
  change when a Probable player plays.
- Doubtful carries nothing to estimate (65 test listings, one played), so its
  `P_PLAY`, `N_HISTORY` and effect columns were **removed**. Its counter stays,
  and it keeps `TOP1_DOUBTFUL_FORM_{MIN,PTS,PACE_PER40}`: those are the slot
  player's own box-score form, not an estimate conditioned on the status, so
  the 65-listing sample does not enter them. Expect a sparse column: only
  **2.6%** of covered team-games list a Doubtful player at all (mean 0.03 per
  team-game), so the form slot is the neutral 0 on the other 97.4%.

### Control build (`--no-injury-report-features`)

`N_AVAILABLE_ROSTER_PLAYERS` is now gated with the injury-report switch
(`include_available_roster_count`). The control is written as
`training_data_2_3_<date>_rebuild.csv`, and compared column by column with
`training_data_2_3_20260909.csv`.

Result (`training_data_2_3_20260909_rebuild.csv`, `sha256:1439a54f9b791430`):
- same 11,543 games and the same 1,804 columns;
- **0 differing values** in any column.

Only the column order differed: 108 odds spread/moneyline price columns. Their
books came from `list(set(...))` in `merge_game_df_with_odds_by_game_id.py`,
which follows the per-process hash seed. That is now `sorted(...)`, so future
builds have a stable order.

The build with Doubtful history columns (`sha256:3104c26b7f3fc147`, 1,888
columns) kept 63 of 84 new columns through cleaning. It was superseded by the
Questionable + Probable build.

**Current 2_4 file:** `training_data_2_4_20260909.csv`,
`sha256:2dad49c93b0fbf0d`, 1,892 columns (control 1,804). Supersedes
`sha256:4e331761674fef75` (1,868 columns), which had no `FORM_*` block and no
Doubtful slot.
- Doubtful columns: the two `N_REPORT_DOUBTFUL_PLAYERS` counters and the
  `TOP1_DOUBTFUL_FORM_*` slot.
- With the 2_3 follow-up cleaning config, **66 of 88 new columns** reach the
  feature matrix (1,251 features). 17 of the 24 `FORM_*` columns survive,
  including `TOP1_DOUBTFUL_FORM_{MIN,PACE_PER40}` both sides and
  `TOP1_DOUBTFUL_FORM_PTS` away.
- The 22 dropped, from `cleaning_report.why_dropped`: exact home/away
  duplicates (report age, both league rates), then correlation pruning at 0.95 —
  `TOP1_<STATUS>_P_PLAY` vs `MEAN_<STATUS>_P_PLAY` (0.97-0.99),
  `SUM_QUESTIONABLE_EXP_{PLAYERS,PTS}` vs `SUM_QUESTIONABLE_EXP_MIN` (0.96),
  `SUM_PROBABLE_EXP_{PLAYERS,MIN}` vs `N_REPORT_PROBABLE_PLAYERS` (0.96-0.99),
  and every dropped form column against **its own slot's** `FORM_MIN`
  (0.952-0.958): the pace and points form of one player are nearly the same
  column as his minutes form on a slot that is 0 on most rows.
