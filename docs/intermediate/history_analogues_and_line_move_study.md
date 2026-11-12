# Using history better: analogue games, market memory, line-move target, and what the production recipe earns over four seasons

Status: **Research** (2026-11-11, rerun on the rebuilt data 2026-11-12) — measurements only; nothing here is
implemented in the pipeline. Read this before testing nearest-neighbour /
"similar past games" features, team market-error memory, robust targets or a
line-move model for `line_error` again.

Scripts: `lab/analog_history_2026_11/`. Outputs (git-ignored):
`artifacts/research/analog_history_2026_11/` (`report*.txt`, per-game
predictions `wf_t*.parquet`, `stage{2,3,4}_t*.parquet`, the live-prediction
extract `live_line_error_predictions.parquet`).

## 1. Question

`line_error` models do not predict the score; they predict on which side the
bookmaker's total will be wrong. The question was whether history can be used
more directly for that: find past games that looked like tonight's and read off
how the market moved and how it erred. Follow-ups: a model trained on the
future line move, noise-robust targets, and a check of what the production
recipe earns outside its holdout window and live.

## 2. Summary of findings

| # | Finding | Verdict |
|---|---|---|
| 1 | Analogue (kNN) features do not predict `line_error` and add nothing to the production model | Closed, do not repeat |
| 2 | Analogue forecasts of the line move rediscover the existing Ridge move feature | Closed |
| 3 | Team-level closing-line-error memory adds nothing (the pipeline already has `DIFF_FROM_ODDS_LINE_<book>_*` rolling features) | Closed |
| 4 | "bet365 is away from the other books, bet toward them" is a coin flip at every horizon | Closed |
| 5 | Robust targets (no-OT training, clipping, Huber, sign) are all +0.5 to +1.5pp over base, none outside noise | Weak lead, untested in the real pipeline |
| 6 | A model trained on the line move predicts the outcome slightly but consistently better than the `line_error` model: 50.4–52.1% on all bets, 16 of 16 cells above 50% | The only positive lead; small, near break-even |
| 7 | The production recipe, walked forward over 2022-10..2026-11 with daily refits, wins 50.4% (T-360) and 51.4% (T-720); its 56–58% holdout is a favourable window | Important; changes how holdout numbers should be read |
| 8 | Live 2026-27 `line_error`: 48.1% over all 159 games; 52.7% live / 50.0% replay on the dashboard's default window (from 28 October, 103 games), T-480 the best slot at 58.4% | Consistent with 7; the first week (44.6%) and the weeks since differ by more than any slot does; sample too small to conclude alone |
| 9 | Rerun on the rebuilt data (`20261111`, 176 of 3,080 columns changed): T-720 51.1% vs 51.4%, move model at T-60 unchanged | Conclusions hold |
| 10 | The Ridge move feature alone, bet by its sign, wins 50.4–51.2%; the tree move model adds to it only at T-720 and T-60 | A tree move model is a small step over the Ridge, not a new signal |

Break-even at −110 is 52.38%.

## 3. Method

**Data.** `intermediate_line_data_2_5_20261105.parquet`, one horizon at a
time, prepared with `training_pipeline.data.prepare_dataset` and the
production configs of `experiments/retrain_line_error_2_5_2026_10/` (same
cleaning, NaN budget 800, 2019 floor, no playoffs): about 8,440 games and
2,280 features per horizon (`prep.py`).

**Walk-forward harness.** Test period 2022-10-18 .. 2026-11-04, about 5,000
games per horizon (May and June excluded). Each test month is predicted by a
model fitted on the last 6,225 games strictly before the month, with the
production hyperparameters and tree count of that horizon
(`backtest_summary.json` of the 2026-10 retrain run), no early stopping, mean
of 3–5 seeds. A bet is placed when `|predicted edge| >= 0.1`; pushes are
dropped. Paired differences against the base arm use a bootstrap clustered by
game day.

**Fidelity check.** With daily refits (stage 3) the harness reproduces the
production holdout exactly at T-720 (58.5% vs 58.5% on 2026-01-18..04-17) and
within noise at T-360 (53.0% vs 56.3%, ~570 bets). Monthly instead of daily
refits cost 0.6–0.8pp over the whole period.

**Limits.** Hyperparameters were tuned in 2026-10 on data that includes the
test seasons, which favours the base model if anything. The 2022-23 test
months train on fewer than 6,225 games. Horizons share games, so results
across horizons are not independent replications. About 100 cells were looked
at in total; treat any single z≈2 as unconfirmed.

## 4. Analogue games (kNN)

`feats.py`. For each game, the K=50 and K=200 most similar games dated
strictly before it, same horizon, z-scored with past-only statistics, in three
spaces:

- **MKT**: consensus move from open, recent move, bet365 deviation from
  consensus, cross-book std, steam net, the Ridge expected move, spread move
  from open, |spread|, line level relative to the league's last 30 days.
- **FUND**: season pace, offensive/defensive ratings, last-5 ratings, rest,
  season points, line minus the teams' scoring averages, |spread|.
- **TOP**: the 30 highest-gain features of a model fitted only on pre-2022-10
  games.

From the neighbours: mean `line_error`, over-minus-under rate, mean move from
snapshot to close, distance to the K-th neighbour.

**Model-free** (`direct.py`, test period only):

- Correlation of any neighbour statistic with `line_error`: −0.04 to +0.03.
  Betting the sign of the neighbours' mean error wins 49–51%.
- Neighbour mean move in MKT space predicts the real move (r 0.11–0.17) but is
  the Ridge again: correlation 0.67–0.79 with
  `ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE`, and its residual after
  the Ridge has r −0.016..+0.002 with `line_error`.
- The top-gain features that define the TOP space share almost nothing between
  horizons, i.e. the `line_error` model's importances are mostly noise.

**In the model** (`evalwf.py`, `report.py`; win rate, 5-seed mean):

| Arm | T-720 | T-360 | T-60 |
|---|---|---|---|
| Base (production features) | 50.6% | 49.8% | 51.1% |
| + analogue features | 50.8% | 50.1% | 51.0% |
| + team market memory | 50.7% | 50.3% | 50.9% |
| + both | 50.7% | 50.7% | 51.4% |
| + 38 Gaussian noise columns (control) | 50.5% | 49.7% | 51.2% |
| Only the 38 new features | 49.8% | 50.4% | 49.8% |

Every paired interval against base contains zero (largest: +0.7pp, CI −0.2 to
+1.6). Adding noise columns is indistinguishable from adding analogues. Seed
range is 1–2pp.

**Why it fails.** `line_error` has σ≈18 points; a 200-neighbour mean carries
about 1.3 points of noise, more than any plausible market bias. XGBoost is
already an adaptive neighbourhood method on the same columns, so explicit
neighbours add no information, only variance.

## 5. Team market-error memory and the stale-book rule

- Mean closing-line error of each team's last 5/10/20/40 games (home, away,
  sum): sign bets win 49.5–51.1%; no gain in the model (table above). The
  dataset already has per-book `DIFF_FROM_ODDS_LINE_<book>_*` rolling features.
- Rule "bet365's total is ≥0.5 / ≥1.0 away from the other books: bet toward
  them at bet365", all seasons, 17 horizons: 49–52% at ≥0.5 (n≈2,500–3,500 per
  horizon), no pattern at ≥1.0, seasons from 35% to 54%. The deviation does
  predict bet365's own later move (+0.15 to +0.28 points), which is not enough
  to matter.

## 6. How much is the line move worth

Perfect foresight of the move from snapshot to close, betting the snapshot
line in that direction (test period):

| Horizon | Mean abs. move | Win if move ≥0.5 | ≥1.0 | ≥2.0 |
|---|---|---|---|---|
| T-720 | 1.80 | 55.7% (89% of games) | 57.0% | 58.9% |
| T-360 | 1.11 | 54.2% (78%) | 55.9% | 59.5% |
| T-60 | 0.49 | 54.2% (50%) | 54.4% | 56.7% |

This is a ceiling. The regression slope of `line_error` on the realised move
is about 1.1–1.5 (not distinguishable from 1): a point of anticipated move is
worth roughly a point of edge, about 2.2pp of win rate.

## 7. A model trained on the line move

`stage2.py`, `stage4.py`. Target: closing bet365 total minus the snapshot
bet365 total. All production features, untuned XGBoost (300 trees, lr 0.05,
`min_child_weight` 20, subsample 0.7, colsample 0.3; M1 depth 4, M2 depth 2).
The bet is the snapshot line in the direction of the predicted move, graded
against the real outcome.

| Horizon | r(pred, move) | r(pred, line_error) | Win, all bets (M1 / M2) | Win, top 25% by abs. pred | Win, top 10% |
|---|---|---|---|---|---|
| T-30 | 0.21 | 0.05 | 51.2 / 50.5 | 52.0 / 52.8 | 53.7 / 56.5 |
| T-60 | 0.20 | 0.05 | 52.0 / 51.6 | 53.9 / 56.1 | 57.5 / 58.3 |
| T-120 | 0.14 | 0.05 | 51.1 / 51.6 | 53.1 / 51.4 | 54.2 / 54.0 |
| T-180 | 0.16 | 0.04 | 51.2 / 51.3 | 51.9 / 52.4 | 51.3 / 50.5 |
| T-240 | 0.16 | 0.04 | 51.4 / 51.5 | 51.6 / 53.3 | 50.9 / 51.5 |
| T-360 | 0.11 | 0.02 | 50.4 / 51.0 | 49.9 / 49.9 | 49.5 / 51.3 |
| T-480 | 0.10 | 0.03 | 51.0 / 50.8 | 52.0 / 51.6 | 52.9 / 53.3 |
| T-720 | 0.21 | 0.05 | 51.4 / 52.1 | 52.6 / 52.9 | 55.4 / 55.2 |

(About 5,000 bets in "all", 1,250 in the top 25%, 500 in the top 10%; ±1.4,
±2.8 and ±4.4pp.)

- The move model's correlation with `line_error` (0.02–0.05) is at or above
  the `line_error` model's own (0.00–0.04), and it is positive in every cell.
  All-bets win rate is above 50% in 16 of 16 cells, around 51.2% on average.
- T-60 with depth 2 (56.1%, z=4.3, 12 of 13 quarters above 50%) was the cell
  that prompted the replication. Neighbouring horizons give 52–53%, so read it
  as the upper tail of a real effect of about +1–3pp, not as 56%.
- The size is what the mechanics predict: the bets capture 0.2–0.85 points of
  later move, worth about 0.5–2pp.
- Direction of the move is predictable (64–78% in the top quartile), most of
  it from `ODDS_SNAP_TOT_BET365_DEVIATION_Z` and the Ridge feature; the rest
  of the gain is spread thinly over snapshot velocities and Questionable-player
  features.
- No sign of an in-play leak at T-60: the win rate is 55% on games whose line
  then did not move at all and is not concentrated in large moves.
- Agreement filter: when the `line_error` model and the move model agree the
  bet wins 50.1–52.4%; when they disagree the `line_error` model's side wins
  48.5–49.2% (three horizons).

Not profitable on its own at −110. It is the only direction that showed a
consistent positive sign.

## 8. Noise-robust targets for the `line_error` model

`stage2.py`, production features and hyperparameters, win rate:

| Arm | T-720 | T-360 | T-60 |
|---|---|---|---|
| Base | 50.6% | 49.6% | 50.9% |
| Overtime games excluded from training | 50.8% | 51.2% | 51.3% |
| Target clipped to ±25 | 50.4% | 50.7% | 51.4% |
| Pseudo-Huber (slope 18) | 51.0% | 50.8% | 52.1% |
| Target = 10·sign(error) | 50.7% | 50.5% | 52.1% |

Each is within ±1.4pp of base. Huber and no-OT are above base at all three
horizons; that is a weak lead, and the hyperparameters were tuned for squared
error on the raw target, so the comparison is not clean.

## 9. The production recipe over four seasons

`stage3.py`: daily refit on the last 6,225 games before each game day,
production hyperparameters, 3 seeds.

| | T-720 | T-360 |
|---|---|---|
| Win rate 2022-10..2026-11 (~4,700 bets, ±1.4pp) | 51.4% | 50.4% |
| MAE model / line | 14.41 / 14.39 | 14.37 / 14.35 |
| Production holdout window, this harness | 58.5% | 53.0% |
| Production holdout window, production artifact | 58.5% | 56.3% |
| Season 2022-23 | 48.9% | 48.4% |
| Season 2023-24 | 49.6% | 50.2% |
| Season 2024-25 | 54.4% | 51.3% |
| Season 2025-26 | 53.0% | 51.9% |
| 2026-27 so far (~110 bets) | 46.9% | 47.7% |

- The 90-day holdout used by every 2_5 campaign (2026-01-18..04-17) is a
  favourable window for this recipe. Outside it the same recipe is at 50–51%.
- Filtering by predicted edge does not help (≥1 point: 51.4–52.5%; ≥2: 51.0–51.5%).
- Two fits of the same recipe (daily vs monthly refit) agree on the side in
  only 75–80% of games.
- The last two full seasons are better than the first two, most at T-720
  (53.7% pooled, roughly 51.7–55.7%). Whether that is more and better training
  data or noise is open.

**Consequence for evaluation.** A 90-day holdout of ~600 games cannot rank or
validate these models; it has been reused across campaigns and happens to be a
good stretch. Multi-season walk-forward at fixed hyperparameters costs about
15 minutes per horizon on the GTX 1060 (monthly refit) and should be the first
check for any future "A beats B".

## 10. Live predictions, 2026-27

`prediction.predictions_full`, scheduled runs only, 159 graded games
(2026-10-20..11-10), each registry slot on its own snapshot, one pick per game:

| Target | Picks | Win rate | Slots above 52.4% |
|---|---|---|---|
| line_error | 2,697 | 48.1% | 1 of 17 |
| spread_error | 2,685 | 47.8% | 3 of 17 |
| total_points | 2,697 | 47.0% | 1 of 17 |

`line_error` detail: intermediate slots pooled 48.1%, game-clustered interval
43.5–52.8%; live daily runs 48.9% (all picks), 48.7% (last pick before tip);
majority vote of slots 70–75; random baseline 47.7%, historical-average
baseline 50.4%, TabPFN 45.2% closing and 48.7% intermediate. Slots agree on
the side in 76% of picks, so the effective sample is the 159 games (±8pp). The
models picked OVER 55% of the time; 46% of games went OVER.

### 10.1 The dashboard's Historical page

The dashboard's default window starts on 28 October, not on opening night, and
its "own snapshot" cells show the live daytime runs as well as the end-of-day
replay. Reproduced with the dashboard's own code (`dashboard_check.py`,
`line_error`, XGBoost, minimum edge 0.1, 103 final games, 2026-10-28..11-11):

| Source | Own-snapshot picks | Win rate | Game-clustered interval | Slots above 50% / 52.4% | Best slot | Selection-aware p |
|---|---|---|---|---|---|---|
| Live predictions | 1,399 | 52.7% | 46.6–59.0% | 12 / 9 of 16 | T-480, 58.4% (52–37) | 0.56 |
| Snapshot replay | 1,583 | 50.0% | 44.2–55.6% | 8 / 5 of 16 | T-30, 55.9% (57–45) | 0.66 |

The selection-aware p flips each game's result at random, the same flip for
every slot, and asks how often the best of the 16 slots does at least as well.
A best slot at 58% among 16 slots on 103 shared games is what chance gives more
than half the time.

The first week is what separates this from the 48.1% above: 2026-10-20..27, 56
games, replay 44.6% over 854 picks, with 63% of picks on OVER. T-480 itself
won 37% that week (20–34) and 55% since (54–44): 48.7% over the season. The
walk-forward harness agrees with the live numbers on the same window (T-480
56.7% and T-720 58.6% since 28 October, section 12), so serving matches
training; the open question is only whether three weeks say anything, and at
±6pp they do not.

## 11. What is worth doing next, and what is not

Not worth repeating: analogue/kNN features in any of the three spaces, team
closing-error memory, the bet365-vs-consensus rule, selecting bets by the
`line_error` model's predicted edge.

Open, in order of expected value:

1. **Move-to-close as a first-class target.** Tune it (it was untuned), test
   it as a stacked input to the `line_error` model, and test it for bet
   *timing* (bet now when the line is expected to move against the pick, wait
   otherwise) rather than for side selection. Evaluate on the multi-season
   walk-forward, all 16 horizons.
2. **Robust loss / no-OT training** inside the real pipeline with its own
   tuning, judged on the multi-season walk-forward.
3. **Why 2024-26 looks better than 2022-24** at T-720: training-set size and
   data-era effects, before attributing it to the model.
4. **Line shopping**, which earlier work measured at about +1.5% ROI, remains
   larger than anything found here.

## 12. Rerun on the rebuilt data (2026-11-12)

Sections 4 to 9 used `intermediate_line_data_2_5_20261105.parquet` (partitions
at `9d96300`). Every season partition was rebuilt on 2026-11-11 at `22dd8b8`,
after the season-std and season-opener travel fixes (`defect_fixes_2_5.md`,
items 11 and 12) and the 2023-24 consensus backfill. `export_partitions.py`
writes the current partitions to `intermediate_line_data_2_5_20261111.parquet`
and `diff_builds.py` compares the two files on their 149,075 shared rows:

- 176 of 3,080 columns differ. Every `*_SEASON_BEFORE_STD_*` column differs in
  about 97% of rows, by 0.15 to 0.3 of the column's standard deviation on
  average, and `PTS_FORM_Z_*_VS_SEASON_BEFORE` with them; the consensus pick
  percentage families differ in 15–74% of rows (their season averages most).
- 38 new games (through 2026-11-10); no column added or removed.

Same harness, same production hyperparameters, daily refit (`report_rerun.py`,
outputs in `artifacts/research/analog_history_2026_11/v20261111/`):

| Horizon | Build | 2022-10..2026-11 | 2022 | 2023 | 2024 | 2025 | Holdout window | Since 2026-10-28 |
|---|---|---|---|---|---|---|---|---|
| T-720 | 20261105 | 51.4% (4,674) | 48.9 | 49.6 | 54.4 | 53.0 | 58.5% | 55.7% (61) |
| T-720 | 20261111 | 51.1% (4,693) | 49.2 | 50.0 | 51.9 | 53.1 | 58.5% | 58.6% (99) |
| T-480 | 20261111 | 51.9% (4,658) | 50.8 | 49.5 | 51.7 | 55.7 | 55.5% | 56.7% (97) |

T-720 new minus old, paired on the same bets: −0.11pp, interval −1.32 to
+1.11; the two builds pick the same side in 84% of games (prediction
correlation 0.83). Changing 176 columns moves one pick in six and the win rate
not at all, which is the size of a seed change.

Move-to-close models at T-60, all bets / top quarter by predicted move:
depth 4 51.5% / 56.1% (was 52.0 / 53.9), depth 2 51.6% / 56.0% (was 51.6 /
56.1).

**Ridge alone against the tree move model** (`ridge_vs_model.py`, 20261105
build, same games). Betting the sign of
`ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE`:

| Horizon | Correlation with the real move: Ridge / tree (depth 2) | Win rate, all bets: Ridge / tree | Later move captured by the top quarter (points): Ridge / tree |
|---|---|---|---|
| T-30 | 0.20 / 0.21 | 51.1 / 50.6 | 0.20 / 0.22 |
| T-60 | 0.17 / 0.20 | 51.2 / 51.6 | 0.23 / 0.32 |
| T-120 | 0.14 / 0.15 | 50.7 / 51.6 | 0.29 / 0.32 |
| T-180 | 0.14 / 0.16 | 50.8 / 51.3 | 0.30 / 0.31 |
| T-240 | 0.14 / 0.16 | 51.2 / 51.5 | 0.31 / 0.43 |
| T-360 | 0.12 / 0.11 | 50.4 / 51.0 | 0.36 / 0.27 |
| T-480 | 0.12 / 0.10 | 51.2 / 50.8 | 0.37 / 0.25 |
| T-720 | 0.14 / 0.20 | 51.0 / 52.1 | 0.53 / 0.85 |

The Ridge reads nine line-history inputs of the anchor book, pooled over
horizons. The tree model reads the whole feature matrix per horizon. It
predicts the move clearly better only at T-720 (and somewhat at T-60 and
T-240), and is no better or worse at T-360 and T-480: most of what is
predictable in the move is already in the Ridge. A tree move model is worth
building for the early horizons, where injuries and fundamentals are not yet
in the line, and not as a general replacement.

