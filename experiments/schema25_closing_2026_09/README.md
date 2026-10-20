# Schema 2.5 closing-line campaign

The first training runs on `training_data_2_5_20260704.csv` (rebuilt
2026-09-17, pinned to `sha256:239e5cd606ca7d78`). Schema 2.5 adds injury-status
tiers, referee tendencies and BetRivers (125 columns) on top of 2.3. It
predicts all three closing-line targets with the full 2.5 feature set. The
referee-column ablation is a separate campaign, `referee_tendencies_2026_09`.

## Design: 3 targets x 3 CV protocols

| cell | CV folds | boosting rounds | reads against |
|---|---|---|---|
| `*_a` | 12 x 50 games, 60-game step | fold-local early stopping | the matching `fixed50_optuna_2_3_2026_09` cell (data change) |
| `*_b` | 6 x 100 games, 120-game step | fold-local early stopping | `*_a` (fold geometry only) |
| `*_c` | 6 x 100 games, 120-game step | tuned once per trial, fixed everywhere | `*_b` (rounds only) |

Targets: `s_` spread error (4,500 training games), `l_` line error (6,200),
`t_` total points against bet365 (4,000).

Why these two protocol changes:

- **6 x 100 folds** led on schema 2.3 for both totals targets (line error CV
  49.6% -> 53.4%, total points 52.8% -> 53.9%). It was never run for spread.
- **Tuned rounds** remove a known mismatch. With early stopping, each Optuna
  fold stops at its own round, while the refit uses the median. In the first
  2.3 line-error run, directional accuracy was 57.3% inside the trial but 49.6%
  when CV was replayed at the final round count. In `n_estimators_probe_2026_09`,
  tuned rounds beat early stopping on both legacy pairs (line CV 50.7% -> 54.0%,
  total 50.5% -> 52.5%). Each of those is a single run, so this campaign
  retests the change on current data.

Held fixed in all 9 cells: seed 16 plus `evaluation_seeds: [101, 202]`; seasons
from 2019; regular season plus play-in; overtime included; `max_na_per_row:
300`; correlation pruning at 0.95 (0.99 for `ODDS_`); 150 Optuna trials with no
timeout; pooled MAE objective; lexicographic selection (top 15%, cap 0.04);
90-day daily walk-forward holdout; bet threshold |edge| >= 0.1.

## Feasibility (measured by the preflight)

At `max_na_per_row: 300`: 8,232 rows after cleaning (the 78 dropped are all
Oct 2019 to Jan 2020, plus 2 in Dec 2020). That leaves 7,622 dev rows and a
610-game holdout. Every fold of every cell reaches its training window.

Line error runs at **6,200 instead of the 2.3 value of 6,250**. On 2.5, 50
fewer rows survive 300 NaNs than on 2.3 (8,232 vs 8,282), so 6,250 did not
fit the earliest fold (6,238 at 12 x 50, 6,216 at 6 x 100). That fold would
have trained on fewer games without raising an error. The line-error `a` cell
therefore differs from 2.3 in two things, but dropping 50 of the oldest games
is small next to the data change.

## Pre-registered reading

Decide in this order, and stop at the first failure:

1. **Seed noise.** A difference smaller than a cell's own `seed_roi_range` is
   not a result.
2. **Volume.** Break-even is 52.38%. Report the Wilson 95% interval for every
   win rate you cite.
3. **CV vs holdout agreement.** Rank on `cv_win_rate`. Use the holdout to
   estimate, not to rank. A large gap in either direction is a warning.
4. **Legitimacy.** Compare only at the common threshold (re-score in
   `survey_experiments.ipynb` with `RESCORE_EDGE_THRESHOLD = 0.1`). Ignore
   rows scored against `*_consensus_opener`: the model sees the closing line,
   so opener wins cannot be realised.

Expectations: `l_b` and `t_b` should reproduce the 2.3 ordering (b >= a).
`*_c` should close the gap between trial CV and replayed CV. Spread is not
expected to clear break-even.

Multiple-comparisons budget: 9 cells, so expect about one to look good by luck.
A single cell beating its neighbour is a hypothesis for the next campaign, not
a winner.

## Known caveats

- **Reused holdout.** The 90-day holdout covers the same calendar window that
  ~20 schema-2.3 runs have already been scored on, so it is no longer clean for
  selection. Use it to confirm, not to choose. 2026-27 in shadow mode is the
  first clean forward test.
- **Bet365 anchors.** The bet365 spread anchor disagrees with the peer-book
  median by more than 1 point on about 7.7% of rows, and the totals anchor has
  the same pattern. This was unfixed as of 2026-09-06, and this campaign does
  not change it. It corrupts part of the spread target and baseline.
- **Checksum.** Rebuilding the CSV changes the checksum. Every cell will then
  refuse to run until `expected_checksum` is updated. This is intended.

## Run

```bash
poetry run python scripts/preflight_campaign.py experiments/schema25_closing_2026_09
bash experiments/runners/run_schema25_closing_2026_09_part1_totals.sh   # l_a..l_c, t_a..t_c
bash experiments/runners/run_schema25_closing_2026_09_part2_spread.sh   # s_a..s_c
```

The two parts are independent and can run on separate GPUs. To resume without
repeating finished cells, prefix either command with `SKIP_EXISTING=1`. All
cells use `--no-save-model`: this is an evaluation campaign, not a model
promotion.

Each cell takes roughly 1.5-2 h for 150 trials on 2.3, plus the two
evaluation-seed replays of the holdout. Part 1 is about 12 h; part 2 is about
6 h.
