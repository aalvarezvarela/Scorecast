# History depth vs row missingness, line error (schema 2.5)

Seven runs: four `(season_year_floor, max_na_per_row)` arms at T-720 and three
at T-60. The question is whether the schema-2.5 intermediate dataset should keep
its 2021 season floor, or reach further back by paying for the older seasons
with a looser per-row NaN budget.

It was designed as eight runs. Arm `d` at T-60 was dropped on 2026-09-22 after
measurement showed it would reproduce arm `a` exactly -- see "What arm `d`
actually measures".

## Why the question exists

The intermediate dataset is one row per (game, snapshot), and about 600-720 of
its columns simply do not exist before 2021-22 -- the book, consensus and
snapshot-market families. Measured on this file at T-0, NaNs per row by season:

| season | games | median NaN | p90 | % over 500 |
| --- | ---: | ---: | ---: | ---: |
| 2019-20 | 1,142 | 737 | 948 | 83.1% |
| 2020-21 | 1,170 | 606 | 703 | **100.0%** |
| 2021-22 | 1,322 | 20 | 26 | 3.2% |
| 2022-23 | 1,320 | 20 | 24 | 0.0% |
| 2023-24 | 1,305 | 34 | 73 | 0.0% |
| 2024-25 | 1,321 | 20 | 24 | 0.0% |
| 2025-26 | 1,322 | 20 | 26 | 0.0% |

This is a wall, not a gradient: a typical modern row misses 20 values, a typical
2020-21 row misses 606. At the current `max_na_per_row: 500` the 2021 floor is
not a conservative choice -- it is the only floor that admits anything, because
100% of 2020-21 rows and 83% of 2019-20 rows exceed the budget on their own.

**Read that table as raw-file counts, not as what the budget does in a run.**
`max_na_per_row` is the last row filter in `clean_dataframe_for_training`: the
NaN-heavy columns are dropped first, and only the surviving columns are counted
per row. A row carrying 606 raw NaNs is far under budget once the 600 columns
that are empty for its whole season are gone. The consequence, measured below,
is that at the 2021 floor the 500 budget turns out to bind on nothing at all --
which is the opposite of what these percentages suggest, and why this campaign's
first attempt mis-sized every training window.

Raising the budget is therefore the only way to reach those seasons short of
`extend_history_dropping_season_gated_columns`, which would drop every column
whose availability identifies the season -- on the order of 600 columns here,
i.e. most of what the intermediate dataset exists to carry. This campaign tests
the cheaper lever instead.

## CV and cleaning

Six latest non-overlapping `test_anchored` folds of 100 games, 120-game step --
the geometry used by `schema25_closing_2026_09` cell `b` and
`intermediate_horizons_2_5_2026_09`, rather than the parent early campaign's
8x80. Correlation pruning keeps the 0.95 general threshold but tightens the
`ODDS_` override from 0.99 to **0.98**, so more near-duplicate odds columns are
pruned. Both are held identical across all seven cells, so neither can explain a
difference between arms; they do mean this campaign is not directly comparable
to the 8x80 / 0.99 runs in the parent campaigns.

## Design: 4 arms x 2 timepoints, 7 cells

| arm | floor | `max_na_per_row` | what it adds | run at |
| --- | ---: | ---: | --- | --- |
| `a` | 2021 | 500 | baseline -- the current production season floor and row budget | both |
| `b` | 2020 | 700 | 2020-21 | both |
| `c` | 2019 | 800 | 2019-20 **and** 2020-21 | both |
| `d` | 2021 | 800 | **nothing, measured** -- intended control | T-720 only |

`b` and `c` each change two things at once, and this is unavoidable: you cannot
admit 2020-21 without raising the budget, since every one of its rows exceeds
500. Arm `d` was meant to make the pair separable: it applies the 800 budget at
an unchanged 2021 floor, so it should admit only the modern rows the 500 budget
currently drops, and whatever column-level cleaning changes follow.

**Measured, it admits almost nothing.** At T-60 arms `a` and `d` clean to the
same 6,165 games; at T-720 `d` admits 13 more than `a`'s 6,004. Since 800 admits
a superset of what 500 admits at the same floor, an equal count is an identical
row set -- and an identical row set means identical correlation pruning, so at
T-60 arm `d` is a bit-for-bit replicate of arm `a` under the shared
`random_state: 16`, not a control. See "What arm `d` actually measures" below.

The decomposition was designed to be read in this order:

- **`d` vs `a`** -- what the looser budget costs or buys with no extra history.
- **`c` vs `d`** -- the history effect, both arms at an 800 budget. This is the
  comparison that actually answers the question.
- **`b` vs `a`** -- the intermediate step, and a check that any effect is
  monotone in how much history is admitted rather than an artefact of one arm.

Since `d` is measurably indistinguishable from `a`, the first step returns
nothing and the second collapses onto `c` vs `a`. Read `c` vs `a` and `b` vs `a`
directly, knowing the budget admits nothing at the 2021 floor.

### Measured feasibility

Not counted from the raw file -- **cleaned** by the real pipeline, one arm per
process, on 2026-09-22. The earlier version of this table counted raw rows and
was wrong in both directions, which is what cost the campaign its first attempt.
`ceiling` is `min_history_games`: the games available before the arm's earliest
CV fold, and the hard maximum for `train_games`.

| arm | horizon | cleaned | dev | ceiling | vs `a` | ladder |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `a` 2021 / 500 | T-60 | 6,165 | 5,555 | 4,149 | -- | 3500 .. 4100 |
| `b` 2020 / 700 | T-60 | 7,250 | 6,640 | 5,234 | +17.6% | 3500 .. 5200 |
| `c` 2019 / 800 | T-60 | 8,301 | 7,691 | 6,285 | +34.6% | 3500 .. 6250 |
| `d` 2021 / 800 | T-60 | 6,165 | 5,555 | 4,149 | +0.0% | *not run* |
| `a` 2021 / 500 | T-720 | 6,004 | 5,394 | 3,988 | -- | 3500 .. 3950 |
| `b` 2020 / 700 | T-720 | 7,010 | 6,400 | 4,994 | +16.8% | 3500 .. 4950 |
| `c` 2019 / 800 | T-720 | 7,736 | 7,126 | 5,720 | +28.8% | 3500 .. 5675 |
| `d` 2021 / 800 | T-720 | 6,017 | 5,407 | 4,001 | +0.2% | 3500 .. 3975 |

The 610-game holdout is identical in all seven cells, so the arms differ only in
how far back their development history reaches.

Arm `c` buys +35% games at T-60 and +29% at T-720: the older seasons' coverage
does decay with lead time, but far less sharply than the raw-file counts
suggested. The two horizons are therefore a weaker contrast than this campaign
was designed around -- see "Why T-60 and T-720".

### What arm `d` actually measures

At T-60, nothing. Arms `a` and `d` clean to the same 6,165 games and therefore
the same columns, and with `random_state: 16` and every other field equal the
two runs are identical -- 3-4 GPU hours for a reproducibility check. At T-720 it
admits 13 extra games out of 6,004, which is 0.2% against a seed noise floor of
1.0-4.4pp on holdout win rate: also not a measurement.

The cause is the ordering described under "Why the question exists": by the time
`max_na_per_row` is applied, the season-gated columns are already gone, so at
the 2021 floor no row is anywhere near 500 surviving NaNs. **The budget lever
this campaign was built to isolate does not exist at the 2021 floor**, and no
value of `max_na_per_row` can make `d` differ from `a`.

That is a result, not only a defect. It means `b` and `c` do not carry the
budget confound the design feared: at the 2021 floor the budget admits nothing,
so what separates them from `a` is the older seasons themselves, plus the
column-level cleaning changes those seasons induce (different NaN profiles and
different correlations produce a different surviving column set -- that
confound is real and remains).

Read the decomposition as `c` vs `a` and `b` vs `a` directly. **The T-60 `d`
cell was therefore dropped from part 1** rather than spend 3-4 GPU hours
reproducing arm `a`; `d` survives at T-720, where the 13 extra games at least
make it a distinct run. To restore it, copy `l_t720_d_floor2021_na800.yaml`,
set `snapshot_minutes: 60`, take arm `a`'s T-60 ladder (ceiling 4,149) and add
the file back to the part 1 runner.

### Why T-60 and T-720

Both are in `DEFAULT_SNAPSHOT_GRID`, so a model at either is promotable and the
daily build samples it. T-720 also has a 6x100 reference already --
`intermediate25_t720_line_error` -- though at a 3,500 fixed window and a 0.99
`ODDS_` correlation override, so it is a rough comparison rather than a control.
The pair was chosen to bracket
the effect size -- the original estimate had the relaxation buying twice as much
data at T-60 as at T-720. Measured on cleaned data the gap is +35% against +29%,
so the two horizons bracket much less than intended. They are still worth
running as a sign-agreement check across horizons, which is what the
pre-registered reading leans on, but they are no longer a contrast between a
high-volume horizon and a low-volume one.

### The window ceiling is part of the arm

Each arm's `train_games_choices` tops out just under **its own** ceiling -- the
cleaned development history before that arm's earliest CV fold, as MEASURED by
the pre-flight and listed in the table above. This is not a detail. `tail(N)` takes the most recent N games, so
with a window fixed across arms the extra seasons would never enter training at
all and the campaign would measure nothing but a cleaning change -- a silent
no-op of exactly the kind this repo keeps finding. An arm that admits more
history must be able to train on it.

The consequence is that `b` and `c` differ from `a` in history *and* in
available window. That is the proposition being tested ("is deeper history worth
the missingness it brings?"), not a confound to remove -- the extra games are
worthless if they cannot be trained on.

**The ladder is five rungs, evenly spaced from 3,500 to each arm's own
ceiling.** Two deliberate choices:

- **It starts at 3,500, not the parent campaign's 2,500.** The question here is
  whether *more* history helps, so the arms are not given the option to answer
  it by training on less. In the parent campaign 4 of 17 line-error horizons
  selected the 2,500 floor; leaving that rung in would let an arm that admits
  three extra seasons quietly ignore them.
- **Every arm gets the same number of rungs.** Spacing five rungs across each
  arm's own range means arm `c` steps 3,500 -> 4,200 -> 4,875 -> 5,550 -> 6,250
  at T-60 rather than jumping straight from 3,500 to its ceiling, while the
  search space stays the same size in every arm. Giving the deeper arms more
  rungs would have spent more of their 150 trials on the window dimension, which
  is a difference between arms that has nothing to do with history depth.
- **The top rung sits at least 25 games below the measured ceiling**, on a
  multiple of 25. A rung above the ceiling is a hard `ValueError` from
  `SplitProvider.splits_for` that costs the whole cell, and the margin is worth
  0.6% of the largest window -- far below anything this campaign can resolve.

The cost is that arm `a` no longer reproduces the existing
`early25_t{60,720}_line_error_window` runs bit-for-bit -- its ladder has changed
-- so those runs are a rough reference rather than an exact control. Arm `a` is
the control that matters here, and it is run fresh.

## Held fixed in all seven cells

Same CSV and `expected_checksum`; `dataset_type: intermediate_line`; regular
season plus play-in with playoffs excluded and overtime kept; correlation
pruning at 0.95 (**0.98** for `ODDS_`); CV of six latest non-overlapping
`test_anchored` folds of 100 games with a 120-game step; 90-day daily walk-forward holdout; time decay
off; 150 Optuna trials with no timeout; pooled MAE objective; lexicographic
selection; `random_state: 16` with `evaluation_seeds: [101, 202]`; bets at
`|edge| >= 0.1` scored against `ODDS_CLOSING_TOTAL_LINE_bet365`.

## Pre-registered reading

Stop at the first failure.

1. **Seed noise first.** Read `seed_roi_range` before anything else. On this
   dataset the same config has moved holdout win rate by 1.0-4.4pp across three
   seeds. A difference smaller than an arm's own seed range is not a result.
2. **Volume.** Break-even is 52.38%. Quote the Wilson 95% interval for every win
   rate. At ~590 holdout bets that interval is roughly +/-4pp, so no single arm
   will reach significance on its own -- the decomposition and the sign
   agreement across two horizons are what carry the evidence.
3. **CV and holdout together.** CV win rate in these runs carries ~4.7pp of
   early-stopping inflation (each Optuna fold stops on the fold it is then
   scored on; measured median 4.2pp across the 6x100 line-error runs, 4.7pp at
   8x84), and across the 28 arms of the early campaigns CV correlated with
   holdout at **r = -0.06**. Do not rank arms on CV. Use the 3-seed mean holdout
   as the estimate, and treat a large CV/holdout gap as a warning.
4. **Legitimacy.** Arms admit different games, so the holdout cohorts are *not*
   identical -- arm `c` at T-60 has more history but the same 90-day holdout
   window. Confirm `holdout_n_games` before comparing, and re-score at a common
   `RESCORE_EDGE_THRESHOLD = 0.1` in `survey_experiments.ipynb`.

**Expectation.** The one prior test of this trade, `pubbet_c_drop_and_extend` on
2.0 closing data, went *against* extending: seed-mean 51.79% -> 50.14%. It is
weak evidence here -- different dataset, different mechanism (it dropped columns
rather than raising the budget), and its isolating cell never ran -- but the
prior is "no improvement". The two extra seasons are the 2019-20 bubble and the
72-game 2020-21, so this trades distribution match for volume.

**Multiple comparisons.** Seven cells against a ~4pp noise floor: expect about
one to look good by luck. A single arm beating its baseline at one horizon is a
hypothesis, not a result. What would be believable is the same sign in the same
decomposition step at both horizons.

**If nothing separates**, that is the useful answer: keep 2021/500, and the 17
promotable line-error models need no rebuild on this axis.

## Running

```bash
poetry run python scripts/preflight_campaign.py experiments/history_depth_line_error_2_5_2026_09
bash experiments/runners/run_history_depth_line_error_2_5_2026_09_part1_t60.sh
bash experiments/runners/run_history_depth_line_error_2_5_2026_09_part2_t720.sh
```

Run the full pre-flight, never `--skip-data`: the per-arm window ceiling is
measured there, and a ceiling that is too large by even one game is a hard
`ValueError` at run time -- which is how five cells of
`early_totals_spread_window_2_5_2026_09` were lost.

**Running the two parts concurrently on this box is a memory race, and part 2
is deliberately set up to enter it.** The constraint is RAM, not the GPU:
`prepare_dataset` loads the whole 4.6GB CSV before filtering to a single
snapshot, which peaks near 14GB of RSS against 15GB of physical memory and 31GB
of swap. On 2026-09-22 both parts were launched 46 seconds apart and the OOM
killer took part 2's pre-flight -- exit 137, no traceback, the log said only
`Killed`. It applies to the cells too, not just the pre-flight: every run
reloads the CSV at start, so a collision can land mid-campaign and cost a cell
that is hours in.

The two runners are asymmetric on purpose:

- **Part 1** takes an exclusive `flock` on
  `artifacts/logs/.history_depth_line_error_2_5_2026_09.lock` before its
  pre-flight, so a second copy of part 1 queues rather than races.
- **Part 2 takes no lock.** Launched alongside part 1 it competes for memory,
  which either thrashes through swap or ends with the kernel killing one of
  them. Both runners now report the exit status and name the signal, so this
  shows up as `killed by signal 9 -- OOM killer` instead of a bare `Killed`.

Run them sequentially and budget 21-28h in total. Run them together and the
wall clock is shorter only if nothing dies.

Within a part the baseline arm `a` runs first, so an interrupted part still
leaves the reference behind it. `SKIP_EXISTING=1` resumes. Each runner re-runs
the pre-flight and aborts the part if it fails, now reporting the exit status
and naming the signal when one killed it.

Running the standalone pre-flight across all seven configs at once cleans the
dataset four times in one process. That is fine on its own, but nothing else
memory-hungry may be running beside it.

Budget roughly 3-4h per cell (Optuna was 1h03m-3h06m per run in the parent
campaign, plus the daily walk-forward holdout across three seeds): about 9-12h
for part 1's three cells and 12-16h for part 2's four, and since the parts are
serialised, 21-28h end to end.
