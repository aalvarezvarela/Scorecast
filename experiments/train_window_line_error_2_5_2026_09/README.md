# Training-window sweep, line error, schema 2.5 (2026-09)

**Does training on older games help?** Twelve cells, six per horizon. Five fix
`train_games` at a different value on one identical dataset; the sixth repeats
the 3,500 arm under a different seed so the campaign measures its own noise
floor instead of discovering it by accident.

## Why this replaces `history_depth_line_error_2_5_2026_09`

That campaign ran 7/7 clean and answered nothing. Two independent reasons, both
corrected here.

**It changed the feature set while it changed the history.** Arms differed in
`season_year_floor`, and cleaning runs per arm: a different floor means a
different correlation structure, so a different set of columns survives pruning.
The gap was not subtle.

| arm | season floor | features into the model |
|---|---|---|
| a | 2021 | 1,380 |
| b | 2020 | 1,388 |
| c | 2019 | **2,407** |

Arm `c` had the best T-60 result and was the only cell flagged significant. It
was also the arm carrying 1,027 extra features. Nothing in the design separates
those two facts.

Most of that gap was a bug, since fixed — see
`src/nba_ou/data_processing/missing_data/column_redundancy.py`. `TIME_TO_MATCH_MIN`
is protected (the scoring-sidecar join needs the `(game, snapshot)` key) and
becomes constant the moment a single horizon is selected. Protection made it
rank first in the redundancy order, so it was the first column kept and the
yardstick every later column was measured against — and a zero-variance
denominator returned `inf`, not `NaN`. `inf > 0.95` is `True`, so ~1,300
features were dropped as "redundant" with a constant. It fired at the 2021 and
2020 floors but not 2019. With the fix, arm `a` cleans to 2,250 features instead
of 1,380, and the gap to arm `c` falls from 1,027 columns to 158.

**It tuned the window, so its answer was an argmax.** Arms `a` and `d` at T-720
are near-replicates: identical 1,410 features, 6,004 vs 6,017 rows — 0.2% apart.

| T-720 | selected window | CV MAE | holdout MAE | win rate | ROI |
|---|---|---|---|---|---|
| a | 3,950 | 14.2412 | 14.4158 | 55.67% | +6.2% |
| d | 3,625 | 14.2320 | 14.5514 | 51.71% | −1.3% |

Across three seeds they differ by **1.87pp win rate and 0.033 MAE**. The entire
spread across arms a/b/c at T-720 was 0.76pp and 0.032 MAE. The replicate gap
was larger than every effect the campaign was built to detect, so "the tuner did
not choose the longest window" carried no information about history at all.

## Design

One dataset per horizon, cleaned once, `season_year_floor: 2019` and
`max_na_per_row: 800` in **every** arm. Cleaning happens before any split, so
all six cells at a horizon are handed the same rows and the same columns. The
only thing that differs is how far back the training window reaches.

| config | `train_games` | seed | ≈ seasons | reaches back through |
|---|---|---|---|---|
| `w_<h>_g1500` | 1,500 | 16 | 1.2 | last winter |
| `w_<h>_g2500` | 2,500 | 16 | 2.0 | — |
| `w_<h>_g3500` | 3,500 | 16 | 2.8 | the anchor |
| `w_<h>_g3500_rep` | 3,500 | **17** | 2.8 | the noise floor |
| `w_<h>_g4500` | 4,500 | 16 | 3.7 | — |
| `w_t60_g6250` / `w_t720_g5675` | 6,250 / 5,675 | 16 | 5.1 / 4.6 | 2020-21 no-crowd, into the 2019-20 bubble |

3,500 is the anchor because it is where both deep arms of the predecessor
campaign pinned when the window was tuned, with median CV MAE rising
monotonically above it (arm b 14.094 → 14.211; arm c 14.066 → 14.193). That
looked like evidence that older games hurt. It was also the **lowest rung
offered**, so the campaign could only say the optimum was ≤ 3,500, never where.
1,500 and 2,500 exist to see below it.

`train_games_choices: null` is set explicitly in every config, not omitted:
`experiments/_base.yaml` supplies `[2500, 3000, 3500, 4000]`, and an absent key
inherits it. That would put the window back under Optuna and turn all six cells
into one tuned run repeated. The pre-flight catches it — every arm reports
"reaches 4000", the inherited maximum, instead of its own window.

## Measured feasibility

Ceilings measured 2026-09-22 with the corrected cleaner. Every top rung sits at
least 25 games under its ceiling; a rung above it is a hard `ValueError` that
costs the cell.

| horizon | cleaned | dev | holdout | ceiling | top rung |
|---|---|---|---|---|---|
| T-60 | 8,301 | 7,691 | 610 | 6,285 | 6,250 |
| T-720 | 7,734 | 7,124 | 610 | ~5,718 | 5,675 |

Pre-flight passes for both parts: all six folds reach the requested window in
every arm.

## Reading the result

**Read the 3,500-vs-3,500-replicate gap first.** Two runs, identical in every
field but `random_state`, on the same data. Whatever separates them is noise.
No window difference smaller than that gap means anything, however monotone the
trend across the ladder looks, and the predecessor campaign is the standing
proof that a clean monotone trend can be entirely noise.

Then read the ladder. Each arm also carries `evaluation_seeds: [101, 202]`, so
every cell has three holdout measurements — use the 3-seed mean, not the primary
seed, which is what made arm `a` look like a +6.2% ROI strategy.

A real "older games help" result looks like the long arms beating the short ones
by more than the replicate gap, at both horizons, on the 3-seed mean. Anything
less is the same non-answer as last time.

## Running

```bash
bash experiments/runners/run_train_window_line_error_2_5_2026_09_part1_t60.sh
bash experiments/runners/run_train_window_line_error_2_5_2026_09_part2_t720.sh
```

**Neither part takes a lock**, so the two can be launched together and will run
concurrently. They then compete for memory: one raw load of the 4.6GB
intermediate CSV peaks near 14GB of RSS against 15GB of physical memory plus
31GB of swap, so two at once either thrashes through swap or ends with the
kernel killing one. Both runners name signal 9 as the OOM killer in the log when
that happens, and `SKIP_EXISTING=1` resumes whichever part lost. The peaks only
overlap while a cell loads its CSV — the first ~2 minutes of each cell — so a
collision is possible rather than certain. Each runner's header carries the
three lines that serialise them again if you want that back.

Budget roughly 2h per cell (the 2,400-feature frames are the slow ones), so
~12h per part: ~24h sequentially, or ~12h together if nothing dies.

T-60 is worth running first. At T-720 the predecessor's replicate noise
exceeded every between-arm effect, so that horizon may simply lack the
resolution to answer the question at this sample size.
