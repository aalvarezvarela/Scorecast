# Top-6 early horizons, window fixed at maximum, decay on/off (schema 2.5)

Twelve runs: the six best horizons by **holdout** win rate from
`early_line_error_window_2_5_2026_09`, each run twice — once unweighted and
once with exponential time decay — with the training window no longer tuned but
**fixed at that horizon's maximum**.

## Which six horizons, and why

Holdout win rate in the parent campaign (18 arms, 610 holdout bets each except
T-1080's 601):

| Horizon | Parent holdout win | Window Optuna picked | Fixed here |
| ---: | ---: | ---: | ---: |
| T-600 | 58.28% | 4,727 (max) | 4,727 |
| T-360 | 57.33% | 4,778 (max) | 4,778 |
| T-1080 | 57.00% | 2,500 | 3,775 |
| T-540 | 56.91% | 4,755 (max) | 4,755 |
| T-240 | 56.29% | 4,788 (max) | 4,788 |
| T-660 | 55.46% | 2,500 | 4,699 |

The parent campaign's top entry by run — not by horizon — was
`early25_t360_line_error_window_decay` at 58.67%. It is not a seventh horizon:
it is T-360 again, and its selected trial set `use_sample_weight: false`, so
that run applied **no decay at all**. The 58.67% is an unweighted T-360 model
that drew different hyperparameters, not evidence for decay. That is the direct
reason the decay arms here set `allow_unweighted: false` (see below).

## The two deliberate changes

1. **`train_games` is fixed, not tuned.** `train_games_choices` is absent, so
   the window leaves the Optuna search space entirely and all 150 trials are
   spent on the remaining hyperparameters. Each horizon's value is the maximum
   from the parent campaign's README: the cleaned development games available
   before the earliest of the eight CV validation folds, at `max_na_per_row:
   500` and a 90-day holdout. Four of the six horizons already chose that
   maximum, so for them this campaign shrinks the search space rather than
   moving the window; T-1080 and T-660 get a genuinely larger window (2,500 →
   3,775 and 2,500 → 4,699).

2. **Decay on versus off, one arm each.** The no-decay arm is
   `sample_weight.enabled: false`. The decay arm tunes `lambda_` over
   `[0.0005, 0.005]` with **`allow_unweighted: false`**, which departs from
   `_base.yaml`'s default of `true`. With the default the sampler can select
   "no weighting" and the decay arm silently becomes a second copy of the
   baseline — which is exactly what happened to the parent campaign's one decay
   arm. Forcing every decay trial to weight is what makes the pair an A/B.

Everything else is held to the parent campaign: same CSV and checksum, same
`season_year_floor`, same `max_na_per_row: 500`, same 90-day holdout, same
eight latest nonoverlapping `test_anchored` folds of ≥80 games, `random_state:
16`, 150 trials, `evaluation_seeds: [101, 202]`, and the same line-error target
scored against `ODDS_CLOSING_TOTAL_LINE_bet365`.

## The CV folds are the parent campaign's

Fixing the window moves fold construction from the tuned path (build histories
unbounded, then tail each candidate) to `make_test_anchored_walk_forward_splits`
with `train_games` passed in. That is a different code path, and under
`test_anchored` the window is part of the fold layout — so it is worth being
explicit that it does not change the folds here.

A fold is accepted only where `len(train_idx) >= min_train_games`. With
`train_games=None` that is `len(pool) >= 1250`; with `train_games=N` it is
`min(N, len(pool)) >= 1250`, and since every N here is far above 1250 the two
conditions coincide. `tail(N)` never requires N rows, so no fold is dropped and
none is shifted. Both paths then apply `apply_training_filter` to train indices
only, and with no training-row filters enabled in these configs that is a no-op
on both sides.

Because each N is by construction the earliest fold's full pool, no fold trains
on fewer games than requested. The runners therefore preflight with
`--skip-data`; set `PREFLIGHT_FULL=1` to re-verify per-fold training sizes
against the data, which is the slow check.

## Reading it

The contrast is **within a horizon**, decay arm against its own no-decay arm —
six paired comparisons, not twelve independent ones. At ~610 holdout bets a
Wilson interval on a 56% win rate spans roughly ±4pp and the parent campaign's
entire 18-arm spread was 51.75–58.67%, so expect one or two pairs to separate by
luck alone. A decay effect worth believing shows the same sign across most of
the six horizons; a single large gap at one horizon does not.

Read `train_games_tuned` and `sample_weighting` in
`summary_experiments.ipynb` — both are already in `factors.FACTOR_SOURCES`, so
this campaign's runs will be matched against the parent's on one knob at a time.

## Running

```bash
bash experiments/runners/run_early_line_error_fixedmax_2_5_2026_09_part1.sh
bash experiments/runners/run_early_line_error_fixedmax_2_5_2026_09_part2.sh
```

Part 1 is T-600, T-360, T-1080; part 2 is T-540, T-240, T-660. Within each
part a horizon's no-decay baseline runs immediately before its decay arm, so an
interrupted part still leaves complete pairs behind it. The two parts run
concurrently on the one GPU the parent campaign used. `SKIP_EXISTING=1` resumes.
