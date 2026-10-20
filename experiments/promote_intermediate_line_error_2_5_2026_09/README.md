# Intermediate line-error promotion sweep, schema 2.5 (2026-09)

**One promotion candidate per pre-game horizon.** Sixteen cells, T-30 through
T-1080; T-0 is already served by the closing-line slot
`2_5/line_error/t0000/main`. Every cell uses the same recipe, so the cells
differ only in the horizon, and each one maps to the slot
`2_5/line_error/t{HHHH}/main`.

## The recipe

| setting | value | why |
|---|---|---|
| data | `intermediate_line_data_2_5_20260613.csv`, one snapshot per cell | schema 2.5 |
| season floor | 2019 | the widest available |
| `max_na_per_row` | 800 | admits essentially every game with a snapshot; see below |
| `train_games` | fixed at each horizon's maximum | see below |
| CV | `test_anchored`, 6 folds × 100 games, step 120 | same protocol as the promoted closing models |
| Optuna | 150 trials, early stopping (1000 cap, 70 rounds), lexicographic selection | as in `train_window_line_error_2_5_2026_09` |
| cleaning | corr 0.95, `ODDS_` 0.98 | as in `train_window_line_error_2_5_2026_09` |
| holdout | 90-day daily walk-forward (2026-01-18 .. 04-17, 610 games) | the production analogue |
| seeds | `random_state` 16, evaluation seeds 101 / 202 | three-seed holdout per cell |

## The NaN budget only decides how much old data gets in

Measured on 2026-09-24 with the real `prepare_dataset` and fold builder, at
each horizon and budget (floor 2019):

| horizon | features (same at every budget) | cleaned games at 300 / 500 / 800 | window ceiling at 300 / 500 / 800 | window used |
|---|---|---|---|---|
| T-30 | 2,397 | 6,119 / 7,287 / 8,301 | 4,103 / 5,271 / 6,285 | **6,275** |
| T-60 | 2,408 | 6,119 / 7,281 / 8,301 | 4,103 / 5,265 / 6,285 | **6,275** |
| T-120 | 2,419 | 6,119 / 7,265 / 8,301 | 4,103 / 5,249 / 6,285 | **6,275** |
| T-180 | 2,438 | 6,119 / 7,237 / 8,294 | 4,103 / 5,221 / 6,278 | **6,275** |
| T-240 | 2,445 | 6,119 / 7,181 / 8,285 | 4,103 / 5,216 / 6,269 | **6,250** |
| T-300 | 2,446 | 6,119 / 7,108 / 8,273 | 4,103 / 5,143 / 6,257 | **6,250** |
| T-360 | 2,449 | 6,100 / 7,014 / 8,243 | 4,090 / 4,998 / 6,227 | **6,225** |
| T-420 | 2,456 | 6,095 / 6,903 / 8,223 | 4,085 / 4,887 / 6,207 | **6,200** |
| T-480 | 2,458 | 6,076 / 6,848 / 8,201 | 4,066 / 4,832 / 6,185 | **6,175** |
| T-540 | 2,469 | 6,062 / 6,785 / 8,164 | 4,052 / 4,769 / 6,148 | **6,125** |
| T-600 | 2,474 | 6,039 / 6,617 / 8,039 | 4,029 / 4,601 / 6,023 | **6,000** |
| T-660 | 2,466 | 6,001 / 6,460 / 7,878 | 3,991 / 4,444 / 5,862 | **5,850** |
| T-720 | 2,444 | 5,910 / 6,348 / 7,734 | 3,900 / 4,332 / 5,718 | **5,700** |
| T-840 | 2,451 | 5,627 / 6,097 / 7,367 | 3,619 / 4,081 / 5,351 | **5,350** |
| T-960 | 2,454 | 5,240 / 5,853 / 6,875 | 3,282 / 3,849 / 4,870 | **4,850** |
| T-1080 | 2,449 | 4,872 / 5,446 / 6,299 | 2,928 / 3,520 / 4,422 | **4,400** |

- **The feature set does not depend on the budget.** `max_na_per_row` is the
  last row filter, applied after column cleaning, so at every horizon 300, 500
  and 800 hand the model identical columns.
- **The budget works as a history cutoff.**
  - **300** drops every 2019-20 and 2020-21 game, so it is effectively a
    2021-22 floor.
  - **500** admits roughly half of the older games.
  - **800** admits essentially all of them: 1,051 of the 2019-20 and 1,085 of
    the 2020-21 games at T-30. A higher budget would add almost nothing.
- **Why the longest window.** The window is the smallest of the six fold pools,
  rounded down to 25. The earlier campaigns found:
  - At T-60 the 6,250 window was the best cell in
    `train_window_line_error_2_5_2026_09`: 58.3% over three seeds, with no seed
    below 57.5%.
  - At T-720 the longest window (5,675) came second.
  - All three `closing_*_6x100_2_5_2026_09` targets were best at 6,200.

  The production refit takes the same number of most recent games.
- **Long horizons have less data.** Fewer games have an 18-hour snapshot than a
  30-minute one, so from T-600 up the window shrinks: 6,000 at T-600, down to
  4,400 at T-1080.

## Controls (end of part 2)

Five more cells, none of them promotion candidates. They run after part 2's
eight horizons, and each is read against its own main cell.

| config | differs from its main cell in | answers |
|---|---|---|
| `p_t240_line_error_rep` | `random_state` 17 | noise floor |
| `p_t840_line_error_rep` | `random_state` 17 | noise floor |
| `p_t240_line_error_recent` | `train_games` 4,100 (was 6,250) | all history vs 2021-22 onward |
| `p_t480_line_error_recent` | `train_games` 4,050 (was 6,175) | all history vs 2021-22 onward |
| `p_t960_line_error_recent` | `train_games` 3,275 (was 4,850) | all history vs 2021-22 onward |

**The recent-only controls use the same data with a shorter window, not
`max_na_per_row: 300`.** At long horizons a budget of 300 also drops recent
games whose snapshots are sparse: about 400 of the 5,640 games from 2021-22 on
at T-960. That would mix "no old seasons" with "fewer sparse games".

Keeping the budget at 800 and only shortening the window leaves the rows and
features identical. Each window fits inside the games from 2021-22 on in the
smallest fold pool (4,149 / 4,126 / 3,635), so no fold and no holdout day
trains on a 2019-20 or 2020-21 game.

## How to run

The runners are unlocked by choice, so both parts can run at the same time:

```bash
nohup bash experiments/runners/run_promote_intermediate_line_error_2_5_2026_09_part1_t30_t420.sh > /dev/null 2>&1 &
nohup bash experiments/runners/run_promote_intermediate_line_error_2_5_2026_09_part2_t480_t1080_controls.sh > /dev/null 2>&1 &
```

Each cell reloads the 4.6GB CSV, which peaks near 14GB of RAM for its first
couple of minutes. If both parts load at once, one may be OOM-killed. The log
names signal 9, and rerunning that part with `SKIP_EXISTING=1` resumes it.

Budget roughly 2–3 hours per cell: part 1 (8 cells) about 16–24 hours, part 2
(13 cells) about 26–39 hours.

## How to read it

0. **Read the controls first.**
   - The two replicate gaps are the error bar. No difference between horizons
     smaller than them is a result.
   - Then each recent-only control against its main cell. If recent-only wins
     at all three horizons by more than the replicate gap, redo the sweep with
     short windows before promoting anything.
1. **Compare against the earlier `early_line_error_window_2_5_2026_09` sweep,
   with care.** That sweep used a 2021 floor, budget 500, a tuned window, and
   ran before the correlation-pruning fix. Treat it as context, not as a
   control.
2. **Don't pick horizons by rank.** The T-30 audit found the 18 horizon arms of
   that sweep to be one statistical population (homogeneity p = 0.45 on the
   holdout). With 16 cells on the same 610-game holdout, expect one or two to
   look good by luck. Judge each cell on its own:
   - does every seed clear break-even (52.4%)?
   - does the model beat the line on MAE?
   - do CV and the holdout agree in sign?

   A horizon that fails these should not be promoted just because it exists.
3. **Check T-60 against `w_t60_g6250`** from the train-window campaign. It has
   the same recipe and seed, with a window 25 games longer, so it should land
   close to 58.3%. A large gap means something changed.

## Before any of these can serve

Training and promoting a slot works today:
`python -m training_pipeline.promote <run_dir> --to-s3`. **Serving and daily
refit do not.**

- **Prediction.** `create_df_to_predict` builds closing-line features only. The
  intermediate models depend on the `ODDS_SNAP_*` family, which is 1,063 of
  2,444 features at T-60 and is built only by `create_intermediate_line_df`.
- **Daily refit.** `resolve_training_frame` raises `TrainingFrameUnavailable`
  for `intermediate_line` specs (`docs/model_registry_reorg_plan.md` §5, work
  item 10). `retrain_prediction_models.py` counts that as a failure and exits
  1, and the workflow's smoke-test and promote steps run only on success. **So
  adding an intermediate row to `ENABLED_MODELS` today would stop the closing
  models from being promoted every day.**

Do not add these slots to `ENABLED_MODELS` until both are in place.
