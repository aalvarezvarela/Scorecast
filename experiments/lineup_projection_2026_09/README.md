# Lineup projection family: with and without (schema 2_6)

The go/no-go G campaign of `docs/lineup_projection_plan.md` (§8.4, §8.5). Four
cells, two per totals strategy, and each pair differs **only** in its CSV:

| Cell | Strategy | Dataset |
|---|---|---|
| `a_line_error_control` | `line_error_regressor` | 2_6 without `LU_*` |
| `b_line_error_lineup` | `line_error_regressor` | 2_6 with the `LU_*` columns |
| `c_total_points_control` | `total_points_regressor` | 2_6 without `LU_*` |
| `d_total_points_lineup` | `total_points_regressor` | 2_6 with the `LU_*` columns |

`tests/test_lineup_campaign_configs.py` pins the one-difference design.

## The datasets

```bash
# Treatment: the ordinary builder with the family switched on.
python scripts/create_train_data/create_train_data.py --lineup-features --limit 2026-07-04
# -> data/train_data/training_data_2_6_20260704_with_lineup_features.csv

# Control: the same file with the LU_* columns removed, at the text level
# so every other cell stays byte-identical (a pandas round-trip would strip the
# leading zeros from GAME_ID).
python experiments/lineup_projection_2026_09/make_control_csv.py
# -> data/train_data/training_data_2_6_20260704_lineup_control.csv
```

Deriving the control this way is exact, not an approximation:
`attach_lineup_features` returns the frame untouched when the flag is off and
adds only the `LU_*` columns when it is on (`tests/test_lineup_features.py`),
and none of the later stages reads those columns. Both checksums are pinned in
the configs.

**Running on another machine.** `data/` is not in git. Copy both CSVs to the
same paths; the pinned checksums make a wrong or partial copy fail at
pre-flight rather than run. Nothing else is needed at run time: the rating
cache and stints are only inputs to building the CSVs.

```bash
poetry run python scripts/preflight_campaign.py experiments/lineup_projection_2026_09
nohup bash experiments/runners/run_lineup_projection_2026_09.sh > /dev/null 2>&1 &
```

## Design choices

- **`season_year_floor: 2021`.** The lineup ratings start in 2021-22. On an
  earlier floor every `LU_*` column is 100% NaN in the early seasons and ~0%
  after, `find_season_gated_columns` drops the whole family, and the treatment
  becomes a silent copy of the control. The control uses the same floor, so
  the pair still differs in the columns alone.
- **Holdout: the last 90 days.** That falls inside 2025-26, the season none of
  the lineup parameters (lambdas, minutes window, calibration window) was
  tuned on.
- **Seeds 16 + [101, 202, 303].** Past campaigns measured a 4.9-12.0 point ROI
  range from seed alone. Nothing smaller is a result.
- Windows (4,500 / 4,000 training games), trials (150), thresholds and
  cleaning follow the referee campaign, so these runs sit beside it.

## Pre-registration (written before any run)

Status: **not run.** Waiting for the 2019-2020 backfill; the dataset and
checksums will be regenerated then. See plan §8.6 for the current evidence
(the defense and pace channels, slope +0.45) and the pre-registered test on
2019-20 and 2020-21.

What the lineup family can plausibly do is small. Its best columns' univariate
hit rate is 55-58% on the largest values, against a 52.38% break-even. So:

- **Expected:** at most a small improvement, strongest for
  `line_error_regressor`, and concentrated in games with a projected absence
  (`LU_ABSENCE_IMPACT_PTS_BEFORE != 0`).
- **Null:** a lineup-minus-control difference inside the control's own seed
  range, on either CV or holdout. That is the most likely outcome and is not
  a failure of the campaign.
- **Only a pass:** the lineup cell beats its control beyond the seed range on
  CV **and** the holdout agrees in sign, for the same strategy. Two
  strategies means two chances; with this noise floor, one of them looking
  good by luck is unremarkable.
- **Also read:** the `LU_*` columns' feature importance (a family the model
  never splits on cannot be the cause of a difference), and win rate on the
  holdout's absence and no-absence games separately.

Only if it passes: flip the `lineup_features` default, bump to schema 2_7,
wire serving (plan §8.2, G′).
