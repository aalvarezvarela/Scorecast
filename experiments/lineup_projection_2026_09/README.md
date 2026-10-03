# Lineup projection family: with and without (schema 2_6)

The go/no-go G campaign of `docs/lineup_projection_plan.md` (§8.4, §8.5). Four
cells, two per totals strategy, and each pair differs **only** in its dataset file:

| Cell | Strategy | Dataset |
|---|---|---|
| `a_line_error_control` | `line_error_regressor` | 2_6 without `LU_*` |
| `b_line_error_lineup` | `line_error_regressor` | 2_6 with the `LU_*` columns |
| `c_total_points_control` | `total_points_regressor` | 2_6 without `LU_*` |
| `d_total_points_lineup` | `total_points_regressor` | 2_6 with the `LU_*` columns |

`tests/test_lineup_campaign_configs.py` pins the one-difference design.

## The datasets

```bash
# Treatment: the ordinary 2_6 build (writes the 2_5 base and layers 2_6 on it).
python scripts/create_train_data/create_train_data.py --limit 2026-07-04
# -> data/train_data/closing_line_data_2_6_20260704.parquet
#    (or, from an existing 2_5 file:
#     python scripts/create_train_data/build_schema_version.py \
#         data/train_data/closing_line_data_2_5_20260704.parquet --to 2_6)

# Control: the same file with the LU_* columns removed, verified to load as
# exactly the treatment minus those columns.
python experiments/lineup_projection_2026_09/make_control_csv.py
# -> data/train_data/closing_line_data_2_6_20260704_lineup_control.parquet
```

Deriving the control this way is exact, not an approximation: the 2_6 layer
(`nba_ou.create_training_data.schema_layers.v2_6`) only appends columns to the
2_5 file, and nothing else reads the `LU_*` columns. The control keeps the 2_6
starter-history columns, so the pair differs in the lineup family alone; the
2_5 file would not do as a control, because it also lacks starter history. Both
checksums are pinned in the configs.

**Running on another machine.** `data/` is not in git. Copy both files to the
same paths; the pinned checksums make a wrong or partial copy fail at
pre-flight rather than run. Nothing else is needed at run time: the rating
cache and stints are only inputs to building the files.

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
- **Temporal evaluation partition: the last 90 days.** The training framework
  calls this a `holdout`, and none of the lineup lambdas or windows was tuned
  there. Feature selection did inspect 2025-26, including candidate absence
  channels, so this partition is not an untouched test of the selected family.
  The independent retrospective 2019-20/2020-21 check and future prospective
  evaluation have not run.
- **Seeds 16 + [101, 202, 303].** Past campaigns measured a 4.9-12.0 point ROI
  range from seed alone. Nothing smaller is a result.
- Windows (4,500 / 4,000 training games), trials (150), thresholds and
  cleaning follow the referee campaign, so these runs sit beside it.

## Pre-registration (written before any run)

Status: **not run.** The 2016-2020 stints now exist, the rating cache was
rebuilt with solver version 2 (2026-09-27), and the pre-registered 2019-20 /
2020-21 check has run (plan §8.7): the defense + pace slope replicated
(+0.41 pooled [+0.09, +0.74]), directional accuracy ~55% did not clearly clear
break-even, and the bench claim split by season. Still to do before running:
regenerate the control/treatment files and their checksums from the new cache,
and decide `season_year_floor` now that coverage reaches back to 2016-17
(decide before any run, not after).

What the lineup family can plausibly do is small. Its best columns' univariate
hit rate is 55-58% on the largest values, against a 52.38% break-even. So:

- **Expected:** at most a small improvement, strongest for
  `line_error_regressor`, and concentrated in games with a projected absence
  (`LU_ABSENCE_IMPACT_PTS_BEFORE != 0`).
- **Null:** a lineup-minus-control difference inside the control's own seed
  range, on either CV or the temporal evaluation partition. That is the most likely outcome and is not
  a failure of the campaign.
- **Only a promising result for further testing:** the lineup cell beats its
  control beyond the seed range on CV **and** the temporal partition agrees in
  sign, for the same strategy. Two
  strategies means two chances; with this noise floor, one of them looking
  good by luck is unremarkable.
- **Also read:** the `LU_*` columns' feature importance (a family the model
  never splits on cannot be the cause of a difference), and win rate on the
  temporal partition's absence and no-absence games separately.

Only if it passes: flip the `lineup_features` default, bump to schema 2_7,
wire serving (plan §8.2, G′).
