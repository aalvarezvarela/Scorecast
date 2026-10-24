"""Which cleaning behaviour each dataset type gets, and the line between.

Cleaning has two modes, chosen by the DECLARED data.dataset_type rather than by
sniffing the frame:

* **closing-line** (one row per game) -- correlation over every row and every
  column, exactly as before any repeated-measures work existed.
* **intermediate-line** (one row per game AND snapshot) -- snapshot/market
  columns exempt from correlation entirely, everything else judged on one row
  per game.

The risk worth testing is not that the intermediate mode works -- that is
covered in test_cleaning_hygiene.py -- but that it can never engage on the
closing-line dataset, and that a forgotten declaration fails loudly instead of
pruning against the wrong view. Nothing would error on its own if it did; the
run would simply report different features, which is precisely the kind of
silent divergence this repo keeps finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)

from training_pipeline.config import CleaningConfig, DatasetType
from training_pipeline.data import (
    assert_dataset_type_matches_frame,
    clean_for_training,
    redundancy_policy_for,
)

CORRELATED_STEP = "correlated_columns"
GROUPED_STEP = "correlated_columns_one_row_per_group"


def _closing_frame(n: int = 60) -> pd.DataFrame:
    """One row per game, as the closing-line CSV is."""
    rng = np.random.default_rng(23)
    pace = rng.normal(100, 4, n)
    return pd.DataFrame(
        {
            "GAME_ID": [f"002240000{i:02d}" for i in range(n)],
            "TOTAL_POINTS": rng.normal(225, 20, n).round(),
            "ODDS_TOTAL_LINE_bet365": rng.normal(224, 15, n).round(),
            "PACE_BEFORE_TEAM_HOME": pace,
            # a near-copy, so the correlation step has something to decide
            "PACE_NEAR_BEFORE_TEAM_HOME": pace + rng.normal(0, 0.01, n),
            "OFF_RATING_BEFORE_TEAM_HOME": rng.normal(112, 5, n),
            "ODDS_SNAP_TOT_BET365_NORM_LINE": rng.normal(224, 15, n),
        }
    )


def _cleaning() -> CleaningConfig:
    return CleaningConfig(verbose=0, corr_threshold=0.95)


def _clean(df: pd.DataFrame, dataset_type: DatasetType = DatasetType.CLOSING_LINE):
    return clean_for_training(
        df,
        _cleaning(),
        force_keep_columns=["ODDS_TOTAL_LINE_bet365"],
        dataset_type=dataset_type,
    )


def _clean_intermediate(df: pd.DataFrame):
    return _clean(df, DatasetType.INTERMEDIATE_LINE)


# ---------------------------------------------------------------------------
# the closing-line dataset keeps the behaviour it always had
# ---------------------------------------------------------------------------


def test_closing_frame_gets_no_repeated_measures_policy():
    _, report = _clean(_closing_frame())

    assert report.redundancy_view == {}
    assert GROUPED_STEP not in report.columns_by_step()


def test_closing_frame_still_prunes_by_correlation_over_every_row():
    """The behaviour must be preserved, not merely 'not the new one'."""
    _, report = _clean(_closing_frame())

    assert report.columns_by_step().get(CORRELATED_STEP, 0) >= 1
    assert report.why_dropped("PACE_NEAR_BEFORE_TEAM_HOME")["step"] == CORRELATED_STEP


def test_closing_frame_does_not_exempt_snapshot_named_columns():
    """The exemption is part of the intermediate policy, not a naming rule. A
    column called ODDS_SNAP_* in a one-row-per-game frame is pruned like any
    other -- otherwise the closing-line dataset would quietly change behaviour
    for any column that happened to match the prefix."""
    df = _closing_frame()
    df["ODDS_SNAP_TOT_COPY_NORM_LINE"] = df["ODDS_SNAP_TOT_BET365_NORM_LINE"]

    cleaned, _ = _clean(df)

    kept = [c for c in cleaned.columns if c.startswith("ODDS_SNAP_")]
    assert len(kept) == 1


def test_closing_output_is_identical_with_and_without_the_policy_argument():
    """The policy path must be inert when unused: passing repeated_measures=None
    explicitly and omitting it entirely must give the same frame, so the
    default cannot drift away from the closing-line behaviour."""
    df = _closing_frame()

    omitted = clean_dataframe_for_training(df, corr_threshold=0.95, verbose=0)
    explicit = clean_dataframe_for_training(
        df, corr_threshold=0.95, repeated_measures=None, verbose=0
    )

    pd.testing.assert_frame_equal(omitted, explicit)


# ---------------------------------------------------------------------------
# the intermediate-line dataset gets the other behaviour
# ---------------------------------------------------------------------------


def _intermediate_frame(snapshots=(30, 60, 720)) -> pd.DataFrame:
    base = _closing_frame()
    return pd.concat([base.assign(TIME_TO_MATCH_MIN=t) for t in snapshots]).reset_index(
        drop=True
    )


def test_intermediate_frame_gets_the_policy():
    _, report = _clean_intermediate(_intermediate_frame())

    view = report.redundancy_view
    assert view["group_col"] == "GAME_ID"
    assert view["rows_in_view"] == 60
    assert view["rows_total"] == 180
    assert GROUPED_STEP in report.columns_by_step()
    assert CORRELATED_STEP not in report.columns_by_step()


def test_intermediate_frame_exempts_snapshot_columns():
    cleaned, _ = _clean_intermediate(_intermediate_frame())
    assert "ODDS_SNAP_TOT_BET365_NORM_LINE" in cleaned.columns


# ---------------------------------------------------------------------------
# the boundary
# ---------------------------------------------------------------------------


def test_single_horizon_slice_is_the_same_either_way():
    """slice_intermediate_snapshot.py writes one horizon per file. That frame
    has a TIME_TO_MATCH_MIN column but one row per game, so the pooled
    correlation already IS the one-row-per-game correlation. Declaring it
    either way must therefore give the same frame -- the policy has nothing to
    correct, and must not invent a difference."""
    df = _intermediate_frame(snapshots=(60,))

    as_closing, _ = _clean(df)
    as_intermediate, report = _clean_intermediate(df)

    pd.testing.assert_frame_equal(as_closing, as_intermediate)
    # ...and the view really did cover every row, rather than sampling one.
    assert (
        report.redundancy_view["rows_in_view"] == report.redundancy_view["rows_total"]
    )


def test_declaring_closing_on_an_intermediate_frame_raises():
    """The dangerous direction, and the whole point of declaring the type: this
    would correlate every historical feature over ten copies of each game and
    prune on that, with nothing downstream noticing."""
    with pytest.raises(ValueError, match="intermediate_line"):
        _clean(_intermediate_frame())


def test_the_contradiction_check_names_both_sides():
    with pytest.raises(ValueError) as excinfo:
        assert_dataset_type_matches_frame(
            _intermediate_frame(), DatasetType.CLOSING_LINE
        )

    message = str(excinfo.value)
    assert "closing_line" in message and "TIME_TO_MATCH_MIN" in message


def test_declaring_intermediate_on_a_closing_frame_is_allowed():
    """The harmless direction. A closing-line frame has one row per game, so
    the policy resolves to a view of every row and changes nothing."""
    df = _closing_frame()
    assert_dataset_type_matches_frame(df, DatasetType.INTERMEDIATE_LINE)


def test_policy_mapping_covers_every_dataset_type():
    """A new DatasetType member must be given a branch in redundancy_policy_for
    rather than silently falling through to the closing-line default."""
    policies = {t: redundancy_policy_for(t) for t in DatasetType}

    assert policies[DatasetType.CLOSING_LINE] is None
    assert policies[DatasetType.INTERMEDIATE_LINE] is not None
    assert policies[DatasetType.INTERMEDIATE_LINE].group_col == "GAME_ID"
    assert policies[DatasetType.INTERMEDIATE_LINE].snapshot_col == "TIME_TO_MATCH_MIN"


def test_duplicate_rows_alone_do_not_switch_a_closing_frame_over():
    """A closing-line CSV that picked up a duplicated GAME_ID is a data bug, and
    must stay one. Switching cleaning modes underneath it would hide the bug
    behind a second, unrelated behaviour change."""
    df = _closing_frame()
    doubled = pd.concat([df, df.iloc[:5]]).reset_index(drop=True)
    assert doubled["GAME_ID"].duplicated().any()

    _, report = _clean(doubled)

    assert report.redundancy_view == {}
    assert GROUPED_STEP not in report.columns_by_step()


def test_a_frame_with_no_game_id_is_left_alone():
    _, report = _clean(_intermediate_frame(snapshots=(60,)).drop(columns=["GAME_ID"]))

    assert report.redundancy_view == {}
    assert GROUPED_STEP not in report.columns_by_step()


@pytest.mark.parametrize("mode", ["closing", "intermediate"])
def test_both_modes_report_which_one_ran(mode):
    """Whichever mode ran must be readable from the report alone, without
    re-running: exactly one of the two correlation steps appears, and
    redundancy_view is populated only for the grouped one."""
    if mode == "closing":
        _, report = _clean(_closing_frame())
    else:
        _, report = _clean_intermediate(_intermediate_frame())

    steps = report.columns_by_step()
    grouped = GROUPED_STEP in steps
    plain = CORRELATED_STEP in steps
    assert grouped != plain
    assert bool(report.redundancy_view) == grouped
