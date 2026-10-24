"""IS_OVERTIME must survive into the training CSV.

It used to be dropped in apply_final_transformations, which made it impossible
to filter overtime games out of training while still scoring them. It is a
post-game fact and must never be a model FEATURE -- training_pipeline enforces
that separately -- but it has to exist as a row attribute for the filter to be
possible at all.
"""

import pandas as pd
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    apply_final_transformations,
)
from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
    STATIC_COLUMNS,
)


def _minimal_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MATCHUP_TEAM_HOME": [0],
            "MATCHUP_TEAM_AWAY": [0],
            "TEAM_ABBREVIATION_TEAM_HOME": ["BOS"],
            "TEAM_ABBREVIATION_TEAM_AWAY": ["LAL"],
            "IS_OVERTIME": [1],
            "TOTAL_POINTS": [235.0],
        }
    )


def test_is_overtime_survives_final_transformations():
    result = apply_final_transformations(_minimal_training_frame())
    assert "IS_OVERTIME" in result.columns
    assert result["IS_OVERTIME"].iloc[0] == 1


def test_is_overtime_is_whitelisted_as_a_static_column():
    """It reaches the CSV the same way GAME_ID and SEASON_TYPE do -- as a row
    attribute, not as a _BEFORE feature.
    """
    assert "IS_OVERTIME" in STATIC_COLUMNS


def test_training_pipeline_still_refuses_it_as_a_feature():
    """Retaining the column must not make it usable as an input: an overtime
    game already played five extra minutes and therefore already scored more.
    """
    import pytest

    from training_pipeline.config import LEAKING_TARGET_COLUMNS
    from training_pipeline.data import assert_no_leaking_features

    assert "IS_OVERTIME" in LEAKING_TARGET_COLUMNS
    with pytest.raises(ValueError):
        assert_no_leaking_features(pd.DataFrame({"IS_OVERTIME": [1]}))
