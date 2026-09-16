"""The referee crew can only enter a snapshot after its game-day release."""

import numpy as np
import pandas as pd
from nba_ou.create_training_data.intermediate_referees import (
    _legacy_referee_features,
    add_snapshot_referee_interactions,
    mask_referees_before_release,
)
from nba_ou.create_training_data.select_intermediate_columns import (
    select_intermediate_training_columns,
)


def test_referees_are_masked_until_nine_eastern_on_game_day():
    # March 8, 2026 is the spring DST change: 09:00 ET is 13:00 UTC.
    tipoff = pd.Timestamp("2026-03-08 23:00:00+00:00")
    snapshots = pd.to_datetime(
        [
            "2026-03-07 15:00:00+00:00",  # 10:00 ET on the previous day
            "2026-03-08 12:59:00+00:00",  # 08:59 EDT
            "2026-03-08 13:00:00+00:00",  # 09:00 EDT
            "2026-03-08 13:01:00+00:00",
        ],
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "GAME_ID": ["1"] * 4,
            "TIPOFF_UTC": [tipoff] * 4,
            "SNAPSHOT_TS_UTC": snapshots,
            "REF_CREW_FTA_TENDENCY_BEFORE": [0.5] * 4,
            "REF_AVG_TOTAL_POINTS_DIFF_BEFORE": [2.0] * 4,
            "REF_CREW_FTA_X_ABS_SPREAD_BEFORE": [1.5] * 4,
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME": [110.0] * 4,
        }
    )
    masked = mask_referees_before_release(frame)
    referee_columns = [c for c in frame if c.startswith("REF_")]
    assert masked.loc[:1, referee_columns].isna().all().all()
    assert (
        (masked.loc[2:, referee_columns] == frame.loc[2:, referee_columns]).all().all()
    )
    assert masked["PTS_SEASON_BEFORE_AVG_TEAM_HOME"].eq(110.0).all()
    assert frame[referee_columns].notna().all().all()  # input is untouched

    gated = select_intermediate_training_columns(masked)
    assert set(referee_columns).issubset(gated.columns)
    assert np.isnan(gated.loc[0, "REF_CREW_FTA_TENDENCY_BEFORE"])


def test_nine_eastern_uses_winter_utc_offset():
    frame = pd.DataFrame(
        {
            "TIPOFF_UTC": ["2026-01-12 23:00:00+00:00"] * 2,
            "SNAPSHOT_TS_UTC": [
                "2026-01-12 13:59:00+00:00",
                "2026-01-12 14:00:00+00:00",
            ],
            "REF_CREW_PF_TENDENCY_BEFORE": [0.3, 0.3],
        }
    )
    masked = mask_referees_before_release(frame)
    assert pd.isna(masked.loc[0, "REF_CREW_PF_TENDENCY_BEFORE"])
    assert masked.loc[1, "REF_CREW_PF_TENDENCY_BEFORE"] == 0.3


def test_legacy_referee_history_excludes_the_current_game_result():
    games = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "GAME_DATE": pd.to_datetime(
                ["2025-11-01"] * 2 + ["2025-11-02"] * 2 + ["2025-11-03"] * 2
            ),
            "SEASON_YEAR": [2025] * 6,
            "PTS": [50, 50, 60, 60, 70, 70],
            "PF": [10] * 6,
        }
    )
    odds = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "total_bet365_line_over": [105.0, 115.0, 130.0],
        }
    )
    refs = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2", "g3"],
            "FIRST_NAME": ["Ref", "Ref", "Ref"],
            "LAST_NAME": ["A", "B", "A"],
        }
    )
    initial = _legacy_referee_features(
        games, odds, refs, first_output_year=2025, book="bet365"
    ).set_index("GAME_ID")
    changed = games.copy()
    changed.loc[changed.GAME_ID.eq("g3"), "PTS"] = 500
    rebuilt = _legacy_referee_features(
        changed, odds, refs, first_output_year=2025, book="bet365"
    ).set_index("GAME_ID")
    column = "REF_AVG_TOTAL_POINTS_DIFF_BEFORE"
    assert initial.loc["g3", column] == -20.0
    assert rebuilt.loc["g3", column] == initial.loc["g3", column]


def test_referee_spread_interaction_uses_snapshot_quote():
    frame = pd.DataFrame(
        {
            "ODDS_SPREAD_bet365": [-15.0],  # closing handicap from the base frame
            "ODDS_SPREAD_LINE_HOME_bet365": [2.0],  # snapshot home margin
            "REF_CREW_FTA_TENDENCY_BEFORE": [0.5],
            "REF_CREW_POSS_TENDENCY_BEFORE": [0.25],
        }
    )
    result = add_snapshot_referee_interactions(frame, book="bet365")
    assert result.loc[0, "REF_CREW_FTA_X_ABS_SPREAD_BEFORE"] == 1.0
    assert result.loc[0, "REF_CREW_POSS_X_ABS_SPREAD_BEFORE"] == 0.5
    assert "ODDS_SPREAD_bet365" not in result.columns
    assert frame.loc[0, "ODDS_SPREAD_bet365"] == -15.0
