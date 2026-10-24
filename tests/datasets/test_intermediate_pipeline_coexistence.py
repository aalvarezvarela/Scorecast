"""The intermediate pipeline must not change the closing-line pipeline.

Both datasets are built from the same feature functions. The intermediate one
composes them differently and adds a stricter gate on top; it must never patch,
wrap or reconfigure anything the existing pipeline depends on.

These are contract tests rather than an end-to-end comparison: building either
dataset needs database access and takes minutes, which does not belong in the
unit suite. What is checked here is every seam where a change could leak across
-- the shared entry points still existing with the signatures the new code
relies on, and importing the new pipeline leaving the old one's behaviour
untouched.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest


def test_shared_team_pipeline_keeps_the_signature_the_new_code_calls():
    """``create_base_game_features`` calls this with keyword arguments.

    If it is ever renamed or its keywords change, the intermediate pipeline
    should fail here rather than at dataset-build time.
    """
    from nba_ou.create_training_data.create_df_to_predict import (
        process_team_statistics_for_training,
    )

    parameters = inspect.signature(process_team_statistics_for_training).parameters
    for expected in ["df", "df_odds", "scheduled_games", "spread_ml_book"]:
        assert expected in parameters


def test_importing_the_intermediate_pipeline_does_not_alter_the_existing_gate():
    """No monkeypatching of the shared leakage filter."""
    from nba_ou.data_processing.merged_home_away_data import select_train_columns

    before = select_train_columns.select_training_columns

    import nba_ou.create_training_data.create_intermediate_line_df  # noqa: F401

    assert select_train_columns.select_training_columns is before


def test_the_two_gates_disagree_exactly_where_they_are_meant_to():
    """The stricter gate is stricter, and specifically about closing prices.

    ``_BEFORE`` means "safe" in the closing-line dataset and does not here: a
    closing line is known when that model bets and unknown when this one does.
    ``IMPLIED_PTS_*_BEFORE`` is the sharpest case -- the two columns sum to the
    closing line exactly.
    """
    from nba_ou.create_training_data.select_intermediate_columns import is_kept_column

    frame = pd.DataFrame(
        {
            "GAME_ID": ["0022300001"],
            "SEASON_YEAR": [2023],
            "TOTAL_POINTS": [221.0],
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME": [112.0],
            "IMPLIED_PTS_HOME_BEFORE": [113.0],
            "IMPLIED_PTS_AWAY_BEFORE": [111.5],
            "THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE": [0.4],
        }
    )

    # The existing gate keeps every _BEFORE column, which is correct for it.
    from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
        select_training_columns,
    )

    existing = select_training_columns(frame, original_columns=[])
    assert "IMPLIED_PTS_HOME_BEFORE" in existing.columns

    # The intermediate gate does not.
    assert not is_kept_column("IMPLIED_PTS_HOME_BEFORE")
    assert not is_kept_column("THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE")
    assert is_kept_column("PTS_SEASON_BEFORE_AVG_TEAM_HOME")


def test_placeholder_injection_does_not_mutate_its_input():
    """The shim must not write player columns back into a caller's frame."""
    from nba_ou.create_training_data.create_base_game_features import (
        _inject_player_placeholders,
    )

    original = pd.DataFrame({"GAME_ID": ["1"]})
    injected = _inject_player_placeholders(original)

    assert "TOP1_PLAYER_PTS_BEFORE" in injected.columns
    assert "TOP1_PLAYER_PTS_BEFORE" not in original.columns


def test_snapshots_of_one_game_never_straddle_a_split():
    """The property that makes a 6-row-per-game panel safe to split by date.

    All six snapshots of a game share a GAME_DATE, and both splitters cut on
    dates -- ``split_latest_days_holdout`` directly, and the walk-forward
    builder by assembling test windows from whole ``unique_dates`` and taking
    the train pool as ``_date < test_start_date``. This is asserted rather than
    assumed because the guarantee silently disappears if ``data.date_col`` is
    ever pointed at the snapshot timestamp: a 12h snapshot falls on the previous
    calendar day.
    """
    import numpy as np
    from nba_ou.modeling.modeling import make_test_anchored_walk_forward_splits

    from training_pipeline.splits import split_latest_days_holdout

    snapshots = [30, 60, 120, 240, 480, 720]
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    frame = pd.DataFrame(
        [
            {
                "GAME_ID": f"00223{index:05d}",
                "GAME_DATE": date,
                "SEASON_YEAR": 2023,
                "TIME_TO_MATCH_MIN": snapshot,
            }
            for index, date in enumerate(dates)
            for snapshot in snapshots
        ]
    )

    dev, test = split_latest_days_holdout(frame, date_col="GAME_DATE", test_days=10)
    assert not set(dev["GAME_ID"]) & set(test["GAME_ID"])

    splits, _info = make_test_anchored_walk_forward_splits(
        dev,
        date_col="GAME_DATE",
        season_col="SEASON_YEAR",
        test_games=60,
        step_games_between_tests=60,
        train_games=120,
        min_train_games=60,
        exclude_test_months=(),
        max_folds=3,
    )
    assert splits
    for train_idx, valid_idx in splits:
        train_games = set(dev.iloc[train_idx]["GAME_ID"])
        valid_games = set(dev.iloc[valid_idx]["GAME_ID"])
        assert not train_games & valid_games
        # And every snapshot of a validation game is present, not a subset.
        counts = dev.iloc[valid_idx].groupby("GAME_ID").size().unique()
        assert set(counts) == {len(snapshots)}
        assert isinstance(train_idx, np.ndarray)


def test_all_star_injury_dependent_families_are_excluded():
    """Team all-star star-power is kept; the injured-player variants are not.

    ``injured_dict`` feeds ONLY the ``*_INJURED_*`` columns, so the team-level
    vote share is available without any injury input. That separation is the
    whole reason all-star can be included in a dataset that must not see
    timestamped injury data.
    """
    from nba_ou.create_training_data.create_base_game_features import (
        _ALL_STAR_KEEP_PREFIXES,
    )

    kept = [
        "ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_HOME",
        "ALL_STAR_MIN_SCORE_BEFORE_TEAM_AWAY",
    ]
    dropped = [
        "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_TEAM_HOME",
        "ALL_STAR_MIN_INJURED_SCORE_BEFORE_TEAM_AWAY",
        "ALL_STAR_CANDIDATE_COUNT_BEFORE",
        "ALL_STAR_SEASON_YEAR_BEFORE",
    ]
    for column in kept:
        assert column.startswith(_ALL_STAR_KEEP_PREFIXES), column
    for column in dropped:
        assert not column.startswith(_ALL_STAR_KEEP_PREFIXES), column


def test_all_star_ballot_is_always_a_completed_one():
    """Leakage: a game must never see the vote published after it was played."""
    from nba_ou.data_processing.all_star_voting.attach_all_star_voting_features import (
        all_star_season_year_for_game_date,
    )

    # A November 2024 game maps to the 2023-24 ballot (voted January 2024),
    # not the January 2025 one that had not happened yet.
    assert all_star_season_year_for_game_date(pd.Timestamp("2024-11-15")) == 2023
    # And a February 2025 game still uses the previous completed ballot.
    assert all_star_season_year_for_game_date(pd.Timestamp("2025-02-15")) == 2023


def test_roster_continuity_accepts_no_injury_reports():
    """``injured_dict=None`` is a supported path, not a workaround.

    Injury reports are one of two sources the function can use to assign a
    player to a team. Dropping that source leaves box scores, which is strictly
    more conservative: a newly acquired player counts once he has played.
    """
    import inspect

    from nba_ou.data_processing.players.roster_continuity import (
        add_roster_continuity_feature,
    )

    annotation = (
        inspect.signature(add_roster_continuity_feature)
        .parameters["injured_dict"]
        .annotation
    )
    assert "None" in str(annotation)


def test_team_identity_is_one_hot_encoded():
    """60 columns, exactly one active per side."""
    from nba_ou.data_processing.merged_home_away_data.team_one_hot_features import (
        add_team_one_hot_features,
    )

    frame = pd.DataFrame(
        {"TEAM_ID_TEAM_HOME": ["1610612738"], "TEAM_ID_TEAM_AWAY": ["1610612747"]}
    )
    out = add_team_one_hot_features(frame, categorical_team_encoding=False)
    home = [
        c for c in out.columns if c.startswith("TEAM_HOME_") and c.endswith("_BEFORE")
    ]
    away = [
        c for c in out.columns if c.startswith("TEAM_AWAY_") and c.endswith("_BEFORE")
    ]
    assert len(home) == 30 and len(away) == 30
    assert out[home].sum(axis=1).tolist() == [1]
    assert out[away].sum(axis=1).tolist() == [1]
    # They carry _BEFORE, which is what gets them through both leakage gates.
    from nba_ou.create_training_data.select_intermediate_columns import is_kept_column

    assert all(is_kept_column(c) for c in home + away)


def test_lagged_injury_dict_never_lets_a_game_see_its_own_report():
    """Roster membership without the same-day timestamp assumption.

    The injury feed is used only to assign a player to a team, never to read his
    status -- but ``add_roster_continuity_feature`` treats a game's own report as
    pre-tip information, which a 12h snapshot cannot assume. Re-keying each
    report onto the team's NEXT game removes that exposure by construction while
    still capturing 74-89% of the report's measured effect.
    """
    from nba_ou.create_training_data.create_base_game_features import (
        _lag_injured_dict_by_one_team_game,
    )

    context = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2", "g3"],
            "TEAM_ID": ["t", "t", "t"],
            "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
        }
    )
    original = {"g1": {"t": ["p1"]}, "g2": {"t": ["p2"]}, "g3": {"t": ["p3"]}}
    lagged = _lag_injured_dict_by_one_team_game(original, context)

    # Each report surfaces one game later, so no game sees its own.
    assert lagged == {"g2": {"t": ["p1"]}, "g3": {"t": ["p2"]}}
    for game_id, team_map in lagged.items():
        for team_id, players in team_map.items():
            assert set(players).isdisjoint(original[game_id][team_id])

    # A team's final game has no successor; its report is dropped, not carried
    # forward into a season it does not belong to.
    assert "g3" not in {g for g, t in lagged.items() if "p3" in t.get("t", [])}


def test_roster_injury_report_mode_is_validated():
    from nba_ou.create_training_data.create_base_game_features import (
        _build_roster_injury_dict,
    )

    assert _build_roster_injury_dict("none", [], pd.DataFrame(), pd.DataFrame()) is None
    with pytest.raises(ValueError, match="none', 'lagged' or 'full'"):
        _build_roster_injury_dict("sometimes", [], pd.DataFrame(), pd.DataFrame())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"combine_fanatics_and_caesars": True, "exclude_fanatics": True},
        {"combine_fanatics_and_caesars": True, "exclude_caesars": True},
        {
            "combine_fanatics_and_caesars": True,
            "exclude_fanatics": True,
            "exclude_caesars": True,
        },
    ],
)
def test_combine_fanatics_and_caesars_rejects_conflicting_exclusions(kwargs):
    """Validated before any DB access, so this needs no mocking."""
    from nba_ou.create_training_data.create_intermediate_line_df import (
        create_intermediate_line_df,
    )

    with pytest.raises(ValueError, match="cannot be combined with"):
        create_intermediate_line_df(**kwargs)
