"""Historical player availability must not depend on target-game minutes."""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.players.attach_player_features import (
    add_player_history_features,
)

TEAM = "1610612738"
OTHER_TEAM = "1610612744"
SEASON_ID = "22024"
SEASON_YEAR = 2024
TARGET_GAME = "0022400003"


def _team_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["0022400001", "0022400002", TARGET_GAME],
            "TEAM_ID": [TEAM] * 3,
            "SEASON_ID": [SEASON_ID] * 3,
            "SEASON_YEAR": [SEASON_YEAR] * 3,
            "GAME_DATE": pd.to_datetime(["2024-10-20", "2024-10-22", "2024-10-24"]),
        }
    )


def _player_row(
    game_id: str,
    game_date: str,
    player_id: str,
    points: float | None,
    minutes: float,
    *,
    team_id: str = TEAM,
    comment: str = "",
) -> dict:
    return {
        "GAME_ID": game_id,
        "TEAM_ID": team_id,
        "SEASON_ID": SEASON_ID,
        "SEASON_YEAR": SEASON_YEAR,
        "GAME_DATE": pd.Timestamp(game_date),
        "PLAYER_ID": player_id,
        "PLAYER_NAME": f"Player {player_id}",
        "MIN": minutes,
        "PTS": points,
        "PACE_PER40": 99.0 if minutes > 0 else None,
        "COMMENT": comment,
    }


def _players(
    *,
    target_star_minutes: float,
    target_star_points: float | None,
    target_star_comment: str | None = None,
):
    rows = []
    for game_id, game_date in [
        ("0022400001", "2024-10-20"),
        ("0022400002", "2024-10-22"),
    ]:
        rows.extend(
            [
                _player_row(game_id, game_date, "star", 30.0, 32.0),
                _player_row(game_id, game_date, "rotation", 15.0, 24.0),
            ]
        )
    rows.extend(
        [
            _player_row(
                TARGET_GAME,
                "2024-10-24",
                "star",
                target_star_points,
                target_star_minutes,
                comment=(
                    target_star_comment
                    if target_star_comment is not None
                    else ("DNP - Coach's Decision" if target_star_minutes == 0 else "")
                ),
            ),
            _player_row(TARGET_GAME, "2024-10-24", "rotation", 15.0, 24.0),
        ]
    )
    return pd.DataFrame(rows)


def _run(
    *,
    target_star_minutes: float = 0.0,
    target_star_points: float | None = None,
    target_star_comment: str | None = None,
    injuries: pd.DataFrame | None = None,
) -> pd.Series:
    if injuries is None:
        injuries = pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"])
    out, _ = add_player_history_features(
        _team_rows(),
        _players(
            target_star_minutes=target_star_minutes,
            target_star_points=target_star_points,
            target_star_comment=target_star_comment,
        ),
        injuries,
        stat_cols=["PTS"],
    )
    return out.loc[out["GAME_ID"].eq(TARGET_GAME)].iloc[0]


def test_unexplained_dnp_stays_available() -> None:
    """A DNP with no reportable reason is available, whatever the box score says.

    The star sits with "DNP - Coach's Decision" and zero minutes. Neither fact
    may move them into the injured bucket: a coach's decision is never on the
    pre-game injury report, so production would see this player as available,
    and target-game minutes cannot tell a coach's decision apart from a late
    scratch in the first place.
    """
    out, injury_dict, availability_dict = add_player_history_features(
        _team_rows(),
        _players(target_star_minutes=0.0, target_star_points=None),
        pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"]),
        stat_cols=["PTS"],
        return_availability_dict=True,
    )
    target = out.loc[out["GAME_ID"].eq(TARGET_GAME)].iloc[0]

    assert target["TOP1_PLAYER_ID_PTS_BEFORE"] == "star"
    assert pd.isna(target["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"])
    assert target["N_INJURED_PLAYERS_BEFORE"] == 0
    assert injury_dict == {}
    assert "star" in availability_dict[TARGET_GAME][TEAM]["available"]
    assert availability_dict[TARGET_GAME][TEAM]["injured"] == []


def test_boxscore_injury_comment_remains_an_injury_report_source() -> None:
    players = _players(target_star_minutes=0.0, target_star_points=None)
    target_star = players["GAME_ID"].eq(TARGET_GAME) & players["PLAYER_ID"].eq("star")
    players.loc[target_star, "COMMENT"] = "DND - Injury/Illness"

    _, injury_dict = add_player_history_features(
        _team_rows(),
        players,
        pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"]),
        stat_cols=["PTS"],
    )

    assert injury_dict[TARGET_GAME][TEAM] == ["star"]


def _roster_split(target: pd.Series) -> tuple[list, list]:
    """Available and injured slot ids, with empty slots normalised for equality."""

    def _slots(template: str, count: int) -> list:
        values = [target[template.format(i=i)] for i in range(1, count + 1)]
        return [None if pd.isna(value) else value for value in values]

    return (
        _slots("TOP{i}_PLAYER_ID_PTS_BEFORE", 6),
        _slots("TOP{i}_INJURED_PLAYER_ID_PTS_BEFORE", 4),
    )


@pytest.mark.parametrize(
    "comment",
    [
        "DNP - Injury/Illness",
        "DND - Injury/Illness - Left Knee; Soreness",
        "DNP - Rest",
        "DNP - Personal Reasons",
        "NWT - League Suspension",
        "NWT - Trade Pending",
    ],
)
def test_absence_reasons_visible_on_the_injury_report_count_as_injured(
    comment: str,
) -> None:
    """Every reason here would also have been published pre-game.

    That is the whole selection rule: the box score records the reason after
    the fact, but the decision behind each of these was made and posted before
    tip-off, so the training label stays reproducible from the injury report at
    prediction time.
    """
    target = _run(target_star_minutes=0.0, target_star_comment=comment)
    available, injured = _roster_split(target)

    assert "star" in injured, comment
    assert "star" not in available, comment


@pytest.mark.parametrize(
    "comment",
    [
        "DNP - Coach's Decision",
        "NWT - G League - Two-Way",
        "NWT - G-League Assignment",
    ],
)
def test_reasons_that_do_not_count_as_an_absence(comment: str) -> None:
    """Neither reason belongs in the injured bucket, for different causes.

    A coach's decision is never published pre-game, so production could not see
    it. A G League assignment is published, but it is a roster fact rather than
    an absence: a two-way player was not part of the rotation being measured, so
    counting them would inflate the injured aggregates with players whose
    absence costs the team nothing.
    """
    target = _run(target_star_minutes=0.0, target_star_comment=comment)
    available, injured = _roster_split(target)

    assert "star" in available, comment
    assert "star" not in injured, comment


def test_target_game_minutes_cannot_move_a_player_between_buckets() -> None:
    """The roster split must be identical whether or not the star played.

    Membership that depends on a target-game quantity is the rotation-depth
    leak documented in ``nba_ou.config.leakage``; holding the reason fixed and
    varying only minutes is the direct check that it has not come back.
    """
    played = _run(target_star_minutes=30.0, target_star_comment="")
    sat = _run(target_star_minutes=0.0, target_star_comment="")

    assert _roster_split(played) == _roster_split(sat)


def test_target_game_points_cannot_change_player_features() -> None:
    ordinary = _run(target_star_minutes=41.0, target_star_points=20.0)
    high_scoring = _run(target_star_minutes=41.0, target_star_points=55.0)

    feature_columns = [
        column for column in ordinary.index if column.endswith("_BEFORE")
    ]
    pd.testing.assert_series_equal(
        ordinary[feature_columns],
        high_scoring[feature_columns],
        check_names=False,
        check_dtype=False,
    )


def test_scheduled_placeholder_is_not_treated_as_a_dnp() -> None:
    target = _run(target_star_minutes=float("nan"), target_star_points=None)

    assert target["TOP1_PLAYER_ID_PTS_BEFORE"] == "star"
    assert pd.isna(target["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"])


def test_injury_membership_overrides_a_same_game_roster_row() -> None:
    injuries = pd.DataFrame(
        [{"GAME_ID": TARGET_GAME, "TEAM_ID": TEAM, "PLAYER_ID": "star"}]
    )

    target = _run(injuries=injuries)

    assert target["TOP1_PLAYER_ID_PTS_BEFORE"] == "rotation"
    assert target["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"] == "star"
    assert target["N_INJURED_PLAYERS_BEFORE"] == 1


def test_same_game_boxscore_does_not_assign_a_healthy_trade_debut() -> None:
    team_rows = _team_rows()
    players = _players(target_star_minutes=24.0, target_star_points=15.0)
    trade_history = pd.DataFrame(
        [
            _player_row(
                "0022400000",
                "2024-10-18",
                "new_star",
                35.0,
                34.0,
                team_id=OTHER_TEAM,
            ),
            _player_row(
                TARGET_GAME,
                "2024-10-24",
                "new_star",
                None,
                0.0,
                team_id=TEAM,
                comment="DNP - Coach's Decision",
            ),
        ]
    )
    players = pd.concat([players, trade_history], ignore_index=True)

    out, _ = add_player_history_features(
        team_rows,
        players,
        pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"]),
        stat_cols=["PTS"],
    )
    target = out.loc[out["GAME_ID"].eq(TARGET_GAME)].iloc[0]

    top_ids = [target[f"TOP{i}_PLAYER_ID_PTS_BEFORE"] for i in range(1, 7)]
    assert "new_star" not in top_ids


def _traded_player_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _player_row(
                "0022400000",
                "2024-10-18",
                "new_star",
                35.0,
                34.0,
                team_id=OTHER_TEAM,
            ),
            _player_row(
                "0022400000b",
                "2024-10-20",
                "new_star",
                35.0,
                34.0,
                team_id=OTHER_TEAM,
            ),
        ]
    )


@pytest.mark.parametrize("scheduled", [False, True], ids=["historical", "scheduled"])
def test_injury_report_assigns_trade_to_reported_team(scheduled: bool) -> None:
    players = pd.concat(
        [
            _players(target_star_minutes=24.0, target_star_points=15.0),
            _traded_player_history(),
        ],
        ignore_index=True,
    )
    injury_entry = {TARGET_GAME: {TEAM: ["new_star"]}}
    historical_injuries = (
        pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"])
        if scheduled
        else pd.DataFrame(
            [{"GAME_ID": TARGET_GAME, "TEAM_ID": TEAM, "PLAYER_ID": "new_star"}]
        )
    )

    other_team_target = pd.DataFrame(
        [
            {
                "GAME_ID": TARGET_GAME,
                "TEAM_ID": OTHER_TEAM,
                "SEASON_ID": SEASON_ID,
                "SEASON_YEAR": SEASON_YEAR,
                "GAME_DATE": pd.Timestamp("2024-10-24"),
            }
        ]
    )
    team_rows = pd.concat([_team_rows(), other_team_target], ignore_index=True)

    out, _ = add_player_history_features(
        team_rows,
        players,
        historical_injuries,
        stat_cols=["PTS"],
        injury_dict_scheduled=injury_entry if scheduled else None,
    )
    target_game = out.loc[out["GAME_ID"].eq(TARGET_GAME)]
    target = target_game.loc[target_game["TEAM_ID"].eq(TEAM)].iloc[0]
    previous_team = target_game.loc[target_game["TEAM_ID"].eq(OTHER_TEAM)].iloc[0]

    assert target["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"] == "new_star"
    assert target["TOP1_INJURED_PLAYER_PTS_BEFORE"] == pytest.approx(35.0)
    assert pd.isna(previous_team["TOP1_INJURED_PLAYER_ID_PTS_BEFORE"])
