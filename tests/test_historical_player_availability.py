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
    }


def _players(*, target_star_minutes: float, target_star_points: float | None):
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
                comment=("DNP - Coach's Decision" if target_star_minutes == 0 else ""),
            ),
            _player_row(TARGET_GAME, "2024-10-24", "rotation", 15.0, 24.0),
        ]
    )
    return pd.DataFrame(rows)


def _run(
    *,
    target_star_minutes: float = 0.0,
    target_star_points: float | None = None,
    injuries: pd.DataFrame | None = None,
) -> pd.Series:
    if injuries is None:
        injuries = pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"])
    out, _ = add_player_history_features(
        _team_rows(),
        _players(
            target_star_minutes=target_star_minutes,
            target_star_points=target_star_points,
        ),
        injuries,
        stat_cols=["PTS"],
    )
    return out.loc[out["GAME_ID"].eq(TARGET_GAME)].iloc[0]


def test_uninjured_dnp_remains_in_the_available_top() -> None:
    target = _run()

    assert target["TOP1_PLAYER_ID_PTS_BEFORE"] == "star"
    assert target["TOP1_PLAYER_PTS_BEFORE"] == pytest.approx(30.0)


def test_target_game_minutes_and_points_cannot_change_player_features() -> None:
    dnp = _run(target_star_minutes=0.0, target_star_points=None)
    played = _run(target_star_minutes=41.0, target_star_points=55.0)

    feature_columns = [column for column in dnp.index if column.endswith("_BEFORE")]
    pd.testing.assert_series_equal(
        dnp[feature_columns],
        played[feature_columns],
        check_names=False,
        check_dtype=False,
    )


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
