"""``filter_valid_games``: complete games, without preseason and All-Star."""

import pandas as pd
from nba_ou.data_processing.team.filters import filter_valid_games


def _team_rows(game_id, teams):
    return [
        {
            "GAME_ID": game_id,
            "TEAM_ID": team,
            "TEAM_NAME": f"Team {team}",
            "GAME_DATE": "2024-10-22",
            "SEASON_ID": "22024",
        }
        for team in teams
    ]


def test_exhibitions_go_before_the_incomplete_entry_check(capsys):
    df = pd.DataFrame(
        _team_rows("0012400001", [1])  # preseason against a non-NBA club
        + _team_rows("0032400001", [1, 2])  # All-Star
        + _team_rows("0022400001", [1, 2])
        + _team_rows("0022400002", [3])  # an incomplete regular-season game
    )
    out = filter_valid_games(df)

    assert out["GAME_ID"].unique().tolist() == ["0022400001"]
    assert out["SEASON_TYPE"].unique().tolist() == ["Regular Season"]
    warning = capsys.readouterr().out
    assert "Filtering out 1 games" in warning
    assert "0022400002" in warning and "0012400001" not in warning
