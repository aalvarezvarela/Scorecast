import pandas as pd
import pytest
from nba_ou.data_processing.lineups.rating_gate import (
    ratings_only_game_projections,
    score_rating_gate,
)


def _gate_fixture():
    games = []
    players = []
    for number, (date, points) in enumerate(
        [
            ("2025-01-01", 120),
            ("2025-01-02", 120),
            ("2025-01-03", 120),
            ("2025-01-04", 105),
        ],
        start=1,
    ):
        game_id = f"002250000{number}"
        games.extend(
            [
                {
                    "GAME_ID": game_id,
                    "GAME_DATE": date,
                    "TEAM_ID": "1",
                    "HOME": True,
                    "PTS": points,
                },
                {
                    "GAME_ID": game_id,
                    "GAME_DATE": date,
                    "TEAM_ID": "2",
                    "HOME": False,
                    "PTS": points,
                },
            ]
        )
        for team, ids in (("1", range(1, 6)), ("2", range(11, 16))):
            players.extend(
                {
                    "GAME_ID": game_id,
                    "TEAM_ID": team,
                    "PLAYER_ID": str(pid),
                    "MIN": 48.0,
                }
                for pid in ids
            )
    ratings = pd.DataFrame(
        [
            {
                "as_of_date": "2025-01-04",
                "player_id": str(pid),
                "o_rating": 1.0,
                "d_rating": 0.0,
                "pace_rating": 0.0,
                "league_ortg": 100.0,
                "league_pace": 100.0,
                "fit_max_game_date": "2025-01-03",
            }
            for pid in (*range(1, 6), *range(11, 16))
        ]
    )
    return pd.DataFrame(games), pd.DataFrame(players), ratings


def test_rating_gate_beats_prior_team_total_baseline():
    games, players, ratings = _gate_fixture()
    projections = ratings_only_game_projections(
        games,
        players,
        ratings,
        validation_from="2025-01-04",
        validation_to="2025-01-04",
    )
    assert projections.proj_total.iloc[0] == pytest.approx(210.0)
    assert projections.baseline_total.iloc[0] == pytest.approx(240.0)
    assert projections.fit_max_game_date.lt(projections.game_date).all()
    assert score_rating_gate(projections) == {
        "games": 1,
        "ratings_mae": pytest.approx(0.0),
        "baseline_mae": pytest.approx(30.0),
        "mae_improvement": pytest.approx(30.0),
        "passed": True,
    }


def test_target_game_box_minutes_cannot_change_its_projection():
    games, players, ratings = _gate_fixture()
    baseline = ratings_only_game_projections(
        games,
        players,
        ratings,
        validation_from="2025-01-04",
        validation_to="2025-01-04",
    )
    players.loc[players.GAME_ID.eq("0022500004"), "MIN"] = 0
    changed = ratings_only_game_projections(
        games,
        players,
        ratings,
        validation_from="2025-01-04",
        validation_to="2025-01-04",
    )
    pd.testing.assert_frame_equal(baseline, changed)


def test_baseline_sums_each_teams_prior_points_scored():
    games, players, ratings = _gate_fixture()
    prior = games.GAME_DATE.lt("2025-01-04")
    games.loc[prior & games.TEAM_ID.eq("1"), "PTS"] = 130
    games.loc[prior & games.TEAM_ID.eq("2"), "PTS"] = 100
    projections = ratings_only_game_projections(
        games,
        players,
        ratings,
        validation_from="2025-01-04",
        validation_to="2025-01-04",
    )
    assert projections.baseline_total.iloc[0] == pytest.approx(230.0)


def test_rating_gate_rejects_same_day_or_future_fits():
    games, players, ratings = _gate_fixture()
    ratings["fit_max_game_date"] = ratings["as_of_date"]
    with pytest.raises(ValueError, match="strictly before"):
        ratings_only_game_projections(
            games,
            players,
            ratings,
            validation_from="2025-01-04",
            validation_to="2025-01-04",
        )


def test_long_absent_player_does_not_absorb_projected_minutes():
    """A player who missed the whole recent window is not projected minutes."""
    games, players, ratings = _gate_fixture()
    # A star who only played the first game, then never appeared again, and
    # whose rating is far from his replacements'.
    players = pd.concat(
        [
            players,
            pd.DataFrame(
                [
                    {
                        "GAME_ID": "0022500001",
                        "TEAM_ID": "1",
                        "PLAYER_ID": "99",
                        "MIN": 48.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ratings = pd.concat(
        [
            ratings,
            pd.DataFrame(
                [
                    {
                        "as_of_date": "2025-01-04",
                        "player_id": "99",
                        "o_rating": 50.0,
                        "d_rating": 0.0,
                        "pace_rating": 0.0,
                        "league_ortg": 100.0,
                        "league_pace": 100.0,
                        "fit_max_game_date": "2025-01-03",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    projections = ratings_only_game_projections(
        games,
        players,
        ratings,
        validation_from="2025-01-04",
        validation_to="2025-01-04",
        recent_games=2,
    )
    assert projections.proj_total.iloc[0] == pytest.approx(210.0)
