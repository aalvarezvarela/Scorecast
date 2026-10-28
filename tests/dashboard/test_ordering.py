import pandas as pd
from nba_ou.dashboard.domain.ordering import order_board

from .conftest import NOW


def test_largest_edge_ranks_games_and_table_rows_with_missing_edges_last():
    games = pd.DataFrame(
        {
            "game_id": [1, 2, 3],
            "tipoff_at": [
                NOW + pd.Timedelta(hours=4),
                NOW + pd.Timedelta(hours=2),
                NOW + pd.Timedelta(hours=3),
            ],
        }
    )
    predictions = pd.DataFrame(
        {
            "game_id": [2, 1, 1, 3, 1],
            "comparison_edge": [5, 2, -8, float("nan"), float("nan")],
            "dataset_type": [1, 1, 2, 1, 2],
            "family_key": ["a", "a", "a", "a", "b"],
        }
    )
    ordered_games, ordered_rows = order_board(games, predictions, "Largest edge")
    assert ordered_games["game_id"].tolist() == [1, 2, 3]
    assert ordered_rows["game_id"].tolist() == [1, 1, 1, 2, 3]
    assert ordered_rows["comparison_edge"].iloc[:2].tolist() == [-8, 2]
    assert pd.isna(ordered_rows["comparison_edge"].iloc[2])
    tipoff_games, tipoff_rows = order_board(games, predictions, "Tip-off")
    assert tipoff_games["game_id"].tolist() == [2, 3, 1]
    assert tipoff_rows["game_id"].tolist() == [2, 3, 1, 1, 1]
    assert games["game_id"].tolist() == [1, 2, 3]


def test_equal_game_edges_use_tipoff_and_empty_predictions_still_sort_games(today_data):
    predictions = pd.DataFrame(
        {
            "game_id": [1, 2],
            "comparison_edge": [-2, 2],
            "dataset_type": [1, 1],
            "family_key": ["a", "a"],
        }
    )
    games, rows = order_board(today_data.games, predictions, "Largest edge")
    assert games["game_id"].tolist() == [2, 1]
    assert rows["game_id"].tolist() == [2, 1]
    games, rows = order_board(today_data.games, predictions.iloc[:0], "Largest edge")
    assert games["game_id"].tolist() == [2, 1]
    assert rows.empty
