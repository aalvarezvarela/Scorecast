"""Keep card and table ordering consistent, including missing market edges."""

import pandas as pd


def order_board(
    games: pd.DataFrame, predictions: pd.DataFrame, sort_by: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_games = games.sort_values(["tipoff_at", "game_id"], kind="stable")
    if predictions.empty:
        return ordered_games, predictions.copy()
    ordered_predictions = predictions.assign(
        _abs_edge=predictions["comparison_edge"].abs()
    )
    if sort_by == "Largest edge":
        rank = ordered_predictions.groupby("game_id")["_abs_edge"].max()
        ordered_games = (
            ordered_games.assign(_rank=ordered_games["game_id"].map(rank))
            .sort_values(
                ["_rank", "tipoff_at", "game_id"],
                ascending=[False, True, True],
                kind="stable",
                na_position="last",
            )
            .drop(columns="_rank")
        )
    positions = {
        game_id: position for position, game_id in enumerate(ordered_games["game_id"])
    }
    ordered_predictions["_game_order"] = ordered_predictions["game_id"].map(positions)
    columns = ["_game_order"]
    ascending = [True]
    if sort_by == "Largest edge":
        columns.append("_abs_edge")
        ascending.append(False)
    columns.extend(["dataset_type", "family_key"])
    ascending.extend([True, True])
    ordered_predictions = ordered_predictions.sort_values(
        columns, ascending=ascending, kind="stable", na_position="last"
    ).drop(columns=["_abs_edge", "_game_order"])
    return ordered_games, ordered_predictions
