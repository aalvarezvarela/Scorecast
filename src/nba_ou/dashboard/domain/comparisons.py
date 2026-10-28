"""Keep a model's original anchor separate from the market used for comparison."""

import math

import pandas as pd

from nba_ou.dashboard.config import DEFAULT_MIN_EDGE_POINTS, REFERENCE_COLUMNS


def signal(
    value, *, spread: bool = False, min_edge: float = DEFAULT_MIN_EDGE_POINTS
) -> str:
    if not math.isfinite(min_edge) or min_edge < 0:
        raise ValueError("min_edge must be a finite, non-negative number of points")
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "UNAVAILABLE"
    magnitude = abs(float(value))
    # Evaluation places bets only for abs(edge) > min_edge. Avoid turning
    # subtraction noise at the boundary (e.g. 0.10000000000000009) into a pick.
    if magnitude <= min_edge or math.isclose(
        magnitude, min_edge, rel_tol=0, abs_tol=1e-9
    ):
        return "NO EDGE"
    if spread:
        return "HOME" if value > 0 else "AWAY"
    return "OVER" if value > 0 else "UNDER"


def compare_predictions(
    selected: pd.DataFrame,
    markets: pd.DataFrame,
    reference: str = "Bet365",
    *,
    min_edge: float = DEFAULT_MIN_EDGE_POINTS,
) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    work = selected.copy()
    output = pd.to_numeric(work["prediction_value"], errors="coerce")
    anchor = pd.to_numeric(work["model_line"], errors="coerce")
    spread = work["target"].eq(3)
    work["predicted_total"] = output.where(work["target"].eq(2), anchor + output)
    work.loc[spread, "predicted_total"] = float("nan")
    work["predicted_home_margin"] = (anchor + output).where(spread)
    work["predicted_error"] = output.where(~work["target"].eq(2), output - anchor)
    market_columns = [
        "game_id",
        "bet365_line",
        "books_mean_line",
        "books_median_line",
        "market_at",
    ]
    # Drop at-prediction market fields before attaching the shared current snapshot.
    work = work.drop(columns=[c for c in market_columns[1:] if c in work])
    if markets.empty:
        for column in market_columns[1:]:
            work[column] = pd.NaT if column == "market_at" else float("nan")
    else:
        work = work.merge(
            markets[market_columns], on="game_id", how="left", validate="many_to_one"
        )
    implied_prediction = work["predicted_total"].where(
        ~work["target"].eq(3), work["predicted_home_margin"]
    )
    for column in REFERENCE_COLUMNS.values():
        work[f"edge_{column}"] = implied_prediction - pd.to_numeric(
            work[column], errors="coerce"
        )
    work["comparison_edge"] = work[f"edge_{REFERENCE_COLUMNS[reference]}"]
    work["signal"] = [
        signal(edge, spread=int(target) == 3, min_edge=min_edge)
        for edge, target in zip(work["comparison_edge"], work["target"], strict=True)
    ]
    return work
