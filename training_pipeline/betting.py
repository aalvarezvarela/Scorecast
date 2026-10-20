"""Betting-profitability metrics: the decision-relevant scoring layer.

Why this exists: MAE/RMSE against the closing total line move by ~0.1 points
between a good and a bad model, which is noise. What actually determines
whether a model is worth betting is whether its directional calls clear the
break-even win rate implied by the price, *and* whether it does so on enough
bets for that to be distinguishable from luck.

Conventions used here:
- Odds are DECIMAL (the repo's odds pipeline already converts American to
  decimal; see merge_yahoo_sportsbook_odds). -110 American == 1.909091 decimal.
- Break-even win rate for decimal odds d is ``1 / d``. At -110 that is 52.38%.
- Stakes are 1 unit per bet. Profit is ``d - 1`` on a win, ``-1`` on a loss,
  ``0`` on a push (stake returned).
- ROI is profit divided by total units *placed* (pushes included in the
  denominator, since that capital was committed). This is return on turnover.
- The win-rate confidence interval is a Wilson score interval: closed-form,
  deterministic (no bootstrap seed), and well-behaved at small n and at
  proportions near 0.5 -- which is exactly the regime here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

# -110 American odds, the standard price on NBA totals.
DECIMAL_ODDS_MINUS_110 = 1.0 + 100.0 / 110.0

DEFAULT_EDGE_THRESHOLDS: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


#: Column holding the realised outcome in a saved predictions frame.
#:
#: Runs predating the spread market wrote this as ``TOTAL_POINTS``, which was
#: unambiguous while totals were the only market. It no longer is: a spread run's
#: outcome is HOME_MARGIN, and writing home margins into a column named
#: TOTAL_POINTS would be a lie that every reader would believe.
#:
#: New runs therefore write ``actual_outcome`` (always) and ``TOTAL_POINTS``
#: (only when that is genuinely what the outcome is). Read via
#: ``outcome_from_predictions``, which falls back so archived runs keep loading.
OUTCOME_COLUMN = "actual_outcome"


def outcome_from_predictions(predictions: pd.DataFrame) -> np.ndarray:
    """The realised outcome from a predictions frame, old layout or new.

    Prefers ``actual_outcome``; falls back to ``TOTAL_POINTS`` so every run
    already in ``artifacts/experiments`` still reads. Raises rather than
    returning NaNs if neither is present -- a scorer silently producing an
    all-NaN outcome column reports a plausible-looking zero-bet result.
    """
    for column in (OUTCOME_COLUMN, "TOTAL_POINTS"):
        if column in predictions.columns:
            return pd.to_numeric(predictions[column], errors="coerce").to_numpy(
                dtype=float
            )
    raise KeyError(
        f"Predictions frame has neither {OUTCOME_COLUMN!r} nor 'TOTAL_POINTS'; "
        "there is no realised outcome to score against."
    )


def decimal_odds_from_american(american_odds: float) -> float:
    """Convert American odds to decimal odds (stake included)."""
    if american_odds > 0:
        return 1.0 + american_odds / 100.0
    return 1.0 + 100.0 / abs(american_odds)


def break_even_win_rate(decimal_odds: float) -> float:
    """Win rate needed to break even at the given decimal odds."""
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be > 1.0")
    return 1.0 / decimal_odds


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because win counts here are small
    (often tens of bets) and the proportion sits near 0.5, where the naive
    interval is badly calibrated.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))

    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2.0 * trials)) / denominator
    margin = (
        z
        / denominator
        * np.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2))
    )
    return (float(center - margin), float(center + margin))


class BettingMetrics(BaseModel):
    """Outcome of applying a bet-selection rule to one set of predictions."""

    min_edge: float
    n_candidates: int
    n_bets: int
    bet_rate: float
    n_wins: int
    n_losses: int
    n_pushes: int
    win_rate: float | None
    win_rate_ci_low: float | None
    win_rate_ci_high: float | None
    break_even_rate: float | None
    edge_vs_break_even: float | None
    profit_units: float
    roi: float | None
    beats_break_even: bool
    is_significant: bool


def _resolve_prices(
    n: int,
    decimal_odds_over: np.ndarray | None,
    decimal_odds_under: np.ndarray | None,
    flat_decimal_odds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row over/under decimal prices, falling back to a flat price.

    Real per-book prices are used when supplied; any missing/invalid row falls
    back to the flat price so a few gaps don't silently drop bets.
    """
    over = (
        np.full(n, flat_decimal_odds, dtype=float)
        if decimal_odds_over is None
        else np.asarray(decimal_odds_over, dtype=float).copy()
    )
    under = (
        np.full(n, flat_decimal_odds, dtype=float)
        if decimal_odds_under is None
        else np.asarray(decimal_odds_under, dtype=float).copy()
    )

    over[~np.isfinite(over) | (over <= 1.0)] = flat_decimal_odds
    under[~np.isfinite(under) | (under <= 1.0)] = flat_decimal_odds
    return over, under


def expected_value(win_probability: np.ndarray, decimal_odds: np.ndarray) -> np.ndarray:
    """Expected profit per unit staked: ``p*d - 1``.

    Positive exactly when ``p`` beats the break-even rate ``1/d``, which is what
    makes EV the right selection score for a probability model: unlike a raw
    probability it already accounts for the price, so it is directly comparable
    between the OVER and UNDER sides and across books.
    """
    return np.asarray(win_probability, dtype=float) * np.asarray(
        decimal_odds, dtype=float
    ) - 1.0


def evaluate_betting(
    *,
    predicted_edge: np.ndarray | pd.Series,
    actual_total: np.ndarray | pd.Series,
    line: np.ndarray | pd.Series,
    min_edge: float = 0.0,
    flat_decimal_odds: float = DECIMAL_ODDS_MINUS_110,
    decimal_odds_over: np.ndarray | pd.Series | None = None,
    decimal_odds_under: np.ndarray | pd.Series | None = None,
    selection_score: np.ndarray | pd.Series | None = None,
) -> BettingMetrics:
    """Score a bet-selection rule.

    ``predicted_edge`` decides WHICH SIDE to bet: positive => OVER. For a
    LINE_ERROR model that is the prediction itself; for a TOTAL_POINTS model it
    is ``prediction - line``; for the classifier it is the difference in
    expected value between the two sides.

    ``selection_score`` decides WHETHER to bet, compared against ``min_edge``.
    It defaults to ``abs(predicted_edge)``, which is the natural magnitude for
    a regressor and preserves the original behaviour exactly. A classifier
    passes expected value instead, because the magnitude of a probability
    difference is not in points and cannot be compared to a points threshold.
    Separating the two is what lets one betting layer serve both.

    A bet is placed when ``selection_score > min_edge``. Rows where the actual
    total lands exactly on the line are pushes: stake returned, no profit, and
    excluded from the win rate.
    """
    edge = np.asarray(pd.to_numeric(pd.Series(predicted_edge), errors="coerce"), dtype=float)
    actual = np.asarray(pd.to_numeric(pd.Series(actual_total), errors="coerce"), dtype=float)
    line_values = np.asarray(pd.to_numeric(pd.Series(line), errors="coerce"), dtype=float)

    score = (
        np.abs(edge)
        if selection_score is None
        else np.asarray(
            pd.to_numeric(pd.Series(selection_score), errors="coerce"), dtype=float
        )
    )

    valid = (
        np.isfinite(edge)
        & np.isfinite(actual)
        & np.isfinite(line_values)
        & np.isfinite(score)
    )
    n_candidates = int(valid.sum())

    over_prices, under_prices = _resolve_prices(
        len(edge),
        None if decimal_odds_over is None else np.asarray(decimal_odds_over, dtype=float),
        None if decimal_odds_under is None else np.asarray(decimal_odds_under, dtype=float),
        flat_decimal_odds,
    )

    placed = valid & (score > min_edge)
    n_bets = int(placed.sum())

    if n_bets == 0:
        return BettingMetrics(
            min_edge=min_edge,
            n_candidates=n_candidates,
            n_bets=0,
            bet_rate=0.0,
            n_wins=0,
            n_losses=0,
            n_pushes=0,
            win_rate=None,
            win_rate_ci_low=None,
            win_rate_ci_high=None,
            break_even_rate=None,
            edge_vs_break_even=None,
            profit_units=0.0,
            roi=None,
            beats_break_even=False,
            is_significant=False,
        )

    bet_over = edge[placed] > 0
    actual_margin = actual[placed] - line_values[placed]
    price = np.where(bet_over, over_prices[placed], under_prices[placed])

    is_push = actual_margin == 0
    won = np.where(bet_over, actual_margin > 0, actual_margin < 0) & ~is_push
    lost = ~won & ~is_push

    n_wins = int(won.sum())
    n_losses = int(lost.sum())
    n_pushes = int(is_push.sum())

    profit_units = float(np.sum(price[won] - 1.0) - n_losses)
    roi = profit_units / n_bets

    n_decided = n_wins + n_losses
    win_rate = n_wins / n_decided if n_decided > 0 else None
    ci_low, ci_high = wilson_interval(n_wins, n_decided) if n_decided > 0 else (None, None)

    # Break-even is price-weighted over the bets actually placed.
    break_even = float(np.mean(1.0 / price)) if n_bets > 0 else None

    edge_vs_be = (
        win_rate - break_even if win_rate is not None and break_even is not None else None
    )
    beats = bool(edge_vs_be is not None and edge_vs_be > 0)
    significant = bool(
        ci_low is not None
        and break_even is not None
        and not np.isnan(ci_low)
        and ci_low > break_even
    )

    return BettingMetrics(
        min_edge=min_edge,
        n_candidates=n_candidates,
        n_bets=n_bets,
        bet_rate=n_bets / n_candidates if n_candidates else 0.0,
        n_wins=n_wins,
        n_losses=n_losses,
        n_pushes=n_pushes,
        win_rate=win_rate,
        win_rate_ci_low=None if ci_low is None or np.isnan(ci_low) else float(ci_low),
        win_rate_ci_high=None if ci_high is None or np.isnan(ci_high) else float(ci_high),
        break_even_rate=break_even,
        edge_vs_break_even=edge_vs_be,
        profit_units=profit_units,
        roi=roi,
        beats_break_even=beats,
        is_significant=significant,
    )


def betting_threshold_sweep(
    *,
    predicted_edge: np.ndarray | pd.Series,
    actual_total: np.ndarray | pd.Series,
    line: np.ndarray | pd.Series,
    thresholds: tuple[float, ...] = DEFAULT_EDGE_THRESHOLDS,
    flat_decimal_odds: float = DECIMAL_ODDS_MINUS_110,
    decimal_odds_over: np.ndarray | pd.Series | None = None,
    decimal_odds_under: np.ndarray | pd.Series | None = None,
    selection_score: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Betting metrics across a range of minimum-edge thresholds.

    This is the table to read when deciding whether a model is usable: a high
    win rate at a large threshold means nothing if ``n_bets`` is tiny, which is
    why volume and the CI are reported alongside the rate.

    ``thresholds`` are in the units of ``selection_score`` -- points for a
    regressor, expected value for the classifier.
    """
    rows = [
        evaluate_betting(
            predicted_edge=predicted_edge,
            actual_total=actual_total,
            line=line,
            min_edge=threshold,
            flat_decimal_odds=flat_decimal_odds,
            decimal_odds_over=decimal_odds_over,
            decimal_odds_under=decimal_odds_under,
            selection_score=selection_score,
        ).model_dump()
        for threshold in thresholds
    ]
    return pd.DataFrame(rows)


def evaluate_alternative_lines(
    *,
    predicted_total_points: np.ndarray | pd.Series,
    actual_total: np.ndarray | pd.Series,
    lines: dict[str, np.ndarray | pd.Series],
    min_edge: float,
    flat_decimal_odds: float = DECIMAL_ODDS_MINUS_110,
) -> pd.DataFrame:
    """Re-score one set of predictions against several different total lines.

    The model is left completely untouched: only the number bets are placed
    *at* and settled *into* changes. For each candidate line the edge is
    recomputed as ``predicted_total_points - line``, which is why this takes
    predictions in POINTS space rather than a precomputed edge -- an edge is
    only meaningful relative to the line it was derived from.

    The point of this table is that the closing line is not a line you can bet.
    A model that clears break-even against the close but not against the open
    has demonstrated information without demonstrating a capturable edge, and
    those are different claims. ``line_mae`` (each line's own forecasting error)
    sits alongside so you can see whether a line is worse *as a forecast* --
    which is what creates the opportunity in the first place -- separately from
    whether betting into it actually paid.

    One row per line, in the order given. Reference points such as
    ``mean_abs_move_vs_first`` are measured against the first line, which by
    convention is the one bets were really settled against.
    """
    predicted = np.asarray(
        pd.to_numeric(pd.Series(predicted_total_points), errors="coerce"), dtype=float
    )
    actual = np.asarray(
        pd.to_numeric(pd.Series(actual_total), errors="coerce"), dtype=float
    )

    reference: np.ndarray | None = None
    rows: list[dict[str, object]] = []

    for line_col, line_values in lines.items():
        line_array = np.asarray(
            pd.to_numeric(pd.Series(line_values), errors="coerce"), dtype=float
        )
        if reference is None:
            reference = line_array

        metrics = evaluate_betting(
            predicted_edge=predicted - line_array,
            actual_total=actual,
            line=line_array,
            min_edge=min_edge,
            flat_decimal_odds=flat_decimal_odds,
        )

        finite_line = np.isfinite(actual) & np.isfinite(line_array)
        line_mae = (
            float(np.mean(np.abs(actual[finite_line] - line_array[finite_line])))
            if finite_line.any()
            else float("nan")
        )
        moved = np.isfinite(reference) & np.isfinite(line_array)
        mean_abs_move = (
            float(np.mean(np.abs(line_array[moved] - reference[moved])))
            if moved.any()
            else float("nan")
        )

        rows.append(
            {
                "line_col": line_col,
                # How good this line is as a forecast, independent of betting.
                "line_mae": line_mae,
                # How far this line sits from the one bets were settled against.
                # Compare it to min_edge: if the move is the larger of the two,
                # the bet you would really have placed is a different bet.
                "mean_abs_move_vs_first": mean_abs_move,
                **{
                    key: value
                    for key, value in metrics.model_dump().items()
                    if key != "min_edge"
                },
                "min_edge": metrics.min_edge,
            }
        )

    return pd.DataFrame(rows)
