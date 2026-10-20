"""Turning any fitted model into a betting decision, in exactly one place.

Three strategies predict three different quantities -- a total, an error
against the line, a probability -- but every downstream consumer (CV betting,
the daily walk-forward, holdout scoring, the leaderboard) needs the same three
things: which side to bet, how strongly, and what the model thinks the total
will be.

Without this module each of those consumers would branch on the strategy
itself, which is how the ``line_error`` sample-weight bug got in: the same
decision expressed in several places, and one copy wrong. ``get_strategy`` is
the only branch point for *training*; this is the only branch point for
*prediction*.

The selection score is where the classifier differs most. A regressor's
magnitude is in points and compares naturally against a points threshold. A
probability has no such scale, so the classifier uses expected value
(``p*d - 1``), which prices the bet and is positive exactly when the
probability beats break-even. That keeps the same betting layer serving both,
but it does mean classifier thresholds live in EV units, not points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBModel

from training_pipeline.betting import expected_value
from training_pipeline.config import ExperimentConfig, Market, PredictionStrategy

#: Strategies whose prediction is already the edge, so no line is subtracted.
#: Both markets' residual regressors behave identically here -- the only
#: difference is which market's line reconstructs the outcome level.
_RESIDUAL_REGRESSORS = frozenset(
    {
        PredictionStrategy.LINE_ERROR_REGRESSOR,
        PredictionStrategy.SPREAD_ERROR_REGRESSOR,
    }
)


@dataclass(frozen=True)
class Decisions:
    """Everything the betting layer needs, whatever produced it."""

    #: The model's raw output, in the units it was trained on. Points for
    #: TOTAL_POINTS, points-vs-line for LINE_ERROR, P(OVER) for the classifier.
    raw_prediction: np.ndarray
    #: Sign picks the side: positive => OVER (totals) / HOME (spread).
    predicted_edge: np.ndarray
    #: Magnitude compared against a threshold to decide whether to bet at all.
    #: Points for regressors, expected value for the classifier.
    selection_score: np.ndarray
    #: P(OVER), classifier only.
    p_over: np.ndarray | None
    #: Expected value of the side actually chosen, classifier only.
    expected_value: np.ndarray | None
    #: Predicted total in points, or None for the classifier -- a probability
    #: carries no view on the total, which is why classifiers cannot be
    #: re-scored against an alternative line the way regressors can.
    predicted_total: np.ndarray | None

    @property
    def bets_over(self) -> np.ndarray:
        return self.predicted_edge > 0


def _resolve_prices(
    n: int,
    over: np.ndarray | None,
    under: np.ndarray | None,
    flat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row prices, falling back to the flat price where absent/invalid."""
    resolved = []
    for values in (over, under):
        if values is None:
            resolved.append(np.full(n, flat, dtype=float))
            continue
        array = np.asarray(values, dtype=float).copy()
        array[~np.isfinite(array) | (array <= 1.0)] = flat
        resolved.append(array)
    return resolved[0], resolved[1]


def predict_decisions(
    model: XGBModel,
    X: pd.DataFrame,
    *,
    config: ExperimentConfig,
    target_line: np.ndarray,
    decimal_odds_over: np.ndarray | None = None,
    decimal_odds_under: np.ndarray | None = None,
) -> Decisions:
    """Run a fitted model and express its output as a betting decision."""
    line = np.asarray(target_line, dtype=float)

    if config.strategy == PredictionStrategy.OVER_UNDER_CLASSIFIER:
        if not isinstance(model, XGBClassifier):
            raise TypeError(
                "over_under_classifier requires an XGBClassifier; got "
                f"{type(model).__name__}. A regressor's output is not a probability."
            )
        # Column 1 is P(class 1) == P(OVER); classes_ is [0, 1] by construction
        # because the label is built as 0/1 in training_pipeline.data.
        p_over = np.asarray(model.predict_proba(X), dtype=float)[:, 1]
        over_prices, under_prices = _resolve_prices(
            len(p_over),
            decimal_odds_over,
            decimal_odds_under,
            config.betting.flat_decimal_odds,
        )
        ev_over = expected_value(p_over, over_prices)
        ev_under = expected_value(1.0 - p_over, under_prices)

        # Bet whichever side prices better; the EV difference carries the side
        # and the max carries the strength. Comparing EV rather than probability
        # is what makes this correct under asymmetric prices.
        return Decisions(
            raw_prediction=p_over,
            predicted_edge=ev_over - ev_under,
            selection_score=np.maximum(ev_over, ev_under),
            p_over=p_over,
            expected_value=np.maximum(ev_over, ev_under),
            predicted_total=None,
        )

    y_pred = np.asarray(model.predict(X), dtype=float)
    if config.strategy in _RESIDUAL_REGRESSORS:
        # The prediction already IS the edge; adding the line back recovers the
        # implied outcome level (a total for LINE_ERROR, a home margin for
        # SPREAD_ERROR).
        edge = y_pred
        predicted_total = line + y_pred
    else:
        edge = y_pred - line
        predicted_total = y_pred

    return Decisions(
        raw_prediction=y_pred,
        predicted_edge=edge,
        selection_score=np.abs(edge),
        p_over=None,
        expected_value=None,
        predicted_total=predicted_total,
    )


def decisions_from_pooled_predictions(
    raw_prediction: np.ndarray,
    *,
    target_line: np.ndarray,
    config: ExperimentConfig,
    decimal_odds_over: np.ndarray | None = None,
    decimal_odds_under: np.ndarray | None = None,
) -> Decisions:
    """Same as :func:`predict_decisions`, but the model has already been run.

    The daily walk-forward fits and predicts one day at a time and only keeps
    the pooled output, so by the time the results are scored there is no model
    left to call -- just an array whose meaning depends on the strategy. This
    keeps that interpretation in the same place as the live one, rather than
    letting the walk-forward re-derive it and drift.
    """
    values = np.asarray(raw_prediction, dtype=float)
    line = np.asarray(target_line, dtype=float)

    if config.strategy == PredictionStrategy.OVER_UNDER_CLASSIFIER:
        over_prices, under_prices = _resolve_prices(
            len(values),
            decimal_odds_over,
            decimal_odds_under,
            config.betting.flat_decimal_odds,
        )
        ev_over = expected_value(values, over_prices)
        ev_under = expected_value(1.0 - values, under_prices)
        return Decisions(
            raw_prediction=values,
            predicted_edge=ev_over - ev_under,
            selection_score=np.maximum(ev_over, ev_under),
            p_over=values,
            expected_value=np.maximum(ev_over, ev_under),
            predicted_total=None,
        )

    if config.strategy in _RESIDUAL_REGRESSORS:
        edge = values
        predicted_total = line + values
    else:
        edge = values - line
        predicted_total = values

    return Decisions(
        raw_prediction=values,
        predicted_edge=edge,
        selection_score=np.abs(edge),
        p_over=None,
        expected_value=None,
        predicted_total=predicted_total,
    )


def price_column(df: pd.DataFrame, column: str | None) -> np.ndarray | None:
    """A decimal-odds column as an array, or None to fall back to a flat price.

    Missing entirely is fine (it may not have survived cleaning); per-row gaps
    are repaired in :func:`_resolve_prices`. American odds are rejected here
    because negative values would otherwise silently fall back to the flat
    price, while positive values such as +100 would be treated as decimal 100.
    """
    if not column or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    american_like = finite[(finite < 0.0) | (finite >= 20.0)]
    if american_like.size:
        sample = np.unique(american_like)[:5].tolist()
        raise ValueError(
            f"Configured betting price column {column!r} must contain decimal odds; "
            f"found American-looking values such as {sample}. Convert the prices "
            "before settlement or leave the price columns unconfigured to use "
            "betting.flat_decimal_odds."
        )
    return values


def collect_prices(
    df: pd.DataFrame,
    config: ExperimentConfig,
    *,
    positions: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The configured (over, under) decimal-odds arrays.

    These belong to the DECISION, not only to settlement. For a classifier the
    side is chosen on expected value, so an asymmetric price can make the less
    likely side the better bet -- selecting the side on flat odds and then
    settling on real ones would score a bet the model never would have placed.

    ``positions`` positionally selects rows, for callers that predict a subset.

    On a spread run the two sides are HOME and AWAY rather than OVER and UNDER.
    They occupy the same slots because they play the same role: ``predicted_edge
    > 0`` selects the first element in both markets (OVER for totals, HOME for
    the spread), so everything downstream is identical arithmetic.
    """
    if config.market is Market.SPREAD:
        over = price_column(df, config.betting.home_price_col)
        under = price_column(df, config.betting.away_price_col)
    else:
        over = price_column(df, config.betting.over_price_col)
        under = price_column(df, config.betting.under_price_col)
    if positions is not None:
        over = None if over is None else over[positions]
        under = None if under is None else under[positions]
    return over, under


def primary_threshold(config: ExperimentConfig) -> float:
    """The headline bet-selection threshold, in this strategy's units."""
    return (
        config.betting.primary_ev_threshold
        if config.is_classifier
        else config.betting.primary_edge_threshold
    )


def threshold_sweep_values(config: ExperimentConfig) -> tuple[float, ...]:
    """The thresholds to sweep, in this strategy's units."""
    return (
        config.betting.ev_thresholds
        if config.is_classifier
        else config.betting.edge_thresholds
    )
