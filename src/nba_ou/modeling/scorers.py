import numpy as np
import pandas as pd


def evaluate_total_points_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test_total: pd.Series,
    line_col: str,
    thresholds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    y_pred_total = np.asarray(model.predict(X_test), dtype=float)
    betting_line = pd.to_numeric(X_test[line_col], errors="coerce").to_numpy(
        dtype=float
    )

    pred_edge = y_pred_total - betting_line
    margin = np.abs(pred_edge)

    rows = []
    n_total = len(y_test_total)
    y_true_total_np = y_test_total.to_numpy(dtype=float)

    for t in thresholds:
        mask = margin > t
        n = int(mask.sum())

        acc = (
            np.nan
            if n == 0
            else over_under_betting_accuracy_total_points(
                y_true=y_true_total_np[mask],
                y_pred=y_pred_total[mask],
                betting_line=betting_line[mask],
            )
        )

        rows.append(
            {
                "threshold_abs_pred_edge_gt": t,
                "n_games": n,
                "pct_of_test": (n / n_total) if n_total else np.nan,
                "ou_betting_accuracy": acc,
            }
        )

    return pd.DataFrame(rows), y_pred_total


def over_under_betting_accuracy_total_points(y_true, y_pred, betting_line) -> float:
    """
    Betting accuracy when the model predicts total points directly.

    Parameters
    ----------
    y_true : array-like
        True game total points.
    y_pred : array-like
        Predicted game total points.
    betting_line : array-like
        Sportsbook total line for each game.

    Returns
    -------
    float
        Fraction of non-push cases where predicted side matches actual side.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    betting_line = np.asarray(betting_line, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(betting_line)
    if not np.any(valid):
        return 0.0

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    betting_line = betting_line[valid]

    true_side = np.sign(y_true - betting_line)
    pred_side = np.sign(y_pred - betting_line)

    # remove pushes
    non_push = (true_side != 0) & (pred_side != 0)
    if not np.any(non_push):
        return 0.0

    return float(np.mean(true_side[non_push] == pred_side[non_push]))


class OverUnderScorerTotalPoints:
    """
    sklearn-compatible scorer for models predicting total points.
    It reads the betting line from X[line_col].
    """

    def __init__(self, line_col: str):
        self.line_col = line_col

    def __call__(self, estimator, X, y_true):
        if self.line_col not in X.columns:
            raise KeyError(f"{self.line_col} not found in X for scoring")

        y_pred = estimator.predict(X)
        betting_line = pd.to_numeric(X[self.line_col], errors="coerce").to_numpy(
            dtype=float
        )

        return over_under_betting_accuracy_total_points(
            y_true=np.asarray(y_true, dtype=float),
            y_pred=np.asarray(y_pred, dtype=float),
            betting_line=betting_line,
        )


def over_under_betting_accuracy_total_points_with_min_edge(
    y_true,
    y_pred,
    betting_line,
    min_edge: float = 1.0,
) -> float:
    """
    OU betting accuracy using only predictions with absolute model edge > min_edge.

    A game is included only if:
        abs(y_pred - betting_line) > min_edge

    Pushes on the true side are excluded.
    Predictions exactly on the threshold are excluded.

    Returns
    -------
    float
        Fraction of qualifying non-push cases where predicted side matches actual side.
        Returns np.nan if no qualifying cases remain.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    betting_line = np.asarray(betting_line, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(betting_line)
    if not np.any(valid):
        return np.nan

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    betting_line = betting_line[valid]

    pred_edge = y_pred - betting_line
    take_bet = np.abs(pred_edge) > min_edge
    if not np.any(take_bet):
        return np.nan

    y_true = y_true[take_bet]
    y_pred = y_pred[take_bet]
    betting_line = betting_line[take_bet]

    true_side = np.sign(y_true - betting_line)
    pred_side = np.where(y_pred > betting_line, 1, -1)

    non_push = true_side != 0
    if not np.any(non_push):
        return np.nan

    return float(np.mean(true_side[non_push] == pred_side[non_push]))

class OverUnderScorerTotalPointsMinEdge:
    """
    sklearn-compatible scorer for total-points models that only scores
    predictions with |y_pred - line| > min_edge.
    """

    def __init__(self, line_col: str, min_edge: float = 1.0):
        self.line_col = line_col
        self.min_edge = min_edge

    def __call__(self, estimator, X, y_true):
        if self.line_col not in X.columns:
            raise KeyError(f"{self.line_col} not found in X for scoring")

        y_pred = estimator.predict(X)
        betting_line = pd.to_numeric(X[self.line_col], errors="coerce").to_numpy(
            dtype=float
        )

        score = over_under_betting_accuracy_total_points_with_min_edge(
            y_true=np.asarray(y_true, dtype=float),
            y_pred=np.asarray(y_pred, dtype=float),
            betting_line=betting_line,
            min_edge=self.min_edge,
        )

        return score if np.isfinite(score) else 0.0


def over_under_betting_accuracy_error_line(y_true_error, y_pred_error) -> float:
    """
    Betting accuracy when the model predicts line error directly.

    Parameters
    ----------
    y_true_error : array-like
        True line error (actual total points - betting line).
    y_pred_error : array-like
        Predicted line error.

    Returns
    -------
    float
        Fraction of non-push cases where predicted side matches actual side.
    """
    y_true_error = np.asarray(y_true_error, dtype=float)
    y_pred_error = np.asarray(y_pred_error, dtype=float)

    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return 0.0

    y_true_error = y_true_error[valid]
    y_pred_error = y_pred_error[valid]

    true_side = np.sign(y_true_error)
    pred_side = np.sign(y_pred_error)

    non_push = (true_side != 0) & (pred_side != 0)
    if not np.any(non_push):
        return 0.0

    return float(np.mean(true_side[non_push] == pred_side[non_push]))


def over_under_betting_accuracy_error_line_with_min_edge(
    y_true_error,
    y_pred_error,
    min_edge: float = 1.0,
) -> float:
    """
    Directional accuracy using only predictions with absolute predicted error
    above a minimum edge threshold.

    Returns np.nan if no qualifying non-push cases remain.
    """
    y_true_error = np.asarray(y_true_error, dtype=float)
    y_pred_error = np.asarray(y_pred_error, dtype=float)

    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan

    y_true_error = y_true_error[valid]
    y_pred_error = y_pred_error[valid]

    take_bet = np.abs(y_pred_error) > min_edge
    if not np.any(take_bet):
        return np.nan

    y_true_error = y_true_error[take_bet]
    y_pred_error = y_pred_error[take_bet]

    true_side = np.sign(y_true_error)
    pred_side = np.where(y_pred_error > 0, 1, -1)

    non_push = true_side != 0
    if not np.any(non_push):
        return np.nan

    return float(np.mean(true_side[non_push] == pred_side[non_push]))



def evaluate_error_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test_error: pd.Series,
    thresholds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
):
    y_pred_error = np.asarray(model.predict(X_test), dtype=float)
    margin = np.abs(y_pred_error)

    rows = []
    n_total = len(y_test_error)
    y_true_error_np = pd.to_numeric(y_test_error, errors="coerce").to_numpy(dtype=float)

    for t in thresholds:
        mask = margin > t
        n = int(mask.sum())

        acc = (
            np.nan
            if n == 0
            else over_under_betting_accuracy_error_line(
                y_true_error=y_true_error_np[mask],
                y_pred_error=y_pred_error[mask],
            )
        )

        rows.append(
            {
                "threshold_abs_pred_error_gt": t,
                "n_games": n,
                "pct_of_test": (n / n_total) if n_total else np.nan,
                "directional_accuracy": acc,
            }
        )

    return pd.DataFrame(rows), y_pred_error


class OverUnderScorerLineError:
    """
    sklearn-compatible scorer for models predicting line error directly.
    """

    def __call__(self, estimator, X, y_true):
        y_pred = estimator.predict(X)
        return over_under_betting_accuracy_error_line(
            y_true_error=np.asarray(y_true, dtype=float),
            y_pred_error=np.asarray(y_pred, dtype=float),
        )


class OverUnderScorerLineErrorMinEdge:
    """
    sklearn-compatible scorer for direct line-error models that only scores
    predictions with |predicted_error| > min_edge.
    """

    def __init__(self, min_edge: float = 1.0):
        self.min_edge = min_edge

    def __call__(self, estimator, X, y_true):
        y_pred = estimator.predict(X)
        score = over_under_betting_accuracy_error_line_with_min_edge(
            y_true_error=np.asarray(y_true, dtype=float),
            y_pred_error=np.asarray(y_pred, dtype=float),
            min_edge=self.min_edge,
        )
        return score if np.isfinite(score) else 0.0
