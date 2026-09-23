"""How a lineup projection column relates to the closing line's error.

The phase-G question (``docs/lineup_projection_plan.md`` section 8.4) is not
"does the projection beat the bookmaker" but "does it carry information the
closing line has not priced". Two cheap reads of that, used by
``scripts/lineups/evaluate_game_projection.py``:

- the OLS slope of ``LINE_ERROR`` on the column, with a bootstrap interval
  **clustered by game date** -- games on one night share league-wide news and
  officiating tendencies, so resampling them independently understates the
  uncertainty;
- directional accuracy of the univariate rule "OVER when the column is
  positive", above thresholds of the column's size.

Both are weak lower bounds: the model sees the column beside ~3,000 others.
They are here so every number the plan quotes can be regenerated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Break-even hit rate at -110.
BREAK_EVEN_110 = 110.0 / 210.0


def line_error_slope(
    frame: pd.DataFrame,
    column: str,
    *,
    target: str = "LINE_ERROR",
    date_col: str = "GAME_DATE",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Slope of ``target`` on ``column`` with a date-clustered 95% interval."""
    data = frame[[column, target, date_col]].dropna()
    if len(data) < 3:
        return {"n": len(data), "slope": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    xs = data[column].to_numpy(float)
    ys = data[target].to_numpy(float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    groups = list(data.groupby(date_col, sort=True).indices.values())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[i] for i in picks])
        if np.ptp(xs[index]) == 0:
            continue
        boots.append(np.polyfit(xs[index], ys[index], 1)[0])
    low, high = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {
        "n": len(data),
        "slope": slope,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def directional_accuracy(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
    *,
    target: str = "LINE_ERROR",
) -> dict[str, float]:
    """Hit rate of "OVER when ``column`` > 0" where ``|column| >= threshold``.

    Pushes (``target == 0``) and ``column == 0`` rows have no direction and are
    left out. The interval is a normal approximation.
    """
    data = frame[[column, target]].dropna()
    data = data.loc[
        data[column].abs().ge(threshold) & data[column].ne(0) & data[target].ne(0)
    ]
    n = len(data)
    if n == 0:
        return {
            "threshold": threshold,
            "n": 0,
            "accuracy": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    hit = float((np.sign(data[column]) == np.sign(data[target])).mean())
    half = 1.96 * np.sqrt(hit * (1.0 - hit) / n)
    return {
        "threshold": threshold,
        "n": n,
        "accuracy": hit,
        "ci_low": hit - half,
        "ci_high": hit + half,
    }
