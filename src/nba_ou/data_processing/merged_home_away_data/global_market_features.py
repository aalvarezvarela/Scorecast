"""
Global Market Regime Features for NBA Over/Under Predictor.

Computes league-wide, game-date-level market calibration features using only
information available **strictly before** each game starts (no data leakage).

Feature families
----------------
- Market bias (signed error)
- Market MAE (unsigned error)
- Market error volatility (std)
- Robust bias / MAE (median)
- Tail-miss rates (|error| > 10, 15, 20)
- Over / under / push directional rates
- Global close-total and actual-total regime levels
- Pricing gap (actual − close averages)
- Regime ratios (short-window / long-window)
- Acceleration / correction-speed diffs
- League activity (game-count per window)
- Opening-line features (move magnitude, direction, correction success, open-based error)
- Cross-book disagreement (per-game and rolling)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import (
    extract_total_line_books,
    resolve_main_total_line_col,
)
from nba_ou.utils.general_utils import _with_before_suffix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-6

# Rolling game-count windows
GAME_WINDOWS = [15, 30, 75, 150]

# Rolling calendar-day windows
DAY_WINDOWS = [3, 7, 14]

# EWM spans (in terms of game-count)
EWM_SPANS = [15, 30, 75]

# Tail thresholds
TAIL_THRESHOLDS = [10, 15, 20]

# Opening-line column (consensus opener kept after merge)
OPEN_LINE_COL = "ODDS_TOTAL_LINE_consensus_opener"

#: Not a book's close but an opening consensus: across books it would measure
#: line movement, not disagreement.
CROSSBOOK_EXCLUDED_BOOKS = ("consensus_opener",)

#: A game with one book has no disagreement to measure; ``ddof=0`` would call it
#: 0 ("the books agree"), so it is NaN instead.
CROSSBOOK_MIN_BOOKS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rolling_game_agg(
    series: pd.Series,
    dates: pd.Series,
    window: int,
    func: str,
) -> pd.Series:
    """
    Rolling aggregation over the last *window* games from strictly earlier dates.

    All rows on the same calendar date receive the same value, so same-day games
    can never influence each other.
    """
    result = np.full(len(series), np.nan, dtype="float64")
    date_days = pd.to_datetime(dates, errors="coerce").values.astype("datetime64[D]")
    values = pd.to_numeric(series, errors="coerce").astype("float64").values

    if len(result) == 0:
        return pd.Series(result, index=series.index, dtype="float64")

    group_starts = np.flatnonzero(
        np.r_[True, date_days[1:] != date_days[:-1]]
    )
    group_ends = np.r_[group_starts[1:], len(values)]

    for start, end in zip(group_starts, group_ends, strict=True):
        window_vals = values[max(0, start - window) : start]
        valid_vals = window_vals[~np.isnan(window_vals)]

        if func == "count":
            agg_value = float(len(valid_vals))
        elif len(valid_vals) == 0:
            agg_value = np.nan
        elif func == "mean":
            agg_value = float(np.mean(valid_vals))
        elif func == "std":
            agg_value = float(np.std(valid_vals, ddof=0))
        elif func == "median":
            agg_value = float(np.median(valid_vals))
        elif func == "sum":
            agg_value = float(np.sum(valid_vals))
        else:
            raise ValueError(f"Unsupported func: {func}")

        result[start:end] = agg_value

    return pd.Series(result, index=series.index, dtype="float64")


def _rolling_day_agg(
    values: pd.Series,
    dates: pd.Series,
    window_days: int,
    func: str,
) -> pd.Series:
    """
    For each row, aggregate *values* from games whose GAME_DATE falls within
    [current_date − window_days, current_date) — strictly before the game.
    """
    result = np.full(len(values), np.nan, dtype="float64")
    dates_arr = dates.values.astype("datetime64[D]")
    vals_arr = values.values.astype("float64")
    delta = np.timedelta64(window_days, "D")

    # The masked rows of a date-sorted array are one contiguous run, in the
    # same order, so the window is the same array, found by binary search;
    # and every row of one date has the same window, computed once.
    in_date_order = not np.isnat(dates_arr).any() and bool(
        (dates_arr[1:] >= dates_arr[:-1]).all()
    )
    by_date: dict = {}

    for i in range(len(result)):
        cur_date = dates_arr[i]
        if in_date_order:
            if cur_date in by_date:
                result[i] = by_date[cur_date]
                continue
            lo = np.searchsorted(dates_arr, cur_date - delta, side="left")
            hi = np.searchsorted(dates_arr, cur_date, side="left")
            window_vals = vals_arr[lo:hi]
        else:
            mask = (dates_arr < cur_date) & (dates_arr >= cur_date - delta)
            window_vals = vals_arr[mask]
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals) == 0:
            pass
        elif func == "mean":
            result[i] = np.mean(window_vals)
        elif func == "std":
            result[i] = np.std(window_vals, ddof=0)
        elif func == "median":
            result[i] = np.median(window_vals)
        elif func == "sum":
            result[i] = np.sum(window_vals)
        elif func == "count":
            result[i] = float(len(window_vals))
        if in_date_order:
            by_date[cur_date] = result[i]
    return pd.Series(result, index=values.index, dtype="float64")


def _ewm_game_mean(series: pd.Series, dates: pd.Series, span: int) -> pd.Series:
    return _ewm_game_mean_strict_prior_dates(series, dates, span)


def _ewm_game_std(series: pd.Series, dates: pd.Series, span: int) -> pd.Series:
    return _ewm_game_std_strict_prior_dates(series, dates, span)


def _ewm_game_mean_strict_prior_dates(
    series: pd.Series, dates: pd.Series, span: int
) -> pd.Series:
    """EWM mean over games from strictly earlier calendar dates only."""
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    date_days = pd.to_datetime(dates, errors="coerce").values.astype("datetime64[D]")
    ewm_full = (
        pd.to_numeric(series, errors="coerce")
        .astype("float64")
        .ewm(span=span, min_periods=1)
        .mean()
    )

    if len(result) == 0:
        return result

    group_starts = np.flatnonzero(
        np.r_[True, date_days[1:] != date_days[:-1]]
    )
    group_ends = np.r_[group_starts[1:], len(result)]

    for start, end in zip(group_starts, group_ends, strict=True):
        agg_value = np.nan if start == 0 else float(ewm_full.iloc[start - 1])
        result.iloc[start:end] = agg_value

    return result


def _ewm_game_std_strict_prior_dates(
    series: pd.Series, dates: pd.Series, span: int
) -> pd.Series:
    """EWM std over games from strictly earlier calendar dates only."""
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    date_days = pd.to_datetime(dates, errors="coerce").values.astype("datetime64[D]")
    ewm_full = (
        pd.to_numeric(series, errors="coerce")
        .astype("float64")
        .ewm(span=span, min_periods=1)
        .std()
    )

    if len(result) == 0:
        return result

    group_starts = np.flatnonzero(
        np.r_[True, date_days[1:] != date_days[:-1]]
    )
    group_ends = np.r_[group_starts[1:], len(result)]

    for start, end in zip(group_starts, group_ends, strict=True):
        agg_value = np.nan if start == 0 else float(ewm_full.iloc[start - 1])
        result.iloc[start:end] = agg_value

    return result


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def add_global_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach global market-regime features to a merged game-level DataFrame.

    The DataFrame must contain at least:
    - ``GAME_DATE`` (datetime-like)
    - ``TOTAL_POINTS`` (actual combined score)
    - A closing total-line column resolvable via ``resolve_main_total_line_col``

    All output columns carry the ``_BEFORE`` suffix so that
    ``select_training_columns`` automatically picks them up.

    Parameters
    ----------
    df : pd.DataFrame
        Merged game-level DataFrame (one row per game), sorted or unsorted.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new global market feature columns appended.
    """
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"])

    # Sort chronologically for rolling calculations; GAME_ID as stable tie-breaker
    out = out.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    # ---- resolve columns ----
    close_col = resolve_main_total_line_col(out)
    if close_col is None:
        raise ValueError(
            "Cannot compute global market features: no ODDS_TOTAL_LINE_* column found."
        )

    close_line = pd.to_numeric(out[close_col], errors="coerce").astype("float64")
    actual_pts = pd.to_numeric(out["TOTAL_POINTS"], errors="coerce").astype("float64")
    dates = out["GAME_DATE"]

    # ---- base derived series ----
    valid_close_actual = actual_pts.notna() & close_line.notna()

    market_error = actual_pts - close_line
    market_abs_error = market_error.abs()

    market_over_flag = pd.Series(
        np.where(valid_close_actual, (actual_pts > close_line).astype(float), np.nan),
        index=out.index,
    )
    market_under_flag = pd.Series(
        np.where(valid_close_actual, (actual_pts < close_line).astype(float), np.nan),
        index=out.index,
    )
    market_push_flag = pd.Series(
        np.where(valid_close_actual, (actual_pts == close_line).astype(float), np.nan),
        index=out.index,
    )
    tail_flags = {
        t: pd.Series(
            np.where(valid_close_actual, (market_abs_error > t).astype(float), np.nan),
            index=out.index,
        )
        for t in TAIL_THRESHOLDS
    }
    ones = pd.Series(1.0, index=out.index)

    # ---- opening-line availability ----
    has_open = OPEN_LINE_COL in out.columns and close_col != OPEN_LINE_COL
    if has_open:
        open_line = pd.to_numeric(out[OPEN_LINE_COL], errors="coerce").astype("float64")
        open_to_close_move = close_line - open_line
        abs_open_to_close_move = open_to_close_move.abs()
        open_error = actual_pts - open_line
        valid_open = valid_close_actual & open_line.notna()
        close_beats_open_flag = pd.Series(
            np.where(
                valid_open,
                (market_abs_error < open_error.abs()).astype(float),
                np.nan,
            ),
            index=out.index,
        )

    # ---- cross-book per-game features (section 8.1) ----
    total_line_books = [
        b
        for b in extract_total_line_books(out)
        if b not in CROSSBOOK_EXCLUDED_BOOKS
    ]
    if len(total_line_books) >= CROSSBOOK_MIN_BOOKS:
        book_lines = pd.DataFrame(
            {
                b: pd.to_numeric(out[f"ODDS_TOTAL_LINE_{b}"], errors="coerce")
                for b in total_line_books
            }
        )
        enough_books = book_lines.notna().sum(axis=1) >= CROSSBOOK_MIN_BOOKS
        crossbook_std = book_lines.std(axis=1, skipna=True, ddof=0).where(
            enough_books
        )
        crossbook_range = (
            book_lines.max(axis=1, skipna=True) - book_lines.min(axis=1, skipna=True)
        ).where(enough_books)
    else:
        crossbook_std = None
        crossbook_range = None

    # ---- collect all new feature columns ----
    new_cols: dict[str, pd.Series] = {}

    # =====================================================================
    # 6.1  Global market bias  (mean of market_error)
    # =====================================================================
    for w in GAME_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_BIAS_{w}G")] = _rolling_game_agg(
            market_error, dates, w, "mean"
        )
    for d in DAY_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_BIAS_{d}D")] = _rolling_day_agg(
            market_error, dates, d, "mean"
        )
    for s in EWM_SPANS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_BIAS_EWM_{s}G")] = _ewm_game_mean(
            market_error, dates, s
        )

    # =====================================================================
    # 6.2  Global market MAE  (mean of |market_error|)
    # =====================================================================
    for w in GAME_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_MAE_{w}G")] = _rolling_game_agg(
            market_abs_error, dates, w, "mean"
        )
    for d in DAY_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_MAE_{d}D")] = _rolling_day_agg(
            market_abs_error, dates, d, "mean"
        )
    for s in EWM_SPANS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_MAE_EWM_{s}G")] = _ewm_game_mean(
            market_abs_error, dates, s
        )

    # =====================================================================
    # 6.3  Global market error std  (std of market_error)
    # =====================================================================
    for w in GAME_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_ERROR_STD_{w}G")] = (
            _rolling_game_agg(market_error, dates, w, "std")
        )
    for d in DAY_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_ERROR_STD_{d}D")] = (
            _rolling_day_agg(market_error, dates, d, "std")
        )
    for s in EWM_SPANS:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_ERROR_STD_EWM_{s}G")] = (
            _ewm_game_std(market_error, dates, s)
        )

    # =====================================================================
    # 6.4  Robust market bias / MAE  (median)
    # =====================================================================
    for w in [15, 30, 75]:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_MEDIAN_ERROR_{w}G")] = (
            _rolling_game_agg(market_error, dates, w, "median")
        )
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_MEDIAN_ABS_ERROR_{w}G")] = (
            _rolling_game_agg(market_abs_error, dates, w, "median")
        )

    # =====================================================================
    # 6.5  Tail miss-rate features
    # =====================================================================
    for t in TAIL_THRESHOLDS:
        for w in [15, 30, 75]:
            new_cols[_with_before_suffix(f"GLOBAL_MARKET_TAIL_GT_{t}_{w}G")] = (
                _rolling_game_agg(tail_flags[t], dates, w, "mean")
            )

    # =====================================================================
    # 6.6  Over / under / push directional frequency
    # =====================================================================
    for w in [15, 30, 75]:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_OVER_RATE_{w}G")] = (
            _rolling_game_agg(market_over_flag, dates, w, "mean")
        )
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_UNDER_RATE_{w}G")] = (
            _rolling_game_agg(market_under_flag, dates, w, "mean")
        )
    for w in [30, 75]:
        new_cols[_with_before_suffix(f"GLOBAL_MARKET_PUSH_RATE_{w}G")] = (
            _rolling_game_agg(market_push_flag, dates, w, "mean")
        )

    # =====================================================================
    # 6.7  Global close-total average (line regime)
    # =====================================================================
    for w in GAME_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_CLOSE_TOTAL_AVG_{w}G")] = (
            _rolling_game_agg(close_line, dates, w, "mean")
        )
    for d in DAY_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_CLOSE_TOTAL_AVG_{d}D")] = (
            _rolling_day_agg(close_line, dates, d, "mean")
        )

    # =====================================================================
    # 6.8  Global actual-total average (scoring regime)
    # =====================================================================
    for w in GAME_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_ACTUAL_TOTAL_AVG_{w}G")] = (
            _rolling_game_agg(actual_pts, dates, w, "mean")
        )
    for d in DAY_WINDOWS:
        new_cols[_with_before_suffix(f"GLOBAL_ACTUAL_TOTAL_AVG_{d}D")] = (
            _rolling_day_agg(actual_pts, dates, d, "mean")
        )

    # =====================================================================
    # 6.9  Market pricing gap  (actual_avg − close_avg  ≈  bias again,
    #       but here computed from windowed averages)
    # =====================================================================
    for w in GAME_WINDOWS:
        act_key = _with_before_suffix(f"GLOBAL_ACTUAL_TOTAL_AVG_{w}G")
        clo_key = _with_before_suffix(f"GLOBAL_CLOSE_TOTAL_AVG_{w}G")
        new_cols[_with_before_suffix(f"GLOBAL_ACTUAL_MINUS_CLOSE_AVG_{w}G")] = (
            new_cols[act_key] - new_cols[clo_key]
        )
    for d in DAY_WINDOWS:
        act_key = _with_before_suffix(f"GLOBAL_ACTUAL_TOTAL_AVG_{d}D")
        clo_key = _with_before_suffix(f"GLOBAL_CLOSE_TOTAL_AVG_{d}D")
        new_cols[_with_before_suffix(f"GLOBAL_ACTUAL_MINUS_CLOSE_AVG_{d}D")] = (
            new_cols[act_key] - new_cols[clo_key]
        )

    # =====================================================================
    # 6.10  Regime ratios  (short / long + eps)
    # =====================================================================
    _ratio_pairs = [(15, 75), (30, 150)]
    for short_w, long_w in _ratio_pairs:
        # Normalized bias diff: (short - long) / (std_long + eps)
        bias_short = new_cols[_with_before_suffix(f"GLOBAL_MARKET_BIAS_{short_w}G")]
        bias_long = new_cols[_with_before_suffix(f"GLOBAL_MARKET_BIAS_{long_w}G")]
        std_long = new_cols[_with_before_suffix(f"GLOBAL_MARKET_ERROR_STD_{long_w}G")]
        new_cols[
            _with_before_suffix(f"GLOBAL_MARKET_BIAS_NORM_DIFF_{short_w}G_VS_{long_w}G")
        ] = (bias_short - bias_long) / (std_long + EPS)

        mae_short = new_cols[_with_before_suffix(f"GLOBAL_MARKET_MAE_{short_w}G")]
        mae_long = new_cols[_with_before_suffix(f"GLOBAL_MARKET_MAE_{long_w}G")]
        new_cols[
            _with_before_suffix(f"GLOBAL_MARKET_MAE_RATIO_{short_w}G_VS_{long_w}G")
        ] = mae_short / (mae_long + EPS)

        std_short = new_cols[_with_before_suffix(f"GLOBAL_MARKET_ERROR_STD_{short_w}G")]
        new_cols[
            _with_before_suffix(f"GLOBAL_MARKET_STD_RATIO_{short_w}G_VS_{long_w}G")
        ] = std_short / (std_long + EPS)

    # =====================================================================
    # 6.11  Acceleration / correction-speed diffs  (short − long)
    # =====================================================================
    for short_w, long_w in _ratio_pairs:
        new_cols[
            _with_before_suffix(f"GLOBAL_MARKET_BIAS_DIFF_{short_w}G_{long_w}G")
        ] = (
            new_cols[_with_before_suffix(f"GLOBAL_MARKET_BIAS_{short_w}G")]
            - new_cols[_with_before_suffix(f"GLOBAL_MARKET_BIAS_{long_w}G")]
        )
        new_cols[
            _with_before_suffix(f"GLOBAL_MARKET_MAE_DIFF_{short_w}G_{long_w}G")
        ] = (
            new_cols[_with_before_suffix(f"GLOBAL_MARKET_MAE_{short_w}G")]
            - new_cols[_with_before_suffix(f"GLOBAL_MARKET_MAE_{long_w}G")]
        )
        new_cols[
            _with_before_suffix(f"GLOBAL_CLOSE_TOTAL_AVG_DIFF_{short_w}G_{long_w}G")
        ] = (
            new_cols[_with_before_suffix(f"GLOBAL_CLOSE_TOTAL_AVG_{short_w}G")]
            - new_cols[_with_before_suffix(f"GLOBAL_CLOSE_TOTAL_AVG_{long_w}G")]
        )
        new_cols[
            _with_before_suffix(f"GLOBAL_ACTUAL_TOTAL_AVG_DIFF_{short_w}G_{long_w}G")
        ] = (
            new_cols[_with_before_suffix(f"GLOBAL_ACTUAL_TOTAL_AVG_{short_w}G")]
            - new_cols[_with_before_suffix(f"GLOBAL_ACTUAL_TOTAL_AVG_{long_w}G")]
        )

    # =====================================================================
    # 6.12  League activity / sample-size features
    # =====================================================================
    for d in [1, 3, 7, 14]:
        new_cols[_with_before_suffix(f"LEAGUE_GAMES_LAST_{d}D")] = _rolling_day_agg(
            ones, dates, d, "count"
        )
    for w in GAME_WINDOWS:
        # Count is trivially the window size once enough games exist,
        # but early in the season it will be < window.  Use rolling count.
        new_cols[_with_before_suffix(f"LEAGUE_GAMES_LAST_{w}G")] = _rolling_game_agg(
            ones, dates, w, "count"
        )

    # =====================================================================
    # 7  Opening-line features  (only when opening line is available
    #    AND is different from the closing line column)
    # =====================================================================
    if has_open:
        # 7.1  Absolute open-to-close move
        for w in [15, 30, 75]:
            new_cols[_with_before_suffix(f"GLOBAL_ABS_OPEN_TO_CLOSE_MOVE_AVG_{w}G")] = (
                _rolling_game_agg(abs_open_to_close_move, dates, w, "mean")
            )
        for d in DAY_WINDOWS:
            new_cols[_with_before_suffix(f"GLOBAL_ABS_OPEN_TO_CLOSE_MOVE_{d}D")] = (
                _rolling_day_agg(abs_open_to_close_move, dates, d, "mean")
            )

        # 7.2  Directional repricing (signed move)
        for w in [15, 30, 75]:
            new_cols[_with_before_suffix(f"GLOBAL_OPEN_TO_CLOSE_MOVE_AVG_{w}G")] = (
                _rolling_game_agg(open_to_close_move, dates, w, "mean")
            )

        # 7.3  Close-better-than-open rate
        for w in [15, 30, 75]:
            new_cols[
                _with_before_suffix(f"GLOBAL_CLOSE_BETTER_THAN_OPEN_RATE_{w}G")
            ] = _rolling_game_agg(close_beats_open_flag, dates, w, "mean")

        # 7.4  Opening-line error features
        open_market_error = actual_pts - open_line
        open_abs_error = open_market_error.abs()
        for w in [15, 30]:
            new_cols[_with_before_suffix(f"GLOBAL_OPEN_MARKET_BIAS_{w}G")] = (
                _rolling_game_agg(open_market_error, dates, w, "mean")
            )
            new_cols[_with_before_suffix(f"GLOBAL_OPEN_MARKET_MAE_{w}G")] = (
                _rolling_game_agg(open_abs_error, dates, w, "mean")
            )
            new_cols[_with_before_suffix(f"GLOBAL_OPEN_MARKET_ERROR_STD_{w}G")] = (
                _rolling_game_agg(open_market_error, dates, w, "std")
            )

    # =====================================================================
    # 8  Cross-book disagreement features
    # =====================================================================
    # 8.1  Per-game cross-book features (already computed above)
    if crossbook_std is not None:
        new_cols[_with_before_suffix("THIS_GAME_CROSSBOOK_TOTAL_STD")] = crossbook_std
        new_cols[_with_before_suffix("THIS_GAME_CROSSBOOK_TOTAL_RANGE")] = (
            crossbook_range
        )

    # 8.2  Rolling league-wide cross-book disagreement
    if crossbook_std is not None:
        for w in [15, 30, 75]:
            new_cols[_with_before_suffix(f"GLOBAL_CROSSBOOK_TOTAL_STD_AVG_{w}G")] = (
                _rolling_game_agg(crossbook_std, dates, w, "mean")
            )

    # =====================================================================
    # Attach all new columns at once (avoids DataFrame fragmentation)
    # =====================================================================
    features_df = pd.DataFrame(new_cols, index=out.index)
    out = pd.concat([out, features_df], axis=1)

    out = out.reset_index(drop=True)

    n_features = len(new_cols)
    print(f"Added {n_features} global market regime features")

    return out
