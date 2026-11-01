"""
NBA Over/Under Predictor - Statistical Analysis Module

This module contains functions for computing rolling statistics, cumulative averages,
trend analysis, and other statistical computations on NBA game and player data.
"""

import numpy as np
import pandas as pd


def _shifted_group_window(
    series: pd.Series,
    keys: list,
    *,
    window: int | None,
    how: str = "mean",
    ddof: int = 1,
) -> pd.Series:
    """``series.groupby(keys).transform(lambda s: s.shift(1).<window>.<how>())``.

    ``window`` is a row count for ``rolling(window, min_periods=1)`` or None
    for ``expanding(min_periods=1)``; ``how`` is ``"mean"`` or ``"std"``.

    ``groupby(...).rolling``/``expanding`` run the same window kernel over
    the groups laid end to end, and the kernel restarts its running sums at
    every group's first row, because that window starts where the previous
    one ended -- exactly the state a per-group call starts from. So each value
    is the same float, without one Python call per group. Rows whose keys are
    missing belong to no group and stay NaN, as with ``transform``.
    """
    positions = pd.RangeIndex(len(series))
    values = series.set_axis(positions)
    group_keys = [k.set_axis(positions) for k in keys]
    shifted = values.groupby(group_keys).shift(1)
    grouped = shifted.groupby(group_keys)
    windows = (
        grouped.expanding(min_periods=1)
        if window is None
        else grouped.rolling(window, min_periods=1)
    )
    result = windows.mean() if how == "mean" else windows.std(ddof=ddof)
    result = result.droplevel(list(range(len(keys)))).reindex(positions)
    return pd.Series(
        result.to_numpy(dtype="float64"), index=series.index, name=series.name
    )


def compute_rolling_stats(
    df: pd.DataFrame,
    param: str = "PTS",
    window: int = 5,
    add_extra_season_avg: bool = False,
    group_by_season: bool = False,
    add_relative_column: bool = True,
    include_home_away_relative: bool = True,
    relative_to_window: int | None = None,
) -> pd.DataFrame:
    """Computes rolling averages for a given `param`, excluding the current row's game.

    Creates:
      - f"{param}_LAST_ALL_{window}_MATCHES_BEFORE"
      - Relative column (if add_relative_column=True):
          - home/away diff vs strict (legacy name reused):
            f"{param}_LAST_HOME_AWAY_{window}_MATCHES_BEFORE"
          - OR strict-window diff:
            f"{param}_LAST_{relative_to_window}_MINUS_LAST_{window}_MATCHES_BEFORE"
      - f"{param}_SEASON_BEFORE_AVG" (if add_extra_season_avg=True)

    Requirements:
      - Columns: TEAM_ID, HOME (bool), GAME_DATE, SEASON_YEAR, and `param`.
        (SEASON_YEAR is used to prevent cross-season contamination when group_by_season=True)

    Parameters:
      - df: Input DataFrame
      - param: Column name to compute rolling stats for
      - window: Number of games in rolling window
      - add_extra_season_avg: If True, add season-to-date average column
      - group_by_season: If True, computes rolling stats within each SEASON_YEAR to prevent
        cross-season contamination. If False, allows rolling across seasons to fill windows.
      - add_relative_column: If True, compute second (relative) feature column.
      - include_home_away_relative: If True, second column is home/away minus strict.
        If False, uses strict-window diff when relative_to_window is provided.
      - relative_to_window: Reference strict window for relative diff
        (e.g., 5 for LAST_5_MINUS_LAST_10).

    Notes:
      - Uses shift(1) everywhere to exclude the current game.
      - When group_by_season=True: computes rolling within (TEAM_ID, SEASON_YEAR) for "ANY"
        and (TEAM_ID, SEASON_YEAR, HOME) for home/away split when needed.
      - When group_by_season=False: computes rolling within (TEAM_ID) for "ANY"
        and (TEAM_ID, HOME) for home/away split when needed, allowing previous seasons.
      - Season averages group by (TEAM_ID, SEASON_YEAR, HOME). Missing values
        fall back to the previous season's mean for the same team/location, then
        the strict rolling average. Without any history they remain NaN.
      - Keeps final sort by GAME_DATE descending to match your pipeline style
    """
    if param not in df.columns:
        return df

    required_cols = {"TEAM_ID", "HOME", "GAME_DATE", "SEASON_YEAR"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_rolling_stats missing required columns: {sorted(missing)}"
        )

    out = df.copy()

    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")

    # Stable deterministic sort for rolling computations:
    # sort within team/season by date, and break ties by GAME_ID if present.
    sort_cols = ["TEAM_ID", "GAME_DATE"]
    if group_by_season:
        sort_cols.insert(1, "SEASON_YEAR")
    if "GAME_ID" in out.columns:
        sort_cols.append("GAME_ID")

    out.sort_values(sort_cols, ascending=True, inplace=True)

    last_n_avg_col = f"{param}_LAST_ALL_{window}_MATCHES_BEFORE"
    season_avg_col = f"{param}_SEASON_BEFORE_AVG"

    # Ensure numeric (keeps NaNs if coercion fails)
    series = pd.to_numeric(out[param], errors="coerce")

    # Build groupby keys based on group_by_season flag
    if group_by_season:
        group_keys_any = [out["TEAM_ID"], out["SEASON_YEAR"]]
        group_keys_homeaway = [out["TEAM_ID"], out["SEASON_YEAR"], out["HOME"]]
    else:
        group_keys_any = [out["TEAM_ID"]]
        group_keys_homeaway = [out["TEAM_ID"], out["HOME"]]

    # 1) Last N games (ANY), excluding current game
    out[last_n_avg_col] = _shifted_group_window(series, group_keys_any, window=window)

    # Optional relative column
    if add_relative_column:
        if include_home_away_relative:
            # Keep legacy column name but store relative value:
            # (last same-home/away N) - (last strict N)
            last_n_homeaway_col = f"{param}_LAST_HOME_AWAY_{window}_MATCHES_BEFORE"
            homeaway_roll = _shifted_group_window(
                series, group_keys_homeaway, window=window
            )
            out[last_n_homeaway_col] = homeaway_roll - out[last_n_avg_col]
        elif relative_to_window is not None:
            relative_col = (
                f"{param}_LAST_{relative_to_window}_MINUS_LAST_{window}_MATCHES_BEFORE"
            )
            if relative_to_window <= 0:
                raise ValueError("relative_to_window must be > 0 when provided.")

            ref_col = f"{param}_LAST_ALL_{relative_to_window}_MATCHES_BEFORE"
            if ref_col in out.columns:
                ref_roll = pd.to_numeric(out[ref_col], errors="coerce")
            else:
                ref_roll = _shifted_group_window(
                    series, group_keys_any, window=relative_to_window
                )

            out[relative_col] = ref_roll - out[last_n_avg_col]

    # 3) Season-to-date average (within season and home/away), excluding current game
    if add_extra_season_avg:
        # Season average always groups by season (even if group_by_season=False for rolling)
        out[season_avg_col] = _shifted_group_window(
            series, [out["TEAM_ID"], out["SEASON_YEAR"], out["HOME"]], window=None
        )

        # Additionally, fill missing season averages using previous season's mean
        # to handle cases where no prior games exist in the current season.
        prev_season_mean = (
            out.assign(_param=series)
            .groupby(["TEAM_ID", "SEASON_YEAR", "HOME"], as_index=False)["_param"]
            .mean()
            .rename(columns={"_param": "_prev_season_mean"})
        )

        # Map previous season's mean onto the next season
        prev_season_mean["SEASON_YEAR"] = prev_season_mean["SEASON_YEAR"] + 1

        out = out.merge(
            prev_season_mean[["TEAM_ID", "SEASON_YEAR", "HOME", "_prev_season_mean"]],
            on=["TEAM_ID", "SEASON_YEAR", "HOME"],
            how="left",
        )

        # Fill order:
        # 1) season-to-date (preferred)
        # 2) previous season mean
        # 3) strict rolling average
        out[season_avg_col] = out[season_avg_col].fillna(out["_prev_season_mean"])
        out[season_avg_col] = out[season_avg_col].fillna(out[last_n_avg_col])

        out.drop(columns=["_prev_season_mean"], inplace=True)

    # Return to your preferred ordering
    out.sort_values(["GAME_DATE"], ascending=False, inplace=True)

    return out


def compute_season_std(df, param="PTS"):
    """
    Computes the season standard deviation for a given parameter (e.g., "PTS"),
    excluding the current row's game. The standard deviation is computed over
    all prior games (using expanding window after a shift).

    The DataFrame must have columns:
        TEAM_ID, SEASON_YEAR, HOME, GAME_DATE, and `param`.
    It sorts by (TEAM_ID, GAME_DATE) ascending so that the "previous" games appear first.

    After calling this function, df will have a new column:
        f"{param}_SEASON_BEFORE_STD"
    which contains the expanding standard deviation of `param` for each group,
    computed using only games before the current one.

    If STD is NaN (not enough games in current season), falls back to previous season's STD.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the necessary columns.
    param : str, default "PTS"
        The column for which to compute the season standard deviation.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an additional column f"{param}_SEASON_BEFORE_STD".
    """
    season_std_col = f"{param}_SEASON_BEFORE_STD"

    # Sort by TEAM_ID and GAME_DATE ascending to ensure proper order.
    df = df.sort_values(["TEAM_ID", "GAME_DATE"], ascending=True).copy()

    # Ensure the column is float so pd.NA values become NaN (avoids TypeError in .std())
    df[param] = df[param].fillna(np.nan).astype(float)

    # Compute expanding standard deviation after shifting by 1 to exclude the current game.
    df[season_std_col] = _shifted_group_window(
        df[param],
        [df["TEAM_ID"], df["SEASON_YEAR"], df["HOME"]],
        window=None,
        how="std",
        ddof=0,
    )

    # Compute previous season's STD as fallback for missing values
    prev_season_std = (
        df.groupby(["TEAM_ID", "SEASON_YEAR", "HOME"], as_index=False)[param]
        .std()
        .rename(columns={param: "_prev_season_std"})
    )

    # Map previous season's STD onto the next season
    prev_season_std["SEASON_YEAR"] = prev_season_std["SEASON_YEAR"] + 1

    df = df.merge(
        prev_season_std[["TEAM_ID", "SEASON_YEAR", "HOME", "_prev_season_std"]],
        on=["TEAM_ID", "SEASON_YEAR", "HOME"],
        how="left",
    )

    # Fill NaN values with previous season's STD
    df[season_std_col] = df[season_std_col].fillna(df["_prev_season_std"])

    # Clean up temporary column
    df.drop(columns=["_prev_season_std"], inplace=True)

    # Optionally, sort the DataFrame back by GAME_DATE descending.
    df = df.sort_values(["GAME_DATE"], ascending=False).reset_index(drop=True)

    return df


def compute_rolling_weighted_stats(
    df,
    param="PTS",
    window=10,
    group_by_season=False,
    add_relative_column: bool = True,
    include_home_away_relative: bool = True,
    relative_to_window: int | None = None,
):
    """
    Computes weighted moving average for a given param, excluding the current row's game.

    Creates:
      - f"{param}_LAST_{window}_WMA_BEFORE"
      - Relative column (if add_relative_column=True):
          - home/away WMA diff vs strict (legacy name reused):
            f"{param}_LAST_HOME_AWAY_{window}_WMA_BEFORE"
          - OR strict-window WMA diff:
            f"{param}_LAST_{relative_to_window}_MINUS_LAST_{window}_WMA_BEFORE"

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TEAM_ID, HOME, GAME_DATE, GAME_ID, SEASON_YEAR, and `param`.
    param : str
        The column for which to compute the weighted moving average.
    group_by_season : bool, default True
        If True, computes rolling stats within each SEASON_YEAR to prevent
        cross-season contamination. If False, allows rolling across seasons.

    Returns
    -------
    pd.DataFrame
        DataFrame with weighted moving average columns added.
    """

    last_n_wma_col = f"{param}_LAST_{window}_WMA_BEFORE"
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")

    sort_cols = ["TEAM_ID", "GAME_DATE", "GAME_ID"]
    if group_by_season:
        sort_cols = ["TEAM_ID", "SEASON_YEAR", "GAME_DATE", "GAME_ID"]

    out.sort_values(sort_cols, ascending=True, inplace=True)

    series = pd.to_numeric(out[param], errors="coerce")

    def weighted_moving_average(x: np.ndarray) -> float:
        """
        Compute weighted moving average with consistent relative weighting.
        Uses exponential-like weights that scale properly for any window size.

        Takes the raw window (``raw=True``): the same float64 values and the
        same numpy reductions as the Series version it replaced, without
        building a Series for every window.
        """
        n = len(x)
        mask = ~np.isnan(x)
        if not mask.any():
            return np.nan

        # Generate weights for the actual window size (1 to n)
        # This ensures consistent relative weighting regardless of window size
        w = np.arange(1, n + 1, dtype=float)

        return float((x[mask] * w[mask]).sum() / w[mask].sum())

    # Build groupby keys
    if group_by_season:
        group_keys_any = [out["TEAM_ID"], out["SEASON_YEAR"]]
        group_keys_homeaway = [out["TEAM_ID"], out["SEASON_YEAR"], out["HOME"]]
    else:
        group_keys_any = [out["TEAM_ID"]]
        group_keys_homeaway = [out["TEAM_ID"], out["HOME"]]

    # Any (team-level)
    out[last_n_wma_col] = series.groupby(group_keys_any).transform(
        lambda s: (
            s.shift(1)
            .rolling(window, min_periods=1)
            .apply(weighted_moving_average, raw=True)
        )
    )

    # Optional relative column
    if add_relative_column:
        if include_home_away_relative:
            # Keep legacy name but store relative value:
            # (last same-home/away WMA) - (last strict WMA)
            last_n_split_col = f"{param}_LAST_HOME_AWAY_{window}_WMA_BEFORE"
            split_wma = series.groupby(group_keys_homeaway).transform(
                lambda s: (
                    s.shift(1)
                    .rolling(window, min_periods=1)
                    .apply(weighted_moving_average, raw=True)
                )
            )
            out[last_n_split_col] = split_wma - out[last_n_wma_col]
        elif relative_to_window is not None:
            relative_col = (
                f"{param}_LAST_{relative_to_window}_MINUS_LAST_{window}_WMA_BEFORE"
            )
            if relative_to_window <= 0:
                raise ValueError("relative_to_window must be > 0 when provided.")

            ref_col = f"{param}_LAST_{relative_to_window}_WMA_BEFORE"
            if ref_col in out.columns:
                ref_wma = pd.to_numeric(out[ref_col], errors="coerce")
            else:
                ref_wma = series.groupby(group_keys_any).transform(
                    lambda s: (
                        s.shift(1)
                        .rolling(relative_to_window, min_periods=1)
                        .apply(weighted_moving_average, raw=True)
                    )
                )
            out[relative_col] = ref_wma - out[last_n_wma_col]

    out.sort_values("GAME_DATE", ascending=False, inplace=True)

    return out
