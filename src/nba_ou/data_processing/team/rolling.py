import numpy as np
import pandas as pd
from tqdm import tqdm

from nba_ou.config.odds_columns import moneyline_col, spread_col, total_line_col
from nba_ou.data_processing.statistics.statistics import (
    compute_rolling_stats,
    compute_rolling_weighted_stats,
    compute_season_std,
)
from nba_ou.data_processing.team.style_matchups import STYLE_SOURCE_COLUMNS

MAIN_TOTAL_LINE_COL = total_line_col()
MAIN_SPREAD_COL = spread_col()
MAIN_MONEYLINE_COL = moneyline_col()

# Module-level constants for rolling statistics computation
COLS_TO_AVERAGE = [
    "PTS",
    "PTS_PER_40",
    "TOTAL_POINTS",
    "OFF_RATING",
    "DEF_RATING",
    "NET_RATING",
    "EFG_PCT",
    "PACE_PER40",
    "FG3A",
    "FG3M",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3_PCT",
    "FTA",
    "FTM",
    "EFG_PCT",
    "TS_PCT",
    "POSS",
    "PIE",
    "PF",
]

COLS_TO_AVERAGE_ODDS = [
    MAIN_TOTAL_LINE_COL,
    "TOTAL_POINTS",
    MAIN_MONEYLINE_COL,
    MAIN_SPREAD_COL,
]

COLS_FOR_WEIGHTED_STATS = [
    "PTS",
    "PTS_PER_40",
    "TOTAL_POINTS",
    MAIN_TOTAL_LINE_COL,
]

COLS_FOR_SEASON_STD = [
    "PTS",
    "PTS_PER_40",
    "TOTAL_POINTS",
    MAIN_TOTAL_LINE_COL,
]

COLS_FOR_SHORT_WINDOWS = [
    "PTS",
    "PTS_PER_40",
    "DIFF_FROM_LINE_bet365",
    MAIN_TOTAL_LINE_COL,
]

IS_OVERTIME_LAST_GAME_BEFORE = "IS_OVERTIME_LAST_GAME_BEFORE"
OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE = "OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE"
OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE = "OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE"
OVERTIME_HISTORY_FEATURE_COLUMNS = (
    IS_OVERTIME_LAST_GAME_BEFORE,
    OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE,
    OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE,
)


def add_overtime_history_features(df):
    """Add strictly pre-game overtime history for each team.

    Last-game and last-five history continue across season boundaries. The
    season frequency resets for each ``SEASON_YEAR``. Rows without a completed
    game outcome (for example, scheduled games with a null ``IS_OVERTIME``)
    receive the latest available history but never enter that history.
    """
    required_columns = {
        "TEAM_ID",
        "GAME_DATE",
        "SEASON_YEAR",
        "IS_OVERTIME",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Cannot compute overtime history; missing required columns: {missing}"
        )

    original_order_column = "__OVERTIME_HISTORY_ORIGINAL_ORDER"
    parsed_date_column = "__OVERTIME_HISTORY_GAME_DATE"
    source_column = "__OVERTIME_HISTORY_SOURCE"
    temporary_columns = {
        original_order_column,
        parsed_date_column,
        source_column,
    }
    collisions = temporary_columns.intersection(df.columns)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(
            f"Cannot compute overtime history; reserved columns already exist: {names}"
        )

    result = df.copy()
    result[original_order_column] = np.arange(len(result))
    result[parsed_date_column] = pd.to_datetime(result["GAME_DATE"], errors="coerce")
    if result[parsed_date_column].isna().any():
        raise ValueError(
            "Cannot compute overtime history; GAME_DATE contains missing or invalid values"
        )

    result[source_column] = pd.to_numeric(result["IS_OVERTIME"], errors="coerce")
    non_numeric_overtime = result["IS_OVERTIME"].notna() & result[source_column].isna()
    invalid_overtime = result[source_column].notna() & ~result[source_column].isin(
        [0, 1]
    )
    if non_numeric_overtime.any() or invalid_overtime.any():
        raise ValueError(
            "Cannot compute overtime history; IS_OVERTIME must contain only 0, 1, or null"
        )

    sort_columns = ["TEAM_ID", parsed_date_column]
    if "GAME_ID" in result.columns:
        sort_columns.append("GAME_ID")
    result = result.sort_values(sort_columns, kind="mergesort")

    team_histories = {}
    season_histories = {}
    last_game_values = []
    last_five_frequencies = []
    season_frequencies = []

    history_columns = ["TEAM_ID", "SEASON_YEAR", source_column]
    for team_id, season_year, overtime_value in result[history_columns].itertuples(
        index=False, name=None
    ):
        team_history = team_histories.setdefault(team_id, [])
        season_history = season_histories.setdefault((team_id, season_year), [])

        last_game_values.append(int(team_history[-1]) if team_history else 0)
        last_five = team_history[-5:]
        last_five_frequencies.append(
            float(sum(last_five) / len(last_five)) if last_five else 0.0
        )
        season_frequencies.append(
            float(sum(season_history) / len(season_history)) if season_history else 0.0
        )

        if pd.notna(overtime_value):
            overtime_value = int(overtime_value)
            team_history.append(overtime_value)
            season_history.append(overtime_value)

    result[IS_OVERTIME_LAST_GAME_BEFORE] = last_game_values
    result[OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE] = last_five_frequencies
    result[OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE] = season_frequencies

    return result.sort_values(original_order_column, kind="mergesort").drop(
        columns=list(temporary_columns)
    )


def _is_total_over_under_price_column(column: str) -> bool:
    """Return whether a column is a raw total-market over/under price."""
    return column.startswith("total_") and column.endswith(
        ("_price_over", "_price_under")
    )


def _get_rolling_price_columns(columns) -> list[str]:
    """Find price columns eligible for historical rolling features.

    Current total-market over/under prices remain in the dataset, but their
    historical rolling/season aggregates are intentionally excluded. Spread
    and moneyline prices continue to receive rolling features.
    """
    return [
        column
        for column in columns
        if "_price" in column
        and column.startswith(("total_", "spread_", "ml_"))
        and not _is_total_over_under_price_column(column)
    ]


#: Season types the previous-season trend fallback is allowed to read.
#: Deliberately regular-season only. A playoff run is a different competitive
#: regime and only 16 teams have one, so sourcing the carried-over trend from
#: whatever a team happened to play last would make the fallback mean something
#: different for a finalist than for a lottery team. Restricting it keeps the
#: value comparable across all 30 teams, which matters because this model is
#: aimed at regular-season games.
TREND_FALLBACK_SEASON_TYPES: tuple[str, ...] = ("Regular Season",)

#: Used when a team has no prior regular season at all -- its first games in the
#: dataset. Matches ``calculate_slope``'s own "not enough data" return value, so
#: "no trend" has one representation throughout this module rather than two.
TREND_NO_HISTORY_VALUE = 0.0

SEASON_TYPE_COLUMN = "SEASON_TYPE"


def _previous_regular_season_trend(df, values, *, extra_keys=()):
    """The trend each team ended its PREVIOUS regular season with.

    Same shape of fallback as ``_SEASON_BEFORE_AVG`` in
    ``statistics.compute_rolling_stats``, adapted to a slope: for a mean the
    natural carried-over value is the previous season's mean, and for a trend it
    is the last trend the team actually held -- the slope over its final regular
    season games. Both are strictly-past quantities, so neither leaks.

    ``extra_keys`` mirrors the grouping of the column being filled, so the
    home/away slope falls back to the previous season's home/away slope rather
    than to its overall one.

    Returns a positional array; the caller never relies on index alignment,
    since the frame may carry a duplicated index.
    """
    values = pd.Series(values).to_numpy(dtype="float64", na_value=np.nan)
    if SEASON_TYPE_COLUMN not in df.columns:
        # Minimal frames (tests, ad hoc use) have no season type to filter on.
        # Falling back to "no history" is the honest answer -- better than
        # quietly sourcing the value from playoff games.
        return np.full(len(df), np.nan)

    keys = ["TEAM_ID", *extra_keys]
    frame = df[[*keys, "SEASON_YEAR"]].reset_index(drop=True)
    frame["_trend"] = values

    is_regular = (
        df[SEASON_TYPE_COLUMN].isin(TREND_FALLBACK_SEASON_TYPES).to_numpy(dtype=bool)
    )
    # groupby.last() skips NaN, so this is the last *computed* trend of the
    # season -- not whatever happened to sit in the final row.
    season_end = (
        frame[is_regular]
        .groupby([*keys, "SEASON_YEAR"], as_index=False, dropna=False)["_trend"]
        .last()
    )
    if season_end.empty:
        return np.full(len(df), np.nan)

    # Carry each season's closing trend forward into the season that follows it.
    season_end["SEASON_YEAR"] = season_end["SEASON_YEAR"] + 1
    season_end = season_end.rename(columns={"_trend": "_previous_trend"})

    merged = frame.merge(season_end, on=[*keys, "SEASON_YEAR"], how="left")
    return merged["_previous_trend"].to_numpy(dtype="float64", na_value=np.nan)


def _fill_trend_from_history(df, values, *, extra_keys=()):
    """Resolve a trend column through the full fallback chain.

    Current season (preferred) -> previous regular season's closing trend -> 0.

    Without this the first two games of every season are NaN in every slope
    column: the rolling window resets at the season boundary and ``min_periods=2``
    needs two prior games to produce anything. That is ~1,100 columns in the
    closing-line dataset, which is enough NaN to get the whole row discarded
    downstream -- so the model never sees the start of a season at all.
    """
    values = pd.Series(values).to_numpy(dtype="float64", na_value=np.nan)
    previous = _previous_regular_season_trend(df, values, extra_keys=extra_keys)
    filled = np.where(np.isnan(values), previous, values)
    return np.where(np.isnan(filled), TREND_NO_HISTORY_VALUE, filled)


def _linregress_slope(x: np.ndarray, y: np.ndarray) -> float:
    """``linregress(x, y).slope``, bit for bit, without the rest of the fit.

    ``linregress`` derives the slope as ``ssxym / ssxm`` from
    ``np.cov(x, y, bias=1)`` and then spends most of its time on the
    correlation, p-value and standard errors, which the trend features discard.
    This is that same expression on the same arrays. ``x`` is always
    ``1..n`` with ``n >= 2`` here, so linregress's constant-``x`` error
    cannot occur.
    """
    ssxm, ssxym, _, _ = np.cov(x, y, bias=1).flat
    return ssxym / ssxm


def compute_trend_slope(
    df,
    parameter="PTS",
    window=10,
    shift_current_game=True,
    add_relative_column: bool = True,
    include_home_away_relative: bool = True,
    relative_to_window: int | None = None,
):
    """
    Computes the slope of a linear regression line over the last `window` games
    to determine whether a team's performance is increasing, decreasing, or stable.

    Works on a dataframe with one row per team per game (2 rows per match).

    Args:
        df (pd.DataFrame): Must contain columns "TEAM_ID", "SEASON_YEAR", "GAME_DATE", "HOME", and `parameter`.
        parameter (str): The statistic to analyze (e.g., "PTS").
        window (int): Number of last games to consider.
        shift_current_game (bool): Whether to exclude the current game from the trend calculation.

    Returns:
        pd.DataFrame: A modified DataFrame with new columns:
            - f"{parameter}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE" (strict all-games trend)
            - Relative column (if add_relative_column=True):
                - home/away minus strict (legacy name reused):
                  f"{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_GAMES_BEFORE"
                - OR strict-window diff:
                  f"{parameter}_TREND_SLOPE_LAST_{relative_to_window}_MINUS_LAST_{window}_GAMES_BEFORE"
    """

    def calculate_slope(series):
        """Applies linear regression to compute the trend slope."""
        # Remove NaN and None values
        clean_series = [x for x in series if x is not None and not np.isnan(x)]

        if len(clean_series) < 2:
            return 0  # Not enough data for a trend

        X = np.arange(1, len(clean_series) + 1)  # Time index [1, 2, ..., N]
        Y = np.array(clean_series)  # Convert to array for linregress

        return _linregress_slope(X, Y)

    # Sort by team, season, and date
    df = df.sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True)

    # 1. Overall trend (all games for each team)
    trend_col = f"{parameter}_TREND_SLOPE_LAST_{window}_GAMES_BEFORE"
    within_season_trend = df.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter].transform(
        lambda s: (
            (s.shift(1) if shift_current_game else s)
            .rolling(window, min_periods=2)
            .apply(calculate_slope, raw=True)
        )
    )
    # The grouping stays per season on purpose -- a slope spanning the offseason
    # is not a trend. The gap it leaves at the start of each season is closed by
    # the fallback chain instead.
    df[trend_col] = _fill_trend_from_history(df, within_season_trend)

    if add_relative_column:
        if include_home_away_relative:
            # 2A. Relative trend using home/away context:
            # (home/away trend) - (strict all-games trend)
            trend_col_location = (
                f"{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_GAMES_BEFORE"
            )
            trend_col_location_raw = (
                f"__{parameter}_TREND_SLOPE_LAST_{window}_HOME_AWAY_RAW"
            )

            home_mask = df["HOME"] == 1
            df_temp_home = (
                df[home_mask]
                .copy()
                .sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True)
            )
            df.loc[home_mask, trend_col_location_raw] = (
                df_temp_home.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter]
                .transform(
                    lambda s: (
                        (s.shift(1) if shift_current_game else s)
                        .rolling(window, min_periods=2)
                        .apply(calculate_slope, raw=True)
                    )
                )
                .values
            )

            away_mask = df["HOME"] == 0
            df_temp_away = (
                df[away_mask]
                .copy()
                .sort_values(["TEAM_ID", "SEASON_YEAR", "GAME_DATE"], ascending=True)
            )
            df.loc[away_mask, trend_col_location_raw] = (
                df_temp_away.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter]
                .transform(
                    lambda s: (
                        (s.shift(1) if shift_current_game else s)
                        .rolling(window, min_periods=2)
                        .apply(calculate_slope, raw=True)
                    )
                )
                .values
            )

            # Filled before the subtraction, and keyed on HOME so the fallback is
            # the previous season's home/away trend rather than its overall one.
            # Filling afterwards would be too late: NaN - filled is still NaN.
            df[trend_col_location_raw] = _fill_trend_from_history(
                df, df[trend_col_location_raw], extra_keys=["HOME"]
            )

            df[trend_col_location] = df[trend_col_location_raw] - df[trend_col]
            df.drop(columns=[trend_col_location_raw], inplace=True)
        elif relative_to_window is not None:
            # 2B. Relative strict trend across windows (e.g., last_5 - last_10)
            relative_col = f"{parameter}_TREND_SLOPE_LAST_{relative_to_window}_MINUS_LAST_{window}_GAMES_BEFORE"
            if relative_to_window <= 0:
                raise ValueError("relative_to_window must be > 0 when provided.")

            ref_col = f"{parameter}_TREND_SLOPE_LAST_{relative_to_window}_GAMES_BEFORE"
            if ref_col in df.columns:
                # Already produced by an earlier call, so already filled.
                reference_trend = df[ref_col].to_numpy(dtype="float64", na_value=np.nan)
            else:
                reference_trend = _fill_trend_from_history(
                    df,
                    df.groupby(["TEAM_ID", "SEASON_YEAR"])[parameter].transform(
                        lambda s: (
                            (s.shift(1) if shift_current_game else s)
                            .rolling(relative_to_window, min_periods=2)
                            .apply(calculate_slope, raw=True)
                        )
                    ),
                )

            df[relative_col] = reference_trend - df[trend_col]

    return df


#: Columns the rolling stage reads besides the statistics themselves.
_ROLLING_KEY_COLUMNS = (
    "TEAM_ID",
    "HOME",
    "GAME_DATE",
    "SEASON_YEAR",
    "GAME_ID",
    SEASON_TYPE_COLUMN,
)
#: Scratch columns the statistics helpers create and drop again.
_ROLLING_SCRATCH_COLUMNS = ("_prev_season_mean", "_prev_season_std")
_ROW_POSITION = "__rolling_row_position__"


def _rolling_source_columns(columns, exclude_yahoo: bool) -> list[str]:
    """Columns the rolling stage can read or write, in frame order.

    Every column the discovery rules in ``_compute_all_rolling_statistics_on``
    can pick up is included, so discovery on this subset returns the same lists
    as on the full frame. Every column the stage can create is named
    ``<source>_...`` or ``__<source>_...``, so any such column already in the
    frame is included too and gets overwritten or dropped exactly as before.
    """
    columns = list(columns)
    discovered = {
        c
        for c in columns
        if c.startswith(("DIFF_FROM_", "ODDS_TOTAL_LINE_"))
        or c.startswith(
            (
                "total_consensus_pct_",
                "spread_consensus_pct_",
                "moneyline_consensus_pct_",
            )
        )
        or (not exclude_yahoo and ("_pct_bets" in c or "_pct_money" in c))
    }
    discovered.update(_get_rolling_price_columns(columns))
    sources = discovered | set(
        COLS_TO_AVERAGE
        + COLS_TO_AVERAGE_ODDS
        + COLS_FOR_WEIGHTED_STATS
        + COLS_FOR_SEASON_STD
        + COLS_FOR_SHORT_WINDOWS
        + list(STYLE_SOURCE_COLUMNS)
    )
    prefixes = tuple(f"{s}_" for s in sources) + tuple(f"__{s}_" for s in sources)
    return [
        c
        for c in columns
        if c in sources
        or c in _ROLLING_KEY_COLUMNS
        or c in _ROLLING_SCRATCH_COLUMNS
        or c.startswith(prefixes)
    ]


def compute_all_rolling_statistics(df, exclude_yahoo=False):
    """``_compute_all_rolling_statistics_on`` without dragging the wide frame.

    Each of the ~150 helper calls in that stage copies, sorts, merges and
    re-sorts the frame it is given, to add one to three columns. Given the full
    team frame -- hundreds of columns wide by then -- that copying was nearly
    all of the stage's time. The helpers only read the key columns and the
    statistics themselves, and every row permutation they apply depends only on
    the keys and the current row order. So the stage runs unchanged on just
    those columns, and the wide frame is reordered once at the end: same rows,
    order, index, columns and values as running it on the whole frame.
    """
    columns = pd.Index(df.columns)
    if not columns.is_unique or _ROW_POSITION in columns:
        return _compute_all_rolling_statistics_on(df, exclude_yahoo=exclude_yahoo)

    narrow_columns = _rolling_source_columns(columns, exclude_yahoo)
    narrow = df[narrow_columns].copy()
    narrow[_ROW_POSITION] = np.arange(len(df))
    narrow = _compute_all_rolling_statistics_on(narrow, exclude_yahoo=exclude_yahoo)

    narrow_set = set(narrow_columns)
    created = [c for c in narrow.columns if c not in narrow_set and c != _ROW_POSITION]
    if set(created) & set(columns):
        # Unreachable given _rolling_source_columns; kept as the safe path.
        return _compute_all_rolling_statistics_on(df, exclude_yahoo=exclude_yahoo)

    positions = narrow.pop(_ROW_POSITION).to_numpy()
    untouched = [c for c in columns if c not in narrow_set]
    result = pd.concat(
        [
            df[untouched].take(positions).reset_index(drop=True),
            narrow.reset_index(drop=True),
        ],
        axis=1,
    )
    final_columns = [
        c for c in columns if c not in narrow_set or c in narrow.columns
    ] + created
    result = result[final_columns]
    result.index = narrow.index
    result.columns.name = columns.name
    return result


def _compute_all_rolling_statistics_on(df, exclude_yahoo=False):
    """
    Compute rolling statistics, weighted averages, and seasonal standard deviations,
    dynamically including new DIFF_FROM_* columns, ODDS_TOTAL_LINE_* columns, and
    selected odds data.

    Args:
        df (pd.DataFrame): DataFrame with game statistics
        exclude_yahoo (bool): If True, exclude Yahoo-specific betting columns (pct_bets, pct_money)
                             from rolling statistics. Default is False (include Yahoo columns).

    Includes all total line columns (ODDS_TOTAL_LINE_*) and their
    corresponding DIFF_FROM_* columns in rolling stats, weighted stats, and season std.
    Also includes odds percentages and spread/moneyline prices. Raw total-market
    over/under prices are retained without historical rolling derivatives.
    """
    original_columns = set(df.columns)

    # 1) Dynamically discover new diff columns
    new_diff_cols = [c for c in df.columns if c.startswith("DIFF_FROM_")]

    # 2) Dynamically discover all ODDS_TOTAL_LINE_* columns
    new_total_line_cols = [c for c in df.columns if c.startswith("ODDS_TOTAL_LINE_")]

    # 3) Dynamically discover Yahoo betting columns (percentage of bets/money)
    # Note: spread/ml yahoo columns are now team-specific (without _home/_away suffix after merge)
    yahoo_cols = []
    if not exclude_yahoo:
        yahoo_patterns = ["_pct_bets", "_pct_money"]
        yahoo_cols = [
            c for c in df.columns if any(pattern in c for pattern in yahoo_patterns)
        ]

    # 4) Dynamically discover consensus percentage columns
    consensus_pct_cols = [
        c
        for c in df.columns
        if (
            c.startswith("total_consensus_pct_")
            or c.startswith("spread_consensus_pct_")
            or c.startswith("moneyline_consensus_pct_")
        )
    ]

    # 5) Dynamically discover rolling-eligible prices. Current total over/under
    # prices stay as raw game-level inputs and are deliberately not rolled.
    # Spread/ml prices are team-specific (without _home/_away suffix after merge).
    price_cols = _get_rolling_price_columns(df.columns)

    # 6) Optional: ensure we only include diffs that correspond to totals lines
    # (keeps things tight if you have other DIFF_FROM_* features in the future)
    total_line_cols = {c for c in df.columns if c.startswith("ODDS_TOTAL_LINE_")}
    allowed_suffixes = set()
    for tl in total_line_cols:
        allowed_suffixes.add(
            tl.replace("TOTAL_", "")
        )  # OVER_UNDER_LINE or LINE_betmgm, etc.
    new_diff_cols = [
        c for c in new_diff_cols if c.replace("DIFF_FROM_", "") in allowed_suffixes
    ]

    # 7) Build local versions of lists (do not mutate module-level constants)
    cols_to_average_odds = (
        COLS_TO_AVERAGE_ODDS
        + new_diff_cols
        + new_total_line_cols
        + yahoo_cols
        + consensus_pct_cols
        + price_cols
    )

    # Weighted stats: include total lines, prices, and consensus percentages
    cols_for_weighted_stats = (
        COLS_FOR_WEIGHTED_STATS + new_total_line_cols + consensus_pct_cols
    )

    # Season std: include diffs, total lines, yahoo, consensus, and is_over columns
    cols_for_season_std = (
        COLS_FOR_SEASON_STD
        + new_diff_cols
        + new_total_line_cols
        + yahoo_cols
        + consensus_pct_cols
    )
    cols_for_season_std = list(dict.fromkeys(cols_for_season_std))

    # 8) Rolling stats loop
    rolling_columns = list(
        dict.fromkeys(
            COLS_TO_AVERAGE + cols_to_average_odds + list(STYLE_SOURCE_COLUMNS)
        )
    )
    for col in tqdm(rolling_columns, desc="Computing rolling stats"):
        is_style_source = col in STYLE_SOURCE_COLUMNS
        df = compute_rolling_stats(
            df,
            col,
            window=5,
            add_extra_season_avg=True,
            group_by_season=False,
            add_relative_column=not is_style_source,
        )

        # Style history is an intermediate input for the matchup layer. Keep
        # only the stable season-before estimate; the last-five column is used
        # as its early-season fallback inside compute_rolling_stats and then
        # removed to avoid exposing a large parallel family of raw features.
        if is_style_source:
            df = df.drop(
                columns=[f"{col}_LAST_ALL_5_MATCHES_BEFORE"],
                errors="ignore",
            )

        if (
            col
            in COLS_FOR_SHORT_WINDOWS
            + new_total_line_cols
            + new_diff_cols
            + consensus_pct_cols
        ):
            df = compute_rolling_stats(
                df,
                col,
                window=10,
                add_extra_season_avg=False,
                group_by_season=False,
                include_home_away_relative=False,
                relative_to_window=5,
            )

        if col in cols_for_weighted_stats:
            df = compute_rolling_weighted_stats(
                df, col, window=5, group_by_season=False
            )

        # Extra short windows for all DIFF columns (legacy + new ones)
        if col in COLS_FOR_SHORT_WINDOWS + new_diff_cols:
            df = compute_rolling_stats(
                df,
                col,
                window=1,
                add_extra_season_avg=False,
                add_relative_column=False,
            )
            df = compute_rolling_stats(
                df,
                col,
                window=2,
                add_extra_season_avg=False,
                add_relative_column=False,
            )
            df = compute_rolling_stats(
                df,
                col,
                window=3,
                add_extra_season_avg=False,
                add_relative_column=False,
            )

    # 9) Seasonal std loop
    for param in tqdm(cols_for_season_std, desc="Computing seasonal std"):
        df = compute_season_std(df, param=param)

    # 10) Compute trend slopes for teams
    print("Computing team performance trends...")
    df = compute_trend_slope(df, parameter="PTS", window=5, shift_current_game=True)
    df = compute_trend_slope(
        df,
        parameter="PTS",
        window=10,
        shift_current_game=True,
        include_home_away_relative=False,
        relative_to_window=5,
    )
    df = compute_trend_slope(
        df, parameter="PTS_PER_40", window=5, shift_current_game=True
    )
    df = compute_trend_slope(
        df,
        parameter="PTS_PER_40",
        window=10,
        shift_current_game=True,
        include_home_away_relative=False,
        relative_to_window=5,
    )

    # 11) Diff-from-line source columns (post-game): exclude current game.
    # Use the list captured before rolling features were created. Rediscovering
    # from df here would also match derived columns such as LAST/SEASON/WMA and
    # recursively create trends of already-derived features.
    for col in tqdm(new_diff_cols, desc="Computing diff-from-line trends"):
        df = compute_trend_slope(df, parameter=col, window=5, shift_current_game=True)

    # 12) Total-line source columns (pre-game known). As above, only use columns
    # discovered at function entry so rolling/season features are not trended.
    total_line_trend_cols = new_total_line_cols.copy()
    if MAIN_TOTAL_LINE_COL in total_line_trend_cols:
        total_line_trend_cols = [MAIN_TOTAL_LINE_COL] + [
            c for c in total_line_trend_cols if c != MAIN_TOTAL_LINE_COL
        ]
    for col in tqdm(total_line_trend_cols, desc="Computing total line trends"):
        if col in df.columns:
            df = compute_trend_slope(
                df, parameter=col, window=5, shift_current_game=True
            )

    # 13) Consensus percentage trends (pre-game known): shift current game
    for col in tqdm(consensus_pct_cols, desc="Computing consensus % trends"):
        if col in df.columns:
            df = compute_trend_slope(
                df, parameter=col, window=5, shift_current_game=True
            )

    # 14) Yahoo percentage trends (if included)
    if not exclude_yahoo:
        for col in tqdm(yahoo_cols, desc="Computing Yahoo % trends"):
            if col in df.columns:
                df = compute_trend_slope(
                    df, parameter=col, window=5, shift_current_game=True
                )

    # 15) Enforce naming convention: all newly computed rolling/stat columns must contain _BEFORE
    # Only apply to columns created inside this function (keep source columns untouched).
    new_columns = set(df.columns) - original_columns
    rename_map = {}
    for col in new_columns:
        is_computed_stat = (
            ("_LAST_" in col and ("_MATCHES" in col or "_WMA_" in col))
            or ("_TREND_SLOPE_" in col)
            or ("_SEASON_" in col and (col.endswith("_AVG") or col.endswith("_STD")))
        )
        if is_computed_stat and "_BEFORE" not in col:
            rename_map[col] = f"{col}_BEFORE"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df
