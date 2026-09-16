import numpy as np
import pandas as pd
from nba_ou.config.constants import (
    TEAM_NAME_CONFERENCE_MAP,
    TEAM_NAME_DIVISION_MAP,
    TEAM_NAME_STANDARDIZATION,
)
from nba_ou.config.odds_columns import moneyline_col, spread_col, total_line_col
from nba_ou.utils.general_utils import _with_before_suffix
from pandas.tseries.holiday import USFederalHolidayCalendar

MAIN_TOTAL_LINE_COL = total_line_col()
MAIN_SPREAD_COL = spread_col()
MAIN_MONEYLINE_COL = moneyline_col()


def add_fresh_absence_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Create HOME + AWAY **sums** for the fresh-absence features.

    The betting rollups below get a home-minus-away difference because they
    measure relative strength. Absences on a totals market are the opposite
    case: a star missing on either side moves the same total, so the two sides
    add rather than cancel, and a difference would net them to zero in exactly
    the games where the news is biggest (both teams missing someone).

    Emits ``<name>_SUM_BEFORE`` for every fresh-absence column that has both a
    ``_TEAM_HOME`` and a ``_TEAM_AWAY`` side. Idempotent: an existing sum column
    is refreshed in place.
    """
    new_features = {}
    for home_col in [col for col in df.columns if col.endswith("_BEFORE_TEAM_HOME")]:
        if not home_col.startswith("INJ_"):
            continue
        away_col = home_col.replace("_TEAM_HOME", "_TEAM_AWAY")
        if away_col not in df.columns:
            continue
        sum_col = home_col.replace("_BEFORE_TEAM_HOME", "_SUM_BEFORE")
        new_features[sum_col] = df[home_col] + df[away_col]

    if not new_features:
        return df

    cols_to_drop = [col for col in new_features if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    print(f"Created {len(new_features)} fresh-absence sum feature(s)")
    return df


def add_betting_stats_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create difference features (HOME - AWAY) for all betting-related rolling statistics.

    This function identifies columns representing team-level betting stats with rolling
    windows (those with _TEAM_HOME and _TEAM_AWAY suffixes) and creates difference features
    to capture the betting market's relative assessment of each team.

    For machine learning, differences often provide stronger signal than absolute values
    because they directly encode relative strength/weakness between opponents.

    Args:
        df: Merged DataFrame with _TEAM_HOME and _TEAM_AWAY columns

    Returns:
        DataFrame with additional difference columns (suffix _DIFF_BEFORE)

    Example:
        ODDS_TOTAL_LINE_<book>_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME -
        ODDS_TOTAL_LINE_<book>_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY =
        ODDS_TOTAL_LINE_<book>_LAST_ALL_5_MATCHES_DIFF_BEFORE
    """
    # Patterns that identify betting-related stats
    betting_patterns = [
        MAIN_TOTAL_LINE_COL,
        "ODDS_TOTAL_LINE_",
        "DIFF_FROM_",
        "IS_OVER_",
        MAIN_MONEYLINE_COL,
        MAIN_SPREAD_COL,
        "ODDS_MONEYLINE_",
        "ODDS_SPREAD_",
        "_pct_bets_",
        "_pct_money_",
        "consensus_pct_",
        "_price_",
        "TOTAL_POINTS",  # Include total points predictions/actuals
    ]

    # Patterns that identify rolling/derived stat suffixes (team-specific features)
    rolling_patterns = [
        "_LAST_ALL_",
        "_LAST_HOME_AWAY_",
        "_SEASON_BEFORE_AVG",
        "_SEASON_BEFORE_STD",
        "_WEIGHTED_",
        "_TREND_SLOPE_",
    ]

    # Find all home columns that match betting and rolling patterns
    home_cols = [col for col in df.columns if col.endswith("_TEAM_HOME")]

    new_features = {}
    updated_features = {}

    for home_col in home_cols:
        # Check if this is a betting stat with rolling window/derived stat
        is_betting_stat = any(pattern in home_col for pattern in betting_patterns)
        is_rolling_stat = any(pattern in home_col for pattern in rolling_patterns)

        if is_betting_stat and is_rolling_stat:
            # Get corresponding away column
            away_col = home_col.replace("_TEAM_HOME", "_TEAM_AWAY")

            if away_col in df.columns:
                # Create difference feature name with consistent temporal suffix.
                if "_BEFORE_TEAM_HOME" in home_col:
                    diff_col = home_col.replace("_BEFORE_TEAM_HOME", "_DIFF_BEFORE")
                else:
                    diff_col = home_col.replace("_TEAM_HOME", "_DIFF_BEFORE")

                # Calculate difference (HOME - AWAY)
                # Positive values indicate home team has higher value for this metric
                diff_values = df[home_col] - df[away_col]
                if diff_col in df.columns:
                    updated_features[diff_col] = diff_values
                else:
                    new_features[diff_col] = diff_values

    # Idempotent behavior: overwrite existing diff columns in place, add only missing ones.
    features_to_apply = {**updated_features, **new_features}
    if features_to_apply:
        # Use pd.concat to avoid fragmentation warnings
        # Drop existing columns that will be updated
        cols_to_drop = [col for col in updated_features.keys() if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Create DataFrame from new features and concat all at once
        features_df = pd.DataFrame(features_to_apply, index=df.index)
        df = pd.concat([df, features_df], axis=1)

        print(
            f"Created {len(new_features)} and refreshed {len(updated_features)} "
            "betting statistics difference feature(s)"
        )

    return df


def apply_final_transformations(df_training):
    """
    Apply final transformations to the training dataset.

    Args:
        df_training (pd.DataFrame): DataFrame with derived features

    Returns:
        pd.DataFrame: Final training-ready DataFrame
    """
    df_training.loc[df_training["MATCHUP_TEAM_HOME"] == 0, "MATCHUP_TEAM_HOME"] = (
        df_training["TEAM_ABBREVIATION_TEAM_HOME"]
        + " vs. "
        + df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
    )

    df_training.loc[df_training["MATCHUP_TEAM_AWAY"] == 0, "MATCHUP_TEAM_AWAY"] = (
        df_training["TEAM_ABBREVIATION_TEAM_AWAY"]
        + " @ "
        + df_training["TEAM_ABBREVIATION_TEAM_HOME"]
    )

    # IS_OVERTIME is deliberately RETAINED (it used to be dropped here).
    #
    # It is a post-game fact -- an overtime game has at least five extra minutes
    # of scoring -- so it must never be used as a model feature. It is kept as a
    # row attribute, like GAME_ID and SEASON_TYPE, so downstream code can FILTER
    # on it: training_pipeline can exclude overtime games from the training rows
    # while keeping them in validation and test, which is impossible once the
    # column is gone.
    #
    # Safe to keep: select_train_columns lists IS_OVERTIME in STATIC_COLUMNS, so
    # the leakage guard there already allows it through, and production
    # retraining selects features by name from the saved
    # schema_info.feature_names rather than by dropping an exclude list, so a new
    # column cannot silently become a production feature.
    # training_pipeline additionally names it in LEAKING_TARGET_COLUMNS and
    # asserts it never reaches the feature matrix.

    # Move the configured main total line column to first position (when present)
    first_col = MAIN_TOTAL_LINE_COL
    if first_col in df_training.columns:
        df_training = df_training[
            [first_col] + [col for col in df_training.columns if col != first_col]
        ]

    return df_training


def add_conference_division_features(
    df: pd.DataFrame,
    home_team_col: str = "TEAM_NAME_TEAM_HOME",
    away_team_col: str = "TEAM_NAME_TEAM_AWAY",
) -> pd.DataFrame:
    """
    Adds SAME_CONFERENCE_BEFORE, SAME_DIVISION_BEFORE, IS_HOME_WEST_CONFERENCE_BEFORE,
    IS_AWAY_WEST_CONFERENCE_BEFORE
    based on standardized team names and lookup maps.

    Notes:
    - Unknown team names map to pd.NA for conference/division and yield 0 for boolean flags.
    - SAME_* becomes 1 only when both sides are known AND equal.
    """
    # 1) Standardize names
    home_name = df[home_team_col].astype(str).map(TEAM_NAME_STANDARDIZATION)
    away_name = df[away_team_col].astype(str).map(TEAM_NAME_STANDARDIZATION)

    # If mapping returns None/NaN, fall back to original string
    home_name = home_name.where(home_name.notna(), df[home_team_col].astype(str))
    away_name = away_name.where(away_name.notna(), df[away_team_col].astype(str))

    # 2) Map to conference/division (unmapped -> pd.NA)
    home_conference = home_name.map(TEAM_NAME_CONFERENCE_MAP)
    away_conference = away_name.map(TEAM_NAME_CONFERENCE_MAP)

    home_division = home_name.map(TEAM_NAME_DIVISION_MAP)
    away_division = away_name.map(TEAM_NAME_DIVISION_MAP)

    # 3) SAME_* only if both are known
    same_conference = (
        home_conference.notna()
        & away_conference.notna()
        & (home_conference == away_conference)
    )
    same_division = (
        home_division.notna() & away_division.notna() & (home_division == away_division)
    )

    # 4) West flags (overwrite-safe)
    df = df.assign(
        SAME_CONFERENCE_BEFORE=same_conference.astype(int),
        SAME_DIVISION_BEFORE=same_division.astype(int),
        IS_HOME_WEST_CONFERENCE_BEFORE=(home_conference == "West")
        .fillna(False)
        .astype(int),
        IS_AWAY_WEST_CONFERENCE_BEFORE=(away_conference == "West")
        .fillna(False)
        .astype(int),
    )

    return df


def add_game_date_features(
    df: pd.DataFrame,
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """
    Adds IS_WEEKEND_BEFORE, MONTH_BEFORE, IS_US_HOLIDAY_BEFORE derived from GAME_DATE.

    - IS_WEEKEND_BEFORE: Saturday/Sunday -> 1 else 0
    - MONTH_BEFORE: integer 1..12
    - IS_US_HOLIDAY_BEFORE: US federal holiday -> 1 else 0

    Notes:
    - Robust to strings / datetimes; coerces invalid dates to NaT.
    - If date is NaT, all features default to 0 (and MONTH_BEFORE becomes pd.NA then filled to 0).
    """
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Compute all new columns first
    month_values = dates.dt.month.astype("Int64").fillna(0).astype(int)

    # US Federal holidays within observed min/max date range
    cal = USFederalHolidayCalendar()

    if dates.notna().any():
        start = dates.min().normalize()
        end = dates.max().normalize()
        us_holidays = cal.holidays(start=start, end=end)  # DatetimeIndex (normalized)
        is_us_holiday = dates.dt.normalize().isin(us_holidays).fillna(False).astype(int)
    else:
        is_us_holiday = pd.Series(0, index=df.index, dtype=int)

    # Add all at once (overwrite-safe)
    df = df.assign(
        IS_WEEKEND_BEFORE=(dates.dt.weekday >= 5).fillna(False).astype(int),
        MONTH_BEFORE=month_values,
        IS_US_HOLIDAY_BEFORE=is_us_holiday,
    )

    return df


def add_derived_features_after_computed_stats(df_training):
    """
    Add derived features to the training dataset.

    Args:
        df_training (pd.DataFrame): DataFrame with selected training columns

    Returns:
        pd.DataFrame: DataFrame with additional derived features
    """
    df_training = add_conference_division_features(df_training)

    # Compute and assign derived features (overwrite-safe)
    df_training = df_training.assign(
        TOTAL_PTS_SEASON_AVG_BEFORE=(
            df_training["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
            + df_training["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
        ),
        TOTAL_PTS_LAST_GAMES_AVG_BEFORE=(
            df_training["PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"]
            + df_training["PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY"]
        ),
        BACK_TO_BACK_BEFORE=(
            (df_training["REST_DAYS_BEFORE_MATCH_TEAM_AWAY"] <= 1)
            & (df_training["REST_DAYS_BEFORE_MATCH_TEAM_HOME"] <= 1)
        ),
        DIFERENCE_HOME_OFF_AWAY_DEF_BEFORE=(
            df_training["OFF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"]
            - df_training["DEF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY"]
        ),
        DIFERENCE_AWAY_OFF_HOME_DEF_BEFORE=(
            df_training["OFF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY"]
            - df_training["DEF_RATING_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"]
        ),
    )

    df_training = apply_final_transformations(df_training)

    return df_training


def add_high_value_features_for_team_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of high-value, numeric-only engineered features for predicting
    TEAM points (works well with XGBoost).

    This function is defensive:
      - It only creates features when the required source columns exist.
      - It never uses player NAME columns.
      - It avoids target leakage (does not use same-game points).
      - It ONLY adds features that are not trivially redundant with existing columns.

    Returns a COPY of df with new columns appended.
    """
    out = df.copy()

    # Dictionary to collect all new columns
    new_cols = {}

    def _has(cols):
        return all(c in out.columns for c in cols)

    def _safe_div(num, den):
        den = den.replace(0, np.nan)
        return num / den

    def _add(name, series):
        col_name = _with_before_suffix(name)
        # Add only if it does not already exist (avoid overwriting any preexisting column).
        if col_name in out.columns:
            return
        new_cols[col_name] = pd.to_numeric(series, errors="coerce")

    # ---------------------------------------------------------------------
    # 1) Market-derived: implied team totals (strong, non-leaky)
    # ---------------------------------------------------------------------
    # Uses game-level main ODDS_TOTAL_LINE_<book> and SPREAD (assumes spread is from
    # HOME perspective: home - away). If your spread sign is opposite, flip it.
    # Prefer main close spread from the configured book; fallback to consensus opener.
    spread_col = None
    if f"{MAIN_SPREAD_COL}_TEAM_HOME" in out.columns:
        spread_col = f"{MAIN_SPREAD_COL}_TEAM_HOME"
    elif "spread_consensus_opener_line_home" in out.columns:
        spread_col = "spread_consensus_opener_line_home"

    if spread_col and _has([MAIN_TOTAL_LINE_COL]):
        spread_val = pd.to_numeric(out[spread_col], errors="coerce")
        _add(
            "IMPLIED_PTS_HOME",
            (out[MAIN_TOTAL_LINE_COL] / 2.0) - (spread_val / 2.0),
        )
        _add(
            "IMPLIED_PTS_AWAY",
            (out[MAIN_TOTAL_LINE_COL] / 2.0) + (spread_val / 2.0),
        )
        # NOTE: Do NOT add implied sum/diff checks: they are algebraic restatements of
        # main ODDS_TOTAL_LINE_<book> and spread, hence redundant.

    # ---------------------------------------------------------------------
    # 2) Offense-vs-defense interaction (often better than raw ratings)
    # ---------------------------------------------------------------------
    # Possession proxy: average of both teams' pace.
    if _has(
        [
            "PACE_PER40_SEASON_BEFORE_AVG_TEAM_HOME",
            "PACE_PER40_SEASON_BEFORE_AVG_TEAM_AWAY",
        ]
    ):
        exp_poss = (
            out["PACE_PER40_SEASON_BEFORE_AVG_TEAM_HOME"]
            + out["PACE_PER40_SEASON_BEFORE_AVG_TEAM_AWAY"]
        ) / 2.0
        _add("EXPECTED_POSS_FROM_PACE", exp_poss)

        # Expected points proxy: possessions * offensive rating.
        if _has(["OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"]):
            _add(
                "EXPECTED_PTS_HOME_FROM_OFFR_PACE",
                exp_poss * (out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"] / 100.0),
            )
        if _has(["OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"]):
            _add(
                "EXPECTED_PTS_AWAY_FROM_OFFR_PACE",
                exp_poss * (out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"] / 100.0),
            )

    # Off-Def mismatch features (home offense vs away defense, and vice versa)
    if _has(
        [
            "OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME",
            "DEF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
        ]
    ):
        _add(
            "OFFDEF_MISMATCH_HOME_OFF_MINUS_AWAY_DEF",
            out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"]
            - out["DEF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"],
        )
    if _has(
        [
            "OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY",
            "DEF_RATING_SEASON_BEFORE_AVG_TEAM_HOME",
        ]
    ):
        _add(
            "OFFDEF_MISMATCH_AWAY_OFF_MINUS_HOME_DEF",
            out["OFF_RATING_SEASON_BEFORE_AVG_TEAM_AWAY"]
            - out["DEF_RATING_SEASON_BEFORE_AVG_TEAM_HOME"],
        )

    # ---------------------------------------------------------------------
    # 3) Form and volatility: standardized recent scoring vs season baseline
    # ---------------------------------------------------------------------
    # z = (recent_avg - season_avg) / season_std
    if _has(
        [
            "PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME",
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME",
            "PTS_SEASON_BEFORE_STD_TEAM_HOME",
        ]
    ):
        z_home = _safe_div(
            out["PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"]
            - out["PTS_SEASON_BEFORE_AVG_TEAM_HOME"],
            out["PTS_SEASON_BEFORE_STD_TEAM_HOME"],
        )
        _add("PTS_FORM_Z_HOME_LAST5_VS_SEASON", z_home)

    if _has(
        [
            "PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY",
            "PTS_SEASON_BEFORE_AVG_TEAM_AWAY",
            "PTS_SEASON_BEFORE_STD_TEAM_AWAY",
        ]
    ):
        z_away = _safe_div(
            out["PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY"]
            - out["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"],
            out["PTS_SEASON_BEFORE_STD_TEAM_AWAY"],
        )
        _add("PTS_FORM_Z_AWAY_LAST5_VS_SEASON", z_away)

    # Trend slopes already exist for the previous five games. The home/away
    # suffix is appended to the complete source name during merge_home_away_data.
    trend_home_col = "PTS_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_HOME"
    trend_away_col = "PTS_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_AWAY"
    if _has([trend_home_col, trend_away_col]):
        _add(
            "PTS_TREND_SLOPE_DIFF_HOME_MINUS_AWAY",
            out[trend_home_col] - out[trend_away_col],
        )
        _add(
            "PTS_TREND_SLOPE_SUM_HOME_PLUS_AWAY",
            out[trend_home_col] + out[trend_away_col],
        )

    # ---------------------------------------------------------------------
    # 4) Fatigue and travel compression
    # ---------------------------------------------------------------------

    # Travel intensity: concentration of travel in last 2 vs 14 days
    if _has(
        ["TOTAL_KM_IN_LAST_2_DAYS_HOME_TEAM", "TOTAL_KM_IN_LAST_14_DAYS_HOME_TEAM"]
    ):
        ratio = _safe_div(
            out["TOTAL_KM_IN_LAST_2_DAYS_HOME_TEAM"],
            out["TOTAL_KM_IN_LAST_14_DAYS_HOME_TEAM"],
        )
        _add("TRAVEL_RECENCY_RATIO_HOME_2D_OVER_14D", ratio)

    if _has(
        ["TOTAL_KM_IN_LAST_2_DAYS_AWAY_TEAM", "TOTAL_KM_IN_LAST_14_DAYS_AWAY_TEAM"]
    ):
        ratio = _safe_div(
            out["TOTAL_KM_IN_LAST_2_DAYS_AWAY_TEAM"],
            out["TOTAL_KM_IN_LAST_14_DAYS_AWAY_TEAM"],
        )
        _add("TRAVEL_RECENCY_RATIO_AWAY_2D_OVER_14D", ratio)

    # Rest mismatch
    if _has(["REST_DAYS_BEFORE_MATCH_TEAM_HOME", "REST_DAYS_BEFORE_MATCH_TEAM_AWAY"]):
        _add(
            "REST_DAYS_DIFF_HOME_MINUS_AWAY",
            out["REST_DAYS_BEFORE_MATCH_TEAM_HOME"]
            - out["REST_DAYS_BEFORE_MATCH_TEAM_AWAY"],
        )

    # ---------------------------------------------------------------------
    # 5) Injury impact compression (numeric-only)
    # ---------------------------------------------------------------------
    if _has(
        ["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME", "PTS_SEASON_BEFORE_AVG_TEAM_HOME"]
    ):
        _add(
            "INJURY_PTS_SHARE_HOME",
            _safe_div(
                out["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME"],
                out["PTS_SEASON_BEFORE_AVG_TEAM_HOME"],
            ),
        )

    if _has(
        ["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY", "PTS_SEASON_BEFORE_AVG_TEAM_AWAY"]
    ):
        _add(
            "INJURY_PTS_SHARE_AWAY",
            _safe_div(
                out["TOTAL_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY"],
                out["PTS_SEASON_BEFORE_AVG_TEAM_AWAY"],
            ),
        )

    if _has(
        ["STAR_PTS_PERCENTAGE_BEFORE_TEAM_HOME", "STAR_PTS_PERCENTAGE_BEFORE_TEAM_AWAY"]
    ):
        _add(
            "STAR_PTS_PCT_DIFF_HOME_MINUS_AWAY",
            out["STAR_PTS_PERCENTAGE_BEFORE_TEAM_HOME"]
            - out["STAR_PTS_PERCENTAGE_BEFORE_TEAM_AWAY"],
        )

    # ---------------------------------------------------------------------
    # 6) Efficiency decomposition
    # ---------------------------------------------------------------------
    if _has(["POSS_SEASON_BEFORE_AVG_TEAM_HOME", "TS_PCT_SEASON_BEFORE_AVG_TEAM_HOME"]):
        _add(
            "POSS_X_TSPCT_HOME",
            out["POSS_SEASON_BEFORE_AVG_TEAM_HOME"]
            * out["TS_PCT_SEASON_BEFORE_AVG_TEAM_HOME"],
        )
    if _has(["POSS_SEASON_BEFORE_AVG_TEAM_AWAY", "TS_PCT_SEASON_BEFORE_AVG_TEAM_AWAY"]):
        _add(
            "POSS_X_TSPCT_AWAY",
            out["POSS_SEASON_BEFORE_AVG_TEAM_AWAY"]
            * out["TS_PCT_SEASON_BEFORE_AVG_TEAM_AWAY"],
        )

    # ---------------------------------------------------------------------
    # Final: add all new columns at once to avoid fragmentation
    # ---------------------------------------------------------------------
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=out.index)
        out = pd.concat([out, new_df], axis=1)

    # Replace inf with nan, keep NaNs (XGBoost can handle missing)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out
