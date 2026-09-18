"""``compute_referee_features`` before per-date memoisation, verbatim.

Reference for tests/test_referee_speedups.py only.
"""

import numpy as np
import pandas as pd
from nba_ou.data_processing.referees.add_referee_features import (
    REFEREE_METRICS,
    _normalize_referee_slots,
)
from tqdm import tqdm


def compute_referee_features_before(
    df_refs_pivot: pd.DataFrame, include_ref_trio_features: bool = False
):
    """
    Compute aggregate referee features for each game based on historical performance.

    For the current game's three referees, referee-position is ignored:
    1. For each referee, compute the metric delta:
       mean(metric in games with referee) - mean(metric in games without referee)
    2. Aggregate those per-referee deltas into:
       - mean across current referees
       - standard deviation across current referees
    3. Optionally compute order-invariant trio features:
       - trio delta: mean(metric in games with exact trio) - mean(metric in games without trio)
       - trio std: standard deviation of metric in games with exact trio

    Args:
        df_refs_pivot (pd.DataFrame): DataFrame with columns:
            - GAME_ID: Unique game identifier
            - GAME_DATE: Date of the game
            - SEASON_YEAR: Year of the season
            - TOTAL_POINTS: Total points scored in the game
            - TOTAL_LINE_<main_book>: Main over/under line for the game
            - PF: Personal fouls called in the game
            - REF_1, REF_2, REF_3: Names of the three referees
            - DIFF_FROM_LINE: TOTAL_POINTS - TOTAL_LINE_<main_book>
        include_ref_trio_features (bool): Whether to compute exact-referee-trio
            features. Defaults to False.

    Returns:
        pd.DataFrame: Original DataFrame with additional columns:
            - REF_AVG_<METRIC>_DIFF_BEFORE
            - REF_STD_<METRIC>_DIFF_BEFORE
            - REF_TRIO_<METRIC>_DIFF_BEFORE
            - REF_TRIO_<METRIC>_STD_BEFORE
              (only when include_ref_trio_features=True)
    """

    def _extract_unique_refs(row):
        return (
            _normalize_referee_slots([row["REF_1"], row["REF_2"], row["REF_3"]])
            .dropna()
            .tolist()
        )

    # Ensure GAME_DATE is datetime
    df = df_refs_pivot.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Sort by GAME_DATE to ensure chronological order
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    if include_ref_trio_features:
        # Cache order-invariant trio key per game for faster matching
        df["REF_TRIO_KEY"] = df.apply(
            lambda row: frozenset(_extract_unique_refs(row)), axis=1
        )

    # Initialize aggregate referee features (and trio features when requested)
    for metric in REFEREE_METRICS:
        df[f"REF_AVG_{metric}_DIFF_BEFORE"] = np.nan
        df[f"REF_STD_{metric}_DIFF_BEFORE"] = np.nan
        df[f"REF_SUM_{metric}_DIFF_BEFORE"] = np.nan
        if include_ref_trio_features:
            df[f"REF_TRIO_{metric}_DIFF_BEFORE"] = np.nan
            df[f"REF_TRIO_{metric}_STD_BEFORE"] = np.nan

    def _split_past_games_by_season(
        current_date: pd.Timestamp, current_season: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        same_season = df[
            (df["GAME_DATE"] < current_date) & (df["SEASON_YEAR"] == current_season)
        ].copy()
        prev_season = df[df["SEASON_YEAR"] == current_season - 1].copy()
        return same_season, prev_season

    def _select_games_for_ref(
        past_same_season: pd.DataFrame,
        past_prev_season: pd.DataFrame,
        ref_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        for past_games in (past_same_season, past_prev_season):
            if past_games.empty:
                continue
            ref_participated = (
                (past_games["REF_1"] == ref_name)
                | (past_games["REF_2"] == ref_name)
                | (past_games["REF_3"] == ref_name)
            )
            games_with_ref = past_games[ref_participated]
            games_without_ref = past_games[~ref_participated]
            if len(games_with_ref) > 0 and len(games_without_ref) > 0:
                return games_with_ref, games_without_ref
        return pd.DataFrame(), pd.DataFrame()

    def _select_games_for_trio(
        past_games: pd.DataFrame,
        trio_key: frozenset,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if past_games.empty:
            return pd.DataFrame(), pd.DataFrame()
        trio_mask = past_games["REF_TRIO_KEY"] == trio_key
        games_with_trio = past_games[trio_mask]
        games_without_trio = past_games[~trio_mask]
        if len(games_with_trio) > 0:
            return games_with_trio, games_without_trio
        return pd.DataFrame(), pd.DataFrame()

    # Process each game
    for idx in tqdm(range(len(df)), desc="Computing referee features"):
        current_game = df.iloc[idx]
        current_date = current_game["GAME_DATE"]
        current_season = current_game["SEASON_YEAR"]

        past_games_same_season, past_games_prev_season = _split_past_games_by_season(
            current_date, current_season
        )
        past_games = (
            past_games_same_season
            if not past_games_same_season.empty
            else past_games_prev_season
        )

        # Skip if no past games available
        if past_games.empty:
            continue

        current_refs = _extract_unique_refs(current_game)

        # Compute per-ref deltas and aggregate them into mean/std (position-agnostic)
        for metric in REFEREE_METRICS:
            per_ref_diffs = []
            for ref_name in current_refs:
                games_with_ref, games_without_ref = _select_games_for_ref(
                    past_games_same_season, past_games_prev_season, ref_name
                )

                if len(games_with_ref) > 0 and len(games_without_ref) > 0:
                    per_ref_diffs.append(
                        games_with_ref[metric].mean() - games_without_ref[metric].mean()
                    )

            if per_ref_diffs:
                per_ref_diffs_series = pd.Series(per_ref_diffs, dtype="float64")
                df.at[idx, f"REF_AVG_{metric}_DIFF_BEFORE"] = (
                    per_ref_diffs_series.mean()
                )
                df.at[idx, f"REF_STD_{metric}_DIFF_BEFORE"] = per_ref_diffs_series.std(
                    ddof=0
                )
                df.at[idx, f"REF_SUM_{metric}_DIFF_BEFORE"] = per_ref_diffs_series.sum()

        # Process referee trio (all three referees together, regardless of order)
        # Use a wider lookback (current + 1 prior seasons) since exact trios are rare
        if include_ref_trio_features and len(current_refs) == 3:
            current_trio_key = frozenset(current_refs)
            trio_past_games = df[
                (df["GAME_DATE"] < current_date)
                & (df["SEASON_YEAR"].isin({current_season, current_season - 1}))
            ]
            games_with_trio, games_without_trio = _select_games_for_trio(
                trio_past_games, current_trio_key
            )

            for metric in REFEREE_METRICS:
                if len(games_with_trio) > 0 and len(games_without_trio) > 0:
                    avg_with_trio = games_with_trio[metric].mean()
                    avg_without_trio = games_without_trio[metric].mean()
                    df.at[idx, f"REF_TRIO_{metric}_DIFF_BEFORE"] = (
                        avg_with_trio - avg_without_trio
                    )
                if len(games_with_trio) > 0:
                    trio_std = games_with_trio[metric].std(ddof=0)
                    df.at[idx, f"REF_TRIO_{metric}_STD_BEFORE"] = trio_std

    if include_ref_trio_features:
        df = df.drop(columns=["REF_TRIO_KEY"])
    return df
