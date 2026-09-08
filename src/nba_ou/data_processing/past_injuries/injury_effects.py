from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from nba_ou.config.market_columns import HOME_MARGIN_COL
from nba_ou.config.odds_columns import (
    get_main_book,
    spread_line_home_col,
    total_line_col,
)
from scipy.stats import fisher_exact, ttest_ind
from tqdm import tqdm

MIN_SIGNIFICANCE_GAMES = 3
EFFECT_METRICS = (
    "TOTAL_POINTS",
    "DIFF_FROM_LINE",
    "SPREAD_ERROR",
    "WIN_RATE_DIFF",
)


def _ensure_datetime(df: pd.DataFrame, col: str = "GAME_DATE") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _build_injured_games_index(
    injury_dict: dict[str, dict[str, list[Any]]],
) -> dict[int, dict[int, set]]:
    """
    injury_dict: {GAME_ID: {TEAM_ID: [player_ids...]}}
    returns: {TEAM_ID: {PLAYER_ID: set([GAME_ID, ...])}}
    """
    idx: dict[int, dict[int, set]] = {}
    for game_id_str, team_map in injury_dict.items():
        try:
            game_id = int(game_id_str)
        except Exception:
            # If your GAME_IDs are not int-like, you can keep them as strings.
            # But ensure df GAME_ID matching uses same type.
            continue

        for team_id_str, players in team_map.items():
            try:
                team_id = int(team_id_str)
            except Exception:
                continue

            bucket = idx.setdefault(team_id, {})
            for pid in players:
                if pd.isna(pid) or pid in (0, "0", ""):
                    continue
                try:
                    player_id = int(pid)
                except Exception:
                    continue
                bucket.setdefault(player_id, set()).add(game_id)
    return idx


def _build_availability_games_index(
    availability_dict: dict[str, dict[str, dict[str, list[Any]]]],
) -> dict[int, dict[int, dict[str, set[int]]]]:
    """Invert the local roster split without altering the injury-report data."""
    idx: dict[int, dict[int, dict[str, set[int]]]] = {}
    for game_id_value, team_map in availability_dict.items():
        try:
            game_id = int(game_id_value)
        except (TypeError, ValueError):
            continue

        for team_id_value, status_map in team_map.items():
            try:
                team_id = int(team_id_value)
            except (TypeError, ValueError):
                continue

            team_bucket = idx.setdefault(team_id, {})
            for status in ("available", "injured"):
                for player_id_value in status_map.get(status, []):
                    try:
                        player_id = int(player_id_value)
                    except (TypeError, ValueError):
                        continue
                    player_bucket = team_bucket.setdefault(
                        player_id, {"available": set(), "injured": set()}
                    )
                    player_bucket[status].add(game_id)
    return idx


def _infer_last_two_season_years_for_row(season_year: int) -> tuple[int, int]:
    return (season_year - 1, season_year)


def _get_recent_history_df(
    df_hist: pd.DataFrame,
    *,
    team_id: int,
    season_years: tuple[int, int],
    before_date: pd.Timestamp,
) -> pd.DataFrame:
    y1, y2 = season_years
    mask = (
        (df_hist["TEAM_ID"].astype("int64") == int(team_id))
        & (df_hist["SEASON_YEAR"].astype("int64").isin([y1, y2]))
        & (df_hist["GAME_DATE"] < before_date)
    )
    return df_hist.loc[mask]


def _continuous_effect_and_pvalue(
    present_values: pd.Series,
    injured_values: pd.Series,
    *,
    min_significance_games: int = MIN_SIGNIFICANCE_GAMES,
) -> tuple[float, float, int, int]:
    present = pd.to_numeric(present_values, errors="coerce").dropna().to_numpy()
    injured = pd.to_numeric(injured_values, errors="coerce").dropna().to_numpy()
    n_injured = len(injured)
    n_present = len(present)
    if len(present) == 0 or len(injured) == 0:
        return np.nan, np.nan, n_injured, n_present

    effect = float(present.mean() - injured.mean())
    if len(present) < min_significance_games or len(injured) < min_significance_games:
        return effect, np.nan, n_injured, n_present

    present_var = float(np.var(present))
    injured_var = float(np.var(injured))
    if present_var == 0.0 and injured_var == 0.0:
        return effect, 1.0 if effect == 0.0 else 0.0, n_injured, n_present

    pvalue = float(ttest_ind(present, injured, equal_var=False).pvalue)
    return (
        effect,
        pvalue if np.isfinite(pvalue) else np.nan,
        n_injured,
        n_present,
    )


def _win_rate_effect_and_pvalue(
    present_values: pd.Series,
    injured_values: pd.Series,
    *,
    min_significance_games: int = MIN_SIGNIFICANCE_GAMES,
) -> tuple[float, float, int, int]:
    present = pd.to_numeric(present_values, errors="coerce").dropna().to_numpy()
    injured = pd.to_numeric(injured_values, errors="coerce").dropna().to_numpy()
    n_injured = len(injured)
    n_present = len(present)
    if len(present) == 0 or len(injured) == 0:
        return np.nan, np.nan, n_injured, n_present

    effect = float(present.mean() - injured.mean())
    if len(present) < min_significance_games or len(injured) < min_significance_games:
        return effect, np.nan, n_injured, n_present

    table = np.array(
        [
            [int(present.sum()), int(len(present) - present.sum())],
            [int(injured.sum()), int(len(injured) - injured.sum())],
        ]
    )
    pvalue = float(fisher_exact(table).pvalue)
    return (
        effect,
        pvalue if np.isfinite(pvalue) else np.nan,
        n_injured,
        n_present,
    )


def _empty_effect_result(n_inj=0, n_present=0, n_total=0):
    return (
        (np.nan,) * 4,
        (np.nan,) * 4,
        (0,) * 4,
        (0,) * 4,
        int(n_inj),
        int(n_present),
        int(n_total),
    )


def _compute_player_availability_effect(
    df_team_hist: pd.DataFrame,
    injured_games_for_player: set,
    available_games_for_player: set | None = None,
):
    """
    Return four present-minus-injured effects, their p-values, and sample sizes.

    ``available_games_for_player`` makes membership explicit, so games before a
    player joined or after they left the team are excluded. ``None`` preserves
    the legacy complement behaviour for standalone callers without that map.
    """
    if df_team_hist.empty:
        return _empty_effect_result()

    game_ids = pd.to_numeric(df_team_hist["GAME_ID"], errors="coerce").astype("Int64")
    df_team_hist = df_team_hist.assign(_GAME_ID_INT=game_ids)

    inj_mask = df_team_hist["_GAME_ID_INT"].isin(list(injured_games_for_player))
    if available_games_for_player is None:
        present_mask = ~inj_mask
    else:
        present_mask = (
            df_team_hist["_GAME_ID_INT"].isin(list(available_games_for_player))
            & ~inj_mask
        )
    df_inj = df_team_hist.loc[inj_mask]
    df_present = df_team_hist.loc[present_mask]
    n_inj = int(len(df_inj))
    n_present = int(len(df_present))
    n_total = n_inj + n_present

    if n_inj == 0 or n_present == 0:
        return _empty_effect_result(n_inj, n_present, n_total)

    total_effect, total_pvalue, total_n_inj, total_n_present = (
        _continuous_effect_and_pvalue(
            df_present["TOTAL_POINTS"], df_inj["TOTAL_POINTS"]
        )
    )
    line_effect, line_pvalue, line_n_inj, line_n_present = (
        _continuous_effect_and_pvalue(
            df_present["DIFF_FROM_LINE"], df_inj["DIFF_FROM_LINE"]
        )
    )
    spread_effect, spread_pvalue, spread_n_inj, spread_n_present = (
        _continuous_effect_and_pvalue(
            df_present["SPREAD_ERROR"], df_inj["SPREAD_ERROR"]
        )
    )
    win_effect, win_pvalue, win_n_inj, win_n_present = _win_rate_effect_and_pvalue(
        df_present["WIN"], df_inj["WIN"]
    )

    return (
        (total_effect, line_effect, spread_effect, win_effect),
        (total_pvalue, line_pvalue, spread_pvalue, win_pvalue),
        (total_n_inj, line_n_inj, spread_n_inj, win_n_inj),
        (total_n_present, line_n_present, spread_n_present, win_n_present),
        n_inj,
        n_present,
        n_total,
    )


def _shrink_effect(
    raw_effect: float, n_inj_games: int, n_present_games: int, k: float
) -> float:
    """
    Empirical-Bayes style shrinkage toward zero.
    """
    if pd.isna(raw_effect):
        return np.nan
    n_eff = min(int(n_inj_games), int(n_present_games))
    if n_eff <= 0:
        return np.nan
    if k <= 0:
        return float(raw_effect)
    return float(raw_effect * (n_eff / (n_eff + k)))


def add_top3_availability_effect_features_for_columns(
    df_games: pd.DataFrame,
    injured_dict: dict[str, dict[str, list[Any]]],
    *,
    availability_dict: dict[str, dict[str, dict[str, list[Any]]]] | None = None,
    home_team_id_col: str = "TEAM_ID_TEAM_HOME",
    away_team_id_col: str = "TEAM_ID_TEAM_AWAY",
    game_date_col: str = "GAME_DATE",
    season_year_col: str = "SEASON_YEAR",
    game_id_col: str = "GAME_ID",
    total_points_col: str = "TOTAL_POINTS",
    diff_from_line_col: str = "DIFF_FROM_LINE",
    home_margin_col: str = HOME_MARGIN_COL,
    total_line_book: str | None = None,
    spread_line_book: str | None = None,
    home_player_cols: tuple[str, ...],
    away_player_cols: tuple[str, ...],
    out_prefix: str,
    shrinkage_k: float = 10.0,
    include_per_player_columns: bool = False,
    include_detailed_sample_size_features: bool = False,
) -> pd.DataFrame:
    """
    Compute player availability impact features from past games only (< current game date).

    Changes vs previous version:
      - DIFF_FROM_LINE is always computed against TOTAL_LINE_<main book> selected by
        total_line_book (or configured main book from config).
      - Effects are shrunk toward zero with:
        eff_shrunk = eff_raw * n_eff/(n_eff + k), where n_eff=min(n_inj, n_present).
      - Adds team-oriented spread-error and win-rate effects.
      - Welch p-values are computed for continuous effects and Fisher exact
        p-values for win rate. Both groups require at least three games.
      - A separate availability map limits history to games where the player was
        actually classified on that team's roster. It does not mutate injuries.
      - Compact aggregate summaries include mean and max-abs effects plus total
        sample size. This is the default training schema.
      - Redundant diagnostic counts/flags can be restored with
        include_detailed_sample_size_features=True.
      - Per-player columns can be disabled via include_per_player_columns=False.
    """
    df = df_games.copy()
    df = _ensure_datetime(df, game_date_col)

    selected_total_line_col = total_line_col(total_line_book or get_main_book())
    if selected_total_line_col not in df.columns:
        raise ValueError(
            f"Missing required total line column {selected_total_line_col}. "
            "This function computes DIFF_FROM_LINE using the configured main book."
        )
    if total_points_col not in df.columns:
        raise ValueError(
            f"Missing required column {total_points_col}. "
            "Cannot compute DIFF_FROM_LINE history for injury effects."
        )

    selected_spread_line_col = spread_line_home_col(spread_line_book or get_main_book())
    if selected_spread_line_col not in df.columns:
        raise ValueError(
            f"Missing required spread line column {selected_spread_line_col}. "
            "Cannot compute historical spread availability effects."
        )
    if home_margin_col not in df.columns:
        raise ValueError(
            f"Missing required column {home_margin_col}. "
            "Cannot compute spread or win-rate availability effects."
        )

    # Always align DIFF_FROM_LINE computation to selected main total line.
    internal_diff_col = "__DIFF_FROM_MAIN_LINE_INTERNAL__"
    df[internal_diff_col] = pd.to_numeric(
        df[total_points_col], errors="coerce"
    ) - pd.to_numeric(df[selected_total_line_col], errors="coerce")
    internal_spread_col = "__SPREAD_ERROR_INTERNAL__"
    home_margin_values = pd.to_numeric(df[home_margin_col], errors="coerce")
    df[internal_spread_col] = home_margin_values - pd.to_numeric(
        df[selected_spread_line_col], errors="coerce"
    )
    internal_home_win_col = "__HOME_WIN_INTERNAL__"
    internal_away_win_col = "__AWAY_WIN_INTERNAL__"
    internal_team_win_col = "__TEAM_WIN_INTERNAL__"
    df[internal_home_win_col] = np.where(
        home_margin_values.notna(), (home_margin_values > 0).astype(float), np.nan
    )
    df[internal_away_win_col] = np.where(
        home_margin_values.notna(), (home_margin_values < 0).astype(float), np.nan
    )

    required = [
        home_team_id_col,
        away_team_id_col,
        game_date_col,
        season_year_col,
        game_id_col,
        total_points_col,
        selected_total_line_col,
        selected_spread_line_col,
        home_margin_col,
        *home_player_cols,
        *away_player_cols,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    injured_index = _build_injured_games_index(injured_dict)
    availability_index = (
        _build_availability_games_index(availability_dict)
        if availability_dict is not None
        else None
    )

    # Build TEAM-game history (two rows per game: one per team)
    hist_home = df[
        [
            home_team_id_col,
            game_date_col,
            season_year_col,
            total_points_col,
            internal_diff_col,
            internal_spread_col,
            internal_home_win_col,
            game_id_col,
        ]
    ].copy()
    hist_home = hist_home.rename(
        columns={
            home_team_id_col: "TEAM_ID",
            internal_home_win_col: internal_team_win_col,
        }
    )

    hist_away = df[
        [
            away_team_id_col,
            game_date_col,
            season_year_col,
            total_points_col,
            internal_diff_col,
            internal_spread_col,
            internal_away_win_col,
            game_id_col,
        ]
    ].copy()
    hist_away = hist_away.rename(
        columns={
            away_team_id_col: "TEAM_ID",
            internal_away_win_col: internal_team_win_col,
        }
    )
    hist_away[internal_spread_col] = -hist_away[internal_spread_col]

    df_hist = pd.concat([hist_home, hist_away], ignore_index=True)
    df_hist = df_hist.rename(
        columns={
            game_date_col: "GAME_DATE",
            season_year_col: "SEASON_YEAR",
            total_points_col: "TOTAL_POINTS",
            internal_diff_col: "DIFF_FROM_LINE",
            internal_spread_col: "SPREAD_ERROR",
            internal_team_win_col: "WIN",
            game_id_col: "GAME_ID",
        }
    )
    df_hist["GAME_DATE"] = pd.to_datetime(df_hist["GAME_DATE"], errors="coerce")
    df_hist["SEASON_YEAR"] = pd.to_numeric(
        df_hist["SEASON_YEAR"], errors="coerce"
    ).astype("Int64")

    @lru_cache(maxsize=250_000)
    def _cached_effect(
        team_id: int, season_year: int, date_ordinal: int, player_id: int
    ):
        season_years = _infer_last_two_season_years_for_row(season_year)
        before_date = pd.Timestamp.fromordinal(date_ordinal)
        df_team_hist = _get_recent_history_df(
            df_hist,
            team_id=team_id,
            season_years=season_years,
            before_date=before_date,
        )
        available_games_for_player = None
        if availability_index is None:
            injured_games_for_player = injured_index.get(team_id, {}).get(
                player_id, set()
            )
        else:
            status = availability_index.get(team_id, {}).get(
                player_id, {"available": set(), "injured": set()}
            )
            available_games_for_player = status["available"]
            injured_games_for_player = status["injured"]
        return _compute_player_availability_effect(
            df_team_hist,
            injured_games_for_player,
            available_games_for_player,
        )

    def _out_col(side: str, i: int, metric: str) -> str:
        return f"{out_prefix}_{side}_P{i}_{metric}"

    metric_output_names = {
        "TOTAL_POINTS": "TOTAL_POINTS",
        "DIFF_FROM_LINE": diff_from_line_col,
        "SPREAD_ERROR": "SPREAD_ERROR",
        "WIN_RATE_DIFF": "WIN_RATE_DIFF",
    }

    n = len(df)
    n_home_players = len(home_player_cols)
    n_away_players = len(away_player_cols)

    home_effects = {
        metric: np.full((n, n_home_players), np.nan, dtype="float64")
        for metric in EFFECT_METRICS
    }
    away_effects = {
        metric: np.full((n, n_away_players), np.nan, dtype="float64")
        for metric in EFFECT_METRICS
    }
    home_pvalues = {
        metric: np.full((n, n_home_players), np.nan, dtype="float64")
        for metric in EFFECT_METRICS
    }
    away_pvalues = {
        metric: np.full((n, n_away_players), np.nan, dtype="float64")
        for metric in EFFECT_METRICS
    }
    home_n_inj = np.zeros((n, n_home_players), dtype="float64")
    home_n_present = np.zeros((n, n_home_players), dtype="float64")
    home_n_total = np.zeros((n, n_home_players), dtype="float64")
    away_n_inj = np.zeros((n, n_away_players), dtype="float64")
    away_n_present = np.zeros((n, n_away_players), dtype="float64")
    away_n_total = np.zeros((n, n_away_players), dtype="float64")

    # itertuples is faster than apply for 25k rows
    for i, row in enumerate(
        tqdm(
            df.itertuples(index=False),
            total=len(df),
            desc="Computing availability effects",
        )
    ):
        date = getattr(row, game_date_col)
        season_year = getattr(row, season_year_col)
        home_team = getattr(row, home_team_id_col)
        away_team = getattr(row, away_team_id_col)

        if (
            pd.isna(date)
            or pd.isna(season_year)
            or pd.isna(home_team)
            or pd.isna(away_team)
        ):
            continue

        date_ts = pd.Timestamp(date)
        date_ord = date_ts.toordinal()

        try:
            season_year_int = int(season_year)
            home_team_int = int(home_team)
            away_team_int = int(away_team)
        except Exception:
            continue

        seen_home_pids: set[int] = set()
        for j, col in enumerate(home_player_cols):
            pid = getattr(row, col)
            if pd.isna(pid) or pid in (0, "0"):
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            if pid_int in seen_home_pids:
                continue
            seen_home_pids.add(pid_int)
            (
                raw_effects,
                pvalues,
                metric_injured_counts,
                metric_present_counts,
                n_inj,
                n_present,
                n_total,
            ) = _cached_effect(home_team_int, season_year_int, date_ord, pid_int)
            for metric, raw_effect, pvalue, injured_count, present_count in zip(
                EFFECT_METRICS,
                raw_effects,
                pvalues,
                metric_injured_counts,
                metric_present_counts,
                strict=True,
            ):
                home_effects[metric][i, j] = _shrink_effect(
                    raw_effect, injured_count, present_count, shrinkage_k
                )
                home_pvalues[metric][i, j] = pvalue
            home_n_inj[i, j] = n_inj
            home_n_present[i, j] = n_present
            home_n_total[i, j] = n_total

        seen_away_pids: set[int] = set()
        for j, col in enumerate(away_player_cols):
            pid = getattr(row, col)
            if pd.isna(pid) or pid in (0, "0"):
                continue
            try:
                pid_int = int(pid)
            except Exception:
                continue
            if pid_int in seen_away_pids:
                continue
            seen_away_pids.add(pid_int)
            (
                raw_effects,
                pvalues,
                metric_injured_counts,
                metric_present_counts,
                n_inj,
                n_present,
                n_total,
            ) = _cached_effect(away_team_int, season_year_int, date_ord, pid_int)
            for metric, raw_effect, pvalue, injured_count, present_count in zip(
                EFFECT_METRICS,
                raw_effects,
                pvalues,
                metric_injured_counts,
                metric_present_counts,
                strict=True,
            ):
                away_effects[metric][i, j] = _shrink_effect(
                    raw_effect, injured_count, present_count, shrinkage_k
                )
                away_pvalues[metric][i, j] = pvalue
            away_n_inj[i, j] = n_inj
            away_n_present[i, j] = n_present
            away_n_total[i, j] = n_total

    if include_per_player_columns:
        for j in range(n_home_players):
            for metric in EFFECT_METRICS:
                output_metric = metric_output_names[metric]
                df[_out_col("HOME", j + 1, output_metric)] = home_effects[metric][:, j]
                df[_out_col("HOME", j + 1, f"PVALUE_{output_metric}")] = home_pvalues[
                    metric
                ][:, j]
            df[_out_col("HOME", j + 1, "N_INJ_GAMES")] = home_n_inj[:, j]
            df[_out_col("HOME", j + 1, "N_PRESENT_GAMES")] = home_n_present[:, j]
            df[_out_col("HOME", j + 1, "N_TOTAL_GAMES")] = home_n_total[:, j]

        for j in range(n_away_players):
            for metric in EFFECT_METRICS:
                output_metric = metric_output_names[metric]
                df[_out_col("AWAY", j + 1, output_metric)] = away_effects[metric][:, j]
                df[_out_col("AWAY", j + 1, f"PVALUE_{output_metric}")] = away_pvalues[
                    metric
                ][:, j]
            df[_out_col("AWAY", j + 1, "N_INJ_GAMES")] = away_n_inj[:, j]
            df[_out_col("AWAY", j + 1, "N_PRESENT_GAMES")] = away_n_present[:, j]
            df[_out_col("AWAY", j + 1, "N_TOTAL_GAMES")] = away_n_total[:, j]

    def _nanmean_axis1(arr: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(arr)
        counts = mask.sum(axis=1)
        sums = np.nansum(arr, axis=1)
        out = np.full(arr.shape[0], np.nan, dtype="float64")
        valid = counts > 0
        out[valid] = sums[valid] / counts[valid]
        return out

    def _nanmaxabs_axis1(arr: np.ndarray) -> np.ndarray:
        abs_arr = np.abs(arr)
        out = np.full(arr.shape[0], np.nan, dtype="float64")
        valid = ~np.isnan(abs_arr).all(axis=1)
        if valid.any():
            out[valid] = np.nanmax(abs_arr[valid], axis=1)
        return out

    def _bonferroni_pvalue_axis1(arr: np.ndarray) -> np.ndarray:
        """Correct the smallest valid per-player p-value within each row."""
        mask = ~np.isnan(arr)
        n_tests = mask.sum(axis=1)
        out = np.full(arr.shape[0], np.nan, dtype="float64")
        valid = n_tests > 0
        if valid.any():
            out[valid] = np.minimum(1.0, np.nanmin(arr[valid], axis=1) * n_tests[valid])
        return out

    for side, effects, pvalues in (
        ("HOME", home_effects, home_pvalues),
        ("AWAY", away_effects, away_pvalues),
    ):
        for metric in EFFECT_METRICS:
            output_metric = metric_output_names[metric]
            df[f"{out_prefix}_{side}_MEAN_{output_metric}"] = _nanmean_axis1(
                effects[metric]
            )
            df[f"{out_prefix}_{side}_BONFERRONI_PVALUE_{output_metric}"] = (
                _bonferroni_pvalue_axis1(pvalues[metric])
            )

    for side, effects in (("HOME", home_effects), ("AWAY", away_effects)):
        for metric in ("TOTAL_POINTS", "DIFF_FROM_LINE"):
            output_metric = metric_output_names[metric]
            df[f"{out_prefix}_{side}_MAX_ABS_{output_metric}"] = _nanmaxabs_axis1(
                effects[metric]
            )

    # No evidence for any of the players means the fully shrunk estimate, which
    # for an estimator that shrinks toward zero is zero -- `_shrink_effect`
    # computes raw * n/(n+k), and that is 0 at n=0. Leaving NaN made the feature
    # discontinuous at exactly the point the shrinkage was designed to handle
    # smoothly: one weak game gives ~0, and no games gave "unknown".
    #
    # Applied to the aggregates only, never to the per-player values, so rows
    # where at least one player has evidence keep the mean over the players who
    # actually have it rather than being pulled toward zero by the others.
    aggregate_columns = [
        f"{out_prefix}_{side}_{statistic}"
        for side in ("HOME", "AWAY")
        for statistic in (
            "MEAN_TOTAL_POINTS",
            f"MEAN_{diff_from_line_col}",
            "MEAN_SPREAD_ERROR",
            "MEAN_WIN_RATE_DIFF",
            "MAX_ABS_TOTAL_POINTS",
            f"MAX_ABS_{diff_from_line_col}",
        )
    ]
    for column in aggregate_columns:
        df[column] = df[column].fillna(0.0)
    df[f"{out_prefix}_HOME_SUM_N_TOTAL_GAMES"] = home_n_total.sum(axis=1)
    df[f"{out_prefix}_AWAY_SUM_N_TOTAL_GAMES"] = away_n_total.sum(axis=1)

    if include_detailed_sample_size_features:
        df[f"{out_prefix}_HOME_SUM_N_INJ_GAMES"] = home_n_inj.sum(axis=1)
        df[f"{out_prefix}_AWAY_SUM_N_INJ_GAMES"] = away_n_inj.sum(axis=1)
        df[f"{out_prefix}_HOME_SUM_N_PRESENT_GAMES"] = home_n_present.sum(axis=1)
        df[f"{out_prefix}_AWAY_SUM_N_PRESENT_GAMES"] = away_n_present.sum(axis=1)
        df[f"{out_prefix}_HOME_N_PLAYERS_WITH_EFFECT"] = (
            (home_n_inj > 0) & (home_n_present > 0)
        ).sum(axis=1)
        df[f"{out_prefix}_AWAY_N_PLAYERS_WITH_EFFECT"] = (
            (away_n_inj > 0) & (away_n_present > 0)
        ).sum(axis=1)
        df[f"{out_prefix}_HOME_HAS_PLAYER_EFFECT"] = (
            df[f"{out_prefix}_HOME_N_PLAYERS_WITH_EFFECT"] > 0
        ).astype(int)
        df[f"{out_prefix}_AWAY_HAS_PLAYER_EFFECT"] = (
            df[f"{out_prefix}_AWAY_N_PLAYERS_WITH_EFFECT"] > 0
        ).astype(int)

    # Cleanup internal helper columns.
    df = df.drop(
        columns=[
            internal_diff_col,
            internal_spread_col,
            internal_home_win_col,
            internal_away_win_col,
        ],
        errors="ignore",
    )

    return df
