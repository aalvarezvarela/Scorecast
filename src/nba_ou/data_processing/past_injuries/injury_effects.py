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
from tqdm import tqdm

#: Minimum games per group before a standard error is reported. The mean
#: difference itself needs one game per side; a spread around it needs two.
MIN_SE_GAMES = 2

#: Prior weight used by the fallback ``n_eff / (n_eff + k)`` shrinkage, applied
#: only to rows too early in the history to fit anything. It is a guess, which
#: is precisely why it is no longer the primary estimator.
DEFAULT_SHRINKAGE_K = 10.0

#: Player-effect observations required before the empirical-Bayes variances are
#: trusted. Below this the fallback prior is used instead.
MIN_SHRINKAGE_FIT_SAMPLES = 200

#: Effects computed per player. ``WIN_RATE_DIFF`` was dropped: a binary outcome
#: over a handful of games is the noisiest of the four and measures nearly what
#: ``SPREAD_ERROR`` measures on a continuous scale.
EFFECT_METRICS = (
    "TOTAL_POINTS",
    "DIFF_FROM_LINE",
    "SPREAD_ERROR",
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


def _continuous_effect_and_se(
    present_values: pd.Series,
    injured_values: pd.Series,
    *,
    min_se_games: int = MIN_SE_GAMES,
) -> tuple[float, float, int, int]:
    """Present-minus-injured mean difference and its Welch standard error.

    The standard error replaces the p-value this function used to return. A
    p-value is a monotone function of the effect size and the two group sizes,
    all of which are already emitted as features, so it carried almost no
    information the model could not reconstruct -- while implying a
    significance test that a handful of games cannot support, over a family of
    per-player, per-metric comparisons that invites false positives. The
    standard error keeps effect magnitude and estimate precision on separate
    axes and leaves the trust rule to the model.
    """
    present = pd.to_numeric(present_values, errors="coerce").dropna().to_numpy()
    injured = pd.to_numeric(injured_values, errors="coerce").dropna().to_numpy()
    n_injured = len(injured)
    n_present = len(present)
    if n_present == 0 or n_injured == 0:
        return np.nan, np.nan, n_injured, n_present

    effect = float(present.mean() - injured.mean())
    if n_present < min_se_games or n_injured < min_se_games:
        return effect, np.nan, n_injured, n_present

    # Welch: no equal-variance assumption, matching the test this replaces.
    standard_error = float(
        np.sqrt(
            np.var(present, ddof=1) / n_present + np.var(injured, ddof=1) / n_injured
        )
    )
    return (
        effect,
        standard_error if np.isfinite(standard_error) else np.nan,
        n_injured,
        n_present,
    )


def _empty_effect_result(n_inj=0, n_present=0, n_total=0):
    n_metrics = len(EFFECT_METRICS)
    return (
        (np.nan,) * n_metrics,
        (np.nan,) * n_metrics,
        (0,) * n_metrics,
        (0,) * n_metrics,
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
    Return the present-minus-injured effects, their standard errors, and sizes.

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

    effects, standard_errors, injured_counts, present_counts = [], [], [], []
    for metric in EFFECT_METRICS:
        effect, standard_error, metric_n_inj, metric_n_present = (
            _continuous_effect_and_se(df_present[metric], df_inj[metric])
        )
        effects.append(effect)
        standard_errors.append(standard_error)
        injured_counts.append(metric_n_inj)
        present_counts.append(metric_n_present)

    return (
        tuple(effects),
        tuple(standard_errors),
        tuple(injured_counts),
        tuple(present_counts),
        n_inj,
        n_present,
        n_total,
    )


def _expanding_empirical_bayes_weights(
    date_ordinals: np.ndarray,
    raw: np.ndarray,
    standard_errors: np.ndarray,
    inverse_n: np.ndarray,
    *,
    min_samples: int = MIN_SHRINKAGE_FIT_SAMPLES,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-observation shrinkage weights ``tau2 / (tau2 + se2)``, fitted on history.

    This replaces a hand-picked ``k``. The observed spread of player effects is
    the sum of real player-to-player variation and sampling noise:

        var(effect) = tau2 + mean(se2)

    so ``tau2 = var(effect) - mean(se2)`` is the signal variance, and the
    posterior weight on a single player's estimate is ``tau2 / (tau2 + se2)``.
    That is the same shape as ``n/(n+k)`` -- since ``se2`` falls like ``1/n`` --
    but with the constant measured rather than assumed, per metric, and using
    each player's own precision instead of their game count alone.

    Both variances are estimated ONLY from observations whose game is strictly
    earlier than the one being shrunk. A single global fit would let a March
    game set the shrinkage applied to an October row; the expanding window keeps
    the feature reproducible at prediction time.

    ``inverse_n`` is ``1/n_present + 1/n_injured``, used to impute ``se2`` for
    players with a single game on one side, where no standard error exists:
    ``se2 ~= sigma2 * inverse_n`` with ``sigma2`` pooled from the fitted window.

    A ``tau2`` of zero is a real answer, not a failure: it says the spread
    between players is fully explained by sampling noise, and the honest
    estimate of every effect is then zero. The returned diagnostics report it.
    """
    weights = np.full(raw.shape, np.nan, dtype="float64")
    diagnostics = {"tau2": np.nan, "sigma2": np.nan, "implied_k": np.nan, "n_fit": 0.0}

    informative = (
        np.isfinite(raw)
        & np.isfinite(standard_errors)
        & np.isfinite(inverse_n)
        & (inverse_n > 0)
    )
    if int(informative.sum()) < min_samples:
        return weights, diagnostics

    order = np.flatnonzero(informative)
    order = order[np.argsort(date_ordinals[order], kind="mergesort")]
    fit_dates = date_ordinals[order]
    fit_raw = raw[order]
    fit_se2 = standard_errors[order] ** 2
    # se2 = sigma2 * (1/n_present + 1/n_injured), so this inverts to sigma2.
    fit_sigma2 = fit_se2 / inverse_n[order]

    counts = np.arange(1, fit_raw.size + 1, dtype="float64")
    cumulative_raw = np.cumsum(fit_raw)
    cumulative_raw2 = np.cumsum(fit_raw**2)
    cumulative_se2 = np.cumsum(fit_se2)
    cumulative_sigma2 = np.cumsum(fit_sigma2)

    # Observations strictly earlier than each row's game date.
    prior_counts = np.searchsorted(fit_dates, date_ordinals, side="left")
    fittable = prior_counts >= min_samples
    if not fittable.any():
        return weights, diagnostics

    last = prior_counts[fittable] - 1
    n_prior = counts[last]
    mean_raw = cumulative_raw[last] / n_prior
    var_raw = np.maximum(cumulative_raw2[last] / n_prior - mean_raw**2, 0.0)
    mean_se2 = cumulative_se2[last] / n_prior
    sigma2 = cumulative_sigma2[last] / n_prior
    tau2 = np.maximum(var_raw - mean_se2, 0.0)

    observed_se2 = standard_errors[fittable] ** 2
    imputed_se2 = sigma2 * inverse_n[fittable]
    se2 = np.where(np.isfinite(observed_se2), observed_se2, imputed_se2)

    usable = np.isfinite(raw[fittable]) & np.isfinite(se2) & ((tau2 + se2) > 0.0)
    fitted = np.full(se2.shape, np.nan, dtype="float64")
    fitted[usable] = tau2[usable] / (tau2[usable] + se2[usable])
    weights[fittable] = fitted

    final_tau2 = float(tau2[-1])
    final_sigma2 = float(sigma2[-1])
    diagnostics = {
        "tau2": final_tau2,
        "sigma2": final_sigma2,
        "implied_k": (
            float(final_sigma2 / final_tau2) if final_tau2 > 0.0 else float("inf")
        ),
        "n_fit": float(n_prior[-1]),
    }
    return weights, diagnostics


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
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    fit_shrinkage: bool = True,
    include_per_player_columns: bool = False,
    include_detailed_sample_size_features: bool = False,
) -> pd.DataFrame:
    """
    Compute player availability impact features from past games only (< current game date).

    Changes vs previous version:
      - DIFF_FROM_LINE is always computed against TOTAL_LINE_<main book> selected by
        total_line_book (or configured main book from config).
      - Effects are shrunk toward zero by an empirical-Bayes weight fitted per
        metric on strictly earlier games: tau2/(tau2 + se2), where tau2 is the
        between-player effect variance net of sampling noise. `shrinkage_k` is
        now only the fallback prior for rows with too little history to fit,
        and `shrinkage_k=0` still disables shrinkage entirely.
      - Adds team-oriented spread-error effects.
      - Each effect carries a Welch standard error instead of a p-value, so
        magnitude and precision stay on separate axes; it needs two games per
        group. See ``_continuous_effect_and_se`` for why the p-value went.
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
            game_id_col,
        ]
    ].copy()
    hist_home = hist_home.rename(columns={home_team_id_col: "TEAM_ID"})

    hist_away = df[
        [
            away_team_id_col,
            game_date_col,
            season_year_col,
            total_points_col,
            internal_diff_col,
            internal_spread_col,
            game_id_col,
        ]
    ].copy()
    hist_away = hist_away.rename(columns={away_team_id_col: "TEAM_ID"})
    hist_away[internal_spread_col] = -hist_away[internal_spread_col]

    df_hist = pd.concat([hist_home, hist_away], ignore_index=True)
    df_hist = df_hist.rename(
        columns={
            game_date_col: "GAME_DATE",
            season_year_col: "SEASON_YEAR",
            total_points_col: "TOTAL_POINTS",
            internal_diff_col: "DIFF_FROM_LINE",
            internal_spread_col: "SPREAD_ERROR",
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
    home_ses = {
        metric: np.full((n, n_home_players), np.nan, dtype="float64")
        for metric in EFFECT_METRICS
    }
    away_ses = {
        metric: np.full((n, n_away_players), np.nan, dtype="float64")
        for metric in EFFECT_METRICS
    }
    # Per-metric counts, because a metric with missing outcomes on some games
    # has a smaller effective sample than the player's overall game counts.
    home_metric_n_inj = {
        metric: np.zeros((n, n_home_players), dtype="float64")
        for metric in EFFECT_METRICS
    }
    home_metric_n_present = {
        metric: np.zeros((n, n_home_players), dtype="float64")
        for metric in EFFECT_METRICS
    }
    away_metric_n_inj = {
        metric: np.zeros((n, n_away_players), dtype="float64")
        for metric in EFFECT_METRICS
    }
    away_metric_n_present = {
        metric: np.zeros((n, n_away_players), dtype="float64")
        for metric in EFFECT_METRICS
    }
    row_date_ordinals = np.full(n, -1, dtype="int64")
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
        row_date_ordinals[i] = date_ord

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
                standard_errors,
                metric_injured_counts,
                metric_present_counts,
                n_inj,
                n_present,
                n_total,
            ) = _cached_effect(home_team_int, season_year_int, date_ord, pid_int)
            for metric, raw_effect, standard_error, injured_count, present_count in zip(
                EFFECT_METRICS,
                raw_effects,
                standard_errors,
                metric_injured_counts,
                metric_present_counts,
                strict=True,
            ):
                # Raw for now. Shrinkage needs the population of effects,
                # so it is applied once the loop has produced all of them.
                home_effects[metric][i, j] = raw_effect
                home_ses[metric][i, j] = standard_error
                home_metric_n_inj[metric][i, j] = injured_count
                home_metric_n_present[metric][i, j] = present_count
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
                standard_errors,
                metric_injured_counts,
                metric_present_counts,
                n_inj,
                n_present,
                n_total,
            ) = _cached_effect(away_team_int, season_year_int, date_ord, pid_int)
            for metric, raw_effect, standard_error, injured_count, present_count in zip(
                EFFECT_METRICS,
                raw_effects,
                standard_errors,
                metric_injured_counts,
                metric_present_counts,
                strict=True,
            ):
                # Raw for now. Shrinkage needs the population of effects,
                # so it is applied once the loop has produced all of them.
                away_effects[metric][i, j] = raw_effect
                away_ses[metric][i, j] = standard_error
                away_metric_n_inj[metric][i, j] = injured_count
                away_metric_n_present[metric][i, j] = present_count
            away_n_inj[i, j] = n_inj
            away_n_present[i, j] = n_present
            away_n_total[i, j] = n_total

    # ---- shrinkage -------------------------------------------------------
    # Applied here rather than inside the loop because the empirical-Bayes
    # weight depends on the spread of effects across the whole player
    # population, which is only known once every effect has been computed.
    if shrinkage_k > 0.0:
        shrinkage_report: dict[str, dict[str, float]] = {}
        for metric in EFFECT_METRICS:
            sides = (
                (
                    "HOME",
                    home_effects,
                    home_ses,
                    home_metric_n_inj,
                    home_metric_n_present,
                ),
                (
                    "AWAY",
                    away_effects,
                    away_ses,
                    away_metric_n_inj,
                    away_metric_n_present,
                ),
            )
            # Home and away effects estimate the same thing -- a player's impact
            # on their own team, with the away spread already sign-flipped to be
            # team-oriented -- so they are pooled into one fit.
            flat_raw, flat_se, flat_inv_n, flat_dates, shapes = [], [], [], [], []
            for _, effects, standard_errors, metric_n_inj, metric_n_present in sides:
                raw = effects[metric]
                n_present = metric_n_present[metric]
                n_injured = metric_n_inj[metric]
                with np.errstate(divide="ignore", invalid="ignore"):
                    inverse_n = np.where(
                        (n_present > 0) & (n_injured > 0),
                        1.0 / np.maximum(n_present, 1.0)
                        + 1.0 / np.maximum(n_injured, 1.0),
                        np.nan,
                    )
                shapes.append(raw.shape)
                flat_raw.append(raw.ravel())
                flat_se.append(standard_errors[metric].ravel())
                flat_inv_n.append(inverse_n.ravel())
                flat_dates.append(
                    np.repeat(row_date_ordinals, raw.shape[1]).astype("int64")
                )

            if fit_shrinkage:
                weights, diagnostics = _expanding_empirical_bayes_weights(
                    np.concatenate(flat_dates),
                    np.concatenate(flat_raw),
                    np.concatenate(flat_se),
                    np.concatenate(flat_inv_n),
                )
            else:
                weights = np.full(
                    sum(a.size for a in flat_raw), np.nan, dtype="float64"
                )
                diagnostics = {"n_fit": 0.0}
            shrinkage_report[metric] = diagnostics

            offset = 0
            for (_, effects, _, metric_n_inj, metric_n_present), shape in zip(
                sides, shapes, strict=True
            ):
                size = shape[0] * shape[1]
                metric_weights = weights[offset : offset + size].reshape(shape)
                offset += size

                # Rows too early to fit anything fall back to the n/(n+k) prior.
                n_eff = np.minimum(metric_n_inj[metric], metric_n_present[metric])
                fallback = np.where(n_eff > 0, n_eff / (n_eff + shrinkage_k), np.nan)
                factor = np.where(np.isfinite(metric_weights), metric_weights, fallback)
                effects[metric] = effects[metric] * factor

        for metric, diagnostics in shrinkage_report.items():
            if diagnostics["n_fit"] > 0:
                print(
                    f"  availability shrinkage [{out_prefix} {metric}]: "
                    f"tau2={diagnostics['tau2']:.4g} sigma2={diagnostics['sigma2']:.4g} "
                    f"implied k={diagnostics['implied_k']:.3g} "
                    f"(n={int(diagnostics['n_fit'])})"
                )
            elif fit_shrinkage:
                print(
                    f"  availability shrinkage [{out_prefix} {metric}]: "
                    f"too little history to fit; using k={shrinkage_k}"
                )
            else:
                print(
                    f"  availability shrinkage [{out_prefix} {metric}]: "
                    f"fitting disabled; using k={shrinkage_k}"
                )

    if include_per_player_columns:
        for j in range(n_home_players):
            for metric in EFFECT_METRICS:
                output_metric = metric_output_names[metric]
                df[_out_col("HOME", j + 1, output_metric)] = home_effects[metric][:, j]
                df[_out_col("HOME", j + 1, f"SE_{output_metric}")] = home_ses[metric][
                    :, j
                ]
            df[_out_col("HOME", j + 1, "N_INJ_GAMES")] = home_n_inj[:, j]
            df[_out_col("HOME", j + 1, "N_PRESENT_GAMES")] = home_n_present[:, j]
            df[_out_col("HOME", j + 1, "N_TOTAL_GAMES")] = home_n_total[:, j]

        for j in range(n_away_players):
            for metric in EFFECT_METRICS:
                output_metric = metric_output_names[metric]
                df[_out_col("AWAY", j + 1, output_metric)] = away_effects[metric][:, j]
                df[_out_col("AWAY", j + 1, f"SE_{output_metric}")] = away_ses[metric][
                    :, j
                ]
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

    # The row-level uncertainty channel, replacing the Bonferroni-corrected
    # p-value that used to sit here. Averaging the per-player standard errors
    # says how precisely this row's effects are pinned down without collapsing
    # effect size and sample size into a single number, and without a multiple
    # -comparison correction over a family of tests that were never powered.
    #
    # Left NaN when no player has evidence, exactly as the p-value aggregate
    # was: "no evidence" means unbounded uncertainty, and filling it with zero
    # would assert the opposite. Only the effect aggregates get filled below,
    # where zero is the correct limit of a shrinkage estimator.
    for side, effects, standard_errors in (
        ("HOME", home_effects, home_ses),
        ("AWAY", away_effects, away_ses),
    ):
        for metric in EFFECT_METRICS:
            output_metric = metric_output_names[metric]
            df[f"{out_prefix}_{side}_MEAN_{output_metric}"] = _nanmean_axis1(
                effects[metric]
            )
            df[f"{out_prefix}_{side}_MEAN_SE_{output_metric}"] = _nanmean_axis1(
                standard_errors[metric]
            )

    for side, effects in (("HOME", home_effects), ("AWAY", away_effects)):
        for metric in ("TOTAL_POINTS", "DIFF_FROM_LINE"):
            output_metric = metric_output_names[metric]
            df[f"{out_prefix}_{side}_MAX_ABS_{output_metric}"] = _nanmaxabs_axis1(
                effects[metric]
            )

    # No evidence for any of the players means the fully shrunk estimate, which
    # for an estimator that shrinks toward zero is zero -- the weight is
    # n/(n+k) or tau2/(tau2+se2), both 0 at n=0. Leaving NaN made the feature
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
        ],
        errors="ignore",
    )

    return df
