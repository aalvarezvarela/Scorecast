"""The original prediction of each model group closest to every Madrid hour.

The hour only chooses a prediction. The chosen one keeps its own line, side and
result; nothing is averaged or graded against the market of that hour.
"""

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from nba_ou.dashboard.config import DISPLAY_SCHEMA_VERSION, MADRID
from nba_ou.dashboard.domain.analysis import (
    MIN_CELL_BETS,
    prepare_analysis,
    wilson_interval,
)
from nba_ou.dashboard.domain.models import without_hidden_variants
from nba_ou.dashboard.domain.strategies import (
    game_permutation_pvalue,
    same_timepoint_combinations,
    selection_chance,
)

# Series name -> (model kind, dataset type: 1 closing, 2 intermediate).
SERIES = {
    "XGBoost · Intermediate": (1, 2),
    "XGBoost · Closing": (1, 1),
    "TabPFN · Intermediate": (3, 2),
    "TabPFN · Closing": (3, 1),
}
SOURCES = ("Live predictions", "Snapshot replay")
TOP_CELLS = 4
HOUR_COLUMN = "Hour (Madrid)"
# A prediction can be chosen for an hour up to this far before or after it.
WINDOW = pd.Timedelta(hours=1)
SLOT = [
    "kind",
    "schema_version",
    "target",
    "dataset_type",
    "model_horizon_minutes",
    "variant",
]
_ORIGINAL_COLUMNS = [
    "schema_version",
    "model_horizon_minutes",
    "variant",
    "prediction_value",
    "model_line",
    "bet_line_x2",
    "bet_price_over_home",
    "bet_price_under_away",
    "input_at",
    "predicted_at",
    "available_at",
    "reference_at",
    "minutes_from_hour",
    "run_id",
    "input_id",
    "model_build_id",
]


def hour_label(hour: int) -> str:
    return f"{hour % 24:02d}:00" + (" (+1 day)" if hour >= 24 else "")


def madrid_cutoff(day: date, hour: int) -> pd.Timestamp:
    """Match PostgreSQL: repeated autumn hours use the second occurrence.

    A missing spring hour normalizes forward on the round trip through UTC.
    """
    local = datetime.combine(day + timedelta(days=hour // 24), time(hour % 24), MADRID)
    repeated = pd.Timestamp(local.replace(fold=1)).tz_convert("UTC")
    if repeated.tz_convert(MADRID).tz_localize(None) != pd.Timestamp(
        local.replace(tzinfo=None)
    ):
        return pd.Timestamp(local.replace(fold=0)).tz_convert("UTC")
    return repeated


def series_names(kind: int | None = None) -> list[str]:
    """The series of one model family, or all four."""
    return [
        name
        for name, (series_kind, _) in SERIES.items()
        if kind is None or series_kind == kind
    ]


def clock_cutoffs(
    games: pd.DataFrame, hours: tuple[int, ...], fetched_at: datetime
) -> pd.DataFrame:
    """One row per game and hour whose window opened before tip-off and the read."""
    rows = []
    for game in games.to_dict("records"):
        for hour in hours:
            cutoff = madrid_cutoff(game["game_date"], hour)
            opens = cutoff - WINDOW
            if opens < pd.Timestamp(game["tipoff_at"]) and opens <= pd.Timestamp(
                fetched_at
            ):
                rows.append({**game, "hour": hour, "cutoff_at": cutoff})
    return pd.DataFrame(rows)


def _with_series(frame: pd.DataFrame) -> pd.DataFrame:
    named = frame.assign(
        series=np.select(
            [
                frame["kind"].eq(series_kind) & frame["dataset_type"].eq(dataset)
                for series_kind, dataset in SERIES.values()
            ],
            list(SERIES),
            default="",
        )
    )
    return named.loc[named["series"].ne("")]


def select_predictions(
    predictions: pd.DataFrame,
    cutoffs: pd.DataFrame,
    source: str,
    kind: int | None = None,
) -> pd.DataFrame:
    """At most one original prediction per game, hour and series.

    Reference time: when a live prediction was generated, and the time of the
    snapshot of a replay, whenever the backfill ran. Only predictions that can
    be bet (an output and their own line) and were made before tip-off, within
    ``WINDOW`` of the hour, are candidates; the closest one is chosen.
    """
    frame = without_hidden_variants(predictions)
    if frame.empty or cutoffs.empty:
        return pd.DataFrame()
    work = frame.loc[
        frame["source"].eq(source)
        & frame["prediction_value"].notna()
        & frame["bet_line_x2"].notna()
    ]
    if kind is not None:
        work = work.loc[work["kind"].eq(kind)]
    work = _with_series(work)
    if work.empty:
        return pd.DataFrame()
    work = work.merge(cutoffs[["game_id", "hour", "cutoff_at"]], on="game_id")
    for column in ("input_at", "predicted_at", "available_at", "tipoff_at_prediction"):
        work[column] = pd.to_datetime(work[column], utc=True)
    work["reference_at"] = (
        work["predicted_at"] if source == SOURCES[0] else work["input_at"]
    )
    offset = work["reference_at"] - work["cutoff_at"]
    work = work.assign(minutes_from_hour=offset.dt.total_seconds() / 60)
    work = work.loc[
        offset.abs().le(WINDOW) & work["reference_at"].le(work["tipoff_at_prediction"])
    ]
    if work.empty:
        return pd.DataFrame()
    minutes_to_tip = (
        work["tipoff_at_prediction"] - work["reference_at"]
    ).dt.total_seconds() / 60
    # Ties: the earlier side of the hour; within one run, the model trained
    # nearest the time to tip; then the latest stored run, as in the other views.
    order = work.assign(
        _distance=work["minutes_from_hour"].abs(),
        _horizon_gap=(minutes_to_tip - work["model_horizon_minutes"]).abs(),
    ).sort_values(
        [
            "_distance",
            "reference_at",
            "_horizon_gap",
            "available_at",
            "predicted_at",
            "run_id",
            "input_id",
            "model_build_id",
        ],
        ascending=[True, True, True, False, False, False, False, False],
        kind="stable",
    )
    return order.drop_duplicates(["game_id", "hour", "series"], keep="first").drop(
        columns=["_distance", "_horizon_gap"]
    )


def clock_bets(
    games: pd.DataFrame,
    predictions: pd.DataFrame,
    source: str,
    hours: tuple[int, ...],
    fetched_at: datetime,
    *,
    kind: int | None = None,
    target: int = 1,
    thresholds: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    """One opportunity per game, hour and series, missing predictions included.

    Each chosen prediction is graded by the shared grader on its own line,
    exactly as the checkpoint views grade it.
    """
    if games.empty:
        return pd.DataFrame()
    cutoffs = clock_cutoffs(games, hours, fetched_at)
    if cutoffs.empty:
        return cutoffs
    work = cutoffs.merge(pd.DataFrame({"series": series_names(kind)}), how="cross")
    keys = ["game_id", "hour", "series"]
    selected = select_predictions(predictions, cutoffs, source, kind)
    if selected.empty:
        work = work.assign(**{column: np.nan for column in _ORIGINAL_COLUMNS})
    else:
        work = work.merge(selected[[*keys, *_ORIGINAL_COLUMNS]], on=keys, how="left")
    available = work["prediction_value"].notna()
    # Prices, lines and outputs of a chosen row are untouched. The fills only
    # label empty rows, so the grader can report them as missing.
    work = work.assign(
        kind=[SERIES[name][0] for name in work["series"]],
        dataset_type=[SERIES[name][1] for name in work["series"]],
        target=target,
        schema_version=work["schema_version"].fillna(DISPLAY_SCHEMA_VERSION),
        variant=work["variant"].fillna("main"),
        model_horizon_minutes=work["model_horizon_minutes"].fillna(0).astype(int),
        source=source,
        bucket=0,
    )
    # prepare_analysis marks supplied rows available. Restore the missing
    # opportunities before computing any summary.
    work = prepare_analysis(work, min_edge=thresholds)
    work["available"] = available
    work.loc[~available, "result"] = "MISSING"
    work.loc[~available, ["absolute_error", "unit_profit"]] = np.nan
    work.loc[~available, "priced_bet"] = False
    return work


def common_cohort(bets: pd.DataFrame) -> pd.DataFrame:
    """Games with a prediction and line at every displayed hour and series.

    It does not require a directional bet.
    """
    if bets.empty:
        return bets
    needed = len(bets["hour"].unique()) * len(bets["series"].unique())
    valid = (
        bets["available"]
        & bets["reference_line"].notna()
        & bets["status"].eq(1)
        & bets["actual_value"].notna()
    )
    counts = bets.loc[valid].groupby("game_id").size()
    return bets.loc[bets["game_id"].isin(counts.index[counts.eq(needed)])]


def clock_summary(bets: pd.DataFrame) -> pd.DataFrame:
    """Pool W/L counts, with eligible final games as the coverage denominator."""
    if bets.empty:
        return pd.DataFrame()
    work = bets
    rows = []
    for (hour, series), group in work.groupby(["hour", "series"], sort=True):
        counts = group["result"].value_counts()
        wins, losses, pushes = (
            int(counts.get(value, 0)) for value in ("WIN", "LOSS", "PUSH")
        )
        final = group["actual_value"].notna() & group["status"].eq(1)
        eligible = int(final.sum())
        covered = int(
            (final & group["available"] & group["reference_line"].notna()).sum()
        )
        priced = int(group["priced_bet"].sum())
        low, high = wilson_interval(wins, losses)
        rows.append(
            dict(
                hour=hour,
                hour_label=hour_label(hour),
                series=series,
                eligible_games=eligible,
                covered_games=covered,
                coverage=100 * covered / eligible if eligible else np.nan,
                wins=wins,
                losses=losses,
                pushes=pushes,
                bets=wins + losses + pushes,
                win_pct=100 * wins / (wins + losses) if wins + losses else np.nan,
                ci_low=low,
                ci_high=high,
                roi=100 * group["unit_profit"].sum() / priced if priced else np.nan,
                priced_bets=priced,
                mae=group.loc[final & group["available"], "absolute_error"].mean(),
                missing=int(counts.get("MISSING", 0)),
                no_bet=int(counts.get("NO BET", 0)),
                pending=int(counts.get("PENDING", 0)),
                void=int(counts.get("VOID", 0)),
                median_minutes_from_hour=(
                    group["minutes_from_hour"].abs().median()
                    if group["minutes_from_hour"].notna().any()
                    else np.nan
                ),
            )
        )
    return pd.DataFrame(rows)


def _eligible_cells(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    return summary.loc[
        summary["win_pct"].notna()
        & summary["wins"].add(summary["losses"]).ge(MIN_CELL_BETS)
    ]


def best_cells(summary: pd.DataFrame, limit: int = TOP_CELLS) -> pd.DataFrame:
    """The hour and series pairs with the highest win %, enough settled bets only."""
    return (
        _eligible_cells(summary)
        .sort_values(["win_pct", "bets"], ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )


def leader_checks(
    cohort: pd.DataFrame, summary: pd.DataFrame
) -> tuple[float, float, int]:
    """Coin-flip chance and game-permutation p of the best cell, and cells tried."""
    eligible = _eligible_cells(summary)
    if eligible.empty:
        return float("nan"), float("nan"), 0
    best = float(eligible["win_pct"].max())
    chance = selection_chance(eligible["wins"] + eligible["losses"], best)
    cells = cohort.assign(cell=cohort["hour"].astype(str) + "|" + cohort["series"])
    names = eligible["hour"].astype(str) + "|" + eligible["series"]
    return chance, game_permutation_pvalue(cells, "cell", names, best), len(eligible)


def coverage_text(covered: int, eligible: int) -> str:
    """Games with a valid prediction out of the final games, with the percentage."""
    return f"{covered}/{eligible} ({100 * covered / eligible:.1f}%)"


def _cell_text(row) -> str:
    """Win %, record and coverage; a cell without wagers still shows coverage."""
    if not row.eligible_games:
        return "—"
    coverage = coverage_text(row.covered_games, row.eligible_games)
    if pd.isna(row.win_pct):
        return f"— · {coverage}"
    return f"{row.win_pct:.1f}% · {row.wins}-{row.losses} · {coverage}"


def hour_model_table(
    summary: pd.DataFrame, names: list[str], hours: tuple[int, ...] | None = None
) -> pd.DataFrame:
    """One row per hour and one column per series, as readable text.

    ``hours`` keeps a fixed list of rows, empty ones included.
    """
    if summary.empty:
        return pd.DataFrame()
    table = (
        summary.assign(cell=[_cell_text(row) for row in summary.itertuples()])
        .pivot(index="hour", columns="series", values="cell")
        .reindex(columns=names)
    )
    table = table.sort_index() if hours is None else table.reindex(index=list(hours))
    table = table.fillna("—")
    table.insert(0, HOUR_COLUMN, [hour_label(hour) for hour in table.index])
    return table.reset_index(drop=True).rename_axis(columns=None)


def hour_combinations(
    bets: pd.DataFrame, *, thresholds: Mapping[int, float] | None = None
) -> pd.DataFrame:
    """Graded rows of every combination of the series chosen at one hour.

    At each hour, each subset of 2–4 series is scored on the games where all of
    them have a chosen prediction with a line, as a mean edge and as a majority
    vote, exactly as the same-timepoint combinations of the historical view.
    """
    if bets.empty:
        return pd.DataFrame()
    members = bets.loc[bets["available"]].assign(member=lambda f: f["series"])
    return same_timepoint_combinations(
        members, min_edge=thresholds, timepoint="hour", label=hour_label
    )
