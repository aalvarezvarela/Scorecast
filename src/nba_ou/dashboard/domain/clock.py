"""One bet per game and model group at each Madrid clock hour."""

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
from nba_ou.dashboard.domain.comparisons import prediction_values
from nba_ou.dashboard.domain.models import without_hidden_variants
from nba_ou.dashboard.domain.strategies import (
    game_permutation_pvalue,
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
SLOT = [
    "kind",
    "schema_version",
    "target",
    "dataset_type",
    "model_horizon_minutes",
    "variant",
]
_MEAN_COLUMNS = [
    "prediction_value",
    "members",
    "input_at",
    "latest_input_at",
    "available_at",
]
_QUOTE_COLUMNS = [
    "bet_line_x2",
    "bet_price_over_home",
    "bet_price_under_away",
    "quote_input_at",
    "quote_age_minutes",
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
    """One row per game and betting hour that had passed before the game started."""
    rows = []
    for game in games.to_dict("records"):
        for hour in hours:
            cutoff = madrid_cutoff(game["game_date"], hour)
            if cutoff < pd.Timestamp(game["tipoff_at"]) and cutoff <= pd.Timestamp(
                fetched_at
            ):
                rows.append({**game, "hour": hour, "cutoff_at": cutoff})
    return pd.DataFrame(rows)


def _known_at_cutoff(
    frame: pd.DataFrame, cutoffs: pd.DataFrame, source: str
) -> pd.DataFrame:
    """Rows of ``source`` per cutoff that had been input and released by then."""
    if frame.empty or cutoffs.empty:
        return pd.DataFrame()
    work = frame.loc[frame["source"].eq(source)]
    if work.empty:
        return pd.DataFrame()
    work = work.merge(cutoffs[["game_id", "hour", "cutoff_at"]], on="game_id")
    for column in ("input_at", "available_at", "tipoff_at_prediction"):
        work[column] = pd.to_datetime(work[column], utc=True)
    # Replay is judged by reconstructed input time, whenever the backfill ran.
    work["selection_at"] = (
        work["available_at"] if source == SOURCES[0] else work["input_at"]
    )
    return work.loc[
        work["input_at"].le(work["cutoff_at"])
        & work["selection_at"].le(work["cutoff_at"])
        & work["cutoff_at"].lt(work["tipoff_at_prediction"])
    ].copy()


def clock_members(
    predictions: pd.DataFrame,
    cutoffs: pd.DataFrame,
    source: str,
    kind: int | None = None,
) -> pd.DataFrame:
    """The previous available output of every model at each hour, per game."""
    frame = without_hidden_variants(predictions)
    if kind is not None and not frame.empty:
        frame = frame.loc[frame["kind"].eq(kind)]
    work = _known_at_cutoff(frame, cutoffs, source)
    if work.empty:
        return work
    work = prediction_values(work)
    work["implied_prediction"] = work["predicted_total"].where(
        work["target"].ne(3), work["predicted_home_margin"]
    )
    # TabPFN republishes old snapshots, so its newest input decides; other
    # models use their latest publication (live) or input snapshot (replay).
    work["_order_at"] = work["selection_at"].where(work["kind"].ne(3), work["input_at"])
    work = work.sort_values(
        [
            "_order_at",
            "input_at",
            "available_at",
            "run_id",
            "input_id",
            "model_build_id",
        ],
        kind="stable",
    ).drop_duplicates(["game_id", "hour", *SLOT], keep="last")
    work["series"] = np.select(
        [
            work["kind"].eq(series_kind) & work["dataset_type"].eq(dataset)
            for series_kind, dataset in SERIES.values()
        ],
        list(SERIES),
        default="",
    )
    return work.loc[work["series"].ne("")].copy()


def clock_quotes(
    quotes: pd.DataFrame, cutoffs: pd.DataFrame, source: str
) -> pd.DataFrame:
    """The newest quote known at each hour, shared by every series of the game."""
    work = _known_at_cutoff(quotes, cutoffs, source)
    if work.empty:
        return work
    work = work.sort_values(
        ["input_at", "available_at", "input_id"], kind="stable"
    ).drop_duplicates(["game_id", "hour"], keep="last")
    work["quote_input_at"] = work["input_at"]
    elapsed = (work["cutoff_at"] - work["input_at"]).dt.total_seconds() / 60
    work["quote_age_minutes"] = elapsed + pd.to_numeric(
        work["quote_age_at_input"], errors="coerce"
    )
    return work


def _left_attach(
    work: pd.DataFrame, frame: pd.DataFrame, keys: list[str], columns: list[str]
) -> pd.DataFrame:
    if frame.empty:
        return work.assign(**{column: np.nan for column in columns})
    return work.merge(frame[[*keys, *columns]], on=keys, how="left")


def clock_bets(
    games: pd.DataFrame,
    predictions: pd.DataFrame,
    quotes: pd.DataFrame,
    source: str,
    hours: tuple[int, ...],
    fetched_at: datetime,
    *,
    kind: int | None = None,
    target: int = 1,
    thresholds: Mapping[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One graded opportunity per game, hour and series, missing ones included.

    Every model of a series contributes its previous available output, in
    absolute points (or home margin), and the series bets their equal-weight mean.
    """
    if games.empty:
        return pd.DataFrame(), pd.DataFrame()
    cutoffs = clock_cutoffs(games, hours, fetched_at)
    if cutoffs.empty:
        return cutoffs, pd.DataFrame()
    members = clock_members(predictions, cutoffs, source, kind)
    means = pd.DataFrame()
    if not members.empty:
        means = (
            members.groupby(["game_id", "hour", "series"], sort=False)
            .agg(
                prediction_value=("implied_prediction", "mean"),
                members=("implied_prediction", "count"),
                input_at=("input_at", "min"),
                latest_input_at=("input_at", "max"),
                available_at=("available_at", "max"),
            )
            .reset_index()
        )
    names = series_names(kind)
    work = cutoffs.merge(pd.DataFrame({"series": names}), how="cross")
    work = _left_attach(work, means, ["game_id", "hour", "series"], _MEAN_COLUMNS)
    work = _left_attach(
        work, clock_quotes(quotes, cutoffs, source), ["game_id", "hour"], _QUOTE_COLUMNS
    )
    work["members"] = work["members"].fillna(0).astype(int)
    available = work["members"].gt(0) & work["prediction_value"].notna()
    # Outputs are absolute by now; a zero anchor lets the existing
    # target-specific grader preserve that scale.
    work = work.assign(
        model_line=0.0,
        kind=[SERIES[name][0] for name in work["series"]],
        dataset_type=[SERIES[name][1] for name in work["series"]],
        target=target,
        schema_version=DISPLAY_SCHEMA_VERSION,
        variant="main",
        model_horizon_minutes=0,
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
    return work, members


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
                no_line=int(counts.get("NO LINE", 0)),
                no_bet=int(counts.get("NO BET", 0)),
                pending=int(counts.get("PENDING", 0)),
                void=int(counts.get("VOID", 0)),
                mean_members=group.loc[group["available"], "members"].mean(),
                median_quote_age_minutes=(
                    group["quote_age_minutes"].median()
                    if group["quote_age_minutes"].notna().any()
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


def _cell_text(row) -> str:
    if pd.isna(row.win_pct):
        return "—"
    text = f"{row.win_pct:.1f}% · {row.wins}-{row.losses}"
    return text if pd.isna(row.roi) else f"{text} · ROI {row.roi:+.1f}%"


def hour_model_table(summary: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """One row per hour and one column per series, as readable text."""
    if summary.empty:
        return pd.DataFrame()
    table = (
        summary.assign(cell=[_cell_text(row) for row in summary.itertuples()])
        .pivot(index="hour", columns="series", values="cell")
        .reindex(columns=names)
        .fillna("—")
        .sort_index()
    )
    table.insert(0, HOUR_COLUMN, [hour_label(hour) for hour in table.index])
    return table.reset_index(drop=True).rename_axis(columns=None)
