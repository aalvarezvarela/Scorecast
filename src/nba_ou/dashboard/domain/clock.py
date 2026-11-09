"""One bet per game/series at a common Madrid clock hour."""

from collections.abc import Mapping
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from nba_ou.dashboard.config import DISPLAY_SCHEMA_VERSION, MADRID
from nba_ou.dashboard.domain.analysis import prepare_analysis, wilson_interval
from nba_ou.dashboard.domain.comparisons import prediction_values
from nba_ou.dashboard.domain.models import without_hidden_variants

SERIES = ("TabPFN", "Closing", "Intermediate")
SOURCES = ("Live predictions", "Snapshot replay")
SLOT = [
    "kind",
    "schema_version",
    "target",
    "dataset_type",
    "model_horizon_minutes",
    "variant",
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


def clock_members(
    predictions: pd.DataFrame, source: str, kind: int | None = None
) -> pd.DataFrame:
    """Keep one latest output per slot, then the closest prior TabPFN snapshot.

    SQL performs the bounded as-of read. Repeat the cutoff checks here so
    callers cannot accidentally grade a future snapshot or duplicate a refit.
    """
    if predictions.empty:
        return predictions.copy()
    work = without_hidden_variants(predictions)
    work = work.loc[work["source"].eq(source)].copy()
    if kind is not None:
        work = work.loc[work["kind"].eq(kind)].copy()
    if work.empty:
        return work
    for column in (
        "cutoff_at",
        "input_at",
        "available_at",
        "selection_at",
        "tipoff_at",
    ):
        work[column] = pd.to_datetime(work[column], utc=True)
    work = work.loc[
        work["input_at"].le(work["cutoff_at"])
        & work["selection_at"].le(work["cutoff_at"])
        & work["cutoff_at"].lt(work["tipoff_at"])
    ].copy()
    if source == SOURCES[0]:
        work = work.loc[work["available_at"].le(work["cutoff_at"])].copy()
    work = prediction_values(work)
    work["implied_prediction"] = work["predicted_total"].where(
        work["target"].ne(3), work["predicted_home_margin"]
    )
    # The closest snapshot takes precedence for TabPFN; other slots use their
    # latest publication (live) or latest input snapshot (replay).
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
        [work["kind"].eq(3), work["dataset_type"].eq(1)],
        ["TabPFN", "Closing"],
        default="Intermediate",
    )
    tab = (
        work.loc[work["series"].eq("TabPFN")]
        .sort_values(
            [
                "input_at",
                "available_at",
                "dataset_type",
                "model_horizon_minutes",
                "variant",
                "run_id",
                "input_id",
                "model_build_id",
            ],
            kind="stable",
        )
        .drop_duplicates(["game_id", "hour"], keep="last")
    )
    return pd.concat([work.loc[work["series"].ne("TabPFN")], tab], ignore_index=True)


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = clock_members(predictions, source, kind)
    names = SERIES if kind is None else (("TabPFN",) if kind == 3 else SERIES[1:])
    if games.empty:
        return pd.DataFrame(), members
    # Missing models remain explicit opportunities; never count the same game
    # twice in a series merely because several horizons voted.
    opportunities = []
    for game in games.to_dict("records"):
        for hour in hours:
            cutoff = madrid_cutoff(game["game_date"], hour)
            if cutoff >= pd.Timestamp(game["tipoff_at"]) or cutoff > pd.Timestamp(
                fetched_at
            ):
                continue
            for name in names:
                opportunities.append(
                    {**game, "hour": hour, "cutoff_at": cutoff, "series": name}
                )
    work = pd.DataFrame(opportunities)
    if work.empty:
        return work, members
    if not members.empty:
        groups = members.groupby(["game_id", "hour", "series"], sort=False)
        means = groups.agg(
            prediction_value=("implied_prediction", "mean"),
            members=("implied_prediction", "count"),
            input_at=("input_at", "min"),
            latest_input_at=("input_at", "max"),
            available_at=("available_at", "max"),
            bet_line_x2=("bet_line_x2", "first"),
            bet_price_over_home=("bet_price_over_home", "first"),
            bet_price_under_away=("bet_price_under_away", "first"),
            quote_age_minutes=("quote_age_minutes", "first"),
            quote_input_at=("quote_input_at", "first"),
        ).reset_index()
        work = work.merge(
            means, on=["game_id", "hour", "series"], how="left", validate="one_to_one"
        )
    else:
        for column in (
            "prediction_value",
            "members",
            "input_at",
            "latest_input_at",
            "available_at",
            "bet_line_x2",
            "bet_price_over_home",
            "bet_price_under_away",
            "quote_age_minutes",
            "quote_input_at",
        ):
            work[column] = np.nan
    work["members"] = work["members"].fillna(0).astype(int)
    work["available"] = work["members"].gt(0) & work["prediction_value"].notna()
    # All member outputs are now absolute points (or home margin); a zero
    # anchor lets the existing target-specific grader preserve that scale.
    work = work.assign(
        model_line=0.0,
        kind=np.where(work["series"].eq("TabPFN"), 3, 1),
        target=target,
        schema_version=DISPLAY_SCHEMA_VERSION,
        variant="main",
        dataset_type=np.where(work["series"].eq("Intermediate"), 2, 1),
        model_horizon_minutes=0,
        source=source,
        bucket=0,
    )
    # prepare_analysis marks supplied rows available. Restore missing
    # opportunities before computing any summary.
    available = work["available"].copy()
    work = prepare_analysis(work, min_edge=thresholds)
    work["available"] = available
    work.loc[~available, "result"] = "MISSING"
    work.loc[~available, ["absolute_error", "unit_profit"]] = np.nan
    work.loc[~available, "priced_bet"] = False
    return work, members


def clock_summary(bets: pd.DataFrame, *, common_games: bool = False) -> pd.DataFrame:
    """Pool W/L counts, with eligible final games as the coverage denominator.

    Common games means a prediction and line exist at every displayed hour
    for every displayed series. It does not require a directional bet.
    """
    if bets.empty:
        return pd.DataFrame()
    work = bets
    if common_games:
        needed = len(work["hour"].unique()) * len(work["series"].unique())
        valid = (
            work["available"]
            & work["reference_line"].notna()
            & work["status"].eq(1)
            & work["actual_value"].notna()
        )
        counts = work.loc[valid].groupby("game_id").size()
        work = work.loc[work["game_id"].isin(counts.index[counts.eq(needed)])]
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
