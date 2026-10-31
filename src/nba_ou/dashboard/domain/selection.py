"""Select each game's latest closing and nearest available intermediate model."""

import pandas as pd

from nba_ou.dashboard.domain.dates import utc_timestamp
from nba_ou.dashboard.domain.models import with_family_metadata
from nba_ou.dashboard.domain.teams import full_team_name

FAMILY_COLUMNS = ["game_id", "kind", "schema_version", "target", "variant"]
SLOT_COLUMNS = [*FAMILY_COLUMNS, "dataset_type", "model_horizon_minutes"]


def live_predictions(frame: pd.DataFrame, now) -> pd.DataFrame:
    """Pre-game outputs from live runs, retained after the game starts or ends."""
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    for column in (
        "predicted_at",
        "tipoff_at",
        "tipoff_at_prediction",
        "run_finished_at",
        "input_at",
    ):
        work[column] = pd.to_datetime(work[column], utc=True, errors="coerce")
    current = utc_timestamp(now)
    valid_mode = (work["dataset_type"].eq(1) & work["input_mode"].eq(0)) | (
        work["dataset_type"].eq(2) & work["input_mode"].isin([1, 2])
    )
    valid = (
        work["workflow"].isin([1, 3])
        & work["kind"].isin([1, 3])
        & work["target"].isin([1, 2, 3])
        & work["status"].isin([0, 1])
        & work["predicted_at"].lt(work["tipoff_at_prediction"])
        & work["predicted_at"].le(current)
        & work["run_finished_at"].le(current)
        & work["time_to_tipoff_seconds"].gt(0)
        & valid_mode
    )
    return work.loc[valid].copy()


def slate_games(frame: pd.DataFrame, game_date) -> pd.DataFrame:
    """Keep scheduled, started and final games belonging to this NBA slate."""
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    work["tipoff_at"] = pd.to_datetime(work["tipoff_at"], utc=True, errors="coerce")
    for side in ("home", "away"):
        work[f"{side}_team"] = [
            full_team_name(name, team_id)
            for name, team_id in zip(
                work[f"{side}_team"], work[f"{side}_team_id"], strict=True
            )
        ]
    return work.loc[
        work["status"].isin([0, 1])
        & pd.to_datetime(work["game_date"]).dt.date.eq(game_date)
    ].sort_values(["tipoff_at", "game_id"])


def select_predictions(
    frame: pd.DataFrame,
    target: int | tuple[int, ...],
    now,
    family_keys: list[str] | None = None,
) -> pd.DataFrame:
    work = live_predictions(frame, now)
    if work.empty:
        return work
    targets = (target,) if isinstance(target, int) else target
    work = with_family_metadata(work.loc[work["target"].isin(targets)])
    if family_keys is not None:
        work = work.loc[work["family_key"].isin(family_keys)].copy()
    if work.empty:
        return work

    # A refit supersedes a build in the same slot. Never group by fit_id.
    work = work.sort_values(
        ["predicted_at", "run_finished_at", "run_id", "input_id", "model_build_id"],
        kind="stable",
    ).drop_duplicates(SLOT_COLUMNS, keep="last")
    work["minutes_remaining"] = (
        work["tipoff_at"] - utc_timestamp(now)
    ).dt.total_seconds().clip(lower=0) / 60
    work["horizon_gap_minutes"] = (
        pd.to_numeric(work["model_horizon_minutes"], errors="coerce")
        - work["minutes_remaining"]
    ).abs()
    closing = work.loc[work["dataset_type"].eq(1)].copy()
    intermediate = work.loc[
        work["dataset_type"].eq(2)
        & pd.to_numeric(work["model_horizon_minutes"], errors="coerce").gt(0)
    ].copy()
    # Resolve equal distances toward the longer horizon, then the newest output.
    intermediate = intermediate.sort_values(
        ["horizon_gap_minutes", "model_horizon_minutes", "predicted_at"],
        ascending=[True, False, False],
        kind="stable",
    ).drop_duplicates(FAMILY_COLUMNS, keep="first")
    selected = pd.concat([closing, intermediate], ignore_index=True)
    selected["phase"] = selected["dataset_type"].map({1: "Closing", 2: "Intermediate"})
    return selected.sort_values(["tipoff_at", "dataset_type", "family_key"])


def market_snapshots(frame: pd.DataFrame, target: int, now) -> pd.DataFrame:
    """One coherent latest stored market observation per game, before model filters."""
    work = live_predictions(frame, now)
    if work.empty:
        return work
    targets = [3] if target == 3 else [1, 2]
    work = work.loc[
        work["target"].isin(targets)
        & work["anchor_book"].str.lower().isin(["bet365", "bet_365"])
        & work["input_at"].le(utc_timestamp(now))
        & work["input_at"].lt(work["tipoff_at_prediction"])
    ].copy()
    work = work.sort_values(
        ["input_at", "predicted_at", "run_id", "input_id"], kind="stable"
    ).drop_duplicates("game_id", keep="last")
    work["bet365_line"] = pd.to_numeric(work["bet_line_x2"], errors="coerce") / 2
    columns = [
        "game_id",
        "bet365_line",
        "books_mean_line",
        "books_median_line",
        "books_count",
        "input_at",
        "bet_line_age_minutes",
        "line_source",
    ]
    return work[columns].rename(columns={"input_at": "market_at"})
