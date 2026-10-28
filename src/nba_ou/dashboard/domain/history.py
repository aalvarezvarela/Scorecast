"""Historical checkpoints and grading against each prediction's stored quote."""

from collections.abc import Mapping

import pandas as pd

from nba_ou.dashboard.config import DEFAULT_MIN_EDGE_BY_KIND
from nba_ou.dashboard.domain.comparisons import prediction_values, signal
from nba_ou.dashboard.domain.dates import horizon_label, utc_timestamp
from nba_ou.dashboard.domain.models import (
    displayed_models,
    model_families,
    with_family_metadata,
)
from nba_ou.dashboard.domain.teams import full_team_name

HISTORY_SOURCES = ("Live predictions", "Snapshot replay")
CHECKPOINT_TOLERANCE = pd.Timedelta(minutes=30)
CLOSING_CHECKPOINTS = (360, 240, 0)
INTERMEDIATE_CHECKPOINTS = (480, 360, 240)


def historical_games(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if work.empty:
        return work
    work["tipoff_at"] = pd.to_datetime(work["tipoff_at"], utc=True)
    for side in ("home", "away"):
        work[f"{side}_team"] = [
            full_team_name(name, team_id)
            for name, team_id in zip(
                work[f"{side}_team"], work[f"{side}_team_id"], strict=True
            )
        ]
    return work.sort_values(["tipoff_at", "game_id"])


def select_history(
    frame: pd.DataFrame, games: pd.DataFrame, target: int, source: str, now
) -> pd.DataFrame:
    """One row per game/family/checkpoint, including explicit missing slots.

    Closing checkpoints freeze availability at the cutoff, with at most 30
    minutes of age. Intermediate live rows use that model near its own input
    horizon; up to 30 minutes of publication delay is disclosed, never hidden.
    Replay uses exact own-horizon rows, allowing post-game reconstruction.
    """
    if source not in HISTORY_SOURCES:
        raise ValueError("Unknown historical prediction source")
    if frame.empty or games.empty:
        return pd.DataFrame()
    work = displayed_models(frame)
    work = work.loc[work["target"].eq(target) & work["workflow"].isin([1, 2, 3])].copy()
    for column in ("predicted_at", "run_finished_at", "input_at", "tipoff_at"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="coerce")
    current = utc_timestamp(now)
    work = work.loc[
        work["predicted_at"].le(current)
        & work["run_finished_at"].le(current)
        & work["input_at"].le(work["predicted_at"])
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work = with_family_metadata(work)
    work["available_at"] = work[["predicted_at", "run_finished_at"]].max(axis=1)
    work = work.sort_values(
        ["available_at", "predicted_at", "run_id", "input_id", "model_build_id"],
        kind="stable",
    )
    families = model_families(work, target)
    rows = []
    for _, game in games.iterrows():
        tipoff = game["tipoff_at"]
        for family in families:
            candidates = work.loc[
                work["game_id"].eq(game["game_id"]) & work["family_key"].eq(family.key)
            ]
            for dataset_type, checkpoints in (
                (1, CLOSING_CHECKPOINTS),
                (2, INTERMEDIATE_CHECKPOINTS),
            ):
                if not (family.closing if dataset_type == 1 else family.intermediate):
                    continue
                for order, horizon in enumerate(checkpoints):
                    cutoff = tipoff - pd.Timedelta(minutes=horizon)
                    eligible = candidates.loc[
                        candidates["dataset_type"].eq(dataset_type)
                    ]
                    if dataset_type == 1:
                        eligible = eligible.loc[
                            eligible["workflow"].isin([1, 3])
                            & eligible["input_mode"].eq(0)
                            & eligible["model_horizon_minutes"].eq(0)
                            & eligible["time_to_tipoff_seconds"].gt(0)
                            & eligible["available_at"].lt(tipoff)
                            & eligible["available_at"].le(cutoff)
                        ]
                        if horizon:
                            eligible = eligible.loc[
                                eligible["available_at"].ge(
                                    cutoff - CHECKPOINT_TOLERANCE
                                )
                                & eligible["predicted_at"].ge(
                                    cutoff - CHECKPOINT_TOLERANCE
                                )
                            ]
                    else:
                        eligible = eligible.loc[
                            eligible["model_horizon_minutes"].eq(horizon)
                        ]
                        if source == "Snapshot replay":
                            eligible = eligible.loc[
                                eligible["input_mode"].eq(1)
                                & eligible["snapshot_horizon_minutes"].eq(horizon)
                                & (
                                    eligible["workflow"].eq(2)
                                    | (
                                        eligible["kind"].eq(3)
                                        & eligible["workflow"].eq(3)
                                    )
                                )
                            ]
                        else:
                            eligible = eligible.loc[
                                eligible["workflow"].isin([1, 3])
                                & eligible["input_mode"].isin([1, 2])
                                & eligible["time_to_tipoff_seconds"].gt(0)
                                & eligible["available_at"].lt(tipoff)
                                & eligible["input_at"].between(
                                    cutoff - CHECKPOINT_TOLERANCE, cutoff
                                )
                                & eligible["available_at"].between(
                                    cutoff - CHECKPOINT_TOLERANCE,
                                    cutoff + CHECKPOINT_TOLERANCE,
                                )
                            ]
                    label = (
                        "Closing · Final pre-game"
                        if dataset_type == 1 and not horizon
                        else f"{'Closing' if dataset_type == 1 else 'Intermediate'} · {horizon_label(horizon)}"
                    )
                    row = eligible.iloc[-1].to_dict() if not eligible.empty else {}
                    row.update(
                        {
                            "game_id": game["game_id"],
                            "family_key": family.key,
                            "model_label": family.label,
                            "kind": family.kind,
                            "target": target,
                            "dataset_type": dataset_type,
                            "checkpoint": label,
                            "checkpoint_order": (dataset_type - 1) * 3 + order,
                            "checkpoint_at": cutoff,
                            "checkpoint_horizon_minutes": horizon,
                            "available": not eligible.empty,
                            "source": "Live predictions"
                            if dataset_type == 1
                            else source,
                        }
                    )
                    for column in (
                        "home_team",
                        "away_team",
                        "home_points",
                        "away_points",
                        "status",
                        "tipoff_at",
                    ):
                        row[column] = game[column]
                    rows.append(row)
    return pd.DataFrame(rows)


def grade_history(
    frame: pd.DataFrame, *, min_edge: Mapping[int, float] | None = None
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    # Entirely missing checkpoint rows still have a complete, typed display schema.
    for column in ("prediction_value", "model_line", "bet_line_x2"):
        if column not in work:
            work[column] = float("nan")
    for column in ("home_points", "away_points"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = prediction_values(work)
    work["reference_line"] = pd.to_numeric(work["bet_line_x2"], errors="coerce") / 2
    work["predicted_value"] = work["predicted_total"].where(
        work["target"].ne(3), work["predicted_home_margin"]
    )
    work["comparison_edge"] = work["predicted_value"] - work["reference_line"]
    thresholds = DEFAULT_MIN_EDGE_BY_KIND if min_edge is None else min_edge
    work["min_edge"] = work["kind"].map(
        lambda kind: thresholds.get(int(kind), DEFAULT_MIN_EDGE_BY_KIND[int(kind)])
    )
    work["signal"] = [
        signal(edge, spread=target == 3, min_edge=threshold)
        for edge, target, threshold in zip(
            work["comparison_edge"], work["target"], work["min_edge"], strict=True
        )
    ]
    final = (
        work["status"].eq(1) & work["home_points"].notna() & work["away_points"].notna()
    )
    work["actual_value"] = (
        (work["home_points"] + work["away_points"])
        .where(work["target"].ne(3), work["home_points"] - work["away_points"])
        .where(final)
    )
    work["prediction_error"] = work["predicted_value"] - work["actual_value"]
    work["absolute_error"] = work["prediction_error"].abs()
    outcomes = []
    for _, row in work.iterrows():
        if not row["available"]:
            result = "MISSING"
        elif row["status"] == 2:
            result = "VOID"
        elif row["signal"] == "UNAVAILABLE":
            result = "NO LINE"
        elif row["signal"] == "NO EDGE":
            result = "NO BET"
        elif pd.isna(row["actual_value"]):
            result = "PENDING"
        else:
            actual_edge = row["actual_value"] - row["reference_line"]
            if actual_edge == 0:
                result = "PUSH"
            else:
                positive = row["signal"] in ("OVER", "HOME")
                result = "WIN" if (actual_edge > 0) == positive else "LOSS"
        outcomes.append(result)
    work["result"] = outcomes
    return work


def history_summary(
    frame: pd.DataFrame, *, by_checkpoint: bool = False
) -> pd.DataFrame:
    """Daily metrics per model, optionally separating the selected checkpoints."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["family_key"]
    if by_checkpoint:
        group_columns += ["checkpoint", "checkpoint_order"]
    for _, group in frame.groupby(group_columns, sort=False):
        counts = group["result"].value_counts()
        wins, losses = int(counts.get("WIN", 0)), int(counts.get("LOSS", 0))
        errors = group.loc[group["available"] & group["status"].eq(1), "absolute_error"]
        rows.append(
            {
                "Family": group["family_key"].iloc[0],
                "Kind": int(group["kind"].iloc[0]),
                "Model": group["model_label"].iloc[0],
                "Games": int(group.loc[errors.dropna().index, "game_id"].nunique()),
                "Available": int(group["available"].sum()),
                "Expected": len(group),
                "Wins": wins,
                "Losses": losses,
                "Pushes": int(counts.get("PUSH", 0)),
                "No bet": int(counts.get("NO BET", 0)),
                "Missing": int(counts.get("MISSING", 0)),
                "No line": int(counts.get("NO LINE", 0)),
                "Pending": int(counts.get("PENDING", 0)),
                "Void": int(counts.get("VOID", 0)),
                "Win rate": 100 * wins / (wins + losses)
                if wins + losses
                else float("nan"),
                "MAE (pts)": errors.mean(),
                "Scored predictions": int(errors.notna().sum()),
            }
        )
        if by_checkpoint:
            rows[-1]["Checkpoint"] = group["checkpoint"].iloc[0]
            rows[-1]["Checkpoint order"] = int(group["checkpoint_order"].iloc[0])
    order = ["Kind", "Model"] + (["Checkpoint order"] if by_checkpoint else [])
    return pd.DataFrame(rows).sort_values(order).reset_index(drop=True)
