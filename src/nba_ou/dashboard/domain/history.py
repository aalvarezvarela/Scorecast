"""Historical checkpoints and grading against each prediction's stored quote."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from nba_ou.dashboard.config import (
    DEFAULT_MIN_EDGE_BY_KIND,
    YESTERDAY_INTERMEDIATE_CORE,
    YESTERDAY_INTERMEDIATE_EXTRA,
)
from nba_ou.dashboard.domain.comparisons import prediction_values
from nba_ou.dashboard.domain.dates import horizon_label, utc_timestamp
from nba_ou.dashboard.domain.models import (
    displayed_models,
    model_families,
    with_family_metadata,
)
from nba_ou.dashboard.domain.teams import full_team_name

HISTORY_SOURCES = ("Live predictions", "Snapshot replay")
CHECKPOINT_TOLERANCE = pd.Timedelta(minutes=60)
# Core checkpoints are listed for every family, so a missing slot stays
# visible; extra ones only when the family has a prediction there on the slate.
CLOSING_CHECKPOINTS = (480, 360, 240, 0)
INTERMEDIATE_CHECKPOINTS = YESTERDAY_INTERMEDIATE_CORE
INTERMEDIATE_EXTRA_CHECKPOINTS = YESTERDAY_INTERMEDIATE_EXTRA
# TabPFN runs on its own schedule: its core is the original three intermediate
# checkpoints, and every other horizon it stored is an extra.
TABPFN_INTERMEDIATE_CHECKPOINTS = (480, 360, 240)
TABPFN_CLOSING_EXTRA_CHECKPOINTS = (720, 600, 120)


def _checkpoint_grid(
    family, dataset_type: int, candidates: pd.DataFrame
) -> list[tuple[int, bool]]:
    """``(horizon, core)`` pairs, longest horizon first and the close last."""
    tabpfn = family.kind == 3
    if dataset_type == 1:
        core = CLOSING_CHECKPOINTS
        extra = TABPFN_CLOSING_EXTRA_CHECKPOINTS if tabpfn else ()
    elif tabpfn:
        core = TABPFN_INTERMEDIATE_CHECKPOINTS
        horizons = pd.to_numeric(
            candidates.loc[candidates["dataset_type"].eq(2), "model_horizon_minutes"],
            errors="coerce",
        ).dropna()
        extra = tuple(int(h) for h in horizons.unique() if h > 0)
    else:
        core = INTERMEDIATE_CHECKPOINTS
        extra = INTERMEDIATE_EXTRA_CHECKPOINTS
    grid = {int(h): True for h in core}
    for horizon in extra:
        grid.setdefault(int(horizon), False)
    return sorted(grid.items(), key=lambda item: -item[0])


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

    Live checkpoints select the nearest input within one hour on either side,
    with publication also within that window. Final live closing remains the
    last completed pre-game output. Replay uses exact reconstructed snapshots
    for closing and intermediate models, allowing post-game publication.
    """
    if source not in HISTORY_SOURCES:
        raise ValueError("Unknown historical prediction source")
    if frame.empty or games.empty:
        return pd.DataFrame()
    work = displayed_models(frame)
    work = work.loc[
        work["target"].eq(target) & work["workflow"].isin([1, 2, 3, 4])
    ].copy()
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
    grids = {
        (family.key, dataset_type): _checkpoint_grid(
            family, dataset_type, work.loc[work["family_key"].eq(family.key)]
        )
        for family in families
        for dataset_type in (1, 2)
    }
    rows = []
    for _, game in games.iterrows():
        tipoff = game["tipoff_at"]
        for family in families:
            candidates = work.loc[
                work["game_id"].eq(game["game_id"]) & work["family_key"].eq(family.key)
            ]
            for dataset_type in (1, 2):
                if not (family.closing if dataset_type == 1 else family.intermediate):
                    continue
                for horizon, core in grids[family.key, dataset_type]:
                    cutoff = tipoff - pd.Timedelta(minutes=horizon)
                    eligible = candidates.loc[
                        candidates["dataset_type"].eq(dataset_type)
                    ]
                    if dataset_type == 1 and source == "Snapshot replay":
                        # Past-snapshot reconstructions (replay mode) and
                        # historical backfill refits, never live outputs.
                        eligible = eligible.loc[
                            (
                                (
                                    eligible["workflow"].eq(2)
                                    & eligible["input_mode"].eq(3)
                                )
                                | (
                                    eligible["workflow"].eq(4)
                                    & eligible["input_mode"].eq(0)
                                )
                            )
                            & eligible["model_horizon_minutes"].eq(0)
                            & eligible["snapshot_horizon_minutes"].eq(horizon)
                        ]
                    elif dataset_type == 1:
                        eligible = eligible.loc[
                            eligible["workflow"].isin([1, 3])
                            & eligible["input_mode"].eq(0)
                            & eligible["model_horizon_minutes"].eq(0)
                            & eligible["time_to_tipoff_seconds"].gt(0)
                            & eligible["available_at"].lt(tipoff)
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
                                    eligible["workflow"].isin([2, 4])
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
                            ]
                    if source == "Live predictions" and horizon:
                        eligible = eligible.loc[
                            eligible["input_at"].between(
                                cutoff - CHECKPOINT_TOLERANCE,
                                cutoff + CHECKPOINT_TOLERANCE,
                            )
                            & eligible["available_at"].between(
                                cutoff - CHECKPOINT_TOLERANCE,
                                cutoff + CHECKPOINT_TOLERANCE,
                            )
                        ]
                        # The stable initial sort breaks equal-distance ties in
                        # favour of the latest completed output/build.
                        eligible = eligible.assign(
                            _distance=(eligible["input_at"] - cutoff).abs()
                        ).sort_values("_distance", ascending=False, kind="stable")
                    label = (
                        "Closing · Final pre-game"
                        if dataset_type == 1 and not horizon
                        else f"{'Closing' if dataset_type == 1 else 'Intermediate'} · {horizon_label(horizon)}"
                    )
                    row = (
                        eligible.iloc[-1]
                        .drop(labels=["_distance"], errors="ignore")
                        .to_dict()
                        if not eligible.empty
                        else {}
                    )
                    row.update(
                        {
                            "game_id": game["game_id"],
                            "family_key": family.key,
                            "model_label": family.label,
                            "kind": family.kind,
                            "target": target,
                            "dataset_type": dataset_type,
                            "checkpoint": label,
                            # Closing first, then the longest horizon first.
                            "checkpoint_order": dataset_type * 10_000 - horizon,
                            "_core_checkpoint": core,
                            "checkpoint_at": cutoff,
                            "checkpoint_horizon_minutes": horizon,
                            "available": not eligible.empty,
                            "source": source,
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
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # An extra checkpoint is listed only where its family has a prediction.
    shown = result["_core_checkpoint"] | result.groupby(["family_key", "checkpoint"])[
        "available"
    ].transform("any")
    return result.loc[shown].drop(columns="_core_checkpoint").reset_index(drop=True)


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
    # Baselines (kind 2) have no default; callers pass their threshold.
    work["min_edge"] = work["kind"].map(
        lambda kind: (
            thresholds[int(kind)]
            if int(kind) in thresholds
            else DEFAULT_MIN_EDGE_BY_KIND[int(kind)]
        )
    )
    if not (np.isfinite(work["min_edge"]) & work["min_edge"].ge(0)).all():
        raise ValueError("min_edge must be a finite, non-negative number of points")
    edge = work["comparison_edge"]
    neutral = edge.abs().le(work["min_edge"]) | np.isclose(
        edge.abs(), work["min_edge"], rtol=0, atol=1e-9
    )
    work["signal"] = np.select(
        [~np.isfinite(edge), neutral, work["target"].eq(3), edge.gt(0)],
        ["UNAVAILABLE", "NO EDGE", np.where(edge.gt(0), "HOME", "AWAY"), "OVER"],
        default="UNDER",
    )
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
    actual_edge = work["actual_value"] - work["reference_line"]
    positive = work["signal"].isin(["OVER", "HOME"])
    work["result"] = np.select(
        [
            ~work["available"],
            work["status"].eq(2),
            work["signal"].eq("UNAVAILABLE"),
            work["signal"].eq("NO EDGE"),
            work["actual_value"].isna(),
            actual_edge.eq(0),
            actual_edge.gt(0).eq(positive),
        ],
        ["MISSING", "VOID", "NO LINE", "NO BET", "PENDING", "PUSH", "WIN"],
        default="LOSS",
    )
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
                "Win rate": (
                    100 * wins / (wins + losses) if wins + losses else float("nan")
                ),
                "MAE (pts)": errors.mean(),
                "Scored predictions": int(errors.notna().sum()),
            }
        )
        if by_checkpoint:
            rows[-1]["Checkpoint"] = group["checkpoint"].iloc[0]
            rows[-1]["Checkpoint order"] = int(group["checkpoint_order"].iloc[0])
    order = ["Kind", "Model"] + (["Checkpoint order"] if by_checkpoint else [])
    return pd.DataFrame(rows).sort_values(order).reset_index(drop=True)


def pregame_line(rows: pd.DataFrame) -> float | None:
    """The stored quote of a game's final pre-game checkpoint.

    Registry models first, then TabPFN; ``None`` when no final checkpoint has a
    line. Used only to describe the result: picks keep their own row's quote.
    """
    if rows.empty or "reference_line" not in rows:
        return None
    final = rows.loc[
        rows["checkpoint"].eq("Closing · Final pre-game")
        & rows["available"].astype(bool)
        & np.isfinite(pd.to_numeric(rows["reference_line"], errors="coerce"))
    ].sort_values("kind", kind="stable")
    return None if final.empty else float(final["reference_line"].iloc[0])
