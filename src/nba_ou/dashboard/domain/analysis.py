"""Historical cells and pooled statistics; see docs/serving/historical_analysis.md."""

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd

from nba_ou.dashboard.domain.dates import horizon_label
from nba_ou.dashboard.domain.history import grade_history
from nba_ou.dashboard.domain.models import with_family_metadata

# Lightweight copy of the serving snapshot contract (checked in tests). Do not
# import the training/market-data stack into the dashboard.
TIME_GRID = (
    30,
    60,
    120,
    180,
    240,
    300,
    360,
    420,
    480,
    540,
    600,
    660,
    720,
    840,
    960,
    1080,
)
EARLY_BUCKET = 1081
FINAL_BUCKET = 0
TIME_BUCKETS = (EARLY_BUCKET, *reversed(TIME_GRID), FINAL_BUCKET)
BREAK_EVEN = 100 * 110 / 210
MIN_CELL_BETS = 30
SEASON_TYPES = {
    2: "Regular season",
    4: "Playoffs",
    5: "Play-in",
    1: "Preseason",
    3: "All-Star",
}
CELL_KEYS = (
    "family_key",
    "model_label",
    "dataset_type",
    "model_horizon_minutes",
    "source",
    "bucket",
)


def season_start(day: date) -> date:
    return date(day.year if day.month >= 10 else day.year - 1, 10, 1)


def time_bucket(minutes: float) -> int | None:
    if not np.isfinite(minutes) or minutes <= 0:
        return None
    return next((h for h in TIME_GRID if minutes <= h), EARLY_BUCKET)


def bucket_label(bucket: int) -> str:
    if bucket == FINAL_BUCKET:
        return "Final"
    return "T−18h+" if bucket == EARLY_BUCKET else horizon_label(bucket)


def wilson_interval(wins: int, losses: int) -> tuple[float, float]:
    n = wins + losses
    if not n:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = wins / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (center - radius), 100 * (center + radius)


def prepare_analysis(
    frame: pd.DataFrame, *, min_edge: Mapping[int, float] | None = None
) -> pd.DataFrame:
    """Grade SQL-reduced rows using exactly Yesterday's stored-quote policy."""
    if frame.empty:
        return frame.copy()
    work = with_family_metadata(frame)
    work["available"] = True
    work = grade_history(work, min_edge=min_edge)
    work["input_at"] = pd.to_datetime(work["input_at"], utc=True)
    work["available_at"] = pd.to_datetime(work["available_at"], utc=True)
    work["publication_lag_minutes"] = (
        work["available_at"] - work["input_at"]
    ).dt.total_seconds() / 60
    work["on_horizon"] = work["dataset_type"].eq(2) & work["bucket"].eq(
        work["model_horizon_minutes"]
    )
    work["time_label"] = work["bucket"].map(bucket_label)
    work["slot_label"] = np.where(
        work["dataset_type"].eq(1),
        "Closing",
        work["model_horizon_minutes"].map(horizon_label),
    )
    positive = work["signal"].isin(["OVER", "HOME"])
    price = pd.Series(np.nan, index=work.index)
    for side, mask in (("over_home", positive), ("under_away", ~positive)):
        column = f"bet_price_{side}"
        if column in work:
            price.loc[mask] = pd.to_numeric(work.loc[mask, column], errors="coerce")
    work["bet_price"] = price
    work["priced_bet"] = (
        price.abs().ge(100)
        & np.isfinite(price)
        & work["result"].isin(["WIN", "LOSS", "PUSH"])
    )
    profit = pd.Series(
        np.where(price.gt(0), price / 100, 100 / price.abs()), index=work.index
    )
    work["unit_profit"] = np.select(
        [work["result"].eq("WIN"), work["result"].eq("LOSS")],
        [profit, -1.0],
        default=0.0,
    )
    work["unit_profit"] = work["unit_profit"].where(work["priced_bet"])
    return work


def aggregate_cells(
    frame: pd.DataFrame,
    by: Sequence[str] = CELL_KEYS,
    *,
    settled_games_total: int | None = None,
) -> pd.DataFrame:
    """Pool counts/errors, never average win percentages or game-level MAEs.

    A pool may contain repeated game observations (e.g. one per horizon). Games
    remain distinct; the UI must label the number of prediction rows separately.
    """
    if frame.empty:
        return pd.DataFrame()
    rows = []
    groups = frame.groupby(list(by), sort=False, dropna=False) if by else [((), frame)]
    for keys, group in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys, strict=True))
        counts = group["result"].value_counts()
        wins, losses, pushes = (
            int(counts.get(result, 0)) for result in ("WIN", "LOSS", "PUSH")
        )
        settled = group["actual_value"].notna() & group["status"].eq(1)
        expected = (
            int(settled.sum())
            if settled_games_total is None
            else (settled_games_total * len(group[list(CELL_KEYS)].drop_duplicates()))
        )
        bets = wins + losses + pushes
        low, high = wilson_interval(wins, losses)
        row.update(
            games=int(group["game_id"].nunique()),
            settled_games=int(group.loc[settled, "game_id"].nunique()),
            rows=len(group),
            settled_rows=int(settled.sum()),
            wins=wins,
            losses=losses,
            pushes=pushes,
            bets=bets,
            win_pct=100 * wins / (wins + losses) if wins + losses else np.nan,
            ci_low=low,
            ci_high=high,
            coverage=100 * bets / expected if expected else np.nan,
            mae=group.loc[settled & group["available"], "absolute_error"].mean(),
            median_lag_minutes=group["publication_lag_minutes"].median(),
            priced_bets=int(group["priced_bet"].sum()),
            roi=100 * group["unit_profit"].sum() / group["priced_bet"].sum()
            if group["priced_bet"].any()
            else np.nan,
            no_bet=int(counts.get("NO BET", 0)),
            no_line=int(counts.get("NO LINE", 0)),
            pending=int(counts.get("PENDING", 0)),
            void=int(counts.get("VOID", 0)),
            low_sample=wins + losses < MIN_CELL_BETS,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_snapshot_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare live/replay on the same games/slots, not different coverages."""
    diagonal = frame.loc[frame["on_horizon"]] if not frame.empty else frame
    if diagonal.empty:
        return pd.DataFrame()
    keys = [
        "game_id",
        "family_key",
        "target",
        "dataset_type",
        "model_horizon_minutes",
        "bucket",
    ]
    # Only settled, directional bets in BOTH sources can inform a paired win-rate delta.
    settled = diagonal.loc[diagonal["result"].isin(["WIN", "LOSS"])]
    live = settled.loc[settled["source"].eq("Live predictions")]
    replay = settled.loc[settled["source"].eq("Snapshot replay")]
    pairs = live.merge(
        replay, on=keys, suffixes=("_live", "_replay"), validate="one_to_one"
    )
    if pairs.empty:
        return pd.DataFrame()
    rows = []
    for (family, horizon), group in pairs.groupby(
        ["model_label_live", "model_horizon_minutes"], sort=False
    ):
        live_wins = group["result_live"].eq("WIN")
        replay_wins = group["result_replay"].eq("WIN")
        rows.append(
            {
                "Model": family,
                "Horizon": horizon_label(horizon),
                "Matched games": len(group),
                "Live win %": 100 * live_wins.mean(),
                "Replay win %": 100 * replay_wins.mean(),
                "Replay − live (pp)": 100 * (replay_wins.mean() - live_wins.mean()),
                "Different picks": int(
                    group["signal_live"].ne(group["signal_replay"]).sum()
                ),
            }
        )
    return pd.DataFrame(rows)
