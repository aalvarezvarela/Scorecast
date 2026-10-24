"""Source comparisons only; matching SBR is never an ingestion condition."""

from pathlib import Path

import pandas as pd

from nba_ou.postgre_db.line_history_aiven.fetch import (
    DEFAULT_MIN_MINUTES_BEFORE_TIP,
    TICK_COLUMNS,
    _decode_ticks,
)
from nba_ou.postgre_db.line_history_aiven.schema import MARKETS

KEY = ["game_id", "market", "book"]
QUOTE = ["left_line", "left_price", "right_line", "right_price"]


def ticks_from_dir(directory: Path) -> pd.DataFrame:
    rows = pd.read_parquet(directory / "lh_line.parquet")
    rows = rows[
        rows["is_pregame"].astype(bool)
        & (rows["mins_to_tip"] <= -DEFAULT_MIN_MINUTES_BEFORE_TIP)
    ].copy()
    rows["market"] = rows["market_id"].map(dict(MARKETS))
    rows["book"] = rows["book_id"].map({1: "bet365"})
    rows["minutes_before_tip"] = rows["mins_to_tip"]
    return _decode_ticks(rows[TICK_COLUMNS].reset_index(drop=True))


def asof(ticks: pd.DataFrame, minutes: int) -> pd.DataFrame:
    # Select before testing availability. Dropping null rows first would
    # resurrect the quote before a suspension.
    seen = ticks[
        ticks["minutes_before_tip"] >= max(minutes, DEFAULT_MIN_MINUTES_BEFORE_TIP)
    ]
    return (
        seen.sort_values("line_ts")
        .groupby(KEY)
        .tail(1)
        .set_index(KEY)[QUOTE + ["minutes_before_tip"]]
    )


def snapshot_comparison(
    movi: pd.DataFrame, sbr: pd.DataFrame, horizons=(0, 30, 60, 240, 720, 1080)
) -> pd.DataFrame:
    frames = []
    for horizon in horizons:
        joined = asof(movi, horizon).join(
            asof(sbr, horizon), how="outer", lsuffix="_moviodds", rsuffix="_sbr"
        )
        market = joined.index.get_level_values("market")
        for source in ("moviodds", "sbr"):
            available = (
                joined[[f"{s}_price_{source}" for s in ("left", "right")]]
                .notna()
                .all(axis=1)
            )
            available &= (market == "money_line") | joined[
                [f"{s}_line_{source}" for s in ("left", "right")]
            ].notna().all(axis=1)
            joined[f"{source}_available"] = available
        both = joined["moviodds_available"] & joined["sbr_available"]
        for kind in ("line", "price"):
            same = pd.Series(True, index=joined.index)
            for side in ("left", "right"):
                same &= (
                    joined[f"{side}_{kind}_moviodds"]
                    .eq(joined[f"{side}_{kind}_sbr"])
                    .fillna(False)
                )
            if kind == "line":
                same |= market == "money_line"
            joined[f"same_{kind}s"] = (both & same).astype("boolean").where(both)
        joined["same_quote"] = joined["same_lines"] & joined["same_prices"]
        joined["same_lines"] = joined["same_lines"].where(market != "money_line")
        joined["snapshot_minutes"] = horizon
        frames.append(joined.reset_index())
    return pd.concat(frames, ignore_index=True)


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    return (
        comparison.groupby(["snapshot_minutes", "market"])
        .agg(
            games=("game_id", "nunique"),
            moviodds_available=("moviodds_available", "sum"),
            sbr_available=("sbr_available", "sum"),
            compared=("same_quote", "count"),
            same_lines=("same_lines", lambda values: values.sum(min_count=1)),
            same_prices=("same_prices", "sum"),
            same_quote=("same_quote", "sum"),
            moviodds_age_minutes=("minutes_before_tip_moviodds", "median"),
            sbr_age_minutes=("minutes_before_tip_sbr", "median"),
        )
        .reset_index()
    )
