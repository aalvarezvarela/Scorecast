"""The bet history: money staked, returned and won, overall and by group.

Works on the rows of ``app.bets_full`` (already graded there). Money figures
count **settled** bets only (win, loss, push): a pending bet has no outcome
yet and a void one (postponed game) returns its stake. Pending bets are
reported apart, with what they have at stake.
"""

from __future__ import annotations

import pandas as pd

from nba_ou.dashboard.config import MODEL_NAMES
from nba_ou.dashboard.domain.bets import MARKET_LABELS, SIDE_LABELS
from nba_ou.postgre_db.app_aiven.schema import MARKET_SPREAD, MARKET_TOTALS

SETTLED = ("win", "loss", "push")
ALL_MARKETS = 0
MARKET_FILTERS = {ALL_MARKETS: "All", MARKET_TOTALS: "Totals", MARKET_SPREAD: "Spread"}
NO_MODEL = "No model"
#: The groupings the history page offers: label -> the column it groups by.
GROUPINGS = {
    "Market and side": "side_label",
    "Model followed": "model_label",
    "Month": "month",
}


def _model_label(kind, dataset_type) -> str:
    if pd.isna(kind):
        return NO_MODEL
    name = MODEL_NAMES.get(int(kind), "Model")
    return f"{name} {'closing' if int(dataset_type) == 1 else 'intermediate'}"


def prepare_history(bets: pd.DataFrame) -> pd.DataFrame:
    """``bets`` with numeric money columns and the labels it is grouped by."""
    history = bets.copy()
    for column in ("stake", "odds_decimal", "profit", "line"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history["game_date"] = pd.to_datetime(history["game_date"]).dt.date
    history["side_label"] = [
        f"{MARKET_LABELS[int(market)]} · {SIDE_LABELS[int(market)][int(side)]}"
        for market, side in zip(history["market"], history["side"], strict=True)
    ]
    history["model_label"] = [
        _model_label(kind, dataset_type)
        for kind, dataset_type in zip(
            history.get("model_kind", pd.Series(pd.NA, index=history.index)),
            history.get("model_dataset_type", pd.Series(pd.NA, index=history.index)),
            strict=True,
        )
    ]
    history["month"] = pd.to_datetime(history["game_date"]).dt.strftime("%Y-%m")
    return history


def filter_history(
    history: pd.DataFrame, start=None, end=None, market: int = ALL_MARKETS
) -> pd.DataFrame:
    """The bets on games from ``start`` to ``end`` (inclusive) of ``market``."""
    keep = pd.Series(True, index=history.index)
    if start is not None:
        keep &= history["game_date"] >= start
    if end is not None:
        keep &= history["game_date"] <= end
    if market != ALL_MARKETS:
        keep &= history["market"].astype(int).eq(market)
    return history.loc[keep]


def performance(history: pd.DataFrame) -> dict[str, float]:
    """Staked, returned, profit and ROI over the settled bets; the win rate
    over wins and losses; the pending bets and their stake apart."""
    settled = history.loc[history["result"].isin(SETTLED)]
    pending = history.loc[history["result"].eq("pending")]
    staked = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    wins = int(settled["result"].eq("win").sum())
    losses = int(settled["result"].eq("loss").sum())
    decided = wins + losses
    return {
        "bets": int(len(history)),
        "settled": int(len(settled)),
        "wins": wins,
        "losses": losses,
        "pushes": int(settled["result"].eq("push").sum()),
        "staked": staked,
        "returned": staked + profit,
        "profit": profit,
        "roi": profit / staked if staked else 0.0,
        "win_rate": wins / decided if decided else 0.0,
        "average_odds": float(settled["odds_decimal"].mean()) if len(settled) else 0.0,
        "pending": int(len(pending)),
        "pending_stake": float(pending["stake"].sum()),
    }


def daily_profit(history: pd.DataFrame) -> pd.DataFrame:
    """Per slate with a settled bet: its bets, stake and profit, and the
    running profit and stake since the first one."""
    settled = history.loc[history["result"].isin(SETTLED)]
    columns = ["game_date", "bets", "staked", "profit", "total_profit", "total_staked"]
    if settled.empty:
        return pd.DataFrame(columns=columns)
    days = (
        settled.groupby("game_date")
        .agg(bets=("bet_id", "size"), staked=("stake", "sum"), profit=("profit", "sum"))
        .sort_index()
        .reset_index()
    )
    days["total_profit"] = days["profit"].cumsum()
    days["total_staked"] = days["staked"].cumsum()
    return days[columns]


def breakdown(history: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per value of ``column``: record, stake, profit and ROI of its
    settled bets, and its pending ones."""
    rows = []
    for label, group in history.groupby(column, sort=True):
        totals = performance(group)
        rows.append(
            {
                "Group": label,
                "Bets": totals["settled"],
                "W-L-P": f"{totals['wins']}-{totals['losses']}-{totals['pushes']}",
                "Win %": totals["win_rate"] * 100 if totals["settled"] else None,
                "Staked (€)": totals["staked"],
                "Profit (€)": totals["profit"],
                "ROI %": totals["roi"] * 100 if totals["staked"] else None,
                "Pending": totals["pending"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Group",
            "Bets",
            "W-L-P",
            "Win %",
            "Staked (€)",
            "Profit (€)",
            "ROI %",
            "Pending",
        ],
    )
