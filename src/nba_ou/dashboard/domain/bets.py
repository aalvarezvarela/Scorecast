"""Bets: what the bet form proposes and how a bet is graded.

The grading here mirrors the ``app.bets_full`` view (the stored history is
graded there); both must agree. Lines follow the books' convention: a totals
bet names the total, a spread bet the HOME handicap (home -5.5 is ``-5.5``),
whichever side was taken. The prediction store keeps the spread as the
market-implied home margin instead (``bet_line_x2 / 2``, +5.5 for the same
line), hence ``handicap_from_margin``.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from nba_ou.postgre_db.app_aiven.schema import MARKET_SPREAD, MARKET_TOTALS

DEFAULT_STAKE = 10.0
DEFAULT_ODDS = 1.909  # -110
DEFAULT_BOOK = "bet365"
SPREAD_TARGET = 3

SIDE_LABELS = {
    MARKET_TOTALS: {1: "Over", -1: "Under"},
    MARKET_SPREAD: {1: "Home", -1: "Away"},
}
MARKET_LABELS = {MARKET_TOTALS: "Totals", MARKET_SPREAD: "Spread"}


def market_of_target(target: int) -> int:
    """Line error and total points bet the totals; spread error the spread."""
    return MARKET_SPREAD if int(target) == SPREAD_TARGET else MARKET_TOTALS


def handicap_from_margin(implied_home_margin: float) -> float:
    """The home handicap books print for an implied home margin (+3.5 -> -3.5)."""
    return 0.0 - float(implied_home_margin)


def proposed_line(target: int, bet_line_x2: float | None) -> float | None:
    """The line the form starts from: the stored bet365 line of the prediction."""
    if bet_line_x2 is None or (
        isinstance(bet_line_x2, float) and math.isnan(bet_line_x2)
    ):
        return None
    line = float(bet_line_x2) / 2
    return (
        handicap_from_margin(line)
        if market_of_target(target) == MARKET_SPREAD
        else line
    )


def newest_quote(rows: pd.DataFrame) -> pd.Series:
    """The newest prediction of a card's market with a stored line (the newest
    of all without one): its line and prices are the ones the card offers."""
    quoted = rows.loc[rows["bet_line_x2"].notna()]
    source = quoted if not quoted.empty else rows
    return source.sort_values("predicted_at", kind="stable").iloc[-1]


def followed_prediction(rows: pd.DataFrame, side: int) -> pd.Series | None:
    """The prediction backing ``side`` with the largest edge, None without one."""
    edges = pd.to_numeric(rows["bet_edge"], errors="coerce")
    backing = pd.to_numeric(rows["bet_side"], errors="coerce").eq(side) & edges.notna()
    if not backing.any():
        return None
    return rows.loc[backing].iloc[int(edges[backing].abs().to_numpy().argmax())]


def favored_side(rows: pd.DataFrame) -> int | None:
    """The side most predictions back, None on a tie."""
    sides = pd.to_numeric(rows["bet_side"], errors="coerce")
    over, under = int(sides.eq(1).sum()), int(sides.eq(-1).sum())
    if over == under:
        return None
    return 1 if over > under else -1


def american_to_decimal(price: float | None) -> float | None:
    """-110 -> 1.909, +150 -> 2.5; None for no price."""
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return None
    price = float(price)
    if -100 < price < 100:
        return None
    decimal = 1 + (100 / -price if price < 0 else price / 100)
    return round(decimal, 3)


def stored_odds(
    side: int, price_over_home: float | None, price_under_away: float | None
) -> float | None:
    """The stored bet365 price of the side as decimal odds, None without one."""
    return american_to_decimal(price_over_home if side == 1 else price_under_away)


def proposed_odds(
    side: int, price_over_home: float | None, price_under_away: float | None
) -> float:
    """The stored bet365 price of the side, as decimal odds (else -110)."""
    return stored_odds(side, price_over_home, price_under_away) or DEFAULT_ODDS


def default_stake(last_stake: float | None) -> float:
    """The user's last stake, or 10."""
    if last_stake is None or float(last_stake) <= 0:
        return DEFAULT_STAKE
    return float(last_stake)


def cover_margin(
    market: int, line: float, home_points: float | None, away_points: float | None
) -> float | None:
    """Positive: OVER/HOME covers; negative: UNDER/AWAY; 0: push."""
    if home_points is None or away_points is None:
        return None
    if any(isinstance(v, float) and math.isnan(v) for v in (home_points, away_points)):
        return None
    if market == MARKET_SPREAD:
        return float(home_points) - float(away_points) + float(line)
    return float(home_points) + float(away_points) - float(line)


def grade(
    *,
    market: int,
    side: int,
    line: float,
    home_points: float | None,
    away_points: float | None,
    game_status: int,
    voided: bool = False,
) -> str:
    """``pending``, ``win``, ``loss``, ``push`` or ``void``, as ``bets_full``."""
    if voided or game_status == 2:
        return "void"
    margin = cover_margin(market, line, home_points, away_points)
    if game_status != 1 or margin is None:
        return "pending"
    if margin == 0:
        return "push"
    return "win" if (margin > 0) == (side > 0) else "loss"


def profit(result: str, stake: float, odds_decimal: float) -> float | None:
    """Net return: stake x (odds - 1) won, the stake lost; None while pending."""
    if result == "pending":
        return None
    if result == "win":
        value = Decimal(str(stake)) * (Decimal(str(odds_decimal)) - 1)
        return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if result == "loss":
        return -float(stake)
    return 0.0


def describe_bet(market: int, side: int, line: float, home: str, away: str) -> str:
    """``Over 224.5``, ``Boston Celtics -3.5``, ``New York Knicks +3.5``."""
    if market == MARKET_TOTALS:
        return f"{SIDE_LABELS[market][side]} {line:g}"
    team, handicap = (home, line) if side == 1 else (away, -line)
    return f"{team} {handicap:+g}"


def summary(bets) -> dict[str, float]:
    """Staked, profit and ROI over settled bets, wins, losses, pushes, pending."""
    settled = bets[bets["result"].isin(["win", "loss", "push"])]
    staked = float(settled["stake"].astype(float).sum())
    won = float(settled["profit"].astype(float).sum())
    return {
        "bets": int(len(bets)),
        "pending": int(bets["result"].eq("pending").sum()),
        "wins": int(settled["result"].eq("win").sum()),
        "losses": int(settled["result"].eq("loss").sum()),
        "pushes": int(settled["result"].eq("push").sum()),
        "staked": staked,
        "profit": won,
        "roi": won / staked if staked else 0.0,
    }
