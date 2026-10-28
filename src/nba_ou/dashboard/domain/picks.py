"""Reduce a game's model signals to one plain betting instruction."""

import math
from dataclasses import dataclass, field

import pandas as pd

DIRECTIONAL = ("OVER", "UNDER", "HOME", "AWAY")


@dataclass(frozen=True)
class GamePick:
    """``status``: BET, TIE, NO BET, NO LINE, PENDING."""

    status: str
    side: str | None = None
    agreeing: int = 0
    predictions: int = 0
    counts: dict[str, int] = field(default_factory=dict)


def game_pick(predictions: pd.DataFrame) -> GamePick:
    """Follow the simple majority of directional signals; a tie is no bet.

    A prediction without a market line cannot vote; one below its minimum edge
    abstains, and ``agreeing`` out of ``predictions`` discloses it.
    """
    if predictions.empty:
        return GamePick("PENDING")
    signals = predictions["signal"]
    available = int(signals.ne("UNAVAILABLE").sum())
    if not available:
        return GamePick("NO LINE", predictions=len(signals))
    counts = {
        side: int(signals.eq(side).sum())
        for side in DIRECTIONAL
        if signals.eq(side).any()
    }
    if not counts:
        return GamePick("NO BET", predictions=available)
    agreeing = max(counts.values())
    winners = [side for side, votes in counts.items() if votes == agreeing]
    if len(winners) > 1:
        return GamePick("TIE", predictions=available, counts=counts)
    side = winners[0]
    return GamePick("BET", side, agreeing, available, counts)


def bet_text(side: str, line, *, home: str, away: str) -> str:
    """``OVER 230.5``; spread lines are implied home margins, so the home
    handicap is the line negated (margin 3.5 -> ``Celtics -3.5``)."""
    if line is None or pd.isna(line) or not math.isfinite(float(line)):
        return side
    line = float(line)
    if side in ("OVER", "UNDER"):
        return f"{side} {line:.1f}"
    handicap = -line if side == "HOME" else line
    team = home if side == "HOME" else away
    return f"{team} {'PK' if handicap == 0 else f'{handicap:+.1f}'}"
