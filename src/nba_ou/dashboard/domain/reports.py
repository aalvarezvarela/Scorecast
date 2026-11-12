"""The injury-report flag of an upcoming game.

The slate's games carry ``home_report_filed`` / ``away_report_filed`` (the
injury-report store at the moment the page read it) when that store is
readable. A team that has not filed matters: the closing models and the
intermediate ones near tip-off skip its game, and the longer horizons predict
it without injury information.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SIDES = ("away", "home")


@dataclass(frozen=True)
class ReportFlag:
    """``missing``: the sides (``away``, ``home``) whose report is not filed."""

    missing: tuple[str, ...]
    read_at: pd.Timestamp | None

    @property
    def filed(self) -> bool:
        return not self.missing


def report_flag(game, now) -> ReportFlag | None:
    """None when there is nothing to flag: the filing state is unknown, or the
    game has started, finished or been postponed."""
    filed = {side: game.get(f"{side}_report_filed") for side in SIDES}
    if any(value is None or pd.isna(value) for value in filed.values()):
        return None
    if game["status"] != 0 or pd.Timestamp(game["tipoff_at"]) <= pd.Timestamp(now):
        return None
    read_at = game.get("reports_read_at")
    return ReportFlag(
        missing=tuple(side for side in SIDES if not bool(filed[side])),
        read_at=None if read_at is None or pd.isna(read_at) else pd.Timestamp(read_at),
    )
