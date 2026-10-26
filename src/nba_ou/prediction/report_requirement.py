"""Which intermediate models skip a game whose injury reports are not filed.

Near tip-off the training rows almost always had both teams' report filed
(2_5 intermediate dataset, 2021-22 on: 99.6-100% from T-240 in, 97.3% at
T-300; 81% at T-360 and ~73% from T-420 out). A model of those horizons has
hardly seen a row without one, so it skips such a game rather than predict it
(``[PredictionModels] REPORT_REQUIRED_MAX_HORIZON``). Longer horizons trained
on such rows keep predicting them, their injury features NaN as in training.

The closing frame leaves such games out itself (``games_not_updated``); this
is the intermediate counterpart, read per row at the row's own snapshot: the
run's moment for the latest input, the snapshot time for the own one.
"""

from __future__ import annotations

import pandas as pd
from nba_ou.data_processing.injury_status.report_state import COVERED_COL

#: Per side, whether the team's report was filed at the row's snapshot.
COVERED_COLUMNS = (f"{COVERED_COL}_TEAM_HOME", f"{COVERED_COL}_TEAM_AWAY")
#: Set on a slot's output: the games it skipped for a missing report.
NO_REPORT_ATTR = "no_report_games"


def requires_report(horizon_minutes: int | None, max_horizon: int | None) -> bool:
    """Whether a model of ``horizon_minutes`` needs both reports filed.

    A pooled model (``horizon_minutes is None``) trained on every horizon,
    filed or not, and never does; nor does any model when ``max_horizon`` is
    None.
    """
    if horizon_minutes is None or max_horizon is None:
        return False
    return int(horizon_minutes) <= int(max_horizon)


def reports_filed(rows: pd.DataFrame) -> pd.Series:
    """Per row, whether both teams had filed; a missing flag reads as not filed."""
    missing = [column for column in COVERED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(
            f"the intermediate rows carry no injury-report coverage ({missing})"
        )
    flags = rows[list(COVERED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    return flags.eq(1).all(axis=1)
