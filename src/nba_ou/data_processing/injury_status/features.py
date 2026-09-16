"""Attach report counters and per-status history features to team-game rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .status_history import add_status_features
from .report_state import COVERED_COL, InjuryReportState, report_counter_features


def add_injury_report_features(
    df_team: pd.DataFrame,
    state: InjuryReportState,
    box: pd.DataFrame,
    *,
    status_top_n: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return ``df_team`` with the report counters and per-status columns.

    ``status_top_n`` maps status to the number of per-player slots; default
    ``status_history.DEFAULT_TOP_N`` (Questionable 2, Probable 1, Doubtful 1).

    Row order and index are preserved. Needs ``GAME_ID``, ``TEAM_ID``,
    ``GAME_DATE`` and ``SEASON_YEAR``.
    """
    positional = df_team.reset_index(drop=True)
    counters = report_counter_features(positional, state)
    statuses = add_status_features(positional, state, box, top_n=status_top_n)
    added = pd.concat([counters, statuses], axis=1)
    added.index = df_team.index
    overlap = [c for c in added.columns if c in df_team.columns]
    return pd.concat([df_team.drop(columns=overlap), added], axis=1)


def mask_uncovered_group_columns(
    df_merged: pd.DataFrame,
    out_prefix: str,
    *,
    covered_col: str = COVERED_COL,
) -> pd.DataFrame:
    """NaN the ``{out_prefix}_{SIDE}_*`` columns wherever that side is uncovered.

    ``add_top3_availability_effect_features_for_columns`` is coverage-blind by
    design: it fills a missing effect aggregate with 0, the correct limit of an
    estimator that shrinks toward zero, and reports 0 sample games. For the
    questionable group that limit is only correct when the report says nobody is
    at risk. On a team-game no report covered there is no questionable group at
    all, and 0 would assert "nobody at risk" from an absence of information --
    the distinction rule 4 of :mod:`.report_state` exists to keep.

    Both sides are masked independently, since one team can file while its
    opponent has not.
    """
    for side in ("HOME", "AWAY"):
        columns = [c for c in df_merged.columns if c.startswith(f"{out_prefix}_{side}_")]
        flag = f"{covered_col}_TEAM_{side}"
        if not columns or flag not in df_merged.columns:
            continue
        uncovered = pd.to_numeric(df_merged[flag], errors="coerce").fillna(0) != 1
        df_merged.loc[uncovered, columns] = np.nan
    return df_merged
