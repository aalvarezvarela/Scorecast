"""Where in the season a game sits, so CV and the holdout can be compared.

Motivation, measured rather than assumed. Pooled over 36 runs' CV folds at the
0.1-point threshold, betting win rate was **52.1% on games played Oct-Dec** and
**53.5% on games played Jan-Apr**. Every holdout to date is the *tail* of the
data (60 days ending wherever the CSV ends, i.e. Feb-Apr), so it contains no
Oct-Dec games at all, while CV spans the whole calendar. Roughly half of the
persistent "CV looks worse than the holdout" gap is that mixture difference and
not a property of any model.

So a pooled CV number and a holdout number are not measuring the same
population. This module exists to make the phase-matched comparison available
alongside the pooled one, from a single definition both sides import -- the
repo previously had no season-phase concept anywhere, only
``walk_forward.exclude_test_months``.

The phase boundaries are calendar months, which is what the data supports: a
game's month is always known, whereas "game N of the season" is not recorded and
would have to be re-derived per season. October-November is the stretch where
rolling team features are still filling up; December-January is mid-season;
February onward is the run-in, when every ``_BEFORE`` average is saturated.

May and June are labelled ``playoffs`` for completeness. They are normally
absent -- ``data.exclude_playoffs`` drops them and
``walk_forward.exclude_test_months`` keeps them out of validation windows -- but
a run that deliberately keeps playoffs should not silently get ``unknown``.
"""

from __future__ import annotations

import pandas as pd

#: Month -> phase. Deliberately exhaustive over 1-12 so no month can fall
#: through to a null and be quietly dropped from a phase-matched subset.
_MONTH_TO_PHASE: dict[int, str] = {
    10: "early",
    11: "early",
    12: "mid",
    1: "mid",
    2: "late",
    3: "late",
    4: "late",
    5: "playoffs",
    6: "playoffs",
    # The NBA does not play these; they exist so the mapping is total.
    7: "offseason",
    8: "offseason",
    9: "offseason",
}

#: Declared order, for readable groupby output.
PHASE_ORDER: tuple[str, ...] = ("early", "mid", "late", "playoffs", "offseason")


def month_to_phase(month: int) -> str:
    """Phase label for a calendar month (1-12)."""
    try:
        return _MONTH_TO_PHASE[int(month)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{month!r} is not a calendar month in 1-12.") from exc


def game_months(dates: pd.Series) -> pd.Series:
    """Calendar month per game, as Int64 so a NaT stays missing rather than 0."""
    return pd.to_datetime(dates, errors="coerce").dt.month.astype("Int64")


def season_phases(dates: pd.Series) -> pd.Series:
    """Phase label per game. NaT dates yield ``<NA>``, never a wrong bucket."""
    months = game_months(dates)
    return months.map(_MONTH_TO_PHASE).astype("string")


def phases_present(dates: pd.Series) -> frozenset[str]:
    """The distinct phases a set of games covers.

    This is what "phase-matched" is matched *against*: call it on the holdout's
    dates, then keep only the CV games whose phase is in the result.
    """
    labels = season_phases(dates).dropna().unique().tolist()
    return frozenset(str(label) for label in labels)


def describe_phases(phases: frozenset[str] | set[str] | None) -> str:
    """Stable, readable rendering for metadata and summaries."""
    if not phases:
        return ""
    ordered = [phase for phase in PHASE_ORDER if phase in phases]
    ordered += sorted(phase for phase in phases if phase not in PHASE_ORDER)
    return "+".join(ordered)


def annotate(
    frame: pd.DataFrame, dates: pd.Series, *, prefix: str = ""
) -> pd.DataFrame:
    """Add ``game_month`` and ``season_phase`` columns to ``frame``.

    Returns a copy. ``prefix`` lets a per-fold table carry the columns under
    names that say they describe the fold's validation window
    (``valid_game_month``) rather than one game.
    """
    out = frame.copy()
    out[f"{prefix}game_month"] = game_months(dates).to_numpy()
    out[f"{prefix}season_phase"] = season_phases(dates).to_numpy()
    return out
