"""Reading injury state back out, as of a point in time.

The one query the model needs:

    for game G at instant T, what was each player's reported status, and how
    stale was that information?

Three rules are baked in here rather than left to callers, because each is a way
to leak the future into a feature:

1. **Strictly ``<``.** A report stamped exactly at the cutoff is published *at*
   that instant. 67.3% of tipoffs are on the hour, so at T-30m two thirds of the
   sample lands exactly on a report stamp and ``<=`` would leak on all of them.
2. **``report_age_minutes`` always comes back.** It is ~30 min in the 24/day era
   and <=15 min in the 96/day era. A model that cannot see the age cannot learn
   that older-era features are stale.
3. **Never fall forward.** No span before the cutoff means no row -- not the
   next report.

Absence is *not* health. A player missing from the result is only "no
designation filed" if a report existed at T **and** their team had filed by
then; otherwise it is unknown. :func:`coverage_at` answers that, and
:func:`status_as_of` will not guess on the caller's behalf.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import psycopg

from .schema import SCHEMA

_STATUS_AS_OF = f"""
    SELECT s.game_id,
           s.player_id,
           s.team_id,
           t.tricode,
           st.code            AS status,
           s.status_id,
           c.code             AS reason_category,
           r.detail           AS reason_detail,
           s.valid_from,
           s.mins_to_tip,
           EXTRACT(EPOCH FROM (%(as_of)s - s.valid_from)) / 60
                              AS report_age_minutes
    FROM {SCHEMA}.ir_status_span s
    JOIN {SCHEMA}.ir_team            t  ON t.team_id     = s.team_id
    JOIN {SCHEMA}.ir_reason          r  ON r.reason_id   = s.reason_id
    JOIN {SCHEMA}.ir_reason_category c  ON c.category_id = r.category_id
    LEFT JOIN {SCHEMA}.ir_status     st ON st.status_id  = s.status_id
    WHERE s.game_id = ANY(%(game_ids)s)
      AND s.valid_from <  %(as_of)s
      AND s.valid_to   >= %(as_of)s
      AND s.status_id IS NOT NULL
    ORDER BY s.game_id, s.player_id
"""

_COVERAGE = f"""
    SELECT f.game_id, f.team_id, f.submitted
    FROM {SCHEMA}.ir_filing_span f
    WHERE f.game_id = ANY(%(game_ids)s)
      AND f.valid_from <  %(as_of)s
      AND f.valid_to   >= %(as_of)s
"""

_LAST_REPORT = f"""
    SELECT max(observed_at) AS last_report
    FROM {SCHEMA}.ir_report
    WHERE observed_at < %(as_of)s AND parse_ok
"""


def status_as_of(
    conn: psycopg.Connection,
    game_ids: list[str],
    as_of: datetime,
) -> pd.DataFrame:
    """Reported status for each listed player, as known strictly before ``as_of``.

    ``as_of`` must be timezone-aware. Ordering and comparison are in UTC
    throughout: on a DST fall-back date the ET wall clock repeats 01:00-01:59,
    so an ET-naive comparison can place a report on the wrong side of a cutoff
    by up to an hour.
    """
    _require_aware(as_of)
    return pd.read_sql_query(
        _STATUS_AS_OF, conn, params={"game_ids": list(game_ids), "as_of": as_of}
    )


def coverage_at(
    conn: psycopg.Connection,
    game_ids: list[str],
    as_of: datetime,
) -> pd.DataFrame:
    """Which teams had filed by ``as_of``, and whether any report existed.

    This is what separates "not listed, so no designation" from "we know
    nothing". Treating the second as health is how an unknown silently becomes a
    favourable known.
    """
    _require_aware(as_of)
    filings = pd.read_sql_query(
        _COVERAGE, conn, params={"game_ids": list(game_ids), "as_of": as_of}
    )
    last = pd.read_sql_query(_LAST_REPORT, conn, params={"as_of": as_of})
    filings.attrs["last_report"] = None if last.empty else last["last_report"].iloc[0]
    return filings


def _require_aware(as_of: datetime) -> None:
    if as_of.tzinfo is None or as_of.tzinfo.utcoffset(as_of) is None:
        raise ValueError(
            "as_of must be timezone-aware; a naive timestamp cannot be ordered "
            "correctly across a DST fall-back"
        )
