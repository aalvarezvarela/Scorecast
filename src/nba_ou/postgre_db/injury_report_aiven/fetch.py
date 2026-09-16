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


# --------------------------------------------------------------------------------
# Bulk reads at each game's own tipoff (closing-line features)
# --------------------------------------------------------------------------------
#
# The per-instant reads above take one ``as_of`` for every game. The closing-line
# dataset needs a different instant per game -- its own tipoff -- so these join
# ``ir_game`` and apply the same rules against ``tipoff_utc``: strictly
# ``valid_from < tipoff_utc``, and ``valid_to >= tipoff_utc`` selects the span
# still open at tip (spans are clamped there, so it is the last one).

_SEASON_FILTER = "(%(season_years)s::smallint[] IS NULL OR g.season_year = ANY(%(season_years)s))"

_LAST_STATUS_BEFORE_TIP = f"""
    SELECT s.game_id,
           t.nba_team_id::text AS team_id,
           s.player_id::text   AS player_id,
           st.code             AS status,
           c.code              AS reason_category,
           r.detail            AS reason_detail,
           g.game_date,
           g.season_year
    FROM {SCHEMA}.ir_status_span s
    JOIN {SCHEMA}.ir_game            g  ON g.game_id     = s.game_id
    JOIN {SCHEMA}.ir_team            t  ON t.team_id     = s.team_id
    JOIN {SCHEMA}.ir_reason          r  ON r.reason_id   = s.reason_id
    JOIN {SCHEMA}.ir_reason_category c  ON c.category_id = r.category_id
    JOIN {SCHEMA}.ir_status          st ON st.status_id  = s.status_id
    WHERE s.valid_from <  g.tipoff_utc
      AND s.valid_to   >= g.tipoff_utc
      AND {_SEASON_FILTER}
"""

_FILING_AT_TIP = f"""
    SELECT f.game_id, t.nba_team_id::text AS team_id, f.submitted
    FROM {SCHEMA}.ir_filing_span f
    JOIN {SCHEMA}.ir_game g ON g.game_id = f.game_id
    JOIN {SCHEMA}.ir_team t ON t.team_id = f.team_id
    WHERE f.valid_from <  g.tipoff_utc
      AND f.valid_to   >= g.tipoff_utc
      AND {_SEASON_FILTER}
"""

_REPORT_AGE_AT_TIP = f"""
    SELECT g.game_id,
           EXTRACT(EPOCH FROM (g.tipoff_utc - last.observed_at)) / 60 AS report_age_minutes
    FROM {SCHEMA}.ir_game g
    CROSS JOIN LATERAL (
        SELECT max(r.observed_at) AS observed_at
        FROM {SCHEMA}.ir_report r
        WHERE r.observed_at < g.tipoff_utc AND r.parse_ok
    ) last
    WHERE {_SEASON_FILTER}
"""

#: Every (game, player, status) for the statuses with history features: the
#: player held that status on ANY pre-tip report of the game, and
#: ``at_last_report`` says whether it was still the status on the last one.
#: Doubtful is not here: at the last report it plays ~2% of the time (3 players
#: since 2021), so its history is noise.
LISTED_STATUS_CODES: tuple[str, ...] = ("questionable", "probable")

_LISTED_STATUS_EVENTS = f"""
    SELECT s.game_id,
           t.nba_team_id::text AS team_id,
           s.player_id::text   AS player_id,
           st.code             AS status,
           g.game_date,
           g.season_year,
           bool_or(s.valid_to >= g.tipoff_utc) AS at_last_report
    FROM {SCHEMA}.ir_status_span s
    JOIN {SCHEMA}.ir_game   g  ON g.game_id    = s.game_id
    JOIN {SCHEMA}.ir_team   t  ON t.team_id    = s.team_id
    JOIN {SCHEMA}.ir_status st ON st.status_id = s.status_id
    WHERE st.code = ANY(%(statuses)s)
      AND s.valid_from < g.tipoff_utc
      AND {_SEASON_FILTER}
    GROUP BY 1, 2, 3, 4, 5, 6
"""

_LISTED_PAIRS = f"""
    SELECT DISTINCT s.game_id, s.player_id::text AS player_id
    FROM {SCHEMA}.ir_status_span s
    JOIN {SCHEMA}.ir_game g ON g.game_id = s.game_id
    WHERE s.status_id IS NOT NULL
      AND s.valid_from < g.tipoff_utc
      AND {_SEASON_FILTER}
"""


def _season_params(season_years: list[int] | None) -> dict:
    return {"season_years": None if season_years is None else list(season_years)}


def last_status_before_tip(
    conn: psycopg.Connection, season_years: list[int] | None = None
) -> pd.DataFrame:
    """Each listed player's status on the last report published before tipoff."""
    return pd.read_sql_query(
        _LAST_STATUS_BEFORE_TIP, conn, params=_season_params(season_years)
    )


def filing_at_tip(
    conn: psycopg.Connection, season_years: list[int] | None = None
) -> pd.DataFrame:
    """``submitted`` per (game, team) on the last report before tipoff.

    A team-game with no row had no report listing its game before tip.
    """
    return pd.read_sql_query(_FILING_AT_TIP, conn, params=_season_params(season_years))


def report_age_at_tip(
    conn: psycopg.Connection, season_years: list[int] | None = None
) -> pd.DataFrame:
    """Minutes between the last parsed report and each game's tipoff."""
    return pd.read_sql_query(
        _REPORT_AGE_AT_TIP, conn, params=_season_params(season_years)
    )


def listed_status_events(
    conn: psycopg.Connection,
    season_years: list[int] | None = None,
    statuses: tuple[str, ...] = LISTED_STATUS_CODES,
) -> pd.DataFrame:
    """Every game a player held one of ``statuses`` at any point before tip."""
    params = _season_params(season_years) | {"statuses": list(statuses)}
    return pd.read_sql_query(_LISTED_STATUS_EVENTS, conn, params=params)


def listed_pairs(
    conn: psycopg.Connection, season_years: list[int] | None = None
) -> pd.DataFrame:
    """Every (game, player) with any pre-tip designation, of any status."""
    return pd.read_sql_query(_LISTED_PAIRS, conn, params=_season_params(season_years))


def _require_aware(as_of: datetime) -> None:
    if as_of.tzinfo is None or as_of.tzinfo.utcoffset(as_of) is None:
        raise ValueError(
            "as_of must be timezone-aware; a naive timestamp cannot be ordered "
            "correctly across a DST fall-back"
        )
