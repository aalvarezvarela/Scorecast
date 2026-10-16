"""Tonight's injury reports, read in memory -- never written.

The store holds a game's report spans once the game is played
(``scripts/injury_reports/daily_injury_reports.py``). Before tip-off a
prediction reads tonight's reports live: the reports published so far, parsed
and turned into spans by the ingest's own code (``ingest_season`` in a dry run,
which writes nothing), then read with the store's own queries -- the same SQL,
with tonight's rows passed in as arrays in place of the tables (``state_at_
snapshots``, ``status_spans``, ``filing_spans``, ``team_game_schedule``). The
server then makes every value, cast and date exactly as for a stored game.

A game's spans need every report that lists it, which starts the day before
the game, so the window opens then.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg

from nba_ou.postgre_db.row_order import canonical_row_order, read_sql_canonical

from . import fetch
from .schema import SCHEMA

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class LiveInjuryReports:
    """Tonight's games' report spans, as of ``as_of``.

    ``spans`` and ``filing_spans`` carry NBA team ids, the status id (NaN for a
    player who dropped off the list) and the reason as category code and
    detail; ``reports`` every report instant read; ``games`` the game
    dimension rows of tonight's games.
    """

    as_of: pd.Timestamp
    spans: pd.DataFrame
    filing_spans: pd.DataFrame
    reports: pd.Series
    games: pd.DataFrame

    @property
    def game_ids(self) -> list[str]:
        return sorted(self.games["game_id"].astype(str))


def report_window(game_date: date, as_of: pd.Timestamp) -> tuple[date, pd.Timestamp]:
    """First report day and last instant read for ``game_date``'s games."""
    return game_date - timedelta(days=1), pd.Timestamp(as_of)


def build_live_injury_reports(
    manifest: pd.DataFrame,
    storage,
    source: psycopg.Connection,
    *,
    game_date: date,
    game_ids: list[str],
    as_of: pd.Timestamp,
    season: str,
) -> LiveInjuryReports:
    """Parse the reports published before ``as_of`` into tonight's spans.

    ``manifest`` is the archive's for ``season``; only reports from the day
    before ``game_date`` and strictly before ``as_of`` are read -- a report
    stamped at the cutoff is not yet visible, the snapshot reads' rule.
    """
    from nba_ou.fetch_data.injury_reports.archive import manifest as mf

    from .ingest import ingest_season

    as_of = pd.Timestamp(as_of)
    first_day, _ = report_window(game_date, as_of)
    published = mf.published_utc(manifest)
    window = manifest[
        (published.dt.tz_convert(ET).dt.date >= first_day) & (published < as_of)
    ]
    collected: dict = {}
    if not window.empty:
        ingest_season(
            season,
            manifest=window,
            storage=storage,
            aiven=None,
            source=source,
            dry_run=True,
            quiet=True,
            write_game_dates=(game_date, game_date),
            collect=collected,
        )
    ids = {str(g) for g in game_ids}
    spans = collected.get("spans", pd.DataFrame(columns=_SPAN_KEYS))
    filings = collected.get("filing_spans", pd.DataFrame(columns=_FILING_KEYS))
    games = collected.get("games", pd.DataFrame(columns=_GAME_KEYS))
    reports = collected.get("reports", pd.DataFrame(columns=["observed_at"]))
    return LiveInjuryReports(
        as_of=as_of,
        spans=spans[spans["game_id"].astype(str).isin(ids)].reset_index(drop=True),
        filing_spans=filings[filings["game_id"].astype(str).isin(ids)].reset_index(
            drop=True
        ),
        reports=pd.to_datetime(reports["observed_at"], utc=True),
        games=games[games["game_id"].astype(str).isin(ids)].reset_index(drop=True),
    )


_SPAN_KEYS = ["game_id", "team_id", "player_id", "valid_from", "valid_to", "status_id"]
_FILING_KEYS = ["game_id", "team_id", "valid_from", "valid_to", "submitted"]
_GAME_KEYS = ["game_id", "game_date", "season_year", "tipoff_utc"]


def _int_or_none(value):
    return None if pd.isna(value) else int(value)


def _span_params(live: LiveInjuryReports) -> dict:
    spans = live.spans
    games = live.games.set_index(live.games["game_id"].astype(str))
    reason = spans["reason_id"].astype(str).str.split("|", n=1)
    game = spans["game_id"].astype(str)
    return {
        "s_game": game.tolist(),
        "s_team": [str(_int_or_none(v)) for v in spans["team_id"]],
        "s_player": [str(_int_or_none(v)) for v in spans["player_id"]],
        "s_from": [
            v.to_pydatetime() for v in pd.to_datetime(spans["valid_from"], utc=True)
        ],
        "s_to": [
            v.to_pydatetime() for v in pd.to_datetime(spans["valid_to"], utc=True)
        ],
        "s_status": [_int_or_none(v) for v in spans["status_id"]],
        "s_category": [r[0] for r in reason],
        "s_detail": [r[1] if len(r) > 1 else "" for r in reason],
        "s_date": [games.loc[g, "game_date"] for g in game],
        "s_season": [int(games.loc[g, "season_year"]) for g in game],
        "s_tip": [
            pd.Timestamp(games.loc[g, "tipoff_utc"]).to_pydatetime() for g in game
        ],
    }


_LIVE_SPANS = """
    live_span AS (
        SELECT * FROM unnest(
            %(s_game)s::text[], %(s_team)s::text[], %(s_player)s::text[],
            %(s_from)s::timestamptz[], %(s_to)s::timestamptz[],
            %(s_status)s::smallint[], %(s_category)s::text[], %(s_detail)s::text[],
            %(s_date)s::date[], %(s_season)s::smallint[], %(s_tip)s::timestamptz[]
        ) AS s(game_id, team_id, player_id, valid_from, valid_to, status_id,
               reason_category, reason_detail, game_date, season_year, tipoff_utc)
    )
"""


def _filing_params(live: LiveInjuryReports) -> dict:
    filings = live.filing_spans
    games = live.games.set_index(live.games["game_id"].astype(str))
    game = filings["game_id"].astype(str)
    return {
        "f_game": game.tolist(),
        "f_team": [str(_int_or_none(v)) for v in filings["team_id"]],
        "f_from": [
            v.to_pydatetime() for v in pd.to_datetime(filings["valid_from"], utc=True)
        ],
        "f_to": [
            v.to_pydatetime() for v in pd.to_datetime(filings["valid_to"], utc=True)
        ],
        "f_submitted": [bool(v) for v in filings["submitted"]],
        "f_tip": [
            pd.Timestamp(games.loc[g, "tipoff_utc"]).to_pydatetime() for g in game
        ],
    }


_LIVE_FILINGS = """
    live_filing AS (
        SELECT * FROM unnest(
            %(f_game)s::text[], %(f_team)s::text[], %(f_from)s::timestamptz[],
            %(f_to)s::timestamptz[], %(f_submitted)s::boolean[],
            %(f_tip)s::timestamptz[]
        ) AS f(game_id, team_id, valid_from, valid_to, submitted, tipoff_utc)
    )
"""

_CUTOFFS = """
    WITH cutoffs AS (
        SELECT * FROM unnest(
            %(game_ids)s::text[],
            %(snapshot_minutes)s::integer[],
            %(as_of)s::timestamptz[]
        ) AS c(game_id, snapshot_minutes, as_of)
    ),
"""

#: ``fetch._STATUS_AT_SNAPSHOTS`` over tonight's spans: the same output
#: expressions, the dimension joins already resolved into the rows.
_LIVE_STATUS_AT_SNAPSHOTS = (
    _CUTOFFS
    + _LIVE_SPANS
    + f"""
    SELECT c.game_id, c.snapshot_minutes,
           s.team_id AS team_id, s.player_id AS player_id,
           st.code AS status, s.reason_category AS reason_category,
           s.reason_detail AS reason_detail, s.game_date, s.season_year
    FROM cutoffs c
    JOIN live_span s ON s.game_id = c.game_id
        AND s.valid_from < c.as_of AND s.valid_to >= c.as_of
    JOIN {SCHEMA}.ir_status st ON st.status_id = s.status_id
"""
)

_LIVE_FILING_AT_SNAPSHOTS = (
    _CUTOFFS
    + _LIVE_FILINGS
    + """
    SELECT c.game_id, c.snapshot_minutes, f.team_id AS team_id, f.submitted
    FROM cutoffs c
    JOIN live_filing f ON f.game_id = c.game_id
        AND f.valid_from < c.as_of AND f.valid_to >= c.as_of
"""
)

#: ``fetch._REPORT_AGE_AT_SNAPSHOTS``, the last report taken over the stored
#: reports and tonight's together.
_LIVE_REPORT_AGE_AT_SNAPSHOTS = (
    _CUTOFFS
    + f"""
    live_report AS (
        SELECT * FROM unnest(%(reports)s::timestamptz[]) AS r(observed_at)
    )
    SELECT c.game_id, c.snapshot_minutes,
           EXTRACT(EPOCH FROM (c.as_of - last.observed_at)) / 60
               AS report_age_minutes
    FROM cutoffs c
    LEFT JOIN LATERAL (
        SELECT max(u.observed_at) AS observed_at
        FROM (
            SELECT r.observed_at FROM {SCHEMA}.ir_report r
            WHERE r.observed_at < c.as_of AND r.parse_ok
            UNION ALL
            SELECT l.observed_at FROM live_report l WHERE l.observed_at < c.as_of
        ) u
    ) last ON TRUE
"""
)


def _cutoff_params(cutoffs: pd.DataFrame) -> dict:
    timestamps = pd.to_datetime(cutoffs["as_of"], utc=True)
    return {
        "game_ids": cutoffs["game_id"].astype(str).tolist(),
        "snapshot_minutes": cutoffs["snapshot_minutes"].astype(int).tolist(),
        "as_of": [value.to_pydatetime() for value in timestamps],
    }


def state_at_snapshots(
    conn: psycopg.Connection, cutoffs: pd.DataFrame, live: LiveInjuryReports
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``fetch.report_state_at_snapshots`` for cutoffs of tonight's games."""
    params = _cutoff_params(cutoffs)
    return (
        read_sql_canonical(
            _LIVE_STATUS_AT_SNAPSHOTS, conn, params={**params, **_span_params(live)}
        ),
        read_sql_canonical(
            _LIVE_FILING_AT_SNAPSHOTS, conn, params={**params, **_filing_params(live)}
        ),
        read_sql_canonical(
            _LIVE_REPORT_AGE_AT_SNAPSHOTS,
            conn,
            params={
                **params,
                "reports": [v.to_pydatetime() for v in live.reports],
            },
        ),
    )


def report_state_at_snapshots(
    conn: psycopg.Connection,
    cutoffs: pd.DataFrame,
    live: LiveInjuryReports | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The stored reads for stored games, the live ones for tonight's, merged
    into the order one read over every cutoff gives (the canonical order is a
    function of the rows alone)."""
    if live is None:
        return fetch.report_state_at_snapshots(conn, cutoffs)
    tonight = cutoffs["game_id"].astype(str).isin(set(live.game_ids))
    stored = fetch.report_state_at_snapshots(conn, cutoffs[~tonight])
    if not tonight.any():
        return stored
    current = state_at_snapshots(conn, cutoffs[tonight], live)
    return tuple(_merged(a, b) for a, b in zip(stored, current, strict=True))


def _merged(stored: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    parts = [frame for frame in (stored, live) if not frame.empty]
    if not parts:
        return stored
    return canonical_row_order(pd.concat(parts, ignore_index=True))


_LIVE_STATUS_SPANS = (
    "WITH "
    + _LIVE_SPANS
    + f"""
    SELECT s.game_id, s.team_id AS team_id, s.player_id AS player_id,
           st.code AS status, s.reason_category AS reason_category,
           s.valid_from, s.valid_to, s.game_date, s.season_year, s.tipoff_utc
    FROM live_span s
    LEFT JOIN {SCHEMA}.ir_status st ON st.status_id = s.status_id
    WHERE s.valid_from < s.tipoff_utc
"""
)

_LIVE_FILING_SPANS = (
    "WITH "
    + _LIVE_FILINGS
    + """
    SELECT f.game_id, f.team_id AS team_id, f.valid_from, f.valid_to, f.submitted
    FROM live_filing f
    WHERE f.valid_from < f.tipoff_utc
"""
)


def status_spans(
    conn: psycopg.Connection,
    season_years: list[int] | None,
    live: LiveInjuryReports | None,
) -> pd.DataFrame:
    """``fetch.status_spans`` plus tonight's, in one read's order."""
    stored = fetch.status_spans(conn, season_years)
    if live is None or live.spans.empty:
        return stored
    current = read_sql_canonical(_LIVE_STATUS_SPANS, conn, params=_span_params(live))
    return _merged(stored, current)


def filing_spans(
    conn: psycopg.Connection,
    season_years: list[int] | None,
    live: LiveInjuryReports | None,
) -> pd.DataFrame:
    """``fetch.filing_spans`` plus tonight's, in one read's order."""
    stored = fetch.filing_spans(conn, season_years)
    if live is None or live.filing_spans.empty:
        return stored
    current = read_sql_canonical(_LIVE_FILING_SPANS, conn, params=_filing_params(live))
    return _merged(stored, current)


#: ``fetch._TEAM_GAME_SCHEDULE`` over the stored games with tonight's in place:
#: a game the store does not hold yet still finds its teams' previous games.
_LIVE_TEAM_GAME_SCHEDULE = f"""
    WITH games AS (
        SELECT g.game_id, g.game_date, g.season_year, g.tipoff_utc,
               g.team_home, g.team_away
        FROM {SCHEMA}.ir_game g
        WHERE NOT (g.game_id = ANY(%(g_game)s::text[]))
        UNION ALL
        SELECT * FROM unnest(
            %(g_game)s::text[], %(g_date)s::date[], %(g_season)s::smallint[],
            %(g_tip)s::timestamptz[], %(g_home)s::text[], %(g_away)s::text[]
        )
    ), team_games AS (
        SELECT g.game_id, g.game_date, g.season_year, g.tipoff_utc,
               t.nba_team_id::text AS team_id
        FROM games g
        JOIN {SCHEMA}.ir_team t ON t.tricode IN (g.team_home, g.team_away)
        WHERE left(g.game_id, 3) NOT IN ('001', '003')
    ), ordered AS (
        SELECT *, lag(game_id) OVER (
                   PARTITION BY team_id ORDER BY tipoff_utc, game_id
               ) AS prev_game_id
        FROM team_games
    )
    SELECT game_id, team_id, game_date, season_year, tipoff_utc, prev_game_id
    FROM ordered g
    WHERE {fetch._SEASON_FILTER}
"""


def team_game_schedule(
    conn: psycopg.Connection,
    season_years: list[int] | None,
    live: LiveInjuryReports | None,
) -> pd.DataFrame:
    """``fetch.team_game_schedule`` with tonight's games in it."""
    if live is None or live.games.empty:
        return fetch.team_game_schedule(conn, season_years)
    games = live.games
    params = {
        **fetch._season_params(season_years),
        "g_game": games["game_id"].astype(str).tolist(),
        "g_date": [pd.Timestamp(d).date() for d in games["game_date"]],
        "g_season": [int(s) for s in games["season_year"]],
        "g_tip": [
            pd.Timestamp(t).to_pydatetime()
            for t in pd.to_datetime(games["tipoff_utc"], utc=True)
        ],
        "g_home": games["team_home"].astype(str).tolist(),
        "g_away": games["team_away"].astype(str).tolist(),
    }
    return read_sql_canonical(_LIVE_TEAM_GAME_SCHEDULE, conn, params=params)
