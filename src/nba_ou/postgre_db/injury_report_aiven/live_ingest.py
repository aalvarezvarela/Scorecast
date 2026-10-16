"""Replace only affected game timelines, atomically with report receipts.

Existing spans expand at their original report instants before being combined
with new PDFs. This reuses the lossless compressed history in Aiven: completed
PDFs are never downloaded or parsed again, including for late-arriving reports.
"""

from __future__ import annotations

import logging

import pandas as pd
from nba_api.stats.static import players as nba_players

from nba_ou.postgre_db.injury_report_aiven import documents, load
from nba_ou.postgre_db.injury_report_aiven import resolve as R
from nba_ou.postgre_db.injury_report_aiven.ingest import (
    _Read,
    _stack,
    build_filing_observations,
    build_removal_observations,
    build_report_frame,
)
from nba_ou.postgre_db.injury_report_aiven.parse import category_code
from nba_ou.postgre_db.injury_report_aiven.schema import create_season_partition
from nba_ou.postgre_db.injury_report_aiven.spans import build_filing_spans, build_spans
from nba_ou.postgre_db.row_order import read_sql_canonical

log = logging.getLogger(__name__)


def resolve_live_players(
    rows: pd.DataFrame,
    rosters: pd.DataFrame,
    report: R.ResolutionReport | None = None,
) -> pd.DataFrame:
    """Use historical team rosters, then unique exact NBA names for new signings.

    Names neither route resolves are dropped and appended to
    ``report.unresolved``, as the historical ingest does. One unknown name (a
    rookie missing from nba_api's bundled player list, say) must not block
    every other player's report.
    """
    rows, rosters = rows.copy(), rosters.copy()
    rows["nba_team_id"] = rows.nba_team_id.astype(str)
    rosters["nba_team_id"] = rosters.nba_team_id.astype(str)
    resolved = R.resolve_players(rows, rosters, R.ResolutionReport())
    keys = ["game_id", "raw_team", "raw_name", "observed_at"]
    found = resolved[keys].drop_duplicates().assign(_found=True)
    pending = rows.merge(found, on=keys, how="left")
    pending = pending.loc[pending._found.isna()].drop(columns="_found")
    if not pending.empty:
        names = pd.DataFrame(
            [
                (R.normalise_name(R._flip_flat_name(p["full_name"])), p["id"])
                for p in nba_players.get_players()
            ],
            columns=["norm_name", "player_id"],
        )
        # Ambiguous exact names are refused; never use fuzzy global matches.
        names = names.drop_duplicates().loc[
            lambda f: ~f.norm_name.duplicated(keep=False)
        ]
        aliases = R.CURATED_PLAYER_ALIASES
        pending["norm_name"] = pending.raw_name.map(R.normalise_name)
        pending = pending.merge(names, on="norm_name", how="left")
        pending["player_id"] = pending.player_id.fillna(pending.norm_name.map(aliases))
        unknown = pending.player_id.isna()
        if unknown.any():
            if report is not None:
                report.unresolved.extend(
                    pending.loc[unknown]
                    .reindex(columns=["raw_name", "raw_team", "season_year"])
                    .to_dict("records")
                )
            log.warning(
                "Unresolved live injury players dropped: %s",
                sorted(pending.loc[unknown, "raw_name"].unique()),
            )
            pending = pending.loc[~unknown].copy()
            pending["player_id"] = pending.player_id.astype("int64")
        pending["method"] = "nba_exact_name"
        resolved = pd.concat([resolved, pending], ignore_index=True)
    return resolved


def read_existing_observations(conn, game_ids: list[str], end):
    """Reconstruct already-ingested snapshots using inclusive event instants.

    These are source observations, not feature cutoffs. A status starting at a
    report instant belongs to that report. Prediction queries remain strict <.
    """
    params = {"ids": game_ids, "end": end}
    common = """
        FROM injury_report.ir_report r
        JOIN injury_report.ir_filing_span f
          ON f.game_id = ANY(%(ids)s)
         AND f.valid_from <= r.observed_at AND f.valid_to > r.observed_at
        JOIN injury_report.ir_game g ON g.game_id = f.game_id
        JOIN injury_report.ir_team t ON t.team_id = f.team_id
    """
    listed = read_sql_canonical(
        """
        SELECT DISTINCT g.game_id, g.game_date, g.team_home, g.team_away,
               g.season_year, g.tipoff_utc, r.observed_at
    """
        + common
        + " WHERE r.parse_ok AND r.observed_at <= %(end)s",
        conn,
        params=params,
    )
    not_filed = read_sql_canonical(
        """
        SELECT g.game_id, t.team_id, t.report_name AS raw_team, r.observed_at
    """
        + common
        + " WHERE r.parse_ok AND r.observed_at <= %(end)s AND NOT f.submitted",
        conn,
        params=params,
    )
    observations = read_sql_canonical(
        """
        SELECT g.game_id, g.game_date, g.team_home, g.team_away, g.season_year,
               g.tipoff_utc, r.observed_at, t.team_id, t.nba_team_id,
               t.report_name AS raw_team, '' AS raw_name, s.player_id,
               s.status_id, c.label AS reason_category, reason.detail AS reason_detail
    """
        + common
        + """
        JOIN injury_report.ir_status_span s
          ON s.game_id = f.game_id AND s.team_id = f.team_id
         AND s.valid_from <= r.observed_at AND s.valid_to > r.observed_at
        JOIN injury_report.ir_reason reason ON reason.reason_id = s.reason_id
        JOIN injury_report.ir_reason_category c ON c.category_id = reason.category_id
        WHERE r.parse_ok AND r.observed_at <= %(end)s AND f.submitted
          AND s.status_id IS NOT NULL
    """,
        conn,
        params=params,
    )
    return observations, listed, not_filed


def combine_observations(old, new, new_times):
    """A newly read report replaces that instant's reconstructed observation."""
    frames = [old.loc[~old.observed_at.isin(new_times)], new]
    nonempty = [frame for frame in frames if not frame.empty]
    return (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.concat(frames, ignore_index=True)
    )


def rebuild_timelines(observations, listed, not_filed, report_times, team_map):
    """Shared historical removal/filing rules, applied to a full affected game."""
    # A previously listed game can disappear with two empty filed lists, then
    # reappear after a new designation. Materialize those intermediate empty
    # listings too; the historical helpers alone handle only final disappearance.
    missing = []
    if not listed.empty:
        for _game_id, game in listed.groupby("game_id"):
            first = game.sort_values("observed_at").iloc[0]
            present = set(game.observed_at)
            for at in report_times:
                if first.observed_at < at < first.tipoff_utc and at not in present:
                    row = first.to_dict()
                    row["observed_at"] = at
                    missing.append(row)
        if missing:
            listed = pd.concat([listed, pd.DataFrame(missing)], ignore_index=True)
    observations = observations.copy()
    if not observations.empty:
        removals = build_removal_observations(
            observations, listed, not_filed, report_times
        )
        observations = pd.concat([observations, removals], ignore_index=True)
        observations = observations.drop_duplicates(
            ["game_id", "player_id", "observed_at"], keep="last"
        )
    filings = build_filing_observations(listed, not_filed, report_times, team_map)
    return observations, filings


def ingest_live_reports(
    conn, parsed: list[_Read], urls: list[str], *, teams, games, rosters
) -> None:
    """Commit every successful report, affected spans, and its receipt together."""
    if not parsed:
        return
    reports = build_report_frame(parsed)
    new_times = set(reports.observed_at)
    end = reports.observed_at.max()
    # Callers finish bookkeeping transactions before entering this one.
    with conn.transaction():
        with conn.cursor() as cur:
            # Serialize writers, including runs that discovered reports concurrently.
            cur.execute("SELECT pg_advisory_xact_lock(873520241)")
            cur.execute(
                "SELECT source_url FROM injury_report.ir_document WHERE source_url = ANY(%s) AND ingested_at IS NOT NULL",
                (urls,),
            )
            done = {row[0] for row in cur.fetchall()}
        pairs = [
            (report, url)
            for report, url in zip(parsed, urls, strict=True)
            if url not in done
        ]
        if not pairs:
            return
        parsed, urls = map(list, zip(*pairs, strict=True))
        reports = build_report_frame(parsed)
        new_times = set(reports.observed_at)
        end = reports.observed_at.max()
        team_map = load.upsert_teams(conn, teams, commit=False)
        tricode_map = {r.tricode: team_map[r.report_name] for r in teams.itertuples()}
        raw_listed = _stack(parsed, "games").reindex(
            columns=[
                "game_date",
                "team_home",
                "team_away",
                "game_time_raw",
                "observed_at",
            ]
        )
        raw_players = _stack(parsed, "players").reindex(
            columns=[
                "game_date",
                "team_home",
                "team_away",
                "raw_team",
                "raw_name",
                "status_id",
                "reason_category",
                "reason_detail",
                "observed_at",
            ]
        )
        raw_nys = _stack(parsed, "not_submitted").reindex(
            columns=["game_date", "team_home", "team_away", "raw_team", "observed_at"]
        )
        resolution = R.ResolutionReport()
        listed = R.resolve_games(raw_listed, games, resolution)
        if resolution.unmatched_games:
            raise ValueError(
                f"Unresolved injury report games: {resolution.unmatched_games}"
            )
        load.upsert_games(
            conn, games.loc[games.game_id.isin(listed.game_id)], commit=False
        )
        not_filed = R.resolve_games(raw_nys, games, R.ResolutionReport())
        not_filed["team_id"] = not_filed.raw_team.map(team_map)
        if not_filed.team_id.isna().any():
            raise ValueError("Unknown injury report team filing")
        observations = R.resolve_games(raw_players, games, R.ResolutionReport())
        if not observations.empty:
            observations["nba_team_id"] = observations.raw_team.map(
                teams.set_index("report_name").nba_team_id
            )
            if observations.nba_team_id.isna().any():
                raise ValueError("Unknown injury report team")
            unresolved = R.ResolutionReport()
            observations = resolve_live_players(observations, rosters, unresolved)
            observations["team_id"] = observations.raw_team.map(team_map)
            load.record_unresolved(conn, unresolved.unresolved, commit=False)
        # New reports can remove a game entirely; include existing pre-game
        # timelines affected by any newly inserted publication time too.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT game_id FROM injury_report.ir_filing_span
                WHERE valid_from < %s AND valid_to > %s
            """,
                (end, reports.observed_at.min()),
            )
            affected = {row[0] for row in cur.fetchall()} | set(listed.game_id)
        ids = sorted(affected)
        # Include later old reports when a late-arriving PDF fills an earlier gap.
        with conn.cursor() as cur:
            cur.execute(
                "SELECT max(observed_at) FROM injury_report.ir_report WHERE parse_ok"
            )
            last = cur.fetchone()[0]
        end = max(end, pd.Timestamp(last)) if last is not None else end
        old_obs, old_listed, old_nys = read_existing_observations(conn, ids, end)
        observations = combine_observations(old_obs, observations, new_times)
        listed = combine_observations(old_listed, listed, new_times)
        not_filed = combine_observations(old_nys, not_filed, new_times)
        with conn.cursor() as cur:
            start = (
                min(reports.observed_at.min(), listed.observed_at.min())
                if not listed.empty
                else reports.observed_at.min()
            )
            cur.execute(
                "SELECT observed_at FROM injury_report.ir_report WHERE parse_ok AND observed_at >= %s AND observed_at <= %s",
                (start, end),
            )
            times = pd.Series(
                sorted(new_times | {pd.Timestamp(row[0]) for row in cur.fetchall()})
            )
        observations, filings = rebuild_timelines(
            observations, listed, not_filed, times, tricode_map
        )
        report_map = load.upsert_reports(conn, reports, commit=False)
        if not observations.empty:
            reasons = sorted(
                {
                    (category_code(c), c, d)
                    for c, d in zip(
                        observations.reason_category,
                        observations.reason_detail,
                        strict=True,
                    )
                }
            )
            reason_map = load.upsert_reasons(conn, reasons)
            observations["reason_id"] = [
                reason_map[(category_code(c), d)]
                for c, d in zip(
                    observations.reason_category,
                    observations.reason_detail,
                    strict=True,
                )
            ]
            observations["report_id"] = observations.observed_at.map(report_map)
        for season in set(
            games.loc[games.game_id.isin(ids), "season_year"].astype(int)
        ):
            create_season_partition(conn, season, commit=False)
        spans = build_spans(observations)
        filing_spans = build_filing_spans(filings)
        with conn.cursor() as cur:
            for table in ("ir_status_span", "ir_filing_span"):
                # Table names are fixed, never supplied by the caller.
                cur.execute(
                    f"DELETE FROM injury_report.{table} WHERE game_id = ANY(%s)", (ids,)
                )
        load.insert_spans(conn, spans, commit=False)
        load.insert_filing_spans(conn, filing_spans, commit=False)
        documents.mark_ingested(conn, urls)
