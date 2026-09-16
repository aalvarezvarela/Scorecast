"""Phase B: attach ``game_id``, ``team_id`` and ``player_id`` to parsed rows.

The dimensions come from three places, none of them the injury PDFs:

* **Games and tipoffs** from ``fetch_nba_schedule``. It returns explicit UTC
  (``gdtutc``/``utctm``) with full coverage, which is the only reason ``ir_game``
  can be populated at all -- ``nba_games`` carries no time, and the report's own
  ``Game Time`` column prints ``10:30 (ET)`` with **no AM/PM** and is unusable.
* **Teams** from ``nba_players``. The report prints the team as ``city + name``
  ("Houston Rockets"), which matches that table exactly for all 30 clubs.
* **Players** from ``nba_players`` (box scores) **union** ``nba_injuries``
  (inactive lists), scoped to the game being resolved. The union is what makes
  it work: a player who is ``Out`` -- precisely who the report is about -- never
  appears in a box score, and the inactive list is what covers him.

  These two disagree about how a name is spelled, which drives the tiering
  below. ``nba_injuries`` carries ``first_name``/``last_name`` in full for every
  season. ``nba_players`` carries them **only for 2025**; every earlier season
  has a null ``familyname`` and an abbreviated ``player_name`` ("B. Adebayo").
  So a full-name match alone silently loses any player who appeared in a box
  score but not on an inactive list -- which is why there is a surname + first
  initial tier. Within one club's ~15-man roster that is effectively unique, and
  where it is not, the match is refused rather than guessed.

Resolution is deliberately narrow before it is wide. Once ``game_id`` and
``team_id`` are known the candidate set is ~15 players rather than ~2,300, so an
exact normalised match is nearly always right and a wrong match is nearly
impossible. Anything that does not resolve is recorded, never guessed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
import psycopg

TIPOFF_SCHEDULE = "schedule"
TIPOFF_GAME_TIME_INDEX = "game_time_index"
TIPOFF_INJURY_REPORT = "injury_report"

METHOD_GAME_ROSTER = "game_roster"
METHOD_GAME_INITIAL = "game_roster_initial"
METHOD_SEASON_ROSTER = "season_roster"
METHOD_SEASON_INITIAL = "season_roster_initial"
METHOD_CURATED_ALIAS = "curated_alias"

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

#: Report names the four roster tiers cannot resolve, mapped to ``player_id`` by
#: hand. Each was checked against box scores and inactive lists (2026-09-15).
#: Keys are :func:`normalise_name` output. Only the case this table exists for:
#: the report and our rosters spell the player differently, or the surname +
#: initial collides within the team. Traded or waived players listed by a club
#: they never played for are NOT here -- they stay unresolved, correctly.
#:
#: The alias is still scoped: it applies only when that ``player_id`` is on the
#: listing team's roster that season, so a stale entry cannot attach a player to
#: a team he never belonged to.
CURATED_PLAYER_ALIASES: dict[str, int] = {
    # Legal name on the report; box scores carry "B. Carrington" / "Bub".
    "carrington|carlton": 1642267,
    # HOU 2024-25 box scores abbreviate both Jalen and Jeff Green to "J. Green",
    # so the initial tier refuses the match as ambiguous.
    "green|jalen": 1630224,
    # Reported under his old surname; our tables carry "Enes Freedom".
    "kanter|enes": 202683,
    # Reported by full legal name; our tables carry "Didi Louzada".
    "louzadasilva|marcos": 1629712,
    # Reported without his second surname; our tables carry "Jones Garcia".
    "jones|david": 1642357,
}


def normalise_name(raw: object) -> str:
    """``"Gilgeous-Alexander, Shai"`` -> ``"gilgeousalexander|shai"``.

    Strips accents, punctuation and generational suffixes so the many ways the
    NBA and the box scores spell the same person collapse together. Keeps family
    and given name separate so "Smith, Jaden" and "Jaden, Smith" cannot collide.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    family, _, given = text.partition(",")

    def clean(part: str) -> str:
        tokens = [
            re.sub(r"[^a-z0-9]", "", tok.lower())
            for tok in part.replace("-", " ").split()
        ]
        tokens = [t for t in tokens if t and t not in _SUFFIXES]
        return "".join(tokens)

    return f"{clean(family)}|{clean(given)}"


def initial_key(norm: str) -> str:
    """``"adebayo|bam"`` -> ``"adebayo|b"``.

    Collapses a full given name onto its first letter so the report's
    "Adebayo, Bam" meets the box score's abbreviated "B. Adebayo".
    """
    family, _, given = str(norm).partition("|")
    return f"{family}|{given[:1]}"


@dataclass
class ResolutionReport:
    """What resolved, what did not, and by which route."""

    resolved: int = 0
    by_method: dict[str, int] = field(default_factory=dict)
    unresolved: list[dict] = field(default_factory=list)
    unmatched_games: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.resolved + len(self.unresolved)

    @property
    def rate(self) -> float:
        return self.resolved / self.total if self.total else 1.0

    def summary(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.by_method.items()))
        return (
            f"resolved {self.resolved}/{self.total} ({self.rate:.2%})"
            + (f" [{parts}]" if parts else "")
            + (f", {len(self.unresolved)} unresolved" if self.unresolved else "")
            + (
                f", {len(self.unmatched_games)} unmatched games"
                if self.unmatched_games
                else ""
            )
        )


def load_team_dimension(conn: psycopg.Connection) -> pd.DataFrame:
    """``report_name`` -> ``(nba_team_id, tricode)`` for all 30 clubs."""
    query = """
        SELECT DISTINCT team_id, team_abbreviation, team_city, team_name
        FROM nba_players.nba_players
        WHERE season_year >= 2019
          AND team_city IS NOT NULL AND team_name IS NOT NULL
    """
    frame = pd.read_sql_query(query, conn)
    frame["report_name"] = (
        frame["team_city"].str.strip() + " " + frame["team_name"].str.strip()
    )
    frame = frame.rename(
        columns={"team_id": "nba_team_id", "team_abbreviation": "tricode"}
    )
    frame = frame[["nba_team_id", "tricode", "report_name"]].drop_duplicates(
        subset=["report_name"], keep="last"
    )
    return frame.reset_index(drop=True)


def load_game_dimension(
    season_years: list[int], conn: psycopg.Connection | None = None
) -> pd.DataFrame:
    """Games with tipoffs, from the schedule endpoint plus the game-time index.

    Two sources because neither is complete on its own:

    * ``fetch_schedules`` is one request per season and carries explicit UTC,
      including playoffs and play-in for 2019-20 through 2024-25 (zero null
      tipoffs). **Its 2025-26 file stopped updating on 2026-04-12**, so it lacks
      that season's play-in, playoffs and NBA Cup knockout week (122 games), and it
      still carries pre-reschedule dates for 9 more -- see
      :func:`_drop_stale_schedule_rows`.
    * ``nba_game_time_index`` is per-game and covers everything including the
      playoffs, but it is populated by ``sync_game_time_index`` and may be empty
      or absent.

    A game with no tipoff from either source cannot clamp a span and is dropped
    here rather than loaded without one. The caller reports the count; see
    ``docs/injury_report_db_plan.md`` section 5.
    """
    from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules

    seasons = sorted(set(season_years))
    columns = [
        "game_id",
        "game_date",
        "season_year",
        "tipoff_utc",
        "team_home",
        "team_away",
    ]
    frames = [fetch_schedules(seasons)[columns].assign(tipoff_source=TIPOFF_SCHEDULE)]

    indexed = _load_game_time_index(seasons, conn)
    if indexed is not None and not indexed.empty:
        frames.append(indexed[columns].assign(tipoff_source=TIPOFF_GAME_TIME_INDEX))

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["tipoff_utc"])
    frame = _drop_stale_schedule_rows(frame, seasons, conn)
    # Schedule first, index second: identical data, but a keep="first" makes the
    # bulk source authoritative and the per-game one purely additive.
    frame = frame.drop_duplicates(subset=["game_id"], keep="first")
    return frame.reset_index(drop=True)


def _drop_stale_schedule_rows(
    frame: pd.DataFrame, seasons: list[int], conn: psycopg.Connection | None
) -> pd.DataFrame:
    """Discard rows whose date disagrees with when the game was actually played.

    The 2025-26 schedule feed froze mid-season, so games rescheduled afterwards
    keep their *original* date and tipoff in it -- DEN@MEM reads 2026-01-25
    15:30 but was played on 2026-03-18. Keeping such a row would store a wrong
    tipoff and clamp every span against it. ``nba_games`` records the played
    date, so a disagreement means the feed is stale; the game then falls through
    to :func:`load_unscheduled_games` and takes its tipoff from the latest
    report. Verified: 9 games in 2025-26, and none in 2019-20 through 2024-25.
    """
    if conn is None or frame.empty:
        return frame
    played = pd.read_sql_query(
        "SELECT DISTINCT game_id, game_date AS played FROM nba_games.nba_games "
        "WHERE season_year = ANY(%(seasons)s)",
        conn,
        params={"seasons": seasons},
    )
    played["played"] = pd.to_datetime(played["played"]).dt.date
    merged = frame.merge(played, how="left", on="game_id")
    stale = merged["played"].notna() & (
        pd.to_datetime(merged["game_date"]).dt.date != merged["played"]
    )
    return merged.loc[~stale].drop(columns=["played"]).reset_index(drop=True)


def _load_game_time_index(
    seasons: list[int], conn: psycopg.Connection | None
) -> pd.DataFrame | None:
    """Read ``nba_game_time_index`` if it has been synced; ``None`` otherwise."""
    if conn is None:
        return None
    from nba_ou.postgre_db.config.db_config import get_schema_name_game_time_index

    schema = get_schema_name_game_time_index()
    query = f"""
        SELECT game_id, game_date, season_year, game_time_utc AS tipoff_utc,
               home_team_tricode AS team_home, away_team_tricode AS team_away
        FROM {schema}.{schema}
        WHERE season_year = ANY(%(seasons)s) AND game_time_utc IS NOT NULL
    """
    try:
        return pd.read_sql_query(query, conn, params={"seasons": seasons})
    except Exception as exc:  # pandas wraps psycopg errors in DatabaseError
        # Never synced. Regular-season tipoffs still come from the schedule;
        # playoffs will simply have no tipoff and be reported as unmatched.
        conn.rollback()
        if _is_missing_table(exc):
            return None
        raise


def _is_missing_table(exc: BaseException) -> bool:
    for err in (exc, exc.__cause__, exc.__context__):
        if isinstance(err, psycopg.errors.UndefinedTable):
            return True
    return "does not exist" in str(exc)


def load_unscheduled_games(
    conn: psycopg.Connection, season_years: list[int], known_game_ids: set[str]
) -> pd.DataFrame:
    """Games in ``nba_games`` that no tipoff source covers.

    All-Star exhibitions are excluded: they have no injury report. Home and away
    come from the ``"AAA @ HHH"`` matchup string, which is the only form of the
    matchup that carries direction.
    """
    query = """
        SELECT game_id, game_date, season_year,
               max(CASE WHEN matchup LIKE '%%@%%' THEN matchup END) AS matchup
        FROM nba_games.nba_games
        WHERE season_year = ANY(%(seasons)s) AND season_type <> 'All Star'
        GROUP BY game_id, game_date, season_year
    """
    frame = pd.read_sql_query(
        query, conn, params={"seasons": sorted(set(season_years))}
    )
    frame = frame.loc[~frame["game_id"].astype(str).isin(known_game_ids)].copy()
    parts = frame["matchup"].fillna("").str.extract(r"^([A-Z]{3}) @ ([A-Z]{3})$")
    frame["team_away"], frame["team_home"] = parts[0], parts[1]
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.date
    return frame.dropna(subset=["team_away", "team_home"]).drop(columns=["matchup"])


def tipoffs_from_reports(
    unscheduled: pd.DataFrame, listings: pd.DataFrame
) -> pd.DataFrame:
    """Fill tipoffs the schedule lacks from the injury reports themselves.

    Takes the time printed by the **latest** report that lists the game, because
    the report tracks reschedules: POR@DAL on 2025-01-09 read ``08:30 (ET)``
    until 13:30 that day and ``07:30 (ET)`` afterwards, and only the later value
    matches the schedule. Rows are tagged ``tipoff_source = injury_report`` so
    they stay distinguishable from schedule-sourced tipoffs.
    """
    from .parse import decode_game_time

    columns = [
        "game_id",
        "game_date",
        "season_year",
        "tipoff_utc",
        "team_home",
        "team_away",
        "tipoff_source",
    ]
    if unscheduled.empty or listings.empty:
        return pd.DataFrame(columns=columns)

    latest = (
        listings.sort_values("observed_at")
        .drop_duplicates(subset=["game_date", "team_away", "team_home"], keep="last")
        .copy()
    )
    latest["tipoff_utc"] = [
        decode_game_time(d, t)
        for d, t in zip(latest["game_date"], latest["game_time_raw"], strict=True)
    ]
    merged = unscheduled.merge(
        latest[["game_date", "team_away", "team_home", "tipoff_utc"]],
        how="inner",
        on=["game_date", "team_away", "team_home"],
    )
    merged = merged.dropna(subset=["tipoff_utc"])
    merged["tipoff_source"] = TIPOFF_INJURY_REPORT
    return merged[columns].reset_index(drop=True)


def load_rosters(conn: psycopg.Connection, season_years: list[int]) -> pd.DataFrame:
    """Per-game rosters: box-score appearances union inactive lists.

    Returns ``(game_id, nba_team_id, player_id, norm_name, season_year)``.
    """
    query = """
        SELECT game_id, team_id AS nba_team_id, player_id, season_year,
               firstname, familyname, player_name
        FROM nba_players.nba_players
        WHERE season_year = ANY(%(seasons)s)
        UNION ALL
        SELECT game_id, team_id AS nba_team_id, player_id, season_year,
               first_name AS firstname, last_name AS familyname,
               NULL AS player_name
        FROM nba_injuries.nba_injuries
        WHERE season_year = ANY(%(seasons)s)
    """
    frame = pd.read_sql_query(
        query, conn, params={"seasons": sorted(set(season_years))}
    )
    if frame.empty:
        return frame.assign(norm_name=pd.Series(dtype=str))

    family = frame["familyname"].fillna("").astype(str)
    given = frame["firstname"].fillna("").astype(str)
    combined = family.str.strip() + ", " + given.str.strip()
    # Fall back to "Firstname Lastname" where the split columns are empty.
    flat = frame["player_name"].fillna("").astype(str)
    needs_flat = combined.str.strip(", ") == ""
    combined = combined.where(~needs_flat, flat.apply(_flip_flat_name))

    frame["norm_name"] = combined.map(normalise_name)
    frame = frame[frame["norm_name"].str.strip("|") != ""]
    frame["norm_initial"] = frame["norm_name"].map(initial_key)
    return frame[
        [
            "game_id",
            "nba_team_id",
            "player_id",
            "season_year",
            "norm_name",
            "norm_initial",
        ]
    ].drop_duplicates()


def _flip_flat_name(name: str) -> str:
    """``"Shai Gilgeous-Alexander"`` -> ``"Gilgeous-Alexander, Shai"``."""
    parts = str(name).strip().split()
    if len(parts) < 2:
        return name
    return f"{' '.join(parts[1:])}, {parts[0]}"


def resolve_games(
    rows: pd.DataFrame, games: pd.DataFrame, report: ResolutionReport
) -> pd.DataFrame:
    """Attach ``game_id``, ``tipoff_utc`` and ``season_year``.

    Joined on (ET game date, away tricode, home tricode). Both sides are ET
    calendar dates so they compare directly; nothing here uses a wall clock.
    """
    if rows.empty:
        return rows.assign(game_id=None, tipoff_utc=pd.NaT, season_year=None)

    merged = rows.merge(
        games,
        how="left",
        left_on=["game_date", "team_away", "team_home"],
        right_on=["game_date", "team_away", "team_home"],
    )
    missing = merged["game_id"].isna()
    if missing.any():
        for _, row in (
            merged.loc[missing, ["game_date", "team_away", "team_home"]]
            .drop_duplicates()
            .iterrows()
        ):
            report.unmatched_games.append(
                {
                    "game_date": row["game_date"],
                    "matchup": f"{row['team_away']}@{row['team_home']}",
                }
            )
    return merged.loc[~missing].reset_index(drop=True)


def _match_tier(
    pending: pd.DataFrame,
    rosters: pd.DataFrame,
    keys: list[str],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Try one scope/name-form combination; return ``(matched, still_pending)``.

    A key that maps to more than one ``player_id`` is dropped from the lookup
    entirely, so an ambiguous surname + initial falls through to the next tier
    and ultimately to ``ir_unresolved``. Guessing which of two players a report
    meant would be worse than admitting we do not know.
    """
    if pending.empty:
        return pending, pending

    lookup = rosters[keys + ["player_id"]].drop_duplicates()
    counts = lookup.groupby(keys, dropna=False)["player_id"].transform("size")
    lookup = lookup.loc[counts == 1]

    merged = pending.merge(lookup, how="left", on=keys)
    hit = merged["player_id"].notna()
    matched = merged.loc[hit].copy()
    matched["method"] = method
    still = merged.loc[~hit].drop(columns=["player_id"]).reset_index(drop=True)
    return matched.reset_index(drop=True), still


def _match_curated_alias(
    pending: pd.DataFrame,
    rosters: pd.DataFrame,
    aliases: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve through :data:`CURATED_PLAYER_ALIASES`, scoped to the team-season.

    Returns ``(matched, still_pending)`` like :func:`_match_tier`.
    """
    aliases = CURATED_PLAYER_ALIASES if aliases is None else aliases
    if pending.empty or not aliases:
        return pending.iloc[0:0], pending

    candidate = pending["norm_name"].map(aliases)
    members = set(
        zip(
            rosters["season_year"].astype(int),
            rosters["nba_team_id"].astype(int),
            rosters["player_id"].astype(int),
            strict=True,
        )
    )
    hit = pd.Series(
        [
            pd.notna(pid)
            and pd.notna(season)
            and pd.notna(team)
            and (int(season), int(team), int(pid)) in members
            for pid, season, team in zip(
                candidate, pending["season_year"], pending["nba_team_id"], strict=True
            )
        ],
        index=pending.index,
    )
    matched = pending.loc[hit].copy()
    matched["player_id"] = candidate.loc[hit].astype("int64")
    matched["method"] = METHOD_CURATED_ALIAS
    return matched.reset_index(drop=True), pending.loc[~hit].reset_index(drop=True)


def resolve_players(
    rows: pd.DataFrame, rosters: pd.DataFrame, report: ResolutionReport
) -> pd.DataFrame:
    """Attach ``player_id``, narrowest scope first, recording every miss.

    Four tiers, in decreasing confidence: the game's own roster by full name,
    the game's roster by surname + initial, then the same two widened to the
    team's whole season. The widening exists because a G League assignee or a
    just-signed player can appear on the report without being on either list for
    that specific game.
    """
    if rows.empty:
        return rows.assign(player_id=None, method=None)

    pending = rows.copy()
    pending["norm_name"] = pending["raw_name"].map(normalise_name)
    pending["norm_initial"] = pending["norm_name"].map(initial_key)

    tiers = (
        (["game_id", "nba_team_id", "norm_name"], METHOD_GAME_ROSTER),
        (["game_id", "nba_team_id", "norm_initial"], METHOD_GAME_INITIAL),
        (["season_year", "nba_team_id", "norm_name"], METHOD_SEASON_ROSTER),
        (["season_year", "nba_team_id", "norm_initial"], METHOD_SEASON_INITIAL),
    )

    matched_frames: list[pd.DataFrame] = []
    for keys, method in tiers:
        found, pending = _match_tier(pending, rosters, keys, method)
        if not found.empty:
            matched_frames.append(found)
        if pending.empty:
            break

    # Last, so a curated alias can never override a roster match.
    found, pending = _match_curated_alias(pending, rosters)
    if not found.empty:
        matched_frames.append(found)

    for _, row in pending.iterrows():
        report.unresolved.append(
            {
                "raw_name": row["raw_name"],
                "raw_team": row["raw_team"],
                "season_year": row.get("season_year"),
            }
        )

    if not matched_frames:
        return pending.assign(player_id=None, method=None).iloc[0:0]

    resolved = pd.concat(matched_frames, ignore_index=True)
    report.resolved += len(resolved)
    for method, count in resolved["method"].value_counts().items():
        report.by_method[method] = report.by_method.get(method, 0) + int(count)
    return resolved
