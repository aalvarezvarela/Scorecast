"""Fetch authoritative NBA tipoff times from the season-schedule feeds.

One request returns a whole season, which makes this the cheap way to get
``tipoff_utc`` for every ``game_id`` -- the per-game ``cdn.nba.com`` boxscore
endpoint used by ``game_time_index`` needs one request per game instead.

The source is stats.nba.com's ``ScheduleLeagueV2``. The legacy ``data.nba.com``
feed it replaced was abandoned (its 2025-26 file froze on 2026-04-12, the
2026-27 one was never published) and is kept only as a fallback for past
seasons. Checked on 2016-17 through 2025-26, the two agree on every regular-
season game except three season openers, where V2 has the actual tip (one
minute later) instead of the scheduled one; V2 also carries games the frozen
2025-26 file lacks, and has no 2019-20 preseason.

The legacy feed exposes both an Eastern wall-clock time (``etm``) and a UTC
date/time pair (``gdtutc`` / ``utctm``); we prefer the UTC pair and fall back
to localizing the Eastern value.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from tqdm import tqdm

from nba_ou.fetch_data.nba_schedule.tipoff_corrections import (
    apply_tipoff_corrections,
)

SCHEDULE_URL = (
    "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/"
    "{season_year}/league/00_full_schedule.json"
)

# data.nba.com returns 403 without a browser-like Referer/Origin pair.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
}

EASTERN_TZ = ZoneInfo("America/New_York")

SCHEDULE_COLUMNS = [
    "game_id",
    "season_year",
    "game_date",
    "tipoff_utc",
    "tipoff_et",
    "team_home",
    "team_away",
    "arena_name",
    "arena_city",
    "game_status",
]


def _parse_tipoff_utc(game: dict) -> pd.Timestamp | None:
    """Prefer the explicit UTC pair; fall back to localizing Eastern time."""
    utc_date = (game.get("gdtutc") or "").strip()
    utc_time = (game.get("utctm") or "").strip()
    if utc_date and utc_time:
        parsed = pd.to_datetime(f"{utc_date} {utc_time}", errors="coerce", utc=True)
        if not pd.isna(parsed):
            return parsed

    eastern = (game.get("etm") or "").strip()
    if eastern:
        try:
            naive = datetime.fromisoformat(eastern)
        except ValueError:
            return None
        return pd.Timestamp(naive.replace(tzinfo=EASTERN_TZ)).tz_convert("UTC")

    return None


def fetch_season_schedule(season_year: int, *, timeout: int = 30) -> pd.DataFrame:
    """Return one row per game for ``season_year`` (2024 == the 2024-25 season).

    From :func:`fetch_season_schedule_v2`. If stats.nba.com fails every retry or
    returns no games, the legacy feed is tried instead; it still serves past
    seasons, not 2026-27 on, so there the legacy error is what propagates.
    """
    try:
        schedule = fetch_season_schedule_v2(season_year, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - nba_api raises many types
        print(
            f"! ScheduleLeagueV2 unavailable for {season_year} ({exc}); "
            "trying the legacy feed"
        )
        return fetch_legacy_season_schedule(season_year, timeout=timeout)
    if schedule.empty:
        print(
            f"! ScheduleLeagueV2 has no games for {season_year}; "
            "trying the legacy feed"
        )
        return fetch_legacy_season_schedule(season_year, timeout=timeout)
    return schedule


def fetch_legacy_season_schedule(
    season_year: int, *, timeout: int = 30
) -> pd.DataFrame:
    """One row per game from the legacy ``data.nba.com`` season feed."""
    response = requests.get(
        SCHEDULE_URL.format(season_year=season_year),
        headers=_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()

    rows: list[dict] = []
    for month in response.json().get("lscd", []):
        for game in month.get("mscd", {}).get("g", []):
            game_id = (game.get("gid") or "").strip()
            if not game_id:
                continue

            eastern = (game.get("etm") or "").strip()
            rows.append(
                {
                    "game_id": game_id,
                    "season_year": season_year,
                    "game_date": pd.to_datetime(
                        game.get("gdte"), errors="coerce"
                    ).date(),
                    "tipoff_utc": _parse_tipoff_utc(game),
                    "tipoff_et": eastern or None,
                    "team_home": (game.get("h") or {}).get("ta"),
                    "team_away": (game.get("v") or {}).get("ta"),
                    "arena_name": game.get("an"),
                    "arena_city": game.get("ac"),
                    "game_status": game.get("st"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    return pd.DataFrame(rows)[SCHEDULE_COLUMNS]


SCHEDULE_V2_RETRIES = 3
SCHEDULE_V2_BACKOFF_S = 5.0


def _request_schedule_v2(season_year: int, timeout: int) -> dict:
    """The season's ``ScheduleLeagueV2`` payload; raises once every retry failed."""
    from nba_api.stats.endpoints import ScheduleLeagueV2

    season = f"{season_year}-{str(season_year + 1)[-2:]}"
    for attempt in range(SCHEDULE_V2_RETRIES):
        try:
            return ScheduleLeagueV2(
                league_id="00", season=season, timeout=timeout
            ).get_dict()
        except Exception:
            if attempt == SCHEDULE_V2_RETRIES - 1:
                raise
            time.sleep(SCHEDULE_V2_BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")


def _schedule_v2_rows(payload: dict, season_year: int) -> list[dict]:
    """Rows shaped like the legacy feed's.

    ``gameDateTimeEst`` is Eastern wall-clock time despite its trailing ``Z``;
    without it the value matches the legacy ``etm``. ``gameStatus`` is an int
    here and a string there. Placeholder fixtures (NBA Cup knockouts, playoffs)
    carry no teams yet and keep ``None`` for them.
    """
    rows: list[dict] = []
    for day in payload.get("leagueSchedule", {}).get("gameDates", []):
        for game in day.get("games", []):
            game_id = str(game.get("gameId") or "").strip()
            if not game_id:
                continue
            status = game.get("gameStatus")
            rows.append(
                {
                    "game_id": game_id,
                    "season_year": season_year,
                    "game_date": pd.to_datetime(
                        (game.get("gameDateEst") or "")[:10], errors="coerce"
                    ).date(),
                    "tipoff_utc": pd.to_datetime(
                        game.get("gameDateTimeUTC"), errors="coerce", utc=True
                    ),
                    "tipoff_et": (game.get("gameDateTimeEst") or "").removesuffix("Z")
                    or None,
                    "team_home": (game.get("homeTeam") or {}).get("teamTricode")
                    or None,
                    "team_away": (game.get("awayTeam") or {}).get("teamTricode")
                    or None,
                    "arena_name": game.get("arenaName") or None,
                    "arena_city": game.get("arenaCity") or None,
                    "game_status": None if status is None else str(status),
                }
            )
    return rows


def fetch_season_schedule_v2(season_year: int, *, timeout: int = 30) -> pd.DataFrame:
    """One row per game from stats.nba.com's ``ScheduleLeagueV2``.

    Same columns and value formats as :func:`fetch_legacy_season_schedule`, but
    it keeps updating through the season (rescheduled games, NBA Cup knockouts,
    play-in and playoffs as they are set).
    """
    rows = _schedule_v2_rows(_request_schedule_v2(season_year, timeout), season_year)
    if not rows:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    return pd.DataFrame(rows)[SCHEDULE_COLUMNS]


BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"


#: stats.nba.com throttles bursts by holding requests open rather than refusing
#: them, so a long timeout turns throttling into minutes per date. A short one
#: plus backoff recovers in seconds.
SCOREBOARD_RETRIES = 4
SCOREBOARD_BACKOFF_S = 2.0


def _request_scoreboard(game_date: str, timeout: int) -> dict | None:
    """One date's ScoreboardV3 payload, or None once every retry has failed."""
    from nba_api.stats.endpoints import ScoreboardV3

    for attempt in range(SCOREBOARD_RETRIES):
        try:
            return ScoreboardV3(game_date=game_date, timeout=timeout).get_dict()
        except Exception:  # noqa: BLE001 - a bad date must not abort the load
            if attempt < SCOREBOARD_RETRIES - 1:
                time.sleep(SCOREBOARD_BACKOFF_S * 2**attempt)
    return None


def _scoreboard_rows(
    payload: dict, game_date: str, season_year: int | None
) -> list[dict]:
    rows = []
    for game in payload.get("scoreboard", {}).get("games", []):
        tipoff = pd.to_datetime(game.get("gameTimeUTC"), errors="coerce", utc=True)
        if pd.isna(tipoff) or not game.get("gameId"):
            continue
        rows.append(
            {
                "game_id": game["gameId"],
                "season_year": season_year,
                "game_date": pd.to_datetime(game_date).date(),
                "tipoff_utc": tipoff,
                "tipoff_et": game.get("gameEt"),
                "team_home": (game.get("homeTeam") or {}).get("teamTricode"),
                "team_away": (game.get("awayTeam") or {}).get("teamTricode"),
                "arena_name": None,
                "arena_city": None,
                "game_status": game.get("gameStatusText"),
            }
        )
    return rows


def _rows_to_schedule(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    return apply_tipoff_corrections(pd.DataFrame(rows)[SCHEDULE_COLUMNS])


def fetch_tipoffs_for_date(
    game_date: str, *, timeout: int = 15, season_year: int | None = None
) -> pd.DataFrame | None:
    """One date's tipoffs from ``ScoreboardV3``, or None if it is unavailable.

    Unlike :func:`fetch_tipoffs_for_dates`, a failed request is not an empty
    frame, so a caller can tell an off day from a scoreboard outage.
    """
    payload = _request_scoreboard(str(game_date), timeout)
    if payload is None:
        return None
    return _rows_to_schedule(_scoreboard_rows(payload, str(game_date), season_year))


def fetch_tipoffs_for_dates(
    game_dates: list, *, timeout: int = 15, season_year: int | None = None
) -> pd.DataFrame:
    """Tipoff fallback, by date, for games missing from the season feed.

    The static season feed is published before the season and never backfilled,
    so it has a hole wherever fixtures are scheduled later -- notably the NBA
    Cup window, where the non-qualifying teams' makeup games are added once the
    bracket is known. The daily scoreboard is authoritative for those.

    ``ScoreboardV3`` rather than V2: V2 has known line-score defects over
    2025-10-22..2025-12-25, which is exactly the affected window.

    For long date ranges use :func:`cache_scoreboards`, which survives the
    API cutting a run off.
    """
    rows: list[dict] = []
    failed: list[str] = []
    for game_date in tqdm(
        sorted({str(d) for d in game_dates}), desc="Scoreboard", unit="date"
    ):
        payload = _request_scoreboard(game_date, timeout)
        if payload is None:
            failed.append(game_date)
            continue
        rows.extend(_scoreboard_rows(payload, game_date, season_year))

    if failed:
        print(
            f"! scoreboard unavailable for {len(failed)} date(s) after "
            f"{SCOREBOARD_RETRIES} attempts: {', '.join(failed[:10])}"
        )
    return _rows_to_schedule(rows)


def _cache_path(cache_dir: Path, game_date: str) -> Path:
    return cache_dir / f"scoreboard_{game_date}.json"


def cache_scoreboards(
    game_dates: list,
    cache_dir: str | Path,
    *,
    max_requests: int | None = None,
    timeout: int = 15,
) -> list[str]:
    """Download each date's scoreboard into ``cache_dir``; return dates still missing.

    Dates already cached are never requested again, so a run the API cuts off
    loses nothing. The run stops early -- leaving the rest for the next one --
    once ``max_requests`` dates have been requested, or at the first date that
    fails every retry: stats.nba.com blocks an address after roughly 300 rapid
    requests, and past that point further attempts only waste time.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dates = sorted({str(pd.Timestamp(d).date()) for d in game_dates})
    pending = [d for d in dates if not _cache_path(cache_dir, d).exists()]
    if len(pending) < len(dates):
        print(f"Scoreboard cache: {len(dates) - len(pending)}/{len(dates)} date(s) already cached")

    budget = len(pending) if max_requests is None else min(max_requests, len(pending))
    requested = 0
    with tqdm(total=len(pending), desc="Scoreboard", unit="date") as bar:
        for game_date in pending:
            if requested >= budget:
                break
            payload = _request_scoreboard(game_date, timeout)
            requested += 1
            # A blocked address can get an answer without a scoreboard in it;
            # caching that would hide the date from every later run.
            if payload is None or "scoreboard" not in payload:
                print(f"\n! scoreboard refused {game_date}; stopping this run")
                break
            path = _cache_path(cache_dir, game_date)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(path)
            bar.update(1)

    return [d for d in dates if not _cache_path(cache_dir, d).exists()]


def tipoffs_from_cache(game_dates: list, cache_dir: str | Path) -> pd.DataFrame:
    """Schedule rows parsed from cached scoreboards (uncached dates are skipped)."""
    cache_dir = Path(cache_dir)
    rows: list[dict] = []
    for game_date in sorted({str(pd.Timestamp(d).date()) for d in game_dates}):
        path = _cache_path(cache_dir, game_date)
        if path.exists():
            rows.extend(_scoreboard_rows(json.loads(path.read_text()), game_date, None))
    return _rows_to_schedule(rows)


def fetch_schedules(season_years: list[int], *, timeout: int = 30) -> pd.DataFrame:
    """Concatenate :func:`fetch_season_schedule` across seasons.

    Known-wrong feed tipoffs (games the league moved) are overridden here; see
    :mod:`nba_ou.fetch_data.nba_schedule.tipoff_corrections`.
    """
    frames = [
        fetch_season_schedule(season_year, timeout=timeout)
        for season_year in season_years
    ]
    if not frames:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
    return apply_tipoff_corrections(out)
