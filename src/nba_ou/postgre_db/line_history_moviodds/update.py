"""Daily, insert-only MoviOdds ingestion. No serving or training integrations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from nba_ou.fetch_data.odds_moviodds.archive import bundle_key
from nba_ou.fetch_data.odds_moviodds.client import MoviOddsError, utc_timestamp
from nba_ou.fetch_data.odds_moviodds.parse import history_ticks, team_code
from nba_ou.postgre_db.line_history_aiven.schema import market_ids
from nba_ou.postgre_db.line_history_aiven.transform import (
    GAME_DIM_COLUMNS,
    OUTPUT_COLUMNS,
    _encode_line,
    _encode_price,
)

from . import schema, store

FIRST_DATE = date(2026, 10, 20)
DEFAULT_RETRY_DAYS = 3
EASTERN = ZoneInfo("America/New_York")


def today_eastern(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(EASTERN).date()


def game_targets(games_df: pd.DataFrame, schedule: pd.DataFrame | None) -> pd.DataFrame:
    """Played nba_games joined to unambiguous NBA clocks and confirmed teams.

    Keep rows lacking a clock so the daily report can distinguish a feed gap
    from no work. Never substitute MoviOdds' approximate kickoff.
    """
    # Preserve competing identities until after the join; the shared SBR
    # index drops duplicates by matchup before we can reject ambiguous ids.
    columns = ["game_id", "game_date", "season_year", "team_home", "team_away"]
    if games_df.empty:
        targets = pd.DataFrame(columns=columns)
    else:
        games = games_df.copy()
        games["game_id"] = games["game_id"].astype(str)
        games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
        home = games[games["home"].eq(True)].rename(columns={"team_name": "team_home"})
        away = games[games["home"].eq(False)].rename(columns={"team_name": "team_away"})
        targets = home[["game_id", "game_date", "season_year", "team_home"]].merge(
            away[["game_id", "game_date", "team_away"]], on=["game_id", "game_date"]
        )
    targets["tipoff_utc"] = pd.NaT
    targets["tipoff_utc"] = pd.to_datetime(targets["tipoff_utc"], utc=True)
    for side in ("team_home", "team_away"):
        targets[side] = targets[side].map(team_code)
    targets = targets.dropna(
        subset=["team_home", "team_away", "game_date", "season_year"]
    )
    # Multiple different identities for one NBA id are refused, not first-wins.
    targets = targets.drop_duplicates()
    targets = targets[~targets["game_id"].duplicated(keep=False)].copy()
    targets = targets[
        ~targets.duplicated(["game_date", "team_home", "team_away"], keep=False)
    ].copy()
    if schedule is None or schedule.empty:
        return targets
    grouped = {str(gid): frame for gid, frame in schedule.groupby("game_id")}
    for idx, target in targets.iterrows():
        clocks = grouped.get(str(target.game_id))
        if clocks is None:
            continue
        valid = []
        for row in clocks.itertuples(index=False):
            try:
                tip = utc_timestamp(row.tipoff_utc)
            except (ValueError, TypeError):
                continue
            if (team_code(row.team_home), team_code(row.team_away)) != (
                target.team_home,
                target.team_away,
            ):
                continue
            if pd.Timestamp(row.game_date).date() != target.game_date:
                continue
            valid.append(tip)
        if len(set(valid)) == 1:
            targets.at[idx, "tipoff_utc"] = valid[0]
    return targets.reset_index(drop=True)


def plan_requests(
    targets: pd.DataFrame,
    stored: dict,
    *,
    today: date,
    retry_days: int = DEFAULT_RETRY_DAYS,
    dates: list[date] | None = None,
) -> pd.DataFrame:
    if retry_days < 0:
        raise ValueError("retry_days must be non-negative")
    wanted = []
    for row in targets.itertuples(index=False):
        if not FIRST_DATE <= row.game_date < today:
            continue
        if dates is not None:
            if row.game_date in dates:
                wanted.append(row.game_id)
        elif row.game_id not in stored or (
            not stored[row.game_id][1]
            and row.game_date >= today - timedelta(days=retry_days)
        ):
            wanted.append(row.game_id)
    return (
        targets[targets["game_id"].isin(wanted)]
        .sort_values(["game_date", "game_id"])
        .reset_index(drop=True)
    )


def build_frames(target, event_id: str, ticks) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = int(event_id)
    if not 0 < event <= 2**31 - 1:
        raise ValueError("MoviOdds event_id does not fit the shared INTEGER column")
    dim = pd.DataFrame(
        [
            {
                "game_id": str(target.game_id),
                "game_date": target.game_date,
                "season_year": int(target.season_year),
                "tipoff_utc": target.tipoff_utc,
                "event_id": event,
                "team_home": target.team_home,
                "team_away": target.team_away,
            }
        ],
        columns=GAME_DIM_COLUMNS,
    )
    mids = market_ids()
    rows = pd.DataFrame(
        [
            {
                "game_id": str(target.game_id),
                "season_year": int(target.season_year),
                "market_id": mids[t.market],
                "book_id": 1,
                "line_ts": t.line_ts,
                "mins_to_tip": t.minutes_to_tip,
                "is_pregame": t.is_pregame,
                "is_opener": t.is_opener,
                "left_line": t.left_line,
                "left_price": t.left_price,
                "right_line": t.right_line,
                "right_price": t.right_price,
            }
            for t in ticks
        ],
        columns=OUTPUT_COLUMNS,
    )
    if not rows.empty:
        for column in ("left_line", "right_line"):
            rows[column] = _encode_line(rows[column])
        for column in ("left_price", "right_price"):
            rows[column] = _encode_price(rows[column])
    return rows, dim


@dataclass
class UpdateResult:
    planned_games: int = 0
    fetched_games: int = 0
    inserted_games: int = 0
    inserted_ticks: int = 0
    archive_failures: int = 0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    frames: list[tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)
    repaired: Counter = field(default_factory=Counter)
    seasons: dict[int, dict] = field(default_factory=dict)


def update_moviodds_line_history(
    plan: pd.DataFrame,
    *,
    stored: dict,
    client,
    conn=None,
    archive=None,
    dry_run=False,
    now: datetime | None = None,
    progress=print,
) -> UpdateResult:
    now = utc_timestamp(now or datetime.now(UTC))
    result = UpdateResult(planned_games=len(plan))

    def archive_records(day, label, records, **metadata):
        if dry_run or archive is None:
            return
        try:
            fetched = datetime.now(UTC)
            archive.put(
                bundle_key(day, label, fetched),
                {
                    "format": 1,
                    "fetched_at": fetched.isoformat(),
                    **metadata,
                    "requests": records,
                },
            )
        except Exception as exc:
            result.archive_failures += 1
            result.warnings.append(f"archive {label}: {type(exc).__name__}: {exc}")

    results_by_day = {}
    for day in sorted(
        {
            d
            for g in plan.itertuples()
            for d in (g.game_date, g.game_date + timedelta(days=1))
        }
    ):
        before = len(client.records)
        try:
            payload = client.results(day)
            results_by_day[day] = payload["results"]
            if payload.get("source") == "memory":
                result.warnings.append(
                    f"{day}: results fell back to memory; older fixtures may be absent"
                )
        except (MoviOddsError, ValueError, TypeError, KeyError) as exc:
            result.failures.append(f"{day} results: {exc}")
            results_by_day[day] = None
        finally:
            archive_records(day, "results", client.records[before:])

    for target in plan.itertuples(index=False):
        gid = str(target.game_id)
        if pd.isna(target.tipoff_utc):
            result.warnings.append(
                f"{gid}: missing authoritative NBA tipoff; postponed"
            )
            continue
        tip = utc_timestamp(target.tipoff_utc)
        if tip >= now:
            result.warnings.append(f"{gid}: game not started; postponed")
            continue
        if gid in stored and utc_timestamp(stored[gid][0]) != tip:
            result.warnings.append(
                f"{gid}: stored tipoff differs from NBA clock; postponed"
            )
            continue
        candidates = {}
        days = (target.game_date, target.game_date + timedelta(days=1))
        if any(results_by_day[d] is None for d in days):
            result.warnings.append(f"{gid}: incomplete results discovery; postponed")
            continue
        for day in days:
            for match in results_by_day[day]:
                if not isinstance(match, dict):
                    continue
                if match.get("league") != "NBA" or match.get("status") != "finished":
                    continue
                if (team_code(match.get("home")), team_code(match.get("away"))) != (
                    target.team_home,
                    target.team_away,
                ):
                    continue
                try:
                    finished = utc_timestamp(match.get("finished_at"))
                except (ValueError, TypeError):
                    continue
                if (
                    match.get("fixture_id")
                    and tip < finished <= now
                    and finished - tip < pd.Timedelta(days=1)
                ):
                    candidates[str(match["fixture_id"])] = match
        if not candidates:
            result.warnings.append(f"{gid}: no finished NBA fixture matched; postponed")
            continue
        fixture = min(candidates, key=lambda v: int(v) if v.isdigit() else float("inf"))
        before = len(client.records)
        metadata = {
            "game_id": gid,
            "game_date": str(target.game_date),
            "nba_tipoff_utc": tip.isoformat(),
            "candidates": list(candidates.values()),
        }
        try:
            history = client.history(fixture)
            result.fetched_games += 1
            metadata["fixture_ids"] = history.fixture_ids
            if not set(candidates) <= set(history.fixture_ids):
                raise ValueError("Ambiguous fixtures not linked by the API")
            parsed = history_ticks(
                history.points,
                home=target.team_home,
                away=target.team_away,
                tipoff_utc=tip,
            )
            result.dropped.update(parsed.dropped)
            result.repaired.update(parsed.repaired)
            if not parsed.ticks:
                raise ValueError("No identifiable main-market history; postponed")
            valid_markets = {
                t.market
                for t in parsed.ticks
                if t.is_pregame
                and t.left_price is not None
                and t.right_price is not None
            }
            if len(valid_markets) < 3:
                result.warnings.append(
                    f"{gid}: incomplete markets {sorted(valid_markets)}; eligible for recent-game retry"
                )
            rows, games = build_frames(target, fixture, parsed.ticks)
            result.frames.append((rows, games))
            counts = result.seasons.setdefault(
                int(target.season_year),
                {
                    "source_rows": 0,
                    "loaded_rows": 0,
                    "dropped": Counter(),
                },
            )
            counts["source_rows"] += len(history.points)
            counts["dropped"].update(parsed.dropped)
            if not dry_run and conn is not None:
                schema.create_season_partition(conn, int(target.season_year))
                # A failed COPY must not leave a stored game with no history.
                with conn.transaction():
                    inserted_games = store.insert_games(conn, games, commit=False)
                    inserted_ticks = store.insert_lines(
                        conn, rows, int(target.season_year), commit=False
                    )
                result.inserted_games += inserted_games
                result.inserted_ticks += inserted_ticks
                counts["loaded_rows"] += inserted_ticks
            progress(
                f"  {target.game_date} {target.team_away}@{target.team_home} ({gid}): {len(rows)} ticks"
            )
        except Exception as exc:
            result.failures.append(
                f"{gid} fixture {fixture}: {type(exc).__name__}: {exc}"
            )
        finally:
            archive_records(
                target.game_date, fixture, client.records[before:], **metadata
            )
    return result
