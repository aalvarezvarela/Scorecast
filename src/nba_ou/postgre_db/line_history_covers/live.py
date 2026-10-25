"""Tonight's Covers line history, read live as the prediction's SBR fallback.

When SBR's feed is down or frozen (``line_history_aiven.feed_freshness``), the
prediction reads tonight's lines from Covers instead: the same feed, the same
books. Nothing is stored or archived; this is the Covers twin of
``line_history_aiven.live``.

A day is fetched once (``fetch_covers_live_day``) and serves every reader:

* the feed check (``covers_freshness``),
* the closing frame's day-page odds (``fetch_data.odds_covers.day_odds``),
* the intermediate frame's line history (``covers_live_line_history``), which
  goes through SBR's own ``live_frames_from_scraped`` and so comes out exactly
  as SBR's live read does. Only SBR's books are kept: ``thescore`` and the
  rebuilt ``consensus_opener`` are not in the SBR store the models trained on.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime

import pandas as pd

from nba_ou.fetch_data.odds_covers.client import CoversClient, CoversError
from nba_ou.fetch_data.odds_covers.parse import (
    PickConsensus,
    parse_consensus,
    parse_matchups,
    scraped_game,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven import ingest as sbr_ingest
from nba_ou.postgre_db.line_history_aiven.feed_freshness import (
    FeedFreshness,
    freshness_from_scraped,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    ALL_MARKETS,
    DEFAULT_MIN_MINUTES_BEFORE_TIP,
    TICK_COLUMNS,
)
from nba_ou.postgre_db.line_history_aiven.live import (
    LiveLineHistory,
    live_frames_from_scraped,
    scheduled_team_games,
)

from . import schema as schema_mod
from .update import _resolve, feed_tipoffs

#: Covers books the SBR store does not hold, so no model has seen them.
COVERS_ONLY_BOOKS: tuple[str, ...] = ("consensus_opener", "thescore")

#: Status of a game read live: not final (``ingest.is_finished``).
LIVE_STATUS = "Scheduled"


@dataclass
class CoversLiveDay:
    """Tonight's Covers games by NBA GAME_ID, as fetched from ``as_of`` on.

    ``as_of`` is when the first request went out, the cut-off every reader
    applies: a tick fetched later is not used. ``missing`` says why a wanted
    game has no entry. ``consensus`` holds the pick counts by Covers id (the
    games' ``event_id``), when they were fetched.
    """

    game_date: date
    as_of: pd.Timestamp
    games: dict[str, ScrapedGame] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    unknown_books: set[int] = field(default_factory=set)
    consensus: dict[int, PickConsensus] = field(default_factory=dict)
    #: GAME_ID -> why its pick counts are missing.
    missing_consensus: dict[str, str] = field(default_factory=dict)

    def update(self, other: CoversLiveDay) -> None:
        """Add another fetch of the same day; the earlier cut-off stands."""
        self.games.update(other.games)
        self.consensus.update(other.consensus)
        self.missing_consensus.update(other.missing_consensus)
        self.missing = {
            game_id: why
            for game_id, why in {**self.missing, **other.missing}.items()
            if game_id not in self.games
        }
        self.unknown_books |= other.unknown_books
        self.as_of = min(self.as_of, other.as_of)


def fetch_covers_live_day(
    game_date: date,
    *,
    scheduled_games: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
    game_ids: Iterable[str] | None = None,
    client: CoversClient | None = None,
    consensus: bool = True,
) -> CoversLiveDay:
    """Fetch tonight's games from Covers: one request for the date, one per game
    (two with ``consensus``: its pick counts, for the closing frame).

    ``scheduled_games`` is ``get_schedule_games``'s frame; ``game_ids`` limits
    the fetch to some of its games. Tip-offs come from the season schedule
    feed (``schedule``, fetched when not given), as in the store. The games
    carry the schedule's team names, so they resolve and merge as SBR's do.
    A failure is recorded per game in ``missing``; nothing raises.
    """
    from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules

    as_of = pd.Timestamp.now(tz="UTC")
    season = season_year_for_date(game_date)
    team_games = scheduled_team_games(scheduled_games, season_year=season)
    names = {
        str(game_id): (
            group.loc[~group["home"], "team_name"].iloc[0],
            group.loc[group["home"], "team_name"].iloc[0],
        )
        for game_id, group in team_games.groupby("game_id")
    }
    wanted = sorted(names if game_ids is None else set(map(str, game_ids)) & set(names))
    day = CoversLiveDay(game_date=game_date, as_of=as_of)
    if not wanted:
        return day
    if schedule is None:
        schedule = fetch_schedules([season])
    tipoffs = feed_tipoffs(schedule)
    lookup = sbr_ingest.build_game_lookup(sbr_ingest.build_game_index(team_games))

    owned = client is None
    client = client or CoversClient()
    try:
        try:
            listed = _resolve(
                parse_matchups(client.matchups(game_date), game_date), lookup
            )
        except CoversError as exc:
            day.missing = {
                game_id: f"matchups page failed: {exc}" for game_id in wanted
            }
            return day
        for game_id in wanted:
            game = listed.get(game_id)
            tipoff = tipoffs.get(game_id)
            if game is None:
                day.missing[game_id] = "not on the Covers matchups page"
                continue
            if tipoff is None:
                day.missing[game_id] = "no schedule-feed tipoff"
                continue
            try:
                text = client.line_history(game.covers_id)
            except CoversError as exc:
                day.missing[game_id] = f"request failed: {exc}"
                continue
            try:
                parsed = scraped_game(
                    game,
                    text=text,
                    tipoff_utc=tipoff,
                    fetched_at=datetime.now(UTC),
                    status_text=LIVE_STATUS,
                )
            except Exception as exc:  # one odd page must not cost the others
                day.missing[game_id] = f"parse error: {type(exc).__name__}: {exc}"
                continue
            day.unknown_books |= parsed.unknown_books
            if not parsed.game.ticks:
                day.missing[game_id] = "no line history"
                continue
            away, home = names[game_id]
            day.games[game_id] = replace(parsed.game, team_away=away, team_home=home)
            if consensus:
                _fetch_consensus(client, day, game_id, game.covers_id)
    finally:
        if owned:
            client.close()
    return day


def _fetch_consensus(
    client: CoversClient, day: CoversLiveDay, game_id: str, covers_id: int
) -> None:
    """The game's pick counts into ``day``; a failure costs only them."""
    try:
        counts = parse_consensus(client.consensus(covers_id))
    except CoversError as exc:
        day.missing_consensus[game_id] = f"request failed: {exc}"
        return
    if counts is None:
        day.missing_consensus[game_id] = "no pick counts on the page"
        return
    day.consensus[int(covers_id)] = counts


def covers_freshness(day: CoversLiveDay, **kwargs) -> FeedFreshness:
    """``feed_freshness``'s verdict on the Covers feed, as of the day's fetch."""
    return freshness_from_scraped(
        list(day.games.values()),
        game_date=day.game_date,
        as_of=day.as_of,
        source="Covers",
        **kwargs,
    )


def covers_live_line_history(
    day: CoversLiveDay,
    *,
    team_games: pd.DataFrame,
    schedule: pd.DataFrame | None = None,
    game_ids: Iterable[str] | None = None,
    as_of: pd.Timestamp | None = None,
    markets: tuple[str, ...] = ALL_MARKETS,
    exclude_books: tuple[str, ...] = (),
    min_minutes_before_tip: int = DEFAULT_MIN_MINUTES_BEFORE_TIP,
) -> LiveLineHistory:
    """The day's games as ``fetch_live_line_history`` returns SBR's. Pure.

    ``as_of`` defaults to the fetch's cut-off and is never later than it.
    """
    wanted = None if game_ids is None else set(map(str, game_ids))
    games = [
        game
        for game_id, game in sorted(day.games.items())
        if wanted is None or game_id in wanted
    ]
    as_of = day.as_of if as_of is None else min(pd.Timestamp(as_of), day.as_of)
    return live_frames_from_scraped(
        games,
        team_games=team_games,
        book_ids=schema_mod.book_ids(),
        as_of=as_of,
        schedule=schedule,
        markets=markets,
        exclude_books=(*COVERS_ONLY_BOOKS, *exclude_books),
        min_minutes_before_tip=min_minutes_before_tip,
    )


def games_with_ticks(history: LiveLineHistory) -> set[str]:
    return set(history.ticks["game_id"].astype(str))


def freshest_minutes_before_tip(history: LiveLineHistory) -> dict[str, float]:
    """Per game, how long before tip its latest tick came."""
    if history.ticks.empty:
        return {}
    freshest = history.ticks.groupby(history.ticks["game_id"].astype(str))[
        "minutes_before_tip"
    ].min()
    return {str(game_id): float(minutes) for game_id, minutes in freshest.items()}


def truncated_game_ids(history: LiveLineHistory, as_of: pd.Timestamp) -> set[str]:
    """Games already tipped at ``as_of`` whose history stops well before tip.

    ``ingest.is_truncated``'s rule, the signature of a feed freeze: the latest
    pre-tip tick more than ``TRUNCATED_HISTORY_MINUTES`` before tip-off. Only
    a tipped game can tell; before tip a quiet game is just quiet.
    """
    if history.games.empty:
        return set()
    as_of = pd.Timestamp(as_of)
    as_of = (
        as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    )
    tipoffs = pd.to_datetime(history.games["tipoff_utc"], utc=True)
    tipped = set(history.games.loc[tipoffs <= as_of, "game_id"].astype(str))
    freshest = freshest_minutes_before_tip(history)
    return {
        game_id
        for game_id in tipped
        if freshest.get(game_id, 0.0) > sbr_ingest.TRUNCATED_HISTORY_MINUTES
    }


def fill_live_line_history(
    primary: LiveLineHistory, fallback: LiveLineHistory, game_ids: Iterable[str]
) -> LiveLineHistory:
    """``primary`` with ``game_ids`` (games and ticks) taken from ``fallback``.

    In the store's read order, so downstream groupbys see the sequence a
    single read gives.
    """
    ids = set(map(str, game_ids))
    if not ids:
        return primary
    taken_ticks = fallback.ticks[fallback.ticks["game_id"].astype(str).isin(ids)]
    kept_ticks = primary.ticks[~primary.ticks["game_id"].astype(str).isin(ids)]
    ticks = taken_ticks if kept_ticks.empty else pd.concat([kept_ticks, taken_ticks])
    ticks = ticks.sort_values(["game_id", "market", "book", "line_ts"])
    taken_games = fallback.games[fallback.games["game_id"].astype(str).isin(ids)]
    kept_games = primary.games[~primary.games["game_id"].astype(str).isin(ids)]
    games = taken_games if kept_games.empty else pd.concat([kept_games, taken_games])
    games = games.sort_values(["game_date", "game_id"])
    return LiveLineHistory(
        ticks.reset_index(drop=True).reindex(columns=TICK_COLUMNS),
        games.reset_index(drop=True),
        primary.as_of,
        sorted(set(primary.unmatched_games) | set(fallback.unmatched_games)),
    )
