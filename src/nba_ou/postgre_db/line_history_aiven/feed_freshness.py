"""Is SBR's line feed still moving? Checked before tonight's predictions.

SBR's feed sometimes freezes for the rest of a day: lines, scores and game
statuses all stop at one clock time, while the site still serves the frozen
state as if it were current (2024-03-28, 2026-03-23, 2026-04-20, 2026-05-01,
2026-10-20). Every input the prediction reads from SBR -- the day page's
current lines and the live line history -- then belongs to that moment, so the
predictions would be made on lines hours old.

The tell is silence. Measured 2026-10-21 over 632 game days (2023-24 to
2025-26): from 08:00 ET to the first tip-off, the longest stretch with no tick
at any book on any of the day's games is at most 115 minutes, median 10; every
freeze is 213+ minutes. Before 08:00 ET overnight quiet is normal, so the clock
starts there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    ScrapedGame,
    discover_games_for_date,
    new_session,
    scrape_events,
)

#: Minutes without a tick, counted from ``WATCH_START_HOUR_ET``, that mean the
#: feed is frozen rather than quiet.
FEED_STALE_MINUTES = 150

#: Hour (ET) from which a game day's silence counts.
WATCH_START_HOUR_ET = 8

EASTERN = "America/New_York"


@dataclass(frozen=True)
class FeedFreshness:
    """How long a line feed has been silent for one game day, as of ``as_of``.

    The feed is SBR's unless ``source`` says otherwise: Covers serves the same
    ticks (``fetch_data.odds_covers.live``), so the same rule judges both.
    """

    game_date: date
    as_of: pd.Timestamp
    latest_tick: pd.Timestamp | None
    #: Silence since the later of the latest tick and 08:00 ET; None when no
    #: game is still to tip, so there is nothing left to predict.
    quiet_minutes: float | None
    pending_games: int
    stale_minutes: int = FEED_STALE_MINUTES
    source: str = "SBR"

    @property
    def stale(self) -> bool:
        return (
            self.quiet_minutes is not None and self.quiet_minutes > self.stale_minutes
        )

    def describe(self) -> str:
        latest = (
            "none"
            if self.latest_tick is None
            else self.latest_tick.tz_convert(EASTERN).strftime("%Y-%m-%d %H:%M ET")
        )
        quiet = "n/a" if self.quiet_minutes is None else f"{self.quiet_minutes:.0f} min"
        return (
            f"{self.source} feed for {self.game_date}: latest tick {latest}, quiet {quiet} "
            f"(limit {self.stale_minutes}), {self.pending_games} game(s) still to tip"
        )


def assess_feed_freshness(
    tick_times: Sequence | pd.Series,
    tipoffs: Sequence | pd.Series,
    *,
    game_date: date,
    as_of: pd.Timestamp,
    stale_minutes: int = FEED_STALE_MINUTES,
    watch_start_hour_et: int = WATCH_START_HOUR_ET,
    source: str = "SBR",
) -> FeedFreshness:
    """Pure: the day's tick times and tip-offs -> how long the feed has been silent."""
    as_of = pd.Timestamp(as_of)
    as_of = (
        as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    )
    ticks = pd.to_datetime(pd.Series(list(tick_times), dtype="object"), utc=True)
    ticks = ticks[ticks <= as_of]
    tips = pd.to_datetime(pd.Series(list(tipoffs), dtype="object"), utc=True)
    pending = int((tips > as_of).sum())
    latest = None if ticks.empty else ticks.max()

    quiet = None
    if pending:
        watch_start = (
            pd.Timestamp(game_date).tz_localize(EASTERN)
            + pd.Timedelta(hours=watch_start_hour_et)
        ).tz_convert("UTC")
        reference = watch_start if latest is None else max(latest, watch_start)
        quiet = max((as_of - reference).total_seconds() / 60.0, 0.0)

    return FeedFreshness(
        game_date=game_date,
        as_of=as_of,
        latest_tick=latest,
        quiet_minutes=quiet,
        pending_games=pending,
        stale_minutes=stale_minutes,
        source=source,
    )


def freshness_from_scraped(
    games: Sequence[ScrapedGame], *, game_date: date, as_of: pd.Timestamp, **kwargs
) -> FeedFreshness:
    return assess_feed_freshness(
        [tick.line_ts for game in games for tick in game.ticks],
        [game.tipoff_utc for game in games],
        game_date=game_date,
        as_of=as_of,
        **kwargs,
    )


def check_live_feed(
    game_date: date, *, as_of: pd.Timestamp | None = None, **kwargs
) -> FeedFreshness:
    """Scrape ``game_date``'s games from SBR and assess the feed. One request per game."""
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    session = new_session()
    try:
        summaries = discover_games_for_date(session, game_date)
        games = list(scrape_events([s.event_id for s in summaries], session=session))
    finally:
        session.close()
    return freshness_from_scraped(games, game_date=game_date, as_of=as_of, **kwargs)
