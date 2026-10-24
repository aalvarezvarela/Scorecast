"""Where tonight's lines come from: SBR, or Covers when SBR is down or frozen.

SBR's feed sometimes freezes for the rest of a day, or serves its games with no
odds at all (``line_history_aiven.feed_freshness``). Covers serves the same
feed for the same books (``docs/odds/covers_line_history.md``), so a run that
finds SBR silent reads tonight's lines from Covers instead of predicting
nothing.

The source is decided once per run, before anything reads a line
(``resolve_line_source``):

* SBR fresh -> ``sbr``. A game SBR lists without lines, or not at all, is
  still filled from Covers by the readers (``fill_ids``).
* SBR frozen, or its check failing -> Covers is fetched and checked by the same
  rule. Fresh -> ``covers`` for every game. Frozen too -> ``stale``: the run
  stops as it always did, unless stale lines are allowed (then SBR, as before).
  A failed check with no usable Covers carries on with SBR, unchecked.

A reader that finds SBR empty when it was chosen (the day page with no odds)
switches the run to Covers there (``switch_to_covers``). Every frame row then
records the source its lines came from (``LINE_SOURCE_COLUMN``), and the
prediction store keeps it per input row (``prediction.inputs.line_source``).

Covers is fetched once per game for the feed check and the closing frame
(``covers_day``); the intermediate frame reads it again, live, as SBR's live
line history is read at that moment too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date

import pandas as pd
from nba_ou.postgre_db.line_history_aiven.feed_freshness import (
    FeedFreshness,
    check_live_feed,
)
from nba_ou.postgre_db.line_history_covers.live import (
    CoversLiveDay,
    covers_freshness,
    fetch_covers_live_day,
)
from nba_ou.utils.ci import annotate

SBR = "sbr"
COVERS = "covers"

#: Frame column naming each row's line source (``SBR`` or ``COVERS``).
LINE_SOURCE_COLUMN = "LINE_SOURCE"

CoversFetcher = Callable[..., CoversLiveDay]


class LineSource:
    """The run's line source and the Covers fetch its readers share."""

    def __init__(
        self,
        game_date: date,
        *,
        source: str = SBR,
        reason: str | None = None,
        sbr_freshness: FeedFreshness | None = None,
        fetch_covers: CoversFetcher = fetch_covers_live_day,
    ) -> None:
        if source not in (SBR, COVERS):
            raise ValueError(f"Unknown line source: {source!r}")
        self.game_date = game_date
        self.source = source
        self.reason = reason
        self.sbr_freshness = sbr_freshness
        self.covers_freshness: FeedFreshness | None = None
        #: Both feeds frozen: nothing fresh to predict on.
        self.stale = False
        #: GAME_ID -> source of its day-page odds, set by the closing reader.
        self.closing_sources: dict[str, str] = {}
        #: GAME_ID -> source of its live line history, set by the intermediate
        #: reader.
        self.history_sources: dict[str, str] = {}
        self._fetch_covers = fetch_covers
        self._covers: CoversLiveDay | None = None
        self._requested: set[str] = set()

    @property
    def uses_covers(self) -> bool:
        return self.source == COVERS

    def switch_to_covers(self, reason: str) -> None:
        """Read every game from Covers from here on (SBR came back empty)."""
        if self.uses_covers:
            return
        annotate("warning", f"{reason}; using Covers lines for this run.")
        self.source = COVERS
        self.reason = reason

    def covers_day(
        self,
        scheduled_games: pd.DataFrame,
        *,
        game_ids: Iterable[str] | None = None,
        schedule: pd.DataFrame | None = None,
        refresh: bool = False,
    ) -> CoversLiveDay:
        """Covers' games for ``game_ids`` (all scheduled: None).

        Each game is fetched once, with its pick counts (the closing frame's
        consensus), unless ``refresh`` asks for its lines as of now; the
        returned day then holds only the refetched games, with their own
        cut-off and no pick counts (the intermediate frame has no use for
        them).
        """
        wanted = set(
            scheduled_games["GAME_ID"].astype(str)
            if game_ids is None
            else map(str, game_ids)
        )
        todo = sorted(wanted if refresh else wanted - self._requested)
        if todo or self._covers is None:
            fetched = self._fetch_covers(
                self.game_date,
                scheduled_games=scheduled_games,
                schedule=schedule,
                game_ids=todo,
                consensus=not refresh,
            )
            for game_id, why in fetched.missing.items():
                annotate("warning", f"Covers has no lines for {game_id}: {why}")
            for game_id, why in fetched.missing_consensus.items():
                annotate("warning", f"Covers has no pick counts for {game_id}: {why}")
            if refresh or self._covers is None:
                self._covers = fetched
                self._requested = set(todo)
            else:
                self._covers.update(fetched)
                self._requested |= set(todo)
        return self._covers

    def describe(self) -> str:
        text = f"Lines from {self.source.upper()}"
        return f"{text} ({self.reason})" if self.reason else text


def resolve_line_source(
    game_date: date,
    *,
    load_scheduled_games: Callable[[], pd.DataFrame],
    check_sbr: Callable[[date], FeedFreshness] = check_live_feed,
    fetch_covers: CoversFetcher = fetch_covers_live_day,
) -> LineSource:
    """Decide the run's line source; Covers is only fetched if SBR is not fresh."""
    try:
        sbr = check_sbr(game_date)
    except Exception as exc:  # noqa: BLE001 -- the check must not cost the run
        sbr, why = None, f"SBR feed check failed: {exc}"
    else:
        if not sbr.stale:
            return LineSource(game_date, sbr_freshness=sbr, fetch_covers=fetch_covers)
        why = f"SBR line feed looks frozen: {sbr.describe()}"

    decision = LineSource(
        game_date,
        source=COVERS,
        reason=why,
        sbr_freshness=sbr,
        fetch_covers=fetch_covers,
    )
    try:
        scheduled_games = load_scheduled_games()
        day = decision.covers_day(scheduled_games)
    except Exception as exc:  # noqa: BLE001 -- judged below like an empty day
        annotate("warning", f"Covers fallback failed: {exc}")
        day = None
    if day is not None and day.games:
        decision.covers_freshness = covers_freshness(day)
        if not decision.covers_freshness.stale:
            return decision
        why = f"{why}; {decision.covers_freshness.describe()}"
    else:
        why = f"{why}; Covers has no lines either"
    decision.source = SBR
    decision.reason = why
    # A failed SBR check never stopped a run; only a frozen feed does.
    decision.stale = sbr is not None
    return decision


def check_line_feed(
    game_date: date,
    *,
    allow_stale_lines: bool,
    load_scheduled_games: Callable[[], pd.DataFrame],
    report: Callable[..., None] = print,
    force: str | None = None,
    **kwargs,
) -> LineSource:
    """``resolve_line_source`` for a prediction script, with its exit rule.

    Prints each feed's verdict through ``report(message, ok=...)``. Both feeds
    frozen stops the run with exit 1 unless ``allow_stale_lines``.
    ``force="covers"`` reads Covers without checking either feed (for tests of
    the fallback on a day SBR is fine).
    """
    if force == COVERS:
        decision = LineSource(game_date, source=COVERS, reason="forced", **kwargs)
        report(decision.describe(), ok=False)
        return decision
    if force not in (None, "auto"):
        raise ValueError(f"Unknown forced line source: {force!r}")
    decision = resolve_line_source(
        game_date, load_scheduled_games=load_scheduled_games, **kwargs
    )
    for freshness in (decision.sbr_freshness, decision.covers_freshness):
        if freshness is not None:
            report(freshness.describe(), ok=not freshness.stale)
    if decision.stale:
        message = f"{decision.reason}. Predictions would use stale lines"
        if not allow_stale_lines:
            annotate(
                "error",
                f"{message}; nothing predicted (--allow-stale-lines overrides).",
            )
            raise SystemExit(1)
        annotate("warning", f"{message}; predicting anyway (--allow-stale-lines).")
    elif decision.uses_covers:
        annotate("warning", f"{decision.reason}; using Covers lines for this run.")
    elif decision.reason:
        annotate("warning", f"{decision.reason}, continuing unchecked.")
    report(decision.describe(), ok=not decision.uses_covers)
    return decision


def fill_ids(scheduled_ids: Iterable[str], served_ids: Iterable[str]) -> list[str]:
    """Scheduled games the chosen source did not serve."""
    served = set(map(str, served_ids))
    return sorted(set(map(str, scheduled_ids)) - served)


def priced_game_ids(day_odds: pd.DataFrame) -> set[str]:
    """Games of a day-page frame with at least one book's totals line.

    SBR can list a day's games with every odds cell empty (2026-10-24): such a
    game is not served, whatever its row says.
    """
    if day_odds is None or day_odds.empty or "game_id" not in day_odds.columns:
        return set()
    lines = [
        column
        for column in day_odds.columns
        if column.startswith("total_")
        and "_line_" in column
        and "consensus" not in column
    ]
    if not lines:
        return set()
    priced = day_odds[lines].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    return set(day_odds.loc[priced, "game_id"].astype(str))
