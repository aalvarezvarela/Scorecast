"""Tonight's line history: read live for predictions, never stored until final.

Two halves of one rule. The store only takes finished games (``is_finished``),
so a partial pre-game history can never be persisted. The prediction job reads
that same history live (``live_frames_from_scraped``) in exactly the shape the
store would have returned it, and the builder splices it in
(``append_live_line_history``).
"""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from nba_ou.config.constants import TEAM_ID_MAP
from nba_ou.create_training_data.create_intermediate_line_df import (
    append_live_line_history,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)
from nba_ou.postgre_db.line_history_aiven import ingest as ing
from nba_ou.postgre_db.line_history_aiven.fetch import GAME_COLUMNS, TICK_COLUMNS
from nba_ou.postgre_db.line_history_aiven.live import (
    LiveLineHistory,
    live_frames_from_scraped,
    scheduled_team_games,
)

TIPOFF = datetime(2025, 4, 4, 23, 0, tzinfo=UTC)
BOOK_IDS = {"betmgm": 2, "fanduel": 6}


def _tick(minutes_to_tip: int, **overrides) -> LineTick:
    row = {
        "market": "totals",
        "book_slug": "betmgm",
        "book_name": "BetMGM",
        "line_ts": TIPOFF + pd.Timedelta(minutes=minutes_to_tip),
        "minutes_to_tip": minutes_to_tip,
        "is_opener": False,
        "left_line": 216.5,
        "left_price": -110,
        "right_line": 216.5,
        "right_price": -110,
    }
    row.update(overrides)
    return LineTick(**row)


def _game(status_text: str = "7:00 PM ET", ticks=None) -> ScrapedGame:
    return ScrapedGame(
        event_id=316538,
        game_date=date(2025, 4, 4),
        season_year=2024,
        tipoff_utc=TIPOFF,
        team_away="Sacramento Kings",
        team_home="Charlotte Hornets",
        status_text=status_text,
        away_score=None,
        home_score=None,
        ticks=tuple(ticks if ticks is not None else [_tick(-120)]),
    )


def _team_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022401118", "0022401118"],
            "game_date": [date(2025, 4, 4)] * 2,
            "team_name": ["Charlotte Hornets", "Sacramento Kings"],
            "home": [True, False],
            "season_year": [2024, 2024],
        }
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {"game_id": ["0022401118"], "team_home": ["CHA"], "team_away": ["SAC"],
         "tipoff_utc": [TIPOFF]}
    )


# --- the store never takes an unfinished game -------------------------------


@pytest.mark.parametrize(
    ("status", "finished"),
    [("Final", True), ("Final/OT", True), ("final ", True),
     ("7:30 PM ET", False), ("Q3 5:21", False), ("Postponed", False), ("", False)],
)
def test_only_final_games_count_as_finished(status, finished):
    assert ing.is_finished(_game(status)) is finished


def test_ingest_writes_only_finished_games(monkeypatch):
    """Tonight's partial history is dropped before anything is resolved or
    encoded, let alone written: it is stored the day after, whole."""
    monkeypatch.setattr(ing, "build_game_index", lambda games_df: pd.DataFrame(
        {"game_id": ["0022401118"], "game_date": [date(2025, 4, 4)],
         "game_season_year": [2024], "team_home": ["Charlotte Hornets"],
         "team_away": ["Sacramento Kings"]}))
    monkeypatch.setattr(ing, "_existing_books", lambda conn, slugs: BOOK_IDS)

    stats = ing.ingest_scraped_games(
        None, [_game("Final"), _game("7:30 PM ET")], games_df=pd.DataFrame(), dry_run=True
    )

    assert stats.dropped["unfinished_game"] == 1
    assert stats.matched_games == 1


def test_a_batch_of_only_unfinished_games_touches_nothing():
    # conn=None: any database access would raise.
    stats = ing.ingest_scraped_games(None, [_game("Q2 3:10")], games_df=pd.DataFrame())
    assert stats.dropped == {"unfinished_game": 1}
    assert stats.inserted_ticks == 0


# --- the live read looks exactly like a stored read -------------------------


def _live(game: ScrapedGame, as_of: datetime, **kwargs) -> LiveLineHistory:
    return live_frames_from_scraped(
        [game], team_games=_team_games(), book_ids=BOOK_IDS, as_of=pd.Timestamp(as_of),
        schedule=_schedule(), **kwargs,
    )


def test_live_ticks_have_the_stored_read_shape_and_decoding():
    live = _live(_game(), as_of=TIPOFF)

    assert list(live.ticks.columns) == TICK_COLUMNS
    assert list(live.games.columns) == GAME_COLUMNS
    row = live.ticks.iloc[0]
    assert (row.game_id, row.market, row.book) == ("0022401118", "totals", "betmgm")
    assert row.minutes_before_tip == 120  # positive minutes, as the store decodes
    assert (row.left_line, row.left_price) == (216.5, -110)
    # Plain numpy dtypes, as a psycopg read decodes -- not pandas nullables.
    assert str(live.ticks["left_line"].dtype) == "float64"
    assert str(live.ticks["left_price"].dtype) == "int64"
    # Team codes from the schedule feed, as the store writes them.
    assert live.games.loc[0, ["team_home", "team_away"]].tolist() == ["CHA", "SAC"]


def test_as_of_truncates_the_history_to_what_was_known():
    game = _game(ticks=[_tick(-600), _tick(-300), _tick(-60)])
    as_of = TIPOFF - pd.Timedelta(minutes=300)

    live = _live(game, as_of=as_of)

    assert sorted(live.ticks["minutes_before_tip"]) == [300, 600]


def test_same_leakage_margin_as_the_store_read():
    """Pre-game only, and nothing inside the 5-minute margin before tip."""
    game = _game(ticks=[_tick(-30), _tick(-3), _tick(4)])

    live = _live(game, as_of=TIPOFF + pd.Timedelta(hours=1))

    assert live.ticks["minutes_before_tip"].tolist() == [30]


def test_an_unfinished_game_is_read_live():
    live = _live(_game("7:00 PM ET"), as_of=TIPOFF)
    assert live.game_ids == ["0022401118"]


def test_scheduled_games_resolve_to_their_game_ids():
    scheduled = pd.DataFrame(
        {
            "GAME_ID": ["0022401118"],
            "GAME_DATE_EST": ["2025-04-04T00:00:00"],
            "HOME_TEAM_ID": [TEAM_ID_MAP["Charlotte Hornets"]],
            "VISITOR_TEAM_ID": [TEAM_ID_MAP["Sacramento Kings"]],
        }
    )
    team_games = scheduled_team_games(scheduled, season_year=2024)

    live = live_frames_from_scraped(
        [_game()], team_games=team_games, book_ids=BOOK_IDS, as_of=pd.Timestamp(TIPOFF)
    )

    assert live.game_ids == ["0022401118"]
    assert live.unmatched_games == []


# --- the builder splices it in ----------------------------------------------


def _stored_ticks(game_id: str, minutes: list[int], book: str = "betmgm") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": game_id, "season_year": 2024, "market": "totals", "book": book,
            "line_ts": [TIPOFF - pd.Timedelta(minutes=m) for m in minutes],
            "minutes_before_tip": [float(m) for m in minutes], "is_opener": False,
            "left_line": 216.5, "left_price": -110, "right_line": 216.5, "right_price": -110,
        },
        columns=TICK_COLUMNS,
    )


def _games(*game_ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"game_id": list(game_ids), "game_date": pd.Timestamp("2025-04-04"),
         "season_year": 2024, "tipoff_utc": TIPOFF, "team_home": "CHA", "team_away": "SAC"},
        columns=GAME_COLUMNS,
    )


def test_live_games_replace_stored_rows_and_keep_the_rest():
    stored = pd.concat([_stored_ticks("A", [600, 300, 60]), _stored_ticks("B", [90])])
    live = LiveLineHistory(
        ticks=pd.concat([_stored_ticks("A", [600]), _stored_ticks("A", [500], book="hardrock")]),
        games=_games("A"),
        as_of=pd.Timestamp(TIPOFF),
    )

    ticks, games = append_live_line_history(
        stored, _games("A", "B"), live, exclude_books=("hardrock",)
    )

    # A's stored full history is gone -- only the live, as-of view remains, and
    # the excluded book is filtered exactly as the store read filters it.
    assert ticks[ticks.game_id == "A"]["minutes_before_tip"].tolist() == [600.0]
    assert ticks[ticks.game_id == "B"]["minutes_before_tip"].tolist() == [90.0]
    assert sorted(games["game_id"]) == ["A", "B"]
