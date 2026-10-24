"""Covers sportsbook ids -> the slugs of the line-history store.

The ids are the ``data-book-id`` of each book's table in the line-history
answer, for the US view (``countryCode=us``; every state gives the same nine
books). Books SBR also carries reuse SBR's slug, so a later SBR/Covers merge
lands on the same book. The id is stored as ``lh_book.book_id``, as in the
OddsLogic store.

``consensus_opener`` is not a Covers book: it is SBR's opener column, which is
BetMGM's opening quote (``parse.consensus_opener_ticks``), stored under a
synthetic id no Covers book uses.
"""

from __future__ import annotations

#: Synthetic book id of the rebuilt SBR opener. Covers ids run below 600.
CONSENSUS_OPENER_ID = 1000

#: Covers sportsbook id -> (store slug, display name).
COVERS_BOOKS: dict[int, tuple[str, str]] = {
    1: ("bet365", "bet365"),
    26: ("betmgm", "BetMGM"),
    18: ("draftkings", "DraftKings"),
    19: ("fanduel", "FanDuel"),
    142: ("fanatics_sportsbook", "Fanatics Sportsbook"),
    29: ("betrivers", "BetRivers"),
    573: ("caesars", "Caesars"),
    6: ("hard_rock_bet", "Hard Rock Bet"),
    # Not on SBR.
    44: ("thescore", "theScore Bet"),
    CONSENSUS_OPENER_ID: ("consensus_opener", "Consensus Opener"),
}

#: The book the consensus opener is rebuilt from.
CONSENSUS_SOURCE_ID = 26

#: Books a stored game is expected to hold; a recent game missing one is
#: fetched again (``line_history_covers.update.plan_requests``).
ACTIVE_BOOKS: tuple[int, ...] = tuple(COVERS_BOOKS)


def book_slug(book_id: int) -> str:
    return COVERS_BOOKS[int(book_id)][0]


def book_name(book_id: int) -> str:
    return COVERS_BOOKS[int(book_id)][1]
