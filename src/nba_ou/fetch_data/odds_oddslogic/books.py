"""OddsLogic sportsbook ids -> the slugs of the line-history store.

Books SBR also carries reuse SBR's slug (``fanatics_sportsbook``,
``hard_rock_bet``, ...), so a later SBR/OddsLogic merge lands on the same book.
The id is stored as ``lh_book.book_id``, which keeps ids identical across
reloads without the first-seen numbering SBR's store needs.

The free widget lists all of these except 60. That one answers on the
line-history endpoint without a name; matched against SBR on 40 games of
2025-26 it is BetRivers (as-of line 0.88, line moves 0.87, against 0.10 or less
for every other book). The widget's other unnamed ids are copies of the
market or of DraftKings/Fanatics and are not stored. bet365 is not among them.
"""

from __future__ import annotations

#: OddsLogic sportsbook id -> (store slug, display name).
OL_BOOKS: dict[int, tuple[str, str]] = {
    109: ("draftkings", "DraftKings"),
    112: ("fanduel", "FanDuel"),
    106: ("betmgm", "BetMGM"),
    107: ("caesars", "Caesars"),
    111: ("fanatics_sportsbook", "Fanatics Sportsbook"),
    113: ("hard_rock_bet", "Hard Rock Bet"),
    60: ("betrivers", "BetRivers"),
    110: ("espn_bet", "ESPN BET"),
    108: ("circa", "Circa"),
    117: ("south_point", "South Point"),
    139: ("kalshi", "Kalshi"),
    # OddsLogic's own market consensus, not a book.
    115: ("oddslogic_consensus", "OddsLogic Consensus"),
    # The opening line ("Open" column), the analogue of SBR's opener.
    116: ("consensus_opener", "Consensus Opener"),
}

#: Books mapped above that the daily update does not request, with why. Still
#: registered in ``lh_book`` and fetchable with ``--books``.
INACTIVE_BOOKS: dict[int, str] = {
    60: (
        "BetRivers has no 2026-27 lines on the widget: only August-September "
        "quotes on the three 2026-10-20 games (checked 2026-10-23)"
    ),
}

#: Books the daily update requests.
ACTIVE_BOOKS: tuple[int, ...] = tuple(
    book_id for book_id in OL_BOOKS if book_id not in INACTIVE_BOOKS
)

#: Public-betting series (share of tickets / of money). Not fetched: their
#: cells mix markets and sides, the format is not decoded, and their answers
#: are the largest (200-450 KB per game). Kept here to document the ids.
PUBLIC_FEEDS: dict[int, str] = {25: "tickets", 26: "money"}


def book_slug(book_id: int) -> str:
    return OL_BOOKS[int(book_id)][0]


def book_name(book_id: int) -> str:
    return OL_BOOKS[int(book_id)][1]
