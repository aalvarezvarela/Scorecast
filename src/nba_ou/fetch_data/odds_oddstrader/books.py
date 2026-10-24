"""OddsTrader ``paid`` (provider account id) -> the book slug used in ``lh_book``.

Books that SBR also carries take SBR's slug (``slugify_bookmaker`` of SBR's
display name), so a future merge lands them on the same ``lh_book`` row. Every
other book gets the slug of its OddsTrader name.

The names come from two places, both read on 2026-10-22:

* ``sportsbook(sbid: X) { paid nam }`` scanned over sbid 1-4000. Several sbids
  can point at one ``paid`` (per-state feeds of the US books, rebrands), so the
  name below is the one that is the book's identity, not always the first hit.
* ``docs/odds/odds_source_alternatives.md`` for the US books (101 is the old William
  Hill US feed, i.e. Caesars; 122 is SBK).

A ``paid`` not listed here gets ``oddstrader_<paid>``. That keeps every row
stored -- the raw archive carries the ``paid`` -- and naming it later is a
rebuild from raw, not a refetch.
"""

from __future__ import annotations

#: paid -> (slug, display name)
PAID_BOOKS: dict[int, tuple[str, str]] = {
    # US regulated books: SBR's slugs where SBR carries the book.
    106: ("bet365", "bet365"),
    99: ("draftkings", "DraftKings"),
    98: ("fanduel", "FanDuel"),
    109: ("betmgm", "BetMGM"),
    113: ("betrivers", "BetRivers"),
    101: ("caesars", "Caesars"),
    102: ("sugarhouse", "SugarHouse"),
    107: ("unibet", "Unibet"),
    115: ("pointsbet", "PointsBet"),
    122: ("sbk", "SBK"),
    # bet365's international feed: a different market from bet365 US.
    5: ("bet365_international", "bet365 (international)"),
    # Offshore and international books.
    3: ("5dimes", "5Dimes"),
    4: ("abcislands", "ABCislands"),
    8: ("betonline", "BetOnline"),
    9: ("bovada", "Bovada"),
    10: ("bookmaker", "Bookmaker"),
    11: ("betdsi", "BetDSI"),
    14: ("betjamaica", "BetJamaica"),
    16: ("justbet", "JustBet"),
    17: ("legends", "Legends"),
    18: ("matchbook", "Matchbook"),
    20: ("pinnacle", "Pinnacle"),
    25: ("wsex", "WSEX"),
    26: ("canbet", "Canbet"),
    27: ("internationalallsports", "InternationalAllSports"),
    28: ("betphoenix", "BetPhoenix"),
    29: ("intertops", "Intertops"),
    31: ("rebatewager", "RebateWager"),
    33: ("betus", "BetUS"),
    35: ("sportsinteraction", "SportsInteraction"),
    36: ("william_hill", "William Hill"),
    38: ("youwager", "YouWager"),
    40: ("888sport", "888Sport"),
    42: ("betislands", "BetIslands"),
    43: ("ladbrokes", "Ladbrokes"),
    44: ("heritage_sports", "Heritage Sports"),
    45: ("sbr_sportsbook", "SBR Sportsbook"),
    47: ("bwin", "bwin"),
    52: ("sbobet", "SBOBet"),
    54: ("wagerweb", "WagerWeb"),
    55: ("marathonbet", "MarathonBet"),
    57: ("titanbet", "TitanBet"),
    63: ("10bet", "10bet"),
    65: ("gtbets", "GTbets"),
    66: ("1vice", "1Vice"),
    68: ("sportingbet", "SportingBet"),
    71: ("betway", "BetWay"),
    73: ("interwetten", "Interwetten"),
    74: ("betathome", "BetAtHome"),
    75: ("betclic", "BetClic"),
    76: ("unibet_international", "Unibet (international)"),
    77: ("coral", "Coral"),
    80: ("guts", "Guts"),
    82: ("mybookie", "MyBookie"),
    83: ("betmania", "BetMania"),
    84: ("skybook", "Skybook"),
    85: ("hrwager", "HRWager"),
    92: ("nitrogen_sports", "Nitrogen Sports"),
    93: ("cloudbet", "Cloudbet"),
    130: ("bet105", "Bet105"),
}


def book_for_paid(paid: int) -> tuple[str, str]:
    """(slug, display name) for an OddsTrader ``paid``."""
    paid = int(paid)
    return PAID_BOOKS.get(paid, (f"oddstrader_{paid}", f"OddsTrader book {paid}"))
