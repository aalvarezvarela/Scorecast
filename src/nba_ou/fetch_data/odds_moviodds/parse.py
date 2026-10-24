"""Reconstruct paired bet365 states without borrowing future quotes or lines.

Only stable selection identity (market/team/side) may be resolved from another
point of the same download. The handicap, price and availability always come
from the actual historical point. Unknown or conflicting identities invalidate
their market; a suspension/incoherent pair emits an all-null tick.

The provider sometimes relabels a few snapshots of a selection with the other
team's name while its handicap and price stay with the selection (seen in 3 of
the first 16 games, 1-6% of points). Such a selection keeps its usual identity
when that identity holds ``MIN_DOMINANT_SHARE`` of its labelled points and
every disagreeing point continues the selection's own numbers but not the
opposite selection's; otherwise the market is invalidated as before.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import groupby

from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import LineTick

from .client import MAIN_GROUP, utc_timestamp

MARKETS = {"1454": "totals", "1453": "point_spread", "960": "money_line"}
TEAM_NAMES = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}
_NAME_CODES = {name.casefold(): code for code, name in TEAM_NAMES.items()}
_ALIASES = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "PHO": "PHX"}
NULL_QUOTE = (None, None, None, None)

# Relabelled selections: share of the usual identity, and the largest jump in
# line (points) or implied probability from the nearest complete point under
# that identity. A disagreeing point must stay within both bounds of its own
# selection and exceed one of them against the opposite selection, so a
# near-even market, where both sides' numbers fit, stays ambiguous.
MIN_DOMINANT_SHARE = 0.9
MAX_LINE_JUMP = 2.0
MAX_PROB_JUMP = 0.10


def team_code(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    full = TEAM_NAME_STANDARDIZATION.get(text, text)
    if full and full.casefold() in _NAME_CODES:
        return _NAME_CODES[full.casefold()]
    if text.upper() in TEAM_NAMES:
        return text.upper()
    # The provider names teams "PHX Suns", "SA Spurs", "GS Warriors", etc.
    prefix = text.split(" ", 1)[0].upper()
    if prefix in _ALIASES:
        return _ALIASES[prefix]
    if prefix in TEAM_NAMES:
        return prefix
    return None


def american_odds(value) -> int | None:
    try:
        decimal = float(value)
        if not math.isfinite(decimal) or decimal <= 1:
            return None
        price = round(100 * (decimal - 1) if decimal >= 2 else -100 / (decimal - 1))
        return price if -32768 <= price <= 32767 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _identity(point: dict, home: str, away: str) -> tuple[str, str] | None:
    mid = str(point.get("market_id") or "")
    if mid and mid not in MARKETS:
        return None
    side, line = str(point.get("side") or ""), str(point.get("line") or "")
    label = str(point.get("label") or "")
    total_sides = set()
    for value in (side, line, label):
        if re.match(r"^(o|over)(?:\s|$)", value, re.I):
            total_sides.add("left")
        elif re.match(r"^(u|under)(?:\s|$)", value, re.I):
            total_sides.add("right")
    if len(total_sides) > 1:
        return None
    total_side = next(iter(total_sides), None)
    market = MARKETS.get(mid)
    if market is None:
        if total_side is not None or "total" in (side.casefold(), line.casefold()):
            market = "totals"
        elif {side.casefold(), line.casefold()} & {"line", "spread"}:
            market = "point_spread"
        elif {side.casefold(), line.casefold()} & {"to win", "money line"}:
            market = "money_line"
    if market == "totals":
        return (market, total_side) if total_side else None
    if market in ("point_spread", "money_line"):
        teams = {
            code for value in (side, line) if (code := team_code(value)) in (home, away)
        }
        if len(teams) == 1:
            return market, "right" if teams.pop() == home else "left"
    return None


def _historical_line(point: dict) -> float | None:
    value = point.get("handicap")
    if value is None or value == "":
        value = point.get("line")
        try:
            float(value)
        except (ValueError, TypeError):
            label = str(point.get("label") or "")
            match = re.fullmatch(
                r"(?:[oOuU]|[Oo]ver|[Uu]nder)?\s*([+-]?\d+(?:\.\d+)?)", label
            )
            value = match[1] if match else None
    try:
        number = 0.0 if str(value).casefold() in ("pk", "pick") else float(value)
        if (
            math.isfinite(number)
            and number * 2 == round(number * 2)
            and abs(number * 2) <= 32767
        ):
            return number
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _quote(point: dict, market: str) -> tuple[float | None, int | None]:
    if point.get("suspended", False) is not False:
        return None, None
    price = american_odds(point.get("odds_dec"))
    line = None if market == "money_line" else _historical_line(point)
    return line, price


def _paired(market: str, state: dict) -> tuple:
    if set(state) != {"left", "right"}:
        return NULL_QUOTE
    left, lp = state["left"]
    right, rp = state["right"]
    if lp is None or rp is None:
        return NULL_QUOTE
    if market != "money_line":
        if left is None or right is None:
            return NULL_QUOTE
        if market == "totals" and (left != right or not 150 <= left <= 300):
            return NULL_QUOTE
        if market == "point_spread" and left != -right:
            return NULL_QUOTE
    return left, lp, right, rp


def _implied(point: dict) -> float | None:
    try:
        decimal = float(point.get("odds_dec"))
    except (ValueError, TypeError):
        return None
    return 1 / decimal if math.isfinite(decimal) and decimal > 1 else None


def _numbers(point: dict, market: str) -> tuple[float | None, float] | None:
    """(line, implied probability), or None unless complete for ``market``."""
    line = None if market == "money_line" else _historical_line(point)
    probability = _implied(point)
    if probability is None or (market != "money_line" and line is None):
        return None
    return line, probability


def _fits(numbers: tuple, reference: tuple) -> bool:
    """Whether ``numbers`` are within every jump bound of ``reference``."""
    return all(
        value is None or abs(value - previous) <= limit
        for value, previous, limit in zip(
            numbers, reference, (MAX_LINE_JUMP, MAX_PROB_JUMP), strict=True
        )
    )


class _References:
    """Complete points of one identity, for nearest-in-time lookups."""

    def __init__(self, points: list[tuple]):
        points.sort(key=lambda p: p[0])
        self.moments = [moment for moment, _ in points]
        self.numbers = [numbers for _, numbers in points]

    def nearest(self, moment) -> tuple | None:
        if not self.moments:
            return None
        i = bisect_left(self.moments, moment)
        j = min(
            (k for k in (i - 1, i) if 0 <= k < len(self.moments)),
            key=lambda k: abs(self.moments[k] - moment),
        )
        return self.numbers[j]


def _resolve_selections(labelled: dict) -> dict[str, tuple[str, str] | None]:
    """sid -> identity from each selection's (moment, point, identity) points."""
    usual = {
        sid: Counter(identity for _, _, identity in points).most_common(1)[0]
        for sid, points in labelled.items()
        if sid
    }
    own, by_identity = defaultdict(list), defaultdict(list)
    for sid, (identity, _) in usual.items():
        for moment, point, label in labelled[sid]:
            if label == identity and (numbers := _numbers(point, identity[0])):
                own[sid].append((moment, numbers))
                by_identity[identity].append((moment, numbers))
    own = {sid: _References(points) for sid, points in own.items()}
    by_identity = {i: _References(points) for i, points in by_identity.items()}

    resolved = {}
    for sid, (identity, count) in usual.items():
        points = labelled[sid]
        if count == len(points):
            resolved[sid] = identity
            continue
        resolved[sid] = None
        if count / len(points) < MIN_DOMINANT_SHARE or sid not in own:
            continue
        market, side = identity
        opposite = by_identity.get((market, "right" if side == "left" else "left"))
        if opposite is None:
            continue
        for moment, point, label in points:
            if label == identity:
                continue
            numbers = _numbers(point, market)
            if label[0] != market or numbers is None:
                break
            if not _fits(numbers, own[sid].nearest(moment)) or _fits(
                numbers, opposite.nearest(moment)
            ):
                break
        else:
            resolved[sid] = identity
    return resolved


@dataclass
class ParsedHistory:
    ticks: list[LineTick]
    dropped: dict[str, int]
    repaired: dict[str, int] = field(default_factory=dict)


def history_ticks(
    points: list[dict], *, home: str, away: str, tipoff_utc
) -> ParsedHistory:
    tip = utc_timestamp(tipoff_utc)
    dropped = Counter()
    repaired = Counter()
    rows = []
    labelled = defaultdict(list)
    for point in points:
        if str(point.get("group_id")) != MAIN_GROUP:
            dropped["non_main_group"] += 1
            continue
        mid = str(point.get("market_id") or "")
        if mid and mid not in MARKETS:
            dropped["non_main_market"] += 1
            continue
        try:
            moment = utc_timestamp(point["ts"])
        except (KeyError, ValueError, TypeError):
            raise ValueError(
                "Invalid history timestamp; refusing a partial timeline"
            ) from None
        sid = str(point.get("sel_id") or "")
        identity = _identity(point, home, away)
        if identity:
            labelled[sid].append((moment, point, identity))
        rows.append((moment, sid, point))

    resolved = _resolve_selections(labelled)
    for sid, identity in resolved.items():
        if identity is not None:
            repaired["relabelled_selection"] += sum(
                i != identity for _, _, i in labelled[sid]
            )

    options_by_sid = {
        sid: {identity for _, _, identity in points} for sid, points in labelled.items()
    }
    identities, blocked = {}, set()
    for _, sid, point in rows:
        options = options_by_sid.get(sid, set())
        if resolved.get(sid) is not None:
            identities[sid] = resolved[sid]
        else:
            blocked.update(market for market, _ in options)
            if point.get("market_id"):
                blocked.add(MARKETS[str(point["market_id"])])
            elif not options:
                blocked.update(MARKETS.values())
            dropped["ambiguous_selection"] += 1

    for market in blocked:
        dropped[f"ambiguous_market:{market}"] += 1

    state = defaultdict(dict)
    minutes = defaultdict(dict)
    previous = {}
    for moment, batch in groupby(sorted(rows, key=lambda r: r[0]), key=lambda r: r[0]):
        changes = defaultdict(set)
        for _, sid, point in batch:
            if sid not in identities:
                continue
            market, side = identities[sid]
            if market not in blocked:
                changes[(market, side)].add(_quote(point, market))
        touched = set()
        for (market, side), quotes in changes.items():
            # Simultaneous conflicting prices have no reliable ordering.
            state[market][side] = (
                next(iter(quotes)) if len(quotes) == 1 else (None, None)
            )
            if len(quotes) > 1:
                dropped["conflicting_same_timestamp"] += 1
            touched.add(market)
        for market in touched:
            pair = _paired(market, state[market])
            if pair != previous.get(market):
                minutes[market][moment.floor("min")] = (pair, moment)
                previous[market] = pair

    ticks = []
    for market, by_minute in sorted(minutes.items()):
        opened = False
        for moment, (quote, source_time) in sorted(by_minute.items()):
            valid = quote != NULL_QUOTE
            offset = round((moment - tip).total_seconds() / 60)
            # NBA clocks can carry seconds. Flooring a post-tip point into
            # that minute must never relabel it as pregame.
            offset = min(-1, offset) if source_time < tip else max(0, offset)
            ticks.append(
                LineTick(
                    market=market,
                    book_slug="bet365",
                    book_name="bet365",
                    line_ts=moment.to_pydatetime(),
                    minutes_to_tip=offset,
                    is_opener=valid and not opened,
                    left_line=quote[0],
                    left_price=quote[1],
                    right_line=quote[2],
                    right_price=quote[3],
                )
            )
            opened |= valid
    return ParsedHistory(ticks, dict(dropped), {k: v for k, v in repaired.items() if v})
