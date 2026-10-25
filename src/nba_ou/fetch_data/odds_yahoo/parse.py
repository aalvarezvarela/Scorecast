"""Parse Yahoo Sports NBA pages from their server-rendered payload.

Yahoo's pages are Next.js App Router pages. Every one carries its data in the
React Server Components payload, the strings passed to
``self.__next_f.push([1, "..."])``, so no browser is needed:

* the schedule page (``/nba/schedule/?season=&date=``) links each matchup page
  as ``/nba/<away>-<home>-<YYYYMMDD><home team id>/``; the ten digits are
  Yahoo's game id and the date in it is the US (Eastern) game date. The page
  itself is laid out in the server's guess of the reader's time zone (from
  the IP), so one schedule date does not hold one US date.
* the matchup page (``?section=odds``) holds the game object: ``gameId``,
  ``awayTeam``/``homeTeam``, ``season`` and ``bets``, one entry per market
  (``SPREAD``, ``OVER_UNDER``, ``MONEY_LINE``), each with two options. An
  option names its side (``teamIds`` or the ``over``/``under`` detail) and
  carries the public splits: ``wagerPercentage`` is the "% of bets" Yahoo
  shows, ``stakePercentage`` the "% of money".

The two options come in Yahoo's internal id order, which since 2026-27 is not
always away/over first. Each value here is read from the option's own label,
never from its position.

A page whose server render failed still answers 200: its payload carries an
error row (``<id>:E{"digest": ...}``) and the game object, if any, lacks
``bets``. The browser recovers from that by rendering on the client; a plain
GET has to retry (``render_failed``).
"""

from __future__ import annotations

import json
import re
from datetime import date

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)')
_RENDER_ERROR_RE = re.compile(r'(?:^|\n)[0-9a-f]+:E\{"digest"')
_MATCHUP_PATH_RE = re.compile(r'href="(/nba/[a-z0-9-]+-(\d{10})/)"')
_DECODER = json.JSONDecoder()

#: How far back, in opening braces, ``find_game`` looks for the object that
#: holds a ``gameId`` key that is not its first.
_MAX_BRACES_BACK = 64

#: The ``odds_yahoo`` columns but ``game_id``, which the caller maps from the
#: games table (``create_odds_yahoo_db._yahoo_columns`` is the table).
YAHOO_COLUMNS = [
    "game_date",
    "season_year",
    "team_home",
    "team_away",
    "team_home_abbr",
    "team_away_abbr",
    "spread_home",
    "spread_away",
    "moneyline_home",
    "moneyline_away",
    "total_line",
    "total_pct_bets_over",
    "total_pct_bets_under",
    "total_pct_money_over",
    "total_pct_money_under",
    "spread_pct_bets_away",
    "spread_pct_bets_home",
    "spread_pct_money_away",
    "spread_pct_money_home",
    "moneyline_pct_bets_away",
    "moneyline_pct_bets_home",
    "moneyline_pct_money_away",
    "moneyline_pct_money_home",
]

_PCT_FIELDS = {"pct_bets": "wagerPercentage", "pct_money": "stakePercentage"}

#: The line columns each market must fill for the game to count as priced.
_LINE_COLUMNS = {
    "SPREAD": ("spread_home", "spread_away"),
    "OVER_UNDER": ("total_line",),
    "MONEY_LINE": ("moneyline_home", "moneyline_away"),
}

_NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


class YahooParseError(ValueError):
    """A game's odds do not have the shape this parser reads."""


def flight_payload(html: str) -> str:
    """The page's React Server Components payload, its pushes joined."""
    return "".join(json.loads(f'"{chunk}"') for chunk in _PUSH_RE.findall(html))


def render_failed(payload: str) -> bool:
    """True when the server render of the page failed (an error row)."""
    return not payload or _RENDER_ERROR_RE.search(payload) is not None


def season_year_for_date(d: date) -> int:
    # Yahoo season start year. 2020 was delayed, so threshold is November there.
    month_threshold = 11 if d.year == 2020 else 10
    return d.year if d.month >= month_threshold else (d.year - 1)


def game_date_from_id(yahoo_game_id: str) -> date:
    """The US game date in a ten-digit Yahoo game id (``YYYYMMDD`` + team)."""
    if not re.fullmatch(r"\d{10}", yahoo_game_id):
        raise ValueError(f"Not a Yahoo game id: {yahoo_game_id!r}")
    return date(
        int(yahoo_game_id[:4]), int(yahoo_game_id[4:6]), int(yahoo_game_id[6:8])
    )


def matchup_paths(html: str) -> dict[str, str]:
    """Yahoo game id -> matchup path, for every matchup the page links."""
    return {game_id: path for path, game_id in _MATCHUP_PATH_RE.findall(html)}


def _decode_object_at(payload: str, start: int) -> dict | None:
    try:
        obj, _ = _DECODER.raw_decode(payload, start)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def find_game(payload: str, yahoo_game_id: str) -> dict | None:
    """The game object of ``yahoo_game_id`` that carries its ``bets``.

    The payload names a game several times: the season series, references
    such as ``"bets": "$ab:..."`` and, for a game of the scoreboard strip
    (today's games), a short copy whose ``bets`` has the lines but neither
    the splits nor ``season``. The one wanted is the fullest object whose
    ``bets`` is a list. None if there is none, as on a page whose render
    failed.
    """
    key = f'"gameId":"nba.g.{yahoo_game_id}"'
    found: list[dict] = []
    for match in re.finditer(re.escape(key), payload):
        # ``gameId`` is the first key of the game object; walk back over
        # opening braces for the rare object that lists it later.
        start = match.start()
        for _ in range(_MAX_BRACES_BACK):
            start = payload.rfind("{", 0, start)
            if start < 0:
                break
            obj = _decode_object_at(payload, start)
            if obj is not None and obj.get("gameId") == f"nba.g.{yahoo_game_id}":
                if isinstance(obj.get("bets"), list):
                    found.append(obj)
                break
    return max(found, key=len, default=None)


def _pct(option: dict, field: str) -> float | None:
    value = option.get(field)
    return None if value in (None, "") else float(value)


def _team_side(option: dict, away_id: str, home_id: str) -> str:
    team_ids = option.get("teamIds") or [t.get("teamId") for t in option["teams"]]
    if team_ids == [away_id]:
        return "away"
    if team_ids == [home_id]:
        return "home"
    raise YahooParseError(f"Option for teams {team_ids} is neither side")


def _number(value: object) -> float | None:
    if value in (None, ""):
        return None
    # Old labels can trail the team name ("+1NewYorkKnicks").
    match = _NUMBER_RE.match(str(value).replace("−", "-").strip())
    if match is None:
        raise YahooParseError(f"Unreadable number {value!r}")
    return float(match.group())


def _detail(option: dict, *keys: str) -> tuple[str, object] | None:
    for detail in option.get("optionDetails") or []:
        if detail.get("key") in keys:
            return detail["key"], detail.get("value")
    return None


def _total_side(option: dict) -> tuple[str, float | None]:
    detail = _detail(option, "over", "under")
    if detail is not None:
        side, value = detail
    else:
        # Older pages: no details, the side only in the label ("O 228.5").
        label = (option.get("shortName") or "").split()
        if len(label) != 2 or label[0] not in {"O", "U"}:
            raise YahooParseError(f"Unreadable total option {option!r}")
        side = "over" if label[0] == "O" else "under"
        value = label[1]
    return side, _number(value)


def _spread_points(option: dict) -> float | None:
    detail = _detail(option, "points")
    return _number(detail[1] if detail is not None else option.get("shortName"))


def _add_pcts(row: dict, prefix: str, side: str, option: dict) -> None:
    for name, field in _PCT_FIELDS.items():
        row[f"{prefix}_{name}_{side}"] = _pct(option, field)


def _read_team_market(
    market: str, options: list[dict], away_id: str, home_id: str
) -> dict:
    prefix = "spread" if market == "SPREAD" else "moneyline"
    row: dict[str, object] = {}
    for option in options:
        side = _team_side(option, away_id, home_id)
        if f"{prefix}_{side}" in row:
            raise YahooParseError(f"Both {market} options are the {side} side")
        row[f"{prefix}_{side}"] = (
            _spread_points(option) if market == "SPREAD" else option.get("americanOdds")
        )
        _add_pcts(row, prefix, side, option)
    if market == "SPREAD" and None not in (row["spread_home"], row["spread_away"]):
        if row["spread_home"] != -row["spread_away"]:
            raise YahooParseError(
                f"Spreads {row['spread_home']} / {row['spread_away']} do not mirror"
            )
    return row


def _read_total_market(options: list[dict]) -> dict:
    row: dict[str, object] = {}
    lines = set()
    for option in options:
        side, line = _total_side(option)
        if f"total_pct_bets_{side}" in row:
            raise YahooParseError(f"Both total options are the {side} side")
        lines.add(line)
        _add_pcts(row, "total", side, option)
    if len(lines) != 1:
        raise YahooParseError(f"Over and under at different lines {sorted(lines)}")
    row["total_line"] = lines.pop()
    return row


def _pregame_lines(game: dict) -> tuple[float | None, float | None]:
    """|spread| and total of the summary line, e.g. ``"-10.5, O/U 228.5"``."""
    text = (game.get("gameOddsSummary") or {}).get("pregameOddsDisplay") or ""
    spread = re.match(r"\s*([+-]?\d+(?:\.\d+)?)\s*,", text)
    total = re.search(r"O/U\s*(\d+(?:\.\d+)?)", text)
    return (
        abs(float(spread.group(1))) if spread else None,
        float(total.group(1)) if total else None,
    )


def _choose(market: str, readings: list[dict], game: dict) -> dict | None:
    """The market's one reading. A game can list a market twice, the second
    copy a line taken later (often in play, the same splits); the pre-game
    line in the summary picks the copy."""
    if len(readings) <= 1:
        return readings[0] if readings else None
    pregame_spread, pregame_total = _pregame_lines(game)
    if market == "OVER_UNDER":
        matches = [r for r in readings if r["total_line"] == pregame_total]
    elif market == "SPREAD":
        matches = [r for r in readings if abs(r["spread_home"]) == pregame_spread]
    else:
        matches = []
    if len(matches) != 1:
        raise YahooParseError(f"{len(readings)} different {market} markets")
    return matches[0]


def game_odds_row(game: dict) -> dict | None:
    """One row of ``odds_yahoo`` (every column but ``game_id``) from a game.

    None when the game has no complete full-game lines (no odds offered, or
    a market or a price missing); the percentages may be missing.
    """
    away_id, home_id = game["awayTeamId"], game["homeTeamId"]
    readings: dict[str, list[dict]] = {market: [] for market in _LINE_COLUMNS}
    errors: dict[str, YahooParseError] = {}
    for bet in game["bets"]:
        market, options = bet.get("type"), bet.get("options") or []
        if market not in readings or len(options) != 2:
            continue
        try:
            if market == "OVER_UNDER":
                reading = _read_total_market(options)
            else:
                reading = _read_team_market(market, options, away_id, home_id)
        except YahooParseError as exc:
            # Old pages can carry a broken extra copy (teamless "1"/"2"
            # options); it only matters when no copy can be read.
            errors[market] = exc
            continue
        # A copy without its line or price is no reading of the market.
        complete = all(reading[c] is not None for c in _LINE_COLUMNS[market])
        if complete and reading not in readings[market]:
            readings[market].append(reading)

    row: dict[str, object] = {}
    for market, market_readings in readings.items():
        if not market_readings and market in errors:
            raise errors[market]
        chosen = _choose(market, market_readings, game)
        if chosen is None:
            return None
        row.update(chosen)

    game_date = game_date_from_id(game["gameId"].removeprefix("nba.g."))
    row.update(
        {
            "game_date": game_date,
            "season_year": int(game.get("season") or season_year_for_date(game_date)),
            "team_home": game["homeTeam"]["fullName"],
            "team_away": game["awayTeam"]["fullName"],
            "team_home_abbr": game["homeTeam"]["abbreviation"],
            "team_away_abbr": game["awayTeam"]["abbreviation"],
        }
    )
    return {c: row[c] for c in YAHOO_COLUMNS}
