"""Rebuild GameRotation from PlayByPlayV2 for games the NBA will not serve.

Before 2021-22, ``GameRotation`` is built on demand behind a ~30 s cap and
caches its failures, so a fifth of old games at best come back per retry pass
(audited 2026-09-25). ``PlayByPlayV2``, republished season by season in the
``shufinskiy/nba_data`` archive (``nbastats_YYYY``, through 2024-25), carries
what a rotation needs: its substitution rows name both players by ID, the
outgoing one in ``PLAYER1_ID`` and the incoming one in ``PLAYER2_ID``. (V3
names the incoming player only by surname in the description.)

The one thing play-by-play does not state is who starts each period. A player
starts it if he is subbed out, or does anything, before being subbed in. A
starter who does nothing all period is invisible; the box score's minutes fill
that slot with the teammate whose recorded time falls furthest short.

Checked on 446 games of 2020-21 that also have an API rotation: median 99.97%
of seconds with identical lineups. Every game that passed ``check_rotation``
matched, apart from two where the API's own rotation is corrupt. The output is
GameRotation-shaped JSON marked with ``SOURCE``, so the stint builder reads it
unchanged and validates it again against points and box minutes.
"""

from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from nba_ou.fetch_data.nba_lineups.manifest import REBUILT_SOURCE as SOURCE

HEADERS = (
    "GAME_ID", "TEAM_ID", "TEAM_CITY", "TEAM_NAME", "PERSON_ID", "PLAYER_FIRST",
    "PLAYER_LAST", "IN_TIME_REAL", "OUT_TIME_REAL", "PLAYER_PTS", "PT_DIFF", "USG_PCT",
)

# PlayByPlayV2 EVENTMSGTYPE codes.
FOUL, SUBSTITUTION, TIMEOUT, EJECTION, INSTANT_REPLAY = 6, 8, 9, 11, 18
#: Foul action types that can be called on a player sitting on the bench
#: (technicals, delay and taunting). Seeing one proves nothing about the floor.
BENCH_FOUL_ACTIONS = frozenset({11, 12, 13, 16, 18, 19, 25, 30})
NOT_ON_COURT_EVENTS = frozenset({TIMEOUT, EJECTION, INSTANT_REPLAY})
#: PERSONnTYPE for home and visiting players; 2/3 are the teams themselves.
PLAYER_PERSON_TYPES = frozenset({4, 5})

#: Same tolerance the stint builder applies to box-score minutes.
MINUTES_TOLERANCE_SECONDS = 60


class RotationRebuildError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def period_bounds(period: int) -> tuple[int, int]:
    """Deciseconds from tip-off, the GameRotation scale."""
    if period <= 4:
        start = (period - 1) * 7200
        return start, start + 7200
    start = 28800 + (period - 5) * 3000
    return start, start + 3000


def _elapsed(period: int, clock: str) -> int:
    minutes, seconds = str(clock).split(":")
    start, end = period_bounds(period)
    remaining = round((int(minutes) * 60 + float(seconds)) * 10)
    if not 0 <= remaining <= end - start:
        raise RotationRebuildError("bad_pbp_clock")
    return end - remaining


def _box_seconds(value: object) -> float:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes) * 60 + float(seconds)
    return float(text) * 60


def _int(value: object) -> int:
    number = pd.to_numeric(value, errors="coerce")
    return 0 if pd.isna(number) else int(number)


def _prepare(pbp: pd.DataFrame) -> pd.DataFrame:
    events = pbp.copy()
    for column in ("EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD"):
        events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0).astype(int)
    for k in (1, 2, 3):
        for column in (f"PLAYER{k}_ID", f"PLAYER{k}_TEAM_ID", f"PERSON{k}TYPE"):
            events[column] = pd.to_numeric(events[column], errors="coerce").fillna(0).astype(int)
    events["T"] = [
        _elapsed(period, clock) for period, clock in zip(events.PERIOD, events.PCTIMESTRING, strict=True)
    ]
    # EVENTNUM is occasionally out of chronological order; within one clock
    # value it is the only order there is, so it breaks the ties.
    return events.sort_values(["PERIOD", "T", "EVENTNUM"], kind="stable")


def _players(row, team_id: int) -> list[int]:
    found = []
    for k in (1, 2, 3):
        if (
            getattr(row, f"PLAYER{k}_TEAM_ID") == team_id
            and getattr(row, f"PERSON{k}TYPE") in PLAYER_PERSON_TYPES
            and getattr(row, f"PLAYER{k}_ID") > 0
        ):
            found.append(getattr(row, f"PLAYER{k}_ID"))
    return found


def _proves_on_court(row) -> bool:
    if row.EVENTMSGTYPE in NOT_ON_COURT_EVENTS:
        return False
    return not (row.EVENTMSGTYPE == FOUL and row.EVENTMSGACTIONTYPE in BENCH_FOUL_ACTIONS)


def _period_starters(events: pd.DataFrame, team_id: int) -> tuple[list[int], set[int]]:
    """Players seen on the floor before any sub-in, and everyone the period touches."""
    starters: list[int] = []
    subbed_in: set[int] = set()
    touched: set[int] = set()
    for row in events.itertuples(index=False):
        if row.EVENTMSGTYPE == SUBSTITUTION:
            if row.PLAYER1_TEAM_ID != team_id:
                continue
            out, into = row.PLAYER1_ID, row.PLAYER2_ID
            if out <= 0 or into <= 0:
                raise RotationRebuildError("bad_substitution")
            if out not in subbed_in and out not in starters:
                starters.append(out)
            subbed_in.add(into)
            touched.update((out, into))
            continue
        players = _players(row, team_id)
        touched.update(players)
        if not _proves_on_court(row):
            continue
        for player in players:
            if player not in subbed_in and player not in starters:
                starters.append(player)
    return starters, touched


def _intervals(
    events: pd.DataFrame, team_id: int, period: int, starters: list[int]
) -> list[tuple[int, int, int]]:
    start, end = period_bounds(period)
    on_court = {player: start for player in starters}
    spans = []
    for row in events.itertuples(index=False):
        if row.EVENTMSGTYPE != SUBSTITUTION or row.PLAYER1_TEAM_ID != team_id:
            continue
        out, into = row.PLAYER1_ID, row.PLAYER2_ID
        # A sub of someone not on the floor is a feed error; check_rotation
        # rejects the game if it leaves the wrong number of players out there.
        if out in on_court:
            spans.append((out, on_court.pop(out), row.T))
        on_court.setdefault(into, row.T)
    spans.extend((player, entered, end) for player, entered in on_court.items())
    return spans


def _merge(spans: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Join a player's back-to-back intervals, as GameRotation reports them."""
    merged: dict[int, list[list[int]]] = defaultdict(list)
    for player, start, end in sorted(spans, key=lambda span: (span[0], span[1])):
        if end <= start:
            continue
        runs = merged[player]
        if runs and runs[-1][1] == start:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    return [(player, start, end) for player, runs in merged.items() for start, end in runs]


def rebuild_rotation(
    pbp: pd.DataFrame, box: pd.DataFrame, team_ids: tuple[int, int]
) -> dict[int, list[tuple[int, int, int]]]:
    """``{team_id: [(player_id, in, out), ...]}`` for one game.

    ``pbp`` is the game's PlayByPlayV2 rows; ``box`` its box score with
    ``TEAM_ID``, ``PLAYER_ID`` and ``MIN``. Raises ``RotationRebuildError`` if
    the result does not pass ``check_rotation``.
    """
    events = _prepare(pbp)
    if events.empty:
        raise RotationRebuildError("no_pbp")
    box_seconds = {
        (int(team), int(player)): _box_seconds(minutes)
        for team, player, minutes in zip(box.TEAM_ID, box.PLAYER_ID, box.MIN, strict=True)
    }
    periods = sorted(events.PERIOD.unique())
    if periods != list(range(1, len(periods) + 1)) or len(periods) < 4:
        raise RotationRebuildError("missing_period")

    lineups: dict[tuple[int, int], tuple[list[int], set[int]]] = {}
    for period in periods:
        period_events = events.loc[events.PERIOD.eq(period)]
        for team_id in team_ids:
            starters, touched = _period_starters(period_events, team_id)
            if len(starters) > 5:
                raise RotationRebuildError("too_many_starters")
            lineups[(period, team_id)] = (starters, touched)

    def spans_for(team_id: int) -> list[tuple[int, int, int]]:
        spans = []
        for period in periods:
            starters, _ = lineups[(period, team_id)]
            spans += _intervals(
                events.loc[events.PERIOD.eq(period)], team_id, period, starters
            )
        return spans

    # Fill each starter the play-by-play never shows, largest deficit first,
    # re-measuring after each pick so one player is not credited twice.
    for (_period, team_id), (starters, touched) in lineups.items():
        while len(starters) < 5:
            played = defaultdict(float)
            for player, start, end in spans_for(team_id):
                played[player] += (end - start) / 10
            candidates = {
                player: seconds - played[player]
                for (team, player), seconds in box_seconds.items()
                if team == team_id and seconds > 0 and player not in touched
            }
            if not candidates:
                raise RotationRebuildError("unresolved_starter")
            pick = max(candidates, key=candidates.get)
            starters.append(pick)
            touched.add(pick)

    rotation = {team_id: _merge(spans_for(team_id)) for team_id in team_ids}
    check_rotation(rotation, box_seconds, periods)
    return rotation


def check_rotation(
    rotation: dict[int, list[tuple[int, int, int]]],
    box_seconds: dict[tuple[int, int], float],
    periods: list[int],
) -> None:
    """Five a side for the whole game, and every player's minutes as boxed."""
    game_end = period_bounds(max(periods))[1]
    for team_id, spans in rotation.items():
        changes: dict[int, int] = defaultdict(int)
        for _, start, end in spans:
            changes[start] += 1
            changes[end] -= 1
        on_court, previous = 0, 0
        for moment in sorted(set(changes) | {0, game_end}):
            if previous < moment and on_court != 5:
                raise RotationRebuildError("not_five_on_court")
            on_court += changes.get(moment, 0)
            previous = moment
        played: dict[int, float] = defaultdict(float)
        for player, start, end in spans:
            played[player] += (end - start) / 10
        boxed = {
            player: seconds
            for (team, player), seconds in box_seconds.items()
            if team == team_id and seconds > 0
        }
        for player in set(played) | set(boxed):
            if abs(played.get(player, 0.0) - boxed.get(player, 0.0)) > MINUTES_TOLERANCE_SECONDS:
                raise RotationRebuildError("minutes_mismatch")


def rotation_payload(
    game_id: str,
    rotation: dict[int, list[tuple[int, int, int]]],
    *,
    home_team_id: int,
    pbp: pd.DataFrame,
) -> bytes:
    """GameRotation JSON the stint builder decodes like an API response.

    Only TEAM_ID, PERSON_ID and the two times are rebuilt; names and team
    labels come from the play-by-play for readability, and the per-stint
    points, plus-minus and usage GameRotation reports are left null.
    """
    names: dict[int, str] = {}
    teams: dict[int, tuple[str, str]] = {}
    for k in (1, 2, 3):
        ids = pd.to_numeric(pbp[f"PLAYER{k}_ID"], errors="coerce")
        team_ids = pd.to_numeric(pbp[f"PLAYER{k}_TEAM_ID"], errors="coerce")
        for pid, name, tid, city, nick in zip(
            ids, pbp[f"PLAYER{k}_NAME"], team_ids,
            pbp[f"PLAYER{k}_TEAM_CITY"], pbp[f"PLAYER{k}_TEAM_NICKNAME"],
            strict=True,
        ):
            if pd.notna(pid) and isinstance(name, str):
                names.setdefault(int(pid), name.strip())
            if pd.notna(tid) and isinstance(city, str):
                teams.setdefault(int(tid), (city.strip(), str(nick).strip()))

    def rows(team_id: int) -> list[list]:
        city, nickname = teams.get(team_id, ("", ""))
        out = []
        for player, start, end in sorted(rotation[team_id], key=lambda s: (s[1], s[0])):
            first, _, last = names.get(player, "").partition(" ")
            out.append([
                game_id, int(team_id), city, nickname, int(player), first, last,
                int(start), int(end), None, None, None,
            ])
        return out

    away_team_id = next(team for team in rotation if team != home_team_id)
    document = {
        "resource": "gamerotation",
        "parameters": {"GameID": game_id, "LeagueID": "00"},
        "source": SOURCE,
        "resultSets": [
            {"name": "AwayTeam", "headers": list(HEADERS), "rowSet": rows(away_team_id)},
            {"name": "HomeTeam", "headers": list(HEADERS), "rowSet": rows(home_team_id)},
        ],
    }
    return json.dumps(document).encode("utf-8")


def rotation_from_payload(raw: bytes) -> dict[int, list[tuple[int, int, int]]]:
    """Read a GameRotation response, fetched or rebuilt, back into spans."""
    rotation: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for part in json.loads(raw)["resultSets"]:
        for row in part["rowSet"]:
            record = dict(zip(part["headers"], row, strict=True))
            rotation[int(record["TEAM_ID"])].append(
                (int(record["PERSON_ID"]), int(record["IN_TIME_REAL"]),
                 int(record["OUT_TIME_REAL"]))
            )
    return dict(rotation)


def lineup_agreement(
    first: dict[int, list[tuple[int, int, int]]],
    second: dict[int, list[tuple[int, int, int]]],
) -> float:
    """Share of team-seconds on which both rotations put the same five out."""
    game_end = max(end for spans in (*first.values(), *second.values()) for *_, end in spans)
    moments = range(5, game_end, 10)  # the middle of every second
    agree = total = 0
    for team_id in set(first) | set(second):
        for moment in moments:
            on_first = {p for p, start, end in first.get(team_id, []) if start <= moment < end}
            on_second = {p for p, start, end in second.get(team_id, []) if start <= moment < end}
            agree += on_first == on_second
            total += 1
    return agree / total if total else 0.0
