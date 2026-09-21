"""Reconstruct five-on-five segments from completed-game rotation and PBP.

All times are tenths of a second elapsed from tip. A malformed game is rejected
instead of silently entering the adjusted-ratings training data.
"""

from __future__ import annotations

import bisect
import json
import re
from collections import defaultdict

import pandas as pd


class StintValidationError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


_CLOCK = re.compile(r"^PT(?:(\d+)M)?(\d+(?:\.\d+)?)S$")
_REBOUND_COUNTS = re.compile(r"\bOff:(\d+)\s+Def:(\d+)\b", re.IGNORECASE)
_STATS = ("fga", "fg3a", "fta", "oreb", "dreb", "tov")


def elapsed_ds(period: int, clock: str) -> int:
    """NBA ISO period-remaining clock to elapsed deciseconds."""
    match = _CLOCK.fullmatch(str(clock))
    if not match or period < 1:
        raise StintValidationError("bad_pbp_clock")
    remaining = float(match[1] or 0) * 60 + float(match[2])
    length = 720 if period <= 4 else 300
    if not 0 <= remaining <= length:
        raise StintValidationError("bad_pbp_clock")
    start = (period - 1) * 7200 if period <= 4 else 28800 + (period - 5) * 3000
    return start + round((length - remaining) * 10)


def _period_start(period: int) -> int:
    return (period - 1) * 7200 if period <= 4 else 28800 + (period - 5) * 3000


def _rotation_intervals(frame: pd.DataFrame) -> tuple[str, list[tuple[int, int, str]]]:
    required = {"TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"}
    if not required.issubset(frame) or frame.empty:
        raise StintValidationError("missing_rotation")
    ids = frame["TEAM_ID"].dropna().astype(str).unique()
    if len(ids) != 1:
        raise StintValidationError("rotation_team_mismatch")
    intervals = []
    for row in frame.itertuples(index=False):
        start = int(row.IN_TIME_REAL)
        end = int(row.OUT_TIME_REAL)
        if start < 0 or end < start:
            raise StintValidationError("bad_rotation_interval")
        # The NBA feed sometimes includes a substitution with zero playing
        # time. It contributes no lineup overlap or box-score minutes.
        if end == start:
            continue
        intervals.append((start, end, str(row.PERSON_ID)))
    per_player = defaultdict(list)
    for start, end, player in intervals:
        per_player[player].append((start, end))
    for spans in per_player.values():
        spans.sort()
        if any(b[0] < a[1] for a, b in zip(spans, spans[1:], strict=False)):
            raise StintValidationError("overlapping_player_intervals")
    return ids[0], intervals


def _lineup_at(intervals: list[tuple[int, int, str]], midpoint: float) -> tuple[str, ...]:
    lineup = tuple(sorted({pid for start, end, pid in intervals if start <= midpoint < end}))
    if len(lineup) != 5:
        raise StintValidationError("not_five_players")
    return lineup


def _score(value: object, previous: int) -> int:
    if value is None or pd.isna(value) or value == "":
        return previous
    return int(value)


def _is_true(value: object) -> bool:
    return str(value).lower() in {"1", "true"}


def _id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def build_game_stints(
    rotation_home: pd.DataFrame,
    rotation_away: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Return one validated segment per lineup change or period boundary."""
    home_id, home = _rotation_intervals(rotation_home)
    away_id, away = _rotation_intervals(rotation_away)
    if home_id == away_id:
        raise StintValidationError("rotation_team_mismatch")
    required = {"period", "clock", "actionNumber", "actionType", "teamId"}
    if pbp.empty or not required.issubset(pbp):
        raise StintValidationError("missing_pbp")
    max_period = int(pd.to_numeric(pbp["period"], errors="coerce").max())
    if max_period < 4:
        raise StintValidationError("incomplete_game")
    game_end = _period_start(max_period) + (7200 if max_period <= 4 else 3000)
    boundaries = sorted(
        {0, game_end}
        | {_period_start(p) for p in range(2, max_period + 1)}
        | {v for start, end, _ in home + away for v in (start, end)}
    )
    if boundaries[0] != 0 or boundaries[-1] != game_end:
        raise StintValidationError("rotation_game_length")
    rows = []
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        if end <= start:
            raise StintValidationError("bad_segment_length")
        midpoint = (start + end) / 2
        period = max(p for p in range(1, max_period + 1) if _period_start(p) <= start)
        row = {
            "seg_idx": i,
            "period": period,
            "start_ds": start,
            "end_ds": end,
            "seconds": (end - start) / 10,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_lineup": _lineup_at(home, midpoint),
            "away_lineup": _lineup_at(away, midpoint),
            "home_pts": 0,
            "away_pts": 0,
        }
        row.update({f"{side}_{stat}": 0 for side in ("home", "away") for stat in _STATS})
        rows.append(row)

    events = pbp.copy()
    events["_period"] = pd.to_numeric(events["period"], errors="raise").astype(int)
    events["_time"] = [
        elapsed_ds(period, clock)
        for period, clock in zip(events["_period"], events["clock"], strict=True)
    ]
    events["_order"] = pd.to_numeric(events["actionNumber"], errors="raise")
    # Corrected actions may be appended with new actionNumbers while their
    # clock still belongs earlier in the game. Sort by game time first and use
    # actionNumber only to resolve events sharing that clock.
    events = events.sort_values(["_time", "_order"], kind="stable")
    subs_at = defaultdict(list)
    subs_by_team = defaultdict(list)
    for event in events.to_dict("records"):
        if str(event.get("actionType", "")).lower() == "substitution":
            subs_at[event["_time"]].append(event["_order"])
            subs_by_team[(event["_time"], _id(event.get("teamId")))].append(
                event["_order"]
            )

    previous_home = previous_away = 0
    rebound_totals: dict[str, tuple[int, int]] = {}
    for event in events.to_dict("records"):
        t = event["_time"]
        order = event["_order"]
        period = event["_period"]
        action = str(event.get("actionType", "")).lower()
        if action == "substitution":
            continue
        idx = bisect.bisect_right(boundaries, t) - 1
        if idx >= len(rows):
            idx = len(rows) - 1
        if idx < 0:
            raise StintValidationError("pbp_outside_game")
        if t in subs_at and t in boundaries and t > 0:
            orders = subs_at[t]
            if order < min(orders):
                idx -= 1
            elif order <= max(orders):
                # Multiple substitutions can straddle a free throw at the
                # same clock. Rotation gives the before/after lineup, not an
                # intermediate zero-time lineup. Use the shooter's presence
                # when it distinguishes them; otherwise use their team's
                # first substitution in event order.
                actor = _id(event.get("personId"))
                team_id = _id(event.get("teamId"))
                side = "home" if team_id == home_id else "away"
                before = rows[idx - 1][f"{side}_lineup"]
                after = rows[idx][f"{side}_lineup"]
                if actor in before and actor not in after:
                    idx -= 1
                elif actor not in after:
                    idx -= 1
                elif actor in before and actor in after:
                    own_subs = subs_by_team.get((t, team_id), [])
                    if not own_subs or order < min(own_subs):
                        idx -= 1
        elif t in boundaries and t > 0 and idx > 0:
            # At a period end, events belong to the period that just ended;
            # at a period start they belong to the new period.
            if _period_start(period) != t:
                idx -= 1
        if idx < 0:
            raise StintValidationError("pbp_outside_game")
        row = rows[idx]
        home_score = _score(event.get("scoreHome"), previous_home)
        away_score = _score(event.get("scoreAway"), previous_away)
        dh, da = home_score - previous_home, away_score - previous_away
        # NBA V3 occasionally corrects an earlier scoring play (including a
        # *decrease*). Preserve the signed correction so game totals reconcile.
        row["home_pts"] += dh
        row["away_pts"] += da
        previous_home, previous_away = home_score, away_score
        team_id = _id(event.get("teamId"))
        side = "home" if team_id == home_id else "away" if team_id == away_id else None
        if side is None:
            continue
        subtype = str(event.get("subType", "")).lower()
        if action in {"made shot", "missed shot"} and _is_true(
            event.get("isFieldGoal")
        ):
            row[f"{side}_fga"] += 1
            if str(event.get("shotValue", "")) == "3":
                row[f"{side}_fg3a"] += 1
        elif action == "free throw":
            row[f"{side}_fta"] += 1
        elif action == "rebound":
            if "offensive" in subtype:
                row[f"{side}_oreb"] += 1
            elif "defensive" in subtype:
                row[f"{side}_dreb"] += 1
            else:
                # V3 commonly sets subType='Unknown'; individual rebound
                # descriptions carry each player's cumulative Off/Def counts.
                counts = _REBOUND_COUNTS.search(str(event.get("description", "")))
                if counts:
                    player = _id(event.get("personId"))
                    off, defensive = int(counts[1]), int(counts[2])
                    old_off, old_def = rebound_totals.get(player, (0, 0))
                    row[f"{side}_oreb"] += max(0, off - old_off)
                    row[f"{side}_dreb"] += max(0, defensive - old_def)
                    # A corrected action can restate a *lower* cumulative count
                    # than an earlier row. Keep the running maximum so the next
                    # genuine rebound is not counted twice.
                    rebound_totals[player] = (
                        max(off, old_off),
                        max(defensive, old_def),
                    )
        elif action == "turnover":
            row[f"{side}_tov"] += 1
    out = pd.DataFrame(rows)
    for side in ("home", "away"):
        out[f"{side}_poss"] = (
            out[f"{side}_fga"] + 0.44 * out[f"{side}_fta"]
            - out[f"{side}_oreb"] + out[f"{side}_tov"]
        )
    if sum(out.end_ds - out.start_ds) != game_end:
        raise StintValidationError("segment_game_length")
    return out


def validate_game_stints(
    stints: pd.DataFrame,
    rotation_home: pd.DataFrame,
    rotation_away: pd.DataFrame,
    *,
    home_points: int,
    away_points: int,
    box_minutes: pd.DataFrame,
) -> None:
    """Reject a game unless points and each player's minutes reconcile."""
    if (int(stints.home_pts.sum()), int(stints.away_pts.sum())) != (
        home_points, away_points
    ):
        raise StintValidationError("points_mismatch")
    if not {"TEAM_ID", "PLAYER_ID", "MIN"}.issubset(box_minutes):
        raise StintValidationError("missing_box_minutes")
    for rotation in (rotation_home, rotation_away):
        team_id = str(rotation.TEAM_ID.iloc[0])
        for player_id, player in rotation.groupby("PERSON_ID"):
            seconds = (player.OUT_TIME_REAL - player.IN_TIME_REAL).sum() / 10
            if seconds == 0:
                continue
            box = box_minutes.loc[
                box_minutes.TEAM_ID.astype(str).eq(team_id)
                & box_minutes.PLAYER_ID.astype(str).eq(str(player_id)), "MIN"
            ]
            if box.empty or abs(seconds - _minutes(box.iloc[0]) * 60) > 60:
                raise StintValidationError("minutes_mismatch")


def _minutes(value: object) -> float:
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":", 1)
        return float(minutes) + float(seconds) / 60
    return float(value)


def decode_raw_game(
    rotation_json: bytes, pbp_json: bytes
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decode the archived original responses without calling the NBA API."""
    rotation = json.loads(rotation_json)
    sets = {
        part["name"]: pd.DataFrame(part["rowSet"], columns=part["headers"])
        for part in rotation["resultSets"]
    }
    actions = json.loads(pbp_json)["game"]["actions"]
    return sets["HomeTeam"], sets["AwayTeam"], pd.DataFrame(actions)
