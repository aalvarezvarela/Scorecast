"""Rotations rebuilt from PlayByPlayV2 must be ones the stint builder can trust."""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.lineups.stints import _rotation_intervals, decode_raw_game
from nba_ou.fetch_data.nba_lineups.manifest import REBUILT_SOURCE, Manifest
from nba_ou.fetch_data.nba_lineups.rotation_rebuild import (
    RotationRebuildError,
    lineup_agreement,
    rebuild_rotation,
    rotation_from_payload,
    rotation_payload,
)

HOME, AWAY = 1, 2
MADE_SHOT, FOUL, SUB = 1, 6, 8
GAME = "0022000001"


def _event(num, period, clock, etype, p1=0, team1=0, p2=0, team2=0, action=0) -> dict:
    row = {
        "GAME_ID": GAME, "EVENTNUM": num, "EVENTMSGTYPE": etype,
        "EVENTMSGACTIONTYPE": action, "PERIOD": period, "PCTIMESTRING": clock,
    }
    for k, (player, team) in enumerate(((p1, team1), (p2, team2), (0, 0)), start=1):
        row[f"PLAYER{k}_ID"] = player
        row[f"PLAYER{k}_TEAM_ID"] = team or None
        row[f"PERSON{k}TYPE"] = {HOME: 4, AWAY: 5}.get(team, 0)
        row[f"PLAYER{k}_NAME"] = f"First{player} Last{player}" if player else None
        row[f"PLAYER{k}_TEAM_CITY"] = {HOME: "Home", AWAY: "Away"}.get(team)
        row[f"PLAYER{k}_TEAM_NICKNAME"] = {HOME: "H", AWAY: "A"}.get(team)
    return row


def _game(extra: list[dict] = ()) -> pd.DataFrame:
    """Four quarters. Home: 16 replaces 15 at 6:00 of Q1 and stays.

    Away: the same five all game, but 25 does nothing in Q1, so only the box
    score can say he started it. 26 never plays.
    """
    events, num = [], 0

    def add(*args, **kwargs):
        nonlocal num
        num += 1
        events.append(_event(num, *args, **kwargs))

    for period in (1, 2, 3, 4):
        home_five = [11, 12, 13, 14, 15] if period == 1 else [11, 12, 13, 14, 16]
        away_five = [21, 22, 23, 24] if period == 1 else [21, 22, 23, 24, 25]
        for player in home_five:
            add(period, "11:00", MADE_SHOT, player, HOME)
        for player in away_five:
            add(period, "10:00", MADE_SHOT, player, AWAY)
        if period == 1:
            add(1, "6:00", SUB, 15, HOME, 16, HOME)
            add(1, "5:00", MADE_SHOT, 16, HOME)
    return pd.DataFrame(events + list(extra))


BOX = pd.DataFrame(
    [(HOME, player, "48:00") for player in (11, 12, 13, 14)]
    + [(HOME, 15, "6:00"), (HOME, 16, "42:00")]
    + [(AWAY, player, "48:00") for player in (21, 22, 23, 24, 25)]
    + [(AWAY, 26, "")],
    columns=["TEAM_ID", "PLAYER_ID", "MIN"],
)


def _spans(rotation, team):
    return sorted(rotation[team])


def test_substitutions_and_full_game_starters_become_intervals():
    rotation = rebuild_rotation(_game(), BOX, (HOME, AWAY))
    home = {player: (start, end) for player, start, end in _spans(rotation, HOME)}
    assert home[15] == (0, 3600)
    # Back-to-back periods merge, as the API reports them.
    assert home[16] == (3600, 28800)
    assert home[11] == (0, 28800)


def test_a_starter_invisible_in_the_play_by_play_comes_from_the_box_score():
    rotation = rebuild_rotation(_game(), BOX, (HOME, AWAY))
    assert (25, 0, 28800) in rotation[AWAY]
    assert 26 not in {player for player, *_ in rotation[AWAY]}


def test_a_technical_on_the_bench_does_not_put_a_player_on_the_floor():
    technical = _event(900, 2, "11:30", FOUL, 26, AWAY, action=11)
    rotation = rebuild_rotation(_game([technical]), BOX, (HOME, AWAY))
    assert 26 not in {player for player, *_ in rotation[AWAY]}


def test_six_players_seen_before_any_substitution_is_refused():
    stray = _event(900, 2, "11:30", MADE_SHOT, 26, AWAY)
    with pytest.raises(RotationRebuildError, match="too_many_starters"):
        rebuild_rotation(_game([stray]), BOX, (HOME, AWAY))


def test_events_are_ordered_by_clock_not_by_event_number():
    # 16 scores after coming on, but the feed numbered that row before the sub.
    game = _game()
    late = game.index[(game.PLAYER1_ID == 16) & (game.PCTIMESTRING == "5:00")][0]
    game.loc[late, "EVENTNUM"] = 0
    rotation = rebuild_rotation(game, BOX, (HOME, AWAY))
    assert (16, 3600, 28800) in rotation[HOME]


def test_minutes_that_disagree_with_the_box_score_are_refused():
    box = BOX.copy()
    box.loc[box.PLAYER_ID.eq(16), "MIN"] = "30:00"
    with pytest.raises(RotationRebuildError, match="minutes_mismatch"):
        rebuild_rotation(_game(), box, (HOME, AWAY))


def test_a_game_missing_a_quarter_is_refused():
    game = _game()
    with pytest.raises(RotationRebuildError, match="missing_period"):
        rebuild_rotation(game.loc[game.PERIOD.ne(3)], BOX, (HOME, AWAY))


def test_payload_reads_back_and_passes_the_stint_builders_rotation_checks():
    game = _game()
    rotation = rebuild_rotation(game, BOX, (HOME, AWAY))
    raw = rotation_payload(GAME, rotation, home_team_id=HOME, pbp=game)
    assert rotation_from_payload(raw) == {team: sorted(spans, key=lambda s: (s[1], s[0]))
                                          for team, spans in rotation.items()}
    home, away, _ = decode_raw_game(raw, b'{"game": {"actions": []}}')
    assert _rotation_intervals(home)[0] == str(HOME)
    assert _rotation_intervals(away)[0] == str(AWAY)
    assert home.loc[home.PERSON_ID.eq(16), "PLAYER_LAST"].iloc[0] == "Last16"


def test_lineup_agreement_counts_team_seconds():
    rotation = rebuild_rotation(_game(), BOX, (HOME, AWAY))
    assert lineup_agreement(rotation, rotation) == 1.0
    swapped = {team: list(spans) for team, spans in rotation.items()}
    swapped[HOME] = [
        (player, start, end) if player not in (15, 16)
        else (player, 0, 7200) if player == 15 else (player, 7200, 28800)
        for player, start, end in rotation[HOME]
    ]
    # Home disagrees for 6 of 48 minutes, away never: 42/48 of home + 48/48.
    assert lineup_agreement(rotation, swapped) == pytest.approx((42 + 48) / 96)


def test_the_manifest_records_where_a_rotation_came_from(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.record_many(2020, [(GAME, "gamerotation", "ok", 10)], source=REBUILT_SOURCE)
    manifest.record(2020, "0022000002", "gamerotation", "ok", 10)
    reread = Manifest(tmp_path).load(2020).set_index("game_id")
    assert reread.loc[GAME, "source"] == REBUILT_SOURCE
    assert pd.isna(reread.loc["0022000002", "source"])
    assert Manifest(tmp_path).is_ok(2020, GAME, "gamerotation")
