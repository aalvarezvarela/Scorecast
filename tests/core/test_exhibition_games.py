"""Preseason and All-Star games are classified by their game id's prefix."""

import pandas as pd
from nba_ou.config.constants import EXHIBITION_SEASON_TYPES, SEASON_TYPE_MAP
from nba_ou.utils.seasons import (
    EXHIBITION_PREFIXES,
    drop_exhibition_games,
    is_exhibition_game_id,
)

#: One game of every season type, in SEASON_TYPE_MAP's order.
GAME_IDS = [
    "0012400001",
    "0022400001",
    "0032400001",
    "0042400101",
    "0052400101",
    "0062400001",
]


def test_the_exhibitions_are_preseason_and_all_star():
    assert set(EXHIBITION_SEASON_TYPES) == {"Preseason", "All Star"}
    assert EXHIBITION_PREFIXES == {"001", "003"}
    assert {SEASON_TYPE_MAP[p] for p in EXHIBITION_PREFIXES} == set(
        EXHIBITION_SEASON_TYPES
    )


def test_every_season_type_is_classified():
    flags = is_exhibition_game_id(pd.Series(GAME_IDS))
    assert flags.tolist() == [True, False, True, False, False, False]


def test_ids_stored_without_leading_zeros_are_padded():
    flags = is_exhibition_game_id(pd.Series([12400001, 22400001, 32400001]))
    assert flags.tolist() == [True, False, True]


def test_a_missing_id_is_not_an_exhibition():
    flags = is_exhibition_game_id(pd.Series(["0012400001", None, float("nan")]))
    assert flags.tolist() == [True, False, False]


def test_drop_keeps_regular_season_playoffs_play_in_and_cup_final():
    frame = pd.DataFrame({"GAME_ID": GAME_IDS, "PTS": range(6)})
    kept = drop_exhibition_games(frame)
    assert kept["GAME_ID"].str[:3].tolist() == ["002", "004", "005", "006"]
    assert kept["PTS"].tolist() == [1, 3, 4, 5]
    assert kept.index.tolist() == [0, 1, 2, 3]


def test_drop_finds_a_lower_case_game_id_column():
    frame = pd.DataFrame({"game_id": GAME_IDS})
    assert len(drop_exhibition_games(frame)) == 4


def test_drop_leaves_a_frame_without_exhibitions_untouched():
    frame = pd.DataFrame({"GAME_ID": ["0022400001", "0042400101"]})
    assert drop_exhibition_games(frame) is frame


def test_drop_passes_none_and_frames_without_a_game_id_through():
    assert drop_exhibition_games(None) is None
    frame = pd.DataFrame({"PLAYER_ID": [1]})
    assert drop_exhibition_games(frame) is frame
