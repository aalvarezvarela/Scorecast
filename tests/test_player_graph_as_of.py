"""The 2_7 point-in-time layer must never return data dated on or after T."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.as_of import (
    CLOSING,
    PointInTimeData,
    game_date_index,
)

DATES = {
    "0022300001": "2023-10-24",
    "0022300002": "2023-10-24",
    "0022300003": "2023-10-25",
    "0022300004": "2023-10-27",
}


def _stints(game_ids=None):
    game_ids = game_ids or list(DATES)
    return pd.DataFrame(
        {
            "game_id": game_ids,
            # Time of day must not matter: the cutoff is by date.
            "game_date": [
                pd.Timestamp(DATES[g]) + pd.Timedelta(hours=19) for g in game_ids
            ],
            "seg_idx": range(len(game_ids)),
        }
    )


def _box_scores():
    return pd.DataFrame(
        {
            "GAME_ID": [int(g) for g in DATES],  # ids from the DB may lose zeros
            "GAME_DATE": pd.to_datetime(list(DATES.values())),
            "PLAYER_ID": [1, 2, 3, 4],
            "MIN": [30.0, 25.0, 20.0, 35.0],
        }
    )


def _matchups():
    return pd.DataFrame(
        {
            "game_id": list(DATES)[::-1],
            "off_player_id": [1, 2, 3, 4],
            "def_player_id": [5, 6, 7, 8],
            "matchup_seconds": [60.0, 30.0, 90.0, 45.0],
        }
    )


@pytest.fixture
def data():
    return PointInTimeData.from_frames(
        _stints(), _matchups(), _box_scores(), {CLOSING: "closing-state"}
    )


@pytest.mark.parametrize("cutoff", list(DATES.values()) + ["2023-10-26", "2023-10-30"])
def test_every_view_is_strictly_before_its_cutoff(data, cutoff):
    view = data.as_of(cutoff)
    t = pd.Timestamp(cutoff)
    for frame, col in (
        (view.stints(), "game_date"),
        (view.matchups(), "game_date"),
        (view.box_scores(), "GAME_DATE"),
    ):
        assert (frame[col] < t).all()
        expected = sum(pd.Timestamp(d) < t for d in DATES.values())
        assert len(frame) == expected


def test_time_of_day_in_the_cutoff_is_ignored(data):
    # A 1 pm game finished before a 7 pm snapshot is still excluded.
    assert data.as_of("2023-10-25 23:59").stints()["game_id"].tolist() == [
        "0022300001",
        "0022300002",
    ]


def test_since_opens_a_window(data):
    window = data.as_of("2023-10-27").stints(since="2023-10-25")
    assert window["game_id"].tolist() == ["0022300003"]
    assert data.as_of("2023-10-24").stints(since="2023-10-25").empty


def test_matchups_are_dated_from_box_scores_and_sorted(data):
    matchups = data.matchups
    assert matchups["game_date"].is_monotonic_increasing
    assert matchups.set_index("game_id")["game_date"].to_dict() == {
        g: pd.Timestamp(d) for g, d in DATES.items()
    }


def test_matchup_without_a_known_date_is_rejected():
    matchups = pd.concat(
        [_matchups(), pd.DataFrame({"game_id": ["0022399999"]})], ignore_index=True
    )
    with pytest.raises(ValueError, match="no known date"):
        PointInTimeData.from_frames(_stints(), matchups, _box_scores())


def test_sources_that_disagree_on_a_date_are_rejected():
    stints = _stints()
    stints.loc[0, "game_date"] = pd.Timestamp("2023-11-01")
    with pytest.raises(ValueError, match="Conflicting dates"):
        game_date_index(
            (_box_scores(), "GAME_ID", "GAME_DATE"), (stints, "game_id", "game_date")
        )


def test_views_return_copies(data):
    first = data.as_of("2023-10-30").stints()
    first["seg_idx"] = -1
    assert (data.as_of("2023-10-30").stints()["seg_idx"] >= 0).all()


def test_injury_states_are_kept_per_horizon(data):
    assert data.injury_report() == "closing-state"
    with pytest.raises(KeyError, match="horizon 360"):
        data.injury_report(360)


def test_games_table_dates_matchups_without_box_scores():
    # 2017-18 has matchups and a games table but no player box scores.
    matchups = pd.concat(
        [_matchups(), pd.DataFrame({"game_id": ["0021700005"], "off_player_id": [9]})],
        ignore_index=True,
    )
    games = pd.DataFrame(
        {"GAME_ID": ["0021700005", "0021700005"], "GAME_DATE": ["2017-10-18"] * 2}
    )
    data = PointInTimeData.from_frames(
        _stints(), matchups, _box_scores(), game_dates=games
    )
    assert data.as_of("2017-10-19").matchups()["game_id"].tolist() == ["0021700005"]
    assert data.as_of("2017-10-18").matchups().empty


def test_lineup_arrays_cannot_be_edited_through_a_view():
    stints = _stints()
    stints["home_lineup"] = [np.array(["1", "2", "3", "4", "5"])] * len(stints)
    data = PointInTimeData.from_frames(stints, _matchups(), _box_scores())
    lineup = data.as_of("2023-10-30").stints()["home_lineup"].iloc[0]
    with pytest.raises(TypeError):
        lineup[0] = "-1"
    assert data.as_of("2023-10-30").stints()["home_lineup"].iloc[0][0] == "1"


def test_an_empty_stint_store_gives_empty_views():
    # read_stints returns a columnless frame when nothing is validated.
    data = PointInTimeData.from_frames(pd.DataFrame(), _matchups(), _box_scores())
    assert data.as_of("2023-10-30").stints().empty
    assert len(data.as_of("2023-10-30").matchups()) == 4


def test_load_keeps_box_scores_of_the_october_2020_finals(monkeypatch, tmp_path):
    from nba_ou import postgre_db
    from nba_ou.create_training_data.schema_layers import inputs
    from nba_ou.data_processing.lineups import stint_store

    spans = []
    finals = pd.DataFrame(
        {
            "GAME_ID": ["0041900406"],
            "GAME_DATE": [pd.Timestamp("2020-10-11")],
            "PLAYER_ID": [1],
        }
    )
    monkeypatch.setattr(
        inputs, "load_player_history", lambda dates: spans.append(dates) or finals
    )
    monkeypatch.setattr(
        postgre_db,
        "load_games_from_db",
        lambda seasons: pd.DataFrame(
            {"game_id": ["0041900406"], "game_date": ["2020-10-11"]}
        ),
    )
    monkeypatch.setattr(stint_store, "read_stints", lambda *a, **k: pd.DataFrame())
    data = PointInTimeData.load([2019], local_root=tmp_path)
    assert spans[0].max() == pd.Timestamp("2020-10-11")
    assert len(data.as_of("2020-10-12").box_scores()) == 1
