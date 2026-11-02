"""The 2_7 point-in-time layer must never return data dated on or after T."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.as_of import (
    BOX_SOURCE,
    CLOSING,
    PointInTimeData,
    game_date_index,
    with_local_box_scores,
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


# As in the season CSVs: ids without leading zeros, minutes as "MM.000000:SS",
# turnovers called TO, empty minutes for a DNP.
CSV_ROWS = [
    {"GAME_ID": 21700001, "TEAM_ID": 1610612737, "PLAYER_ID": 203145,
     "START_POSITION": "G", "MIN": "30.000000:30", "PTS": 12, "TO": 2},
    {"GAME_ID": 21700001, "TEAM_ID": 1610612737, "PLAYER_ID": 2,
     "START_POSITION": None, "MIN": None, "PTS": 0, "TO": 0},
    # Not in the games table (e.g. an exhibition): dropped, as from the DB.
    {"GAME_ID": 91700001, "TEAM_ID": 1, "PLAYER_ID": 3,
     "START_POSITION": None, "MIN": "10.000000:00", "PTS": 4, "TO": 0},
]  # fmt: skip
GAMES_2017 = pd.DataFrame({"GAME_ID": ["0021700001"], "GAME_DATE": ["2017-10-17"]})


def _write_csv(root, year, rows):
    path = root / f"nba_players_{year}_{(year + 1) % 100:02d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _db_box(game_id="0021800001", date="2018-10-16"):
    return pd.DataFrame(
        {
            "GAME_ID": [game_id],
            "GAME_DATE": [pd.Timestamp(date)],
            "PLAYER_ID": ["9"],
            "MIN": [20.0],
            "TOV": [1.0],
        }
    )


def test_a_season_the_database_lacks_is_read_from_the_csvs(tmp_path):
    _write_csv(tmp_path, 2017, CSV_ROWS)
    box = with_local_box_scores(_db_box(), [2017, 2018], GAMES_2017, tmp_path)
    local = box.loc[box[BOX_SOURCE].eq("csv")].set_index("PLAYER_ID")
    assert box[BOX_SOURCE].value_counts().to_dict() == {"csv": 2, "db": 1}
    assert list(local.index) == ["203145", "2"]
    assert (local["GAME_ID"] == "0021700001").all()
    assert (local["GAME_DATE"] == pd.Timestamp("2017-10-17")).all()
    assert local.loc["203145", "MIN"] == pytest.approx(30.5)
    assert local.loc["2", "MIN"] == 0
    assert local.loc["203145", "TOV"] == 2


def test_a_season_the_database_holds_is_never_mixed_with_csv_rows(tmp_path):
    _write_csv(tmp_path, 2017, CSV_ROWS)
    db = _db_box("0021700001", "2017-10-17")
    box = with_local_box_scores(db, [2017], GAMES_2017, tmp_path)
    assert box[BOX_SOURCE].eq("db").all() and len(box) == 1


def test_game_graph_rosters_read_only_the_database_rows(tmp_path):
    _write_csv(tmp_path, 2017, CSV_ROWS)
    box = with_local_box_scores(_db_box(), [2017, 2018], GAMES_2017, tmp_path)
    data = PointInTimeData.from_frames(pd.DataFrame(), pd.DataFrame(), box)
    assert data.box_scores_2_6["GAME_ID"].tolist() == ["0021800001"]
    assert len(data.as_of("2018-10-17").box_scores()) == 3


def test_load_reads_the_csvs_for_seasons_without_database_box_scores(
    monkeypatch, tmp_path
):
    from nba_ou import postgre_db
    from nba_ou.create_training_data.schema_layers import inputs
    from nba_ou.data_processing.lineups import stint_store

    _write_csv(tmp_path / "season_games_data", 2017, CSV_ROWS)
    monkeypatch.setattr(
        inputs,
        "load_player_history",
        lambda dates: pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "PLAYER_ID"]),
    )
    monkeypatch.setattr(postgre_db, "load_games_from_db", lambda seasons: GAMES_2017)
    monkeypatch.setattr(stint_store, "read_stints", lambda *a, **k: pd.DataFrame())
    data = PointInTimeData.load([2017], local_root=tmp_path)
    assert len(data.as_of("2017-10-18").box_scores()) == 2
    assert data.box_scores_2_6.empty


def test_load_rejects_seasons_before_2_7_starts(tmp_path):
    with pytest.raises(ValueError, match="2_7 starts in 2016"):
        PointInTimeData.load([2015, 2016], local_root=tmp_path)
