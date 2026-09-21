"""The season archive must stand in for the API, not merely resemble it."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest
from nba_ou.fetch_data.nba_lineups.pbp_archive import (
    ACTION_FIELDS,
    archive_urls,
    download_season_csv,
    game_payload,
    games_in_csv,
    season_datasets,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    columns = [field for field in ACTION_FIELDS if field != "shotValue"]
    return pd.DataFrame(rows, columns=columns)


def _action(**overrides) -> dict:
    base = {field: "" for field in ACTION_FIELDS if field != "shotValue"}
    base.update(
        actionNumber=1, period=1, teamId=0, personId=0, xLegacy=0, yLegacy=0,
        shotDistance=0, isFieldGoal=0, pointsTotal=0, videoAvailable=0, actionId=1,
    )
    base.update(overrides)
    return base


def test_payload_matches_the_playbyplayv3_shape_the_parser_decodes():
    payload = game_payload(_frame([_action(description="Start of 1st Period")]), "0021800001")
    document = json.loads(payload)
    assert document["game"]["gameId"] == "0021800001"
    assert list(document["game"]["actions"][0]) == list(ACTION_FIELDS)


def test_three_pointers_are_recovered_from_the_description():
    rows = [
        _action(actionNumber=1, actionType="Made Shot", isFieldGoal=1,
                description="Tatum 26' 3PT Jump Shot (3 PTS)"),
        _action(actionNumber=2, actionType="Missed Shot", isFieldGoal=1,
                description="MISS Embiid 4' Layup"),
    ]
    actions = json.loads(game_payload(_frame(rows), "0021800001"))["game"]["actions"]
    assert [action["shotValue"] for action in actions] == [3, 2]


def test_a_block_row_is_not_mistaken_for_a_field_goal():
    # The API reports the blocked shot's value here, but the stint parser only
    # reads shotValue on made and missed shots, so 0 is the safe reconstruction.
    rows = [_action(actionType="", isFieldGoal=0, description="Horford BLOCK (1 BLK)")]
    actions = json.loads(game_payload(_frame(rows), "0021800001"))["game"]["actions"]
    assert actions[0]["shotValue"] == 0


def test_corrected_actions_keep_the_archives_row_order():
    # A turnover and its steal share an actionNumber; an unstable sort would
    # swap them and silently reattribute both to the wrong team.
    rows = [
        _action(actionNumber=204, actionType="Turnover", teamId=1, description="Bad Pass"),
        _action(actionNumber=204, actionType="", teamId=2, description="STEAL"),
        _action(actionNumber=100, actionType="", teamId=1, description="earlier"),
    ]
    actions = json.loads(game_payload(_frame(rows), "0021800001"))["game"]["actions"]
    assert [action["description"] for action in actions] == ["earlier", "Bad Pass", "STEAL"]


def test_blank_scores_and_padded_text_are_normalised():
    rows = [_action(scoreHome=float("nan"), scoreAway=12.0, actionType="Turnover      ")]
    action = json.loads(game_payload(_frame(rows), "0021800001"))["game"]["actions"][0]
    assert action["scoreHome"] == ""
    assert action["scoreAway"] == "12"
    assert action["actionType"] == "Turnover"


def test_game_ids_are_padded_to_the_archives_ten_digit_key(tmp_path: Path):
    frame = _frame([_action(), _action(actionNumber=2)])
    frame["gameId"] = 21800001
    csv_path = tmp_path / "season.csv"
    frame.to_csv(csv_path, index=False)
    assert [game_id for game_id, _ in games_in_csv(csv_path)] == ["0021800001"]


def test_season_datasets_cover_the_playoffs_by_default():
    assert season_datasets(2018) == ["nbastatsv3_2018", "nbastatsv3_po_2018"]
    assert season_datasets(2018, playoffs=False) == ["nbastatsv3_2018"]


def _fake_opener(payload: bytes):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload

    return lambda url: Response()


def test_the_dataset_listing_is_parsed_into_names_and_urls():
    listing = b"nbastatsv3_2018=https://example.test/a.tar.xz,nbastatsv3_2019=https://example.test/b.tar.xz"
    urls = archive_urls(opener=_fake_opener(listing))
    assert urls["nbastatsv3_2019"] == "https://example.test/b.tar.xz"


def test_a_season_download_yields_the_single_csv_inside(tmp_path: Path):
    body = b"gameId,actionNumber\n21800001,1\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        info = tarfile.TarInfo("nbastatsv3_2018.csv")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    extracted = download_season_csv(
        "https://example.test/a.tar.xz", tmp_path, opener=_fake_opener(buffer.getvalue())
    )
    assert extracted.read_bytes() == body


def test_an_archive_without_exactly_one_csv_is_refused(tmp_path: Path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:xz") as archive:
        info = tarfile.TarInfo("readme.txt")
        info.size = 0
        archive.addfile(info, io.BytesIO(b""))
    with pytest.raises(ValueError, match="Expected one CSV"):
        download_season_csv(
            "https://example.test/a.tar.xz", tmp_path, opener=_fake_opener(buffer.getvalue())
        )
