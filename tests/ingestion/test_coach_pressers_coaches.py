"""Step 2 of the coach presser archive: coach spells, staff and players."""

from __future__ import annotations

import json
from datetime import date

import pytest

from lab.coach_pressers import catalog, coaches
from lab.coach_pressers.names import full_forms, last_name, normalize
from tests.support.coach_pressers import (
    config_2021,
    players_2021,
    season_2021,
    wikipedia_2021,
)

SENTENCES = [
    (
        "On November 21, 2021, the Sacramento Kings fired head coach Luke Walton and "
        "named Alvin Gentry as their interim head coach.",
        ("2021-11-21", "SAC", "Alvin Gentry", "interim"),
    ),
    (
        "On February 19, 2020, the Cleveland Cavaliers named J. B. Bickerstaff the new "
        "head coach of the team after John Beilein resigned from the position.",
        ("2020-02-19", "CLE", "J. B. Bickerstaff", "head"),
    ),
    (
        "On March 7, 2020, the Brooklyn Nets and head coach Kenny Atkinson mutually "
        "agreed to part ways. Jacque Vaughn was named interim head coach.",
        ("2020-03-07", "BKN", "Jacque Vaughn", "interim"),
    ),
    (
        "On February 21, 2021, the Minnesota Timberwolves fired head coach Ryan "
        "Saunders after three seasons with the team.",
        ("2021-02-21", "MIN", None, "head"),
    ),
    (
        "On February 22, 2021, the Minnesota Timberwolves hired Chris Finch as their "
        "new head coach.",
        ("2021-02-22", "MIN", "Chris Finch", "head"),
    ),
    (
        "On April 8, 2025, the Denver Nuggets fired their head coach Michael Malone "
        "after 10 seasons with the team. David Adelman was named the interim head coach.",
        ("2025-04-08", "DEN", "David Adelman", "interim"),
    ),
]


@pytest.mark.parametrize("sentence,expected", SENTENCES)
def test_change_sentences(sentence, expected):
    event = coaches.parse_change(sentence)
    assert (event["date"], event["team"], event["incoming"], event["kind"]) == expected


def test_sentence_without_a_team_needs_a_hint():
    sentence = (
        "On November 2, 2024, Gregg Popovich suffered a mild stroke. Mitch Johnson "
        "was named the interim head coach."
    )
    assert coaches.parse_change(sentence) is None
    pages = {
        "SAS": {"infobox": [{"name": "Gregg Popovich", "note": ""}]},
        "SAC": {"infobox": [{"name": "Mike Brown", "note": "fired"}]},
    }
    events, unparsed = coaches.resolve_by_coach([], [sentence], pages)
    assert unparsed == []
    assert (events[0]["team"], events[0]["incoming"]) == ("SAS", "Mitch Johnson")


def test_team_page_ignores_navbox_and_footnotes():
    page = coaches.parse_team_season(
        wikipedia_2021().pages["2021–22 Sacramento Kings season"]
    )
    assert page["infobox"] == [
        {"name": "Luke Walton", "note": "fired"},
        {"name": "Alvin Gentry", "note": "interim"},
    ]
    assert page["head_coach"] == "Alvin Gentry"
    assert page["assistants"] == ["Doug Christie", "Jonah Herscu"]


def test_spells_and_expected_coach_per_game(tmp_path):
    config = config_2021(tmp_path)
    manifest = catalog.prepare_manifest(config, season_2021())
    built = coaches.build_coaches(
        config, manifest, wikipedia=wikipedia_2021(), players_loader=players_2021
    )
    sac = built["teams"]["SAC"]
    assert [
        (s["coach"], s["start_date"], s["end_date"], s["kind"]) for s in sac["spells"]
    ] == [
        ("Luke Walton", "2021-11-19", "2021-11-20", "head"),
        ("Alvin Gentry", "2021-11-21", "2021-12-01", "interim"),
    ]
    expected = {
        key: entry["coach_expected"]
        for key, entry in manifest["entries"].items()
        if entry["team"] == "SAC"
    }
    assert expected == {
        "0022100010:SAC": "Luke Walton",
        "0022100020:SAC": "Alvin Gentry",
        "0022100030:SAC": "Alvin Gentry",
    }
    assert manifest["entries"]["0022100010:PHX"]["coach_expected"] == "Monty Williams"
    names = {s["name"]: s["role"] for s in sac["staff"]}
    assert names["Luke Walton"] == "head" and names["Jonah Herscu"] == "assistant"
    assert sac["players"] == ["De'Aaron Fox", "Harrison Barnes"]
    assert built["unparsed_changes"] == [
        "On December 30, 2021, something unusual happened to a coach."
    ]
    assert json.loads(config.coaches_path.read_bytes())["season"] == "2021-22"


def test_overrides_win_over_wikipedia(tmp_path):
    config = config_2021(tmp_path)
    config.overrides_csv.write_text(
        "team,coach,start_date,end_date,kind,note\n"
        "SAC,Doug Christie,2021-12-01,2021-12-01,acting,protocols\n"
    )
    manifest = catalog.prepare_manifest(config, season_2021())
    coaches.build_coaches(
        config, manifest, wikipedia=wikipedia_2021(), players_loader=players_2021
    )
    assert manifest["entries"]["0022100030:SAC"]["coach_expected"] == "Doug Christie"
    assert manifest["entries"]["0022100020:SAC"]["coach_expected"] == "Alvin Gentry"


def test_infobox_coach_without_a_dated_change_is_reported():
    page = {
        "infobox": [
            {"name": "A Coach", "note": "fired"},
            {"name": "B Coach", "note": ""},
        ],
        "head_coach": "B Coach",
        "assistants": [],
    }
    spells, warnings = coaches.build_spells(
        page, [], date(2022, 10, 19), date(2023, 4, 9)
    )
    assert [s["coach"] for s in spells] == ["A Coach"]
    assert "B Coach" in warnings[0]


def test_missing_database_is_a_warning_not_a_failure(tmp_path):
    config = config_2021(tmp_path)
    manifest = catalog.prepare_manifest(config, season_2021())

    def broken(season):
        raise ConnectionError("no credentials")

    built = coaches.build_coaches(
        config, manifest, wikipedia=wikipedia_2021(), players_loader=broken
    )
    assert "no credentials" in built["warnings"][0]
    assert built["teams"]["SAC"]["players"] == []


def test_names():
    assert normalize("Darko Rajaković") == "darko rajakovic"
    assert normalize("De'Aaron Fox") == "deaaron fox"
    assert full_forms("J.B. Bickerstaff") == ["j b bickerstaff", "jb bickerstaff"]
    assert last_name("Gary Payton II") == "payton"
