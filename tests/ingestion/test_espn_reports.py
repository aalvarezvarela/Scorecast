"""Historical fallback must retain article provenance and match the right game."""

from copy import deepcopy
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import requests

from lab.game_reports import espn_reports as espn
from lab.game_reports import season_reports as archive


@pytest.fixture
def entry():
    return {
        "game_id": "0021800002",
        "date": "2018-10-16",
        "source_url": "https://www.nba.com/game/okc-vs-gsw-0021800002",
        "away": "Thunder",
        "home": "Warriors",
        "phase": "regular_season",
    }


@pytest.fixture
def summary():
    return {
        "header": {
            "id": "test-event",
            "competitions": [
                {
                    "id": "test-event",
                    "date": "2018-10-17T02:30Z",
                    "status": {"type": {"completed": True}},
                    "competitors": [
                        {
                            "homeAway": "away",
                            "team": {"abbreviation": "OKC"},
                            "score": "100",
                        },
                        {
                            "homeAway": "home",
                            "team": {"abbreviation": "GS"},
                            "score": "108",
                        },
                    ],
                }
            ],
        },
        "article": {
            "type": "Recap",
            "gameId": "test-event",
            "headline": "A fictional opening night recap",
            "source": "Associated Press",
            "originallyPosted": "2018-10-17T06:00:00Z",
            "published": "2018-10-17T07:00:00Z",
            "lastModified": "2018-10-17T07:01:00Z",
            "story": '<p>A <a href="https://example.com">late basket</a>.</p><p>A second paragraph.</p>',
        },
    }


def test_matches_eastern_date_and_normalizes_team_codes(entry, summary):
    report = espn.parse_espn_report(summary, "test-event", entry)
    assert report["game_id"] == entry["game_id"]
    assert report["text"] == "A late basket.\n\nA second paragraph."
    assert report["source"] == "ESPN"
    assert report["agency"] == "Associated Press"
    assert (report["away_score"], report["home_score"]) == (100, 108)
    assert report["nba_source_url"] == entry["source_url"]
    assert "espn.com" in report["source_url"]
    assert report["published_at_utc"] == "2018-10-17T06:00:00+00:00"
    assert report["source_last_modified_at"] == "2018-10-17T07:01:00Z"


@pytest.mark.parametrize(
    ("espn_code", "nba_code"),
    [
        ("GS", "GSW"),
        ("SA", "SAS"),
        ("NO", "NOP"),
        ("NY", "NYK"),
        ("UTAH", "UTA"),
        ("WSH", "WAS"),
    ],
)
def test_espn_abbreviations_map_to_nba_tricodes(entry, summary, espn_code, nba_code):
    competition = summary["header"]["competitions"][0]
    competition["competitors"][1]["team"]["abbreviation"] = espn_code
    entry["source_url"] = (
        f"https://www.nba.com/game/okc-vs-{nba_code.lower()}-0021800002"
    )
    assert espn.matches_game(competition, entry)


@pytest.mark.parametrize("change", ["date", "teams", "header_id", "article_id"])
def test_wrong_event_is_rejected(entry, summary, change):
    competition = summary["header"]["competitions"][0]
    if change == "date":
        competition["date"] = "2018-10-18T02:30Z"
    elif change == "teams":
        competition["competitors"][0]["homeAway"] = "home"
        competition["competitors"][1]["homeAway"] = "away"
    elif change == "header_id":
        summary["header"]["id"] = "wrong"
    else:
        summary["article"]["gameId"] = "wrong"
    with pytest.raises(ValueError, match="different game"):
        espn.parse_espn_report(summary, "test-event", entry)


@pytest.mark.parametrize("change", ["preview", "empty", "unfinished"])
def test_only_complete_written_postgame_articles_are_accepted(entry, summary, change):
    if change == "preview":
        summary["article"]["type"] = "Preview"
    elif change == "empty":
        summary["article"]["story"] = ""
    else:
        summary["header"]["competitions"][0]["status"]["type"]["completed"] = False
    assert espn.parse_espn_report(summary, "test-event", entry) is None


def test_publication_timestamp_must_have_a_timezone(entry, summary):
    summary["article"]["originallyPosted"] = "2018-10-17T06:00:00"
    with pytest.raises(ValueError, match="time zone"):
        espn.parse_espn_report(summary, "test-event", entry)


def test_scoreboard_is_cached_and_each_request_is_paced(entry, summary):
    event = {"id": "test-event", "competitions": summary["header"]["competitions"]}
    session = MagicMock()
    session.get.return_value.json.side_effect = [{"events": [event]}, summary, summary]
    wait = MagicMock()
    source = espn.ESPNRecaps(wait)
    assert source.fetch(session, entry, 10)["espn_event_id"] == "test-event"
    assert source.fetch(session, entry, 10)["espn_event_id"] == "test-event"
    assert session.get.call_count == wait.call_count == 3


def test_no_match_is_unavailable_and_multiple_matches_are_errors(entry, summary):
    session = MagicMock()
    session.get.return_value.json.return_value = {"events": []}
    assert espn.ESPNRecaps(lambda: None).fetch(session, entry, 10) is None
    event = {"id": "test-event", "competitions": summary["header"]["competitions"]}
    session.get.return_value.json.return_value = {"events": [event, deepcopy(event)]}
    with pytest.raises(ValueError, match="Multiple"):
        espn.ESPNRecaps(lambda: None).fetch(session, entry, 10)


def test_nba_has_priority_and_missing_articles_use_opt_in_fallback(
    entry, summary, tmp_path, monkeypatch
):
    config = archive.SeasonConfig(2018, tmp_path, "test-bucket", espn_fallback=True)
    pacer = archive.RequestPacer(0)
    fetch = MagicMock(return_value=espn.parse_espn_report(summary, "test-event", entry))
    monkeypatch.setattr(pacer.espn, "fetch", fetch)
    nba = MagicMock(return_value=b"existing NBA article")
    monkeypatch.setattr(archive, "fetch_nba_report", nba)
    assert archive.fetch_report(entry, config, pacer) == b"existing NBA article"
    fetch.assert_not_called()
    nba.return_value = None
    assert (
        archive.fetch_report(entry, replace(config, espn_fallback=False), pacer) is None
    )
    fetch.assert_not_called()
    report = archive.load_report(archive.fetch_report(entry, config, pacer))
    assert report["source"] == "ESPN"
    assert report["nba_unavailable_reason"]


@pytest.mark.parametrize("status", [403, 404, 410, 429])
def test_only_missing_nba_pages_trigger_fallback_on_http_errors(
    entry, tmp_path, monkeypatch, status
):
    config = archive.SeasonConfig(2018, tmp_path, "test-bucket", espn_fallback=True)
    pacer = archive.RequestPacer(0)
    response = requests.Response()
    response.status_code = status
    monkeypatch.setattr(
        archive,
        "fetch_nba_report",
        MagicMock(side_effect=requests.HTTPError(response=response)),
    )
    fallback = MagicMock(return_value=None)
    monkeypatch.setattr(pacer.espn, "fetch", fallback)
    if status in {404, 410}:
        with pytest.raises(
            archive.ReportUnavailable, match="no matching written recap"
        ):
            archive.fetch_report(entry, config, pacer)
        fallback.assert_called_once()
    else:
        with pytest.raises(requests.HTTPError):
            archive.fetch_report(entry, config, pacer)
        fallback.assert_not_called()
