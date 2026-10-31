"""Check date selection and written-recap extraction without network requests."""

import json
from datetime import date
from unittest.mock import MagicMock

import pytest
import requests

from lab.game_reports import get_game_reports


@pytest.fixture
def reports():
    return get_game_reports


@pytest.fixture
def game():
    return {
        "game_id": "0022600141",
        "away": "Spurs",
        "home": "Jazz",
        "game_status": 3,
        "source_url": "https://www.nba.com/game/sas-vs-uta-0022600141",
    }


@pytest.fixture
def props(game):
    return {
        "game": {
            "gameId": game["game_id"],
            "gameEt": "2026-10-28T21:00:00Z",
            "gameStatus": 3,
            "awayTeam": {"score": 117},
            "homeTeam": {"score": 111},
            "gameRecap": {"excerpt": "This is a video description, not the article."},
        },
        "story": {
            "header": {
                "headline": "A fictional overtime report",
                "byline": "By TEST WRITER",
                "bytitle": "Associated Press",
            },
            "date": "2026-10-28T23:59:12",
            "storyType": "recap",
            "content": [
                "<p>TEST CITY (AP) A late basket decided the game.</p>",
                "The home team rallied — but fell short.",
            ],
        },
    }


def test_extracts_written_story_and_preserves_attribution(reports, props, game):
    report = reports.parse_report(props, game, date(2026, 10, 28))
    assert report["game_id"] == "0022600141"
    assert report["author"] == "By TEST WRITER"
    assert report["agency"] == "Associated Press"
    assert report["published_at"] == "2026-10-28T23:59:12"
    assert report["published_at_utc"] == "2026-10-29T03:59:12+00:00"
    assert "<p>" not in report["text"]
    assert "video description" not in report["text"]
    assert "\n\n" in report["text"]
    assert report["word_count"] > 0


@pytest.mark.parametrize("change", ["id", "date"])
def test_rejects_wrong_game_or_date(reports, props, game, change):
    props["game"]["gameId" if change == "id" else "gameEt"] = "wrong"
    with pytest.raises(ValueError, match="different"):
        reports.parse_report(props, game, date(2026, 10, 28))


@pytest.mark.parametrize("kind", ["missing", "preview", "empty"])
def test_missing_or_pregame_story_is_not_a_report(reports, props, game, kind):
    if kind == "missing":
        props["story"] = None
    elif kind == "preview":
        props["story"]["storyType"] = "preview"
    else:
        props["story"]["content"] = []
    assert reports.parse_report(props, game, date(2026, 10, 28)) is None


def test_reads_embedded_json_and_detects_blocked_pages(reports):
    page = {"props": {"pageProps": {"selectedDate": "2026-10-28"}}}
    html = f'<script type="application/json" id="__NEXT_DATA__">{json.dumps(page)}</script>'
    assert reports.parse_page_props(html) == page["props"]["pageProps"]
    with pytest.raises(ValueError, match="__NEXT_DATA__"):
        reports.parse_page_props("<html>Access denied</html>")


def test_empty_calendar_day_is_valid_but_wrong_date_is_rejected(reports):
    props = {"selectedDate": "2026-10-28", "gameCardFeed": {"modules": []}}
    assert reports.parse_games(props, date(2026, 10, 28)) == []
    with pytest.raises(ValueError, match="calendar date"):
        reports.parse_games(props, date(2026, 10, 29))


def test_download_keeps_processing_after_errors_and_reuses_saved_reports(
    reports, props, game, tmp_path, monkeypatch
):
    games = [
        game,
        {**game, "game_id": "0022600142"},
        {**game, "game_id": "0022600143", "game_status": 1},
    ]
    session = MagicMock()
    monkeypatch.setattr(reports, "create_session", lambda: session)
    monkeypatch.setattr(reports, "parse_games", lambda page, day: games)
    fetch = MagicMock(side_effect=[{}, props, requests.HTTPError("403 Forbidden")])
    monkeypatch.setattr(reports, "fetch_page_props", fetch)
    rows = reports.download_reports("2026-10-28", tmp_path)
    assert [row["status"] for row in rows] == ["saved", "error", "not_final"]
    saved_json = tmp_path / "2026-10-28/0022600141.json"
    saved_text = saved_json.with_suffix(".txt")
    assert json.loads(saved_json.read_text())["game_id"] == "0022600141"
    assert "TEST CITY (AP)" in saved_text.read_text()
    assert (saved_json.parent / "manifest.csv").is_file()

    fetch.reset_mock(side_effect=True)
    fetch.side_effect = [{}, requests.HTTPError("403 Forbidden")]
    rows = reports.download_reports("2026-10-28", tmp_path)
    assert [row["status"] for row in rows] == ["cached", "error", "not_final"]
    assert fetch.call_count == 2  # Only the calendar and the failed game are fetched.


@pytest.mark.parametrize(
    ("eastern", "utc"),
    [
        ("2025-12-17T00:07:19", "2025-12-17T05:07:19+00:00"),  # EST, UTC-5
        ("2026-04-18T01:00:12", "2026-04-18T05:00:12+00:00"),  # EDT, UTC-4
        ("", ""),
        (None, ""),
    ],
)
def test_story_dates_are_us_eastern_across_daylight_saving(reports, eastern, utc):
    assert reports.published_at_utc(eastern) == utc
