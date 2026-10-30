"""Read the training opener from the SBR main page, with canonical NBA IDs."""

import json

import pandas as pd
import pytest
from nba_ou.create_training_data import get_all_info_for_scheduled_games as info
from nba_ou.data_processing.odds.scheduled_opener import OPENER_COLUMNS


@pytest.mark.parametrize("lowercase_schedule", [False, True])
def test_main_opener_is_mapped_from_sbr_to_nba_and_keeps_both_raw_prices(
    monkeypatch, lowercase_schedule
):
    model = {
        "sportsbooks": [],
        "gameRows": [
            {
                "gameView": {
                    "gameId": 123456,
                    "startDate": "2026-10-30T00:00Z",
                    "awayTeam": {"displayName": "Sacramento"},
                    "homeTeam": {"displayName": "New Orleans"},
                    "consensus": {"overPickPercent": 99, "underPickPercent": 1},
                },
                "openingLineViews": [
                    {
                        "openingLine": {
                            "total": 238.5,
                            "overOdds": -115,
                            "underOdds": -105,
                        }
                    }
                ],
            }
        ],
    }
    payload = {
        "props": {
            "pageProps": {"oddsTables": [{"league": "NBA", "oddsTableModel": model}]}
        }
    }
    html = '<script id="__NEXT_DATA__">' + json.dumps(payload) + "</script>"

    def fetch(session, url):
        assert url.endswith("/totals/full-game/?date=2026-10-29")
        return html

    monkeypatch.setattr(info, "fetch_sbr_page", fetch)
    if lowercase_schedule:
        schedule = pd.DataFrame(
            {
                "game_id": ["0022600145"],
                "game_date": [pd.Timestamp("2026-10-29").date()],
                "team_home": ["New Orleans Pelicans"],
                "team_away": ["Sacramento Kings"],
            }
        )
    else:
        schedule = pd.DataFrame(
            {
                "GAME_ID": ["0022600145"],
                "GAME_DATE_EST": ["2026-10-29"],
                "HOME_TEAM_ID": ["1610612740"],
                "VISITOR_TEAM_ID": ["1610612758"],
            }
        )
    result = info.get_sportsbook_opener_data("2026-10-29", schedule)
    assert result.columns.tolist() == ["game_id", *OPENER_COLUMNS]
    assert result.game_id.tolist() == ["0022600145"]
    assert result.iloc[0].total_consensus_opener_line_over == 238.5
    assert result.iloc[0].total_consensus_opener_price_over == -115
    assert result.iloc[0].total_consensus_opener_price_under == -105


def test_main_page_without_games_returns_missing_openers(monkeypatch):
    monkeypatch.setattr(
        info,
        "fetch_sbr_page",
        lambda *a: '<script id="__NEXT_DATA__">{"props":{"pageProps":{}}}</script>',
    )
    result = info.get_sportsbook_opener_data("2026-10-29", pd.DataFrame())
    assert result.empty and result.columns.tolist() == ["game_id", *OPENER_COLUMNS]
