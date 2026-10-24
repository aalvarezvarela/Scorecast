"""Who-guarded-whom: both sources land in one schema, offence and defence intact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from nba_ou.fetch_data.nba_lineups.client import (
    EmptyResponse,
    GameUnavailable,
    LineupClient,
    ServerTimeout,
    _nonempty,
)
from nba_ou.fetch_data.nba_lineups.matchups import (
    COLUMNS,
    STAT_COLUMNS,
    MatchupStore,
    fill_from_api,
    from_api_payload,
    from_archive_csv,
    import_season,
    season_datasets,
)

HOME, AWAY = 1610612760, 1610612759
STATS = {
    "matchupMinutes": "1:12",
    "matchupMinutesSort": 72.0,
    "partialPossessions": 5.0,
    "percentageDefenderTotalTime": 0.117,
    "percentageOffensiveTotalTime": 0.117,
    "percentageTotalTimeBothOn": 0.141,
    "switchesOn": 0,
    "playerPoints": 3,
    "teamPoints": 10,
    "matchupAssists": 2,
    "matchupPotentialAssists": 0,
    "matchupTurnovers": 1,
    "matchupBlocks": 0,
    "matchupFieldGoalsMade": 1,
    "matchupFieldGoalsAttempted": 2,
    "matchupFieldGoalsPercentage": 0.5,
    "matchupThreePointersMade": 1,
    "matchupThreePointersAttempted": 1,
    "matchupThreePointersPercentage": 1.0,
    "helpBlocks": 0,
    "helpFieldGoalsMade": 0,
    "helpFieldGoalsAttempted": 0,
    "helpFieldGoalsPercentage": 0.0,
    "matchupFreeThrowsMade": 0,
    "matchupFreeThrowsAttempted": 0,
    "shootingFouls": 0,
}


def _payload(game_id: str = "0022500010", *, empty: bool = False) -> dict:
    def team(team_id, off_id, def_id):
        players = [] if empty else [
            {
                "personId": off_id, "firstName": "Jalen", "familyName": "Williams",
                "nameI": "J. Williams", "playerSlug": "x", "position": "F",
                "comment": "", "jerseyNum": "8",
                "matchups": [{
                    "personId": def_id, "firstName": "Harrison", "familyName": "Barnes",
                    "nameI": "H. Barnes", "playerSlug": "y", "jerseyNum": "40",
                    "statistics": STATS,
                }],
            }
        ]
        return {"teamId": team_id, "teamCity": "c", "teamName": "n",
                "teamTricode": "T", "teamSlug": "s", "players": players}

    return {"meta": {}, "boxScoreMatchups": {
        "gameId": game_id, "awayTeamId": AWAY, "homeTeamId": HOME,
        "homeTeam": team(HOME, 1631114, 203084),
        "awayTeam": team(AWAY, 203084, 1631114),
    }}


def _archive_frame(game_ids=(22500010,)) -> pd.DataFrame:
    """The archive's flattened snake_case layout for the home team's row."""
    rows = []
    for game_id in game_ids:
        rows.append({
            "game_id": game_id, "away_team_id": AWAY, "home_team_id": HOME,
            "team_id": HOME, "team_name": "n", "team_city": "c", "team_tricode": "T",
            "team_slug": "s", "person_id": 1631114, "first_name": "Jalen",
            "family_name": "Williams", "name_i": "J. Williams", "player_slug": "x",
            "position": "F", "comment": None, "jersey_num": 8.0,
            "matchups_person_id": 203084, "matchups_first_name": "Harrison",
            "matchups_family_name": "Barnes", "matchups_name_i": "H. Barnes",
            "matchups_player_slug": "y", "matchups_jersey_num": 40.0,
            "matchup_minutes": "1:12",
            **{theirs: STATS[_camel(theirs)] for theirs in STAT_COLUMNS},
        })
    return pd.DataFrame(rows)


def _camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(part.title() for part in rest)


def test_outer_player_is_the_offence_and_inner_the_defender():
    rows = from_api_payload(_payload(), 2025)
    home = rows.loc[rows.off_team_id.eq(HOME)].iloc[0]
    assert home.off_player_id == 1631114
    assert home.def_player_id == 203084
    assert home.def_team_id == AWAY
    assert set(rows.def_team_id) == {HOME, AWAY}
    assert (rows.def_team_id != rows.off_team_id).all()


def test_archive_and_api_render_identical_rows():
    api = from_api_payload(_payload(), 2025)
    api = api.loc[api.off_team_id.eq(HOME)].drop(columns="source").reset_index(drop=True)
    archive = from_archive_csv(_archive_frame(), 2025).drop(columns="source")
    pd.testing.assert_frame_equal(api, archive)


def test_schema_ids_and_season_type():
    rows = from_archive_csv(_archive_frame([22500010, 42500401]), 2025)
    assert list(rows.columns) == list(COLUMNS)
    assert list(rows.game_id) == ["0022500010", "0042500401"]
    assert list(rows.season_type) == ["Regular Season", "Playoffs"]
    assert rows.matchup_seconds.iloc[0] == 72.0


def test_a_game_from_another_season_is_refused():
    with pytest.raises(ValueError, match="do not belong to season 2024"):
        from_archive_csv(_archive_frame(), 2024)


def test_an_archive_with_a_renamed_column_is_refused():
    with pytest.raises(ValueError, match="lacks columns"):
        from_archive_csv(_archive_frame().drop(columns="matchups_person_id"), 2025)


def test_an_untracked_game_yields_no_rows_and_is_not_nonempty():
    assert from_api_payload(_payload(empty=True), 2025).empty
    assert not _nonempty(_payload(empty=True), "boxscorematchupsv3")
    assert _nonempty(_payload(), "boxscorematchupsv3")


def test_season_datasets_cover_the_playoffs_by_default():
    assert season_datasets(2020) == ["matchups_2020", "matchups_po_2020"]
    assert season_datasets(2020, playoffs=False) == ["matchups_2020"]


def test_store_replaces_whole_games_and_tracks_status(tmp_path: Path):
    store = MatchupStore(tmp_path)
    store.write(2025, from_archive_csv(_archive_frame([22500010, 22500011]), 2025))
    api = from_api_payload(_payload("0022500011"), 2025)
    store.write(2025, api, empty_games=["0022500012"])

    rows = store.read(2025)
    # Game 11 now holds the API's two rows, not the archive's one.
    assert rows.groupby("game_id").size().to_dict() == {"0022500010": 1, "0022500011": 2}
    status = store.statuses(2025).set_index("game_id")
    assert status.loc["0022500010", "source"] == "archive"
    assert status.loc["0022500011", "source"] == "api"
    assert status.loc["0022500012", "status"] == "empty"
    assert store.settled_games(2025) == {"0022500010", "0022500011", "0022500012"}


def test_import_season_reads_both_datasets(tmp_path: Path):
    def download(url, destination):
        path = destination / "season.csv"
        game = 22500010 if url.endswith("rs") else 42500401
        _archive_frame([game]).to_csv(path, index=False)
        return path

    store = MatchupStore(tmp_path)
    counts = import_season(
        2025, store=store,
        urls={"matchups_2025": "u/rs", "matchups_po_2025": "u/po"},
        download=download,
    )
    assert counts == {"games": 2, "rows": 2, "missing_dataset": 0}
    assert set(store.read(2025).season_type) == {"Regular Season", "Playoffs"}


def test_import_season_without_a_published_dataset_writes_nothing(tmp_path: Path):
    store = MatchupStore(tmp_path)
    counts = import_season(2026, store=store, urls={})
    assert counts["missing_dataset"] == 2
    assert not store.path(2026).exists()


class FakeClient:
    def __init__(self, outcomes: dict):
        self.outcomes = outcomes
        self.calls = []

    def fetch(self, endpoint, game_id):
        self.calls.append((endpoint, game_id))
        outcome = self.outcomes[game_id]
        if isinstance(outcome, Exception):
            raise outcome
        return json.dumps(outcome).encode()


def test_fill_records_data_and_true_empties_but_retries_server_errors(tmp_path: Path):
    store = MatchupStore(tmp_path)
    client = FakeClient({
        "0022500010": _payload("0022500010"),
        "0022500011": EmptyResponse("no rows"),
        "0022500012": GameUnavailable("fast 500"),
        "0022500013": ServerTimeout("slow 500"),
    })
    counts = fill_from_api(2025, list(client.outcomes), store=store, client=client)
    assert counts == {"ok": 1, "empty": 1, "retry_later": 2}
    assert {endpoint for endpoint, _ in client.calls} == {"boxscorematchupsv3"}
    # Neither kind of 500 is settled, so the next run asks again.
    assert store.settled_games(2025) == {"0022500010", "0022500011"}


def test_fill_keeps_progress_when_the_circuit_opens(tmp_path: Path):
    from nba_ou.fetch_data.nba_lineups.client import CircuitOpen

    store = MatchupStore(tmp_path)
    client = FakeClient({
        "0022500010": _payload("0022500010"),
        "0022500011": CircuitOpen("blocked"),
    })
    with pytest.raises(CircuitOpen):
        fill_from_api(2025, list(client.outcomes), store=store, client=client)
    assert store.settled_games(2025) == {"0022500010"}


def test_the_client_knows_the_matchups_endpoint():
    assert "boxscorematchupsv3" in LineupClient().endpoints
