"""Source timings preserve database update skips, retries and failure behavior."""

import importlib
from unittest.mock import Mock

import pytest

update = importlib.import_module("nba_ou.postgre_db.update_all.update_all_databases")


@pytest.fixture
def sources(monkeypatch):
    fakes = {
        "get_live_game_ids": Mock(return_value=["live"]),
        "get_game_ids_for_date": Mock(return_value=["tonight"]),
        "resolve_season_years": Mock(return_value=[2026]),
        "update_team_players_database": Mock(return_value=(False, ["excluded"], True)),
        "update_odds_sportsbook_database": Mock(return_value={}),
        "update_odds_yahoo_database": Mock(return_value={}),
        "update_refs_injuries_database": Mock(return_value=False),
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(update, name, fake)
    monkeypatch.setattr(update.time, "sleep", Mock())
    return fakes


def test_each_source_is_timed_without_changing_its_inputs(sources, capsys):
    update.update_all_databases(headless=True, exclude_game_date="2026-10-29")
    kwargs = sources["update_team_players_database"].call_args.kwargs
    assert kwargs["season_year"] == "2026"
    assert set(kwargs["games_id_to_exclude"]) == {"live", "tonight"}
    sources["update_refs_injuries_database"].assert_called_once_with(
        season_year="2026", exclude_game_ids=["excluded"]
    )
    output = capsys.readouterr().out
    for label in (
        "games and player logs",
        "sportsbook odds",
        "Yahoo odds",
        "referees and injuries",
    ):
        assert f"[start] 2026 {label}" in output
        assert f"[time] 2026 {label}" in output


def test_no_new_games_still_skips_other_sources(sources, capsys):
    sources["update_team_players_database"].return_value = (False, [], False)
    update.update_all_databases(only_new_games=True)
    for name in (
        "update_odds_sportsbook_database",
        "update_odds_yahoo_database",
        "update_refs_injuries_database",
    ):
        sources[name].assert_not_called()
    assert "Skipping odds and refs/injuries" in capsys.readouterr().out


def test_retry_is_timed_separately_and_later_sources_still_run(sources, capsys):
    sources["update_odds_sportsbook_database"].side_effect = [RuntimeError("retry"), {}]
    update.update_all_databases()
    assert sources["update_odds_sportsbook_database"].call_count == 2
    sources["update_refs_injuries_database"].assert_called_once()
    output = capsys.readouterr().out
    assert "sportsbook odds (attempt 1/3)" in output
    assert "sportsbook odds (attempt 2/3)" in output
    assert "did not finish" in output


def test_exhausted_retries_publish_partial_timings_and_still_fail(
    sources, monkeypatch, tmp_path
):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    sources["update_odds_sportsbook_database"].side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="Sportsbook update failed during backfill"):
        update.update_all_databases()
    assert sources["update_odds_sportsbook_database"].call_count == 3
    sources["update_odds_yahoo_database"].assert_not_called()
    markdown = summary.read_text()
    assert "(interrupted)" in markdown
    assert "games and player logs" in markdown
    assert "sportsbook odds (attempt 3/3) (did not finish)" in markdown
