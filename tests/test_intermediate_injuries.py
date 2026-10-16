"""Snapshot injury features must use reports available at prediction time."""

from contextlib import nullcontext

import pandas as pd
import pytest
from nba_ou.create_training_data import intermediate_injuries as injuries
from nba_ou.create_training_data.create_base_game_features import BaseInjuryContext
from nba_ou.create_training_data.select_intermediate_columns import is_kept_column
from nba_ou.data_processing.injury_status.report_state import InjuryReportState
from nba_ou.data_processing.players.attach_player_features import (
    _restrict_snapshot_membership,
)
from nba_ou.postgre_db.injury_report_aiven import fetch


def test_snapshot_report_queries_use_explicit_utc_cutoffs_and_strict_time(monkeypatch):
    queries = []

    def fake_read_sql(query, _conn, params):
        queries.append((query, params))
        return pd.DataFrame()

    monkeypatch.setattr(fetch.pd, "read_sql_query", fake_read_sql)
    cutoffs = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "snapshot_minutes": [60, 30],
            "as_of": ["2025-11-01 22:00:00+00:00", "2025-11-01 22:30:00+00:00"],
        }
    )
    fetch.report_state_at_snapshots(object(), cutoffs)
    assert len(queries) == 3
    assert all("unnest(" in sql for sql, _ in queries)
    assert all("< c.as_of" in sql for sql, _ in queries)
    assert all(params["snapshot_minutes"] == [60, 30] for _, params in queries)
    assert all(timestamp.tzinfo is not None for timestamp in queries[0][1]["as_of"])

    with pytest.raises(ValueError, match="unique per game and horizon"):
        fetch.report_state_at_snapshots(
            object(), pd.concat([cutoffs, cutoffs.iloc[:1]])
        )


def test_snapshot_availability_effect_names_survive_the_gate():
    assert is_kept_column("TOP3_AVAILABILITY_EFFECT_HOME_MEAN_TOTAL_POINTS")
    assert is_kept_column("TOP3_INJURED_AVAILABILITY_EFFECT_AWAY_MEAN_SE_TOTAL_POINTS")
    assert is_kept_column(
        "TOP2_QUESTIONABLE_AVAILABILITY_EFFECT_HOME_SUM_N_TOTAL_GAMES"
    )


def test_current_roster_membership_cannot_use_settled_absences():
    membership = {"past": {"h": ["known"]}, "current": {"h": ["future_inactive"]}}
    result = _restrict_snapshot_membership(
        membership,
        {"current"},
        {("current", "h"): ["reported_out"]},
        {"current": {"h": ["reported_questionable"]}},
        {"current": {"h": ["reported_probable"]}},
    )
    assert result["past"] == {"h": ["known"]}
    assert set(result["current"]["h"]) == {
        "reported_out",
        "reported_questionable",
        "reported_probable",
    }


def test_report_states_are_partitioned_by_snapshot_horizon(monkeypatch):
    statuses = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "snapshot_minutes": [120, 30],
            "team_id": ["h", "h"],
            "player_id": ["p", "p"],
            "status": ["questionable", "out"],
            "reason_category": ["injury_illness"] * 2,
        }
    )
    filings = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "snapshot_minutes": [120, 30],
            "team_id": ["h", "h"],
            "submitted": [True, True],
        }
    )
    ages = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "snapshot_minutes": [120, 30],
            "report_age_minutes": [15.0, 10.0],
        }
    )
    monkeypatch.setattr(injuries, "connect_nba_db", lambda *_: nullcontext(object()))
    monkeypatch.setattr(
        injuries.fetch, "report_state_at_snapshots", lambda *_: (statuses, filings, ages)
    )
    monkeypatch.setattr(
        injuries.fetch,
        "listed_status_events",
        lambda *_: InjuryReportState.empty().status_events,
    )
    monkeypatch.setattr(
        injuries.fetch,
        "listed_pairs",
        lambda *_: InjuryReportState.empty().listed_pairs,
    )
    cutoffs = pd.DataFrame({"game_id": ["g1", "g1"], "snapshot_minutes": [120, 30]})
    states = injuries._states_at_snapshots(cutoffs)
    assert states[120].statuses["status"].tolist() == ["questionable"]
    assert states[30].statuses["status"].tolist() == ["out"]
    assert states[120].report_age["report_age_minutes"].tolist() == [15.0]


def test_unfiled_side_has_nan_injury_features_even_with_legacy_values(monkeypatch):
    game = "g1"
    home, away = "h", "a"
    team = pd.DataFrame(
        {
            "GAME_ID": [game, game],
            "TEAM_ID": [home, away],
            "HOME": [True, False],
            "GAME_DATE": pd.to_datetime(["2025-11-01"] * 2),
        }
    )
    context = BaseInjuryContext(
        team, pd.DataFrame(), ["2025-26"], ["2024-25", "2025-26"]
    )
    base = pd.DataFrame(
        {
            "GAME_ID": [game],
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME": [110.0],
            "PTS_SEASON_BEFORE_AVG_TEAM_AWAY": [108.0],
        }
    )
    state = InjuryReportState.empty()
    state.filings = pd.DataFrame(
        {"game_id": [game], "team_id": [home], "submitted": [True]}
    )

    def fake_player_features(team_frame, *_args, **_kwargs):
        result = team_frame.copy()
        result["TOP1_INJURED_PLAYER_PTS_BEFORE"] = [10.0, 99.0]
        result["TOTAL_INJURED_PLAYER_PTS_BEFORE"] = [10.0, 99.0]
        result["INJ_FRESH_OUT_PTS_BEFORE"] = [10.0, 99.0]
        return result, {}, {}

    def fake_report_features(team_frame, *_args):
        result = team_frame.copy()
        result["INJURY_REPORT_COVERED_BEFORE"] = [1, 0]
        result["LAST_STATUS_REPORT_AGE_MIN_BEFORE"] = [20.0, 20.0]
        return result

    def fake_all_star(team_frame, *_args, **_kwargs):
        result = team_frame.copy()
        for column in injuries.INJURED_ALL_STAR_COLUMNS:
            result[column] = 0.5
        return result

    monkeypatch.setattr(injuries, "add_player_history_features", fake_player_features)
    monkeypatch.setattr(injuries, "add_injury_report_features", fake_report_features)
    monkeypatch.setattr(injuries, "add_all_star_voting_features", fake_all_star)
    monkeypatch.setattr(
        injuries, "_add_availability_effects", lambda games, *_args, **_kwargs: games
    )
    result = injuries._one_horizon(
        base, context, state, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {game}
    ).iloc[0]
    assert result["INJURY_REPORT_COVERED_BEFORE_TEAM_HOME"] == 1
    assert result["INJURY_REPORT_COVERED_BEFORE_TEAM_AWAY"] == 0
    assert result["TOP1_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME"] == 10.0
    assert pd.isna(result["TOP1_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY"])
    assert pd.isna(result["LAST_STATUS_REPORT_AGE_MIN_BEFORE_TEAM_AWAY"])
    assert pd.isna(result["INJ_FRESH_OUT_PTS_SUM_BEFORE"])


def test_horizons_in_processes_equal_the_horizons_in_turn(monkeypatch, tmp_path):
    """Forked horizons give the same rows, memo entries and sums as a loop."""
    from nba_ou.create_training_data import history_memo
    from nba_ou.create_training_data import intermediate_injuries as ii

    game_ids = [f"002{s % 100:02d}{g:05d}" for s in (2023, 2024) for g in range(6)]
    horizons = [30, 60, 120, 240]
    cutoffs = pd.DataFrame(
        [(g, h) for g in game_ids for h in horizons],
        columns=["game_id", "snapshot_minutes"],
    )

    def one_horizon(base, context, state, *args, horizon=None, priors=None, **kw):
        player_rows, effect_cache = args[4], args[8]
        values = []
        for game_id in game_ids:
            key = (game_id, state, 0, 9, 9)
            if player_rows.rows.get(key) is None:
                player_rows.rows[key] = len(game_id) * state
            values.append(player_rows.rows.get(key))
            effect_cache.rows[("e", (int(game_id[3:5]) + 2000,), state)] = state
        priors.setdefault((horizon, "F"), {"sum": float(state)})
        return pd.DataFrame({"GAME_ID": game_ids, "X": values})

    monkeypatch.setattr(ii, "_one_horizon", one_horizon)

    def run(workers):
        backend = history_memo._LocalBackend(tmp_path / str(workers))
        memo = history_memo.HistoryMemo(backend, {2023: "a", 2024: "b"}, 2024)
        caches = [
            history_memo.PersistentRowCache(
                memo, name, season_of=season_of, size_positions=()
            )
            for name, season_of in (
                ("p", lambda key: history_memo.season_of_game_id(key[0])),
                ("e", lambda key: max(key[1])),
            )
        ]
        inputs = ii._HorizonInputs(
            base=None, context=None, states={h: h // 30 for h in horizons},
            cutoffs=cutoffs, injuries=None, box=None, voting=None,
            player_rows=caches[0], all_star_rows=None, prepared_players=None,
            prepared_all_star=None, effect_cache=caches[1], targets=None,
            priors={}, first_current_season=2024,
        )
        calls = []
        if workers > 1:
            parts = ii._horizons_in_processes(
                inputs, memo, workers, None, while_waiting=lambda: calls.append(1)
            )
            assert calls == [1]
        else:
            parts = [ii._horizon_features(inputs, h) for h in horizons]
        tables = {key: dict(table) for key, table in memo._tables.items()}
        return pd.concat(parts, ignore_index=True), tables, inputs.priors

    rows, tables, priors = run(1)
    forked_rows, forked_tables, forked_priors = run(3)
    pd.testing.assert_frame_equal(forked_rows, rows)
    assert forked_tables == tables and len(tables) == 4
    assert forked_priors == priors and len(priors) == 4

    # A failing worker costs time, never a value: the horizons run in turn.
    def broken(horizon):
        raise RuntimeError("worker died")

    monkeypatch.setattr(ii, "_forked_horizon", broken)
    fallback_rows, fallback_tables, fallback_priors = run(2)
    pd.testing.assert_frame_equal(fallback_rows, rows)
    assert fallback_tables == tables and fallback_priors == priors
