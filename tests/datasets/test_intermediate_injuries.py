"""Snapshot injury features must use reports available at prediction time."""

from contextlib import nullcontext
from types import SimpleNamespace

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


def test_historical_replay_bounds_status_form_to_the_same_history_date(monkeypatch):
    snapshots = pd.DataFrame(
        {
            "GAME_ID": ["g1"],
            "TIME_TO_MATCH_MIN": [1080],
            "SNAPSHOT_TS_UTC": [pd.Timestamp("2026-10-24T05:00Z")],
        }
    )
    monkeypatch.setattr(injuries, "_states_at_snapshots", lambda *a, **k: {})
    monkeypatch.setattr(injuries, "get_injury_data_from_db", lambda *a: pd.DataFrame())
    monkeypatch.setattr(
        injuries,
        "load_player_box_history",
        lambda: pd.DataFrame(
            {"game_date": pd.to_datetime(["2026-10-22", "2026-10-23", "2026-10-24"])}
        ),
    )

    class FormPrepared(Exception):
        pass

    def prepare(box):
        # The last two dates may include outcomes after the early snapshot.
        assert box.game_date.tolist() == [pd.Timestamp("2026-10-22")]
        raise FormPrepared

    monkeypatch.setattr(injuries, "prepare_box_and_form", prepare)
    context = SimpleNamespace(
        live_as_of=pd.Timestamp("2026-10-24T05:00Z"), player_context_seasons=["2026-27"]
    )
    with pytest.raises(FormPrepared):
        injuries.add_snapshot_injury_features(
            snapshots,
            pd.DataFrame(),
            context,
            target_game_ids={"g1"},
            history_limit=pd.Timestamp("2026-10-22"),
        )


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
        injuries.fetch,
        "report_state_at_snapshots",
        lambda *_: (statuses, filings, ages),
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


def test_tonights_snapshots_ahead_of_the_cutoff_read_the_reports_before_it(
    monkeypatch,
):
    # A queued run with cutoff 18:00 starts later: the store already holds the
    # reports since. g1's T-360 (20:00) reads them as of 18:00, its report age
    # still counted to 20:00; its T-720 (14:00) and earlier games are as read.
    cutoff = pd.Timestamp("2026-10-21 18:00", tz="UTC")
    cutoffs = pd.DataFrame(
        {
            "game_id": ["g0", "g1", "g1"],
            "snapshot_minutes": [360, 360, 720],
            "as_of": pd.to_datetime(
                ["2026-10-20 20:00", "2026-10-21 20:00", "2026-10-21 14:00"],
                utc=True,
            ),
        }
    )
    reads = []

    def read(_conn, rows):
        reads.append(rows)
        keys = rows[["game_id", "snapshot_minutes"]].reset_index(drop=True)
        return (
            keys.assign(
                team_id="h", player_id="p", status="out", reason_category="injury"
            ),
            keys.assign(team_id="h", submitted=True),
            keys.assign(report_age_minutes=10.0),
        )

    monkeypatch.setattr(injuries.history_memo, "default_backend", lambda: None)
    monkeypatch.setattr(injuries, "connect_nba_db", lambda *_: nullcontext(object()))
    monkeypatch.setattr(injuries.fetch, "report_state_at_snapshots", read)
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
    states = injuries._states_at_snapshots(
        cutoffs, target_game_ids={"g1"}, visible_until=cutoff
    )
    (rows,) = reads
    assert pd.to_datetime(rows["as_of"], utc=True).tolist() == [
        pd.Timestamp("2026-10-20 20:00", tz="UTC"),
        cutoff,
        pd.Timestamp("2026-10-21 14:00", tz="UTC"),
    ]
    ages = states[360].report_age.set_index("game_id")["report_age_minutes"]
    assert ages.to_dict() == {"g0": 10.0, "g1": 130.0}
    assert states[720].report_age["report_age_minutes"].tolist() == [10.0]


@pytest.mark.parametrize("home_filed", [True, False])
def test_unfiled_side_has_nan_injury_features_even_with_legacy_values(
    monkeypatch, home_filed
):
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
        {"game_id": [game], "team_id": [home], "submitted": [home_filed]}
    )

    def fake_player_features(team_frame, *_args, **_kwargs):
        result = team_frame.copy()
        result["TOP1_INJURED_PLAYER_PTS_BEFORE"] = [10.0, 99.0]
        result["TOTAL_INJURED_PLAYER_PTS_BEFORE"] = [10.0, 99.0]
        result["INJ_FRESH_OUT_PTS_BEFORE"] = [10.0, 99.0]
        return result, {}, {}

    def fake_report_features(team_frame, *_args):
        result = team_frame.copy()
        result["INJURY_REPORT_COVERED_BEFORE"] = [int(home_filed), 0]
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
    assert result["INJURY_REPORT_COVERED_BEFORE_TEAM_HOME"] == int(home_filed)
    assert result["INJURY_REPORT_COVERED_BEFORE_TEAM_AWAY"] == 0
    if home_filed:
        assert result["TOP1_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME"] == 10.0
    else:
        assert pd.isna(result["TOP1_INJURED_PLAYER_PTS_BEFORE_TEAM_HOME"])
        assert pd.isna(result["LAST_STATUS_REPORT_AGE_MIN_BEFORE_TEAM_HOME"])
    assert pd.isna(result["TOP1_INJURED_PLAYER_PTS_BEFORE_TEAM_AWAY"])
    assert pd.isna(result["LAST_STATUS_REPORT_AGE_MIN_BEFORE_TEAM_AWAY"])
    assert pd.isna(result["INJ_FRESH_OUT_PTS_SUM_BEFORE"])


def _worker_dies(horizon):
    # Module level so the pool can pickle it and the failure happens inside the
    # worker. A local function fails to pickle in the queue feeder thread, which
    # can deadlock ProcessPoolExecutor's shutdown lock once _stop kills the pool.
    raise RuntimeError("worker died")


def test_horizons_in_processes_equal_the_horizons_in_turn(monkeypatch, tmp_path):
    """Forked horizons give the same rows, memo entries and sums as a loop."""
    from nba_ou.create_training_data import history_memo
    from nba_ou.create_training_data import intermediate_injuries as ii

    game_ids = [f"002{s % 100:02d}{g:05d}" for s in (2023, 2024) for g in range(6)]
    horizons = [30, 60, 120, 240]
    rosters = {(game_ids[-1], "home"): {"rookie": "New player"}}
    cutoffs = pd.DataFrame(
        [(g, h) for g in game_ids for h in horizons],
        columns=["game_id", "snapshot_minutes"],
    )

    def one_horizon(base, context, state, *args, horizon=None, priors=None, **kw):
        assert kw["current_rosters"] == rosters
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
        memo = history_memo.HistoryMemo(backend, {2023: "a", 2024: "b"}, 2024, code="c")
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
            base=None,
            context=None,
            states={h: h // 30 for h in horizons},
            cutoffs=cutoffs,
            injuries=None,
            box=None,
            voting=None,
            player_rows=caches[0],
            all_star_rows=None,
            prepared_players=None,
            prepared_all_star=None,
            effect_cache=caches[1],
            targets=None,
            priors={},
            first_current_season=2024,
            current_rosters=rosters,
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
    monkeypatch.setattr(ii, "_forked_horizon", _worker_dies)
    fallback_rows, fallback_tables, fallback_priors = run(2)
    pd.testing.assert_frame_equal(fallback_rows, rows)
    assert fallback_tables == tables and fallback_priors == priors


def test_only_the_horizons_with_result_rows_are_computed():
    from nba_ou.create_training_data import intermediate_injuries as ii

    states = {h: object() for h in (0, 30, 60, 240, 243, 360)}
    tonight = pd.DataFrame({"GAME_ID": ["a", "b"], "TIME_TO_MATCH_MIN": [243, 360]})
    assert ii._wanted_horizons(states, tonight) == [243, 360]
    every = pd.DataFrame({"GAME_ID": ["a"] * 6, "TIME_TO_MATCH_MIN": list(states)})
    assert ii._wanted_horizons(states, every) == list(states)


def test_a_subset_of_forked_horizons_equals_those_horizons_in_turn(
    monkeypatch, tmp_path
):
    from nba_ou.create_training_data import history_memo
    from nba_ou.create_training_data import intermediate_injuries as ii

    game_ids = [f"00224{g:05d}" for g in range(4)]
    horizons = [30, 60, 120, 240]
    cutoffs = pd.DataFrame(
        [(g, h) for g in game_ids for h in horizons],
        columns=["game_id", "snapshot_minutes"],
    )

    def one_horizon(base, context, state, *args, horizon=None, priors=None, **kw):
        return pd.DataFrame(
            {"GAME_ID": game_ids, "X": [state * 10 + i for i in range(4)]}
        )

    monkeypatch.setattr(ii, "_one_horizon", one_horizon)

    def caches(memo):
        return [
            history_memo.PersistentRowCache(memo, name, season_of=lambda key: 2024)
            for name in ("p", "e")
        ]

    def inputs_for(memo):
        players, effects = caches(memo)
        return ii._HorizonInputs(
            base=None,
            context=None,
            states={h: h // 30 for h in horizons},
            cutoffs=cutoffs,
            injuries=None,
            box=None,
            voting=None,
            player_rows=players,
            all_star_rows=None,
            prepared_players=None,
            prepared_all_star=None,
            effect_cache=effects,
            targets=None,
            priors={},
            first_current_season=None,
            current_rosters={},
        )

    memo = history_memo.HistoryMemo(
        history_memo._LocalBackend(tmp_path), {2024: "a"}, 2024, code="c"
    )
    forked = ii._horizons_in_processes(
        inputs_for(memo), memo, 2, None, horizons=[60, 240]
    )
    in_turn = [ii._horizon_features(inputs_for(memo), h) for h in (60, 240)]
    pd.testing.assert_frame_equal(
        pd.concat(forked, ignore_index=True), pd.concat(in_turn, ignore_index=True)
    )
    assert set(pd.concat(forked)["TIME_TO_MATCH_MIN"]) == {60, 240}
