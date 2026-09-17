"""Scrape provenance for SBR closes, the stored-close repair and tipoff corrections.

SBR day pages show each book's last number, which after tip is a live line. The
scraper must record when it looked and what SBR showed, the table must only
accept a pre-tip refresh, and none of that bookkeeping may reach a model frame.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from nba_ou.config import odds_columns as oc
from nba_ou.data_processing.odds.closing_line_repair import closing_quote_columns
from nba_ou.fetch_data.nba_schedule.tipoff_corrections import (
    TIPOFF_CORRECTIONS,
    apply_tipoff_corrections,
    render_corrections_module,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import attach_scrape_metadata
from nba_ou.postgre_db.odds_sportsbook.create_db import (
    create_odds_sportsbook_db as db,
)
from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db import (
    fetch_data_from_odds_sportsbook_db as loader,
)
from nba_ou.postgre_db.line_history_aiven import ingest
from nba_ou.postgre_db.line_history_aiven.tipoff_corrections import (
    find_tipoff_mismatches,
)
from nba_ou.postgre_db.odds_sportsbook.repair_closes import current_season_year


class TestScrapeMetadata:
    MODEL = {
        "gameRows": [
            {
                "gameView": {
                    "gameId": 362745,
                    "startDate": "2026-01-31T17:00:00+00:00",
                    "gameStatusText": "Final",
                }
            },
            {
                "gameView": {
                    "gameId": 362746,
                    "startDate": "2026-02-01T00:00:00+00:00",
                    "gameStatusText": "7:00 PM ET",
                }
            },
        ]
    }

    def test_each_game_gets_its_own_start_and_status(self):
        day = pd.DataFrame({"game_id": ["362745", "362746"]})
        scraped_at = pd.Timestamp("2026-01-31T19:00:00Z")

        out = attach_scrape_metadata(day, self.MODEL, scraped_at)

        assert (out["scraped_at"] == scraped_at).all()
        assert out["sbr_start_time_utc"].tolist() == [
            pd.Timestamp("2026-01-31T17:00:00Z"),
            pd.Timestamp("2026-02-01T00:00:00Z"),
        ]
        assert out["sbr_status_at_scrape"].tolist() == ["Final", "7:00 PM ET"]
        # Scraped two hours after the first tip: that row is not a close.
        assert (out["scraped_at"] < out["sbr_start_time_utc"]).tolist() == [False, True]

    def test_browser_rows_without_a_model_never_look_pre_tip(self):
        out = attach_scrape_metadata(
            pd.DataFrame({"game_id": ["1"]}), None, pd.Timestamp("2026-01-31T12:00Z")
        )
        assert out["sbr_start_time_utc"].isna().all()


class TestTableAndUpsert:
    def test_table_carries_the_provenance_columns(self):
        names = [name for name, _ in db._sportsbook_columns()]
        for column in oc.SPORTSBOOK_METADATA_COLUMNS:
            assert column in names

    def test_default_upsert_fills_scores_but_gates_odds(self):
        cols = [name for name, _ in db._sportsbook_columns()]
        text = db.build_upsert_query("s", "t", cols).as_string(None)

        assert '"home_points" = COALESCE(EXCLUDED."home_points", "t"."home_points")' in text
        assert '"total_bet365_line_over" = CASE WHEN' in text
        assert '"scraped_at" = CASE WHEN' in text
        # Identity and the repair stamp are never written by a scrape.
        for column in ("game_id", "game_date", "team_home", "closes_repaired_at"):
            assert f'"{column}" =' not in text

    def test_backfill_clears_the_repair_stamp(self):
        cols = [name for name, _ in db._sportsbook_columns()]
        text = db.build_upsert_query(
            "s", "t", cols, ["total_betrivers_line_over"]
        ).as_string(None)
        assert '"closes_repaired_at" = NULL' in text


class TestMetadataNeverReachesFrames:
    def test_drop_helper(self):
        df = pd.DataFrame(columns=["game_id", "scraped_at", "closes_repaired_at", "x"])
        assert list(oc.drop_sportsbook_metadata_columns(df).columns) == ["game_id", "x"]

    def test_closing_loader_drops_them(self, monkeypatch):
        frame = pd.DataFrame(
            {
                "game_id": ["1"],
                "total_bet365_line_over": [220.5],
                "scraped_at": [pd.Timestamp("2026-01-01T00:00Z")],
                "closes_repaired_at": [pd.Timestamp("2026-01-02T00:00Z")],
            }
        )

        class _Conn:
            def close(self):
                pass

        monkeypatch.setattr(loader, "connect_nba_db", lambda: _Conn())
        monkeypatch.setattr(
            loader.sql.Composed, "as_string", lambda self, conn=None: "SELECT"
        )
        monkeypatch.setattr(loader.pd, "read_sql_query", lambda *a, **k: frame.copy())

        out = loader.load_odds_sportsbook_from_db()

        assert list(out.columns) == ["game_id", "total_bet365_line_over"]


class TestRepairHelpers:
    def test_closing_quote_columns_skip_consensus_and_metadata(self):
        df = pd.DataFrame(
            columns=[
                "game_id",
                "total_consensus_opener_line_over",
                "total_consensus_opener_price_over",
                "total_bet365_line_over",
                "total_bet365_line_under",
                "total_bet365_price_over",
                "total_bet365_price_under",
                "ml_bet365_price_away",
                "ml_bet365_price_home",
                "scraped_at",
            ]
        )
        assert sorted(closing_quote_columns(df)) == sorted(
            [
                "total_bet365_line_over",
                "total_bet365_line_under",
                "total_bet365_price_over",
                "total_bet365_price_under",
                "ml_bet365_price_away",
                "ml_bet365_price_home",
            ]
        )

    @pytest.mark.parametrize(
        ("today", "season"),
        [(date(2026, 9, 17), 2026), (date(2026, 3, 1), 2025), (date(2025, 8, 1), 2025)],
    )
    def test_current_season_year(self, today, season):
        assert current_season_year(today) == season


class TestTipoffCorrections:
    def test_moved_game_gets_the_corrected_tip(self):
        schedule = pd.DataFrame(
            {
                "game_id": ["0022500697", "0022500698"],
                "tipoff_utc": [
                    "2026-01-31T20:00:00+00:00",
                    "2026-02-01T00:00:00+00:00",
                ],
                "tipoff_et": ["2026-01-31T15:00:00", "2026-01-31T19:00:00"],
            }
        )

        out = apply_tipoff_corrections(schedule)

        assert out["tipoff_utc"].tolist() == [
            pd.Timestamp("2026-01-31T17:00:00Z"),
            pd.Timestamp("2026-02-01T00:00:00Z"),
        ]
        assert out["tipoff_et"].tolist() == ["2026-01-31T12:00:00", "2026-01-31T19:00:00"]

    def test_postponed_game_is_moved_to_the_date_it_was_played(self):
        # Stored for 2026-04-01, played 2026-03-12: its live ticks were being
        # read as pre-game.
        correction = TIPOFF_CORRECTIONS["0022501111"]
        assert correction.tipoff_utc == pd.Timestamp("2026-03-13T00:00:00Z")
        assert correction.feed_tipoff_utc == pd.Timestamp("2026-04-02T00:00:00Z")

    def test_late_news_game_is_not_corrected(self):
        assert "0022100767" not in TIPOFF_CORRECTIONS

    def test_every_correction_changes_the_tip(self):
        for correction in TIPOFF_CORRECTIONS.values():
            assert correction.tipoff_utc != correction.feed_tipoff_utc

    def test_generated_module_round_trips(self):
        corrections = {
            g: (c.feed_tipoff_utc, c.tipoff_utc) for g, c in TIPOFF_CORRECTIONS.items()
        }
        namespace: dict = {}
        exec(render_corrections_module(corrections), namespace)
        assert {
            g: (pd.Timestamp(f), pd.Timestamp(p))
            for g, (f, p) in namespace["CORRECTED_TIPOFFS"].items()
        } == corrections

    def test_feed_fetch_applies_corrections(self, monkeypatch):
        from nba_ou.fetch_data.nba_schedule import fetch_nba_schedule as feed

        monkeypatch.setattr(
            feed,
            "fetch_season_schedule",
            lambda season_year, timeout=30: pd.DataFrame(
                {
                    "game_id": ["0022500697"],
                    "tipoff_utc": [pd.Timestamp("2026-01-31T20:00:00Z")],
                }
            ),
        )
        out = feed.fetch_schedules([2025])
        assert out["tipoff_utc"].iloc[0] == pd.Timestamp("2026-01-31T17:00:00Z")


class TestTipoffReconciliation:
    def test_mismatches_report_only_games_that_differ(self):
        stored = pd.DataFrame(
            {
                "game_id": ["a", "b", "c", "d"],
                "season_year": [2025, 2025, 2025, 2025],
                "tipoff_utc": pd.to_datetime(
                    [
                        "2026-01-01T00:00Z",
                        "2026-01-02T00:00Z",
                        "2026-01-03T00:00Z",
                        "2026-01-04T00:00Z",
                    ]
                ),
            }
        )
        reference = pd.DataFrame(
            {
                "game_id": ["a", "b", "d"],
                "tipoff_utc": pd.to_datetime(
                    ["2026-01-01T00:00Z", "2026-01-01T23:30Z", "2026-01-04T00:01Z"]
                ),
            }
        )

        out = find_tipoff_mismatches(stored, reference)

        assert out["game_id"].tolist() == ["b"]
        assert out["shift_minutes"].tolist() == [-30.0]

    def test_ingest_retimes_a_stored_game_onto_the_page_tipoff(self, monkeypatch):
        stored = {
            "moved": pd.Timestamp("2026-04-02T00:00Z"),
            "same": pd.Timestamp("2026-03-13T01:00Z"),
        }

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, *args):
                pass

            def fetchall(self):
                return list(stored.items())

        class _Conn:
            def cursor(self):
                return _Cursor()

        calls = {}
        monkeypatch.setattr(
            ingest, "retime_games", lambda conn, tipoffs: calls.update(tipoffs)
        )
        game_dim = pd.DataFrame(
            {
                "game_id": ["moved", "same", "new"],
                "tipoff_utc": pd.to_datetime(
                    ["2026-03-13T00:00Z", "2026-03-13T01:00Z", "2026-03-14T00:00Z"]
                ),
            }
        )

        moved = ingest.retime_to_page_tipoffs(_Conn(), game_dim)

        assert moved == ["moved"]
        assert calls == {"moved": pd.Timestamp("2026-03-13T00:00Z")}


class TestScoreboardCache:
    @staticmethod
    def _payload(game_date: str) -> dict:
        return {
            "scoreboard": {
                "games": [
                    {
                        "gameId": f"g{game_date}",
                        "gameTimeUTC": f"{game_date}T23:00:00Z",
                        "homeTeam": {"teamTricode": "BOS"},
                        "awayTeam": {"teamTricode": "NYK"},
                    }
                ]
            }
        }

    def test_budget_caches_a_batch_and_reports_the_rest(self, tmp_path, monkeypatch):
        from nba_ou.fetch_data.nba_schedule import fetch_nba_schedule as feed

        requested = []
        monkeypatch.setattr(
            feed,
            "_request_scoreboard",
            lambda d, timeout: requested.append(d) or self._payload(d),
        )
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

        remaining = feed.cache_scoreboards(dates, tmp_path, max_requests=2)
        assert remaining == ["2024-01-03"]

        remaining = feed.cache_scoreboards(dates, tmp_path, max_requests=2)
        assert remaining == []
        # Cached dates are never requested twice.
        assert requested == dates

        board = feed.tipoffs_from_cache(dates, tmp_path)
        assert board["game_id"].tolist() == [f"g{d}" for d in dates]

    def test_a_refused_date_stops_the_run_and_is_not_cached(self, tmp_path, monkeypatch):
        from nba_ou.fetch_data.nba_schedule import fetch_nba_schedule as feed

        answers = {"2024-01-01": {"resultSets": []}, "2024-01-02": None}
        requested = []
        monkeypatch.setattr(
            feed,
            "_request_scoreboard",
            lambda d, timeout: requested.append(d) or answers.get(d, self._payload(d)),
        )

        remaining = feed.cache_scoreboards(
            ["2024-01-01", "2024-01-02", "2024-01-03"], tmp_path
        )

        assert requested == ["2024-01-01"]
        assert remaining == ["2024-01-01", "2024-01-02", "2024-01-03"]


class TestSbrTipoffs:
    def test_pages_are_cached_resolved_and_reported(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone

        from nba_ou.fetch_data.odds_sportsbook import scrape_sportsbook_line_history as sbr
        from nba_ou.postgre_db.line_history_aiven import tipoff_corrections as tc

        calls = []

        def discover(session, day, **kwargs):
            calls.append(day)
            if day == date(2026, 3, 13):
                raise RuntimeError("HTTP 503")
            return [
                sbr.GameSummary(
                    event_id=1,
                    game_date=day,
                    tipoff_utc=datetime(2026, 3, 13, 0, 0, tzinfo=timezone.utc),
                    team_away="Dallas Mavericks",
                    team_home="Memphis Grizzlies",
                    status_text="Final",
                )
            ]

        monkeypatch.setattr(sbr, "discover_games_for_date", discover)
        monkeypatch.setattr(ingest, "build_game_index", lambda games: games)
        monkeypatch.setattr(
            ingest,
            "build_game_lookup",
            lambda index: {(date(2026, 3, 12), "MEM", "DAL"): ("0022501111", 2025)},
        )
        monkeypatch.setattr(
            ingest,
            "resolve_game_id",
            lambda lookup, *, game_date, team_away, team_home: lookup.get(
                (game_date, "MEM", "DAL")
            ),
        )
        played = pd.DataFrame(
            {"game_id": ["0022501111", "x"], "game_date": ["2026-03-12", "2026-03-13"]}
        )

        reference, failed = tc.sbr_tipoffs(played, pd.DataFrame(), tmp_path, max_workers=1)

        assert reference["game_id"].tolist() == ["0022501111"]
        assert reference["tipoff_utc"].tolist() == [pd.Timestamp("2026-03-13T00:00Z")]
        assert [f.split(":")[0] for f in failed] == ["2026-03-13"]

        # A rerun only asks for the date that failed.
        calls.clear()
        tc.sbr_tipoffs(played, pd.DataFrame(), tmp_path, max_workers=1)
        assert calls == [date(2026, 3, 13)]
