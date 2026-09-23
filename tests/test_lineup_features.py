"""Phase G: the lineup family as columns, and the switch that gates it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.features import (
    LINEUP_FEATURE_COLUMNS,
    RatingBook,
    add_lineup_features,
    attach_lineup_features,
    walk_forward_offset,
)
from nba_ou.data_processing.lineups.projection_eval import (
    directional_accuracy,
    line_error_slope,
)

H, A, C, D = "1610612739", "1610612738", "1610612737", "1610612736"
ROSTERS = {team: [f"{team[-2:]}p{i}" for i in range(5)] for team in (H, A, C, D)}
STAR = ROSTERS[H][0]
HISTORY_DATES = [f"2025-01-0{day}" for day in range(1, 6)]
TARGET_DATE = "2025-01-10"


def _box() -> pd.DataFrame:
    rows = []
    for index, date in enumerate(HISTORY_DATES):
        for game, teams in ((f"00{index:06d}01", (H, A)), (f"00{index:06d}02", (C, D))):
            for team in teams:
                for player in ROSTERS[team]:
                    rows.append(
                        {
                            "GAME_ID": game,
                            "TEAM_ID": team,
                            "GAME_DATE": date,
                            "PLAYER_ID": player,
                            "MIN": 48.0,
                        }
                    )
    return pd.DataFrame(rows)


def _merged() -> pd.DataFrame:
    rows = []
    for index, date in enumerate(HISTORY_DATES + [TARGET_DATE]):
        for game, (home, away) in (
            (f"00{index:06d}01", (H, A)),
            (f"00{index:06d}02", (C, D)),
        ):
            rows.append(
                {
                    "GAME_ID": game,
                    "GAME_DATE": pd.Timestamp(date),
                    "TEAM_ID_TEAM_HOME": home,
                    "TEAM_ID_TEAM_AWAY": away,
                    "TOTAL_POINTS": np.nan if date == TARGET_DATE else 220.0 + index,
                    "OTHER_FEATURE_BEFORE": float(index),
                }
            )
    return pd.DataFrame(rows)


def _ratings(star_offense: float = 5.0) -> pd.DataFrame:
    rows = []
    for as_of in pd.date_range("2025-01-01", TARGET_DATE):
        for players in ROSTERS.values():
            for player in players:
                rows.append(
                    {
                        "as_of_date": as_of,
                        "player_id": player,
                        "o_rating": star_offense if player == STAR else 0.0,
                        "d_rating": 0.0,
                        "pace_rating": 0.0,
                        "league_ortg": 112.0,
                        "league_pace": 100.0,
                        "fit_max_game_date": as_of - pd.Timedelta(days=1),
                    }
                )
    return pd.DataFrame(rows)


def _statuses(*outs: tuple[str, str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game,
                "team_id": team,
                "player_id": player,
                "status": "out",
                "reason_category": "injury",
            }
            for game, team, player in outs
        ],
        columns=["game_id", "team_id", "player_id", "status", "reason_category"],
    )


TARGET_GAME = f"00{len(HISTORY_DATES):06d}01"


class TestRatingBook:
    def test_reads_the_latest_date_on_or_before_the_game(self):
        book = RatingBook(_ratings().loc[lambda r: r.as_of_date.le("2025-01-04")])
        assert book.as_of(pd.Timestamp("2025-01-06")) == pd.Timestamp("2025-01-04")
        assert book.as_of(pd.Timestamp("2025-01-04")) == pd.Timestamp("2025-01-04")

    def test_nothing_before_the_cache_and_nothing_stale(self):
        book = RatingBook(_ratings(), max_staleness_days=3)
        assert book.as_of(pd.Timestamp("2024-12-31")) is None
        assert book.as_of(pd.Timestamp("2025-01-14")) is None
        assert book.for_date(pd.Timestamp("2025-01-13")) is not None

    def test_rejects_a_fit_that_reaches_its_own_date(self):
        bad = _ratings()
        bad.loc[0, "fit_max_game_date"] = bad.loc[0, "as_of_date"]
        with pytest.raises(ValueError, match="strictly before"):
            RatingBook(bad)


class TestWalkForwardOffset:
    def test_same_day_results_are_held_back(self):
        dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-02"]))
        raw = pd.Series([200.0, 210.0, 210.0])
        actual = pd.Series([204.0, 230.0, 250.0])
        offset = walk_forward_offset(dates, raw, actual)
        assert np.isnan(offset.iloc[0])
        # Both games of the 2nd read only the 1st: +4, not their own +20/+40.
        assert offset.iloc[1] == offset.iloc[2] == pytest.approx(4.0)

    def test_window_and_unknown_results(self):
        dates = pd.Series(pd.date_range("2025-01-01", periods=4))
        raw = pd.Series([200.0, 200.0, 200.0, 200.0])
        actual = pd.Series([210.0, 220.0, np.nan, np.nan])
        offset = walk_forward_offset(dates, raw, actual, window=1)
        # A scheduled game's missing result is skipped, not counted as zero.
        assert offset.iloc[3] == pytest.approx(20.0)


class TestAddLineupFeatures:
    def test_full_health_has_no_absence_impact(self):
        out = add_lineup_features(_merged(), _box(), RatingBook(_ratings()))
        row = out.loc[out.GAME_ID.eq(TARGET_GAME)].iloc[0]
        assert row["LU_ABSENCE_IMPACT_PTS_BEFORE"] == pytest.approx(0.0)
        assert row["LU_PROJ_POSS_BEFORE"] == pytest.approx(100.0)

    def test_an_out_star_lowers_the_projection(self):
        p_out = {(TARGET_GAME, H, STAR): 1.0}
        out = add_lineup_features(_merged(), _box(), RatingBook(_ratings()), p_out)
        row = out.loc[out.GAME_ID.eq(TARGET_GAME)].iloc[0]
        # He plays all 48 minutes (weight 48/48 = 1) at +5 per 100, over 100
        # possessions; his replacements are league average.
        assert row["LU_ABSENCE_IMPACT_PTS_BEFORE"] == pytest.approx(-5.0)
        other = out.loc[out.GAME_ID.eq(f"00{len(HISTORY_DATES):06d}02")].iloc[0]
        assert other["LU_ABSENCE_IMPACT_PTS_BEFORE"] == pytest.approx(0.0)

    def test_games_without_history_or_ratings_are_nan(self):
        out = add_lineup_features(_merged(), _box(), RatingBook(_ratings()))
        first = out.loc[out.GAME_DATE.eq(HISTORY_DATES[0])]
        assert first[list(LINEUP_FEATURE_COLUMNS)].isna().all().all()


class TestSwitch:
    def test_off_returns_the_frame_untouched(self):
        merged = _merged()
        out = attach_lineup_features(
            merged, _box(), enabled=False, injury_statuses=None
        )
        assert out is merged
        assert not [c for c in out.columns if c.startswith("LU_")]

    def test_on_adds_exactly_the_family_and_changes_nothing_else(self):
        merged = _merged()
        before = merged.copy()
        out = attach_lineup_features(
            merged,
            _box(),
            enabled=True,
            injury_statuses=_statuses((TARGET_GAME, H, STAR)),
            ratings=RatingBook(_ratings()),
        )
        assert set(out.columns) - set(before.columns) == set(LINEUP_FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(out[before.columns], before)
        row = out.loc[out.GAME_ID.eq(TARGET_GAME)].iloc[0]
        assert row["LU_ABSENCE_IMPACT_PTS_BEFORE"] < 0

    def test_every_column_passes_the_real_leakage_gate(self):
        """Plan 8.1: ``select_training_columns`` accepts them with no exemption."""
        from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
            select_training_columns,
        )

        out = attach_lineup_features(
            _merged(),
            _box(),
            enabled=True,
            injury_statuses=_statuses(),
            ratings=RatingBook(_ratings()),
        )
        selected = select_training_columns(out, original_columns=[])
        assert set(LINEUP_FEATURE_COLUMNS) <= set(selected.columns)

    def test_on_without_a_report_is_an_error(self):
        with pytest.raises(ValueError, match="injury report"):
            attach_lineup_features(
                _merged(), _box(), enabled=True, injury_statuses=None
            )


class TestProjectionEval:
    def _frame(self, n=400, slope=0.5, seed=1):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 2, n)
        return pd.DataFrame(
            {
                "x": x,
                "LINE_ERROR": slope * x + rng.normal(0, 1, n),
                "GAME_DATE": pd.date_range("2025-01-01", periods=n // 4).repeat(4),
            }
        )

    def test_slope_and_interval_cover_the_truth(self):
        result = line_error_slope(self._frame(), "x", n_boot=300)
        assert result["n"] == 400
        assert result["ci_low"] < 0.5 < result["ci_high"]
        assert result["slope"] == pytest.approx(0.5, abs=0.05)

    def test_accuracy_skips_pushes_and_applies_the_threshold(self):
        frame = pd.DataFrame(
            {"x": [3.0, -3.0, 0.5, 2.0], "LINE_ERROR": [1.0, 2.0, -1.0, 0.0]}
        )
        result = directional_accuracy(frame, "x", 1.0)
        assert result["n"] == 2
        assert result["accuracy"] == pytest.approx(0.5)
