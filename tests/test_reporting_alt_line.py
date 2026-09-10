"""Tests for re-settling closing-line predictions at an earlier snapshot line.

The analysis this supports is a deliberate counterfactual, which is exactly why
its mechanics have to be exact: the leaks are stated in prose, so any further
error is one nobody is looking for. Three things are protected here -- the join
never guesses, the side is re-taken against the new line rather than carried
over, and both settlement lines are scored on the same games.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training_pipeline.reporting import alt_line

LINE_COL = "ODDS_TOTAL_LINE_bet365"


def write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "closing.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_snapshots(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "snapshots.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def predictions(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


class TestTargetLineColumn:
    def test_uses_the_configured_line_when_present(self) -> None:
        assert alt_line.target_line_column({"line_col": "ODDS_X"}) == "ODDS_X"

    def test_falls_back_to_the_main_book_for_line_error(self) -> None:
        """line_error_regressor must omit line_col, and its bets settle into
        the main book's line because that is what the target subtracts."""
        assert alt_line.target_line_column({"line_col": None}).startswith(
            "ODDS_TOTAL_LINE_"
        )


class TestAttachGameIds:
    def test_matches_on_date_total_and_line(self, tmp_path: Path) -> None:
        source = write_source(tmp_path, [
            {"GAME_ID": "0022500001", "GAME_DATE": "2026-03-01",
             "TOTAL_POINTS": 220.0, LINE_COL: 218.0},
            {"GAME_ID": "0022500002", "GAME_DATE": "2026-03-01",
             "TOTAL_POINTS": 231.0, LINE_COL: 225.0},
        ])
        frame = predictions([
            {"date": "2026-03-01", "TOTAL_POINTS": 231.0, "target_line": 225.0,
             "predicted_edge": 1.0},
        ])
        matched, report = alt_line.attach_game_ids(frame, source, line_col=LINE_COL)
        assert matched["game_id"].tolist() == ["0022500002"]
        assert report["n_matched"] == 1
        assert report["n_unmatched"] == 0

    def test_a_key_shared_by_two_games_is_discarded_not_guessed(
        self, tmp_path: Path
    ) -> None:
        """Two games agreeing on date, total AND line identify neither.

        Taking the first match would attach the prediction to an arbitrary one
        of them, and every number downstream would be wrong with nothing
        raised. Both sides must be dropped and counted.
        """
        source = write_source(tmp_path, [
            {"GAME_ID": "0022500001", "GAME_DATE": "2026-03-01",
             "TOTAL_POINTS": 220.0, LINE_COL: 218.0},
            {"GAME_ID": "0022500002", "GAME_DATE": "2026-03-01",
             "TOTAL_POINTS": 220.0, LINE_COL: 218.0},
            {"GAME_ID": "0022500003", "GAME_DATE": "2026-03-02",
             "TOTAL_POINTS": 210.0, LINE_COL: 214.0},
        ])
        frame = predictions([
            {"date": "2026-03-01", "TOTAL_POINTS": 220.0, "target_line": 218.0,
             "predicted_edge": 1.0},
            {"date": "2026-03-02", "TOTAL_POINTS": 210.0, "target_line": 214.0,
             "predicted_edge": -1.0},
        ])
        matched, report = alt_line.attach_game_ids(frame, source, line_col=LINE_COL)
        assert matched["game_id"].tolist() == ["0022500003"]
        assert report["n_unmatched"] == 1
        assert report["n_ambiguous_keys_in_source"] == 2

    def test_never_expands_rows(self, tmp_path: Path) -> None:
        """A join that duplicates predictions would double-count bets."""
        source = write_source(tmp_path, [
            {"GAME_ID": f"002250000{i}", "GAME_DATE": "2026-03-01",
             "TOTAL_POINTS": 220.0 + i, LINE_COL: 218.0}
            for i in range(1, 4)
        ])
        frame = predictions([
            {"date": "2026-03-01", "TOTAL_POINTS": 221.0, "target_line": 218.0,
             "predicted_edge": 1.0},
        ])
        matched, _ = alt_line.attach_game_ids(frame, source, line_col=LINE_COL)
        assert len(matched) == 1

    def test_a_missing_source_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(alt_line.AlternativeLineError, match="not found"):
            alt_line.attach_game_ids(
                predictions([{"date": "2026-03-01", "TOTAL_POINTS": 220.0,
                              "target_line": 218.0, "predicted_edge": 1.0}]),
                tmp_path / "missing.csv", line_col=LINE_COL,
            )


class TestSnapshotLineLookup:
    def test_reads_only_the_requested_horizon(self, tmp_path: Path) -> None:
        path = write_snapshots(tmp_path, [
            {"GAME_ID": "A", "TIME_TO_MATCH_MIN": 360, LINE_COL: 218.0},
            {"GAME_ID": "A", "TIME_TO_MATCH_MIN": 0, LINE_COL: 220.0},
            {"GAME_ID": "B", "TIME_TO_MATCH_MIN": 360, LINE_COL: 225.5},
        ])
        lookup = alt_line.snapshot_line_lookup(
            path, snapshot_minutes=360, line_col=LINE_COL
        )
        assert lookup.to_dict() == {"A": 218.0, "B": 225.5}

    def test_an_absent_horizon_raises_rather_than_returning_empty(
        self, tmp_path: Path
    ) -> None:
        """An empty lookup would silently drop every game downstream."""
        path = write_snapshots(tmp_path, [
            {"GAME_ID": "A", "TIME_TO_MATCH_MIN": 0, LINE_COL: 220.0},
        ])
        with pytest.raises(alt_line.AlternativeLineError, match="Available"):
            alt_line.snapshot_line_lookup(
                path, snapshot_minutes=360, line_col=LINE_COL
            )

    def test_two_rows_for_one_game_at_a_horizon_raises(self, tmp_path: Path) -> None:
        path = write_snapshots(tmp_path, [
            {"GAME_ID": "A", "TIME_TO_MATCH_MIN": 360, LINE_COL: 218.0},
            {"GAME_ID": "A", "TIME_TO_MATCH_MIN": 360, LINE_COL: 219.0},
        ])
        with pytest.raises(alt_line.AlternativeLineError, match="ambiguous"):
            alt_line.snapshot_line_lookup(
                path, snapshot_minutes=360, line_col=LINE_COL
            )


def settled(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    margin = frame["TOTAL_POINTS"] - frame["target_line"]
    frame["push"] = margin == 0
    frame["won"] = (
        np.where(frame["predicted_edge"] > 0, margin > 0, margin < 0) & ~frame["push"]
    ).astype(int)
    frame["selection_score"] = frame["predicted_edge"].abs()
    return frame


class TestSwapSettlementLine:
    def test_recovers_the_implied_total_and_re_prices_it(self) -> None:
        """edge + line is the prediction in TOTAL_POINTS space for BOTH
        regressors, which is what makes one expression enough."""
        frame = settled([
            {"game_id": "A", "target_line": 218.0, "TOTAL_POINTS": 225.0,
             "predicted_edge": 3.0},
        ])
        swapped = alt_line.swap_settlement_line(frame, pd.Series({"A": 220.0}))
        assert swapped.loc[0, "implied_total"] == pytest.approx(221.0)
        assert swapped.loc[0, "predicted_edge"] == pytest.approx(1.0)
        assert swapped.loc[0, "line_move"] == pytest.approx(2.0)

    def test_the_side_is_re_taken_against_the_new_line(self) -> None:
        """The mechanism of the whole section.

        Implied total 219 is an OVER against a 218 close and an UNDER against a
        222 line. Carrying the old side over instead of re-deriving it would
        make the swap almost a no-op, and nothing would raise.
        """
        frame = settled([
            {"game_id": "A", "target_line": 218.0, "TOTAL_POINTS": 225.0,
             "predicted_edge": 1.0},
        ])
        swapped = alt_line.swap_settlement_line(frame, pd.Series({"A": 222.0}))
        assert swapped.loc[0, "predicted_edge"] < 0        # now an UNDER
        assert swapped.loc[0, "won"] == 0                  # 225 went over 222

    def test_selection_score_follows_the_new_edge(self) -> None:
        frame = settled([
            {"game_id": "A", "target_line": 218.0, "TOTAL_POINTS": 225.0,
             "predicted_edge": 6.0},
        ])
        swapped = alt_line.swap_settlement_line(frame, pd.Series({"A": 223.0}))
        assert swapped.loc[0, "selection_score"] == pytest.approx(1.0)

    def test_games_without_an_alternative_line_are_dropped(self) -> None:
        frame = settled([
            {"game_id": "A", "target_line": 218.0, "TOTAL_POINTS": 225.0,
             "predicted_edge": 1.0},
            {"game_id": "B", "target_line": 210.0, "TOTAL_POINTS": 208.0,
             "predicted_edge": -1.0},
        ])
        swapped = alt_line.swap_settlement_line(frame, pd.Series({"A": 220.0}))
        assert swapped["game_id"].tolist() == ["A"]

    def test_requires_game_ids(self) -> None:
        frame = settled([
            {"target_line": 218.0, "TOTAL_POINTS": 225.0, "predicted_edge": 1.0},
        ])
        with pytest.raises(alt_line.AlternativeLineError, match="game_id"):
            alt_line.swap_settlement_line(frame, pd.Series({"A": 220.0}))


class TestCompareSettlementLines:
    def test_both_series_are_scored_on_the_same_games(self) -> None:
        """The cohort control.

        Game B has no T-360 line, so it must leave the closing side too --
        otherwise the comparison is 2 games against 1 and the difference is
        which games were counted, not which line they settled into.
        """
        closing = settled([
            {"game_id": "A", "target_line": 218.0, "TOTAL_POINTS": 225.0,
             "predicted_edge": 3.0},
            {"game_id": "B", "target_line": 210.0, "TOTAL_POINTS": 190.0,
             "predicted_edge": -8.0},
        ])
        swapped = alt_line.swap_settlement_line(closing, pd.Series({"A": 220.0}))
        comparison = alt_line.compare_settlement_lines(
            closing, swapped, coverage_grid=(1.0,)
        )
        assert set(comparison["n_bets"]) == {1}

    def test_reports_one_row_per_line_and_coverage(self) -> None:
        closing = settled([
            {"game_id": chr(65 + i), "target_line": 218.0,
             "TOTAL_POINTS": 220.0 + i, "predicted_edge": float(i - 5)}
            for i in range(12)
        ])
        lines = pd.Series({chr(65 + i): 219.0 for i in range(12)})
        comparison = alt_line.compare_settlement_lines(
            closing, alt_line.swap_settlement_line(closing, lines),
            alt_name="T-360", coverage_grid=(1.0, 0.5),
        )
        assert len(comparison) == 4
        assert set(comparison["settled_at"]) == {"closing line", "T-360 line"}

    def test_selection_overlap_is_reported_not_assumed(self) -> None:
        """Same cohort is not the same picks below full coverage.

        The cohort control above holds, but each series then ranks that cohort
        by ITS OWN margin, and swapping the settlement line re-ranks it. At 100%
        that is vacuous and the selections coincide; below it they diverge, and
        part of any win-rate gap is a different subset rather than a different
        price. The chart used to assert the opposite, so the overlap is carried
        in the frame where a reader can see it.
        """
        # Twelve games whose closing edges rank them one way and whose T-360
        # edges rank them close to the reverse: the alt line is placed just
        # past each implied total, so a large closing edge becomes a small one.
        closing = settled([
            {"game_id": chr(65 + i), "target_line": 218.0,
             "TOTAL_POINTS": 220.0, "predicted_edge": float(i + 1)}
            for i in range(12)
        ])
        lines = pd.Series({chr(65 + i): 218.0 + i + 1 - (12 - i) * 0.1
                           for i in range(12)})
        comparison = alt_line.compare_settlement_lines(
            closing, alt_line.swap_settlement_line(closing, lines),
            coverage_grid=(1.0, 0.5),
        )
        overlap = comparison.groupby("target_coverage")[
            "share_selection_shared"
        ].first()
        assert overlap[1.0] == 1.0
        assert overlap[0.5] < 1.0
        # Both sides still keep the same NUMBER of games; only which ones
        # differ, which is precisely what makes the divergence invisible
        # without this column.
        half = comparison[np.isclose(comparison["target_coverage"], 0.5)]
        assert half["n_bets"].nunique() == 1


class TestClosingLineRuns:
    def _runs(self, tmp_path: Path) -> pd.DataFrame:
        rows = [
            ("closing_reg", "closing_line", "line_error_regressor", False),
            ("intermediate_reg", "intermediate_line", "line_error_regressor", False),
            ("closing_clf", "closing_line", "over_under_classifier", True),
        ]
        frame = []
        for name, dataset, strategy, is_classifier in rows:
            (tmp_path / name).mkdir()
            (tmp_path / name / "config.json").write_text(
                f'{{"data": {{"dataset_type": "{dataset}"}}, '
                f'"prediction_strategy": "{strategy}"}}'
            )
            frame.append({
                "label": name, "run_dir": str(tmp_path / name),
                "is_classifier": is_classifier,
            })
        return pd.DataFrame(frame)

    def test_keeps_only_closing_line_regressors(self, tmp_path: Path) -> None:
        """An intermediate run is already scored at its own horizon, so
        re-settling it answers nothing; a classifier's score cannot be rebuilt
        from the artifacts at all."""
        selected = alt_line.closing_line_runs(self._runs(tmp_path))
        assert [run["label"] for run, _ in selected] == ["closing_reg"]

    def test_returns_the_config_beside_each_run(self, tmp_path: Path) -> None:
        selected = alt_line.closing_line_runs(self._runs(tmp_path))
        _, config = selected[0]
        assert config["data.dataset_type"] == "closing_line"


class TestSettlementGap:
    def _comparisons(self) -> pd.DataFrame:
        return pd.DataFrame({
            "label": ["a"] * 4,
            "settled_at": ["closing line", "T-360 line"] * 2,
            "target_coverage": [1.0, 1.0, 0.5, 0.5],
            "win_rate": [0.50, 0.55, 0.52, 0.51],
            "roi": [-0.04, 0.05, 0.00, -0.02],
            "n_bets": [100, 100, 50, 50],
        })

    def test_gain_is_the_alternative_line_minus_the_close(self) -> None:
        gap = alt_line.settlement_gap(self._comparisons(), alt_name="T-360")
        full = gap[gap[("target_coverage", "")] == 1.0].iloc[0]
        assert full[("win_rate", "gain")] == pytest.approx(0.05)
        assert full[("roi", "gain")] == pytest.approx(0.09)

    def test_a_worse_alternative_line_gives_a_negative_gain(self) -> None:
        gap = alt_line.settlement_gap(self._comparisons(), alt_name="T-360")
        half = gap[gap[("target_coverage", "")] == 0.5].iloc[0]
        assert half[("win_rate", "gain")] < 0

    def test_empty_input_gives_an_empty_frame(self) -> None:
        assert alt_line.settlement_gap(pd.DataFrame(), alt_name="T-360").empty


class TestSideFlipSummary:
    def test_counts_flips_and_movement(self) -> None:
        frame = settled([
            # implied 219: OVER at 218, UNDER at 222 -> flips
            {"game_id": "A", "target_line": 218.0, "TOTAL_POINTS": 225.0,
             "predicted_edge": 1.0},
            # implied 228: OVER at both -> no flip, line still moved
            {"game_id": "B", "target_line": 220.0, "TOTAL_POINTS": 230.0,
             "predicted_edge": 8.0},
        ])
        swapped = alt_line.swap_settlement_line(
            frame, pd.Series({"A": 222.0, "B": 221.0})
        )
        summary = alt_line.side_flip_summary(swapped)
        assert summary["n_games"] == 2
        assert summary["n_side_flips"] == 1
        assert summary["share_side_flipped"] == pytest.approx(0.5)
        assert summary["share_line_moved"] == pytest.approx(1.0)
        assert summary["mean_abs_line_move"] == pytest.approx(2.5)

    def test_an_unmoved_line_flips_nothing(self) -> None:
        frame = settled([
            {"game_id": "A", "target_line": 218.0, "TOTAL_POINTS": 225.0,
             "predicted_edge": 1.0},
        ])
        swapped = alt_line.swap_settlement_line(frame, pd.Series({"A": 218.0}))
        summary = alt_line.side_flip_summary(swapped)
        assert summary["n_side_flips"] == 0
        assert summary["share_line_moved"] == pytest.approx(0.0)


class TestAvailableHorizons:
    def test_a_horizon_with_no_line_is_not_available(self) -> None:
        """Its rows exist, but re-settling against it would drop every game."""
        frame = pd.DataFrame({
            "GAME_ID": ["A", "B", "C"],
            "TIME_TO_MATCH_MIN": [60, 360, 720],
            LINE_COL: [218.0, 219.0, np.nan],
        })
        assert alt_line.available_horizons(frame, line_col=LINE_COL) == [60, 360]


class TestResolveHorizons:
    AVAILABLE = [0, 30, 60, 120, 180, 240, 300, 360, 480, 720]

    def test_snaps_each_request_to_the_closest_horizon(self) -> None:
        resolved = alt_line.resolve_horizons([65, 250], self.AVAILABLE)
        assert resolved == {65: 60, 250: 240}

    def test_a_request_with_nothing_in_range_is_dropped_not_raised(self) -> None:
        """A horizon nobody collected is a gap in the data, not a caller error:
        the remaining horizons still answer the question."""
        resolved = alt_line.resolve_horizons([60, 1440], self.AVAILABLE)
        assert resolved == {60: 60}

    def test_two_requests_never_collapse_onto_one_horizon(self) -> None:
        resolved = alt_line.resolve_horizons([60, 70], [60, 480])
        assert list(resolved.values()) == [60]

    def test_nothing_available_resolves_nothing(self) -> None:
        assert alt_line.resolve_horizons([60], []) == {}


class TestHorizonLabel:
    def test_reads_as_hours_with_the_close_at_zero(self) -> None:
        assert alt_line.horizon_label(0) == "close"
        assert alt_line.horizon_label(60) == "1h"
        assert alt_line.horizon_label(720) == "12h"
        assert alt_line.horizon_label(90) == "90m"


class TestHorizonGapTable:
    def _comparisons(self) -> pd.DataFrame:
        rows = []
        for horizon, win_rate in ((60, 0.51), (720, 0.57)):
            rows += [
                {"label": "a", "horizon_minutes": horizon,
                 "settled_at": "closing line", "target_coverage": 1.0,
                 "win_rate": 0.53, "roi": 0.01, "n_bets": 100},
                {"label": "a", "horizon_minutes": horizon,
                 "settled_at": f"T-{horizon} line", "target_coverage": 1.0,
                 "win_rate": win_rate, "roi": 0.04, "n_bets": 100},
            ]
        return pd.DataFrame(rows)

    def test_one_row_per_horizon_with_the_gain_against_the_close(self) -> None:
        table = alt_line.horizon_gap_table(self._comparisons())
        assert list(table["priced at"]) == ["1h", "12h"]
        assert list(table["win_rate_gain"]) == pytest.approx([-0.02, 0.04])

    def test_a_frame_without_horizons_gives_an_empty_table(self) -> None:
        assert alt_line.horizon_gap_table(pd.DataFrame()).empty


class TestPlotSettlementHorizons:
    def _comparisons(self) -> pd.DataFrame:
        rows = []
        for label in ("a", "b"):
            for horizon, win_rate in ((60, 0.515), (240, 0.545), (720, 0.562)):
                rows += [
                    {"label": label, "horizon_minutes": horizon,
                     "settled_at": "closing line", "target_coverage": 1.0,
                     "win_rate": 0.523, "roi": 0.0, "n_bets": 600,
                     "win_rate_ci_low": 0.45, "win_rate_ci_high": 0.60},
                    {"label": label, "horizon_minutes": horizon,
                     "settled_at": f"T-{horizon} line", "target_coverage": 1.0,
                     "win_rate": win_rate, "roi": 0.03, "n_bets": 600,
                     "win_rate_ci_low": win_rate - 0.04,
                     "win_rate_ci_high": win_rate + 0.04},
                ]
        return pd.DataFrame(rows)

    def _figure(self, **kwargs: object) -> tuple[object, object]:
        import matplotlib

        matplotlib.use("Agg")
        return alt_line.plot_settlement_horizons(self._comparisons(), **kwargs)

    def test_one_panel_per_run(self) -> None:
        _, axes = self._figure()
        assert sum(ax.get_visible() for ax in axes.flat) == 2

    def test_the_y_axis_fits_the_win_rates_and_keeps_break_even_in_view(self) -> None:
        """A wedge three points tall is a hairline on a 35-70% axis, so this
        chart fits the data -- but never at the price of hiding break-even."""
        _, axes = self._figure()
        low, high = axes.flat[0].get_ylim()
        assert low < alt_line.BREAK_EVEN < high
        # Fitted, not the notebook's standard window, and the confidence
        # verticals are allowed to run off the top and bottom.
        assert high - low < 0.10
        assert low > 0.45

    def test_an_explicit_window_is_honoured(self) -> None:
        _, axes = self._figure(ylim=(0.35, 0.70))
        low, high = axes.flat[0].get_ylim()
        assert (low, high) == pytest.approx((0.35, 0.70))

    def test_a_frame_without_horizons_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="horizon_minutes"):
            alt_line.plot_settlement_horizons(pd.DataFrame({"label": ["a"]}))
