"""Tests for the margin-threshold (coverage) analysis.

The questions worth guarding here are the ones a wrong answer would not raise
on: a cutoff that quietly drops the games it was meant to keep, a Wilson
interval printed over correlated rows as though they were independent, and a
"cutoff frozen from CV" that silently re-derives itself on holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_pipeline.reporting import coverage


def _frame(scores: list[float], wins: list[int], line: float = 220.0) -> pd.DataFrame:
    """A settled prediction frame with a chosen margin and outcome per row.

    ``predicted_edge`` is signed so that the bet is OVER, and the realised
    total is placed above or below the line to make the row a win or a loss.
    That keeps the fixture readable while going through the same
    ``evaluate_betting`` path the notebook uses.
    """
    edge = np.asarray(scores, dtype=float)
    total = np.where(np.asarray(wins) == 1, line + 5.0, line - 5.0)
    return pd.DataFrame({
        "predicted_edge": edge,
        "selection_score": edge,
        "target_line": line,
        "TOTAL_POINTS": total,
        "push": False,
        "won": np.asarray(wins, dtype=int),
    })


class TestCutoffForCoverage:
    def test_keeps_the_requested_share(self) -> None:
        scores = pd.Series(np.arange(100, dtype=float))
        for target in (1.0, 0.9, 0.5, 0.1):
            cutoff = coverage.cutoff_for_coverage(scores, target)
            kept = int((scores > cutoff).sum())
            assert kept >= round(target * 100), (target, kept)

    def test_ties_do_not_collapse_coverage(self) -> None:
        """A discrete score must not lose its whole boundary group.

        Returning the empirical quantile itself would drop every tied row at
        once, because selection is ``score > threshold`` strictly. With 60 rows
        at 1.0 and 40 at 2.0, a naive cutoff of 1.0 at 90% coverage keeps 40 --
        less than half the request, and no error anywhere.
        """
        scores = pd.Series([1.0] * 60 + [2.0] * 40)
        cutoff = coverage.cutoff_for_coverage(scores, 0.90)
        assert int((scores > cutoff).sum()) == 100

    def test_rejects_coverage_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError):
            coverage.cutoff_for_coverage(pd.Series([1.0, 2.0]), 0.0)
        with pytest.raises(ValueError):
            coverage.cutoff_for_coverage(pd.Series([1.0, 2.0]), 1.5)

    def test_all_nan_scores_give_nan_rather_than_raising(self) -> None:
        assert np.isnan(coverage.cutoff_for_coverage(pd.Series([np.nan] * 5), 0.5))


class TestCoverageTable:
    def test_self_mode_cuts_each_source_independently(self) -> None:
        """Two sources with different score scales get different cutoffs."""
        frames = {
            "cross-validation": _frame([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0]),
            "holdout": _frame([10.0, 20.0, 30.0, 40.0], [1, 0, 1, 0]),
        }
        table = coverage.coverage_table(frames, mode="self", coverage_grid=(0.5,))
        cutoffs = table.set_index("source")["cutoff"]
        assert cutoffs["cross-validation"] < cutoffs["holdout"]

    def test_cv_mode_freezes_one_cutoff_for_every_source(self) -> None:
        """The whole point of mode='cv': holdout must not re-derive its own.

        Both sources have to show the SAME cutoff. If they differ, the cutoff
        was chosen with the holdout in view and the win rate beside it is an
        in-sample number wearing an out-of-sample label.
        """
        frames = {
            "cross-validation": _frame([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0]),
            "holdout": _frame([10.0, 20.0, 30.0, 40.0], [1, 0, 1, 0]),
        }
        table = coverage.coverage_table(frames, mode="cv", coverage_grid=(0.5,))
        assert table["cutoff"].nunique() == 1
        # And that frozen cutoff keeps every holdout row, because the holdout
        # scores all sit above the CV distribution.
        holdout = table[table["source"] == "holdout"].iloc[0]
        assert holdout["realised_coverage"] == pytest.approx(1.0)

    def test_cv_mode_requires_a_calibration_frame(self) -> None:
        with pytest.raises(ValueError, match="cutoff"):
            coverage.coverage_table({"holdout": _frame([1.0, 2.0], [1, 0])}, mode="cv")

    def test_rejects_an_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            coverage.coverage_table(
                {"holdout": _frame([1.0, 2.0], [1, 0])}, mode="in-sample"
            )

    def test_correlated_rows_suppress_the_interval(self) -> None:
        """Above the tolerance, no interval and no verdict may be reported."""
        frames = {"holdout": _frame([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 0])}
        table = coverage.coverage_table(frames, rows_per_game=9.9, coverage_grid=(1.0,))
        assert not table["independent"].any()
        assert table["win_rate_ci_low"].isna().all()
        assert table["win_rate_ci_high"].isna().all()
        assert table["is_significant"].isna().all()
        # The point estimate itself is still reported: it is the interval that
        # is unsupported, not the win rate.
        assert table["win_rate"].notna().all()

    def test_one_row_per_game_keeps_the_interval(self) -> None:
        frames = {"holdout": _frame([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 0])}
        table = coverage.coverage_table(frames, rows_per_game=1.0, coverage_grid=(1.0,))
        assert table["independent"].all()
        assert table["win_rate_ci_low"].notna().all()

    def test_win_rate_vs_full_is_measured_against_full_coverage(self) -> None:
        # Wins concentrated in the top half, losses in the bottom.
        frames = {"holdout": _frame([4.0, 3.0, 2.0, 1.0], [1, 1, 0, 0])}
        table = coverage.coverage_table(frames, coverage_grid=(1.0, 0.5))
        by_coverage = table.set_index("target_coverage")
        assert by_coverage.loc[1.0, "win_rate"] == pytest.approx(0.5)
        assert by_coverage.loc[0.5, "win_rate"] == pytest.approx(1.0)
        assert by_coverage.loc[1.0, "win_rate_vs_full"] == pytest.approx(0.0)
        assert by_coverage.loc[0.5, "win_rate_vs_full"] == pytest.approx(0.5)


class TestMarginBuckets:
    def test_buckets_are_disjoint_and_cover_every_row(self) -> None:
        frame = _frame([float(i) for i in range(100)], [i % 2 for i in range(100)])
        table = coverage.margin_bucket_table(frame, n_buckets=10)
        assert len(table) == 10
        assert table["n_rows"].sum() == 100
        assert table["n_rows"].tolist() == [10] * 10

    def test_bucket_one_holds_the_strongest_margins(self) -> None:
        frame = _frame([float(i) for i in range(100)], [1] * 100)
        table = coverage.margin_bucket_table(frame, n_buckets=10).set_index("bucket")
        assert table.loc[1, "score_min"] > table.loc[10, "score_max"]

    def test_tied_scores_still_produce_equal_buckets(self) -> None:
        """rank(method='first') is load-bearing: 'average' would tie whole
        groups into one bucket and leave others empty."""
        frame = _frame([1.0] * 50 + [2.0] * 50, [1] * 100)
        table = coverage.margin_bucket_table(frame, n_buckets=10)
        assert table["n_rows"].tolist() == [10] * 10

    def test_a_planted_ordering_shows_up_as_a_gradient(self) -> None:
        # Top half all wins, bottom half all losses.
        scores = [float(i) for i in range(100)]
        wins = [1 if i >= 50 else 0 for i in range(100)]
        table = coverage.margin_bucket_table(
            _frame(scores, wins), n_buckets=10
        ).set_index("bucket")
        assert table.loc[1, "win_rate"] == pytest.approx(1.0)
        assert table.loc[10, "win_rate"] == pytest.approx(0.0)

    def test_correlated_rows_suppress_the_interval(self) -> None:
        frame = _frame([float(i) for i in range(100)], [i % 2 for i in range(100)])
        table = coverage.margin_bucket_table(frame, n_buckets=5, rows_per_game=9.9)
        assert not table["independent"].any()
        assert table["win_rate_ci_low"].isna().all()

    def test_rejects_fewer_than_two_buckets(self) -> None:
        with pytest.raises(ValueError, match="n_buckets"):
            coverage.margin_bucket_table(_frame([1.0, 2.0], [1, 0]), n_buckets=1)


class TestScoreboard:
    def _table(self) -> pd.DataFrame:
        """Two runs, both at full and half coverage."""
        frames = {
            "cross-validation": _frame([4.0, 3.0, 2.0, 1.0], [1, 1, 1, 0]),
            "holdout": _frame([4.0, 3.0, 2.0, 1.0], [1, 0, 0, 0]),
        }
        weak = coverage.coverage_table(
            frames, label="weak", coverage_grid=(1.0, 0.5)
        )
        frames = {
            "cross-validation": _frame([4.0, 3.0, 2.0, 1.0], [1, 0, 0, 0]),
            "holdout": _frame([4.0, 3.0, 2.0, 1.0], [1, 1, 1, 1]),
        }
        strong = coverage.coverage_table(
            frames, label="strong", coverage_grid=(1.0, 0.5)
        )
        return pd.concat([weak, strong], ignore_index=True)

    def test_reads_the_requested_coverage_slice(self) -> None:
        board = coverage.scoreboard_table(self._table(), coverage_level=1.0)
        assert len(board) == 2
        assert set(board["label"]) == {"weak", "strong"}
        assert board.set_index("label").loc["strong", "holdout_win_rate"] == (
            pytest.approx(1.0)
        )

    def test_a_coverage_level_not_in_the_grid_raises(self) -> None:
        """Silently returning an empty board would render a blank chart."""
        with pytest.raises(ValueError, match="No rows at coverage"):
            coverage.scoreboard_table(self._table(), coverage_level=0.25)

    def test_best_holdout_run_sorts_last(self) -> None:
        """matplotlib's y axis grows upward, so last == top of the chart."""
        board = coverage.scoreboard_table(self._table())
        assert board["label"].iloc[-1] == "strong"

    def test_holdout_minus_cv_is_the_out_of_sample_move(self) -> None:
        board = coverage.scoreboard_table(self._table()).set_index("label")
        assert board.loc["weak", "holdout_minus_cv"] == pytest.approx(
            board.loc["weak", "holdout_win_rate"] - board.loc["weak", "cv_win_rate"]
        )
        # weak drops out of sample, strong improves.
        assert board.loc["weak", "holdout_minus_cv"] < 0
        assert board.loc["strong", "holdout_minus_cv"] > 0

    def test_a_correlated_run_is_marked_and_keeps_no_interval(self) -> None:
        frames = {
            "cross-validation": _frame([4.0, 3.0, 2.0, 1.0], [1, 1, 0, 0]),
            "holdout": _frame([4.0, 3.0, 2.0, 1.0], [1, 0, 1, 0]),
        }
        board = coverage.scoreboard_table(
            coverage.coverage_table(
                frames, label="pooled", rows_per_game=9.9, coverage_grid=(1.0,)
            )
        )
        assert not board["independent"].iloc[0]
        assert pd.isna(board["holdout_win_rate_ci_low"].iloc[0])

    def test_a_run_with_no_holdout_still_appears(self) -> None:
        """A dropped run looks identical to a run that never existed."""
        board = coverage.scoreboard_table(
            coverage.coverage_table(
                {"cross-validation": _frame([2.0, 1.0], [1, 0])},
                label="cv-only", coverage_grid=(1.0,),
            )
        )
        assert board["label"].tolist() == ["cv-only"]
        assert pd.isna(board["holdout_win_rate"].iloc[0])


def _runs(labels: list[str], rows_per_game: float = 1.0) -> pd.DataFrame:
    """The columns the notebook-facing builders read off a prepared runs frame."""
    return pd.DataFrame({
        "label": labels,
        "panel_label": [f"panel:{label}" for label in labels],
        "experiment_name": [f"exp_{label}" for label in labels],
        "prediction_strategy": ["line_error_regressor"] * len(labels),
        "rows_per_game": [rows_per_game] * len(labels),
    })


def _cache(labels: list[str], *, with_cv: set[str] | None = None) -> dict:
    with_cv = labels if with_cv is None else with_cv
    cache = {}
    for index, label in enumerate(labels):
        frames = {
            "holdout": _frame([4.0, 3.0, 2.0, 1.0], [1, 1, index % 2, 0]),
        }
        if label in with_cv:
            # Overlapping but not identical score ranges, as two periods of the
            # same model actually behave: a cutoff learned on CV keeps a
            # DIFFERENT share of holdout, which is the drift the tables report.
            frames["cross-validation"] = _frame([4.0, 3.0, 2.0, 1.0], [1, 0, 1, 0])
            frames["holdout"] = _frame([5.0, 4.0, 2.0, 1.0], [1, 1, index % 2, 0])
        cache[label] = frames
    return cache


class TestOperatingPointTable:
    def test_freezes_one_cutoff_per_run_across_sources(self) -> None:
        """CV and holdout must share a cutoff, or the holdout column is an
        in-sample number wearing an out-of-sample label."""
        table = coverage.operating_point_table(
            _runs(["a"]), _cache(["a"]), coverage_level=0.5
        )
        assert table["cutoff"].nunique() == 1
        assert set(table["source"]) == {"cross-validation", "holdout"}

    def test_names_the_runs_it_had_to_exclude(self) -> None:
        """A run that vanishes from a comparison looks exactly like a run that
        was never there, so the exclusion has to be reported."""
        table = coverage.operating_point_table(
            _runs(["a", "b"]), _cache(["a", "b"], with_cv={"a"}), coverage_level=1.0
        )
        assert table.attrs["missing_calibration"] == ["b"]
        assert set(table["label"]) == {"a"}

    def test_raises_when_no_run_can_be_calibrated(self) -> None:
        with pytest.raises(ValueError, match="operating point"):
            coverage.operating_point_table(
                _runs(["a"]), _cache(["a"], with_cv=set()), coverage_level=1.0
            )

    def test_carries_the_short_panel_label(self) -> None:
        table = coverage.operating_point_table(
            _runs(["a"]), _cache(["a"]), coverage_level=1.0
        )
        assert table["panel"].unique().tolist() == ["panel:a"]


class TestHeadlineTable:
    def test_one_row_per_run_best_holdout_first(self) -> None:
        table = coverage.operating_point_table(
            _runs(["a", "b"]), _cache(["a", "b"]), coverage_level=1.0
        )
        headline = coverage.headline_table(table)
        assert len(headline) == 2
        assert headline["holdout_win_rate"].is_monotonic_decreasing

    def test_keeps_the_cutoff_and_its_units(self) -> None:
        table = coverage.operating_point_table(
            _runs(["a"]), _cache(["a"]), coverage_level=1.0
        )
        headline = coverage.headline_table(table)
        assert headline.loc[0, "cutoff_units"] == "points"
        assert headline.loc[0, "cutoff"] == table["cutoff"].iloc[0]


class TestExecutableTable:
    def test_pairs_cv_with_the_holdout_scored_at_the_same_cutoff(self) -> None:
        views = coverage.build_views(
            _runs(["a"]), _cache(["a"]), coverage_grid=(1.0, 0.5)
        )
        executable = coverage.executable_table(views.cv_coverage)
        assert len(executable) == 2
        assert executable["holdout_win"].notna().all()

    def test_refuses_to_pair_rows_cut_at_different_cutoffs(self) -> None:
        """The guard against being handed a self-mode frame.

        There each source cut itself at its own cutoff, so pairing them on the
        coverage label alone would put two different rules in one row under a
        heading that says the cutoff was frozen from CV. Joining on the cutoff
        as well leaves the holdout columns empty, which is visible.
        """
        views = coverage.build_views(
            _runs(["a"]), _cache(["a"]), coverage_grid=(0.5,)
        )
        self_mode = views.self_coverage
        cutoffs = self_mode.set_index("source")["cutoff"]
        assert cutoffs["cross-validation"] != cutoffs["holdout"]  # fixture check

        executable = coverage.executable_table(self_mode)
        assert executable["holdout_win"].isna().all()

    def test_empty_input_gives_an_empty_frame(self) -> None:
        assert coverage.executable_table(pd.DataFrame()).empty


class TestBuildViews:
    def test_builds_all_four_views_in_one_pass(self) -> None:
        views = coverage.build_views(
            _runs(["a", "b"]), _cache(["a", "b"]),
            coverage_grid=(1.0, 0.5), n_buckets=2,
        )
        assert len(views.self_coverage) == 2 * 2 * 2   # runs x coverages x sources
        assert len(views.cv_coverage) == 2 * 2 * 2
        assert len(views.buckets) == 2 * 2 * 2         # runs x sources x buckets
        assert len(views.trends) == 2 * 2              # runs x sources
        assert set(views.self_coverage["panel"]) == {"panel:a", "panel:b"}

    def test_a_run_without_cv_is_left_out_of_the_cv_view_only(self) -> None:
        views = coverage.build_views(
            _runs(["a", "b"]), _cache(["a", "b"], with_cv={"a"}), coverage_grid=(1.0,)
        )
        assert set(views.self_coverage["label"]) == {"a", "b"}
        assert set(views.cv_coverage["label"]) == {"a"}

    def test_rows_per_game_reaches_the_interval_suppression(self) -> None:
        """Defaulting this to 1.0 inside the builder would hand a pooled run
        the binomial maths it does not qualify for, silently."""
        views = coverage.build_views(
            _runs(["a"], rows_per_game=9.9), _cache(["a"]), coverage_grid=(1.0,)
        )
        assert not views.self_coverage["independent"].any()
        assert views.self_coverage["win_rate_ci_low"].isna().all()


class TestBuildScoreboard:
    def test_attaches_the_experiment_name_and_spec(self) -> None:
        runs = _runs(["a"])
        runs["strategy_short"] = "line_error"
        runs["train_games"] = 3500
        runs["dataset_type"] = "closing_line"
        runs["snapshot_minutes"] = None
        board = coverage.build_scoreboard(runs, _cache(["a"]))
        assert board.loc[0, "experiment_name"] == "exp_a"
        assert "3,500-game window" in board.loc[0, "spec"]
        assert "closing" in board.loc[0, "spec"]

    def test_raises_on_an_empty_cache(self) -> None:
        with pytest.raises(ValueError, match="nothing to score"):
            coverage.build_scoreboard(_runs(["a"]), {})


class TestWinRateLimits:
    def test_uses_the_focused_window_when_the_data_fits(self) -> None:
        assert coverage.win_rate_limits([0.48, 0.55, 0.62]) == coverage.WIN_RATE_YLIM

    def test_widens_rather_than_hiding_a_point_above(self) -> None:
        """Focusing the axis is a reading aid; cropping a point is a lie.

        A hard set_ylim would draw a 78% win rate off the top of the panel with
        nothing to say it was there.
        """
        low, high = coverage.win_rate_limits([0.50, 0.78])
        assert high > 0.78
        assert low == coverage.WIN_RATE_YLIM[0]

    def test_widens_rather_than_hiding_a_point_below(self) -> None:
        low, high = coverage.win_rate_limits([0.28, 0.55])
        assert low < 0.28
        assert high == coverage.WIN_RATE_YLIM[1]

    def test_all_nan_falls_back_to_the_requested_window(self) -> None:
        assert coverage.win_rate_limits([np.nan, np.nan]) == coverage.WIN_RATE_YLIM


class TestMarginTrend:
    def test_detects_a_planted_relationship(self) -> None:
        scores = [float(i) for i in range(200)]
        wins = [1 if i >= 100 else 0 for i in range(200)]
        trend = coverage.margin_trend(_frame(scores, wins))
        assert trend["spearman_rho"] > 0.5
        assert trend["spearman_p"] < 0.01
        assert trend["half_gap"] == pytest.approx(1.0)
        assert trend["half_gap_p"] < 0.01

    def test_reports_no_effect_when_there_is_none(self) -> None:
        rng = np.random.default_rng(0)
        scores = list(rng.random(400))
        wins = list(rng.integers(0, 2, 400))
        trend = coverage.margin_trend(_frame(scores, wins))
        assert abs(trend["spearman_rho"]) < 0.15
        assert trend["spearman_p"] > 0.05

    def test_too_few_rows_returns_nan_rather_than_a_number(self) -> None:
        trend = coverage.margin_trend(_frame([1.0, 2.0, 3.0], [1, 0, 1]))
        assert np.isnan(trend["spearman_rho"])
        assert trend["n_decided"] == 3

    def test_flags_correlated_rows(self) -> None:
        scores = [float(i) for i in range(100)]
        trend = coverage.margin_trend(
            _frame(scores, [i % 2 for i in range(100)]), rows_per_game=9.9
        )
        assert trend["independent"] is False

    def test_pushes_are_excluded_from_the_trend(self) -> None:
        frame = _frame([float(i) for i in range(100)], [1] * 100)
        frame.loc[frame.index[:50], "push"] = True
        trend = coverage.margin_trend(frame)
        assert trend["n_decided"] == 50
