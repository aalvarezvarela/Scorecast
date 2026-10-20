import json
from pathlib import Path

import pandas as pd
import pytest

from training_pipeline.leaderboard import (
    build_leaderboard,
    discover_run_dirs,
    headline_leaderboard,
    load_run_summary,
)


def _write_run(
    root: Path,
    name: str,
    *,
    target_family: str,
    final_mae: float,
    baseline_mae: float,
    final_rmse: float | None = None,
    baseline_rmse: float | None = None,
    include_baseline: bool = True,
    created_at: str | None = "2026-07-30T09:00:00+00:00",
    roi: float | None = None,
    n_bets: int = 100,
    win_rate: float = 0.55,
    win_rate_ci_low: float | None = None,
    is_significant: bool = False,
    bias_baseline_roi: float = -0.02,
    n_candidates: int = 570,
    holdout_start: str | None = "2026-02-01T00:00:00",
    holdout_end: str | None = "2026-03-18T00:00:00",
    config_fingerprint: str = "abc123def456",
    training_version: str | None = "2.1",
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)

    metadata = {
        "experiment_name": name,
        "training_version": training_version,
        "target_family": target_family,
        # Resolved label + timestamp, as pipeline.run_experiment writes them.
        "window_dir_label": "5000_games",
        "window_name_label": "5000_games",
        "config_fingerprint": config_fingerprint,
    }
    if created_at is not None:
        metadata["created_at"] = created_at
    if holdout_start is not None:
        metadata["holdout_start"] = holdout_start
        metadata["holdout_end"] = holdout_end
        metadata["holdout_n_games"] = n_candidates
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "target_family": target_family,
                # Raw user-supplied value: None whenever the label was
                # auto-derived, which is why the leaderboard must prefer
                # metadata.json for the resolved label.
                "window_dir_label": None,
                "optuna": {"n_trials": 80, "objective_name": "reg:squarederror"},
                "walk_forward": {"train_games": 5000},
                "refit": {"train_games": None},
            }
        )
    )
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "cv": {
                    "mae": final_mae + 0.5,
                    "rmse": (final_rmse or final_mae + 1) + 0.5,
                    "ou_acc": 0.53,
                },
                "holdout": {
                    "mae": final_mae,
                    "rmse": final_rmse or final_mae + 1,
                    "r2": 0.1,
                    "ou_acc": 0.55,
                },
            }
        )
    )
    if roi is not None:
        (run_dir / "betting_metrics.json").write_text(
            json.dumps(
                {
                    "primary": {
                        "min_edge": 2.0,
                        "n_candidates": n_candidates,
                        "n_bets": n_bets,
                        "bet_rate": 0.4,
                        "win_rate": win_rate,
                        "win_rate_ci_low": (
                            win_rate - 0.09
                            if win_rate_ci_low is None
                            else win_rate_ci_low
                        ),
                        "win_rate_ci_high": win_rate + 0.09,
                        "n_pushes": 0,
                        "break_even_rate": 110.0 / 210.0,
                        "edge_vs_break_even": win_rate - 110.0 / 210.0,
                        "roi": roi,
                        "profit_units": roi * n_bets,
                        "is_significant": is_significant,
                    },
                    "baseline_bias_corrected": {"mae": 13.4},
                    "baseline_bias_corrected_betting": {"roi": bias_baseline_roi},
                    "dev_line_error_bias": 0.31,
                }
            )
        )
    if include_baseline:
        (run_dir / "baseline_metrics.json").write_text(
            json.dumps(
                {
                    "cv": {
                        "mae": baseline_mae + 0.5,
                        "rmse": (baseline_rmse or baseline_mae + 1) + 0.5,
                        "r2": 0.0,
                        "ou_accuracy": None,
                    },
                    "holdout": {
                        "mae": baseline_mae,
                        "rmse": baseline_rmse or baseline_mae + 1,
                        "r2": 0.0,
                        "ou_accuracy": None,
                    },
                }
            )
        )
    return run_dir


def test_build_leaderboard_computes_improvement_over_baseline(tmp_path):
    _write_run(tmp_path, "run_a", target_family="total_points", final_mae=12.0, baseline_mae=15.0)
    _write_run(tmp_path, "run_b", target_family="total_points", final_mae=13.5, baseline_mae=15.0)

    df = build_leaderboard(root_dir=tmp_path)

    assert len(df) == 2
    row_a = df.loc[df["run_name"] == "run_a"].iloc[0]
    assert row_a["mae_improvement_over_baseline_pct"] == pytest.approx((15.0 - 12.0) / 15.0)
    assert bool(row_a["beats_baseline_mae"]) is True

    # sorted descending by improvement -> run_a (bigger improvement) is first
    assert df.iloc[0]["run_name"] == "run_a"


def test_build_leaderboard_handles_missing_baseline_metrics_gracefully(tmp_path):
    _write_run(
        tmp_path,
        "run_no_baseline",
        target_family="total_points",
        final_mae=12.0,
        baseline_mae=15.0,
        include_baseline=False,
    )

    df = build_leaderboard(root_dir=tmp_path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["baseline_holdout_mae"] is None
    assert pd.isna(row["mae_improvement_over_baseline_pct"])


def test_build_leaderboard_filters_by_target_family(tmp_path):
    _write_run(tmp_path, "run_tp", target_family="total_points", final_mae=12.0, baseline_mae=15.0)
    _write_run(tmp_path, "run_le", target_family="line_error", final_mae=6.0, baseline_mae=7.0)

    df = build_leaderboard(root_dir=tmp_path, target_family="line_error")

    assert len(df) == 1
    assert df.iloc[0]["run_name"] == "run_le"


def test_leaderboard_reports_created_at_and_resolved_window_label(tmp_path):
    """Regression: both columns used to be silently None -- created_at was
    never written by the pipeline, and window_dir_label was read from the raw
    config (None when auto-derived) instead of the resolved metadata value.
    """
    _write_run(tmp_path, "run_a", target_family="total_points", final_mae=12.0, baseline_mae=15.0)

    df = build_leaderboard(root_dir=tmp_path)
    row = df.iloc[0]

    assert row["created_at"] == "2026-07-30T09:00:00+00:00"
    assert row["window_dir_label"] == "5000_games"
    assert row["config_fingerprint"] == "abc123def456"


def test_leaderboard_falls_back_to_run_dir_timestamp_when_created_at_missing(tmp_path):
    """Runs written before created_at was recorded must still sort by date."""
    run_dir = _write_run(
        tmp_path,
        "legacy_run_20260115_143000",
        target_family="total_points",
        final_mae=12.0,
        baseline_mae=15.0,
        created_at=None,
    )
    assert not json.loads((run_dir / "metadata.json").read_text()).get("created_at")

    df = build_leaderboard(root_dir=tmp_path)

    assert df.iloc[0]["created_at"] == "2026-01-15T14:30:00+00:00"


def test_leaderboard_ranks_by_win_rate_confidence_not_roi(tmp_path):
    """ROI is diagnostic; directional evidence controls the default order."""
    _write_run(
        tmp_path, "high_roi_lower_win_rate", target_family="total_points",
        final_mae=12.0, baseline_mae=15.0, roi=0.30,
        win_rate=0.53, win_rate_ci_low=0.49,
    )
    _write_run(
        tmp_path, "low_roi_stronger_win_rate", target_family="total_points",
        final_mae=13.9, baseline_mae=15.0, roi=-0.30,
        win_rate=0.57, win_rate_ci_low=0.53,
    )

    df = build_leaderboard(root_dir=tmp_path)

    assert df.iloc[0]["run_name"] == "low_roi_stronger_win_rate"
    assert (
        df.iloc[0]["mae_improvement_over_baseline_pct"]
        < df.iloc[1]["mae_improvement_over_baseline_pct"]
    )

    # ROI remains available when explicitly requested as a diagnostic.
    by_roi = build_leaderboard(root_dir=tmp_path, sort_by="roi")
    assert by_roi.iloc[0]["run_name"] == "high_roi_lower_win_rate"


def test_leaderboard_surfaces_bet_volume_and_significance(tmp_path):
    _write_run(
        tmp_path, "small_sample", target_family="total_points",
        final_mae=13.0, baseline_mae=15.0, roi=0.10, n_bets=25, is_significant=False,
    )

    row = build_leaderboard(root_dir=tmp_path).iloc[0]

    assert row["n_bets"] == 25
    assert bool(row["is_significant"]) is False
    assert row["break_even_rate"] == pytest.approx(110.0 / 210.0)
    assert row["win_rate_ci_low"] is not None


def test_leaderboard_computes_roi_edge_over_the_bias_corrected_null(tmp_path):
    _write_run(
        tmp_path, "run_a", target_family="total_points", final_mae=13.0,
        baseline_mae=15.0, roi=0.05, bias_baseline_roi=-0.02,
    )

    row = build_leaderboard(root_dir=tmp_path).iloc[0]

    assert row["bias_baseline_roi"] == pytest.approx(-0.02)
    assert row["roi_vs_bias_baseline"] == pytest.approx(0.07)


def test_leaderboard_handles_runs_without_betting_metrics(tmp_path):
    _write_run(
        tmp_path, "legacy", target_family="total_points",
        final_mae=13.0, baseline_mae=15.0, roi=None,
    )

    df = build_leaderboard(root_dir=tmp_path)

    assert len(df) == 1
    assert pd.isna(df.iloc[0]["roi"])


def test_build_leaderboard_rejects_unknown_sort_column(tmp_path):
    _write_run(
        tmp_path, "run_a", target_family="total_points",
        final_mae=13.0, baseline_mae=15.0, roi=0.05,
    )
    with pytest.raises(KeyError):
        build_leaderboard(root_dir=tmp_path, sort_by="not_a_column")


def test_headline_leaderboard_is_a_subset_of_columns(tmp_path):
    _write_run(
        tmp_path, "run_a", target_family="total_points",
        final_mae=13.0, baseline_mae=15.0, roi=0.05,
    )

    full = build_leaderboard(root_dir=tmp_path)
    headline = headline_leaderboard(root_dir=tmp_path)

    # prediction_strategy rather than target_family: it names the model class
    # as well as the target, so "total_points" no longer identifies a run on its
    # own now that a classifier can share a target with a regressor.
    assert list(headline.columns)[:5] == [
        "experiment_id",
        "comparison_group",
        "training_version",
        "prediction_strategy",
        "window_dir_label",
    ]
    assert "roi" not in headline.columns
    assert "cv_roi" not in headline.columns
    assert "win_rate" in headline.columns
    assert "win_rate_ci_low" in headline.columns
    assert "n_bets" in headline.columns
    assert set(headline.columns).issubset(set(full.columns))


def test_legacy_runs_without_prediction_strategy_get_one_inferred(tmp_path):
    """Runs saved before the field existed must still be identifiable."""
    _write_run(
        tmp_path, "old_run", target_family="line_error",
        final_mae=13.0, baseline_mae=13.5,
    )
    row = build_leaderboard(root_dir=tmp_path).iloc[0]
    assert row["prediction_strategy"] == "line_error_regressor"


def test_leaderboard_surfaces_training_version(tmp_path):
    """The manually curated label must be visible so runs can be grouped by
    the training approach that produced them.
    """
    _write_run(
        tmp_path, "run_a", target_family="total_points", final_mae=13.0,
        baseline_mae=15.0, roi=0.05, training_version="2.1-style-features",
    )

    df = build_leaderboard(root_dir=tmp_path)
    assert df.iloc[0]["training_version"] == "2.1-style-features"
    assert "training_version" in headline_leaderboard(root_dir=tmp_path).columns


def test_leaderboard_groups_runs_by_training_version(tmp_path):
    for i, version in enumerate(["2.0", "2.1", "2.1"]):
        _write_run(
            tmp_path, f"run_{i}", target_family="total_points", final_mae=13.0,
            baseline_mae=15.0, roi=0.05 - i * 0.01, training_version=version,
        )

    df = build_leaderboard(root_dir=tmp_path)
    assert sorted(df["training_version"].tolist()) == ["2.0", "2.1", "2.1"]
    assert len(df.groupby("training_version")) == 2


def test_training_version_is_none_for_runs_without_one(tmp_path):
    _write_run(
        tmp_path, "unlabelled", target_family="total_points", final_mae=13.0,
        baseline_mae=15.0, roi=0.05, training_version=None,
    )
    assert build_leaderboard(root_dir=tmp_path).iloc[0]["training_version"] is None


def test_leaderboard_surfaces_the_evaluation_cohort(tmp_path):
    """ROI is only comparable between runs scored on the same holdout window,
    so the window and candidate count must be visible next to the metrics.
    """
    _write_run(
        tmp_path, "run_a", target_family="total_points", final_mae=13.0,
        baseline_mae=15.0, roi=0.05,
    )

    row = build_leaderboard(root_dir=tmp_path).iloc[0]

    assert row["holdout_start"] == "2026-02-01T00:00:00"
    assert row["holdout_end"] == "2026-03-18T00:00:00"
    assert row["n_candidates"] == 570
    assert row["holdout_n_games"] == 570
    assert row["config_fingerprint"] == "abc123def456"


def test_leaderboard_makes_incomparable_cohorts_visible_without_blocking(tmp_path):
    """Runs measured on different holdout windows are still ranked together --
    comparing configurations is the point -- but the difference is legible
    rather than silent.
    """
    _write_run(
        tmp_path, "run_short_window", target_family="total_points", final_mae=13.0,
        baseline_mae=15.0, roi=0.20, n_bets=15, n_candidates=40,
        win_rate=0.70, win_rate_ci_low=0.45,
        holdout_start="2026-03-01T00:00:00", holdout_end="2026-03-18T00:00:00",
        config_fingerprint="fingerprint_one",
    )
    _write_run(
        tmp_path, "run_long_window", target_family="total_points", final_mae=13.4,
        baseline_mae=15.0, roi=0.04, n_bets=300, n_candidates=1200,
        win_rate=0.56, win_rate_ci_low=0.52,
        holdout_start="2025-10-01T00:00:00", holdout_end="2026-03-18T00:00:00",
        config_fingerprint="fingerprint_two",
    )

    df = build_leaderboard(root_dir=tmp_path)

    # Both runs are present and ranked (nothing is filtered out)...
    assert len(df) == 2
    assert df.iloc[0]["run_name"] == "run_long_window"

    # The raw 70% from 15 bets does not outrank the larger cohort's stronger
    # lower confidence bound.
    top, second = df.iloc[0], df.iloc[1]
    assert top["n_candidates"] > second["n_candidates"]
    assert top["holdout_start"] != second["holdout_start"]
    assert top["config_fingerprint"] != second["config_fingerprint"]


def test_headline_leaderboard_includes_cohort_columns_after_directional_metrics(tmp_path):
    _write_run(
        tmp_path, "run_a", target_family="total_points", final_mae=13.0,
        baseline_mae=15.0, roi=0.05,
    )

    columns = list(headline_leaderboard(root_dir=tmp_path).columns)

    for column in ("holdout_start", "holdout_end", "n_candidates", "config_fingerprint"):
        assert column in columns
    # Cohort context must follow the directional block, not be buried at the end.
    assert columns.index("holdout_start") > columns.index("n_bets")
    assert columns.index("holdout_start") < columns.index("final_test_mae")


def test_cohort_columns_degrade_to_none_for_runs_without_them(tmp_path):
    """Runs written before the holdout window was recorded must still load."""
    _write_run(
        tmp_path, "legacy", target_family="total_points", final_mae=13.0,
        baseline_mae=15.0, roi=0.05, holdout_start=None,
    )

    row = build_leaderboard(root_dir=tmp_path).iloc[0]

    assert row["holdout_start"] is None
    assert row["holdout_end"] is None
    assert row["holdout_n_games"] is None


def test_discover_run_dirs_empty_when_root_missing(tmp_path):
    assert discover_run_dirs(tmp_path / "nonexistent") == []


def test_build_leaderboard_empty_when_no_runs(tmp_path):
    df = build_leaderboard(root_dir=tmp_path)
    assert df.empty


def test_the_leaderboard_reports_the_selected_window_not_the_fallback(tmp_path):
    """Third instance of the same defect. config.walk_forward.train_games is the
    fallback for when tuning is skipped; metadata.json records what the selected
    trial chose and every fit site used.

    train_games is a MATCHING factor, so reading the config value meant two
    tuned runs that selected different windows were compared as though they
    were the same experiment whenever their configs shared a fallback -- and the
    run label carried the wrong number as well."""
    import json

    run_dir = tmp_path / "tuned_20260820_120000"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_name": "tuned",
                "target_family": "line_error",
                "train_games": 2500,          # what the selected trial chose
                "train_games_tuned": True,
            }
        )
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "walk_forward": {
                    "train_games": 3500,       # the untouched _base.yaml fallback
                    "train_games_choices": [2500, 3000, 3500],
                },
                "optuna": {},
                "refit": {},
            }
        )
    )

    row = load_run_summary(run_dir)

    assert row["train_games"] == 2500


def test_a_fixed_window_run_still_reads_its_window_from_the_config(tmp_path):
    """Runs saved before the selected value was recorded in metadata must keep
    working -- the config value is the only thing they have."""
    import json

    run_dir = tmp_path / "legacy_20260101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps({"experiment_name": "legacy"}))
    (run_dir / "config.json").write_text(
        json.dumps({"walk_forward": {"train_games": 3750}, "optuna": {}, "refit": {}})
    )

    assert load_run_summary(run_dir)["train_games"] == 3750
