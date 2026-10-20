"""Tests for the controlled-comparison helpers in reporting.factors.

The thing being protected here is a claim, not a number: "these two runs differ
only in X". Every test below is written against a way that claim has been, or
could be, quietly false -- a factor read from a run's name instead of its
config, a pair matched across different holdout periods, a replicate counted as
an effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from training_pipeline.reporting import factors

REFERENCE_WINDOW = ("2026-02-19T00:00:00", "2026-04-17T00:00:00")


def write_run(
    root: Path,
    name: str,
    *,
    strategy: str = "line_error_regressor",
    train_games: int = 3750,
    csv_path: str = "data/train_data/training_data_2_0_20260704.csv",
    exclude_overtime: bool = False,
    exclude_playoffs: bool = True,
    max_na: int = 80,
    exclude_cols: tuple[str, ...] = ("fanatics_sportsbook",),
    sample_weighting: bool = False,
    max_folds: int = 12,
    n_trials: int = 50,
    window: tuple[str, str] = REFERENCE_WINDOW,
    n_games: int = 416,
    **metrics: float,
) -> Path:
    """A minimal run directory: the config fields factors reads, plus metrics."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "prediction_strategy": strategy,
        "data": {
            "csv_path": csv_path,
            "exclude_overtime_from_training": exclude_overtime,
            "exclude_playoffs": exclude_playoffs,
            "season_year_floor": 2021,
        },
        "cleaning": {
            "max_na_per_row": max_na,
            "nan_threshold": 50.0,
            "exclude_cols_containing": list(exclude_cols),
        },
        "walk_forward": {
            "train_games": train_games,
            "max_folds": max_folds,
            "strategy": "test_anchored",
        },
        "sample_weight": {"enabled": sample_weighting},
        "optuna": {"n_trials": n_trials},
    }))
    return run_dir


def runs_frame(root: Path, specs: dict[str, dict]) -> pd.DataFrame:
    """Build the prepared-runs frame the reporting layer passes around."""
    rows = []
    for name, spec in specs.items():
        metrics = {
            key: spec.pop(key)
            for key in list(spec)
            if key in {"roi", "win_rate", "cv_win_rate", "cv_roi", "seed_roi_range"}
        }
        window = spec.get("window", REFERENCE_WINDOW)
        n_games = spec.get("n_games", 416)
        run_dir = write_run(root, name, **spec)
        rows.append({
            "run_name": name,
            "run_dir": str(run_dir),
            "source_path": str(root),
            "prediction_strategy": spec.get("strategy", "line_error_regressor"),
            "strategy_short": spec.get("strategy", "line_error_regressor")
            .replace("_regressor", "").replace("over_under_classifier", "classifier"),
            "train_games": spec.get("train_games", 3750),
            "holdout_start": window[0],
            "holdout_end": window[1],
            "holdout_n_games": n_games,
            "label": name,
            "created_at": pd.Timestamp("2026-08-03"),
            **{"roi": 0.0, "win_rate": 0.53, "cv_win_rate": 0.53,
               "cv_roi": 0.0, "seed_roi_range": 0.05, **metrics},
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- design matrix
def test_factors_are_read_from_config_not_from_the_run_name(tmp_path):
    """A run named "..._no_ot" that did not exclude overtime must not be
    labelled as though it did. Names are written by hand; configs are not."""
    runs = runs_frame(tmp_path, {
        "line_error_3750_no_ot": {"exclude_overtime": False},
    })
    assert not factors.design_matrix(runs)["exclude_overtime"].iat[0]


def test_old_data_build_detected_from_csv_filename(tmp_path):
    runs = runs_frame(tmp_path, {
        "new": {},
        "old": {"csv_path": "data/train_data/old_training_data_until_20260704.csv"},
    })
    build = factors.design_matrix(runs).set_index("run_name")["data_build"]
    assert build["new"] == "2.0"
    assert build["old"] == "old"


def test_missing_overtime_field_means_the_filter_did_not_exist(tmp_path):
    """Older configs predate the flag. Absent must read as False, not NaN, or
    those runs group separately and silently match nothing."""
    run_dir = write_run(tmp_path, "ancient")
    config = json.loads((run_dir / "config.json").read_text())
    del config["data"]["exclude_overtime_from_training"]
    (run_dir / "config.json").write_text(json.dumps(config))

    runs = pd.DataFrame([{
        "run_name": "ancient", "run_dir": str(run_dir), "source_path": str(tmp_path),
        "prediction_strategy": "line_error_regressor", "strategy_short": "line_error",
        "train_games": 3750, "holdout_start": REFERENCE_WINDOW[0],
        "holdout_end": REFERENCE_WINDOW[1], "holdout_n_games": 416,
        "label": "ancient", "created_at": pd.Timestamp("2026-08-01"),
    }])
    assert not factors.design_matrix(runs)["exclude_overtime"].iat[0]


# ---------------------------------------------------------------- cohort flags
def test_same_window_with_fewer_scored_games_stays_comparable(tmp_path):
    """The classifier prices 409 of the same 416 games. That is seven games it
    could not bet, not a different two months, and excluding it would delete
    the strategy comparison the set exists to make."""
    runs = runs_frame(tmp_path, {
        "regressor": {"n_games": 416},
        "classifier": {"strategy": "over_under_classifier", "n_games": 409},
    })
    flagged = factors.flag_cohorts(runs).set_index("run_name")
    assert flagged.loc["classifier", "cohort_ok"]


def test_different_window_is_flagged_however_close_the_game_count(tmp_path):
    runs = runs_frame(tmp_path, {
        "reference": {},
        "reference_two": {},
        "playoffs": {
            "exclude_playoffs": False, "n_games": 89,
            "window": ("2026-04-15T00:00:00", "2026-06-13T00:00:00"),
        },
    })
    flagged = factors.flag_cohorts(runs).set_index("run_name")
    assert not flagged.loc["playoffs", "cohort_ok"]
    assert "2026-06-13" in flagged.loc["playoffs", "cohort_note"]


# ------------------------------------------------------------------- contrasts
def test_contrast_isolates_a_single_factor(tmp_path):
    runs = runs_frame(tmp_path, {
        "base": {"train_games": 3750},
        "no_ot": {"train_games": 3750, "exclude_overtime": True},
        "wide": {"train_games": 4500},
    })
    table = factors.contrasts(runs, "exclude_overtime")
    # "wide" differs in the window as well, so it belongs to no overtime pair.
    assert set(table["run_name"]) == {"base", "no_ot"}
    assert table["contrast"].nunique() == 1


def test_referee_exclusion_cells_are_distinct_experiments(tmp_path):
    """The referee campaign varies only exclude_cols_containing. Without a
    factor for it every cell matched as a replicate of every other."""
    runs = runs_frame(tmp_path, {
        "no_refs": {"exclude_cols": ("REF_AVG_", "REF_CREW_")},
        "legacy": {"exclude_cols": ("REF_CREW_",)},
        "everything": {"exclude_cols": ()},
    })
    design = factors.design_matrix(runs).set_index("run_name")
    assert design["design"].nunique() == 3
    assert design.loc["everything", "referee_features"] == "all"
    assert factors.contrasts(runs, "referee_features")["run_name"].nunique() == 3


def test_referee_factor_leaves_consensus_contrasts_intact(tmp_path):
    runs = runs_frame(tmp_path, {
        "base": {"exclude_cols": ("fanatics_sportsbook",)},
        "no_consensus": {"exclude_cols": ("fanatics_sportsbook", "consensus_pct")},
    })
    assert set(factors.contrasts(runs, "drop_consensus")["run_name"]) == {"base", "no_consensus"}


def test_run_changing_two_things_at_once_forms_no_contrast(tmp_path):
    """The whole guarantee. A cell that moved two knobs cannot answer for
    either of them, and must not be quietly attributed to one."""
    runs = runs_frame(tmp_path, {
        "base": {},
        "both": {"exclude_overtime": True, "max_na": 200},
    })
    assert factors.contrasts(runs, "exclude_overtime").empty
    assert factors.contrasts(runs, "max_na_per_row").empty


def test_pair_scored_on_different_windows_is_not_a_contrast(tmp_path):
    runs = runs_frame(tmp_path, {
        "base": {},
        "base_two": {},
        "other_period": {
            "exclude_overtime": True, "n_games": 89,
            "window": ("2026-04-15T00:00:00", "2026-06-13T00:00:00"),
        },
    })
    assert factors.contrasts(runs, "exclude_overtime").empty
    assert not factors.contrasts(
        runs, "exclude_overtime", require_cohort=False
    ).empty


def test_replicates_collapse_so_they_are_not_read_as_an_effect(tmp_path):
    """Two runs of one configuration must not appear as two levels; otherwise
    effect() reports a change from a level to itself."""
    runs = runs_frame(tmp_path, {
        "base": {"train_games": 3750},
        "base_rerun": {"train_games": 3750},
        "wide": {"train_games": 4500},
    })
    table = factors.contrasts(runs, "train_games")
    assert list(table["train_games"]) == [3750, 4500]

    effects = factors.effect(table, "train_games")
    assert len(effects) == 1
    assert (effects["from"].iat[0], effects["to"].iat[0]) == (3750, 4500)


def test_contrast_label_names_only_what_separates_the_contrasts(tmp_path):
    """Held-constant factors identical across every contrast are noise in the
    label; the one that differs is the whole point of it."""
    runs = runs_frame(tmp_path, {
        "le_base": {"strategy": "line_error_regressor"},
        "le_no_ot": {"strategy": "line_error_regressor", "exclude_overtime": True},
        "tp_base": {"strategy": "total_points_regressor"},
        "tp_no_ot": {"strategy": "total_points_regressor", "exclude_overtime": True},
    })
    labels = set(factors.contrasts(runs, "exclude_overtime")["contrast"])
    assert labels == {
        "strategy=line_error_regressor", "strategy=total_points_regressor"
    }


def test_unknown_factor_raises_rather_than_returning_empty(tmp_path):
    """An empty frame reads as "no matched runs", which is a real answer and
    the wrong one for a typo."""
    runs = runs_frame(tmp_path, {"base": {}})
    with pytest.raises(KeyError, match="max_dept"):
        factors.contrasts(runs, "max_dept")


# ---------------------------------------------------------------------- effect
def test_effect_compares_against_the_noise_floor(tmp_path):
    runs = runs_frame(tmp_path, {
        "base": {"train_games": 3750, "roi": 0.02, "seed_roi_range": 0.05},
        "wide": {"train_games": 4500, "roi": 0.06, "seed_roi_range": 0.03},
    })
    effects = factors.effect(factors.contrasts(runs, "train_games"), "train_games")
    assert effects["d_roi"].iat[0] == pytest.approx(0.04)
    # 4 points of ROI against a 5-point seed range is not an effect.
    assert effects["seed_roi_range"].iat[0] == pytest.approx(0.05)
    assert not effects["beats_seed_noise"].iat[0]


# ------------------------------------------------------------------ replicates
def test_replicates_finds_reruns_of_one_configuration(tmp_path):
    runs = runs_frame(tmp_path, {
        "base": {}, "base_rerun": {}, "different": {"train_games": 4500},
    })
    found = factors.replicates(runs)
    assert set(found["run_name"]) == {"base", "base_rerun"}


# ---------------------------------------------------------------------- labels
def test_labels_describe_deviations_instead_of_hashes(tmp_path):
    runs = runs_frame(tmp_path, {
        "a": {}, "b": {"train_games": 3000},
        "c": {"exclude_overtime": True, "train_games": 4500},
    })
    labelled = factors.describe_labels(runs).set_index("run_name")
    assert labelled.loc["a", "label"] == "line_error · 3750"
    assert labelled.loc["c", "label"] == "line_error · 4500 · no-OT"


def test_labels_stay_unique_for_identical_configurations(tmp_path):
    """Replicates deviate identically, so labelling alone cannot separate them
    -- and two runs collapsing into one row in a groupby is a silent loss."""
    runs = runs_frame(tmp_path, {"a": {}, "b": {}})
    labels = factors.describe_labels(runs)["label"]
    assert labels.nunique() == 2


# ---------------------------------------------------------------------------
# reading runs that have no seed data at all
# ---------------------------------------------------------------------------


def _runs_frame(with_seeds: bool):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "label": ["a", "b"],
            "roi": [0.02, -0.03],
            "prediction_strategy": ["line_error_regressor"] * 2,
            "holdout_start": ["2026-02-19"] * 2,
            "holdout_end": ["2026-04-17"] * 2,
        }
    )
    if with_seeds:
        frame["seed_roi_range"] = [0.05, 0.06]
    return frame


def test_seed_narrative_survives_a_frame_with_no_seed_column():
    """Single seed is the default, so seed_roi_range is usually absent from the
    frame ENTIRELY -- not merely NaN. dropna(subset=[...]) raises KeyError on a
    missing column, which turns "these runs have no error bars" into "the whole
    report failed to render"."""
    from training_pipeline.reporting import narrative

    assert narrative.seed_noise(_runs_frame(with_seeds=False)) == []


def test_strategy_spread_survives_a_frame_with_no_seed_column():
    import pandas as pd

    from training_pipeline.reporting import narrative

    summary = pd.DataFrame({"mean_roi": [0.1, 0.2]})

    assert narrative.strategy_spread(summary, _runs_frame(with_seeds=False)) == []


def test_seed_narrative_still_speaks_when_the_data_is_there():
    """Guards the guard: archived runs DO carry seed ranges and must keep
    getting the commentary."""
    from training_pipeline.reporting import narrative

    assert narrative.seed_noise(_runs_frame(with_seeds=True))


def test_the_roi_chart_renders_without_seed_ranges():
    """It degrades to plain bars, and must not promise an error bar it is not
    drawing."""
    import matplotlib

    matplotlib.use("Agg")
    from training_pipeline.reporting import charts

    charts.plot_roi_with_seed_noise(_runs_frame(with_seeds=False))
    charts.plot_roi_with_seed_noise(_runs_frame(with_seeds=True))
