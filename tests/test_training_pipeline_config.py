import pytest
from pydantic import ValidationError

from training_pipeline.config import (
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    OptunaConfig,
    RefitConfig,
    SampleWeightConfig,
    TargetFamily,
    WalkForwardConfig,
)


def _base_kwargs(**overrides):
    kwargs = {
        "experiment_name": "test_experiment",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="data/train_data/example.csv"),
    }
    kwargs.update(overrides)
    return kwargs


def test_total_points_requires_line_col():
    with pytest.raises(ValidationError, match="line_col is required"):
        ExperimentConfig(**_base_kwargs(line_col=None))


def test_line_error_forbids_line_col():
    kwargs = _base_kwargs(target_family=TargetFamily.LINE_ERROR, line_col="ODDS_TOTAL_LINE_bet365")
    with pytest.raises(ValidationError, match="line_col must be omitted"):
        ExperimentConfig(**kwargs)


def test_total_points_accepts_recency_weighting():
    """Previously rejected, because upstream's optuna_total_points.py has no
    sample-weight parameters. training_pipeline supplies its own objective and
    final-fit path, so both target families support weighting.
    """
    config = ExperimentConfig(
        **_base_kwargs(sample_weight=SampleWeightConfig(enabled=True, lambda_=0.005))
    )
    assert config.sample_weight.enabled is True
    assert config.sample_weight.lambda_ == 0.005


def test_total_points_accepts_tuned_sample_weight_lambda():
    config = ExperimentConfig(
        **_base_kwargs(
            sample_weight=SampleWeightConfig(
                enabled=True, tune_lambda=True, lambda_bounds=(1e-4, 0.02)
            )
        )
    )
    assert config.sample_weight.tune_lambda is True


def test_sample_weighting_changes_the_fingerprint():
    """Weighted and unweighted training are different problems, so a
    persistent study must not be resumed across them.
    """
    unweighted = ExperimentConfig(**_base_kwargs()).fingerprint()
    weighted = ExperimentConfig(
        **_base_kwargs(sample_weight=SampleWeightConfig(enabled=True, lambda_=0.005))
    ).fingerprint()
    assert unweighted != weighted


def test_line_error_auto_adds_required_exclude_cols():
    kwargs = _base_kwargs(target_family=TargetFamily.LINE_ERROR, line_col=None)
    config = ExperimentConfig(**kwargs)
    assert "LINE_ERROR" in config.exclude_cols
    assert "TOTAL_POINTS" in config.exclude_cols


def test_total_points_auto_adds_total_points_exclude_col():
    kwargs = _base_kwargs(exclude_cols=["SEASON_YEAR", "GAME_DATE"])
    config = ExperimentConfig(**kwargs)
    assert "TOTAL_POINTS" in config.exclude_cols


def test_config_round_trips_through_json():
    config = ExperimentConfig(**_base_kwargs())
    restored = ExperimentConfig.model_validate_json(config.model_dump_json())
    assert restored == config


def test_refit_has_no_separate_train_games_knob():
    """Regression: the CV window and the final-refit window must be a single
    value. Two knobs could disagree, which would select hyperparameters for one
    training-set size and then fit the shipped model on another.
    """
    assert "train_games" not in RefitConfig.model_fields

    config = ExperimentConfig(
        **_base_kwargs(walk_forward=WalkForwardConfig(train_games=1234))
    )
    assert config.walk_forward.train_games == 1234
    assert config.resolved_window_dir_label == "1234_games"


def test_legacy_refit_train_games_in_yaml_is_ignored_not_fatal():
    """Configs written before the unification must still load."""
    config = ExperimentConfig.model_validate(
        {
            **_base_kwargs(),
            "refit": {"strategy": "rolling_window", "train_games": 999},
        }
    )
    assert not hasattr(config.refit, "train_games")


def test_training_version_defaults_to_unset():
    assert ExperimentConfig(**_base_kwargs()).training_version is None


def test_training_version_is_kept_verbatim_when_set():
    config = ExperimentConfig(**_base_kwargs(training_version="2.1-style-features"))
    assert config.training_version == "2.1-style-features"


def test_blank_training_version_is_treated_as_unset():
    """A whitespace-only label is a mistake, not a version."""
    assert ExperimentConfig(**_base_kwargs(training_version="   ")).training_version is None
    assert ExperimentConfig(**_base_kwargs(training_version="")).training_version is None


def test_training_version_is_stripped():
    config = ExperimentConfig(**_base_kwargs(training_version="  v3  "))
    assert config.training_version == "v3"


def test_training_version_does_not_change_the_fingerprint():
    """It is a human label: relabelling must not fork a persistent study, and
    real changes are already captured by the fingerprint on their own.
    """
    base = ExperimentConfig(**_base_kwargs()).fingerprint()
    labelled = ExperimentConfig(**_base_kwargs(training_version="2.1")).fingerprint()
    relabelled = ExperimentConfig(**_base_kwargs(training_version="9.9")).fingerprint()

    assert base == labelled == relabelled


def test_training_version_round_trips_through_json():
    config = ExperimentConfig(**_base_kwargs(training_version="2.1"))
    assert ExperimentConfig.model_validate_json(config.model_dump_json()) == config


def test_playoffs_are_excluded_by_default():
    config = ExperimentConfig(**_base_kwargs())
    assert config.data.exclude_playoffs is True
    assert config.data.allowed_season_types == ("Regular Season", "Play-In Tournament")
    assert "Playoffs" not in config.data.allowed_season_types


def test_allowed_season_types_must_be_known_values():
    with pytest.raises(ValidationError, match="Unknown season types"):
        ExperimentConfig(
            **_base_kwargs(
                data=DataConfig(
                    csv_path="x.csv", allowed_season_types=("Regular Season", "Postseason")
                )
            )
        )


def test_allowed_season_types_cannot_be_empty_while_filtering():
    with pytest.raises(ValidationError, match="must not be empty"):
        ExperimentConfig(
            **_base_kwargs(
                data=DataConfig(csv_path="x.csv", exclude_playoffs=True, allowed_season_types=())
            )
        )


def test_toggling_playoff_exclusion_changes_the_fingerprint():
    """Including playoffs changes the training distribution, so a persistent
    Optuna study must not be resumed across the two settings.
    """
    excluded = ExperimentConfig(**_base_kwargs()).fingerprint()
    included = ExperimentConfig(
        **_base_kwargs(
            data=DataConfig(
                csv_path="data/train_data/example.csv",
                allowed_season_types=("Regular Season", "Play-In Tournament", "Playoffs"),
            )
        )
    ).fingerprint()
    assert excluded != included


def test_fingerprint_ignores_cosmetic_and_output_only_fields():
    """Renaming an experiment or asking for more trials must still resume the
    same persistent Optuna study -- those don't change what a trial means.
    """
    base = ExperimentConfig(**_base_kwargs()).fingerprint()

    assert ExperimentConfig(**_base_kwargs(experiment_name="renamed")).fingerprint() == base
    assert ExperimentConfig(**_base_kwargs(window_dir_label="3_seasons")).fingerprint() == base
    assert (
        ExperimentConfig(**_base_kwargs(optuna=OptunaConfig(n_trials=500))).fingerprint()
        == base
    )
    assert (
        ExperimentConfig(**_base_kwargs(save_experiment_artifacts=False)).fingerprint()
        == base
    )


def test_fingerprint_changes_when_trials_would_not_be_comparable():
    """Regression: persistent Optuna storage used to be keyed only on the
    experiment name, so changing the CSV or fold layout silently appended
    incomparable trials to an existing study.
    """
    base = ExperimentConfig(**_base_kwargs()).fingerprint()

    different_csv = _base_kwargs(data=DataConfig(csv_path="data/train_data/other.csv"))
    assert ExperimentConfig(**different_csv).fingerprint() != base

    different_cleaning = _base_kwargs(cleaning=CleaningConfig(nan_threshold=50.0))
    assert ExperimentConfig(**different_cleaning).fingerprint() != base

    different_window = _base_kwargs(walk_forward=WalkForwardConfig(train_games=2500))
    assert ExperimentConfig(**different_window).fingerprint() != base

    different_objective = _base_kwargs(
        optuna=OptunaConfig(objective_name="reg:pseudohubererror")
    )
    assert ExperimentConfig(**different_objective).fingerprint() != base


def test_fingerprint_is_stable_across_equivalent_constructions():
    assert (
        ExperimentConfig(**_base_kwargs()).fingerprint()
        == ExperimentConfig(**_base_kwargs()).fingerprint()
    )


def test_line_error_config_round_trips_through_json():
    kwargs = _base_kwargs(
        target_family=TargetFamily.LINE_ERROR,
        line_col=None,
        sample_weight=SampleWeightConfig(enabled=True, tune_lambda=True, lambda_=0.005),
    )
    config = ExperimentConfig(**kwargs)
    restored = ExperimentConfig.model_validate_json(config.model_dump_json())
    assert restored == config


# --- holdout sizing: fixed calendar window ----------------------------------


def test_exactly_one_holdout_sizing_rule_is_required():
    from training_pipeline.config import HoldoutConfig

    with pytest.raises(ValueError, match="exactly one"):
        HoldoutConfig(test_size=0.05, test_days=60)
    with pytest.raises(ValueError, match="exactly one"):
        HoldoutConfig(test_size=None, test_games=None, test_days=None)

    assert HoldoutConfig(test_size=None, test_days=60).test_days == 60


def test_test_days_must_be_positive():
    from training_pipeline.config import HoldoutConfig

    with pytest.raises(ValueError, match="must be > 0"):
        HoldoutConfig(test_size=None, test_days=0)


def test_days_holdout_never_splits_a_game_day_across_the_boundary():
    """A count-based cut can put some of a day's games in training and the rest
    in test -- exactly the leak the daily walk-forward exists to avoid.
    """
    import pandas as pd

    from training_pipeline.splits import split_latest_days_holdout

    dates = pd.to_datetime(
        ["2026-01-01"] * 5 + ["2026-01-02"] * 5 + ["2026-01-03"] * 5
    )
    df = pd.DataFrame({"GAME_DATE": dates, "x": range(15)})

    df_dev, df_test = split_latest_days_holdout(df, date_col="GAME_DATE", test_days=1)

    assert set(df_dev["GAME_DATE"]).isdisjoint(set(df_test["GAME_DATE"]))
    assert df_dev["GAME_DATE"].max() < df_test["GAME_DATE"].min()
    # Whole days, so sizes are multiples of the 5 games per day.
    assert len(df_test) == 5 and len(df_dev) == 10


def test_days_holdout_is_measured_from_the_last_game_not_from_today():
    """Re-running an old config must reproduce its split."""
    import pandas as pd

    from training_pipeline.splits import split_latest_days_holdout

    df = pd.DataFrame(
        {"GAME_DATE": pd.date_range("2020-01-01", periods=100, freq="D")}
    )
    df_dev, df_test = split_latest_days_holdout(df, date_col="GAME_DATE", test_days=10)
    assert df_test["GAME_DATE"].min() == pd.Timestamp("2020-03-31")
    assert len(df_test) == 10


def test_days_holdout_refuses_to_consume_the_whole_dataset():
    import pandas as pd

    from training_pipeline.splits import split_latest_days_holdout

    df = pd.DataFrame({"GAME_DATE": pd.date_range("2026-01-01", periods=5, freq="D")})
    with pytest.raises(ValueError, match="nothing left to train on"):
        split_latest_days_holdout(df, date_col="GAME_DATE", test_days=999)


def test_datasets_ending_on_the_same_day_get_the_identical_window():
    """The reason for the change: a 5% fraction gave the 2026-07-04 A/B
    different start dates (Mar 7 vs Mar 8) and game counts (293 vs 287),
    quietly making its ROI columns non-comparable. Row count must not move the
    window.
    """
    import pandas as pd

    from training_pipeline.splits import split_latest_days_holdout

    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    dense = pd.DataFrame({"GAME_DATE": dates.repeat(8)})
    # Half the coverage in the middle, but the same first and last game.
    sparse_dates = dates[::2].union(pd.DatetimeIndex([dates[-1]]))
    sparse = pd.DataFrame({"GAME_DATE": sparse_dates})

    _, test_dense = split_latest_days_holdout(dense, date_col="GAME_DATE", test_days=30)
    _, test_sparse = split_latest_days_holdout(sparse, date_col="GAME_DATE", test_days=30)

    assert test_dense["GAME_DATE"].min() == test_sparse["GAME_DATE"].min()
    assert test_dense["GAME_DATE"].max() == test_sparse["GAME_DATE"].max()
    # Different game counts are expected and are the honest signal: they mean
    # the datasets genuinely cover the window differently.
    assert len(test_dense) != len(test_sparse)


def test_the_window_is_anchored_to_each_datasets_own_last_game():
    """The alignment guarantee has one precondition worth knowing: two datasets
    line up only if they END on the same date. A CSV rebuilt to a later date
    shifts its whole window, so the cohort check in the comparison notebook
    still earns its place.
    """
    import pandas as pd

    from training_pipeline.splits import split_latest_days_holdout

    early = pd.DataFrame({"GAME_DATE": pd.date_range("2026-01-01", periods=100)})
    late = pd.DataFrame({"GAME_DATE": pd.date_range("2026-01-01", periods=110)})

    _, test_early = split_latest_days_holdout(early, date_col="GAME_DATE", test_days=30)
    _, test_late = split_latest_days_holdout(late, date_col="GAME_DATE", test_days=30)

    assert test_early["GAME_DATE"].max() != test_late["GAME_DATE"].max()
