"""``spread_error_regressor`` end to end through prepare_dataset.

Covers the two things that would otherwise fail silently: outcome columns
reaching the feature matrix, and a target built against a different line than
bets settle into.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_pipeline.config import (
    OUTCOME_ONLY_COLUMNS,
    BettingConfig,
    CleaningConfig,
    DataConfig,
    DatasetType,
    ExperimentConfig,
    HoldoutConfig,
    Market,
    PredictionStrategy,
    TargetFamily,
    WalkForwardConfig,
)
from training_pipeline.data import (
    assert_no_leaking_features,
    prepare_dataset,
    verify_spread_error_column,
)

SNAPSHOTS = (30, 60, 120, 240, 720)


def _spread_frame(*, n_days=90, games_per_day=4, snapshots=SNAPSHOTS, drift=True):
    """A (game, snapshot) frame carrying both markets' targets."""
    rows = []
    game_number = 0
    for offset, day in enumerate(
        pd.date_range("2024-10-20", periods=n_days, freq="D")
    ):
        for _ in range(games_per_day):
            game_number += 1
            for position, minutes in enumerate(snapshots):
                rows.append(
                    {
                        "GAME_ID": f"00221{game_number:05d}",
                        "GAME_DATE": day,
                        "SEASON_YEAR": 2024 + (offset > 60),
                        "TIME_TO_MATCH_MIN": minutes,
                        "_snapshot_position": position,
                    }
                )
    df = pd.DataFrame(rows).sort_values(
        ["GAME_DATE", "TIME_TO_MATCH_MIN"], kind="stable"
    )
    df = df.reset_index(drop=True)

    rng = np.random.default_rng(11)
    total_line = rng.uniform(205, 240, len(df)).round(1)
    df["ODDS_TOTAL_LINE_bet365"] = total_line
    df["TOTAL_POINTS"] = (total_line + rng.normal(0, 12, len(df))).round(1)

    # One margin per GAME (an outcome cannot vary by snapshot), but a spread that
    # MOVES across snapshots, which is what makes the targets differ.
    margins = {
        gid: value
        for gid, value in zip(
            sorted(df["GAME_ID"].unique()),
            rng.integers(-25, 26, df["GAME_ID"].nunique()),
            strict=True,
        )
    }
    df["PTS_TEAM_AWAY"] = 110
    df["PTS_TEAM_HOME"] = 110 + df["GAME_ID"].map(margins)
    df["HOME_MARGIN"] = df["PTS_TEAM_HOME"] - df["PTS_TEAM_AWAY"]

    base = df["GAME_ID"].map(
        dict(zip(sorted(margins), rng.normal(0, 5, len(margins)), strict=True))
    ).round(1)
    step = df["_snapshot_position"] * 0.5 if drift else 0.0
    df["ODDS_SPREAD_LINE_HOME_bet365"] = (base + step).round(1)
    df["SPREAD_ERROR"] = df["HOME_MARGIN"] - df["ODDS_SPREAD_LINE_HOME_bet365"]

    df["ODDS_SPREAD_PRICE_HOME"] = 1.909091
    df["ODDS_SPREAD_PRICE_AWAY"] = 1.909091
    df["FEATURE_A"] = rng.normal(size=len(df))
    df["ODDS_SNAP_MOVE_A"] = np.linspace(-1, 1, len(df))
    return df.drop(columns=["_snapshot_position"])


def _config(tmp_path, csv_path, **overrides):
    kwargs = {
        "experiment_name": "spread",
        "prediction_strategy": PredictionStrategy.SPREAD_ERROR_REGRESSOR,
        "data": DataConfig(
            csv_path=csv_path, dataset_type=DatasetType.INTERMEDIATE_LINE
        ),
        "cleaning": CleaningConfig(verbose=0),
        "holdout": HoldoutConfig(test_size=None, test_days=30),
        "walk_forward": WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=60,
            min_train_games=40,
            max_folds=None,
            train_games=100,
        ),
        "betting": BettingConfig(
            home_price_col="ODDS_SPREAD_PRICE_HOME",
            away_price_col="ODDS_SPREAD_PRICE_AWAY",
        ),
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _write(tmp_path, frame=None):
    frame = _spread_frame() if frame is None else frame
    path = tmp_path / "spread.csv"
    frame.to_csv(path, index=False)
    return path


# --- strategy wiring --------------------------------------------------------


def test_strategy_resolves_to_spread_market_and_target():
    strategy = PredictionStrategy.SPREAD_ERROR_REGRESSOR
    assert strategy.target_family is TargetFamily.SPREAD_ERROR
    assert strategy.market is Market.SPREAD
    assert not strategy.is_classifier


def test_spread_strategy_rejects_a_line_col(tmp_path):
    """A residual regressor predicts the edge, so a line would be meaningless."""
    with pytest.raises(ValueError, match="line_col must be omitted"):
        _config(tmp_path, "x.csv", line_col="ODDS_SPREAD_LINE_HOME_bet365")


def test_config_exposes_the_right_target_and_outcome(tmp_path):
    config = _config(tmp_path, "x.csv")
    assert config.target_col == "SPREAD_ERROR"
    assert config.outcome_col == "HOME_MARGIN"
    assert config.market is Market.SPREAD


def test_totals_strategies_are_untouched(tmp_path):
    config = _config(
        tmp_path, "x.csv", prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR
    )
    assert config.target_col == "LINE_ERROR"
    assert config.outcome_col == "TOTAL_POINTS"
    assert config.market is Market.TOTALS


# --- leakage ----------------------------------------------------------------


@pytest.mark.parametrize("column", OUTCOME_ONLY_COLUMNS)
def test_outcome_columns_are_rejected_from_the_feature_matrix(column):
    with pytest.raises(ValueError, match="reached the feature matrix"):
        assert_no_leaking_features(pd.DataFrame({column: [1.0], "FEATURE_A": [0.0]}))


def test_prepare_dataset_keeps_outcome_columns_off_x(tmp_path):
    """The end-to-end version: they must survive cleaning yet never reach X."""
    prepared = prepare_dataset(_config(tmp_path, _write(tmp_path)))

    for column in ("PTS_TEAM_HOME", "PTS_TEAM_AWAY", "HOME_MARGIN", "SPREAD_ERROR"):
        assert column in prepared.df_full.columns, f"{column} must survive for scoring"
        assert column not in prepared.X.columns, f"{column} leaked into X"
    assert "TOTAL_POINTS" not in prepared.X.columns
    assert prepared.y.name == "SPREAD_ERROR"


def test_totals_run_also_keeps_per_team_points_off_x(tmp_path):
    """The restriction is per-strategy-agnostic: a totals model must not see
    PTS_TEAM_HOME either, since it hands over TOTAL_POINTS directly."""
    prepared = prepare_dataset(
        _config(
            tmp_path,
            _write(tmp_path),
            prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
            betting=BettingConfig(),
        )
    )
    assert "PTS_TEAM_HOME" not in prepared.X.columns
    assert "PTS_TEAM_AWAY" not in prepared.X.columns
    assert "HOME_MARGIN" not in prepared.X.columns


# --- target/line agreement --------------------------------------------------


def test_verify_spread_error_accepts_a_consistent_frame():
    frame = pd.DataFrame(
        {
            "HOME_MARGIN": [8.0, -3.0, 0.0],
            "ODDS_SPREAD_LINE_HOME_bet365": [4.0, 1.5, 0.0],
            "SPREAD_ERROR": [4.0, -4.5, 0.0],
        }
    )
    assert verify_spread_error_column(
        frame,
        spread_line_col="ODDS_SPREAD_LINE_HOME_bet365",
        outcome_col="HOME_MARGIN",
    ) is frame


def test_verify_spread_error_catches_a_flipped_sign():
    """The exact failure mode: target built with the opposite convention."""
    frame = pd.DataFrame(
        {
            "HOME_MARGIN": [8.0, -3.0],
            "ODDS_SPREAD_LINE_HOME_bet365": [4.0, 1.5],
            "SPREAD_ERROR": [12.0, -1.5],  # margin + line, not margin - line
        }
    )
    with pytest.raises(ValueError, match="was built against a different line"):
        verify_spread_error_column(
            frame,
            spread_line_col="ODDS_SPREAD_LINE_HOME_bet365",
            outcome_col="HOME_MARGIN",
        )


def test_verify_spread_error_catches_a_per_game_join(tmp_path):
    """A closing line broadcast over every snapshot no longer matches the target."""
    frame = _spread_frame(n_days=6)
    closing = frame.groupby("GAME_ID")["ODDS_SPREAD_LINE_HOME_bet365"].transform("last")
    frame["ODDS_SPREAD_LINE_HOME_bet365"] = closing  # the mistake
    with pytest.raises(ValueError, match="was built against a different line"):
        verify_spread_error_column(
            frame,
            spread_line_col="ODDS_SPREAD_LINE_HOME_bet365",
            outcome_col="HOME_MARGIN",
        )


def test_a_schema_2_0_csv_is_refused_with_a_useful_message(tmp_path):
    frame = _spread_frame(n_days=20).drop(
        columns=["SPREAD_ERROR", "HOME_MARGIN", "PTS_TEAM_HOME", "PTS_TEAM_AWAY"]
    )
    with pytest.raises(KeyError, match="schema 2_1"):
        prepare_dataset(_config(tmp_path, _write(tmp_path, frame)))


# --- pushes -----------------------------------------------------------------


def test_pushes_survive_into_training(tmp_path):
    """SPREAD_ERROR == 0 is a real observation and must not be dropped."""
    frame = _spread_frame(n_days=40)
    # Force a push on a slice of rows.
    push_rows = frame.index[:50]
    frame.loc[push_rows, "ODDS_SPREAD_LINE_HOME_bet365"] = frame.loc[
        push_rows, "HOME_MARGIN"
    ]
    frame["SPREAD_ERROR"] = frame["HOME_MARGIN"] - frame["ODDS_SPREAD_LINE_HOME_bet365"]
    assert (frame["SPREAD_ERROR"] == 0).sum() >= 50

    prepared = prepare_dataset(_config(tmp_path, _write(tmp_path, frame)))
    assert (prepared.y == 0).sum() > 0, "pushes were dropped from the regression target"
    assert prepared.n_pushes_excluded == 0
