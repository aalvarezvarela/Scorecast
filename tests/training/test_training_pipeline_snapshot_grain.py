"""Rows are not games: the intermediate-line dataset's two modes.

The intermediate-line CSV holds one row per (game, pre-game snapshot). Every
``*_games`` knob in the pipeline used to be a row count, so on a ten-snapshot
frame ``train_games: 3500`` trained on 350 games and ``min_validation_games: 25``
accepted folds of 2.5. Nothing errored -- the correction lived as a hand-written
multiplier in a YAML comment.

Two modes are supported now, and the tests below are mostly about the ABSENCE of
the old arithmetic:

  pooled              data.snapshot_minutes = None. Several rows per game, and
                      every knob still means games.
  one per timepoint   data.snapshot_minutes = 720. One row per game, so the two
                      counts coincide and every splitter is correct again.

The closing-line no-op is the load-bearing property here: on a one-row-per-game
frame all of this must reduce to exactly the code that ran before. Every test was
mutation-checked -- the fix reverted, the test observed to fail, the fix restored.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from nba_ou.modeling.modeling import tail_n_games

from training_pipeline.config import (
    CleaningConfig,
    DataConfig,
    DatasetType,
    ExperimentConfig,
    HoldoutConfig,
    PredictionStrategy,
    WalkForwardConfig,
)
from training_pipeline.data import (
    assert_one_row_per_game,
    build_feature_matrix,
    filter_to_snapshot,
    load_scoring_sidecar,
    rolling_window_index,
)
from training_pipeline.splits import (
    _tail_games,
    build_rolling_origin_plan,
    resolve_game_codes,
)

SNAPSHOTS = (30, 60, 120, 240, 720)


# --- fixtures ---------------------------------------------------------------


def _pooled_frame(
    *,
    n_days: int = 120,
    games_per_day: int = 4,
    snapshots: tuple[int, ...] = SNAPSHOTS,
) -> pd.DataFrame:
    """A (game, snapshot) frame: one row per game per horizon, date-sorted.

    Deliberately NOT grouped by game in row order -- rows are interleaved within
    a day the way a date sort leaves them -- so a window that assumed a game's
    rows were contiguous would be caught.
    """
    rows: list[dict] = []
    game_number = 0
    for offset, day in enumerate(pd.date_range("2024-10-20", periods=n_days, freq="D")):
        for _ in range(games_per_day):
            game_number += 1
            for minutes in snapshots:
                rows.append(
                    {
                        "GAME_ID": f"00221{game_number:05d}",
                        "GAME_DATE": day,
                        "SEASON_YEAR": 2024 + (offset > 80),
                        "TIME_TO_MATCH_MIN": minutes,
                    }
                )
    df = pd.DataFrame(rows)
    # Interleave within each day, as a stable date sort would.
    df = df.sort_values(["GAME_DATE", "TIME_TO_MATCH_MIN"], kind="stable")
    df = df.reset_index(drop=True)

    rng = np.random.default_rng(4)
    line = rng.uniform(205, 240, len(df)).round(1)
    df["ODDS_TOTAL_LINE_bet365"] = line
    df["TOTAL_POINTS"] = (line + rng.normal(0, 12, len(df))).round(1)
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]
    df["FEATURE_A"] = rng.normal(size=len(df))
    return df


def _closing_frame(**kwargs) -> pd.DataFrame:
    """The same schedule at one row per game."""
    return _pooled_frame(snapshots=(60,), **kwargs)


def _config(tmp_path, **overrides) -> ExperimentConfig:
    kwargs: dict = {
        "experiment_name": "grain",
        "prediction_strategy": PredictionStrategy.LINE_ERROR_REGRESSOR,
        "data": DataConfig(
            csv_path="x.csv", dataset_type=DatasetType.INTERMEDIATE_LINE
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
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# --- the game key -----------------------------------------------------------


def test_a_one_row_per_game_frame_has_no_game_key_at_all():
    """None, not arange. It is what keeps the closing-line path on the original
    row arithmetic by construction rather than by numerical coincidence."""
    assert resolve_game_codes(_closing_frame(), game_id_col="GAME_ID") is None


def test_a_frame_without_the_id_column_has_no_game_key():
    frame = _closing_frame().drop(columns=["GAME_ID"])
    assert resolve_game_codes(frame, game_id_col="GAME_ID") is None


def test_a_pooled_frame_has_one_code_per_game_not_per_row():
    frame = _pooled_frame(n_days=10)
    codes = resolve_game_codes(frame, game_id_col="GAME_ID")
    assert codes is not None
    assert len(codes) == len(frame)
    assert len(np.unique(codes)) == frame["GAME_ID"].nunique() == 40


# --- the window -------------------------------------------------------------


def test_the_window_keeps_whole_games_never_a_fraction_of_one():
    """A window holding four of a game's five snapshots is neither a row window
    nor a game window, and would train on a game the model half-saw."""
    frame = _pooled_frame(n_days=10)
    codes = resolve_game_codes(frame, game_id_col="GAME_ID")
    idx = np.arange(len(frame))

    kept = _tail_games(codes, idx, 7)

    per_game = frame.iloc[kept].groupby("GAME_ID").size()
    assert len(per_game) == 7
    assert set(per_game) == {len(SNAPSHOTS)}


def test_the_window_takes_the_LATEST_games():
    frame = _pooled_frame(n_days=10)
    codes = resolve_game_codes(frame, game_id_col="GAME_ID")

    kept = _tail_games(codes, np.arange(len(frame)), 7)

    expected = sorted(frame["GAME_ID"].unique())[-7:]
    assert sorted(frame.iloc[kept]["GAME_ID"].unique()) == expected


def test_without_a_game_key_the_window_is_the_original_tail_slice():
    """The closing-line no-op, asserted on the primitive itself."""
    idx = np.arange(500)
    assert np.array_equal(_tail_games(None, idx, 120), idx[-120:])


def test_asking_for_more_games_than_exist_returns_everything():
    frame = _pooled_frame(n_days=5)
    codes = resolve_game_codes(frame, game_id_col="GAME_ID")
    idx = np.arange(len(frame))
    assert np.array_equal(_tail_games(codes, idx, 10_000), idx)


# --- the fold layout --------------------------------------------------------


def test_rolling_origin_counts_folds_in_games_not_rows(tmp_path):
    frame = _pooled_frame()
    plan = build_rolling_origin_plan(frame, _config(tmp_path))

    # eval_span_games=60 is sixty GAMES. At five snapshots each that is 300
    # rows, and the old code would have stopped after 60 rows -- 12 games.
    assert plan.n_validation_games >= 60
    assert plan.n_validation_rows == plan.n_validation_games * len(SNAPSHOTS)
    assert sum(plan.fold_game_counts) == plan.n_validation_games


def test_the_training_window_is_games_on_a_pooled_frame(tmp_path):
    frame = _pooled_frame()
    plan = build_rolling_origin_plan(frame, _config(tmp_path))

    for fold in plan.folds:
        rows = fold.train_idx(50)
        assert frame.iloc[rows]["GAME_ID"].nunique() == 50
        assert len(rows) == 50 * len(SNAPSHOTS)


def test_the_window_ceiling_is_expressed_in_games(tmp_path):
    frame = _pooled_frame()
    plan = build_rolling_origin_plan(frame, _config(tmp_path))

    # min_history_games must be a game count, so it is far below the row count.
    assert plan.min_history_games < len(frame) / len(SNAPSHOTS)
    with pytest.raises(ValueError, match="GAMES"):
        plan.assert_window_fits(plan.min_history_games + 1)


def test_fold_info_reports_both_games_and_rows(tmp_path):
    """"28 folds, 855 validation games" hides whether that is 855 predictions or
    4,275 of them, and the objective is averaged over the second number."""
    plan = build_rolling_origin_plan(_pooled_frame(), _config(tmp_path))
    info = plan.fold_info

    assert (info["test_n_rows"] == info["test_n_games"] * len(SNAPSHOTS)).all()
    assert (info["train_n_rows"] > info["train_n_games"]).all()


def test_the_pooled_and_single_snapshot_layouts_score_the_same_games(tmp_path):
    """The control has to be the same cohort as the arm it controls for, or the
    comparison measures which games each one happened to get."""
    config = _config(tmp_path)
    pooled = build_rolling_origin_plan(_pooled_frame(), config)

    sliced = _pooled_frame()
    sliced = sliced[sliced["TIME_TO_MATCH_MIN"] == 720].reset_index(drop=True)
    single = build_rolling_origin_plan(sliced, config)

    assert pooled.n_validation_games == single.n_validation_games
    assert pooled.fold_game_counts == single.fold_game_counts
    assert pooled.min_history_games == single.min_history_games


def test_a_closing_line_layout_is_unchanged_by_the_presence_of_game_ids(tmp_path):
    """THE no-op. One row per game must produce byte-identical folds whether or
    not the frame carries an identifier at all."""
    config = _config(tmp_path)
    frame = _closing_frame()

    with_ids = build_rolling_origin_plan(frame, config)
    without = build_rolling_origin_plan(frame.drop(columns=["GAME_ID"]), config)

    assert with_ids.n_folds == without.n_folds
    assert with_ids.n_validation_games == without.n_validation_games
    assert with_ids.fold_game_counts == without.fold_game_counts
    for left, right in zip(with_ids.folds, without.folds, strict=True):
        assert np.array_equal(left.valid_idx, right.valid_idx)
        assert np.array_equal(left.train_idx(80), right.train_idx(80))


def test_min_validation_games_is_a_game_floor_not_a_row_floor(tmp_path):
    """At five snapshots a row floor of 25 accepts folds of five games."""
    config = _config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=1,
            eval_span_games=60,
            min_train_games=40,
            max_folds=None,
            train_games=100,
            min_validation_games=12,
        ),
    )
    plan = build_rolling_origin_plan(_pooled_frame(), config)

    assert plan.n_folds_below_min == 0
    assert min(plan.fold_game_counts) >= 12


# --- selecting one horizon --------------------------------------------------


def test_filtering_to_one_horizon_gives_one_row_per_game():
    frame = _pooled_frame(n_days=10)
    sliced = filter_to_snapshot(frame, snapshot_col="TIME_TO_MATCH_MIN", minutes=720)

    assert len(sliced) == frame["GAME_ID"].nunique()
    assert not sliced.duplicated(subset=["GAME_ID"]).any()


def test_an_absent_horizon_names_the_ones_that_exist():
    """Training on zero rows would fail much later with nothing pointing here."""
    with pytest.raises(ValueError, match=r"Horizons present.*\[30, 60, 120, 240, 720\]"):
        filter_to_snapshot(
            _pooled_frame(n_days=3), snapshot_col="TIME_TO_MATCH_MIN", minutes=90
        )


def test_filtering_a_frame_with_no_snapshot_column_is_an_error():
    frame = _closing_frame(n_days=3).drop(columns=["TIME_TO_MATCH_MIN"])
    with pytest.raises(KeyError, match="no 'TIME_TO_MATCH_MIN' column"):
        filter_to_snapshot(frame, snapshot_col="TIME_TO_MATCH_MIN", minutes=720)


def test_a_duplicated_game_and_snapshot_pair_is_refused():
    """It would not error anywhere downstream; it would quietly halve the
    variance of every per-horizon interval."""
    frame = _pooled_frame(n_days=3)
    doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        assert_one_row_per_game(
            doubled,
            game_id_col="GAME_ID",
            snapshot_col="TIME_TO_MATCH_MIN",
            context="test.csv",
        )


def test_the_invariant_check_is_silent_on_a_closing_line_frame():
    frame = _closing_frame(n_days=3).drop(columns=["TIME_TO_MATCH_MIN"])
    assert_one_row_per_game(
        frame,
        game_id_col="GAME_ID",
        snapshot_col="TIME_TO_MATCH_MIN",
        context="test.csv",
    )


# --- config guards ----------------------------------------------------------


def test_a_pooled_frame_may_not_use_a_row_counting_splitter(tmp_path):
    """test_anchored describes a fold as N rows, which on this frame is N/5
    games -- silently, with every number looking ordinary."""
    with pytest.raises(ValueError, match="counts a fold in ROWS"):
        _config(
            tmp_path,
            walk_forward=WalkForwardConfig(
                strategy="test_anchored", test_games=50, train_games=400
            ),
        )


def test_a_single_horizon_frame_may_use_any_splitter(tmp_path):
    """One row per game, so the two counts coincide and nothing is ambiguous."""
    config = _config(
        tmp_path,
        data=DataConfig(
            csv_path="x.csv",
            dataset_type=DatasetType.INTERMEDIATE_LINE,
            snapshot_minutes=720,
        ),
        walk_forward=WalkForwardConfig(
            strategy="test_anchored", test_games=50, train_games=400
        ),
    )
    assert config.data.snapshot_minutes == 720


@pytest.mark.parametrize(
    "field, value",
    [("snapshot_minutes", 720), ("scoring_csv_path", "scoring.csv")],
)
def test_snapshot_knobs_are_refused_on_a_closing_line_dataset(field, value):
    """Both would be silent no-ops there -- no snapshot column to filter or join
    on -- which is the failure dataset_type exists to prevent."""
    with pytest.raises(ValueError, match=f"data.{field} requires"):
        DataConfig(
            csv_path="x.csv",
            dataset_type=DatasetType.CLOSING_LINE,
            **{field: value},
        )


# --- the feature matrix -----------------------------------------------------


def test_the_feature_matrix_is_an_allow_list_when_one_is_given():
    """A deny-list only removes what it was told about, so every non-feature
    column added later reaches the model unless five call sites are updated."""
    frame = _pooled_frame(n_days=3)
    X, _ = build_feature_matrix(
        frame, target_col="LINE_ERROR", feature_names=["FEATURE_A"]
    )
    assert list(X.columns) == ["FEATURE_A"]
    assert "GAME_ID" not in X.columns


def test_a_frame_missing_a_prepared_feature_is_an_error():
    frame = _pooled_frame(n_days=3).drop(columns=["FEATURE_A"])
    with pytest.raises(KeyError, match="absent from this frame"):
        build_feature_matrix(
            frame, target_col="LINE_ERROR", feature_names=["FEATURE_A"]
        )


def test_the_deny_list_still_works_for_prepare_dataset():
    frame = _pooled_frame(n_days=3)
    X, _ = build_feature_matrix(
        frame, target_col="LINE_ERROR", exclude_cols=["LINE_ERROR", "GAME_ID"]
    )
    assert "GAME_ID" not in X.columns
    assert "FEATURE_A" in X.columns


# --- the refit window -------------------------------------------------------


def test_the_refit_window_counts_games_when_given_a_game_key():
    frame = _pooled_frame(n_days=10)
    window = rolling_window_index(
        frame.index, 7, game_ids=frame["GAME_ID"]
    )
    assert frame.loc[window, "GAME_ID"].nunique() == 7
    assert len(window) == 7 * len(SNAPSHOTS)


def test_the_refit_window_without_a_game_key_is_the_original_tail():
    frame = _closing_frame(n_days=10)
    window = rolling_window_index(frame.index, 7, game_ids=None)
    assert list(window) == list(frame.index[-7:])


def test_the_daily_walk_forward_window_keeps_whole_games():
    frame = _pooled_frame(n_days=10)
    kept = tail_n_games(frame, 7, group_col="GAME_ID")
    assert kept["GAME_ID"].nunique() == 7
    assert len(kept) == 7 * len(SNAPSHOTS)


def test_the_daily_walk_forward_window_without_a_group_is_tail():
    frame = _closing_frame(n_days=10)
    assert tail_n_games(frame, 7, group_col=None).equals(frame.tail(7))


# --- the scoring sidecar ----------------------------------------------------


def test_the_sidecar_joins_on_game_and_snapshot(tmp_path):
    frame = _pooled_frame(n_days=4)
    sidecar = frame[["GAME_ID", "TIME_TO_MATCH_MIN"]].copy()
    sidecar["ODDS_CLOSING_TOTAL_LINE_bet365"] = 220.5
    path = tmp_path / "scoring.csv"
    sidecar.to_csv(path, index=False)

    merged, attached = load_scoring_sidecar(
        frame,
        csv_path=path,
        game_id_col="GAME_ID",
        snapshot_col="TIME_TO_MATCH_MIN",
    )

    assert attached == ["ODDS_CLOSING_TOTAL_LINE_bet365"]
    assert len(merged) == len(frame)
    assert (merged["ODDS_CLOSING_TOTAL_LINE_bet365"] == 220.5).all()


def test_a_sidecar_that_would_overwrite_a_training_column_is_refused(tmp_path):
    """Letting a join decide which version of a column wins is how a feature
    silently becomes something else."""
    frame = _pooled_frame(n_days=4)
    sidecar = frame[["GAME_ID", "TIME_TO_MATCH_MIN", "FEATURE_A"]].copy()
    path = tmp_path / "scoring.csv"
    sidecar.to_csv(path, index=False)

    with pytest.raises(ValueError, match="would overwrite"):
        load_scoring_sidecar(
            frame,
            csv_path=path,
            game_id_col="GAME_ID",
            snapshot_col="TIME_TO_MATCH_MIN",
        )


# --- prepare_dataset, end to end --------------------------------------------


def _write_intermediate_csv(tmp_path, snapshots=SNAPSHOTS, n_days=60) -> pathlib.Path:
    frame = _pooled_frame(n_days=n_days, snapshots=snapshots)
    frame = frame.drop(columns=["LINE_ERROR"])  # derived by prepare_dataset
    frame["ODDS_SNAP_MOVE_A"] = np.linspace(-1, 1, len(frame))
    frame["FEATURE_B"] = np.linspace(2, 3, len(frame))
    path = tmp_path / "intermediate.csv"
    frame.to_csv(path, index=False)
    return path


def _prepared_config(tmp_path, csv_path, **data_overrides) -> ExperimentConfig:
    return _config(
        tmp_path,
        data=DataConfig(
            csv_path=csv_path,
            dataset_type=DatasetType.INTERMEDIATE_LINE,
            **data_overrides,
        ),
        # Real cleaning, deliberately: keep_all_cols=True would skip the
        # constant-column removal that these tests are partly about.
        cleaning=CleaningConfig(verbose=0),
    )


def test_game_id_survives_cleaning_but_never_reaches_the_feature_matrix(tmp_path):
    """Both halves matter. The window arithmetic needs it on df_full; a model
    given it would split on a string encoding date and sequence."""
    from training_pipeline.data import prepare_dataset

    csv_path = _write_intermediate_csv(tmp_path)
    prepared = prepare_dataset(_prepared_config(tmp_path, csv_path))

    assert "GAME_ID" in prepared.df_full.columns
    assert "GAME_ID" not in prepared.X.columns
    assert "GAME_ID" not in prepared.feature_names


def test_the_pooled_grain_is_reported_on_the_prepared_dataset(tmp_path):
    from training_pipeline.data import prepare_dataset

    csv_path = _write_intermediate_csv(tmp_path)
    prepared = prepare_dataset(_prepared_config(tmp_path, csv_path))

    assert prepared.n_snapshots == len(SNAPSHOTS)
    assert prepared.rows_per_game == len(SNAPSHOTS)
    assert len(prepared.df_full) == prepared.n_games * len(SNAPSHOTS)
    assert prepared.is_pooled_snapshots


def test_time_to_match_is_a_feature_when_pooled(tmp_path):
    """The whole reason the pooled arm exists: the model conditions on it."""
    from training_pipeline.data import prepare_dataset

    csv_path = _write_intermediate_csv(tmp_path)
    prepared = prepare_dataset(_prepared_config(tmp_path, csv_path))

    assert "TIME_TO_MATCH_MIN" in prepared.feature_names


def test_selecting_one_horizon_gives_a_one_row_per_game_dataset(tmp_path):
    from training_pipeline.data import prepare_dataset

    csv_path = _write_intermediate_csv(tmp_path)
    prepared = prepare_dataset(
        _prepared_config(tmp_path, csv_path, snapshot_minutes=720)
    )

    assert prepared.n_snapshots == 1
    assert prepared.rows_per_game == 1.0
    assert not prepared.is_pooled_snapshots
    assert len(prepared.df_full) == prepared.n_games
    # Constant within the slice, so cleaning removes it rather than handing the
    # model a dead column. Force-keeping it would only smuggle it past.
    assert "TIME_TO_MATCH_MIN" not in prepared.feature_names


def test_the_two_modes_see_the_same_games(tmp_path):
    """The control must be the same cohort as the arm, or it controls nothing."""
    from training_pipeline.data import prepare_dataset

    csv_path = _write_intermediate_csv(tmp_path)
    pooled = prepare_dataset(_prepared_config(tmp_path, csv_path))
    single = prepare_dataset(
        _prepared_config(tmp_path, csv_path, snapshot_minutes=720)
    )

    assert pooled.n_games == single.n_games
    assert set(pooled.df_full["GAME_ID"]) == set(single.df_full["GAME_ID"])


def test_the_sidecar_reaches_df_full_and_not_the_feature_matrix(tmp_path):
    """Ordering is the safety property: joined after X is built, so a closing
    line cannot be handed to a model that is pricing a bet 12 hours out."""
    from training_pipeline.data import prepare_dataset

    csv_path = _write_intermediate_csv(tmp_path)
    source = pd.read_csv(csv_path, dtype={"GAME_ID": str})
    sidecar = source[["GAME_ID", "TIME_TO_MATCH_MIN"]].copy()
    sidecar["ODDS_CLOSING_TOTAL_LINE_bet365"] = 221.5
    sidecar_path = tmp_path / "scoring.csv"
    sidecar.to_csv(sidecar_path, index=False)

    prepared = prepare_dataset(
        _prepared_config(tmp_path, csv_path, scoring_csv_path=sidecar_path)
    )

    assert prepared.scoring_columns == ["ODDS_CLOSING_TOTAL_LINE_bet365"]
    assert "ODDS_CLOSING_TOTAL_LINE_bet365" in prepared.df_full.columns
    assert "ODDS_CLOSING_TOTAL_LINE_bet365" not in prepared.feature_names
    assert "ODDS_CLOSING_TOTAL_LINE_bet365" not in prepared.X.columns


def test_single_horizon_keeps_the_sidecar_join_key_out_of_features(tmp_path):
    """The constant horizon is still needed to join closing-line scoring data,
    but it remains metadata rather than a model input.
    """
    from training_pipeline.data import prepare_dataset

    csv_path = _write_intermediate_csv(tmp_path)
    source = pd.read_csv(csv_path, dtype={"GAME_ID": str})
    sidecar = source[["GAME_ID", "TIME_TO_MATCH_MIN"]].copy()
    sidecar["ODDS_CLOSING_TOTAL_LINE_bet365"] = 221.5
    sidecar_path = tmp_path / "single_horizon_scoring.csv"
    sidecar.to_csv(sidecar_path, index=False)

    prepared = prepare_dataset(
        _prepared_config(
            tmp_path,
            csv_path,
            snapshot_minutes=720,
            scoring_csv_path=sidecar_path,
        )
    )

    assert (prepared.df_full["TIME_TO_MATCH_MIN"] == 720).all()
    assert "TIME_TO_MATCH_MIN" not in prepared.feature_names
    assert "TIME_TO_MATCH_MIN" not in prepared.X.columns
    assert "ODDS_CLOSING_TOTAL_LINE_bet365" in prepared.df_full.columns
    assert "ODDS_CLOSING_TOTAL_LINE_bet365" not in prepared.X.columns


# --- the CLI ----------------------------------------------------------------


def test_the_cli_lets_the_config_decide_whether_to_ship_a_model():
    """It used to pass save_model=True whenever the flag was absent, overriding
    refit.train_production_model: false. Every campaign runner works around it
    by always passing --no-save-model."""
    from training_pipeline.cli import _build_arg_parser

    parser = _build_arg_parser()
    assert parser.parse_args(["c.yaml"]).save_model is None
    assert parser.parse_args(["c.yaml", "--no-save-model"]).save_model is False
    assert parser.parse_args(["c.yaml", "--save-model"]).save_model is True
