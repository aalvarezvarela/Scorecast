"""Tests for per-snapshot re-scoring of an already-trained model's predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_pipeline.betting import DECIMAL_ODDS_MINUS_110, evaluate_betting
from training_pipeline.snapshot_scoring import (
    POOLED_LABEL,
    SnapshotAlignmentError,
    _lookup_by_position,
    _verify_alignment,
    format_snapshot_table,
    save_snapshot_report,
    score_by_snapshot,
)

SNAPSHOTS = (30, 720)


def make_inputs(n_games: int = 40, *, snapshots: tuple[int, ...] = SNAPSHOTS):
    """One deterministic 'run': every game appears once per snapshot.

    The model is right on the first half of the games and wrong on the second,
    identically at every snapshot, so the expected win rate is exactly 0.5 in
    every group and in the pool. That makes the COUNTS the only thing that
    changes between groups, which is what these tests are about.
    """
    rows = []
    for snapshot in snapshots:
        for game in range(n_games):
            correct = game < n_games // 2
            line = 220.0
            actual = 230.0 if correct else 210.0
            rows.append(
                {
                    "game_id": f"g{game:04d}",
                    "snapshot": snapshot,
                    "predicted_edge": 5.0,
                    "line": line,
                    "actual_total": actual,
                }
            )
    return pd.DataFrame(rows)


def score(frame: pd.DataFrame, **kwargs):
    return score_by_snapshot(
        snapshot=frame["snapshot"],
        predicted_edge=frame["predicted_edge"].to_numpy(dtype=float),
        actual_total=frame["actual_total"].to_numpy(dtype=float),
        line=frame["line"].to_numpy(dtype=float),
        game_id=frame["game_id"],
        flat_decimal_odds=DECIMAL_ODDS_MINUS_110,
        **kwargs,
    )


def test_one_row_per_snapshot_plus_a_pooled_row():
    table = score(make_inputs())
    assert list(table["snapshot"]) == [30, 720, POOLED_LABEL]


def test_each_snapshot_counts_games_not_rows():
    table = score(make_inputs(n_games=40)).set_index("snapshot")
    for snapshot in SNAPSHOTS:
        assert table.loc[snapshot, "n_bets"] == 40
        assert table.loc[snapshot, "n_snapshots"] == 1


def test_pooled_row_is_flagged_as_multi_snapshot():
    """The whole point: the pooled row counts one game once per snapshot."""
    table = score(make_inputs(n_games=40)).set_index("snapshot")
    pooled = table.loc[POOLED_LABEL]
    assert pooled["n_bets"] == 80          # 40 games x 2 snapshots
    assert pooled["n_snapshots"] == 2


def test_pooled_row_reports_no_interval_and_no_significance():
    """Correlated repeats break the binomial assumption in the
    anti-conservative direction, so the pooled row must decline to make a
    claim rather than making an overconfident one."""
    table = score(make_inputs(n_games=40)).set_index("snapshot")
    pooled = table.loc[POOLED_LABEL]
    assert pooled["win_rate"] is not None      # the point estimate is fine
    assert pd.isna(pooled["win_rate_ci_low"])
    assert pd.isna(pooled["win_rate_ci_high"])
    assert pd.isna(pooled["is_significant"])

    per_snapshot = table.loc[30]
    assert not pd.isna(per_snapshot["win_rate_ci_low"])
    assert per_snapshot["is_significant"] in (True, False)


def test_n_games_is_absent_when_no_game_ids_are_available():
    """The normal pipeline path. GAME_ID does not survive cleaning and must
    not, so nothing in the report may depend on it."""
    frame = make_inputs(n_games=20)
    table = score_by_snapshot(
        snapshot=frame["snapshot"],
        predicted_edge=frame["predicted_edge"].to_numpy(dtype=float),
        actual_total=frame["actual_total"].to_numpy(dtype=float),
        line=frame["line"].to_numpy(dtype=float),
        flat_decimal_odds=DECIMAL_ODDS_MINUS_110,
    )
    assert table["n_games"].isna().all()
    # The trustworthiness flag still works without it.
    assert set(table["n_snapshots"]) == {1, 2}


def test_a_snapshot_slice_matches_evaluate_betting_called_directly():
    """The grouping changes nothing about how a bet is scored."""
    frame = make_inputs()
    table = score(frame).set_index("snapshot")

    subset = frame[frame["snapshot"] == 720]
    direct = evaluate_betting(
        predicted_edge=subset["predicted_edge"].to_numpy(dtype=float),
        actual_total=subset["actual_total"].to_numpy(dtype=float),
        line=subset["line"].to_numpy(dtype=float),
        min_edge=0.0,
        flat_decimal_odds=DECIMAL_ODDS_MINUS_110,
    )
    assert table.loc[720, "n_bets"] == direct.n_bets
    assert table.loc[720, "win_rate"] == pytest.approx(direct.win_rate)
    assert table.loc[720, "roi"] == pytest.approx(direct.roi)


def test_snapshots_can_disagree():
    """A real per-horizon difference must survive into the table."""
    frame = make_inputs(n_games=40)
    # Make every 720 bet a loser, leaving 30 untouched.
    frame.loc[frame["snapshot"] == 720, "actual_total"] = 210.0
    table = score(frame).set_index("snapshot")
    assert table.loc[30, "win_rate"] == pytest.approx(0.5)
    assert table.loc[720, "win_rate"] == pytest.approx(0.0)


def test_min_edge_is_applied_per_snapshot():
    frame = make_inputs(n_games=40)
    frame.loc[frame["snapshot"] == 720, "predicted_edge"] = 0.25
    table = score(frame, min_edge=1.0).set_index("snapshot")
    assert table.loc[30, "n_bets"] == 40
    assert table.loc[720, "n_bets"] == 0
    # Candidates are still counted: "no bet cleared the threshold" is not the
    # same as "there was nothing to bet on".
    assert table.loc[720, "n_candidates"] == 40


def test_pooled_row_can_be_suppressed():
    table = score(make_inputs(), include_pooled=False)
    assert POOLED_LABEL not in set(table["snapshot"])


def test_misaligned_lengths_raise_rather_than_broadcast():
    frame = make_inputs(n_games=10)
    with pytest.raises(SnapshotAlignmentError, match="row-aligned"):
        score_by_snapshot(
            snapshot=frame["snapshot"].iloc[:5],
            predicted_edge=frame["predicted_edge"].to_numpy(dtype=float),
            actual_total=frame["actual_total"].to_numpy(dtype=float),
            line=frame["line"].to_numpy(dtype=float),
            flat_decimal_odds=DECIMAL_ODDS_MINUS_110,
        )


def test_lookup_rejects_a_missing_snapshot_column():
    source = pd.DataFrame({"GAME_DATE": pd.to_datetime(["2026-01-01"])})
    with pytest.raises(SnapshotAlignmentError, match="TIME_TO_MATCH_MIN"):
        _lookup_by_position(source, np.array([0]), "TIME_TO_MATCH_MIN")


def test_lookup_rejects_out_of_range_positions():
    source = pd.DataFrame({"TIME_TO_MATCH_MIN": [30, 720]})
    with pytest.raises(SnapshotAlignmentError, match="out of range"):
        _lookup_by_position(source, np.array([0, 5]), "TIME_TO_MATCH_MIN")


def test_alignment_check_catches_a_reordered_source():
    """A silent misalignment would attribute every prediction to the wrong
    snapshot, so it must raise instead of returning plausible numbers."""
    source = pd.DataFrame(
        {"GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])}
    )
    predictions = pd.DataFrame(
        {"GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-03"])}
    )
    # Positions 0 and 2 line up; 0 and 1 do not.
    _verify_alignment(predictions, source, np.array([0, 2]), date_col="GAME_DATE")
    with pytest.raises(SnapshotAlignmentError, match="do not line up"):
        _verify_alignment(predictions, source, np.array([0, 1]), date_col="GAME_DATE")


def test_report_is_written_as_csv(tmp_path):
    table = score(make_inputs())
    written = save_snapshot_report({"holdout": table}, tmp_path)
    assert written["holdout"].exists()
    assert pd.read_csv(written["holdout"]).shape[0] == len(table)


def test_format_is_readable_without_a_game_id():
    frame = make_inputs()
    table = score_by_snapshot(
        snapshot=frame["snapshot"],
        predicted_edge=frame["predicted_edge"].to_numpy(dtype=float),
        actual_total=frame["actual_total"].to_numpy(dtype=float),
        line=frame["line"].to_numpy(dtype=float),
        flat_decimal_odds=DECIMAL_ODDS_MINUS_110,
    )
    assert table["n_games"].isna().all()
    assert "snapshot" in format_snapshot_table(table)


def test_pushes_are_excluded_from_the_win_rate_per_snapshot():
    frame = make_inputs(n_games=10)
    frame.loc[frame["snapshot"] == 30, "actual_total"] = 220.0  # lands on the line
    table = score(frame).set_index("snapshot")
    assert table.loc[30, "n_pushes"] == 10
    assert table.loc[30, "n_wins"] == 0
    assert table.loc[30, "n_losses"] == 0


# --------------------------------------------------------------------------
# Integration: the adapters that read a finished ExperimentResult.
#
# These are the tests whose absence let two real defects ship. Every earlier
# test called score_by_snapshot directly with hand-built arrays, so nothing
# exercised the join from a prediction frame back to its source rows -- which
# is where both bugs lived.
# --------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

from training_pipeline.cli import load_config  # noqa: E402
from training_pipeline.snapshot_scoring import (  # noqa: E402
    build_snapshot_report,
    cv_snapshot_metrics,
    holdout_snapshot_metrics,
)

REPO_CONFIG = (
    "experiments/archived/intermediate_line_2026_08/pooled_line_error.yaml"
)


@dataclass
class FakeWalkForward:
    predictions: pd.DataFrame


@dataclass
class FakeCvBetting:
    predictions: pd.DataFrame


@dataclass
class FakeResult:
    config: object
    df_dev: pd.DataFrame
    df_test: pd.DataFrame
    walk_forward_result: object | None = None
    cv_betting: object | None = None


def source_frame(n_games=12, snapshots=SNAPSHOTS):
    """Mimics df_dev / df_test AFTER cleaning: note there is no GAME_ID."""
    rows = []
    for game in range(n_games):
        for snapshot in snapshots:
            rows.append(
                {
                    "GAME_DATE": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=game // 4),
                    "TIME_TO_MATCH_MIN": snapshot,
                    "TOTAL_POINTS": 220.0 + game,
                    "ODDS_TOTAL_LINE_bet365": 218.0 + game,
                }
            )
    return pd.DataFrame(rows)


def prediction_frame(source, positions, *, position_col, date_col):
    picked = source.iloc[positions]
    return pd.DataFrame(
        {
            position_col: positions,
            date_col: picked["GAME_DATE"].to_numpy(),
            "TOTAL_POINTS": picked["TOTAL_POINTS"].to_numpy(),
            "target_line": picked["ODDS_TOTAL_LINE_bet365"].to_numpy(),
            "predicted_edge": np.full(len(positions), 3.0),
            "selection_score": np.full(len(positions), 3.0),
        }
    )


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_CONFIG)


def test_holdout_adapter_groups_by_snapshot(config):
    """Regression: the walk-forward frame names its date column "date", not
    GAME_DATE. The alignment check used to look only for the configured name
    and RETURN SILENTLY when absent, so the holdout join was never verified."""
    source = source_frame()
    positions = np.arange(len(source))
    result = FakeResult(
        config=config,
        df_dev=source,
        df_test=source,
        walk_forward_result=FakeWalkForward(
            prediction_frame(
                source, positions, position_col="row_in_test_final", date_col="date"
            )
        ),
    )
    table = holdout_snapshot_metrics(result)
    assert set(table["snapshot"]) == {*SNAPSHOTS, POOLED_LABEL}
    assert table.set_index("snapshot").loc[30, "n_rows"] == 12


def test_cv_adapter_groups_by_snapshot(config):
    source = source_frame()
    positions = np.arange(len(source))
    result = FakeResult(
        config=config,
        df_dev=source,
        df_test=source,
        cv_betting=FakeCvBetting(
            prediction_frame(
                source, positions, position_col="row_in_dev", date_col="GAME_DATE"
            )
        ),
    )
    table = cv_snapshot_metrics(result)
    assert set(table["snapshot"]) == {*SNAPSHOTS, POOLED_LABEL}


def test_report_contains_both_tables(config):
    source = source_frame()
    positions = np.arange(len(source))
    result = FakeResult(
        config=config,
        df_dev=source,
        df_test=source,
        walk_forward_result=FakeWalkForward(
            prediction_frame(
                source, positions, position_col="row_in_test_final", date_col="date"
            )
        ),
        cv_betting=FakeCvBetting(
            prediction_frame(
                source, positions, position_col="row_in_dev", date_col="GAME_DATE"
            )
        ),
    )
    assert set(build_snapshot_report(result)) == {"cv", "holdout"}


def test_a_within_date_swap_is_caught(config):
    """Regression: a date-only check passes any permutation inside one day,
    and a day here holds several games at several snapshots each."""
    source = source_frame()
    positions = np.arange(len(source))
    predictions = prediction_frame(
        source, positions, position_col="row_in_test_final", date_col="date"
    )
    # Rows are game-major (game0/snap30, game0/snap720, game1/snap30, ...) and
    # games 0-3 share a date, so rows 0 and 2 are DIFFERENT games on the SAME
    # day with different totals. Swapping rows 0 and 1 would prove nothing:
    # they are the two snapshots of one game and share a total.
    swapped = [0, 2]
    predictions.loc[swapped, "TOTAL_POINTS"] = predictions.loc[
        swapped[::-1], "TOTAL_POINTS"
    ].to_numpy()
    assert predictions.loc[0, "date"] == predictions.loc[2, "date"]

    result = FakeResult(
        config=config,
        df_dev=source,
        df_test=source,
        walk_forward_result=FakeWalkForward(predictions),
    )
    with pytest.raises(SnapshotAlignmentError, match="TOTAL_POINTS"):
        holdout_snapshot_metrics(result)


def test_an_unverifiable_join_raises_instead_of_passing_quietly(config):
    """A safety check that cannot run is a failure, not a pass."""
    source = source_frame()
    positions = np.arange(len(source))
    predictions = prediction_frame(
        source, positions, position_col="row_in_test_final", date_col="date"
    ).drop(columns=["date", "TOTAL_POINTS", "target_line"])

    result = FakeResult(
        config=config,
        df_dev=source,
        df_test=source,
        walk_forward_result=FakeWalkForward(predictions),
    )
    with pytest.raises(SnapshotAlignmentError, match="Cannot verify"):
        holdout_snapshot_metrics(result)


def test_adapters_return_none_when_the_run_has_no_such_result(config):
    source = source_frame()
    result = FakeResult(config=config, df_dev=source, df_test=source)
    assert holdout_snapshot_metrics(result) is None
    assert cv_snapshot_metrics(result) is None
    assert build_snapshot_report(result) == {}
