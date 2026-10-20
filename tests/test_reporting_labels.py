"""Tests for run labelling and the tuned-window readout.

Both protect against the same failure: a report that says two runs are the same
when they are not. A label that cannot distinguish a closing-line run from a
pooled-snapshot one puts them side by side on a chart as though the difference
were noise, and a tuned window read from ``config.json`` reports the fallback
rather than what Optuna picked -- neither raises anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training_pipeline.reporting import loaders, narrative, theme


def write_run(
    root: Path,
    name: str,
    *,
    strategy: str = "line_error_regressor",
    dataset_type: str = "closing_line",
    snapshot_minutes: int | None = None,
    season_floor: int = 2021,
    csv_path: str = "data/train_data/training_data_2_0_20260819.csv",
    train_games: int = 3500,
    train_games_choices: list[int] | None = None,
    selected_train_games: int | None = None,
    rows_per_game: float = 1.0,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({
        "prediction_strategy": strategy,
        "data": {
            "csv_path": csv_path,
            "dataset_type": dataset_type,
            "snapshot_minutes": snapshot_minutes,
            "season_year_floor": season_floor,
        },
        "cleaning": {"max_na_per_row": 80, "nan_threshold": 50.0},
        "walk_forward": {
            "train_games": train_games,
            "train_games_choices": train_games_choices,
            "strategy": "rolling_origin",
        },
        "optuna": {"n_trials": 200},
    }))
    (run_dir / "metadata.json").write_text(json.dumps({
        "dataset_type": dataset_type,
        "snapshot_minutes": snapshot_minutes,
        "train_games": selected_train_games or train_games,
        "train_games_choices": train_games_choices or [],
        "train_games_tuned": bool(train_games_choices),
        "rows_per_game": rows_per_game,
        "n_games": 6000,
        "cv_n_validation_games": 855,
        "holdout_n_games": 416,
    }))
    return run_dir


def runs_frame(root: Path, names: list[str]) -> pd.DataFrame:
    """The columns theme.prepare_runs needs, for runs written above."""
    rows = []
    for name in names:
        metadata = json.loads((root / name / "metadata.json").read_text())
        config = json.loads((root / name / "config.json").read_text())
        rows.append({
            "run_name": name,
            "run_dir": str(root / name),
            "source_root": "campaign",
            "source_path": str(root),
            "experiment_id": name[:8],
            "prediction_strategy": config["prediction_strategy"],
            "target_family": None,
            "train_games": metadata["train_games"],
            "created_at": "2026-08-21T00:00:00",
        })
    return pd.DataFrame(rows)


class TestLabelsDistinguishDatasets:
    def test_closing_pooled_and_single_snapshot_get_distinct_labels(
        self, tmp_path: Path
    ) -> None:
        """The snapshot campaign's three cells must not collide.

        All three share a strategy and can share a selected window, so before
        dataset_type and snapshot_minutes reached LABEL_FIELDS every one of
        them read as "line_error - 3500".
        """
        write_run(tmp_path, "closing", dataset_type="closing_line")
        write_run(
            tmp_path, "pooled", dataset_type="intermediate_line",
            snapshot_minutes=None,
            csv_path="data/train_data/intermediate_line_data_10snap.csv",
            rows_per_game=9.92,
        )
        write_run(
            tmp_path, "t360", dataset_type="intermediate_line",
            snapshot_minutes=360,
            csv_path="data/train_data/intermediate_line_data_10snap.csv",
        )
        runs = theme.prepare_runs(runs_frame(tmp_path, ["closing", "pooled", "t360"]))
        labels = dict(zip(runs["run_name"], runs["label"], strict=True))
        assert len(set(labels.values())) == 3, labels
        assert "closing" in labels["closing"]
        assert "T-360" in labels["t360"]

    def test_a_null_snapshot_still_counts_as_a_choice(self, tmp_path: Path) -> None:
        """Null IS the pooled setting, not a missing value.

        Closing-line runs also leave snapshot_minutes null, so treating null as
        absent leaves a single non-null value (360), which the discriminator
        test then rejects as "does not vary" -- and the T-360 token disappears.
        """
        write_run(tmp_path, "closing", dataset_type="closing_line")
        write_run(
            tmp_path, "t360", dataset_type="intermediate_line", snapshot_minutes=360,
        )
        assert "data.snapshot_minutes" in theme.NONE_IS_A_VALUE
        runs = theme.prepare_runs(runs_frame(tmp_path, ["closing", "t360"]))
        labels = dict(zip(runs["run_name"], runs["label"], strict=True))
        assert "T-360" in labels["t360"]

    def test_a_field_common_to_every_run_adds_no_token(self, tmp_path: Path) -> None:
        """Labels say what is UNUSUAL, so a shared setting must stay silent."""
        write_run(tmp_path, "a", dataset_type="closing_line", train_games=3000)
        write_run(tmp_path, "b", dataset_type="closing_line", train_games=3500)
        runs = theme.prepare_runs(runs_frame(tmp_path, ["a", "b"]))
        assert not any("closing" in label for label in runs["label"])


class TestDescribeRuns:
    def test_adds_the_short_panel_label_and_the_prose_one(self, tmp_path: Path) -> None:
        write_run(tmp_path, "a", dataset_type="closing_line")
        write_run(tmp_path, "b", dataset_type="intermediate_line", snapshot_minutes=360)
        described = theme.describe_runs(
            theme.prepare_runs(runs_frame(tmp_path, ["a", "b"]))
        )
        assert described["panel_label"].str.contains("line_error").all()
        assert described["label"].str.startswith("Predicts error vs").all()
        # The prose form is the longer of the two; a chart panel needs the other.
        assert (
            described["label"].str.len() > described["panel_label"].str.len()
        ).all()

    def test_rows_per_game_comes_from_metadata_not_a_default(
        self, tmp_path: Path
    ) -> None:
        """Defaulting this to 1.0 would grant a pooled run binomial intervals
        it does not qualify for, with nothing raised."""
        write_run(tmp_path, "pooled", rows_per_game=9.92)
        described = theme.describe_runs(
            theme.prepare_runs(runs_frame(tmp_path, ["pooled"]))
        )
        assert described.loc[0, "rows_per_game"] == 9.92

    def test_carries_the_dataset_and_horizon_columns(self, tmp_path: Path) -> None:
        write_run(tmp_path, "t360", dataset_type="intermediate_line",
                  snapshot_minutes=360)
        described = theme.describe_runs(
            theme.prepare_runs(runs_frame(tmp_path, ["t360"]))
        )
        assert described.loc[0, "dataset_type"] == "intermediate_line"
        assert described.loc[0, "snapshot_minutes"] == 360


class TestRunSpec:
    def _run(self, **overrides: object) -> pd.Series:
        base = {
            "strategy_short": "line_error", "train_games": 3500,
            "dataset_type": "closing_line", "snapshot_minutes": None,
            "rows_per_game": 1.0,
        }
        return pd.Series({**base, **overrides})

    def test_closing_pooled_and_single_horizon_read_differently(self) -> None:
        """A null snapshot means "pooled" on the intermediate dataset and "not
        applicable" on the closing one; only the pair tells them apart."""
        assert theme.horizon_text(self._run()) == "closing"
        assert theme.horizon_text(
            self._run(dataset_type="intermediate_line")
        ) == "pooled snapshots"
        assert theme.horizon_text(
            self._run(dataset_type="intermediate_line", snapshot_minutes=360)
        ) == "T-360"

    def test_flags_repeated_rows_per_game(self) -> None:
        assert "rows/game" in theme.run_spec(self._run(rows_per_game=9.92))
        assert "rows/game" not in theme.run_spec(self._run())

    def test_a_full_history_run_says_so(self) -> None:
        assert "full-history" in theme.window_text(self._run(train_games=float("nan")))


class TestConfigMatrix:
    def test_shows_only_fields_that_actually_differ(self, tmp_path: Path) -> None:
        write_run(tmp_path, "a", season_floor=2021)
        write_run(tmp_path, "b", season_floor=2020)
        matrix = loaders.config_matrix(runs_frame(tmp_path, ["a", "b"]))
        assert "data.season_year_floor" in matrix.index
        # Shared settings are not differences and must not pad the table.
        assert "cleaning.max_na_per_row" not in matrix.index

    def test_ignored_fields_are_dropped(self, tmp_path: Path) -> None:
        write_run(tmp_path, "a", season_floor=2021)
        write_run(tmp_path, "b", season_floor=2020)
        matrix = loaders.config_matrix(
            runs_frame(tmp_path, ["a", "b"]), ignore={"data.season_year_floor"}
        )
        assert matrix.empty

    def test_identical_runs_give_an_empty_frame(self, tmp_path: Path) -> None:
        write_run(tmp_path, "a")
        write_run(tmp_path, "b")
        assert loaders.config_matrix(runs_frame(tmp_path, ["a", "b"])).empty


class TestTunedWindowNarrative:
    def test_reports_an_untuned_window_as_a_config_fix(self, tmp_path: Path) -> None:
        write_run(tmp_path, "fixed", train_games_choices=None)
        messages = narrative.tuned_window(
            loaders.tuned_window_table(runs_frame(tmp_path, ["fixed"]))
        )
        assert any("NOT tuned" in message for message in messages)
        assert any("train_games_choices" in message for message in messages)

    def test_reports_a_grid_edge_separately_from_tuning(self, tmp_path: Path) -> None:
        write_run(tmp_path, "edge", train_games_choices=[2500, 3000, 4000],
                  selected_train_games=4000)
        messages = narrative.tuned_window(
            loaders.tuned_window_table(runs_frame(tmp_path, ["edge"]))
        )
        assert any("Censored by the search grid" in message for message in messages)
        assert not any("NOT tuned" in message for message in messages)

    def test_reports_repeated_rows_per_game(self, tmp_path: Path) -> None:
        write_run(tmp_path, "pooled", train_games_choices=[2500, 3000],
                  selected_train_games=3000, rows_per_game=9.92)
        messages = narrative.tuned_window(
            loaders.tuned_window_table(runs_frame(tmp_path, ["pooled"]))
        )
        assert any("Several rows per game" in message for message in messages)

    def test_a_clean_run_raises_none_of_the_three_warnings(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path, "clean", train_games_choices=[2500, 3000, 4000],
                  selected_train_games=3000)
        messages = narrative.tuned_window(
            loaders.tuned_window_table(runs_frame(tmp_path, ["clean"]))
        )
        assert len(messages) == 1
        assert "tuned by Optuna" in messages[0]


class TestTunedWindowTable:
    def test_reports_the_selected_window_not_the_config_fallback(
        self, tmp_path: Path
    ) -> None:
        """metadata.json is the record of what ran.

        The fallback here is 3500 and Optuna chose 2500. Reading
        walk_forward.train_games would report 3500 with no error anywhere.
        """
        write_run(
            tmp_path, "tuned", train_games=3500,
            train_games_choices=[2500, 3000, 3500, 4000], selected_train_games=2500,
        )
        table = loaders.tuned_window_table(runs_frame(tmp_path, ["tuned"]))
        assert table.loc[0, "selected"] == 2500
        assert bool(table.loc[0, "tuned"]) is True

    def test_flags_a_window_that_hit_the_edge_of_its_grid(self, tmp_path: Path) -> None:
        write_run(
            tmp_path, "edge", train_games_choices=[2500, 3000, 3500, 4000],
            selected_train_games=4000,
        )
        write_run(
            tmp_path, "interior", train_games_choices=[2500, 3000, 3500, 4000],
            selected_train_games=3000,
        )
        table = loaders.tuned_window_table(
            runs_frame(tmp_path, ["edge", "interior"])
        ).set_index("run_name")
        assert bool(table.loc["edge", "at_grid_edge"]) is True
        assert bool(table.loc["interior", "at_grid_edge"]) is False

    def test_an_untuned_run_is_never_flagged_as_at_the_edge(
        self, tmp_path: Path
    ) -> None:
        """With no grid there is no edge to sit on, and no tuning to report."""
        write_run(tmp_path, "fixed", train_games=3500, train_games_choices=None)
        table = loaders.tuned_window_table(runs_frame(tmp_path, ["fixed"]))
        assert bool(table.loc[0, "tuned"]) is False
        assert bool(table.loc[0, "at_grid_edge"]) is False
        assert table.loc[0, "choices"] == "—"

    def test_carries_rows_per_game_for_the_independence_check(
        self, tmp_path: Path
    ) -> None:
        write_run(tmp_path, "pooled", rows_per_game=9.92)
        table = loaders.tuned_window_table(runs_frame(tmp_path, ["pooled"]))
        assert table.loc[0, "rows_per_game"] == 9.92
