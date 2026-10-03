"""Training datasets are Parquet files that load exactly as the CSV builds did.

Datasets are written straight to Parquet by the builders, and older CSV builds
are converted. Either way the loaded frame must equal what reading the CSV gave
-- dtypes, index, missing values and all -- because every campaign's cleaning
and splits run on whatever load_raw_training_csv returns. "Close" would mean a
silently different run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training_pipeline.data import (
    filter_to_snapshot,
    load_raw_training_csv,
    read_dataset_columns,
)
from training_pipeline.parquet_dataset import (
    ParquetConversionError,
    convert_csv_to_parquet,
    write_training_dataset,
)

SNAPSHOT = "TIME_TO_MATCH_MIN"


def _write_intermediate_csv(path):
    rng = np.random.default_rng(0)
    rows = []
    for game in range(7):
        for minutes in (0, 60, 120):
            rows.append(
                {
                    # Leading zero: dtype inference would turn this into an int.
                    "GAME_ID": f"00221{game:05d}",
                    "GAME_DATE": f"2025-11-{game + 1:02d} 19:30:00",
                    SNAPSHOT: minutes,
                    "SEASON_YEAR": 2025,
                    "TEAM_NAME_TEAM_HOME": None if game == 3 else f"Team {game}",
                    "REF_ID": None if game == 5 else str(1000 + game),
                    "ODDS_SNAP_TOT_LINE": np.nan if game == 2 else 220.5 + rng.normal(),
                    "PTS_LAST_1": float(rng.integers(90, 130)),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def datasets(tmp_path):
    csv = tmp_path / "intermediate_line_data_2_5_20260101.csv"
    _write_intermediate_csv(csv)
    parquet = tmp_path / "intermediate_line_data_2_5_20260101.parquet"
    # 4-row groups, so every horizon is spread across several groups and the
    # file-position index is exercised across group boundaries.
    convert_csv_to_parquet(csv, parquet, row_group_size=4)
    return csv, parquet


def test_full_read_is_identical_to_the_csv(datasets):
    csv, parquet = datasets
    pd.testing.assert_frame_equal(
        load_raw_training_csv(parquet), load_raw_training_csv(csv), check_exact=True
    )


def test_single_horizon_read_matches_the_csv_path_index_included(datasets):
    """Parquet filters while reading; CSV reads everything and filter_to_snapshot
    subsets it. The pipeline must not be able to tell which one ran."""
    csv, parquet = datasets
    from_csv = filter_to_snapshot(
        load_raw_training_csv(csv), snapshot_col=SNAPSHOT, minutes=60
    )
    from_parquet = filter_to_snapshot(
        load_raw_training_csv(parquet, snapshot_col=SNAPSHOT, snapshot_minutes=60),
        snapshot_col=SNAPSHOT,
        minutes=60,
    )
    assert len(from_parquet) == 7
    pd.testing.assert_frame_equal(from_parquet, from_csv, check_exact=True)


def test_missing_text_stays_nan_not_none_and_ids_keep_leading_zeros(datasets):
    _, parquet = datasets
    df = load_raw_training_csv(parquet)
    assert df["GAME_ID"].iloc[0] == "0022100000"
    assert df["TEAM_NAME_TEAM_HOME"].isna().sum() == 3
    missing = df.loc[df["REF_ID"].isna(), "REF_ID"]
    assert len(missing) == 3
    assert all(isinstance(v, float) and np.isnan(v) for v in missing)  # not None
    assert str(df["GAME_DATE"].dtype) == "datetime64[ns]"


def test_an_absent_horizon_names_the_ones_that_exist(datasets):
    _, parquet = datasets
    with pytest.raises(ValueError, match=r"Horizons present .*\[0, 60, 120\]"):
        load_raw_training_csv(parquet, snapshot_col=SNAPSHOT, snapshot_minutes=90)


def test_column_reads_agree_including_missing_ids(datasets):
    csv, parquet = datasets
    for dtype in ({"REF_ID": str}, {"REF_ID": "string", "GAME_DATE": "string"}):
        columns = ["GAME_ID", "REF_ID", "GAME_DATE"]
        from_csv = read_dataset_columns(csv, columns=columns, dtype=dtype)
        from_parquet = read_dataset_columns(parquet, columns=columns, dtype=dtype)
        assert (
            from_parquet["REF_ID"].isna().sum() == from_csv["REF_ID"].isna().sum() == 3
        )
        assert "nan" not in set(from_parquet["REF_ID"].dropna())


def test_a_mixed_type_column_is_named_and_no_file_is_left(tmp_path):
    # Numbers and text in one column, as a builder (or low_memory CSV
    # inference) can leave behind.
    mixed = pd.DataFrame(
        {"GAME_DATE": pd.to_datetime(["2025-11-01"] * 2), "ODD": [1.5, "x"]}
    )
    with pytest.raises(ParquetConversionError, match="ODD"):
        write_training_dataset(mixed, tmp_path / "mixed.parquet", row_group_size=4)
    assert not list(tmp_path.glob("mixed.parquet*"))


# --- builders write Parquet directly ------------------------------------------

import training_pipeline.data as data_module  # noqa: E402
from training_pipeline.data import compute_file_checksum  # noqa: E402


def _builder_frame() -> pd.DataFrame:
    """Typed the way a builder's in-memory frame is, not the way a CSV read is."""
    return pd.DataFrame(
        {
            "GAME_ID": ["0022100001", "0022100002", "0022100003"],
            "TEAM_ID_TEAM_HOME": [1610612737, 1610612738, 1610612739],
            "GAME_DATE": [
                "2025-11-01 19:30:00",
                "2025-11-02 20:00:00",
                "2025-11-03 21:00:00",
            ],
            "SEASON_YEAR": [2025, 2025, 2025],
            "TEAM_NAME_TEAM_HOME": ["Team 1", None, "Team 3"],
            "IS_US_HOLIDAY_BEFORE": [0, 1, 0],
            "ODDS_book_total_line_mid_bet365": [220.5, np.nan, 231.0],
        },
        index=[10, 11, 12],
    )


def test_a_builder_frame_loads_as_its_csv_would_have(tmp_path):
    frame = _builder_frame()
    csv = tmp_path / "train.csv"
    frame.to_csv(csv, index=False)

    parquet = tmp_path / "train.parquet"
    write_training_dataset(frame, parquet, row_group_size=2)

    pd.testing.assert_frame_equal(
        load_raw_training_csv(parquet), load_raw_training_csv(csv), check_exact=True
    )


def test_writing_leaves_the_builder_frame_alone(tmp_path):
    frame = _builder_frame()
    before = frame.copy(deep=True)

    write_training_dataset(frame, tmp_path / "train.parquet")

    pd.testing.assert_frame_equal(frame, before)


def test_the_returned_checksum_is_the_file_s(tmp_path):
    parquet = tmp_path / "train.parquet"
    assert write_training_dataset(_builder_frame(), parquet) == compute_file_checksum(
        parquet
    )


def test_only_parquet_is_written(tmp_path):
    with pytest.raises(ValueError, match="Parquet"):
        write_training_dataset(_builder_frame(), tmp_path / "train.csv")


def test_converting_a_csv_build_writes_a_new_file_with_its_own_checksum(datasets):
    csv, parquet = datasets
    assert compute_file_checksum(parquet) != compute_file_checksum(csv)
    pd.testing.assert_frame_equal(
        load_raw_training_csv(parquet), load_raw_training_csv(csv), check_exact=True
    )


# --- a config reads exactly the file it names ---------------------------------


def _config(path, checksum):
    from training_pipeline.config import DataConfig, ExperimentConfig, TargetFamily

    return ExperimentConfig(
        experiment_name="reads_the_named_file",
        target_family=TargetFamily.LINE_ERROR,
        data=DataConfig(
            csv_path=path,
            expected_checksum=checksum,
            dataset_type="intermediate_line",
            snapshot_minutes=60,
        ),
    )


@pytest.mark.parametrize("named", ["csv", "parquet"])
def test_prepare_dataset_reads_the_file_the_config_names(datasets, monkeypatch, named):
    """No substitution: a config naming the CSV reads the CSV even with a
    Parquet copy beside it."""
    csv, parquet = datasets
    path = csv if named == "csv" else parquet
    read = []

    def spy(path, **kwargs):
        read.append(Path(path))
        raise RuntimeError("stop after the read")

    monkeypatch.setattr(data_module, "load_raw_training_csv", spy)
    with pytest.raises(RuntimeError, match="stop after the read"):
        data_module.prepare_dataset(_config(path, compute_file_checksum(path)))
    assert read == [path]


def test_a_wrong_pin_fails(datasets):
    _, parquet = datasets
    with pytest.raises(ValueError, match="checksum mismatch"):
        data_module.prepare_dataset(_config(parquet, "sha256:0000000000000000"))


def test_a_parquet_scoring_sidecar_joins_like_a_csv_one(tmp_path):
    frame = pd.DataFrame(
        {"GAME_ID": ["0022100001", "0022100002"], SNAPSHOT: [60, 60], "X": [1.0, 2.0]}
    )
    sidecar = pd.DataFrame(
        {
            "GAME_ID": ["0022100002", "0022100001"],
            SNAPSHOT: [60, 60],
            "CLOSE": [9.0, 8.0],
        }
    )
    sidecar.to_csv(tmp_path / "scoring.csv", index=False)
    write_training_dataset(sidecar, tmp_path / "scoring.parquet")

    joined = [
        data_module.load_scoring_sidecar(
            frame, csv_path=path, game_id_col="GAME_ID", snapshot_col=SNAPSHOT
        )
        for path in (tmp_path / "scoring.csv", tmp_path / "scoring.parquet")
    ]

    assert joined[0][1] == joined[1][1] == ["CLOSE"]
    assert joined[1][0]["CLOSE"].tolist() == [8.0, 9.0]
    pd.testing.assert_frame_equal(joined[0][0], joined[1][0], check_exact=True)


# --- standard dataset names ---------------------------------------------------


def test_dataset_names_carry_kind_schema_and_limit_date():
    from nba_ou.config.dataset_versions import training_dataset_filename

    assert (
        training_dataset_filename("closing", "2026-10-03", schema_version="2_5")
        == "closing_line_data_2_5_20261003.parquet"
    )
    assert (
        training_dataset_filename("intermediate", "2026-10-03", schema_version="2_5")
        == "intermediate_line_data_2_5_20261003.parquet"
    )
    assert training_dataset_filename(
        "closing", "2026-10-03", variant="without_injury_reports", schema_version="2_5"
    ) == ("closing_line_data_2_5_20261003_without_injury_reports.parquet")
    with pytest.raises(ValueError, match="kind"):
        training_dataset_filename("pooled", "2026-10-03")


def test_the_registry_reads_the_schema_back_from_a_standard_name():
    from nba_ou.config.dataset_versions import training_dataset_filename

    from training_pipeline.registry import parse_schema_version

    for kind in ("closing", "intermediate"):
        name = training_dataset_filename(kind, "2026-10-03", schema_version="2_5")
        assert parse_schema_version(name) == "2_5"
        assert (
            parse_schema_version(name.replace(".parquet", "_scoring.parquet")) == "2_5"
        )
