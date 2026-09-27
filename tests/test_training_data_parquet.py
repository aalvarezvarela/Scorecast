"""A Parquet copy of a training dataset must load as the identical frame.

The contract is exact equality with the CSV path -- dtypes, index, missing
values and all -- because every campaign's cleaning and splits run on whatever
load_raw_training_csv returns. "Close" would mean a silently different run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training_pipeline.parquet_dataset as parquet_dataset
from training_pipeline.data import (
    filter_to_snapshot,
    load_raw_training_csv,
    read_dataset_columns,
)
from training_pipeline.parquet_dataset import (
    ParquetConversionError,
    convert_csv_to_parquet,
    finish_dataset_output,
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
        assert from_parquet["REF_ID"].isna().sum() == from_csv["REF_ID"].isna().sum() == 3
        assert "nan" not in set(from_parquet["REF_ID"].dropna())


def test_a_mixed_type_column_is_named_and_no_file_is_left(tmp_path, monkeypatch):
    csv = tmp_path / "mixed.csv"
    csv.write_text("GAME_DATE,ODD\n2025-11-01,1.5\n")
    # read_csv would infer a clean type from this file, so hand the converter
    # the frame low_memory inference can leave behind: numbers and text mixed.
    mixed = pd.DataFrame(
        {"GAME_DATE": pd.to_datetime(["2025-11-01"] * 2), "ODD": [1.5, "x"]}
    )
    monkeypatch.setattr(parquet_dataset, "load_raw_training_csv", lambda *a, **k: mixed)
    with pytest.raises(ParquetConversionError, match="ODD"):
        convert_csv_to_parquet(csv, tmp_path / "mixed.parquet", row_group_size=4)
    assert not list(tmp_path.glob("mixed.parquet*"))


@pytest.mark.parametrize(
    ("output_format", "csv_kept", "parquet_written"),
    [("csv", True, False), ("parquet", False, True), ("both", True, True)],
)
def test_builder_output_formats(tmp_path, output_format, csv_kept, parquet_written):
    csv = tmp_path / "training_data_2_5_20260101.csv"
    _write_intermediate_csv(csv)
    reference = load_raw_training_csv(csv)

    pinned = finish_dataset_output(csv, output_format=output_format)

    assert csv.exists() is csv_kept
    assert csv.with_suffix(".parquet").exists() is parquet_written
    assert pinned == (csv.with_suffix(".parquet") if parquet_written else csv)
    # Whatever the build leaves behind loads as the frame the CSV described.
    pd.testing.assert_frame_equal(load_raw_training_csv(pinned), reference, check_exact=True)


def test_unknown_output_format_is_refused(tmp_path):
    with pytest.raises(ValueError, match="output_format"):
        finish_dataset_output(tmp_path / "x.csv", output_format="feather")


# --- a config naming the CSV reads its verified Parquet copy ---------------

import training_pipeline.data as data_module  # noqa: E402
from training_pipeline.data import (  # noqa: E402
    compute_file_checksum,
    resolve_dataset_source,
)


def test_pinned_csv_resolves_to_its_verified_copy(datasets):
    csv, parquet = datasets
    pinned = compute_file_checksum(csv)

    source = resolve_dataset_source(csv, expected_checksum=pinned)

    assert source.read_path == parquet
    # The identity stays the CSV's: registry specs and run metadata remain
    # comparable with every config that pinned it.
    assert source.checksum == pinned


def test_the_csv_can_be_deleted_once_a_verified_copy_exists(datasets):
    csv, parquet = datasets
    pinned = compute_file_checksum(csv)
    reference = load_raw_training_csv(csv)
    csv.unlink()

    source = resolve_dataset_source(csv, expected_checksum=pinned)

    assert source.read_path == parquet
    pd.testing.assert_frame_equal(
        load_raw_training_csv(source.read_path), reference, check_exact=True
    )


def test_a_copy_of_different_bytes_is_never_substituted(datasets):
    """A CSV regenerated in place after conversion: the copy is stale."""
    csv, _ = datasets
    csv.write_text(csv.read_text().replace("Team 1", "Team 9"))
    regenerated = compute_file_checksum(csv)

    assert resolve_dataset_source(csv, expected_checksum=regenerated).read_path == csv
    assert resolve_dataset_source(csv, expected_checksum=None).read_path == csv


def test_an_unpinned_config_uses_the_copy_only_while_the_csv_matches(datasets):
    csv, parquet = datasets
    assert resolve_dataset_source(csv, expected_checksum=None).read_path == parquet


def test_a_copy_without_provenance_is_ignored(tmp_path):
    csv = tmp_path / "training_data_2_5_20260101.csv"
    _write_intermediate_csv(csv)
    load_raw_training_csv(csv).to_parquet(csv.with_suffix(".parquet"), index=False)

    source = resolve_dataset_source(csv, expected_checksum=compute_file_checksum(csv))

    assert source.read_path == csv


def test_a_wrong_pin_still_fails_on_the_csv(datasets):
    csv, _ = datasets
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_dataset_source(csv, expected_checksum="sha256:0000000000000000")


def test_prepare_dataset_reads_the_copy(datasets, monkeypatch):
    from training_pipeline.config import DataConfig, ExperimentConfig, TargetFamily

    csv, parquet = datasets
    config = ExperimentConfig(
        experiment_name="reads_the_copy",
        target_family=TargetFamily.LINE_ERROR,
        data=DataConfig(
            csv_path=csv,
            expected_checksum=compute_file_checksum(csv),
            dataset_type="intermediate_line",
            snapshot_minutes=60,
        ),
    )
    read = []

    def spy(path, **kwargs):
        read.append(Path(path))
        raise RuntimeError("stop after the read")

    monkeypatch.setattr(data_module, "load_raw_training_csv", spy)
    with pytest.raises(RuntimeError, match="stop after the read"):
        data_module.prepare_dataset(config)
    assert read == [parquet]
