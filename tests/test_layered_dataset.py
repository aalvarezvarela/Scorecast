"""Upgrading a dataset file: parent untouched, new columns appended, lineage recorded."""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from nba_ou.create_training_data.schema_layers import registry
from nba_ou.create_training_data.schema_layers.base import (
    CLOSING_LINE,
    INTERMEDIATE_LINE,
    SchemaLayer,
)
from nba_ou.create_training_data.schema_layers.manifest import read_manifest

from training_pipeline import layered_dataset as ld
from training_pipeline.data import compute_file_checksum, load_raw_training_csv
from training_pipeline.parquet_dataset import (
    convert_csv_to_parquet,
    write_training_dataset,
)

COLUMNS = ("NEW_RATE_BEFORE", "NEW_COUNT_BEFORE_TEAM_HOME", "NEW_RESIDUAL_BEFORE")


def _new_columns(inputs, ctx, dataset_type):
    game = inputs["GAME_ID"].astype(str).str[-1].astype(int)
    return pd.DataFrame(
        {
            "NEW_RATE_BEFORE": game / 3,
            "NEW_COUNT_BEFORE_TEAM_HOME": np.where(game == 2, np.nan, game * 2.0),
            # "RESIDUAL" contains "ID". Loaders once read such names as text;
            # identifiers are now declared by exact name, so this stays numeric.
            "NEW_RESIDUAL_BEFORE": np.where(game == 3, np.nan, game / 7),
        },
        index=inputs.index,
    )


@pytest.fixture(autouse=True)
def stub_2_6(monkeypatch):
    layer = SchemaLayer(
        version="2_6",
        parent="2_5",
        summary="stub",
        columns={CLOSING_LINE: COLUMNS, INTERMEDIATE_LINE: COLUMNS},
        requires=("GAME_DATE",),
        build=_new_columns,
    )
    monkeypatch.setitem(registry.LAYERS, "2_6", layer)


def _frame():
    return pd.DataFrame(
        {
            "TEAM_ID_TEAM_HOME": ["1610612737", "1610612738", "1610612739"],
            "GAME_ID": ["0022500001", "0022500002", "0022500003"],
            "GAME_DATE": ["2025-10-21", "2025-10-22", "2025-10-22"],
            "TOTAL_POINTS": [221, 230, 199],
            "ODD_FLOAT_BEFORE": [0.1 + 0.2, 1 / 3, np.nan],
            "LABEL": ["a,b", "plain", None],
        }
    )


@pytest.fixture
def parent(tmp_path):
    path = tmp_path / "closing_line_data_2_5_20260630.parquet"
    write_training_dataset(_frame(), path)
    return path


@pytest.fixture
def parent_csv(tmp_path):
    path = tmp_path / "legacy" / "training_data_2_5_20260630.csv"
    path.parent.mkdir()
    _frame().to_csv(path, index=False)
    return path


def test_a_parquet_parent_is_untouched_and_gains_the_new_columns(parent):
    before = compute_file_checksum(parent)
    out = ld.build_schema_version_file(parent, to_version="2_6")
    assert out.name == "closing_line_data_2_6_20260630.parquet"
    assert compute_file_checksum(parent) == before

    original = pq.read_table(parent)
    copy = pq.read_table(out, columns=original.column_names)
    assert copy.equals(original)  # the parent's arrays, bit for bit

    upgraded = load_raw_training_csv(out)
    loaded_parent = load_raw_training_csv(parent)
    pd.testing.assert_frame_equal(upgraded[loaded_parent.columns], loaded_parent)
    assert list(upgraded.columns) == [*loaded_parent.columns, *COLUMNS]
    assert upgraded["NEW_RATE_BEFORE"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert upgraded["NEW_COUNT_BEFORE_TEAM_HOME"].isna().tolist() == [
        False,
        True,
        False,
    ]
    assert upgraded["NEW_RESIDUAL_BEFORE"].dtype == float


def test_the_manifest_records_lineage_and_checksums(parent):
    out = ld.build_schema_version_file(parent, to_version="2_6")
    manifest = read_manifest(out)
    assert manifest["schema_version"] == "2_6"
    assert manifest["dataset_type"] == CLOSING_LINE
    assert manifest["built_from_version"] == "2_5"
    assert manifest["layers_applied"] == ["2_6"]
    assert manifest["columns_added"] == {"2_6": list(COLUMNS)}
    assert manifest["parent"] == {
        "filename": parent.name,
        "checksum": compute_file_checksum(parent),
        "schema_version": "2_5",
    }
    assert manifest["files"] == {out.name: compute_file_checksum(out)}
    assert (manifest["n_rows"], manifest["n_columns"]) == (3, 9)
    sidecar = out.with_name("closing_line_data_2_6_20260630.manifest.json")
    assert json.loads(sidecar.read_text()) == manifest

    metadata = pq.read_schema(out).metadata
    assert metadata[ld.SCHEMA_VERSION_KEY] == b"2_6"
    assert metadata[ld.PARENT_CHECKSUM_KEY] == compute_file_checksum(parent).encode()


def test_a_legacy_csv_parent_becomes_a_parquet_file_like_its_converted_copy(
    parent_csv, tmp_path
):
    from_csv = ld.build_schema_version_file(parent_csv, to_version="2_6")
    assert from_csv.name == "training_data_2_6_20260630.parquet"
    converted = convert_csv_to_parquet(
        parent_csv, tmp_path / "training_data_2_5_20260630.parquet"
    )
    from_parquet = ld.build_schema_version_file(
        converted, to_version="2_6", out_dir=tmp_path / "via_parquet"
    )
    pd.testing.assert_frame_equal(
        load_raw_training_csv(from_csv), load_raw_training_csv(from_parquet)
    )


def test_an_intermediate_parent_keys_on_game_and_snapshot(tmp_path):
    path = tmp_path / "intermediate_line_data_2_5_20260630.parquet"
    frame = pd.DataFrame(
        {
            "GAME_ID": ["0022500001"] * 2 + ["0022500002"],
            "TIME_TO_MATCH_MIN": [60, 0, 0],
            "GAME_DATE": ["2025-10-21", "2025-10-21", "2025-10-22"],
        }
    )
    write_training_dataset(frame, path)
    out = ld.build_schema_version_file(path, to_version="2_6")
    assert out.name == "intermediate_line_data_2_6_20260630.parquet"
    assert read_manifest(out)["dataset_type"] == INTERMEDIATE_LINE
    assert load_raw_training_csv(out)["NEW_RATE_BEFORE"].tolist() == pytest.approx(
        [1 / 3, 1 / 3, 2 / 3]
    )


def test_a_parent_whose_manifest_disagrees_with_its_name_is_refused(parent):
    parent.with_name("closing_line_data_2_5_20260630.manifest.json").write_text(
        json.dumps({"schema_version": "2_4", "dataset_type": CLOSING_LINE})
    )
    with pytest.raises(ValueError, match="manifest says 2_4"):
        ld.build_schema_version_file(parent, to_version="2_6")


def test_upgrading_to_the_same_version_is_refused(parent):
    with pytest.raises(ValueError, match="already 2_5"):
        ld.build_schema_version_file(parent, to_version="2_5")


@pytest.mark.parametrize(
    ("name", "version", "renamed"),
    [
        (
            "closing_line_data_2_5_20261003.parquet",
            "2_6",
            "closing_line_data_2_6_20261003.parquet",
        ),
        (
            "intermediate_line_data_2_5_20261003_without_injury_reports.parquet",
            "2_7",
            "intermediate_line_data_2_7_20261003_without_injury_reports.parquet",
        ),
        ("training_data_2_5_20260704.csv", "2_6", "training_data_2_6_20260704.parquet"),
    ],
)
def test_layered_filenames_swap_only_the_version(name, version, renamed):
    assert ld.layered_filename(name, version) == renamed
    assert ld.schema_version_from_filename(renamed) == version


def test_a_custom_output_name_is_checked_before_the_build():
    ld.check_layerable_output(
        "x/closing_line_data_2_5_20260630.parquet", base_version="2_5"
    )
    with pytest.raises(ValueError, match="dataset type"):
        ld.check_layerable_output("x/foo.parquet", base_version="2_5")
    with pytest.raises(ValueError, match="names version 2_6"):
        ld.check_layerable_output(
            "closing_line_data_2_6_20260630.parquet", base_version="2_5"
        )
