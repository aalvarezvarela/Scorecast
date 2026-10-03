"""The version registry: how versions chain and which columns each one owns."""

import pytest
from nba_ou.config.dataset_versions import (
    BASE_SCHEMA_VERSION,
    TRAINING_DATA_SCHEMA_VERSION,
)
from nba_ou.create_training_data import schema_layers as sl
from nba_ou.create_training_data.schema_layers.contract import (
    check_declared_columns,
)

from training_pipeline.config import SNAPSHOT_COLUMN, DatasetType


def test_config_constant_is_the_newest_registered_version():
    assert TRAINING_DATA_SCHEMA_VERSION == sl.latest_version()
    assert sl.available_versions()[0] == BASE_SCHEMA_VERSION


def test_dataset_type_spellings_match_the_training_pipeline():
    assert sl.CLOSING_LINE == DatasetType.CLOSING_LINE.value
    assert sl.INTERMEDIATE_LINE == DatasetType.INTERMEDIATE_LINE.value
    assert sl.ROW_KEYS[sl.INTERMEDIATE_LINE][-1] == SNAPSHOT_COLUMN


def test_every_layer_chains_onto_the_one_before_it():
    versions = sl.available_versions()
    for older, newer in zip(versions, versions[1:], strict=False):
        assert sl.LAYERS[newer].parent == older
        assert sl.parse_version(newer) > sl.parse_version(older)


def test_versions_compare_numerically():
    assert sl.parse_version("2_10") > sl.parse_version("2_9")
    with pytest.raises(ValueError):
        sl.parse_version("2.6")


def test_chain_runs_forward_only_between_known_versions():
    assert sl.chain("2_5", "2_5") == []
    assert [layer.version for layer in sl.chain("2_5", "2_6")] == ["2_6"]
    with pytest.raises(ValueError, match="down"):
        sl.chain("2_6", "2_5")
    with pytest.raises(ValueError, match="not buildable"):
        sl.chain("2_4", "2_6")
    with pytest.raises(ValueError, match="not buildable"):
        sl.chain("2_5", "9_9")


@pytest.mark.parametrize("dataset_type", sl.DATASET_TYPES)
def test_declared_columns_are_unique_pregame_and_owned_by_one_version(dataset_type):
    seen: dict[str, str] = {}
    for version, layer in sl.LAYERS.items():
        check_declared_columns(layer, dataset_type)
        for column in layer.columns_for(dataset_type):
            assert (
                column not in seen
            ), f"{column} declared by {seen.get(column)} and {version}"
            seen[column] = version


def test_column_origin_names_the_introducing_version():
    starter = "STARTER_UNIQUE_LAST_5_GAMES_BEFORE_TEAM_HOME"
    origin = sl.column_origin(
        ["GAME_ID", starter, "LU_PROJ_TOTAL_BEFORE"], sl.CLOSING_LINE
    )
    assert origin == {"GAME_ID": "2_5", starter: "2_6", "LU_PROJ_TOTAL_BEFORE": "2_6"}
    # The lineup family is closing-only in 2_6.
    assert sl.column_origin(["LU_PROJ_TOTAL_BEFORE"], sl.INTERMEDIATE_LINE) == {
        "LU_PROJ_TOTAL_BEFORE": "2_5"
    }


def test_required_inputs_put_the_row_keys_first():
    closing = sl.required_input_columns("2_5", "2_6", sl.CLOSING_LINE)
    intermediate = sl.required_input_columns("2_5", "2_6", sl.INTERMEDIATE_LINE)
    assert closing[:1] == ["GAME_ID"]
    assert intermediate[:2] == ["GAME_ID", SNAPSHOT_COLUMN]
    assert sl.required_input_columns("2_5", "2_5", sl.CLOSING_LINE) == ["GAME_ID"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("closing_line_data_2_5_20261003.parquet", sl.CLOSING_LINE),
        ("training_data_2_5_20260704.csv", sl.CLOSING_LINE),
        ("training_data_2_6_20260704.parquet", sl.CLOSING_LINE),
        ("historical_training_data_2_5_20260215.parquet", sl.CLOSING_LINE),
        ("intermediate_line_data_2_5_20260613.csv", sl.INTERMEDIATE_LINE),
    ],
)
def test_dataset_type_is_read_from_the_filename(name, expected):
    assert sl.dataset_type_from_filename(name) == expected


def test_unknown_filenames_are_refused():
    with pytest.raises(ValueError, match="dataset type"):
        sl.dataset_type_from_filename("whatever_2_5_20260704.csv")
