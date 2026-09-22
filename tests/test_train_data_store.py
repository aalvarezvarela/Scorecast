"""The S3 training-data layout and the sweep that produces it.

The interesting case is not the happy path but what the sweep REFUSES to
move: the prefix is also an offered home for unrelated archives, and a rule of
"anything not named like a version" would have swallowed them.
"""

import pytest
from nba_ou.create_training_data.train_data_store import (
    LEGACY_SCHEMA_VERSION,
    TRAIN_DATA_PREFIX,
    historical_filename,
    is_unversioned_key,
    train_data_key,
)

from scripts.organize_s3_train_data import plan_moves
from tests.fake_s3 import FakeS3Client

# --- layout ----------------------------------------------------------------


def test_the_key_carries_the_version_as_its_first_segment():
    assert (
        train_data_key(schema_version="2_5", filename="x.parquet")
        == "train_data/2_5/x.parquet"
    )


def test_a_version_that_is_not_a_version_is_refused():
    """The segment is the whole point of the layout, so a typo must not
    silently create a folder called 'v2_5' beside '2_5'."""
    for bad in ["v2_5", "2.5", "", "latest", "2_5/"]:
        with pytest.raises(ValueError, match="schema_version"):
            train_data_key(schema_version=bad, filename="x.parquet")


def test_the_filename_is_parseable_by_the_training_pipeline():
    """training_pipeline.registry.parse_schema_version wants the version
    immediately before an 8-digit date, as the local CSVs have it."""
    from training_pipeline.registry import parse_schema_version

    name = historical_filename(schema_version="2_5", limit_date="20260215")
    assert name == "historical_training_data_2_5_20260215.parquet"
    assert parse_schema_version(name) == "2_5"


# --- what the sweep selects ------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "train_data/historical_training_data_until_20240218.parquet",
        "train_data/something_undated.parquet",
    ],
)
def test_datasets_loose_at_the_root_are_selected(key):
    assert is_unversioned_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "train_data/1_0/historical_training_data_until_20240218.parquet",
        "train_data/2_5/historical_training_data_2_5_20260215.parquet",
        # Not a version, but in a subfolder, so not ours to move. The injury
        # report storage helper suggests exactly this prefix.
        "train_data/injury_reports/2026-02-15.json",
        # The zero-byte placeholder standing for the prefix itself.
        "train_data/",
        "models/2_5/line_error/t0000/main/channels/production.json",
    ],
)
def test_everything_else_is_left_alone(key):
    assert not is_unversioned_key(key)


# --- the sweep -------------------------------------------------------------


def _client_with(keys):
    client = FakeS3Client()
    for key in keys:
        client.put_object(Bucket="b", Key=key, Body=b"x")
    return client


def test_the_sweep_moves_loose_files_and_keeps_their_names():
    client = _client_with(
        [
            "train_data/",
            "train_data/historical_training_data_until_20240218.parquet",
            "train_data/historical_training_data_until_20260215.parquet",
            "train_data/injury_reports/2026-02-15.json",
        ]
    )
    moves = plan_moves(
        s3_client=client,
        bucket="b",
        root=TRAIN_DATA_PREFIX,
        schema_version=LEGACY_SCHEMA_VERSION,
    )
    assert [(source, target) for source, target, _ in moves] == [
        (
            "train_data/historical_training_data_until_20240218.parquet",
            "train_data/1_0/historical_training_data_until_20240218.parquet",
        ),
        (
            "train_data/historical_training_data_until_20260215.parquet",
            "train_data/1_0/historical_training_data_until_20260215.parquet",
        ),
    ]


def test_running_it_twice_is_a_no_op():
    """The second run must not produce train_data/1_0/1_0/."""
    client = _client_with(
        [
            "train_data/1_0/historical_training_data_until_20240218.parquet",
            "train_data/2_5/historical_training_data_2_5_20260215.parquet",
        ]
    )
    assert (
        plan_moves(
            s3_client=client,
            bucket="b",
            root=TRAIN_DATA_PREFIX,
            schema_version=LEGACY_SCHEMA_VERSION,
        )
        == []
    )
