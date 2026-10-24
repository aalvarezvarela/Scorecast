"""Refit and serving build each slot's frame at the slot's own schema version."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.create_training_data.schema_layers import (
    LayerContext,
    is_layered,
    newest_version,
    registry,
)
from nba_ou.create_training_data.schema_layers.base import CLOSING_LINE, SchemaLayer
from nba_ou.create_training_data.season_partitions import PartitionedFrame
from nba_ou.modeling.refit import TrainingFrameUnavailable, resolve_training_frame


class _Partitions:
    """The base version's closing partitions; counts the reads."""

    def __init__(self):
        self.reads = 0

    def load(self, horizon=None):
        self.reads += 1
        return PartitionedFrame(
            df=pd.DataFrame(
                {"GAME_ID": ["0022500001", "0022500002"], "BASE_BEFORE": [1.0, 2.0]}
            ),
            build_id="closing_line_data_2_5_20260630",
            checksum="sha256:0",
            limit_date=date(2026, 6, 30),
            prefix="p/",
        )


@pytest.fixture
def builds(monkeypatch):
    layer = SchemaLayer(
        version="2_6",
        parent="2_5",
        summary="stub",
        columns={CLOSING_LINE: ("NEW_BEFORE",)},
        requires=(),
        build=lambda i, c, d: pd.DataFrame({"NEW_BEFORE": [5.0, 6.0]}, index=i.index),
    )
    monkeypatch.setitem(registry.LAYERS, "2_6", layer)
    return {CLOSING_LINE: _Partitions()}


def _spec(version, dataset_type="closing_line"):
    return SimpleNamespace(
        spec_id=f"spec-{version}",
        identity=SimpleNamespace(dataset_type=dataset_type, schema_version=version),
    )


def test_each_version_gets_its_own_columns_from_one_base_build(builds):
    base_frames: dict = {}
    ctx = LayerContext()
    older = resolve_training_frame(
        _spec("2_5"), partitions=builds, base_frames=base_frames, layer_context=ctx
    )
    newer = resolve_training_frame(
        _spec("2_6"), partitions=builds, base_frames=base_frames, layer_context=ctx
    )
    assert builds[CLOSING_LINE].reads == 1
    assert list(older.df.columns) == ["GAME_ID", "BASE_BEFORE"]
    assert list(newer.df.columns) == ["GAME_ID", "BASE_BEFORE", "NEW_BEFORE"]
    assert older.build_id == newer.build_id == "closing_line_data_2_5_20260630"
    # Layering never mutates the cached base other slots share.
    assert list(base_frames[CLOSING_LINE].df.columns) == ["GAME_ID", "BASE_BEFORE"]


def test_a_version_this_checkout_cannot_build_is_unavailable_not_a_crash(builds):
    with pytest.raises(TrainingFrameUnavailable, match="9_9"):
        resolve_training_frame(_spec("9_9"), partitions=builds)
    assert builds[CLOSING_LINE].reads == 0


def test_versions_older_than_the_base_are_served_by_the_base(builds):
    frame = resolve_training_frame(_spec("2_4"), partitions=builds)
    assert list(frame.df.columns) == ["GAME_ID", "BASE_BEFORE"]


def test_one_serving_frame_covers_the_newest_enabled_version():
    assert newest_version(["2_5", "2_6", "2_5"]) == "2_6"
    assert newest_version(["2_4", "2_5"]) == "2_5"
    assert newest_version([]) == "2_5"
    assert not is_layered("2_5") and is_layered("2_6")
    with pytest.raises(ValueError, match="not buildable"):
        newest_version(["2_5", "9_9"])
