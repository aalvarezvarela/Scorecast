"""Refit and serving build each slot's frame at the slot's own schema version."""

from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.create_training_data import create_df_to_predict as builder
from nba_ou.create_training_data.schema_layers import (
    LayerContext,
    is_layered,
    newest_version,
    registry,
)
from nba_ou.create_training_data.schema_layers.base import CLOSING_LINE, SchemaLayer
from nba_ou.modeling.refit import TrainingFrameUnavailable, resolve_training_frame


@pytest.fixture
def builds(monkeypatch):
    calls = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {"GAME_ID": ["0022500001", "0022500002"], "BASE_BEFORE": [1.0, 2.0]}
        )

    monkeypatch.setattr(builder, "create_df_to_predict", fake_builder)
    layer = SchemaLayer(
        version="2_6",
        parent="2_5",
        summary="stub",
        columns={CLOSING_LINE: ("NEW_BEFORE",)},
        requires=(),
        build=lambda i, c, d: pd.DataFrame({"NEW_BEFORE": [5.0, 6.0]}, index=i.index),
    )
    monkeypatch.setitem(registry.LAYERS, "2_6", layer)
    return calls


def _spec(version, dataset_type="closing_line"):
    return SimpleNamespace(
        spec_id=f"spec-{version}",
        identity=SimpleNamespace(dataset_type=dataset_type, schema_version=version),
    )


def test_each_version_gets_its_own_columns_from_one_base_build(builds):
    base_frames: dict = {}
    ctx = LayerContext()
    older = resolve_training_frame(
        _spec("2_5"),
        limit_date="2026-06-30",
        base_frames=base_frames,
        layer_context=ctx,
    )
    newer = resolve_training_frame(
        _spec("2_6"),
        limit_date="2026-06-30",
        base_frames=base_frames,
        layer_context=ctx,
    )
    assert len(builds) == 1
    assert list(older.df.columns) == ["GAME_ID", "BASE_BEFORE"]
    assert list(newer.df.columns) == ["GAME_ID", "BASE_BEFORE", "NEW_BEFORE"]
    # Layering never mutates the cached base other slots share.
    assert list(base_frames[("closing_line", "2026-06-30")].columns) == [
        "GAME_ID",
        "BASE_BEFORE",
    ]


def test_a_version_this_checkout_cannot_build_is_unavailable_not_a_crash(builds):
    with pytest.raises(TrainingFrameUnavailable, match="9_9"):
        resolve_training_frame(_spec("9_9"))
    assert builds == []


def test_versions_older_than_the_base_are_served_by_the_base(builds):
    frame = resolve_training_frame(_spec("2_4"))
    assert list(frame.df.columns) == ["GAME_ID", "BASE_BEFORE"]


def test_one_serving_frame_covers_the_newest_enabled_version():
    assert newest_version(["2_5", "2_6", "2_5"]) == "2_6"
    assert newest_version(["2_4", "2_5"]) == "2_5"
    assert newest_version([]) == "2_5"
    assert not is_layered("2_5") and is_layered("2_6")
    with pytest.raises(ValueError, match="not buildable"):
        newest_version(["2_5", "9_9"])
