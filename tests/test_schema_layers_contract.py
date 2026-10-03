"""A layer may add its declared columns and nothing else."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.create_training_data.schema_layers import registry
from nba_ou.create_training_data.schema_layers.base import (
    CLOSING_LINE,
    INTERMEDIATE_LINE,
    LayerContext,
    SchemaLayer,
)
from nba_ou.create_training_data.schema_layers.contract import LayerContractError


def _layer(
    build, *, version="2_6", parent="2_5", columns=("NEW_X_BEFORE",), requires=()
):
    return SchemaLayer(
        version=version,
        parent=parent,
        summary="test layer",
        columns={CLOSING_LINE: tuple(columns)},
        requires=requires,
        build=build,
    )


@pytest.fixture
def base():
    return pd.DataFrame(
        {
            "GAME_ID": ["0022500001", "0022500002", "0022500003"],
            "GAME_DATE": pd.to_datetime(["2025-10-21", "2025-10-22", "2025-10-22"]),
            "OLD_BEFORE": [1.5, np.nan, 3.25],
        }
    )


@pytest.fixture
def use_layers(monkeypatch):
    def install(*layers):
        monkeypatch.setattr(registry, "chain", lambda from_v, to_v: list(layers))

    return install


def test_new_columns_are_appended_and_the_parent_is_untouched(base, use_layers):
    use_layers(
        _layer(
            lambda inputs, ctx, dt: pd.DataFrame(
                {"NEW_X_BEFORE": [7.0, 8.0, 9.0]}, index=inputs.index
            )
        )
    )
    before = base.copy()
    out = registry.apply_layers(base, to_version="2_6", dataset_type=CLOSING_LINE)
    pd.testing.assert_frame_equal(out[before.columns], before)
    assert list(out.columns) == [*before.columns, "NEW_X_BEFORE"]
    pd.testing.assert_frame_equal(base, before)


def test_a_layer_sees_only_keys_and_what_it_requires(base, use_layers):
    seen = {}

    def build(inputs, ctx, dataset_type):
        seen["columns"] = list(inputs.columns)
        return pd.DataFrame({"NEW_X_BEFORE": 0.0}, index=inputs.index)

    use_layers(_layer(build, requires=("GAME_DATE",)))
    registry.apply_layers(base, to_version="2_6", dataset_type=CLOSING_LINE)
    assert seen["columns"] == ["GAME_ID", "GAME_DATE"]


def test_a_later_layer_can_read_an_earlier_layers_output(base, use_layers):
    first = _layer(
        lambda i, c, d: pd.DataFrame({"A_BEFORE": [1.0, 2.0, 3.0]}, index=i.index),
        columns=("A_BEFORE",),
    )
    second = _layer(
        lambda i, c, d: pd.DataFrame({"B_BEFORE": i["A_BEFORE"] * 10}, index=i.index),
        version="2_7",
        parent="2_6",
        columns=("B_BEFORE",),
        requires=("A_BEFORE",),
    )
    use_layers(first, second)
    out = registry.apply_layers(base, to_version="2_7", dataset_type=CLOSING_LINE)
    assert out["B_BEFORE"].tolist() == [10.0, 20.0, 30.0]


def test_the_context_is_shared_across_layers(base, use_layers):
    loads = []

    def build(name):
        def run(inputs, ctx, dataset_type):
            ctx.get_or_load("players", lambda: loads.append(1) or "loaded")
            return pd.DataFrame({name: 0.0}, index=inputs.index)

        return run

    use_layers(
        _layer(build("A_BEFORE"), columns=("A_BEFORE",)),
        _layer(build("B_BEFORE"), version="2_7", parent="2_6", columns=("B_BEFORE",)),
    )
    registry.apply_layers(
        base, to_version="2_7", dataset_type=CLOSING_LINE, ctx=LayerContext()
    )
    assert loads == [1]


def test_a_dataset_the_layer_does_not_cover_gets_no_columns(base, use_layers):
    use_layers(_layer(lambda i, c, d: pytest.fail("must not run")))
    frame = base.assign(TIME_TO_MATCH_MIN=60)
    out = registry.apply_layers(frame, to_version="2_6", dataset_type=INTERMEDIATE_LINE)
    assert out is frame


@pytest.mark.parametrize(
    ("make_output", "message"),
    [
        (lambda i: pd.DataFrame({"NEW_X_BEFORE": [1.0, 2.0]}), "rows"),
        (lambda i: pd.DataFrame({"NEW_X_BEFORE": 1.0}, index=i.index[::-1]), "rows"),
        (lambda i: pd.DataFrame({"OTHER_BEFORE": 1.0}, index=i.index), "undeclared"),
        (lambda i: pd.DataFrame(index=i.index), "missing declared"),
    ],
)
def test_misaligned_or_undeclared_output_is_refused(
    base, use_layers, make_output, message
):
    use_layers(_layer(lambda inputs, ctx, dt: make_output(inputs)))
    with pytest.raises(LayerContractError, match=message):
        registry.apply_layers(base, to_version="2_6", dataset_type=CLOSING_LINE)


def test_an_existing_column_can_never_be_overwritten(base, use_layers):
    use_layers(
        _layer(
            lambda i, c, d: pd.DataFrame({"OLD_BEFORE": 0.0}, index=i.index),
            columns=("OLD_BEFORE",),
        )
    )
    with pytest.raises(LayerContractError, match="overwrite"):
        registry.apply_layers(base, to_version="2_6", dataset_type=CLOSING_LINE)


@pytest.mark.parametrize("column", ["NEW_X", "N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME"])
def test_names_must_follow_the_leakage_conventions(base, use_layers, column):
    use_layers(
        _layer(
            lambda i, c, d: pd.DataFrame({column: 0.0}, index=i.index),
            columns=(column,),
        )
    )
    with pytest.raises(LayerContractError):
        registry.apply_layers(base, to_version="2_6", dataset_type=CLOSING_LINE)


def test_a_missing_required_column_is_reported_by_name(base, use_layers):
    use_layers(
        _layer(
            lambda i, c, d: pytest.fail("must not run"), requires=("TEAM_ID_TEAM_HOME",)
        )
    )
    with pytest.raises(ValueError, match="TEAM_ID_TEAM_HOME"):
        registry.apply_layers(base, to_version="2_6", dataset_type=CLOSING_LINE)


def test_layers_must_be_newer_than_their_parent():
    with pytest.raises(ValueError, match="newer"):
        _layer(lambda i, c, d: None, version="2_5", parent="2_5")
