import pandas as pd
import pytest
from nba_ou.dashboard.config import DEFAULT_MIN_EDGE_POINTS
from nba_ou.dashboard.domain.comparisons import compare_predictions, signal
from nba_ou.dashboard.domain.selection import market_snapshots, select_predictions

from .conftest import NOW, make_prediction


def compared(target=1, reference="Bet365", min_edge=DEFAULT_MIN_EDGE_POINTS, **changes):
    raw = pd.DataFrame([make_prediction(target=target, **changes)])
    return compare_predictions(
        select_predictions(raw, target, NOW),
        market_snapshots(raw, target, NOW),
        reference,
        min_edge=min_edge,
    ).iloc[0]


def test_normalized_line_error_total_and_raw_bet365_direction_can_disagree():
    row = compared(
        target=1, model_line=229, prediction_value=1, bet_line_x2=461, min_edge=0.1
    )
    assert row["predicted_total"] == 230
    assert row["predicted_error"] == 1
    assert row["comparison_edge"] == -0.5
    assert row["signal"] == "UNDER"


def test_total_points_also_exposes_its_error():
    row = compared(target=2, prediction_value=232, model_line=229, bet_line_x2=461)
    assert row["predicted_total"] == 232
    assert row["predicted_error"] == 3
    assert row["comparison_edge"] == 1.5
    assert row["signal"] == "OVER"


def test_reference_changes_edge_without_changing_the_prediction():
    bet365 = compared(reference="Bet365")
    mean = compared(reference="Market mean")
    assert bet365["predicted_total"] == mean["predicted_total"]
    assert mean["comparison_edge"] == 0
    assert mean["signal"] == "NO EDGE"


def test_no_bet365_does_not_fall_back_to_mean():
    row = compared(bet_line_x2=None)
    assert row["predicted_total"] == 230
    assert pd.isna(row["comparison_edge"])
    assert row["signal"] == "UNAVAILABLE"


def test_missing_model_anchor_does_not_invent_a_total():
    assert pd.isna(compared(model_line=None)["predicted_total"])


@pytest.mark.parametrize("value, expected", [(2, "HOME"), (-2, "AWAY"), (0, "NO EDGE")])
def test_spread_preserves_original_error_when_model_and_market_lines_match(
    value, expected
):
    row = compared(target=3, prediction_value=value)
    assert pd.isna(row["predicted_total"])
    assert row["predicted_error"] == value
    assert row["signal"] == expected


@pytest.mark.parametrize(
    "output, line, expected_edge, expected_side",
    [
        (2, 7.5, -2, "AWAY"),  # Original HOME error reverses after the market moves.
        (-2, -0.5, 2, "HOME"),
        (0, 1.5, 2, "HOME"),  # A zero original residual can have a market edge.
    ],
)
def test_spread_direction_uses_the_current_market_line(
    output, line, expected_edge, expected_side
):
    row = compared(
        target=3, prediction_value=output, model_line=3.5, bet_line_x2=line * 2
    )
    assert row["predicted_error"] == output
    assert row["predicted_home_margin"] == 3.5 + output
    assert row["comparison_edge"] == expected_edge
    assert row["signal"] == expected_side


def test_spread_reference_switch_changes_edge_without_changing_model_output():
    inputs = dict(
        target=3,
        model_line=3.5,
        prediction_value=2,
        bet_line_x2=15,
        books_mean_line=3.0,
        books_median_line=5.5,
    )
    bet365 = compared(**inputs)
    mean = compared(reference="Market mean", **inputs)
    median = compared(reference="Market median", **inputs)
    assert [row["predicted_error"] for row in (bet365, mean, median)] == [2, 2, 2]
    assert [row["comparison_edge"] for row in (bet365, mean, median)] == [-2, 2.5, 0]
    assert [row["signal"] for row in (bet365, mean, median)] == [
        "AWAY",
        "HOME",
        "NO EDGE",
    ]


@pytest.mark.parametrize("changes", [{"bet_line_x2": None}, {"model_line": None}])
def test_spread_missing_market_or_model_line_never_falls_back_to_raw_error(changes):
    row = compared(target=3, prediction_value=2, **changes)
    assert row["predicted_error"] == 2
    assert pd.isna(row["comparison_edge"])
    assert row["signal"] == "UNAVAILABLE"


@pytest.mark.parametrize("spread", [False, True])
@pytest.mark.parametrize("edge", [-0.1, -0.05, 0.0, 0.05, 0.1])
def test_small_edges_and_threshold_equality_are_neutral(edge, spread):
    assert signal(edge, spread=spread) == "NO EDGE"


def test_tabpfn_bets_on_a_smaller_edge_than_registry_models():
    raw = pd.DataFrame(
        [make_prediction(kind=kind, target=1, prediction_value=1.55) for kind in (1, 3)]
    )
    rows = compare_predictions(
        select_predictions(raw, 1, NOW),
        market_snapshots(raw, 1, NOW),
        min_edge={1: 0.1, 3: 0.01},
    ).set_index("kind")
    # 229 + 1.55 against Bet365 230.5: an edge of +0.05 for both models.
    assert rows["comparison_edge"].round(9).tolist() == [0.05, 0.05]
    assert rows.loc[1, "signal"] == "NO EDGE"
    assert rows.loc[3, "signal"] == "OVER"
    assert rows["min_edge"].tolist() == [0.1, 0.01]


@pytest.mark.parametrize("spread,side", [(False, "OVER"), (True, "HOME")])
def test_threshold_changes_signal_without_changing_the_market_edge(spread, side):
    assert signal(1.5, spread=spread, min_edge=1.0) == side
    assert signal(1.5, spread=spread, min_edge=2.0) == "NO EDGE"
    assert signal(0.10000000000000009, min_edge=0.1) == "NO EDGE"


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_invalid_signal_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="min_edge"):
        signal(2.0, min_edge=threshold)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_and_nonfinite_values_never_become_under(value):
    assert signal(value) == "UNAVAILABLE"
