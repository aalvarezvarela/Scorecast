import json
from uuid import UUID

import pandas as pd
import pytest
from nba_ou.dashboard.domain.comparisons import compare_predictions
from nba_ou.dashboard.domain.selection import market_snapshots, select_predictions
from nba_ou.dashboard.ui.charts import comparison_chart

from .conftest import NOW, make_prediction


@pytest.mark.parametrize("target", [1, 2, 3])
def test_charts_serialize_real_database_uuid_types_without_sending_raw_rows(
    today_data, target
):
    raw = today_data.predictions.copy()
    raw["run_uuid"] = UUID("00000000-0000-0000-0000-000000000001")
    markets = market_snapshots(raw, target, NOW)
    predictions = compare_predictions(select_predictions(raw, target, NOW), markets)
    chart = comparison_chart(
        predictions.loc[predictions["game_id"].eq(1)],
        markets.iloc[0],
        spread=target == 3,
    )
    serialized = json.dumps(chart.to_dict())
    assert "run_uuid" not in serialized
    assert "HOME" in serialized if target == 3 else "Bet365" in serialized


def test_spread_chart_plots_market_edge_when_it_opposes_original_error():
    raw = pd.DataFrame(
        [make_prediction(target=3, model_line=3.5, prediction_value=2, bet_line_x2=15)]
    )
    markets = market_snapshots(raw, 3, NOW)
    predictions = compare_predictions(select_predictions(raw, 3, NOW), markets)
    spec = comparison_chart(predictions, markets.iloc[0], spread=True).to_dict()
    records = [
        row
        for dataset in spec["datasets"].values()
        for row in dataset
        if "signal" in row
    ]
    assert records[0]["predicted_error"] == 2
    assert records[0]["comparison_edge"] == -2
    assert records[0]["signal"] == "AWAY"
    assert spec["layer"][1]["encoding"]["x"]["field"] == "comparison_edge"
