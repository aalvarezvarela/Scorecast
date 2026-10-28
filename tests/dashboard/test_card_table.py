import pandas as pd
from bs4 import BeautifulSoup
from nba_ou.dashboard.domain.models import family_label
from nba_ou.dashboard.domain.picks import game_pick
from nba_ou.dashboard.ui.formatting import compact_totals_table
from nba_ou.dashboard.ui.totals_table import totals_table_html


def example_predictions():
    return pd.DataFrame(
        [
            dict(
                phase="Closing",
                target=1,
                model_horizon_minutes=0,
                predicted_total=241.1,
                comparison_edge=-0.4,
                signal="UNDER",
            ),
            dict(
                phase="Closing",
                target=2,
                model_horizon_minutes=0,
                predicted_total=240.9,
                comparison_edge=-0.6,
                signal="UNDER",
            ),
            dict(
                phase="Intermediate",
                target=2,
                model_horizon_minutes=660,
                predicted_total=239.9,
                comparison_edge=-1.6,
                signal="UNDER",
            ),
            dict(
                phase="Intermediate",
                target=1,
                model_horizon_minutes=660,
                predicted_total=242.3,
                comparison_edge=0.8,
                signal="OVER",
            ),
        ]
    )


def test_four_predictions_become_two_rows_without_losing_values_or_votes():
    predictions = example_predictions()
    table = compact_totals_table(predictions)
    assert list(table.index) == ["Closing", "T−11h"]
    assert list(table.columns) == [
        ("Total points", "Prediction"),
        ("Total points", "Margin"),
        ("Line error", "Prediction"),
        ("Line error", "Margin"),
    ]
    assert table.loc["Closing", ("Total points", "Prediction")] == "**240.9**"
    assert table.loc["Closing", ("Total points", "Margin")] == "-0.6 :blue-badge[UNDER]"
    assert table.loc["Closing", ("Line error", "Prediction")] == "**241.1**"
    assert table.loc["Closing", ("Line error", "Margin")] == "-0.4 :blue-badge[UNDER]"
    assert table.loc["T−11h", ("Total points", "Prediction")] == "**239.9**"
    assert table.loc["T−11h", ("Total points", "Margin")] == "-1.6 :blue-badge[UNDER]"
    assert table.loc["T−11h", ("Line error", "Prediction")] == "**242.3**"
    assert table.loc["T−11h", ("Line error", "Margin")] == "+0.8 :red-badge[OVER]"
    pick = game_pick(predictions)
    assert (pick.status, pick.side, pick.agreeing, pick.predictions) == (
        "BET",
        "UNDER",
        3,
        4,
    )


def test_different_intermediate_horizons_stay_explicit_in_each_target_cell():
    predictions = example_predictions()
    predictions.loc[2, "model_horizon_minutes"] = 240
    predictions.loc[3, "model_horizon_minutes"] = 120
    table = compact_totals_table(predictions)
    assert table.loc["Intermediate", ("Total points", "Prediction")].startswith(
        "T−4h · "
    )
    assert table.loc["Intermediate", ("Line error", "Prediction")].startswith("T−2h · ")


def test_missing_target_is_empty_not_a_fabricated_prediction():
    predictions = example_predictions().iloc[[0, 2]]
    table = compact_totals_table(predictions)
    assert table.loc["Closing", ("Total points", "Prediction")] == "—"
    assert table.loc["Closing", ("Total points", "Margin")] == "—"
    assert table.loc["T−11h", ("Line error", "Prediction")] == "—"
    assert table.loc["T−11h", ("Line error", "Margin")] == "—"


def test_html_spans_target_headers_and_preserves_all_predictions_and_picks():
    table = pd.concat(
        [compact_totals_table(example_predictions())],
        keys=["XGBoost · 2.5"],
        names=["Model"],
    )
    soup = BeautifulSoup(totals_table_html(table), "html.parser")
    groups = soup.select('th[scope="colgroup"]')
    assert [(th.get_text(), th["colspan"]) for th in groups] == [
        ("Total points", "2"),
        ("Line error", "2"),
    ]
    assert len(soup.select("tbody tr")) == 2
    assert [cell.get_text() for cell in soup.select("td.prediction")] == [
        "240.9",
        "241.1",
        "239.9",
        "242.3",
    ]
    assert [badge.get_text() for badge in soup.select(".pick")] == [
        "UNDER",
        "UNDER",
        "UNDER",
        "OVER",
    ]


def test_html_escapes_model_labels():
    table = pd.concat(
        [compact_totals_table(example_predictions())],
        keys=['<img src=x onerror="alert(1)">'],
        names=["Model"],
    )
    soup = BeautifulSoup(totals_table_html(table), "html.parser")
    assert not soup.select("img, script")
    assert '<img src=x onerror="alert(1)">' in soup.get_text()


def test_tabpfn_label_omits_variant_but_registry_variants_remain_visible():
    assert family_label(3, "2_5", "tabpfn_v3_5") == "TabPFN · 2.5"
    assert (
        family_label(3, "2_5", "tabpfn_experimental") == "TabPFN · 2.5 · experimental"
    )
    assert family_label(1, "2_5", "alt") == "XGBoost · 2.5 · alt"
