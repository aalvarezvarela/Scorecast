"""A sortable, long-format comparison table, with every timestamp in Madrid."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.dates import madrid_time
from nba_ou.dashboard.domain.teams import team_logo_url
from nba_ou.dashboard.ui.formatting import prediction_table, signal_style


def render_prediction_table(
    predictions: pd.DataFrame, reference: str, *, spread: bool
) -> None:
    if predictions.empty:
        st.info("No predictions are available for these filters.")
        return
    display = prediction_table(predictions, reference, spread=spread)
    display.insert(
        0, "Matchup", predictions["away_team"] + " @ " + predictions["home_team"]
    )
    display.insert(1, "Tip-off (Madrid)", predictions["tipoff_at"].map(madrid_time))
    display.insert(0, "Away logo", predictions["away_team"].map(team_logo_url))
    display.insert(1, "Home logo", predictions["home_team"].map(team_logo_url))
    for label, column in (
        ("Bet365", "bet365_line"),
        ("Market mean", "books_mean_line"),
        ("Market median", "books_median_line"),
    ):
        display[label] = predictions[column]
    numeric_columns = [
        "Predicted total",
        "Error vs model line",
        f"Edge vs {reference}",
        "Spread error",
        "Bet365",
        "Market mean",
        "Market median",
    ]
    numeric_formats = {"Min edge": "%g"}
    st.dataframe(
        display.style.map(signal_style, subset=["Signal"]),
        hide_index=True,
        column_config={
            "Away logo": st.column_config.ImageColumn("Away", width="small"),
            "Home logo": st.column_config.ImageColumn("Home", width="small"),
            **{
                label: st.column_config.NumberColumn(label, format="%.1f")
                for label in numeric_columns
                if label in display
            },
            **{
                label: st.column_config.NumberColumn(label, format=pattern)
                for label, pattern in numeric_formats.items()
            },
        },
        width="stretch",
        height="auto",
    )
