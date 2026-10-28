"""English labels and numeric formatting shared by cards and tables."""

import math

import pandas as pd

from nba_ou.dashboard.domain.dates import horizon_label, madrid_time


def number(value, *, signed: bool = False) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "—"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def prediction_table(
    predictions: pd.DataFrame, reference: str, *, spread: bool
) -> pd.DataFrame:
    display = pd.DataFrame(index=predictions.index)
    display["Model"] = predictions["model_label"]
    display["Prediction"] = predictions["phase"]
    display["Horizon"] = predictions["model_horizon_minutes"].map(horizon_label)
    display.loc[predictions["dataset_type"].eq(1), "Horizon"] = "Closing"
    if not spread:
        display["Predicted total"] = predictions["predicted_total"]
        display["Error vs model line"] = predictions["predicted_error"]
    else:
        display["Spread error"] = predictions["predicted_error"]
    display[f"Edge vs {reference}"] = predictions["comparison_edge"]
    display["Min edge"] = predictions["min_edge"]
    display["Signal"] = predictions["signal"]
    display["Predicted at (Madrid)"] = predictions["predicted_at"].map(madrid_time)
    return display


def signal_style(value) -> str:
    color = {
        "OVER": "#FB923C",
        "HOME": "#FB923C",
        "UNDER": "#60A5FA",
        "AWAY": "#60A5FA",
    }.get(value, "#94A3B8")
    return f"color: {color}; font-weight: bold"
