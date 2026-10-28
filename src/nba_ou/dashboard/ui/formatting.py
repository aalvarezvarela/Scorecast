"""English labels and numeric formatting shared by cards and tables."""

import math

import pandas as pd

from nba_ou.dashboard.config import SIGNAL_COLORS
from nba_ou.dashboard.domain.dates import horizon_label, madrid_time


def number(value, *, signed: bool = False) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "—"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def compact_totals_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Two phase rows, with separate prediction and margin columns per target.

    Cells retain predicted total, market edge and pick. Different intermediate
    horizons stay explicit rather than being relabeled as the same snapshot.
    """
    rows = []
    labels = []
    for phase in ("Closing", "Intermediate"):
        part = predictions.loc[predictions["phase"].eq(phase)]
        if part.empty:
            continue
        horizons = part["model_horizon_minutes"].dropna().unique()
        shared_horizon = phase == "Intermediate" and len(horizons) == 1
        label = horizon_label(horizons[0]) if shared_horizon else phase
        labels.append(label)
        row = {}
        for target, column in ((2, "Total points"), (1, "Line error")):
            target_rows = part.loc[part["target"].eq(target)]
            if target_rows.empty:
                row[column, "Prediction"] = "—"
                row[column, "Margin"] = "—"
                continue
            prediction = target_rows.iloc[0]
            pick = prediction["signal"]
            cell = f"**{number(prediction['predicted_total'])}**"
            if phase == "Intermediate" and not shared_horizon:
                cell = f"{horizon_label(prediction['model_horizon_minutes'])} · {cell}"
            row[column, "Prediction"] = cell
            row[column, "Margin"] = (
                f"{number(prediction['comparison_edge'], signed=True)} "
                f":{SIGNAL_COLORS[pick]}-badge[{pick}]"
            )
        rows.append(row)
    columns = pd.MultiIndex.from_product(
        [["Total points", "Line error"], ["Prediction", "Margin"]]
    )
    return pd.DataFrame(rows, index=pd.Index(labels, name="Snapshot"), columns=columns)


def prediction_table(
    predictions: pd.DataFrame, reference: str, *, spread: bool
) -> pd.DataFrame:
    display = pd.DataFrame(index=predictions.index)
    display["Model"] = predictions["model_label"]
    display["Target"] = predictions["target_label"]
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
