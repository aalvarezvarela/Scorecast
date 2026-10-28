"""Historical chart specifications built only from small aggregate tables."""

import altair as alt
import pandas as pd

from nba_ou.dashboard.config import SIGNAL_HEX_COLORS
from nba_ou.dashboard.domain.analysis import BREAK_EVEN, TIME_BUCKETS, bucket_label

METRICS = {"Win %": "win_pct", "Bets": "bets", "MAE": "mae", "ROI": "roi"}
TIME_LABELS = [bucket_label(value) for value in TIME_BUCKETS]
MODEL_COLORS = [
    SIGNAL_HEX_COLORS["UNDER"],
    SIGNAL_HEX_COLORS["OVER"],
    "#D3BDFF",
    "#F6C56F",
]
TOOLTIPS = [
    alt.Tooltip("model_label:N", title="Model"),
    alt.Tooltip("time_label:N", title="Input time bucket"),
    alt.Tooltip("wins:Q", title="Wins"),
    alt.Tooltip("losses:Q", title="Losses"),
    alt.Tooltip("pushes:Q", title="Pushes"),
    alt.Tooltip("games:Q", title="Distinct games"),
    alt.Tooltip("bets:Q", title="Bets (W/L/P)"),
    alt.Tooltip("win_pct:Q", title="Win %", format=".1f"),
    alt.Tooltip("ci_low:Q", title="95% CI low", format=".1f"),
    alt.Tooltip("ci_high:Q", title="95% CI high", format=".1f"),
    alt.Tooltip("coverage:Q", title="Coverage %", format=".1f"),
    alt.Tooltip("mae:Q", title="MAE (points)", format=".2f"),
    alt.Tooltip("roi:Q", title="ROI % (recorded prices only)", format=".1f"),
    alt.Tooltip("priced_bets:Q", title="Bets with valid prices"),
    alt.Tooltip(
        "median_lag_minutes:Q", title="Median publication lag (min)", format=".1f"
    ),
]


def _times(cells: pd.DataFrame) -> pd.DataFrame:
    data = cells.copy()
    data["time_label"] = data["bucket"].map(bucket_label)
    return data


def closing_timeline(cells: pd.DataFrame, metric: str):
    return _prediction_timeline(
        cells,
        metric,
        x_title="Input time before tip-off (Final = last published prediction)",
    )


def intermediate_replay_timeline(cells: pd.DataFrame, metric: str):
    own_replay = cells.loc[
        cells["dataset_type"].eq(2)
        & cells["source"].eq("Snapshot replay")
        & cells["bucket"].eq(cells["model_horizon_minutes"])
    ]
    return _prediction_timeline(
        own_replay,
        metric,
        x_title="Intermediate model horizon before tip-off (own-snapshot replay)",
    )


def _prediction_timeline(cells: pd.DataFrame, metric: str, *, x_title: str):
    data = _times(cells)
    column = METRICS[metric]
    chart = alt.Chart(data).encode(
        x=alt.X(
            "time_label:O",
            sort=TIME_LABELS,
            title=x_title,
            axis=alt.Axis(labelAngle=-35),
        ),
        color=alt.Color(
            "model_label:N", title="Model", scale=alt.Scale(range=MODEL_COLORS)
        ),
    )
    points = chart.mark_line(point=True).encode(
        y=alt.Y(
            f"{column}:Q",
            title="MAE (points)" if metric == "MAE" else metric,
            scale=alt.Scale(zero=False),
        ),
        tooltip=TOOLTIPS,
    )
    if metric == "Win %":
        band = chart.mark_area(opacity=0.15).encode(y="ci_low:Q", y2="ci_high:Q")
        rule = (
            alt.Chart(pd.DataFrame({"reference": [BREAK_EVEN]}))
            .mark_rule(strokeDash=[5, 4], color="#AFBCD4")
            .encode(y="reference:Q")
        )
        labels = chart.mark_text(dy=-12, fontSize=10).encode(
            y="win_pct:Q", text="bets:Q"
        )
        points = band + rule + points + labels
    return (
        points.properties(height=290)
        .configure_legend(orient="bottom")
        .configure_view(strokeWidth=0)
    )


def live_heatmap(cells: pd.DataFrame, metric: str):
    data = _times(cells)
    data["row_label"] = (
        data["model_label"] + " · " + data["model_horizon_minutes"].map(bucket_label)
    )
    data.loc[data["dataset_type"].eq(1), "row_label"] = (
        data["model_label"] + " · Closing"
    )
    order = (
        data.sort_values(
            ["model_label", "dataset_type", "model_horizon_minutes"],
            ascending=[True, True, False],
        )["row_label"]
        .drop_duplicates()
        .tolist()
    )
    data["diagonal"] = data["dataset_type"].eq(2) & data["bucket"].eq(
        data["model_horizon_minutes"]
    )
    column = METRICS[metric]
    if metric == "Win %":
        scale = alt.Scale(
            domain=[35, BREAK_EVEN, 70],
            range=["#A63549", "#33435D", "#2A9C75"],
            clamp=True,
        )
    elif metric == "ROI":
        scale = alt.Scale(
            domain=[-30, 0, 30], range=["#A63549", "#33435D", "#2A9C75"], clamp=True
        )
    else:
        scale = alt.Scale(scheme="blues", reverse=metric == "MAE")
    chart = alt.Chart(data).encode(
        x=alt.X(
            "time_label:O",
            sort=TIME_LABELS,
            title="Input time before tip-off",
            axis=alt.Axis(labelAngle=-35),
        ),
        y=alt.Y("row_label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        tooltip=[alt.Tooltip("row_label:N", title="Slot"), *TOOLTIPS],
    )
    cells_chart = chart.mark_rect(strokeWidth=2).encode(
        color=alt.condition(
            "datum.low_sample",
            alt.value("#3C4659"),
            alt.Color(f"{column}:Q", title=metric, scale=scale),
        ),
        stroke=alt.condition(
            "datum.diagonal", alt.value("#F6C56F"), alt.value("transparent")
        ),
    )
    labels = chart.mark_text(fontSize=11, color="#FFFFFF").encode(
        text=alt.Text(f"{column}:Q", format=".0f" if metric == "Bets" else ".1f")
    )
    return (
        (cells_chart + labels)
        .properties(height=max(140, 30 * len(order)))
        .configure_view(strokeWidth=0)
    )


def snapshot_comparison(cells: pd.DataFrame):
    data = _times(cells)
    data["slot_label"] = (
        data["model_label"] + " · " + data["model_horizon_minutes"].map(bucket_label)
    )
    order = (
        data.sort_values(
            ["model_label", "model_horizon_minutes"], ascending=[True, False]
        )["slot_label"]
        .drop_duplicates()
        .tolist()
    )
    base = alt.Chart(data).encode(
        y=alt.Y("slot_label:N", title=None, sort=order, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("source:N"),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(
                domain=["Live predictions", "Snapshot replay"], range=MODEL_COLORS[:2]
            ),
            title="Source",
        ),
        tooltip=[alt.Tooltip("source:N", title="Source"), *TOOLTIPS],
    )
    intervals = base.mark_rule(strokeWidth=2).encode(
        x=alt.X(
            "ci_low:Q",
            title="Win % (95% per-cell interval)",
            scale=alt.Scale(domain=[0, 100]),
        ),
        x2="ci_high:Q",
    )
    points = base.mark_point(filled=True, size=65).encode(x="win_pct:Q")
    reference = (
        alt.Chart(pd.DataFrame({"reference": [BREAK_EVEN]}))
        .mark_rule(strokeDash=[5, 4], color="#AFBCD4")
        .encode(x="reference:Q")
    )
    return (
        (reference + intervals + points)
        .properties(height=max(140, 45 * len(order)))
        .configure_legend(orient="bottom")
        .configure_view(strokeWidth=0)
    )
