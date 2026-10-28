"""Compact total-point comparisons and spread market edges with Altair."""

import altair as alt
import pandas as pd

from nba_ou.dashboard.domain.dates import horizon_label


def comparison_chart(
    predictions: pd.DataFrame,
    market: pd.Series | None,
    *,
    spread: bool,
    reference: str = "Bet365",
):
    # Keep database-only objects (notably psycopg UUIDs) out of Arrow/Altair.
    data = predictions[
        [
            "model_label",
            "target_label",
            "phase",
            "model_horizon_minutes",
            "predicted_error",
            "comparison_edge",
            "predicted_total",
            "signal",
        ]
    ].copy()
    data["label"] = [
        f"{model} · {target} · {phase}"
        + (f" {horizon_label(horizon)}" if phase == "Intermediate" else "")
        for model, target, phase, horizon in zip(
            data["model_label"],
            data["target_label"],
            data["phase"],
            data["model_horizon_minutes"],
            strict=True,
        )
    ]
    column = "comparison_edge" if spread else "predicted_total"
    data = data.loc[pd.to_numeric(data[column], errors="coerce").notna()]
    if data.empty:
        return None
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=110)
        .encode(
            x=alt.X(
                f"{column}:Q",
                title=f"Edge vs {reference} (points)"
                if spread
                else "Predicted total (points)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color(
                "signal:N",
                legend=None,
                scale=alt.Scale(
                    domain=["OVER", "UNDER", "HOME", "AWAY", "NO EDGE", "UNAVAILABLE"],
                    range=[
                        "#FB923C",
                        "#60A5FA",
                        "#FB923C",
                        "#60A5FA",
                        "#94A3B8",
                        "#94A3B8",
                    ],
                ),
            ),
            shape=alt.Shape(
                "phase:N",
                title="Prediction",
                scale=alt.Scale(
                    domain=["Closing", "Intermediate"], range=["circle", "diamond"]
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Model"),
                alt.Tooltip(f"{column}:Q", title="Prediction", format=".1f"),
                alt.Tooltip("signal:N", title="Signal"),
            ],
        )
    )
    if spread:
        references = pd.DataFrame({"value": [0.0], "reference": ["No edge"]})
    else:
        references = pd.DataFrame(
            [
                {"value": market.get(column), "reference": label}
                for label, column in (
                    ("Bet365", "bet365_line"),
                    ("Market mean", "books_mean_line"),
                    ("Market median", "books_median_line"),
                )
            ]
            if market is not None
            else [],
            columns=["value", "reference"],
        ).dropna(subset=["value"])
    if references.empty:
        chart = points
    else:
        rules = (
            alt.Chart(references)
            .mark_rule(color="#94A3B8", opacity=0.7)
            .encode(
                x="value:Q",
                strokeDash=alt.StrokeDash(
                    "reference:N",
                    title="Market",
                    scale=alt.Scale(range=[[1, 0], [5, 4], [2, 3]]),
                ),
                tooltip=[
                    alt.Tooltip("reference:N", title="Reference"),
                    alt.Tooltip("value:Q", format=".1f"),
                ],
            )
        )
        chart = rules + points
    return (
        chart.properties(height=max(120, 35 * len(data)))
        .configure_legend(orient="bottom", title=None)
        .configure_view(strokeWidth=0)
    )
