"""Native, responsive game cards retaining both teams' existing logo images."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import SIGNAL_COLORS
from nba_ou.dashboard.domain.dates import horizon_label, madrid_time, time_remaining
from nba_ou.dashboard.domain.models import ModelFamily
from nba_ou.dashboard.domain.teams import team_logo_url
from nba_ou.dashboard.ui.charts import comparison_chart
from nba_ou.dashboard.ui.formatting import number, prediction_table


def _team(name: str, team_id, role: str) -> None:
    with st.container(horizontal=True, vertical_alignment="center"):
        logo = team_logo_url(name, team_id)
        if logo:
            # Streamlit 1.54 has no image alt parameter; the visible caption
            # names the team, and the adjacent text remains usable without images.
            st.image(logo, width=48, caption=role)
        st.markdown(f"**{name}**")


def _market(market: pd.Series | None, *, spread: bool, now) -> None:
    cols = st.columns(3)
    for col, label, field in zip(
        cols,
        ("Bet365", "Market mean", "Market median"),
        ("bet365_line", "books_mean_line", "books_median_line"),
        strict=True,
    ):
        with col:
            st.metric(label, number(market.get(field) if market is not None else None))
    if market is None:
        st.caption("Market data unavailable. No substitute line has been assumed.")
        return
    count = market.get("books_count")
    count_label = (
        f"{int(count)} bookmakers" if pd.notna(count) else "Bookmaker count unavailable"
    )
    st.caption(f"{count_label} · Snapshot: {madrid_time(market['market_at'])}")
    age = market.get("bet_line_age_minutes")
    if pd.notna(age):
        elapsed = max(0, (pd.Timestamp(now) - market["market_at"]).total_seconds() / 60)
        if age + elapsed >= 30:
            st.caption(
                f":orange[Bet365 quote is approximately {int(age + elapsed)} min old.]"
            )
    if spread:
        st.caption(
            "Spread lines are implied home margins. Signals use the selected market reference."
        )


def _prediction(row: pd.Series, reference: str, *, spread: bool, now) -> None:
    st.caption(row["model_label"])
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(row["signal"], color=SIGNAL_COLORS[row["signal"]])
        if row["phase"] == "Intermediate":
            st.caption(f"{horizon_label(row['model_horizon_minutes'])} model")
    if spread:
        left, right = st.columns(2)
        left.metric(
            "Predicted spread error",
            number(row["predicted_error"], signed=True),
            help="Original error relative to the line used by the model.",
        )
        right.metric(
            f"Edge vs {reference}", number(row["comparison_edge"], signed=True)
        )
    else:
        left, right = st.columns(2)
        left.metric("Predicted total", number(row["predicted_total"]))
        right.metric(
            f"Edge vs {reference}", number(row["comparison_edge"], signed=True)
        )
        st.caption(
            f"Error vs model line: {number(row['predicted_error'], signed=True)} pts"
        )
    st.caption(f"Predicted: {madrid_time(row['predicted_at'])}")
    if (pd.Timestamp(now) - row["predicted_at"]).total_seconds() > 3600:
        st.caption(":orange[Prediction is more than 1 hour old.]")


def render_game_card(
    game: pd.Series,
    predictions: pd.DataFrame,
    market: pd.Series | None,
    families: list[ModelFamily],
    reference: str,
    *,
    spread: bool,
    show_chart: bool,
    now,
) -> None:
    with st.container(border=True):
        away, home = st.columns(2)
        with away:
            _team(game["away_team"], game["away_team_id"], "Away")
        with home:
            _team(game["home_team"], game["home_team_id"], "Home")
        st.markdown(
            f"**{madrid_time(game['tipoff_at'])}** · {time_remaining(game['tipoff_at'], now)}"
        )
        _market(market, spread=spread, now=now)

        closing, intermediate = st.columns(2)
        for column, phase in ((closing, "Closing"), (intermediate, "Intermediate")):
            with column:
                st.markdown(f"**{phase} predictions**")
                for family in families:
                    rows = (
                        predictions.loc[
                            predictions["family_key"].eq(family.key)
                            & predictions["phase"].eq(phase)
                        ]
                        if not predictions.empty
                        else predictions
                    )
                    if rows.empty:
                        st.caption(f"{family.label} · Not available")
                    else:
                        _prediction(rows.iloc[0], reference, spread=spread, now=now)
                if not families:
                    st.caption("Not available")

        if not predictions.empty and show_chart:
            chart = comparison_chart(
                predictions, market, spread=spread, reference=reference
            )
            if chart is not None:
                st.altair_chart(chart, width="stretch")

        # Gate optional content explicitly: collapsed expanders in 1.54 still run.
        if st.toggle(
            "View details", key=f"details_{int(game['game_id'])}", value=False
        ):
            if predictions.empty:
                st.caption("No prediction details available.")
            else:
                details = prediction_table(predictions, reference, spread=spread)
                details["Model reference line"] = predictions["model_line"]
                details["Original output"] = predictions["prediction_value"]
                details["Input mode"] = predictions["input_mode"].map(
                    {0: "Closing", 1: "Own snapshot", 2: "Latest snapshot"}
                )
                details["Input horizon"] = predictions["snapshot_horizon_minutes"].map(
                    horizon_label
                )
                details["Input timestamp (Madrid)"] = predictions["input_at"].map(
                    madrid_time
                )
                details["Fit"] = predictions["fit_id"]
                st.dataframe(details, hide_index=True, width="stretch")
                st.caption(
                    "Market comparisons preserve the original model anchor and use the shared snapshot above. Closing identifies a model type, not a final market close."
                )
