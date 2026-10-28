"""One card per game: the bet, market lines, and model predictions."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import REFERENCE_COLUMNS, SIGNAL_COLORS
from nba_ou.dashboard.domain.dates import (
    horizon_label,
    madrid_clock,
    madrid_time,
    time_remaining,
)
from nba_ou.dashboard.domain.models import ModelFamily
from nba_ou.dashboard.domain.picks import GamePick, bet_text
from nba_ou.dashboard.domain.teams import full_team_name, team_logo_url
from nba_ou.dashboard.ui.charts import comparison_chart
from nba_ou.dashboard.ui.formatting import (
    compact_totals_table,
    number,
    prediction_table,
)
from nba_ou.dashboard.ui.totals_table import totals_table_html

# Beyond this the stored line may no longer be the one on offer.
STALE_SNAPSHOT_MINUTES = 60


def _team(name: str, team_id, role: str) -> None:
    name = full_team_name(name, team_id)
    with st.container(horizontal=True, vertical_alignment="center"):
        logo = team_logo_url(name, team_id)
        if logo:
            # Streamlit 1.54 has no image alt parameter; the visible caption
            # names the side, and the adjacent text remains usable without images.
            st.image(logo, width=48, caption=role)
        st.markdown(f"**{name}**")


def _headline(
    pick: GamePick, market: pd.Series | None, reference: str, game, *, spread: bool
) -> None:
    if pick.status == "BET":
        line = market.get(REFERENCE_COLUMNS[reference]) if market is not None else None
        text = bet_text(pick.side, line, home=game["home_team"], away=game["away_team"])
        headline = f"### :{SIGNAL_COLORS[pick.side]}[{text}]"
        detail = None
    else:
        text, detail = {
            "TIE": ("No bet: tied vote", None),
            "NO BET": ("No bet", "No prediction clears its minimum edge."),
            "NO LINE": (
                "No bet: no line",
                f"No stored {reference} line to bet against.",
            ),
            "PENDING": ("No predictions yet", "Waiting for the next prediction run."),
        }[pick.status]
        headline = f"### :gray[{text}]"

    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(headline, width="content")
        if market is not None:
            source = reference
            if pick.status != "BET":
                source += f" {number(market.get(REFERENCE_COLUMNS[reference]))}"
            if reference != "Bet365":
                source += f" · Bet365 {number(market.get('bet365_line'))}"
            count = market.get("books_count")
            books = f" · {int(count)} books" if pd.notna(count) else ""
            st.caption(
                ("Implied home margin · " if spread else "")
                + f"{source} · mean {number(market.get('books_mean_line'))} · "
                f"median {number(market.get('books_median_line'))}{books}",
                width="content",
            )
    if pick.counts:
        st.markdown(
            " · ".join(
                f":{SIGNAL_COLORS[side] if side == pick.side else 'gray'}[**{count} {side}**]"
                for side, count in pick.counts.items()
            )
        )
    if detail:
        st.caption(detail)


def _phase(row: pd.Series) -> str:
    if row["phase"] == "Closing":
        return "Closing"
    return horizon_label(row["model_horizon_minutes"])


def _predictions(
    predictions: pd.DataFrame,
    families: list[ModelFamily],
    missing_models: tuple[str, ...],
    *,
    spread: bool,
) -> None:
    if not spread:
        tables = []
        labels = []
        unavailable = []
        for family in families:
            rows = (
                predictions.loc[predictions["family_key"].eq(family.key)]
                if not predictions.empty
                else predictions
            )
            if rows.empty:
                unavailable.append(family.label)
                continue
            tables.append(compact_totals_table(rows))
            labels.append(family.label)
        if tables:
            st.html(totals_table_html(pd.concat(tables, keys=labels, names=["Model"])))
        for label in unavailable:
            st.caption(f"{label} · no prediction for this game")
        for name in missing_models:
            st.caption(f"{name} · no prediction yet for tonight")
        return
    widths = [4, 2, 2, 2]
    head = st.columns(widths)
    for column, label in zip(
        head,
        ("Model", "Pred. margin" if spread else "Pred. total", "Edge", "Pick"),
        strict=True,
    ):
        column.caption(label)
    for _, row in predictions.iterrows():
        name, value, edge, pick = st.columns(widths, vertical_alignment="center")
        name.markdown(
            f"**{row['model_name']} · {row['target_label']}** · {_phase(row)}",
            help=f"{row['model_label']}. Predicted {madrid_time(row['predicted_at'])}. "
            f"Bets when |edge| > {row['min_edge']:g} pts.",
        )
        value.markdown(
            number(row["predicted_home_margin" if spread else "predicted_total"])
        )
        edge.markdown(number(row["comparison_edge"], signed=True))
        with pick:
            st.badge(row["signal"], color=SIGNAL_COLORS[row["signal"]])
    shown = set(predictions["family_key"]) if not predictions.empty else set()
    for family in families:
        if family.key not in shown:
            st.caption(f"{family.label} · no prediction for this game")
    for name in missing_models:
        st.caption(f"{name} · no prediction yet for tonight")


def _market_freshness(market: pd.Series | None, *, now) -> None:
    if market is None:
        st.caption("No stored Bet365 line. No substitute line has been assumed.")
        return
    read_at = market["market_at"]
    # The line history stores changes only: the quote's age at the snapshot is
    # how long Bet365 had gone without moving line or price.
    age = market.get("bet_line_age_minutes")
    moved = (
        f"Last Bet365 move {madrid_clock(read_at - pd.Timedelta(minutes=age))} · "
        if pd.notna(age)
        else ""
    )
    elapsed = max(0, (pd.Timestamp(now) - read_at).total_seconds() / 60)
    snapshot = f"line read {madrid_clock(read_at)} Madrid"
    if elapsed >= STALE_SNAPSHOT_MINUTES:
        st.caption(
            f"{moved}:orange[{snapshot}, {int(elapsed)} min ago: "
            "check the live price before betting.]"
        )
    else:
        st.caption(f"{moved}{snapshot}")


def render_game_card(
    game: pd.Series,
    predictions: pd.DataFrame,
    market: pd.Series | None,
    families: list[ModelFamily],
    reference: str,
    pick: GamePick,
    *,
    spread: bool,
    show_chart: bool,
    now,
    missing_models: tuple[str, ...] = (),
) -> None:
    with st.container(border=True):
        away, home = st.columns(2)
        with away:
            _team(game["away_team"], game["away_team_id"], "Away")
        with home:
            _team(game["home_team"], game["home_team_id"], "Home")
        st.caption(
            f"**{madrid_time(game['tipoff_at'])}** · {time_remaining(game['tipoff_at'], now)}"
        )
        _headline(pick, market, reference, game, spread=spread)
        _market_freshness(market, now=now)
        _predictions(predictions, families, missing_models, spread=spread)

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
