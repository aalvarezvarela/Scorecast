"""Bet history: the user's own bets over time.

Money staked, returned and won on the settled bets, the running profit by
slate, and the same figures by market and side, by the model that was
followed and by month. Deleted (voided) bets are not part of it.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import psycopg

import streamlit as st
from nba_ou.dashboard.config import SIGNAL_HEX_COLORS
from nba_ou.dashboard.domain import bet_history as history_rules
from nba_ou.dashboard.domain.access import AppUser
from nba_ou.dashboard.domain.dates import madrid_time
from nba_ou.dashboard.ui.app_loaders import load_bet_history
from nba_ou.dashboard.ui.bet_summary import graded_bets_table

PROFIT_COLOR = "#77E4A4"
LOSS_COLOR = SIGNAL_HEX_COLORS["OVER"]
NEUTRAL_COLOR = SIGNAL_HEX_COLORS["NO EDGE"]


def profit_chart(days: pd.DataFrame):
    """The running profit by slate (line) over each slate's own profit (bars)."""
    data = days.assign(
        game_date=pd.to_datetime(days["game_date"]),
        outcome=days["profit"].ge(0).map({True: "Winning day", False: "Losing day"}),
    )
    tooltip = [
        alt.Tooltip("game_date:T", title="Slate", format="%d %b %Y"),
        alt.Tooltip("bets:Q", title="Settled bets"),
        alt.Tooltip("staked:Q", title="Staked (€)", format=".2f"),
        alt.Tooltip("profit:Q", title="Day profit (€)", format="+.2f"),
        alt.Tooltip("total_profit:Q", title="Running profit (€)", format="+.2f"),
    ]
    x = alt.X("game_date:T", title=None, axis=alt.Axis(format="%d %b"))
    bars = (
        alt.Chart(data)
        .mark_bar(opacity=0.45)
        .encode(
            x=x,
            y=alt.Y("profit:Q", title="Profit (€)"),
            color=alt.Color(
                "outcome:N",
                title=None,
                scale=alt.Scale(
                    domain=["Winning day", "Losing day"],
                    range=[PROFIT_COLOR, LOSS_COLOR],
                ),
            ),
            tooltip=tooltip,
        )
    )
    line = (
        alt.Chart(data)
        .mark_line(color="#FFD166", point=alt.OverlayMarkDef(color="#FFD166"))
        .encode(x=x, y=alt.Y("total_profit:Q"), tooltip=tooltip)
    )
    zero = (
        alt.Chart(pd.DataFrame({"value": [0.0]}))
        .mark_rule(color=NEUTRAL_COLOR, opacity=0.7)
        .encode(y="value:Q")
    )
    return (
        (zero + bars + line)
        .properties(height=280)
        .configure_legend(orient="bottom", title=None)
        .configure_view(strokeWidth=0)
    )


def _metrics(totals: dict[str, float]) -> None:
    with st.container(horizontal=True, key="bet_history_money"):
        st.metric("Staked", f"{totals['staked']:.2f} €", border=True)
        st.metric("Returned", f"{totals['returned']:.2f} €", border=True)
        st.metric("Profit", f"{totals['profit']:+.2f} €", border=True)
        st.metric("ROI", f"{totals['roi']:+.1%}", border=True)
    with st.container(horizontal=True, key="bet_history_record"):
        st.metric("Settled bets", totals["settled"], border=True)
        st.metric(
            "W-L-P",
            f"{totals['wins']}-{totals['losses']}-{totals['pushes']}",
            border=True,
        )
        st.metric("Win rate", f"{totals['win_rate']:.1%}", border=True)
        st.metric("Average odds", f"{totals['average_odds']:.2f}", border=True)
        st.metric(
            "Pending",
            f"{totals['pending']} · {totals['pending_stake']:.2f} €",
            border=True,
        )
    st.caption(
        "Staked, returned, profit and ROI count settled bets (win, loss, push). "
        "Win rate = wins / (wins + losses). Pending bets are listed apart with "
        "their stake; deleted bets are left out."
    )


def _all_bets(history: pd.DataFrame) -> pd.DataFrame:
    newest = history.sort_values(["tipoff_at", "bet_id"], ascending=False)
    table = graded_bets_table(newest)
    table.insert(0, "Date", newest["game_date"].to_numpy())
    table["Followed"] = newest["model_label"].to_numpy()
    table["Placed (Madrid)"] = newest["placed_at"].map(madrid_time).to_numpy()
    return table


def render_bet_history(user: AppUser) -> None:
    try:
        bets = load_bet_history(user.user_id)
    except (psycopg.Error, ValueError):
        st.error(
            "The bet history is unavailable right now. Try again in a minute.",
            icon=":material/database:",
        )
        return
    if bets.empty:
        st.info("No bets yet. Place one on the Bets page.")
        return
    history = history_rules.prepare_history(bets)
    first, last = history["game_date"].min(), history["game_date"].max()
    dates_col, market_col = st.columns([3, 2], vertical_alignment="bottom")
    with dates_col:
        picked = st.date_input(
            "Slates",
            value=(first, last),
            min_value=first,
            max_value=last,
            format="DD/MM/YYYY",
            key="bet_history_dates",
            help="NBA slate dates, both included.",
        )
    with market_col:
        market = st.segmented_control(
            "Market",
            list(history_rules.MARKET_FILTERS),
            format_func=history_rules.MARKET_FILTERS.get,
            default=history_rules.ALL_MARKETS,
            key="bet_history_market",
            width="stretch",
        )
    # A range being picked holds one date until the second click.
    start, end = (picked[0], picked[-1]) if picked else (first, last)
    shown = history_rules.filter_history(
        history, start, end, market or history_rules.ALL_MARKETS
    )
    if shown.empty:
        st.info("No bets match these filters.")
        return

    _metrics(history_rules.performance(shown))

    days = history_rules.daily_profit(shown)
    st.subheader("Profit over time")
    if days.empty:
        st.caption("No settled bet yet.")
    else:
        st.altair_chart(profit_chart(days), width="stretch")
        st.caption("Bars: each slate's profit. Line: the running profit.")

    st.subheader("Breakdown")
    grouping = st.segmented_control(
        "Group by",
        list(history_rules.GROUPINGS),
        default=next(iter(history_rules.GROUPINGS)),
        key="bet_history_grouping",
    ) or next(iter(history_rules.GROUPINGS))
    st.dataframe(
        history_rules.breakdown(shown, history_rules.GROUPINGS[grouping]),
        hide_index=True,
        width="stretch",
        column_config={
            "Win %": st.column_config.NumberColumn(format="%.1f"),
            "Staked (€)": st.column_config.NumberColumn(format="%.2f"),
            "Profit (€)": st.column_config.NumberColumn(format="%+.2f"),
            "ROI %": st.column_config.NumberColumn(format="%+.1f"),
        },
        alt="Bet results by group",
    )
    if grouping == "Model followed":
        st.caption(
            "The model whose prediction backed the side with the largest edge "
            "when the bet was placed."
        )

    st.subheader("All bets")
    st.dataframe(
        _all_bets(shown),
        hide_index=True,
        width="stretch",
        column_config={
            "Odds": st.column_config.NumberColumn(format="%.3f"),
            "Stake (€)": st.column_config.NumberColumn(format="%.2f"),
            "Profit (€)": st.column_config.NumberColumn(format="%+.2f"),
        },
        alt="All bets, newest first",
    )
