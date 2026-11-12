"""A slate's bets as other pages show them: a graded table and one line per bet."""

from __future__ import annotations

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain import bets as bet_rules

RESULT_COLORS = {"win": "green", "loss": "red"}


def graded_bets_table(bets: pd.DataFrame) -> pd.DataFrame:
    """Game, bet, odds, stake, result and profit of each bet."""
    return pd.DataFrame(
        {
            "Game": bets["away_team"] + " @ " + bets["home_team"],
            "Bet": [
                bet_rules.describe_bet(int(m), int(s), float(line), h, a)
                for m, s, line, h, a in bets[
                    ["market", "side", "line", "home_team", "away_team"]
                ].itertuples(index=False)
            ],
            "Odds": bets["odds_decimal"].astype(float),
            "Stake (€)": bets["stake"].astype(float),
            "Result": bets["result"].str.capitalize(),
            "Profit (€)": bets["profit"].astype(float),
        }
    )


def bet_line(bet: pd.Series) -> str:
    """``Over 230.5 · 10 € at 1.91 · Win +9.09 €``, coloured by its result."""
    text = bet_rules.describe_bet(
        int(bet["market"]),
        int(bet["side"]),
        float(bet["line"]),
        bet["home_team"],
        bet["away_team"],
    )
    result = str(bet["result"])
    outcome = result.capitalize()
    if pd.notna(bet["profit"]) and result in ("win", "loss"):
        outcome += f" {float(bet['profit']):+.2f} €"
    color = RESULT_COLORS.get(result, "gray")
    return (
        f"**{text}** · {float(bet['stake']):g} € at {float(bet['odds_decimal']):.2f} "
        f"· :{color}[**{outcome}**]"
    )


def render_slate_bets(bets: pd.DataFrame, *, title: str = "Your bets") -> None:
    """The user's bets on a slate: what was staked, what it returned, and the
    graded list."""
    st.subheader(title)
    totals = bet_rules.slate_summary(bets)
    record = bet_rules.summary(bets)
    with st.container(horizontal=True, key="slate_bets_metrics"):
        st.metric("Bets", totals["bets"], border=True)
        st.metric("Staked", f"{totals['staked']:.2f} €", border=True)
        if totals["settled"]:
            st.metric("Profit", f"{totals['profit']:+.2f} €", border=True)
            st.metric(
                "W-L-P",
                f"{record['wins']}-{record['losses']}-{record['pushes']}",
                border=True,
            )
        if record["pending"]:
            st.metric("To win", f"{totals['to_win']:.2f} €", border=True)
    st.dataframe(
        graded_bets_table(bets), hide_index=True, width="stretch", alt="Your bets"
    )
