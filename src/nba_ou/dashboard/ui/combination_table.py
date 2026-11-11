"""Heatmap table of combinations: mean edge and majority vote side by side."""

from collections.abc import Callable

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.analysis import BREAK_EVEN, MIN_CELL_BETS
from nba_ou.dashboard.ui.controls import (
    coin_flip_metric,
    permutation_metric,
    profit_metric,
)
from nba_ou.dashboard.ui.formatting import rate_style

MEAN_WIN = "Mean · Win %"
MEAN_BETS = "Mean · Bets"
VOTE_WIN = "Majority · Win %"
VOTE_BETS = "Majority · Bets"
_METHODS = (("mean", MEAN_WIN, MEAN_BETS), ("vote", VOTE_WIN, VOTE_BETS))
_LOW_SAMPLE = "color: #8a8f98"


def display_table(
    table: pd.DataFrame, label: Callable[[int], str], timepoint_title: str
) -> pd.DataFrame:
    """The readable columns of ``combination_table``, numbers kept numeric."""
    display = pd.DataFrame(
        {
            timepoint_title: table["timepoint"].map(label),
            "Slots": table["member_names"],
            "Games": table["games"].astype(int),
        }
    )
    for prefix, win, bets in _METHODS:
        display[win] = table[f"{prefix}_win_pct"]
        display[bets] = table[f"{prefix}_bets"].fillna(0).astype(int)
    return display


def _bets_style(bets: float, most: float) -> str:
    if not most or pd.isna(bets) or not bets:
        return ""
    alpha = 0.08 + 0.47 * bets / most
    return f"background-color: rgba(56, 132, 255, {alpha:.2f})"


def styled_combination_table(table: pd.DataFrame, display: pd.DataFrame):
    """Win % green above break-even and red below, grey text under the minimum
    settled bets; bets shaded by how many games the method could bet."""
    most = float(display[[MEAN_BETS, VOTE_BETS]].to_numpy().max(initial=0))

    def styles(frame: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for prefix, win, bets in _METHODS:
            eligible = table[f"{prefix}_eligible"].to_numpy()
            css[win] = [
                "" if pd.isna(value) else rate_style(value) if ok else _LOW_SAMPLE
                for value, ok in zip(frame[win], eligible, strict=True)
            ]
            css[bets] = [_bets_style(value, most) for value in frame[bets]]
        return css

    return display.style.apply(styles, axis=None).format(
        {MEAN_WIN: "{:.1f}%", VOTE_WIN: "{:.1f}%"}, na_rep="—"
    )


def render_combination_table(
    table: pd.DataFrame,
    *,
    label: Callable[[int], str],
    timepoint_title: str,
    alt: str,
) -> None:
    display = display_table(table, label, timepoint_title)
    st.dataframe(
        styled_combination_table(table, display),
        hide_index=True,
        width="stretch",
        height=min(35 * (len(display) + 1) + 3, 600),
        column_config={
            timepoint_title: st.column_config.TextColumn(width="small", pinned=True),
            "Slots": st.column_config.TextColumn(width="large"),
            "Games": st.column_config.NumberColumn(
                help="Final games where every slot of the combination had a line."
            ),
            MEAN_WIN: st.column_config.NumberColumn(
                help="Mean of the members' edges, bet with the minimum edge of the "
                "member published last. Wins / (wins + losses)."
            ),
            MEAN_BETS: st.column_config.NumberColumn(
                help="Settled bets the mean edge placed (wins, losses and pushes)."
            ),
            VOTE_WIN: st.column_config.NumberColumn(
                help="More than half of the members on one side (with two, both "
                "agree); otherwise no bet. Wins / (wins + losses)."
            ),
            VOTE_BETS: st.column_config.NumberColumn(
                help="Settled bets the majority vote placed (wins, losses and "
                "pushes): the games where a majority agreed."
            ),
        },
        alt=alt,
    )
    st.caption(
        f"Win %: green beats the {BREAK_EVEN:.2f}% break-even at −110, red does not; "
        f"grey text has fewer than {MIN_CELL_BETS} wins + losses. Bets: darker blue "
        "means more games could be bet. Click a column header to sort."
    )


def combination_leader_metrics(leader, chance, permutation, tried: int) -> None:
    """Win %, P(profitable) and the selection checks of the leading combination."""
    with st.container(horizontal=True):
        st.metric(
            "Best combination · win %",
            f"{leader['win_pct']:.1f}%",
            border=True,
            width="content",
            help=f"{leader['strategy']}: {leader['wins']:.0f} wins / "
            f"{leader['losses']:.0f} losses / {leader['pushes']:.0f} pushes.",
        )
        profit_metric(leader["wins"], leader["losses"], border=True, width="content")
        coin_flip_metric(chance, tried, leader["win_pct"], "combinations")
        permutation_metric(permutation, tried, "combinations")
