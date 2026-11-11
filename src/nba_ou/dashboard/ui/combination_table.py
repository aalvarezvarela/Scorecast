"""Heatmap table of combinations: mean edge and majority vote side by side."""

from collections.abc import Callable

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.analysis import BREAK_EVEN, MIN_CELL_BETS
from nba_ou.dashboard.domain.strategies import TOP_COMBINATIONS
from nba_ou.dashboard.ui.controls import (
    coin_flip_metric,
    permutation_metric,
    profit_metric,
)
from nba_ou.dashboard.ui.formatting import rate_style

MEAN_WIN = "Mean · Win %"
MEAN_BETS = "Mean · Bets"
MEAN_SHARE = "Mean · Bets % of games"
VOTE_WIN = "Majority · Win %"
VOTE_BETS = "Majority · Bets"
VOTE_SHARE = "Majority · Bets % of games"
_METHODS = (
    ("mean", MEAN_WIN, MEAN_BETS, MEAN_SHARE),
    ("vote", VOTE_WIN, VOTE_BETS, VOTE_SHARE),
)
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
    games = display["Games"]
    for prefix, win, bets, share in _METHODS:
        display[win] = table[f"{prefix}_win_pct"]
        display[bets] = table[f"{prefix}_bets"].fillna(0).astype(int)
        display[share] = (100 * display[bets] / games).where(games.gt(0))
    return display


def _bets_style(value: float, most: float) -> str:
    if not most or pd.isna(value) or not value:
        return ""
    alpha = 0.08 + 0.47 * min(value / most, 1)
    return f"background-color: rgba(56, 132, 255, {alpha:.2f})"


def styled_combination_table(table: pd.DataFrame, display: pd.DataFrame):
    """Win % green above break-even and red below, grey text under the minimum
    settled bets; bets shaded by how many games the method could bet, and their
    share of the games by that share."""
    most = float(display[[MEAN_BETS, VOTE_BETS]].to_numpy().max(initial=0))

    def styles(frame: pd.DataFrame) -> pd.DataFrame:
        css = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for prefix, win, bets, share in _METHODS:
            eligible = table[f"{prefix}_eligible"].to_numpy()
            css[win] = [
                "" if pd.isna(value) else rate_style(value) if ok else _LOW_SAMPLE
                for value, ok in zip(frame[win], eligible, strict=True)
            ]
            css[bets] = [_bets_style(value, most) for value in frame[bets]]
            css[share] = [_bets_style(value, 100) for value in frame[share]]
        return css

    percent = {
        column: "{:.1f}%" for column in (MEAN_WIN, VOTE_WIN, MEAN_SHARE, VOTE_SHARE)
    }
    return display.style.apply(styles, axis=None).format(percent, na_rep="—")


def render_combination_table(
    table: pd.DataFrame,
    *,
    label: Callable[[int], str],
    timepoint_title: str,
    alt: str,
) -> None:
    """The ``TOP_COMBINATIONS`` first rows of ``combination_table``."""
    shown = table.head(TOP_COMBINATIONS).reset_index(drop=True)
    display = display_table(shown, label, timepoint_title)
    st.dataframe(
        styled_combination_table(shown, display),
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
            MEAN_SHARE: st.column_config.NumberColumn(
                help="Mean · Bets as a share of Games."
            ),
            VOTE_BETS: st.column_config.NumberColumn(
                help="Settled bets the majority vote placed (wins, losses and "
                "pushes): the games where a majority agreed."
            ),
            VOTE_SHARE: st.column_config.NumberColumn(
                help="Majority · Bets as a share of Games: how often a majority "
                "agreed."
            ),
        },
        alt=alt,
    )
    st.caption(
        f"Top {len(shown)} of {len(table)} combinations, those with "
        f"{MIN_CELL_BETS} wins + losses in a method first, by their best win %. "
        f"Win %: green beats the {BREAK_EVEN:.2f}% break-even at −110, red does not; "
        f"grey text has fewer than {MIN_CELL_BETS} wins + losses. Bets: darker blue "
        "means more games could be bet; % of games is that count over Games."
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
