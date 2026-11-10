"""Shared controls with page-specific session keys."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import (
    DEFAULT_MIN_EDGE_BY_KIND,
    MIN_EDGE_LEVELS,
    MODEL_NAMES,
)


def minimum_edge_select(kind: int, *, page: str) -> float:
    default = DEFAULT_MIN_EDGE_BY_KIND[kind]
    return st.selectbox(
        f"Minimum edge · {MODEL_NAMES[kind]} (points)",
        MIN_EDGE_LEVELS,
        index=MIN_EDGE_LEVELS.index(default),
        key=f"{page}_min_edge_{MODEL_NAMES[kind].lower()}",
        help="A prediction bets only when its absolute market edge is strictly "
        "above this threshold. All predictions remain visible.",
    )


def permutation_metric(p_value: float, tried: int, what: str) -> None:
    st.metric(
        "Game-permutation p",
        # 2,000 simulations cannot resolve below 1/2001.
        (
            "—"
            if pd.isna(p_value)
            else "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"
        ),
        border=True,
        width="content",
        help=f"Selection-aware p-value of the leader among {tried} {what}: each "
        "game's result is flipped at random, the same flip for every "
        f"{what[:-1]} that bet it, so shared games stay shared. Share of "
        "simulations whose best reaches the observed win %. Small means the "
        "leader is unlikely to be luck alone.",
    )


def coin_flip_metric(chance: float, tried: int, win_pct: float, what: str) -> None:
    st.metric(
        "Coin-flip chance",
        "—" if pd.isna(chance) else f"{100 * chance:.0f}%",
        border=True,
        width="content",
        help=f"How often the best of {tried} coin-flip {what} with these bet counts "
        f"reaches {win_pct:.1f}%. High means the leader is indistinguishable from "
        "luck. Cells share games, so it is a conservative guide, not a p-value.",
    )
