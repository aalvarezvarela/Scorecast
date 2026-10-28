"""Shared controls with page-specific session keys."""

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
