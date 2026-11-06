from pathlib import Path

from nba_ou.dashboard.domain.dates import slate_date
from nba_ou.dashboard.ui.bets import render_bets

import streamlit as st

with st.container(horizontal=True, vertical_alignment="center"):
    st.image(Path(__file__).resolve().parents[1] / "assets/nba-mark.svg", width=22)
    st.title("Bets", width="content")
render_bets(st.session_state["current_user"], slate_date())
