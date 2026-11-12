from pathlib import Path

from nba_ou.dashboard.ui.bet_history import render_bet_history

import streamlit as st

with st.container(horizontal=True, vertical_alignment="center"):
    st.image(Path(__file__).resolve().parents[1] / "assets/nba-mark.svg", width=22)
    st.title("Bet history", width="content")
render_bet_history(st.session_state["current_user"])
