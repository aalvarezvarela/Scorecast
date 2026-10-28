from pathlib import Path

from nba_ou.dashboard.ui.historical import render_historical

import streamlit as st

with st.container(horizontal=True, vertical_alignment="center"):
    st.image(Path(__file__).resolve().parents[1] / "assets/nba-mark.svg", width=22)
    st.title("Historical analysis", width="content")
render_historical()
