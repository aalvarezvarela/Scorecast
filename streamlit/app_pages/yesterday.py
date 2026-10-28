from pathlib import Path

from nba_ou.dashboard.ui.yesterday import render_yesterday

import streamlit as st

with st.container(horizontal=True, vertical_alignment="center", key="yesterday_header"):
    st.image(Path(__file__).resolve().parents[1] / "assets/nba-mark.svg", width=22)
    st.title("Yesterday's predictions", width="content")
render_yesterday()
