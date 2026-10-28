from pathlib import Path

from nba_ou.dashboard.ui.today import render_today

import streamlit as st

with st.container(horizontal=True, vertical_alignment="center", key="today_header"):
    st.image(Path(__file__).resolve().parents[1] / "assets/nba-mark.svg", width=22)
    st.title("Today's predictions", width="content")
render_today()
