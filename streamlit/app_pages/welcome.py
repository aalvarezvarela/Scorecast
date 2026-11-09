from pathlib import Path

from nba_ou.dashboard.ui.auth import sign_in_button

import streamlit as st

with st.container(horizontal=True, vertical_alignment="center"):
    st.image(Path(__file__).resolve().parents[1] / "assets/nba-mark.svg", width=22)
    st.title("NBA predictions", width="content")
st.markdown(
    "Model predictions for tonight's NBA totals and spreads, with the bets "
    "you place on them. Sign in to continue."
)
sign_in_button()
st.caption(
    "New accounts are approved by the administrator before they can see the app."
)
