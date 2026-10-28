"""Unified Aiven dashboard. Run from the repository root."""

import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent / "src"))

st.set_page_config(
    page_title="NBA · Predictions",
    page_icon=":material/sports_basketball:",
    layout="wide",
)

for key, value in {
    "today_target": 2,
    "today_view": "Cards",
    "today_reference": "Bet365",
    "today_charts": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

page = st.navigation(
    [
        st.Page(
            str(APP_DIR / "app_pages/today.py"),
            title="Today's predictions",
            icon=":material/sports_basketball:",
            default=True,
        ),
        st.Page(
            str(APP_DIR / "app_pages/yesterday.py"),
            title="Yesterday's predictions",
            icon=":material/history:",
        ),
        st.Page(
            str(APP_DIR / "app_pages/historical.py"),
            title="Historical analysis",
            icon=":material/analytics:",
        ),
    ],
    position="top",
)
page.run()
