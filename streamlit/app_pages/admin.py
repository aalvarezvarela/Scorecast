from nba_ou.dashboard.ui.admin import render_admin

import streamlit as st

st.title("Accounts")
render_admin(st.session_state["current_user"])
