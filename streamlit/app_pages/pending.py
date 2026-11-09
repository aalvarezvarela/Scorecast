from nba_ou.dashboard.domain.access import Access
from nba_ou.dashboard.ui.auth import refresh_user, sign_out_button

import streamlit as st

st.title("Access pending")
if st.session_state.get("current_access") == Access.DISABLED:
    st.warning("This account is disabled.", icon=":material/block:")
else:
    st.info(
        "You are signed in. The administrator has to approve your account "
        "before you can use the app.",
        icon=":material/hourglass_top:",
    )
with st.container(horizontal=True):
    if st.button("Check again", icon=":material/refresh:"):
        refresh_user()
        st.rerun()
    sign_out_button(key="pending_sign_out")
