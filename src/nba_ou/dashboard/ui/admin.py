"""Admin: approve the accounts that signed in, or disable one."""

from __future__ import annotations

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.access import AppUser
from nba_ou.dashboard.domain.dates import madrid_time
from nba_ou.dashboard.repositories import app_store
from nba_ou.dashboard.repositories.connection import open_app_connection
from nba_ou.dashboard.ui.app_loaders import load_users
from nba_ou.dashboard.ui.auth import app_secrets
from nba_ou.postgre_db.app_aiven.schema import (
    USER_ACTIVE,
    USER_DISABLED,
    USER_PENDING,
)

STATUS_LABELS = {
    USER_PENDING: "Pending",
    USER_ACTIVE: "Active",
    USER_DISABLED: "Disabled",
}


def render_admin(admin: AppUser) -> None:
    users = load_users()
    pending = int(users["status"].eq(USER_PENDING).sum()) if not users.empty else 0
    st.caption(f"{len(users)} account(s), {pending} waiting for approval.")
    if users.empty:
        return
    st.dataframe(
        pd.DataFrame(
            {
                "Email": users["email"],
                "Name": users["display_name"].fillna(""),
                "Status": users["status"].map(STATUS_LABELS),
                "Admin": users["is_admin"],
                "Created": users["created_at"].map(madrid_time),
                "Last sign-in": users["last_login_at"].map(madrid_time),
            }
        ),
        hide_index=True,
        width="stretch",
        alt="Dashboard accounts",
    )
    others = users.loc[users["user_id"].ne(admin.user_id)]
    if others.empty:
        return
    with st.container(horizontal=True, vertical_alignment="bottom"):
        user_id = st.selectbox(
            "Account",
            options=others["user_id"].tolist(),
            format_func=lambda uid: _account(others, uid),
            key="admin_account",
        )
        for status, label, icon in (
            (USER_ACTIVE, "Approve", ":material/check:"),
            (USER_DISABLED, "Disable", ":material/block:"),
        ):
            if st.button(label, icon=icon, key=f"admin_{label.lower()}"):
                with open_app_connection(app_secrets().app_db_url) as conn:
                    app_store.set_user_status(conn, int(user_id), status, by=admin)
                load_users.clear()
                st.rerun()


def _account(users: pd.DataFrame, user_id: int) -> str:
    row = users.loc[users["user_id"].eq(user_id)].iloc[0]
    return f"{row['email']} ({STATUS_LABELS.get(int(row['status']), '?')})"
