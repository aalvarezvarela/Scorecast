"""The sign-in gate: Google identity (``st.login``) plus the ``app.users`` row.

Without a signed-in, approved account nothing but the welcome (or pending)
page is built: ``streamlit_app.py`` lists the other pages only for
``can_use_app``. Secrets (Streamlit): ``[auth]`` for Google,
``AIVEN_APP_DB_URL`` (the ``dashboard_app`` role), ``GITHUB_DISPATCH_TOKEN``
and ``ADMIN_EMAILS``; see ``docs/serving/streamlit_dashboard.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

import streamlit as st
from nba_ou.dashboard.domain.access import (
    Access,
    AppUser,
    access_of,
    admin_emails,
    normalize_email,
)
from nba_ou.dashboard.repositories.app_store import ensure_user, read_user
from nba_ou.dashboard.repositories.connection import open_app_connection

USER_KEY = "app_user"
#: An approved account's status is read again this often, so a disabled
#: account loses access within minutes, not at its next session.
USER_RECHECK_SECONDS = 120


@dataclass(frozen=True)
class AppSecrets:
    app_db_url: str
    dispatch_token: str
    admins: frozenset[str]


def _secret(name: str, default=""):
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def app_secrets() -> AppSecrets:
    return AppSecrets(
        app_db_url=str(_secret("AIVEN_APP_DB_URL") or ""),
        dispatch_token=str(_secret("GITHUB_DISPATCH_TOKEN") or ""),
        admins=admin_emails(_secret("ADMIN_EMAILS", None)),
    )


def identity() -> tuple[str, str | None] | None:
    """The signed-in Google account's (email, name), or None."""
    # Without an [auth] section (tests, local runs) st.user has no
    # is_logged_in at all: signed out.
    if not st.user.get("is_logged_in", False):
        return None
    return str(st.user.get("email") or ""), st.user.get("name")


@st.cache_data(ttl=USER_RECHECK_SECONDS, max_entries=64, show_spinner=False)
def _stored_user(email: str, dsn: str) -> AppUser | None:
    with open_app_connection(dsn) as conn:
        return read_user(conn, email)


def current_user() -> AppUser | None:
    """The signed-in account's ``app.users`` row; the first call of a session
    creates or refreshes it (``ensure_user``). None when signed out."""
    signed_in = identity()
    if signed_in is None:
        st.session_state.pop(USER_KEY, None)
        return None
    email, name = signed_in
    secrets = app_secrets()
    cached = st.session_state.get(USER_KEY)
    if cached is not None and cached.email == normalize_email(email):
        stored = _stored_user(cached.email, secrets.app_db_url)
        return stored or cached
    with open_app_connection(secrets.app_db_url) as conn:
        user = ensure_user(conn, email, name, admins=secrets.admins)
    st.session_state[USER_KEY] = user
    return user


def refresh_user() -> None:
    """Read the account again on the next run (after an approval)."""
    st.session_state.pop(USER_KEY, None)
    _stored_user.clear()


def resolve_access() -> tuple[Access, AppUser | None]:
    """The gate. A missing or unreachable account store is no access."""
    try:
        user = current_user()
    except (psycopg.Error, ValueError) as exc:
        st.error(
            "Accounts are unavailable right now. "
            + (
                "AIVEN_APP_DB_URL is not configured."
                if isinstance(exc, ValueError) and "AIVEN_APP_DB_URL" in str(exc)
                else "Try again in a minute."
            ),
            icon=":material/lock:",
        )
        return Access.PENDING, None
    return access_of(user), user


def sign_in_button() -> None:
    if st.button("Sign in with Google", icon=":material/login:", type="primary"):
        try:
            st.login()
        except Exception:  # noqa: BLE001 -- an unconfigured [auth] section
            st.error("Sign-in is not configured on this deployment.")


def sign_out_button(key: str = "sign_out") -> None:
    if st.button("Sign out", icon=":material/logout:", key=key):
        st.session_state.pop(USER_KEY, None)
        st.logout()
