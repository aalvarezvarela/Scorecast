"""Cached reads of schema ``app`` for the signed-in pages.

Short TTLs: a request's status moves within minutes, and the Bets page reads
its requests and their predictions every 15 seconds (below that, so each
refresh reads the store). Every write clears the cache it changes
(``clear_requests``, ``clear_bets``).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

import streamlit as st
from nba_ou.dashboard.repositories import app_store
from nba_ou.dashboard.repositories.connection import open_app_connection
from nba_ou.dashboard.ui.auth import app_secrets


def _dsn() -> str:
    return app_secrets().app_db_url


@st.cache_data(ttl=10, max_entries=32, show_spinner=False)
def load_requests(user_id: int, game_date: date) -> pd.DataFrame:
    with open_app_connection(_dsn()) as conn:
        return app_store.user_requests(conn, user_id, game_date)


@st.cache_data(ttl=10, max_entries=32, show_spinner=False)
def load_manual_predictions(user_id: int, game_date: date) -> pd.DataFrame:
    with open_app_connection(_dsn()) as conn:
        return app_store.manual_predictions(conn, user_id, game_date)


@st.cache_data(ttl=60, max_entries=32, show_spinner=False)
def load_bets(user_id: int, game_date: date) -> pd.DataFrame:
    """The user's bets on the slate (voided ones left out)."""
    with open_app_connection(_dsn()) as conn:
        return app_store.slate_bets(conn, user_id, game_date)


@st.cache_data(ttl=60, max_entries=32, show_spinner=False)
def load_last_stake(user_id: int) -> float | None:
    with open_app_connection(_dsn()) as conn:
        return app_store.last_stake(conn, user_id)


@st.cache_data(ttl=30, max_entries=4, show_spinner=False)
def load_users() -> pd.DataFrame:
    with open_app_connection(_dsn()) as conn:
        return app_store.list_users(conn)


def clear_requests() -> None:
    load_requests.clear()
    load_manual_predictions.clear()


def clear_bets() -> None:
    load_bets.clear()
