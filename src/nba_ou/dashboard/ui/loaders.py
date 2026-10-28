"""Cache source reads, leaving time-sensitive selection outside the cache."""

from datetime import date

import streamlit as st
from nba_ou.dashboard.config import CACHE_TTL_SECONDS
from nba_ou.dashboard.repositories.analysis import AnalysisData, read_history_range
from nba_ou.dashboard.repositories.predictions import (
    TodayData,
    read_today,
    read_yesterday,
)


def _aiven_secret() -> str:
    try:
        return str(
            st.secrets.get("AIVEN_DB_URL")
            or st.secrets.get("Aiven", {}).get("AIVEN_DB_URL", "")
        )
    except FileNotFoundError:
        return ""


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=8, show_spinner=False)
def load_today(game_date: date) -> TodayData:
    return read_today(game_date, dsn_override=_aiven_secret())


@st.cache_data(ttl=CACHE_TTL_SECONDS, max_entries=8, show_spinner=False)
def load_yesterday(game_date: date) -> TodayData:
    return read_yesterday(game_date, dsn_override=_aiven_secret())


@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def load_history(
    start: date, end: date, target: int, season_type: int | None
) -> AnalysisData:
    return read_history_range(
        start, end, target, season_type, dsn_override=_aiven_secret()
    )
