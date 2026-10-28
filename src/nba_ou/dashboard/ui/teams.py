"""The shared game header for upcoming and completed games."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.dates import madrid_time, time_remaining
from nba_ou.dashboard.domain.teams import full_team_name, team_logo_url


def render_game_header(game, *, now=None, show_result: bool = False) -> None:
    """A light header with teams, Madrid tip-off time and optional final score."""
    with st.container(key=f"match_header_{int(game['game_id'])}"):
        away, home = st.columns(2)
        with away:
            render_team(game["away_team"], game["away_team_id"], "Away")
        with home:
            render_team(game["home_team"], game["home_team_id"], "Home")
        timing = f"**{madrid_time(game['tipoff_at'])}**"
        if now is not None:
            timing += f" · {time_remaining(game['tipoff_at'], now)}"
        st.caption(timing)
        if show_result:
            _game_result(game)


def _game_result(game) -> None:
    final = (
        game["status"] == 1
        and pd.notna(game["home_points"])
        and pd.notna(game["away_points"])
    )
    if final:
        st.markdown(f"### {int(game['away_points'])} — {int(game['home_points'])}")
        st.caption(
            f"Final · Total points **{int(game['home_points'] + game['away_points'])}** · "
            f"Home margin **{int(game['home_points'] - game['away_points']):+d}**"
        )
    elif game["status"] == 2:
        st.markdown("### :gray[Postponed / void]")
    else:
        st.markdown("### :gray[Result pending]")


def render_team(name: str, team_id, role: str) -> None:
    name = full_team_name(name, team_id)
    with st.container(horizontal=True, vertical_alignment="center"):
        logo = team_logo_url(name, team_id)
        if logo:
            # Streamlit 1.54 has no image alt parameter; use the visible side caption.
            st.image(logo, width=48, caption=role)
        st.markdown(f"**{name}**")
