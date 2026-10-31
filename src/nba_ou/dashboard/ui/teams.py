"""The shared game header for upcoming and completed games."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.dates import madrid_time, time_remaining
from nba_ou.dashboard.domain.teams import full_team_name, team_logo_url


def render_game_header(game, *, now=None, show_result: bool = False) -> None:
    """A balanced matchup panel with a central score and a separate timing row."""
    game_id = int(game["game_id"])
    final = (
        show_result
        and game["status"] == 1
        and pd.notna(game.get("home_points"))
        and pd.notna(game.get("away_points"))
    )
    with st.container(key=f"match_header_{game_id}"):
        with st.container(
            horizontal=True,
            vertical_alignment="center",
            gap="small",
            key=f"match_matchup_{game_id}",
        ):
            render_team(
                game["away_team"],
                game["away_team_id"],
                "Away",
                key=f"match_team_{game_id}_away",
            )
            with st.container(
                width="content",
                horizontal_alignment="center",
                gap="xxsmall",
                key=f"match_center_{game_id}",
            ):
                if show_result:
                    _game_result(game, final=final)
                else:
                    st.markdown("**VS**")
            render_team(
                game["home_team"],
                game["home_team_id"],
                "Home",
                key=f"match_team_{game_id}_home",
            )
        with st.container(key=f"match_footer_{game_id}", gap="xxsmall"):
            timing = f"**{madrid_time(game['tipoff_at'])}**"
            if game["status"] == 1:
                timing += " · Final"
            elif now is not None and game["status"] != 2:
                timing += f" · {time_remaining(game['tipoff_at'], now)}"
            st.caption(timing)
            if final:
                st.caption(
                    f"Total points **{int(game['home_points'] + game['away_points'])}** · "
                    f"Home margin **{int(game['home_points'] - game['away_points']):+d}**"
                )


def _game_result(game, *, final: bool) -> None:
    if final:
        st.markdown(f"### {int(game['away_points'])} — {int(game['home_points'])}")
        st.caption("Final")
    elif game["status"] == 2:
        st.markdown("**Postponed / void**")
    else:
        st.markdown("**Result pending**")


def render_team(name: str, team_id, role: str, *, key: str) -> None:
    name = full_team_name(name, team_id)
    with st.container(horizontal_alignment="center", gap="xxsmall", key=key):
        st.caption(role)
        logo = team_logo_url(name, team_id)
        if logo:
            # Streamlit 1.54 has no image alt parameter; use the visible side caption.
            st.image(logo, width=48)
        st.markdown(f"**{name}**", text_alignment="center")
