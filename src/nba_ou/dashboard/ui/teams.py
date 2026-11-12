"""The shared game header for upcoming and completed games."""

import pandas as pd

import streamlit as st
from nba_ou.dashboard.domain.dates import madrid_clock, madrid_time, time_remaining
from nba_ou.dashboard.domain.reports import report_flag
from nba_ou.dashboard.domain.teams import full_team_name, team_logo_url


def render_game_header(
    game,
    *,
    now=None,
    show_result: bool = False,
    line: float | None = None,
    spread: bool = False,
) -> None:
    """A balanced matchup panel with a central score and a separate timing row.

    A final game adds a result strip: total points and home margin, plus the
    given pre-game ``line`` (totals, or the implied home margin when
    ``spread``) and how the game finished against it. An upcoming game (with
    ``now``) is flagged while a team's injury report is not filed.
    """
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
        if final:
            _result_strip(game, line=line, spread=spread)
        with st.container(key=f"match_footer_{game_id}", gap="xxsmall"):
            timing = f"**{madrid_time(game['tipoff_at'])}**"
            if game["status"] == 1:
                timing += " · Final"
            elif now is not None and game["status"] != 2:
                timing += f" · {time_remaining(game['tipoff_at'], now)}"
            st.caption(timing)
            if now is not None:
                _report_badge(game, now)


def _report_badge(game, now) -> None:
    flag = report_flag(game, now)
    if flag is None:
        return
    if flag.filed:
        st.badge("Injury reports filed", icon=":material/check:", color="green")
        return
    teams = " and ".join(
        _nickname(game[f"{side}_team"], game.get(f"{side}_team_id"))
        for side in flag.missing
    )
    checked = "" if flag.read_at is None else f" · as of {madrid_clock(flag.read_at)}"
    st.badge(
        f"No injury report yet: {teams}{checked}",
        icon=":material/warning:",
        color="orange",
    )


def _nickname(name: str, team_id) -> str:
    """``Boston Celtics`` -> ``Celtics``, short enough for the result strip."""
    return full_team_name(name, team_id).split()[-1]


def result_stats(game, *, line=None, spread: bool = False) -> list[tuple[str, str]]:
    """``(label, value)`` cells describing a final game against its line."""
    total = int(game["home_points"] + game["away_points"])
    margin = int(game["home_points"] - game["away_points"])
    home = _nickname(game["home_team"], game.get("home_team_id"))
    away = _nickname(game["away_team"], game.get("away_team_id"))
    leader = home if margin >= 0 else away
    stats = [
        ("Total points", str(total)),
        ("Margin", "Tied" if not margin else f"{leader} by {abs(margin)}"),
    ]
    if line is None:
        return stats
    if spread:
        # The line is the implied home margin; the home handicap is its negation.
        handicap = -line
        stats.append(
            (
                "Pre-game spread",
                f"{home} {'PK' if handicap == 0 else f'{handicap:+.1f}'}",
            )
        )
        diff = margin - line
        side = home if diff > 0 else away
        outcome = "Push" if diff == 0 else f"{side} cover by {abs(diff):.1f}"
    else:
        stats.append(("Pre-game line", f"{line:.1f}"))
        diff = total - line
        outcome = (
            "Push"
            if diff == 0
            else f"{'Over' if diff > 0 else 'Under'} by {abs(diff):.1f}"
        )
    stats.append(("Result", outcome))
    return stats


def _result_strip(game, *, line, spread: bool) -> None:
    game_id = int(game["game_id"])
    with st.container(horizontal=True, gap="small", key=f"match_result_{game_id}"):
        for index, (label, value) in enumerate(
            result_stats(game, line=line, spread=spread)
        ):
            with st.container(
                gap="xxsmall",
                horizontal_alignment="center",
                key=(
                    f"match_stat_total_{game_id}"
                    if not index
                    else f"match_stat_{game_id}_{index}"
                ),
            ):
                st.caption(label)
                st.markdown(f"**{value}**", text_alignment="center")


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
