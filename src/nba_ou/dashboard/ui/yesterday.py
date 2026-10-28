"""Date-selectable slates: fixed checkpoints, stored quotes, and final scores."""

from datetime import UTC, datetime, timedelta

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import MODEL_NAMES
from nba_ou.dashboard.domain.dates import madrid_time, slate_date
from nba_ou.dashboard.domain.history import (
    HISTORY_SOURCES,
    grade_history,
    historical_games,
    history_summary,
    select_history,
)
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.controls import minimum_edge_select
from nba_ou.dashboard.ui.history_table import (
    checkpoint_summary_table,
    game_results_table,
    history_details,
    history_table_html,
)
from nba_ou.dashboard.ui.loaders import load_yesterday
from nba_ou.dashboard.ui.teams import render_game_header

TARGETS = {1: "Line error", 2: "Total points", 3: "Spread"}


def _performance_summary(games: pd.DataFrame, rows: pd.DataFrame) -> None:
    st.subheader("Daily performance")
    final = (
        games["status"].eq(1)
        & games["home_points"].notna()
        & games["away_points"].notna()
    )
    st.caption(
        f"{int(final.sum())}/{len(games)} games final · Games = evaluated matches / all matches · Wins = wins / settled picks"
    )
    summary = history_summary(rows, by_checkpoint=True)
    if summary.empty:
        st.caption("No predictions available to summarize.")
        return
    with st.container(horizontal=True, gap="medium"):
        for _, checkpoints in summary.groupby("Family", sort=False):
            with st.container(border=True):
                st.markdown(f"**{checkpoints['Model'].iloc[0]}**")
                st.table(
                    checkpoint_summary_table(checkpoints, len(games)),
                    border="horizontal",
                )
                if checkpoints["Checkpoint"].str.startswith("Closing").any():
                    st.caption(
                        "Closing at closing = last available pre-game prediction."
                    )


def _game_card(game, rows: pd.DataFrame, *, spread: bool) -> None:
    with st.container(border=True):
        render_game_header(game, show_result=True)
        if rows.empty:
            st.caption("No stored predictions for this target.")
        else:
            st.html(history_table_html(rows, spread=spread))
            if not rows["available"].any():
                st.caption("No predictions match these checkpoints and source.")
        if st.toggle("View details", key=f"yesterday_details_{int(game['game_id'])}"):
            if rows.empty:
                st.caption("No prediction details available.")
            else:
                st.dataframe(
                    history_details(rows, spread=spread),
                    hide_index=True,
                    width="stretch",
                )


@st.fragment(run_every="60s")
def render_yesterday() -> None:
    default_date = slate_date() - timedelta(days=1)
    game_date = (
        st.date_input(
            "Game date",
            value=default_date,
            key="yesterday_date",
            format="DD/MM/YYYY",
            help="NBA slate date, using the same Pacific date boundary as Today. Game times are displayed in Madrid time.",
            width=220,
        )
        or default_date
    )
    st.caption(f"NBA slate · {game_date.strftime('%d %b %Y')} · All times in Madrid")
    target_col, source_col, view_col, refresh_col = st.columns(
        [3, 3, 2, 1], vertical_alignment="bottom"
    )
    with target_col:
        target = (
            st.segmented_control(
                "Prediction target",
                list(TARGETS),
                format_func=TARGETS.get,
                default=1,
                key="yesterday_target",
                width="stretch",
            )
            or 1
        )
    with source_col:
        source = (
            st.segmented_control(
                "Prediction source",
                HISTORY_SOURCES,
                default=HISTORY_SOURCES[0],
                key="yesterday_source",
                width="stretch",
            )
            or HISTORY_SOURCES[0]
        )
    with view_col:
        view = (
            st.segmented_control(
                "Display",
                ["Cards", "Table"],
                default="Cards",
                key="yesterday_view",
                width="stretch",
            )
            or "Cards"
        )
    with refresh_col:
        if st.button(
            "Refresh",
            icon=":material/refresh:",
            key="yesterday_refresh",
            width="stretch",
        ):
            load_yesterday.clear(game_date)
    with (
        st.container(key="yesterday_settings"),
        st.popover("Settings", icon=":material/tune:"),
        st.container(key="yesterday_settings_options"),
    ):
        thresholds = {
            kind: minimum_edge_select(kind, page="yesterday") for kind in MODEL_NAMES
        }
        st.caption(
            "Closing: T−6h, T−4h and the final available pre-game prediction. "
            "Intermediate: the T−8h, T−6h and T−4h models at their own checkpoints. "
            "Each pick is graded against its stored bookmaker line; missing slots do not count as losses."
        )
        st.caption(
            "Closing checkpoints use predictions available in the preceding 30 minutes. Live intermediate inputs must be within the preceding 30 minutes of their own checkpoint; publication may take up to 30 minutes after it. Details show the exact times."
        )
    if source == "Snapshot replay":
        st.info(
            "Intermediate rows use stored, exact own-horizon snapshots, which may have been reconstructed after the game. Closing rows remain live predictions in both modes."
        )
    try:
        with st.spinner("Reading predictions and results from Aiven…"):
            data = load_yesterday(game_date)
    except DashboardDataError as exc:
        st.error(str(exc), icon=":material/database:")
        return
    games = historical_games(data.games)
    if games.empty:
        st.info("No games were recorded in Aiven for the selected NBA slate.")
        return
    rows = grade_history(
        select_history(data.predictions, games, target, source, datetime.now(UTC)),
        min_edge=thresholds,
    )
    _performance_summary(games, rows)
    st.subheader("Game results")
    if rows.empty:
        st.info(
            "No stored model predictions are available for this target yet. Recorded games and results are shown below."
        )
    if view == "Table":
        if not rows.empty:
            st.dataframe(
                history_details(rows, spread=target == 3),
                hide_index=True,
                width="stretch",
            )
        else:
            st.dataframe(game_results_table(games), hide_index=True, width="stretch")
    else:
        with st.container(horizontal=True, gap="medium", key="yesterday_games_grid"):
            for _, game in games.iterrows():
                game_rows = (
                    rows.loc[rows["game_id"].eq(game["game_id"])]
                    if not rows.empty
                    else rows
                )
                with st.container(
                    width="content", key=f"yesterday_game_slot_{int(game['game_id'])}"
                ):
                    _game_card(game, game_rows, spread=target == 3)
    st.caption(
        f"Results and predictions read {madrid_time(data.fetched_at)} · updates every minute."
    )
