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
    pregame_line,
    select_history,
)
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.controls import minimum_edge_select
from nba_ou.dashboard.ui.history_table import (
    checkpoint_summary_table,
    game_results_table,
    history_details,
    history_table_html,
    slate_summary_tables,
)
from nba_ou.dashboard.ui.loaders import load_yesterday
from nba_ou.dashboard.ui.teams import render_game_header

TARGETS = {1: "Line error", 2: "Total points", 3: "Spread", 0: "Summary"}
# The whole slate's win rate for every target, instead of one target's games.
SUMMARY = 0


def _performance_summary(games: pd.DataFrame, rows: pd.DataFrame) -> None:
    st.subheader("Daily performance")
    st.caption(
        f"{_final_games(games)}/{len(games)} games final · "
        "Games = evaluated matches · Bets = wins + losses · Win rate = Wins / Bets. "
        "No bets, pushes and missing lines are excluded from Bets."
    )
    summary = history_summary(rows, by_checkpoint=True)
    if summary.empty:
        st.caption("No predictions available to summarize.")
        return
    with st.container(horizontal=True, gap="medium", key="yesterday_performance"):
        for _, checkpoints in summary.groupby("Family", sort=False):
            with st.container(border=True, width="content"):
                st.markdown(f"**{checkpoints['Model'].iloc[0]}**")
                st.table(
                    checkpoint_summary_table(checkpoints),
                    border="horizontal",
                    alt=f"Daily prediction coverage and betting results for {checkpoints['Model'].iloc[0]}",
                )
                if checkpoints["Checkpoint"].str.startswith("Closing").any():
                    st.caption(
                        "Closing at closing = T−0 reconstruction in replay; last available pre-game prediction in live."
                    )


def _final_games(games: pd.DataFrame) -> int:
    return int(
        (
            games["status"].eq(1)
            & games["home_points"].notna()
            & games["away_points"].notna()
        ).sum()
    )


def _slate_summary(data, games: pd.DataFrame, source: str, thresholds) -> None:
    """Win rate of every target and model over the slate, not per game."""
    st.subheader("Slate summary")
    st.caption(
        f"{_final_games(games)}/{len(games)} games final · {source} · "
        "Win rate = Wins / Bets (wins–losses). No bets, pushes and missing lines "
        "are excluded. Each prediction is graded against its own stored line."
    )
    pooled, by_checkpoint = {}, {}
    now = datetime.now(UTC)
    for target, label in TARGETS.items():
        if target == SUMMARY:
            continue
        rows = grade_history(
            select_history(data.predictions, games, target, source, now),
            min_edge=thresholds,
        )
        if rows.empty:
            continue
        pooled[label] = history_summary(rows)
        by_checkpoint[label] = history_summary(rows, by_checkpoint=True)
    if not pooled:
        st.info("No stored model predictions are available for this slate yet.")
        return
    with st.container(horizontal=True, key="yesterday_summary_targets"):
        for label, summary in pooled.items():
            wins, losses = int(summary["Wins"].sum()), int(summary["Losses"].sum())
            st.metric(
                label,
                f"{100 * wins / (wins + losses):.1f}%" if wins + losses else "—",
                border=True,
                help=f"{wins} wins / {losses} losses over every model and "
                "checkpoint. Rows repeat games across checkpoints, so they are not "
                "independent bets.",
            )
    with st.container(horizontal=True, gap="medium", key="yesterday_summary"):
        for model, table in slate_summary_tables(pooled, by_checkpoint).items():
            with st.container(border=True, width="content"):
                st.markdown(f"**{model}**")
                st.table(
                    table,
                    border="horizontal",
                    alt=f"Slate win rate by checkpoint and target for {model}",
                )


def _game_card(game, rows: pd.DataFrame, *, spread: bool) -> None:
    with st.container(border=True):
        render_game_header(
            game, show_result=True, line=pregame_line(rows), spread=spread
        )
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
        [4, 3, 2, 1], vertical_alignment="bottom"
    )
    with target_col:
        target = st.segmented_control(
            "Prediction target",
            list(TARGETS),
            format_func=TARGETS.get,
            default=1,
            key="yesterday_target",
            width="stretch",
        )
        # Summary is 0, so a deselected control (None) is told apart explicitly.
        if target is None:
            target = 1
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
                disabled=target == SUMMARY,
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
            "Closing: T−8h, T−6h, T−4h and a final checkpoint (T−0 in replay; last completed pre-game output in live); "
            "TabPFN also T−12h, T−10h and T−2h when it ran there. "
            "Intermediate: the T−8h, T−6h, T−4h and T−2h models at their own checkpoints, plus T−16h, T−12h and T−10h when the slate has them; "
            "TabPFN shows every horizon it ran. "
            "Each pick is graded against its stored bookmaker line; missing slots do not count as losses. "
            "The header's pre-game line is the final checkpoint's stored quote."
        )
        st.caption(
            "Live checkpoints select the nearest input within ±60 minutes; publication must also fall within that window. Inputs and publication can be after the checkpoint. Final live closing uses the last completed pre-game prediction. Details show the exact times."
        )
    if source == "Snapshot replay":
        st.info(
            "Closing and intermediate rows use stored, exact snapshots, which may have been reconstructed after the game. Closing replay uses T−8h, T−6h, T−4h and T−0 pregame inputs. Missing reconstructions remain unavailable."
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
    if target == SUMMARY:
        _slate_summary(data, games, source, thresholds)
        st.caption(
            f"Results and predictions read {madrid_time(data.fetched_at)} · updates every minute."
        )
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
