"""Today's board: controls, one cached source read, and time-sensitive rendering."""

from datetime import UTC, datetime

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import (
    DEFAULT_MIN_EDGE_BY_KIND,
    DEFAULT_TARGET,
    MIN_EDGE_LEVELS,
    MODEL_NAMES,
    REFERENCE_COLUMNS,
    SIGNAL_COLORS,
    TARGET_GROUPS,
    TARGET_LABELS,
)
from nba_ou.dashboard.domain.comparisons import compare_predictions
from nba_ou.dashboard.domain.dates import madrid_clock, madrid_time, slate_date
from nba_ou.dashboard.domain.models import displayed_models, model_families
from nba_ou.dashboard.domain.ordering import order_board
from nba_ou.dashboard.domain.picks import bet_text, game_pick
from nba_ou.dashboard.domain.selection import (
    live_predictions,
    market_snapshots,
    select_predictions,
    upcoming_games,
)
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.game_card import render_game_card
from nba_ou.dashboard.ui.loaders import load_today
from nba_ou.dashboard.ui.prediction_table import render_prediction_table


def _min_edge_select(kind: int) -> float:
    default = DEFAULT_MIN_EDGE_BY_KIND[kind]
    return st.selectbox(
        f"Minimum edge · {MODEL_NAMES[kind]} (points)",
        MIN_EDGE_LEVELS,
        index=MIN_EDGE_LEVELS.index(default),
        key=f"today_min_edge_{MODEL_NAMES[kind].lower()}",
        help="A prediction bets only when its absolute market edge is strictly "
        "above this threshold. All predictions remain visible.",
    )


def _picks_summary(games, picks, markets_by_game, reference) -> None:
    st.subheader("Tonight's bets")
    bets = [game_id for game_id, pick in picks.items() if pick.status == "BET"]
    with st.container(horizontal=True):
        st.metric("Games", len(games), border=True)
        st.metric("Bets", len(bets), border=True)
        st.metric(
            "Tied votes",
            sum(pick.status == "TIE" for pick in picks.values()),
            border=True,
        )
        st.metric(
            "No bet",
            sum(pick.status in ("NO BET", "NO LINE") for pick in picks.values()),
            border=True,
        )
    if not bets:
        st.caption("No bets for these games.")
        return
    with st.container(border=True):
        for _, game in games.loc[games["game_id"].isin(bets)].iterrows():
            pick = picks[game["game_id"]]
            market = (
                markets_by_game.loc[game["game_id"]]
                if game["game_id"] in markets_by_game.index
                else None
            )
            line = (
                market.get(REFERENCE_COLUMNS[reference]) if market is not None else None
            )
            text = bet_text(
                pick.side, line, home=game["home_team"], away=game["away_team"]
            )
            st.markdown(
                f"{madrid_clock(game['tipoff_at'])} · {game['away_team']} @ "
                f"{game['home_team']} → :{SIGNAL_COLORS[pick.side]}[**{text}**] "
                f"· {pick.agreeing}/{pick.predictions}"
            )


@st.fragment(run_every="60s")
def render_today() -> None:
    game_date = slate_date()
    st.caption(f"NBA slate · {game_date.strftime('%d %b %Y')} · All times in Madrid")
    target_col, view_col, refresh_col = st.columns(
        [4, 2, 1], vertical_alignment="bottom"
    )
    with target_col:
        # A running session may still hold the retired Total points option.
        if st.session_state.get("today_target") not in TARGET_LABELS:
            st.session_state["today_target"] = DEFAULT_TARGET
        target = st.segmented_control(
            "Market",
            list(TARGET_LABELS),
            format_func=TARGET_LABELS.get,
            key="today_target",
            width="stretch",
        )
    with view_col:
        view = (
            st.segmented_control(
                "Display", ["Cards", "Table"], key="today_view", width="stretch"
            )
            or "Cards"
        )
    with refresh_col:
        if st.button(
            "Refresh", icon=":material/refresh:", width="stretch", key="today_refresh"
        ):
            load_today.clear(game_date)
    # Segmented controls can be deselected; keep a sensible explicit default.
    target = target or DEFAULT_TARGET
    spread = target == 3
    targets = TARGET_GROUPS[target]
    try:
        # Only the session's first read shows a spinner, not each minute's refresh.
        if st.session_state.get("today_loaded"):
            data = load_today(game_date)
        else:
            with st.spinner("Reading tonight's predictions from Aiven…"):
                data = load_today(game_date)
            st.session_state["today_loaded"] = True
    except DashboardDataError as exc:
        st.error(str(exc), icon=":material/database:")
        return

    now = datetime.now(UTC)
    games = upcoming_games(data.games, now)
    if data.games.empty:
        st.info("No games have been recorded in Aiven for today's NBA slate yet.")
        st.caption(
            "This does not necessarily mean there are no games. Check again after the next prediction run."
        )
        return
    if games.empty:
        st.info(
            "All recorded games for this slate have started or are no longer scheduled."
        )
        return

    live = live_predictions(data.predictions, now)
    families = model_families(displayed_models(live), targets)
    with (
        st.container(key="today_settings"),
        st.popover("Settings", icon=":material/tune:"),
        st.container(key="today_settings_options"),
    ):
        reference = st.selectbox(
            "Compare against",
            list(REFERENCE_COLUMNS),
            key="today_reference",
            help="This changes the displayed edge and pick, never the model's original prediction.",
        )
        min_edge = {kind: _min_edge_select(kind) for kind in MODEL_NAMES}
        show_chart = st.toggle("Show comparison charts", key="today_charts")
        sort_by = st.selectbox(
            "Sort games by", ["Tip-off", "Largest edge"], key="today_sort"
        )

    selected = select_predictions(displayed_models(live), targets, now)
    markets = market_snapshots(live, target, now)
    predictions = compare_predictions(selected, markets, reference, min_edge=min_edge)
    games, predictions = order_board(games, predictions, sort_by)
    if not predictions.empty:
        # Use the schedule's canonical full names in tables, details and picks.
        names = games.set_index("game_id")
        for side in ("home", "away"):
            predictions[f"{side}_team"] = predictions["game_id"].map(
                names[f"{side}_team"]
            )
    # A model with no live output for this target still gets a row, so a
    # TabPFN run that has not happened yet is visible rather than absent.
    missing_models = tuple(
        name
        for kind, name in MODEL_NAMES.items()
        if not any(family.kind == kind for family in families)
    )
    markets_by_game = (
        markets.set_index("game_id") if not markets.empty else pd.DataFrame()
    )
    by_game = {
        game_id: (
            predictions.loc[predictions["game_id"].eq(game_id)]
            if not predictions.empty
            else predictions
        )
        for game_id in games["game_id"]
    }
    picks = {game_id: game_pick(rows) for game_id, rows in by_game.items()}

    latest = live["predicted_at"].max() if not live.empty else None
    thresholds = " · ".join(
        f"{MODEL_NAMES[kind]} > {value:g}" for kind, value in min_edge.items()
    )
    st.caption(
        f"A prediction bets when |edge vs {reference}| clears its minimum ({thresholds} pts); "
        "a game follows the majority of directional votes; tied votes mean no bet. "
        f"Latest prediction {madrid_time(latest)} · read {madrid_clock(data.fetched_at)} · updates every minute."
    )

    if not families:
        st.info(
            "No live predictions are available for this target yet. Recorded games and available market lines are shown below."
        )

    if view == "Table":
        render_prediction_table(predictions, reference, spread=spread)
        missing = len(games) - (
            predictions["game_id"].nunique() if not predictions.empty else 0
        )
        if missing:
            st.caption(
                f"{missing} recorded game(s) have no predictions for these filters. Switch to Cards to see them."
            )
    else:
        # Cards wrap by content width, then grow to share unused row space.
        # The scoped grid styles impose no column count or pixel width.
        with st.container(horizontal=True, gap="medium", key="games_grid"):
            for _, game in games.iterrows():
                market = (
                    markets_by_game.loc[game["game_id"]]
                    if game["game_id"] in markets_by_game.index
                    else None
                )
                with st.container(
                    width="content", key=f"game_slot_{int(game['game_id'])}"
                ):
                    render_game_card(
                        game,
                        by_game[game["game_id"]],
                        market,
                        families,
                        reference,
                        picks[game["game_id"]],
                        spread=spread,
                        show_chart=show_chart,
                        now=now,
                        missing_models=missing_models,
                    )
    _picks_summary(games, picks, markets_by_game, reference)
