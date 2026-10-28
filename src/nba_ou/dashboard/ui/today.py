"""Today's board: controls, one cached source read, and time-sensitive rendering."""

from datetime import UTC, datetime

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import (
    DEFAULT_MIN_EDGE_POINTS,
    MIN_EDGE_LEVELS,
    REFERENCE_COLUMNS,
    TARGET_LABELS,
)
from nba_ou.dashboard.domain.comparisons import compare_predictions
from nba_ou.dashboard.domain.dates import madrid_time, slate_date
from nba_ou.dashboard.domain.models import model_families
from nba_ou.dashboard.domain.ordering import order_board
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


@st.fragment(run_every="60s")
def render_today() -> None:
    game_date = slate_date()
    st.caption(f"NBA slate · {game_date.strftime('%d %b %Y')} · All times in Madrid")
    target_col, refresh_col = st.columns([4, 1], vertical_alignment="bottom")
    with target_col:
        target = st.segmented_control(
            "Prediction target",
            list(TARGET_LABELS),
            format_func=TARGET_LABELS.get,
            key="today_target",
            width="stretch",
        )
    with refresh_col:
        if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
            load_today.clear(game_date)
    # Segmented controls can be deselected; keep a sensible explicit default.
    target = target or 2
    spread = target == 3
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
    families = model_families(live, target)
    labels = {family.key: family.label for family in families}
    models_key = f"today_models_{target}"
    known_key = f"today_known_models_{target}"
    # Keep the user's choices, drop vanished families and select newly seen ones.
    known = st.session_state.get(known_key, set())
    chosen = st.session_state.get(models_key, [])
    st.session_state[models_key] = [k for k in chosen if k in labels] + [
        k for k in labels if k not in known
    ]
    st.session_state[known_key] = known | set(labels)
    model_col, view_col = st.columns([3, 2], vertical_alignment="bottom")
    with model_col:
        family_keys = st.multiselect(
            "Models",
            list(labels),
            format_func=labels.get,
            key=models_key,
            placeholder="Select models",
        )
    with view_col:
        view = (
            st.segmented_control(
                "Display", ["Cards", "Table"], key="today_view", width="stretch"
            )
            or "Cards"
        )
    with st.popover("Display settings", icon=":material/tune:"):
        reference = st.selectbox(
            "Compare against",
            list(REFERENCE_COLUMNS),
            key="today_reference",
            help="This changes the displayed edge and signal, never the model's original prediction.",
        )
        min_edge = st.selectbox(
            "Minimum edge (points)",
            MIN_EDGE_LEVELS,
            index=MIN_EDGE_LEVELS.index(DEFAULT_MIN_EDGE_POINTS),
            key="today_min_edge",
            help="Directional signals require an absolute market edge strictly above this threshold. All predictions remain visible.",
        )
        show_chart = st.toggle("Show comparison charts", key="today_charts")
        sort_by = st.selectbox(
            "Sort games by", ["Tip-off", "Largest edge"], key="today_sort"
        )

    selected = select_predictions(live, target, now, family_keys)
    markets = market_snapshots(live, target, now)
    predictions = compare_predictions(selected, markets, reference, min_edge=min_edge)
    games, predictions = order_board(games, predictions, sort_by)
    visible_families = [f for f in families if f.key in family_keys]

    with st.container(horizontal=True):
        st.metric("Games tonight", len(games), border=True)
        st.metric(
            "With predictions",
            f"{predictions['game_id'].nunique() if not predictions.empty else 0} / {len(games)}",
            border=True,
        )
        st.metric(
            "Closing",
            int(predictions["dataset_type"].eq(1).sum())
            if not predictions.empty
            else 0,
            border=True,
        )
        st.metric(
            "Intermediate",
            int(predictions["dataset_type"].eq(2).sum())
            if not predictions.empty
            else 0,
            border=True,
        )
    latest = live["predicted_at"].max() if not live.empty else None
    st.caption(
        f"Last read: {madrid_time(data.fetched_at, include_date=False)} · Latest available prediction: {madrid_time(latest)} · Updates every minute"
    )
    st.caption(
        "Games recorded in Aiven. Market values are stored snapshots, not live quotes. Each game uses its nearest available intermediate model horizon."
    )
    st.caption(
        f"Signals require |edge vs {reference}| > {min_edge:g} points. Smaller edges are shown as NO EDGE."
    )
    if spread:
        st.caption(
            "Spread lines show the implied home margin. HOME/AWAY signals compare the reconstructed margin with the selected market line; the original model error remains visible."
        )

    if not families:
        st.info(
            "No live predictions are available for this target yet. Recorded games and available market lines are shown below."
        )
    elif not family_keys:
        st.info("Select at least one model to display predictions.")

    if view == "Table":
        render_prediction_table(predictions, reference, spread=spread)
        missing = len(games) - (
            predictions["game_id"].nunique() if not predictions.empty else 0
        )
        if missing:
            st.caption(
                f"{missing} recorded game(s) have no predictions for these filters. Switch to Cards to see them."
            )
        return

    markets_by_game = (
        markets.set_index("game_id") if not markets.empty else pd.DataFrame()
    )
    with st.container(horizontal=True, gap="medium"):
        for _, game in games.iterrows():
            game_predictions = predictions.loc[
                predictions["game_id"].eq(game["game_id"])
            ]
            market = (
                markets_by_game.loc[game["game_id"]]
                if game["game_id"] in markets_by_game.index
                else None
            )
            with st.container(width=640):
                render_game_card(
                    game,
                    game_predictions,
                    market,
                    visible_families,
                    reference,
                    spread=spread,
                    show_chart=show_chart,
                    now=now,
                )
