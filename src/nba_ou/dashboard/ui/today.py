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
    TARGET_LABELS,
)
from nba_ou.dashboard.domain.comparisons import compare_predictions
from nba_ou.dashboard.domain.dates import madrid_clock, madrid_time, slate_date
from nba_ou.dashboard.domain.models import model_families
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
    bets = [game_id for game_id, pick in picks.items() if pick.status == "BET"]
    with st.container(horizontal=True):
        st.metric("Games", len(games), border=True)
        st.metric("Bets", len(bets), border=True)
        st.metric(
            "Models disagree",
            sum(pick.status == "SPLIT" for pick in picks.values()),
            border=True,
        )
        st.metric(
            "No bet",
            sum(pick.status in ("NO BET", "NO LINE") for pick in picks.values()),
            border=True,
        )
    if not bets:
        return
    with st.container(border=True):
        st.markdown("**Tonight's bets**")
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
        target = st.segmented_control(
            "Prediction target",
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
        if st.button("Refresh", icon=":material/refresh:", width="stretch"):
            load_today.clear(game_date)
    # Segmented controls can be deselected; keep a sensible explicit default.
    target = target or DEFAULT_TARGET
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
    with st.popover("Settings", icon=":material/tune:"):
        family_keys = st.multiselect(
            "Models",
            list(labels),
            format_func=labels.get,
            key=models_key,
            placeholder="Select models",
        )
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

    selected = select_predictions(live, target, now, family_keys)
    markets = market_snapshots(live, target, now)
    predictions = compare_predictions(selected, markets, reference, min_edge=min_edge)
    games, predictions = order_board(games, predictions, sort_by)
    visible_families = [f for f in families if f.key in family_keys]
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

    _picks_summary(games, picks, markets_by_game, reference)
    latest = live["predicted_at"].max() if not live.empty else None
    thresholds = " · ".join(
        f"{MODEL_NAMES[kind]} > {value:g}" for kind, value in min_edge.items()
    )
    st.caption(
        f"A prediction bets when |edge vs {reference}| clears its minimum ({thresholds} pts); "
        "a game is a bet when every such prediction points the same way. "
        f"Latest prediction {madrid_time(latest)} · read {madrid_clock(data.fetched_at)} · updates every minute."
    )

    if not families:
        st.info(
            "No live predictions are available for this target yet. Recorded games and available market lines are shown below."
        )
    elif not family_keys:
        st.info("Select at least one model in Settings to display predictions.")

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

    with st.container(horizontal=True, gap="medium"):
        for _, game in games.iterrows():
            market = (
                markets_by_game.loc[game["game_id"]]
                if game["game_id"] in markets_by_game.index
                else None
            )
            with st.container(width=560):
                render_game_card(
                    game,
                    by_game[game["game_id"]],
                    market,
                    visible_families,
                    reference,
                    picks[game["game_id"]],
                    spread=spread,
                    show_chart=show_chart,
                    now=now,
                    missing_models=missing_models,
                )
