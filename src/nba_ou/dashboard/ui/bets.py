"""The Bets page: launch manual predictions, bet on them, keep the history.

1. New prediction: the models to run (TabPFN intermediate by default) and,
   for the intermediate ones, each game's own horizon or a fixed one. Each
   model is a request dispatched to GitHub Actions.
2. Today's requests and their status.
3. The predictions of those requests (optionally every live prediction of
   the slate), to bet on: side, line, odds and stake start from the
   prediction and the last stake, and stay editable.
4. The user's bets, graded, with their profit.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pandas as pd

import streamlit as st
from nba_ou.dashboard.config import MODEL_NAMES, PREDICTION_TARGET_LABELS
from nba_ou.dashboard.domain import bets as bet_rules
from nba_ou.dashboard.domain import manual
from nba_ou.dashboard.domain.access import AppUser
from nba_ou.dashboard.domain.dates import horizon_label, madrid_clock, madrid_time
from nba_ou.dashboard.domain.selection import select_predictions
from nba_ou.dashboard.repositories import app_store
from nba_ou.dashboard.repositories.connection import open_app_connection
from nba_ou.dashboard.repositories.dispatch import DispatchError, dispatch_workflow
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from nba_ou.dashboard.ui.app_loaders import (
    clear_bets,
    clear_requests,
    load_bets,
    load_last_stake,
    load_manual_predictions,
    load_requests,
)
from nba_ou.dashboard.ui.auth import app_secrets, still_allowed
from nba_ou.dashboard.ui.loaders import load_today
from nba_ou.postgre_db.app_aiven.schema import MARKET_SPREAD

#: The user's requests give one candidate per game, model, target and horizon:
#: the newest output. A fixed-horizon request stays beside a per-game one.
CANDIDATE_KEY = [
    "game_id",
    "kind",
    "dataset_type",
    "target",
    "variant",
    "model_horizon_minutes",
]
FLASH_KEY = "bets_flash"


def _now() -> datetime:
    return datetime.now(UTC)


def _flash() -> None:
    message = st.session_state.pop(FLASH_KEY, None)
    if message:
        st.success(message, icon=":material/check_circle:")


# -- 1. new prediction ---------------------------------------------------------


def _minutes_left(games: pd.DataFrame, now: datetime) -> dict[str, float]:
    tipoffs = pd.to_datetime(games["tipoff_at"], utc=True)
    minutes = (tipoffs - pd.Timestamp(now)).dt.total_seconds() / 60
    return dict(zip(games["game_id"].astype(str), minutes, strict=True))


def _launch(user: AppUser, choices: list[str], horizon: int | None, game_date) -> None:
    secrets = app_secrets()
    launched, failed = [], []
    with open_app_connection(secrets.app_db_url) as conn:
        for key in choices:
            choice = manual.CHOICES_BY_KEY[key]
            request_horizon = (
                horizon if choice.dataset_type == manual.INTERMEDIATE else None
            )
            request_id = app_store.create_request(
                conn,
                user_id=user.user_id,
                kind=choice.kind,
                dataset_type=choice.dataset_type,
                horizon_minutes=request_horizon,
                game_date=game_date,
            )
            workflow, inputs = manual.dispatch_inputs(
                request_id, choice.kind, choice.dataset_type, request_horizon
            )
            try:
                dispatch_workflow(secrets.dispatch_token, workflow, inputs)
            except DispatchError as exc:
                app_store.fail_request(conn, request_id, str(exc))
                failed.append(f"{choice.label}: {exc}")
            else:
                launched.append(
                    manual.request_label(
                        choice.kind, choice.dataset_type, request_horizon
                    )
                )
    clear_requests()
    if launched:
        st.session_state[FLASH_KEY] = "Launched: " + ", ".join(launched)
    for message in failed:
        st.error(message, icon=":material/error:")


def _new_prediction(user: AppUser, games: pd.DataFrame, game_date, now) -> None:
    with st.container(border=True):
        st.subheader("New prediction")
        st.caption(
            "Runs on the slate's games not yet started, with the data of the "
            "moment the run starts. Results appear below in a few minutes."
        )
        choices = st.pills(
            "Models",
            options=[choice.key for choice in manual.MODEL_CHOICES],
            format_func=lambda key: manual.CHOICES_BY_KEY[key].label,
            selection_mode="multi",
            default=list(manual.DEFAULT_CHOICES),
            key="manual_models",
        )
        intermediate = any(
            manual.CHOICES_BY_KEY[key].dataset_type == manual.INTERMEDIATE
            for key in choices
        )
        horizon = None
        if intermediate:
            horizon = st.selectbox(
                "Intermediate model",
                options=[None, *manual.HORIZONS],
                format_func=manual.horizon_choice_label,
                key="manual_horizon",
                help=(
                    "Each game's own horizon: a game 4 h before tip-off is "
                    "predicted by the T−4h model, one 8 h before by the T−8h "
                    "model. A fixed horizon predicts every game with that model."
                ),
            )
            upcoming = games.loc[pd.to_datetime(games["tipoff_at"], utc=True) > now]
            if horizon is None and not upcoming.empty:
                preview = manual.preview_horizons(_minutes_left(upcoming, now))
                names = upcoming.set_index(upcoming["game_id"].astype(str))
                st.caption(
                    "Now: "
                    + " · ".join(
                        f"{names.at[game, 'away_team']} @ {names.at[game, 'home_team']} "
                        f"{horizon_label(minutes)}"
                        for game, minutes in preview.items()
                    )
                )
        if st.button(
            "Launch",
            icon=":material/rocket_launch:",
            type="primary",
            disabled=not choices,
            key="manual_launch",
        ):
            _launch(user, list(choices), horizon, game_date)
            st.rerun()


# -- 2. requests ---------------------------------------------------------------


def _requests(user: AppUser, game_date, now) -> None:
    st.subheader("Today's requests")
    requests = load_requests(user.user_id, game_date)
    if requests.empty:
        st.caption("No request today.")
        return
    requested = pd.to_datetime(requests["requested_at"], utc=True)
    table = pd.DataFrame(
        {
            "Model": [
                manual.request_label(k, d, None if pd.isna(h) else int(h))
                for k, d, h in requests[
                    ["kind", "dataset_type", "horizon_minutes"]
                ].itertuples(index=False)
            ],
            "Status": [
                manual.status_label(status, at.to_pydatetime(), now)
                for status, at in zip(requests["status"], requested, strict=True)
            ],
            "Requested": requested.map(madrid_clock),
            "Finished": pd.to_datetime(requests["finished_at"], utc=True).map(
                madrid_clock
            ),
            "Typical run": [
                f"{manual.TYPICAL_MINUTES[(int(k), int(d))]} min"
                for k, d in requests[["kind", "dataset_type"]].itertuples(index=False)
            ],
            "Error": requests["error"].fillna(""),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch", alt="Today's requests")


# -- 3. bet candidates -----------------------------------------------------------


def _model_label(row: pd.Series) -> str:
    name = MODEL_NAMES.get(int(row["kind"]), "Model")
    if int(row["dataset_type"]) == 1:
        return f"{name} closing"
    return f"{name} {horizon_label(row['model_horizon_minutes'])}"


def _scheduled_candidates(today: pd.DataFrame, now) -> pd.DataFrame:
    """What Today shows: each closing slot and, per family, the horizon nearest
    the tip. Manual runs go first, or another user's run could take a slot."""
    scheduled = today.loc[today["request_id"].isna()]
    return select_predictions(scheduled, tuple(PREDICTION_TARGET_LABELS), now)


def _candidates(
    manual_rows: pd.DataFrame, scheduled_rows: pd.DataFrame, now
) -> pd.DataFrame:
    """The user's own outputs beside the scheduled ones, games not yet started."""
    own = manual_rows.sort_values(
        ["predicted_at", "run_id", "input_id"], kind="stable"
    ).drop_duplicates(CANDIDATE_KEY, keep="last")
    parts = [frame for frame in (own, scheduled_rows) if not frame.empty]
    if not parts:
        return own
    rows = pd.concat(parts, ignore_index=True)
    rows = rows.loc[pd.to_datetime(rows["tipoff_at"], utc=True) > pd.Timestamp(now)]
    return rows.sort_values(["tipoff_at", "game_id", "kind", "target"]).reset_index(
        drop=True
    )


def _selection_key(candidates: pd.DataFrame) -> str:
    """A selection is a row position: new rows get a new table, so a refresh
    cannot leave it pointing at another prediction."""
    ids = (
        candidates[["run_id", "input_id", "model_build_id", "target"]]
        .astype(str)
        .agg("-".join, axis=1)
    )
    return "bet_candidates_" + hashlib.sha1("|".join(ids).encode()).hexdigest()[:12]


def _candidate_table(rows: pd.DataFrame) -> pd.DataFrame:
    def line(row):
        value = bet_rules.proposed_line(row["target"], row["bet_line_x2"])
        return None if value is None else value

    def pick(row):
        side = row["bet_side"]
        if pd.isna(side) or int(side) == 0:
            return "—"
        market = bet_rules.market_of_target(row["target"])
        return bet_rules.SIDE_LABELS[market][int(side)]

    return pd.DataFrame(
        {
            "Tip-off": rows["tipoff_at"].map(madrid_clock),
            "Game": rows["away_team"] + " @ " + rows["home_team"],
            "Model": rows.apply(_model_label, axis=1),
            "Target": rows["target"].map(PREDICTION_TARGET_LABELS),
            "Prediction": rows["predicted_value"].astype(float).round(1),
            "Bet365 line": rows.apply(line, axis=1),
            "Edge": rows["bet_edge"].astype(float).round(2),
            "Pick": rows.apply(pick, axis=1),
            "Source": rows["request_id"].map(
                lambda rid: "Scheduled" if pd.isna(rid) else f"Request #{int(rid)}"
            ),
            "Predicted": rows["predicted_at"].map(madrid_clock),
        }
    )


@st.dialog("Place a bet")
def _bet_dialog(row: dict, user: AppUser, stake: float) -> None:
    market = bet_rules.market_of_target(row["target"])
    sides = bet_rules.SIDE_LABELS[market]
    home, away = row["home_team"], row["away_team"]
    # Per prediction: a dialog closed with its X must not lend its values to
    # the next one.
    form = "_".join(
        str(int(row[name]))
        for name in ("game_id", "target", "input_id", "model_build_id")
    )
    st.markdown(f"**{away} @ {home}** · {madrid_time(row['tipoff_at'])}")
    st.caption(
        f"{_model_label(pd.Series(row))} · "
        f"{PREDICTION_TARGET_LABELS.get(int(row['target']), '')} "
        f"{float(row['predicted_value']):.1f}"
    )
    pick = row.get("bet_side")
    default_side = 1 if pd.isna(pick) or int(pick) == 0 else int(pick)
    side = st.segmented_control(
        "Side",
        options=[1, -1],
        format_func=lambda s: (
            (home if s == 1 else away) if market == MARKET_SPREAD else sides[s]
        ),
        default=default_side,
        key=f"bet_side_{form}",
    )
    side = default_side if side is None else int(side)
    proposed = bet_rules.proposed_line(row["target"], row["bet_line_x2"])
    line = st.number_input(
        "Home handicap" if market == MARKET_SPREAD else "Total",
        value=proposed,
        step=0.5,
        format="%.1f",
        placeholder="Enter the line you got",
        key=f"bet_line_{form}",
        help=(
            "The line you got, as the book shows the home side (home -3.5 is -3.5)."
            if market == MARKET_SPREAD
            else "The total you got."
        ),
    )
    prices = (row.get("bet_price_over_home"), row.get("bet_price_under_away"))
    odds = st.number_input(
        "Odds (decimal)",
        min_value=1.01,
        value=bet_rules.proposed_odds(side, *prices),
        step=0.01,
        format="%.3f",
        key=f"bet_odds_{form}_{side}",
    )
    if bet_rules.stored_odds(side, *prices) is None:
        st.caption(
            f"No stored price for this side: {bet_rules.DEFAULT_ODDS} is assumed, "
            "enter the odds you got."
        )
    amount = st.number_input(
        "Stake (€)",
        min_value=0.5,
        value=float(stake),
        step=1.0,
        key=f"bet_stake_{form}",
    )
    book = st.text_input("Book", value=bet_rules.DEFAULT_BOOK, key=f"bet_book_{form}")
    if line is not None:
        st.caption(
            "Bet: " + bet_rules.describe_bet(market, side, float(line), home, away)
        )
    if st.button(
        "Save bet",
        icon=":material/save:",
        type="primary",
        disabled=line is None,
        key=f"bet_save_{form}",
    ):
        if not still_allowed():
            st.rerun()
        bet = {
            "user_id": user.user_id,
            "game_id": int(row["game_id"]),
            "market": market,
            "side": side,
            "line": float(line),
            "odds_decimal": float(odds),
            "stake": float(amount),
            "book": book.strip() or bet_rules.DEFAULT_BOOK,
            "model_build_id": int(row["model_build_id"]),
            "input_id": int(row["input_id"]),
            "request_id": (
                None if pd.isna(row.get("request_id")) else int(row["request_id"])
            ),
        }
        with open_app_connection(app_secrets().app_db_url) as conn:
            saved = app_store.insert_bet(conn, bet)
        if saved is None:
            st.error("This account can no longer place bets.")
            return
        clear_bets()
        st.session_state[FLASH_KEY] = (
            f"Saved: {bet_rules.describe_bet(market, side, float(line), home, away)} "
            f"· {float(amount):g} € at {float(odds):.2f}"
        )
        st.rerun()


def _bet_candidates(user: AppUser, game_date, now) -> None:
    st.subheader("Predictions to bet on")
    include_all = st.toggle(
        "Include the scheduled predictions",
        value=False,
        key="bets_include_scheduled",
        help="Also list the scheduled predictions Today shows for each game.",
    )
    rows = load_manual_predictions(user.user_id, game_date)
    scheduled = (
        _scheduled_candidates(load_today(game_date).predictions, now)
        if include_all
        else pd.DataFrame()
    )
    candidates = _candidates(rows, scheduled, now)
    if candidates.empty:
        st.caption(
            "No prediction to bet on yet: launch one above, or include the "
            "scheduled predictions."
        )
        return
    event = st.dataframe(
        _candidate_table(candidates),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=_selection_key(candidates),
        alt="Predictions to bet on",
    )
    selected = event.selection.rows
    if st.button(
        "Bet on the selected prediction",
        icon=":material/payments:",
        disabled=not selected,
        key="bet_open",
    ):
        stake = bet_rules.default_stake(load_last_stake(user.user_id))
        _bet_dialog(candidates.iloc[selected[0]].to_dict(), user, stake)


# -- 4. history ------------------------------------------------------------------


def _history(user: AppUser, now) -> None:
    st.subheader("My bets")
    history = load_bets(user.user_id)
    if history.empty:
        st.caption("No bet yet.")
        return
    totals = bet_rules.summary(history)
    with st.container(horizontal=True):
        st.metric("Bets", totals["bets"], border=True)
        st.metric("Staked", f"{totals['staked']:.2f} €", border=True)
        st.metric("Profit", f"{totals['profit']:+.2f} €", border=True)
        st.metric("ROI", f"{totals['roi']:+.1%}", border=True)
        st.metric(
            "W-L-P",
            f"{totals['wins']}-{totals['losses']}-{totals['pushes']}",
            border=True,
        )
    table = pd.DataFrame(
        {
            "Date": history["game_date"],
            "Game": history["away_team"] + " @ " + history["home_team"],
            "Bet": [
                bet_rules.describe_bet(int(m), int(s), float(line), h, a)
                for m, s, line, h, a in history[
                    ["market", "side", "line", "home_team", "away_team"]
                ].itertuples(index=False)
            ],
            "Odds": history["odds_decimal"].astype(float),
            "Stake (€)": history["stake"].astype(float),
            "Book": history["book"],
            "Placed": history["placed_at"].map(madrid_time),
            "Result": history["result"].str.capitalize(),
            "Profit (€)": history["profit"].astype(float),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch", alt="My bets")

    voidable = history.loc[
        history["result"].eq("pending")
        & (pd.to_datetime(history["tipoff_at"], utc=True) > pd.Timestamp(now))
    ]
    if voidable.empty:
        return
    with st.container(horizontal=True, vertical_alignment="bottom"):
        bet_id = st.selectbox(
            "Void a bet (before tip-off)",
            options=voidable["bet_id"].tolist(),
            format_func=lambda bid: _bet_option(voidable, bid),
            key="bet_void_choice",
        )
        if st.button("Void", icon=":material/block:", key="bet_void"):
            with open_app_connection(app_secrets().app_db_url) as conn:
                voided = app_store.void_bet(conn, int(bet_id), user.user_id)
            clear_bets()
            st.session_state[FLASH_KEY] = (
                "Bet voided." if voided else "The game has started: not voided."
            )
            st.rerun()


def _bet_option(bets: pd.DataFrame, bet_id: int) -> str:
    bet = bets.loc[bets["bet_id"].eq(bet_id)].iloc[0]
    text = bet_rules.describe_bet(
        int(bet["market"]),
        int(bet["side"]),
        float(bet["line"]),
        bet["home_team"],
        bet["away_team"],
    )
    return f"{bet['away_team']} @ {bet['home_team']}: {text}, {float(bet['stake']):g} €"


# -- page ------------------------------------------------------------------------


@st.fragment(run_every="30s")
def _live_sections(user: AppUser, game_date) -> None:
    # Fragment runs skip the gate in streamlit_app.py.
    if not still_allowed():
        st.rerun()
    now = _now()
    _requests(user, game_date, now)
    _bet_candidates(user, game_date, now)


def render_bets(user: AppUser, game_date) -> None:
    _flash()
    now = _now()
    try:
        games = load_today(game_date).games
    except DashboardDataError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    _new_prediction(user, games, game_date, now)
    _live_sections(user, game_date)
    _history(user, now)
