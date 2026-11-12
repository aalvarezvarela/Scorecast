"""The Bets page: launch manual predictions, bet on them, keep the history.

1. New prediction: the models to run (TabPFN intermediate by default), the
   targets (line error by default) and, for the intermediate models, each
   game's own horizon or a fixed one. Each model is a request dispatched to
   GitHub Actions; TabPFN intermediate and closing together share one run.
2. Today's requests and their status.
3. One card per game with the predictions of those requests (optionally
   every live prediction of the slate). A click on Over or Under, or on Home
   or Away for a game with spread predictions, opens a short form in the
   card: stake, line and odds, which start from the last stake and the stored
   bet365 quote. Saving it reruns only the live sections.
4. Tonight's bets: the user's bets on this slate, editable (line, odds,
   stake) and deletable until tip-off. The history has its own page
   (``ui/bet_history.py``); a past slate's bets also show on Yesterday.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import psycopg

import streamlit as st
from nba_ou.dashboard.config import (
    MODEL_NAMES,
    PREDICTION_TARGET_LABELS,
    SIGNAL_COLORS,
)
from nba_ou.dashboard.domain import bets as bet_rules
from nba_ou.dashboard.domain import manual
from nba_ou.dashboard.domain.access import AppUser
from nba_ou.dashboard.domain.dates import horizon_label, madrid_clock
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
from nba_ou.dashboard.ui.bet_summary import graded_bets_table
from nba_ou.dashboard.ui.bet_table import (
    BetTableRow,
    bet_line_html,
    bet_predictions_html,
)
from nba_ou.dashboard.ui.formatting import number
from nba_ou.dashboard.ui.game_card import STALE_SNAPSHOT_MINUTES
from nba_ou.dashboard.ui.loaders import load_today
from nba_ou.dashboard.ui.teams import render_game_header
from nba_ou.postgre_db.app_aiven.schema import MARKET_SPREAD, MARKET_TOTALS

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
#: The requests and the bet cards are read again this often: a request's
#: status and its predictions should show as soon as they are stored.
LIVE_REFRESH_SECONDS = 15
#: The bet being placed (what its side button offered): its form is open in
#: the game's card. Placing, editing and deleting bets rerun only the live
#: sections (widget callbacks inside the fragment), never the whole page.
BET_OFFER_KEY = "bets_offer"
BET_ERROR_KEY = "bets_offer_error"
#: A message of the live sections, shown as a toast on their next run.
TOAST_KEY = "bets_toast"
#: The stake of the session's last bet: the next form starts from it.
STAKE_KEY = "bets_last_stake"
EDITOR_VERSION_KEY = "bets_editor_version"
SIDE_ICONS = {
    MARKET_TOTALS: {1: ":material/trending_up:", -1: ":material/trending_down:"},
    MARKET_SPREAD: {1: ":material/home:", -1: ":material/flight:"},
}


def _now() -> datetime:
    return datetime.now(UTC)


def _flash() -> None:
    message = st.session_state.pop(FLASH_KEY, None)
    if message:
        st.success(message, icon=":material/check_circle:")


def _toast() -> None:
    message = st.session_state.pop(TOAST_KEY, None)
    if message:
        st.toast(message, icon=":material/check_circle:")


# -- 1. new prediction ---------------------------------------------------------


def _minutes_left(games: pd.DataFrame, now: datetime) -> dict[str, float]:
    tipoffs = pd.to_datetime(games["tipoff_at"], utc=True)
    minutes = (tipoffs - pd.Timestamp(now)).dt.total_seconds() / 60
    return dict(zip(games["game_id"].astype(str), minutes, strict=True))


def _launch(
    user: AppUser,
    choices: list[str],
    targets: list[str],
    horizon: int | None,
    game_date,
) -> None:
    secrets = app_secrets()
    launched, failed = [], []
    joint = manual.runs_jointly(choices)
    intermediate_key, closing_key = manual.JOINT_TABPFN
    with open_app_connection(secrets.app_db_url) as conn:

        def create(choice: manual.ModelChoice) -> tuple[int, int | None]:
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
            return request_id, request_horizon

        for key in choices:
            if joint and key == closing_key:
                continue  # answered by the intermediate request's run
            choice = manual.CHOICES_BY_KEY[key]
            request_id, request_horizon = create(choice)
            request_ids = [request_id]
            label = manual.request_label(
                choice.kind, choice.dataset_type, request_horizon
            )
            closing_id = None
            if joint and key == intermediate_key:
                closing = manual.CHOICES_BY_KEY[closing_key]
                closing_id, _ = create(closing)
                request_ids.append(closing_id)
                label += f" + {closing.label} (one run)"
            workflow, inputs = manual.dispatch_inputs(
                request_id,
                choice.kind,
                choice.dataset_type,
                request_horizon,
                targets,
                closing_request_id=closing_id,
            )
            try:
                dispatch_workflow(secrets.dispatch_token, workflow, inputs)
            except DispatchError as exc:
                for failed_id in request_ids:
                    app_store.fail_request(conn, failed_id, str(exc))
                failed.append(f"{label}: {exc}")
            else:
                launched.append(label)
    clear_requests()
    if launched:
        names = ", ".join(
            manual.target_label(target) for target in manual.ordered_targets(targets)
        )
        st.session_state[FLASH_KEY] = f"Launched: {', '.join(launched)} ({names})"
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
        targets = st.pills(
            "Targets",
            options=list(manual.TARGETS),
            format_func=manual.target_label,
            selection_mode="multi",
            default=list(manual.DEFAULT_TARGETS),
            key="manual_targets",
            help="Only these targets are fitted and predicted.",
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
            disabled=not choices or not targets,
            key="manual_launch",
        ):
            _launch(user, list(choices), list(targets), horizon, game_date)
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


# -- 3. bet cards ----------------------------------------------------------------


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


def _pick(row: pd.Series) -> str:
    side = row["bet_side"]
    if pd.isna(side) or int(side) == 0:
        return "—"
    market = bet_rules.market_of_target(row["target"])
    return bet_rules.SIDE_LABELS[market][int(side)].upper()


def _table_row(row: pd.Series, card_line_x2: float | None = None) -> BetTableRow:
    """``card_line_x2``: the line the card offers, to mark a prediction made
    against another one."""
    target = PREDICTION_TARGET_LABELS[int(row["target"])]
    spread = bet_rules.market_of_target(row["target"]) == MARKET_SPREAD
    pick = _pick(row)
    return BetTableRow(
        model=_model_label(row),
        detail=f"{target} · Scheduled" if pd.isna(row["request_id"]) else target,
        prediction=number(row["predicted_value"], signed=spread),
        edge=number(row["bet_edge"], signed=True),
        pick=pick,
        color=SIGNAL_COLORS.get(pick, "gray"),
        line=bet_rules.line_text(row["target"], row["bet_line_x2"]),
        line_differs=(
            card_line_x2 is not None
            and pd.notna(card_line_x2)
            and pd.notna(row["bet_line_x2"])
            and float(row["bet_line_x2"]) != float(card_line_x2)
        ),
    )


def _offer(game: pd.Series, rows: pd.DataFrame, market: int, side: int) -> dict:
    """What the bet form starts from: the newest quote of the market, and the
    prediction that backs the side with the largest edge."""
    quote = bet_rules.newest_quote(rows)
    followed = bet_rules.followed_prediction(rows, side)
    return {
        "game_id": int(game["game_id"]),
        "home": game["home_team"],
        "away": game["away_team"],
        "tipoff_at": game["tipoff_at"],
        "market": market,
        "side": side,
        "line": bet_rules.proposed_line(quote["target"], quote["bet_line_x2"]),
        "prices": (quote.get("bet_price_over_home"), quote.get("bet_price_under_away")),
        "quote_id": int(quote["input_id"]),
        "followed": (
            None
            if followed is None
            else {
                "label": (
                    f"{_model_label(followed)} · "
                    f"{PREDICTION_TARGET_LABELS[int(followed['target'])]}"
                ),
                "edge": float(followed["bet_edge"]),
                "model_build_id": int(followed["model_build_id"]),
                "input_id": int(followed["input_id"]),
                "request_id": (
                    None
                    if pd.isna(followed["request_id"])
                    else int(followed["request_id"])
                ),
            }
        ),
    }


def _form_key(offer: dict) -> str:
    """Per quote: a form closed without saving must not lend its values to
    the next one."""
    return f"{offer['game_id']}_{offer['market']}_{offer['side']}_{offer['quote_id']}"


def _open_bet(game: pd.Series, rows: pd.DataFrame, market: int, side: int) -> None:
    """A side button's callback: its form opens in the card, in the same
    fragment run as the click (one form at a time)."""
    st.session_state[BET_OFFER_KEY] = _offer(game, rows, market, side)
    st.session_state.pop(BET_ERROR_KEY, None)


def _close_bet() -> None:
    st.session_state.pop(BET_OFFER_KEY, None)
    st.session_state.pop(BET_ERROR_KEY, None)


def _default_stake(user: AppUser) -> float:
    """The stake of the session's last bet, else the last stored one, else 10."""
    last = st.session_state.get(STAKE_KEY)
    if last is None:
        last = load_last_stake(user.user_id)
    return bet_rules.default_stake(last)


def _save_bet(offer: dict, user: AppUser) -> None:
    """The form's Confirm callback: store the bet and close the form. Only the
    live sections run again, so the rest of the page stays as it is."""
    form = _form_key(offer)
    market, side = offer["market"], offer["side"]
    line = st.session_state.get(f"bet_line_{form}")
    odds = st.session_state.get(f"bet_odds_{form}")
    amount = st.session_state.get(f"bet_stake_{form}")
    if line is None or odds is None or amount is None:
        st.session_state[BET_ERROR_KEY] = "Enter the line, the odds and the stake."
        return
    if not still_allowed():
        _close_bet()  # the fragment's own gate then leaves the page
        return
    followed = offer["followed"]
    bet = {
        "user_id": user.user_id,
        "game_id": offer["game_id"],
        "market": market,
        "side": side,
        "line": float(line),
        "odds_decimal": float(odds),
        "stake": float(amount),
        "book": bet_rules.DEFAULT_BOOK,
        "model_build_id": followed["model_build_id"] if followed else None,
        "input_id": followed["input_id"] if followed else None,
        "request_id": followed["request_id"] if followed else None,
    }
    try:
        with open_app_connection(app_secrets().app_db_url) as conn:
            saved = app_store.insert_bet(conn, bet)
    except psycopg.Error:
        st.session_state[BET_ERROR_KEY] = "The bet could not be stored. Try again."
        return
    if saved is None:
        st.session_state[BET_ERROR_KEY] = "This account can no longer place bets."
        return
    clear_bets()
    st.session_state[STAKE_KEY] = float(amount)
    text = bet_rules.describe_bet(
        market, side, float(line), offer["home"], offer["away"]
    )
    st.session_state[TOAST_KEY] = (
        f"Saved: {text} · {float(amount):g} € at {float(odds):.2f}"
    )
    _close_bet()


def _bet_form(offer: dict, user: AppUser) -> None:
    """The open bet, inside its card: stake, line and odds start from the last
    stake and the stored bet365 quote. Typing does not rerun anything; Confirm
    stores the bet."""
    market, side = offer["market"], offer["side"]
    form = _form_key(offer)
    followed = offer["followed"]
    with st.form(f"bet_form_{form}", border=True):
        name = bet_rules.side_name(market, side, offer["home"], offer["away"])
        st.markdown(f"**Bet: {name}**")
        st.caption(
            f"Follows {followed['label']} (edge {followed['edge']:+.1f})"
            if followed
            else "No model backs this side."
        )
        with st.container(horizontal=True):
            st.number_input(
                "Stake (€)",
                min_value=0.5,
                value=_default_stake(user),
                step=1.0,
                key=f"bet_stake_{form}",
            )
            st.number_input(
                "Home handicap" if market == MARKET_SPREAD else "Total",
                value=offer["line"],
                step=0.5,
                format="%.1f",
                placeholder="Enter the line",
                key=f"bet_line_{form}",
                help=(
                    "The line you got, as the book shows the home side "
                    "(home -3.5 is -3.5)."
                    if market == MARKET_SPREAD
                    else "The total you got."
                ),
            )
            st.number_input(
                "Odds",
                min_value=1.01,
                value=bet_rules.proposed_odds(side, *offer["prices"]),
                step=0.01,
                format="%.3f",
                key=f"bet_odds_{form}",
                help="Decimal odds. Change them if your book offers another price.",
            )
        if bet_rules.stored_odds(side, *offer["prices"]) is None:
            st.caption(
                f"No stored price for this side: {bet_rules.DEFAULT_ODDS} is assumed, "
                "enter the odds you got."
            )
        error = st.session_state.get(BET_ERROR_KEY)
        if error:
            st.error(error, icon=":material/error:")
        with st.container(horizontal=True):
            st.form_submit_button(
                "Confirm bet",
                icon=":material/check:",
                type="primary",
                width="stretch",
                on_click=_save_bet,
                args=(offer, user),
                key=f"bet_save_{form}",
            )
            st.form_submit_button(
                "Cancel",
                width="stretch",
                on_click=_close_bet,
                key=f"bet_cancel_{form}",
            )


def _quote_read(read_at, now) -> str:
    moment = f"Line read {madrid_clock(read_at)} Madrid"
    elapsed = (pd.Timestamp(now) - pd.Timestamp(read_at)).total_seconds() / 60
    if elapsed >= STALE_SNAPSHOT_MINUTES:
        return f":orange[{moment}, {int(elapsed)} min ago: check the live price.]"
    return moment


def _market_section(
    game: pd.Series, rows: pd.DataFrame, market: int, now, user: AppUser
) -> None:
    game_id = int(game["game_id"])
    home, away = game["home_team"], game["away_team"]
    quote = bet_rules.newest_quote(rows)
    line = bet_rules.proposed_line(quote["target"], quote["bet_line_x2"])
    label = f"{bet_rules.MARKET_LABELS[market]} · Bet365 line"
    differ = bet_rules.lines_differ(rows)
    if line is None:
        st.html(bet_line_html(label, None))
        st.caption("No stored Bet365 line: enter the one you get.")
    else:
        shown = (
            bet_rules.describe_bet(market, 1, line, home, away)
            if market == MARKET_SPREAD
            else f"{line:g}"
        )
        st.html(bet_line_html(label, shown))
        st.caption(_quote_read(quote["predicted_at"], now))
        if differ:
            st.caption(
                ":orange[The predictions were made against different lines "
                "(Line column). The newest is the one shown.]"
            )
    st.html(
        bet_predictions_html(
            [_table_row(row, quote["bet_line_x2"]) for _, row in rows.iterrows()],
            show_line=differ,
        )
    )
    favored = bet_rules.favored_side(rows)
    with st.container(horizontal=True):
        for side in (1, -1):
            name = bet_rules.side_name(market, side, home, away)
            if line is not None:
                name = bet_rules.describe_bet(market, side, line, home, away)
            st.button(
                name,
                icon=SIDE_ICONS[market][side],
                type="primary" if side == favored else "secondary",
                width="stretch",
                key=f"bet_{game_id}_{market}_{side}",
                on_click=_open_bet,
                args=(game, rows, market, side),
            )
    offer = st.session_state.get(BET_OFFER_KEY)
    if offer and (offer["game_id"], offer["market"]) == (game_id, market):
        _bet_form(offer, user)


def _game_card(game: pd.Series, rows: pd.DataFrame, now, user: AppUser) -> None:
    with st.container(border=True):
        render_game_header(game, now=now)
        markets = rows["target"].map(bet_rules.market_of_target)
        for market in (MARKET_TOTALS, MARKET_SPREAD):
            market_rows = rows.loc[markets.eq(market)]
            if not market_rows.empty:
                _market_section(game, market_rows, market, now, user)


def _bet_cards(user: AppUser, game_date, games: pd.DataFrame, now) -> None:
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
    slate = games.set_index(games["game_id"].astype(int), drop=False)
    with st.container(horizontal=True, gap="medium", key="games_grid"):
        for game_id, game_rows in candidates.groupby("game_id", sort=False):
            with st.container(width="content", key=f"game_slot_{int(game_id)}"):
                _game_card(slate.loc[int(game_id)], game_rows, now, user)


# -- 4. tonight's bets ------------------------------------------------------------


def _bet_changes(open_bets: pd.DataFrame, edited: pd.DataFrame) -> list[dict]:
    """What the editor changed: per bet, a deletion or its new line, odds and
    stake (the line back as stored: the home handicap for a spread bet). A
    cell left empty changes nothing."""
    changes = []
    for bet_id, row in edited.iterrows():
        bet = open_bets.loc[bet_id]
        if bool(row["Delete"]):
            changes.append({"bet_id": int(bet_id), "delete": True})
            continue
        values = row[["Line", "Odds", "Stake (€)"]]
        if values.isna().any():
            continue
        new = {
            "line": bet_rules.side_line(
                int(bet["market"]), int(bet["side"]), float(row["Line"])
            ),
            "odds_decimal": float(row["Odds"]),
            "stake": float(row["Stake (€)"]),
        }
        if any(abs(value - float(bet[name])) > 1e-9 for name, value in new.items()):
            changes.append({"bet_id": int(bet_id), "delete": False, **new})
    return changes


def _apply_bet_changes(user: AppUser, changes: list[dict]) -> None:
    """The Save button's callback: every change in one connection."""
    if not still_allowed():
        return
    updated = deleted = refused = 0
    try:
        with open_app_connection(app_secrets().app_db_url) as conn:
            for change in changes:
                if change["delete"]:
                    done = app_store.void_bet(conn, change["bet_id"], user.user_id)
                    deleted += done
                else:
                    done = app_store.update_bet(
                        conn,
                        change["bet_id"],
                        user.user_id,
                        line=change["line"],
                        odds_decimal=change["odds_decimal"],
                        stake=change["stake"],
                    )
                    updated += done
                refused += not done
    except psycopg.Error:
        st.session_state[TOAST_KEY] = "The changes could not be stored. Try again."
        return
    finally:
        clear_bets()
    # A new editor: the next one starts from the stored bets.
    st.session_state[EDITOR_VERSION_KEY] = (
        st.session_state.get(EDITOR_VERSION_KEY, 0) + 1
    )
    parts = [
        text
        for count, text in (
            (updated, f"{updated} updated"),
            (deleted, f"{deleted} deleted"),
            (refused, f"{refused} not changed (game started)"),
        )
        if count
    ]
    st.session_state[TOAST_KEY] = "Bets: " + ", ".join(parts)


def _open_bets_editor(user: AppUser, open_bets: pd.DataFrame) -> None:
    """The bets whose game has not started: line, odds and stake editable in
    place, a tick to delete, one Save for all the changes."""
    open_bets = open_bets.set_index("bet_id")
    markets, sides = open_bets["market"].astype(int), open_bets["side"].astype(int)
    stakes = open_bets["stake"].astype(float)
    odds = open_bets["odds_decimal"].astype(float)
    table = pd.DataFrame(
        {
            "Game": open_bets["away_team"] + " @ " + open_bets["home_team"],
            "Bet": [
                bet_rules.side_name(m, s, home, away)
                for m, s, home, away in zip(
                    markets,
                    sides,
                    open_bets["home_team"],
                    open_bets["away_team"],
                    strict=True,
                )
            ],
            "Line": [
                bet_rules.side_line(m, s, line)
                for m, s, line in zip(markets, sides, open_bets["line"], strict=True)
            ],
            "Odds": odds,
            "Stake (€)": stakes,
            "To win (€)": (stakes * (odds - 1)).round(2),
            "Delete": False,
        },
        index=open_bets.index,
    )
    version = st.session_state.get(EDITOR_VERSION_KEY, 0)
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["Game", "Bet", "To win (€)"],
        column_config={
            "Line": st.column_config.NumberColumn(
                step=0.5,
                format="%.1f",
                help="The total, or the handicap of the team you backed.",
            ),
            "Odds": st.column_config.NumberColumn(
                min_value=1.01, step=0.001, format="%.3f"
            ),
            "Stake (€)": st.column_config.NumberColumn(
                min_value=0.5, step=0.5, format="%.2f"
            ),
            "To win (€)": st.column_config.NumberColumn(format="%.2f"),
            "Delete": st.column_config.CheckboxColumn(
                help="Tick and save to delete the bet."
            ),
        },
        # Per set of bets: a new bet must not inherit another row's edits.
        key=f"bets_editor_{version}_{'_'.join(map(str, table.index))}",
    )
    changes = _bet_changes(open_bets, edited)
    if not changes:
        st.caption(
            "Edit the line, odds or stake in the table, or tick Delete; "
            "bets can be changed until tip-off."
        )
        return
    deletions = sum(change["delete"] for change in changes)
    updates = len(changes) - deletions
    summary = " and ".join(
        text
        for count, text in (
            (updates, f"update {updates}"),
            (deletions, f"delete {deletions}"),
        )
        if count
    )
    st.button(
        f"Save changes ({summary})",
        icon=":material/save:",
        type="primary",
        key="bets_save_changes",
        on_click=_apply_bet_changes,
        args=(user, changes),
    )


def _tonight_bets(user: AppUser, game_date, now) -> None:
    """The user's bets on this slate only; the history lives elsewhere."""
    st.subheader("Tonight's bets")
    bets = load_bets(user.user_id, game_date)
    if bets.empty:
        st.caption("No bet on tonight's games yet.")
        return
    totals = bet_rules.slate_summary(bets)
    with st.container(horizontal=True):
        st.metric("Bets", totals["bets"], border=True)
        st.metric("Staked", f"{totals['staked']:.2f} €", border=True)
        st.metric("To win", f"{totals['to_win']:.2f} €", border=True)
        if totals["settled"]:
            st.metric("Profit", f"{totals['profit']:+.2f} €", border=True)
    is_open = bets["result"].eq("pending") & (
        pd.to_datetime(bets["tipoff_at"], utc=True) > pd.Timestamp(now)
    )
    if is_open.any():
        _open_bets_editor(user, bets.loc[is_open])
    started = bets.loc[~is_open]
    if started.empty:
        return
    st.caption("Started or finished: no longer editable.")
    st.dataframe(
        graded_bets_table(started), hide_index=True, width="stretch", alt="Started bets"
    )


# -- page ------------------------------------------------------------------------


@st.fragment(run_every=f"{LIVE_REFRESH_SECONDS}s")
def _live_sections(user: AppUser, game_date, games: pd.DataFrame) -> None:
    # Fragment runs skip the gate in streamlit_app.py.
    if not still_allowed():
        st.rerun()
    _toast()
    now = _now()
    _requests(user, game_date, now)
    _bet_cards(user, game_date, games, now)
    _tonight_bets(user, game_date, now)


def render_bets(user: AppUser, game_date) -> None:
    _flash()
    now = _now()
    try:
        games = load_today(game_date).games
    except DashboardDataError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    _new_prediction(user, games, game_date, now)
    _live_sections(user, game_date, games)
