"""The sign-in gate and the signed-in pages (Bets, Accounts), without a database."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from nba_ou.dashboard.domain.access import AppUser
from nba_ou.dashboard.ui import admin as admin_ui
from nba_ou.dashboard.ui import auth, loaders
from nba_ou.dashboard.ui import bets as bets_ui
from streamlit.testing.v1 import AppTest

from .conftest import ACTIVE_USER, ADMIN_USER, NOW

ENTRYPOINT = Path(__file__).resolve().parents[2] / "streamlit/streamlit_app.py"
BETS_PAGE = str(ENTRYPOINT.parent / "app_pages/bets.py")
ADMIN_PAGE = str(ENTRYPOINT.parent / "app_pages/admin.py")


def titles(app):
    return [title.value for title in app.title]


@pytest.fixture
def no_slate(monkeypatch):
    reader = Mock(side_effect=AssertionError("the slate must not load"))
    loaders.load_today.clear()
    monkeypatch.setattr(loaders, "read_today", reader)
    yield reader
    loaders.load_today.clear()


def test_signed_out_sees_only_the_welcome(monkeypatch, no_slate):
    monkeypatch.setattr(auth, "current_user", lambda: None)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert titles(app) == ["NBA predictions"]
    assert [button.label for button in app.button] == ["Sign in with Google"]
    no_slate.assert_not_called()


def test_signed_out_cannot_open_another_page(monkeypatch, no_slate):
    monkeypatch.setattr(auth, "current_user", lambda: None)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    # The page is not even registered for a signed-out session.
    with pytest.raises(ValueError, match="Known pages"):
        app.switch_page(BETS_PAGE)
    no_slate.assert_not_called()


@pytest.mark.parametrize(("status", "text"), [(0, "approve"), (2, "disabled")])
def test_pending_and_disabled_accounts_wait(monkeypatch, no_slate, status, text):
    monkeypatch.setattr(
        auth, "current_user", lambda: AppUser(3, "new@example.com", None, status, False)
    )
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    assert not app.exception
    assert titles(app) == ["Access pending"]
    assert any(text in alert.value for alert in [*app.info, *app.warning])
    no_slate.assert_not_called()


def _manual_rows(today_data) -> pd.DataFrame:
    rows = today_data.predictions.copy()
    rows["kind"] = 3
    rows["request_id"] = 5
    rows["predicted_value"] = rows["model_line"] + rows["prediction_value"]
    rows.loc[rows["target"].eq(2), "predicted_value"] = rows["prediction_value"]
    rows["bet_edge"] = rows["predicted_value"] - rows["bet_line_x2"] / 2
    rows["bet_side"] = rows["bet_edge"].gt(0).map({True: 1, False: -1})
    rows["bet_price_over_home"] = -110
    rows["bet_price_under_away"] = -110
    return rows


@pytest.fixture
def bets_page(monkeypatch, today_data):
    loaders.load_today.clear()
    monkeypatch.setattr(loaders, "read_today", Mock(return_value=today_data))
    monkeypatch.setattr(bets_ui, "_now", lambda: NOW.to_pydatetime())
    monkeypatch.setattr(
        bets_ui,
        "load_requests",
        lambda user_id, game_date: pd.DataFrame(
            {
                "request_id": [5],
                "kind": [3],
                "dataset_type": [2],
                "horizon_minutes": [None],
                "game_date": [NOW.date()],
                "status": [2],
                "requested_at": [NOW - pd.Timedelta(minutes=20)],
                "started_at": [NOW - pd.Timedelta(minutes=18)],
                "finished_at": [NOW - pd.Timedelta(minutes=8)],
                "error": [None],
            }
        ),
    )
    monkeypatch.setattr(
        bets_ui, "load_manual_predictions", lambda user_id, d: _manual_rows(today_data)
    )
    monkeypatch.setattr(bets_ui, "load_bets", lambda user_id, d: pd.DataFrame())
    monkeypatch.setattr(bets_ui, "load_last_stake", lambda user_id: 25.0)
    yield
    loaders.load_today.clear()


def test_bets_page_lists_requests_and_candidates(bets_page, today_data):
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    assert not app.exception
    assert "Bets" in titles(app)
    # TabPFN intermediate is preselected.
    assert app.get("button_group")[0].value == ["tabpfn_intermediate"]
    # Line error alone is the default target.
    assert app.get("button_group")[1].value == ["line_error"]
    [requests] = [frame.value for frame in app.dataframe]
    assert requests["Status"].tolist() == ["done"]
    assert requests["Model"].tolist() == ["TabPFN intermediate · per game"]
    # One row per game, model, dataset type and target: the newest.
    expected = _manual_rows(today_data).drop_duplicates(bets_ui.CANDIDATE_KEY)
    tables = "".join(html.proto.body for html in app.get("html"))
    assert tables.count('scope="row"') == len(expected)
    assert "Scheduled" not in tables
    # Each game offers both sides of both markets, Over 230.5 first.
    assert app.button(key="bet_1_1_1").label == "Over 230.5"
    assert app.button(key="bet_1_1_-1").label == "Under 230.5"
    assert app.button(key="bet_1_2_1").label == "Boston Celtics -3.5"
    assert app.button(key="bet_1_2_-1").label == "New York Knicks +3.5"


def test_a_card_shows_its_line_large_and_each_predictions_own_when_they_differ(
    bets_page, monkeypatch, today_data
):
    rows = _manual_rows(today_data)
    game_one = rows["game_id"].eq(1) & rows["target"].isin([1, 2])
    older = rows.index[game_one][0]
    # One totals prediction of game 1 was made earlier, against 229.5.
    rows.loc[older, "bet_line_x2"] = 459
    rows.loc[older, "predicted_at"] -= pd.Timedelta(minutes=30)
    monkeypatch.setattr(bets_ui, "load_manual_predictions", lambda user_id, d: rows)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    assert not app.exception
    html = [element.proto.body for element in app.get("html")]
    lines = [body for body in html if 'class="bet-line"' in body]
    # The newest quote is the card's line, on every market section.
    assert any(">230.5<" in body for body in lines)
    with_column = [body for body in html if "Line</th>" in body]
    # Only game 1's totals section lists the lines, the older one marked.
    assert len(with_column) == 1
    assert '<td class="line-differs">229.5</td>' in with_column[0]
    assert "<td>230.5</td>" in with_column[0]
    assert any("different lines" in text.value for text in app.caption)


def test_upcoming_games_flag_a_missing_injury_report(bets_page, today_data):
    games = today_data.games
    games["home_report_filed"] = True
    games["away_report_filed"] = games["game_id"].ne(1)
    games["reports_read_at"] = NOW - pd.Timedelta(minutes=30)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    assert not app.exception
    badges = [text.value for text in app.markdown if "badge" in text.value]
    assert sum("No injury report yet: Knicks" in badge for badge in badges) == 1
    assert sum("Injury reports filed" in badge for badge in badges) == 1


def test_launch_creates_a_request_and_dispatches_it(bets_page, monkeypatch):
    @contextmanager
    def connection(dsn):
        yield object()

    created, dispatched = Mock(return_value=41), Mock()
    monkeypatch.setattr(bets_ui, "open_app_connection", connection)
    monkeypatch.setattr(bets_ui.app_store, "create_request", created)
    monkeypatch.setattr(bets_ui, "dispatch_workflow", dispatched)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    app.button(key="manual_launch").click().run()
    assert not app.exception
    assert created.call_args.kwargs | {"game_date": None} == {
        "user_id": 2,
        "kind": 3,
        "dataset_type": 2,
        "horizon_minutes": None,
        "game_date": None,
    }
    _, workflow, inputs = dispatched.call_args.args
    assert workflow == "nba_predictor_daily_tabpfn_client.yml"
    assert inputs["request_id"] == "41"
    assert inputs["horizon_minutes"] == "nearest"
    assert inputs["target"] == "line_error"
    assert any("Launched" in text.value for text in app.success)


def test_both_tabpfn_lines_launch_as_two_requests_and_one_run(bets_page, monkeypatch):
    @contextmanager
    def connection(dsn):
        yield object()

    created, dispatched = Mock(side_effect=[41, 42]), Mock()
    monkeypatch.setattr(bets_ui, "open_app_connection", connection)
    monkeypatch.setattr(bets_ui.app_store, "create_request", created)
    monkeypatch.setattr(bets_ui, "dispatch_workflow", dispatched)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    # Closing picked first: the intermediate request still carries the run.
    app.session_state["manual_models"] = ["tabpfn_closing", "tabpfn_intermediate"]
    app.run()
    app.button(key="manual_launch").click().run()
    assert not app.exception
    assert [call.kwargs["dataset_type"] for call in created.call_args_list] == [2, 1]
    dispatched.assert_called_once()
    _, workflow, inputs = dispatched.call_args.args
    assert workflow == "nba_predictor_daily_tabpfn_client.yml"
    assert inputs["request_id"] == "41"
    assert inputs["closing_request_id"] == "42"
    assert inputs["line"] == "intermediate"
    assert any("one run" in text.value for text in app.success)


def test_a_failed_joint_dispatch_fails_both_requests(bets_page, monkeypatch):
    @contextmanager
    def connection(dsn):
        yield object()

    failed = Mock()
    monkeypatch.setattr(bets_ui, "open_app_connection", connection)
    monkeypatch.setattr(bets_ui.app_store, "create_request", Mock(side_effect=[8, 9]))
    monkeypatch.setattr(bets_ui.app_store, "fail_request", failed)
    monkeypatch.setattr(
        bets_ui,
        "dispatch_workflow",
        Mock(side_effect=bets_ui.DispatchError("GitHub refused (HTTP 403)")),
    )
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    app.session_state["manual_models"] = ["tabpfn_intermediate", "tabpfn_closing"]
    app.run()
    app.button(key="manual_launch").click().run()
    assert [call.args[1] for call in failed.call_args_list] == [8, 9]


def test_failed_dispatch_marks_the_request_failed(bets_page, monkeypatch):
    @contextmanager
    def connection(dsn):
        yield object()

    failed = Mock()
    monkeypatch.setattr(bets_ui, "open_app_connection", connection)
    monkeypatch.setattr(bets_ui.app_store, "create_request", Mock(return_value=8))
    monkeypatch.setattr(bets_ui.app_store, "fail_request", failed)
    monkeypatch.setattr(
        bets_ui,
        "dispatch_workflow",
        Mock(side_effect=bets_ui.DispatchError("GitHub refused (HTTP 403)")),
    )
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    app.button(key="manual_launch").click().run()
    assert failed.call_args.args[1:] == (8, "GitHub refused (HTTP 403)")


def _scheduled(today_data) -> pd.DataFrame:
    return today_data.predictions.assign(request_id=float("nan"))


def test_scheduled_candidates_are_todays_selection(today_data):
    chosen = bets_ui._scheduled_candidates(_scheduled(today_data), NOW)
    intermediate = chosen.loc[chosen["dataset_type"].eq(2)]
    # Game 1 tips in 4 hours, game 2 in 2: T-240 and T-120, whatever ran last.
    horizons = intermediate.groupby("game_id")["model_horizon_minutes"].unique()
    assert [list(h) for h in horizons.loc[[1, 2]]] == [[240], [120]]
    assert chosen["dataset_type"].eq(1).sum() == 6


def test_another_users_manual_run_cannot_take_a_scheduled_slot(today_data):
    rows = _scheduled(today_data)
    rows.loc[
        rows["game_id"].eq(1) & rows["model_horizon_minutes"].eq(240), "request_id"
    ] = 9
    chosen = bets_ui._scheduled_candidates(rows, NOW)
    assert chosen["request_id"].isna().all()
    game = chosen.loc[chosen["game_id"].eq(1) & chosen["dataset_type"].eq(2)]
    assert set(game["model_horizon_minutes"]) != {240}
    assert len(game) == 3


def test_own_and_scheduled_predictions_stay_side_by_side(today_data):
    scheduled = bets_ui._scheduled_candidates(_scheduled(today_data), NOW)
    own = scheduled.assign(request_id=5)
    both = bets_ui._candidates(own, scheduled, NOW)
    assert len(both) == 2 * len(scheduled)
    assert both["request_id"].notna().sum() == len(scheduled)


def test_own_candidates_keep_the_newest_per_horizon(today_data):
    own = _manual_rows(today_data)
    older = own.assign(
        predicted_at=own["predicted_at"] - pd.Timedelta(hours=1),
        run_id=own["run_id"] - 1,
        predicted_value=-1.0,
    )
    rows = bets_ui._candidates(pd.concat([older, own]), pd.DataFrame(), NOW)
    assert len(rows) == len(own.drop_duplicates(bets_ui.CANDIDATE_KEY))
    assert rows["predicted_value"].ne(-1.0).all()
    assert rows["model_horizon_minutes"].nunique() > 1


def test_only_requested_targets_get_their_market(bets_page, monkeypatch, today_data):
    rows = _manual_rows(today_data)
    monkeypatch.setattr(
        bets_ui, "load_manual_predictions", lambda *_: rows.loc[rows["target"].eq(1)]
    )
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    assert not app.exception
    keys = {button.key for button in app.button}
    assert {"bet_1_1_1", "bet_1_1_-1"} <= keys
    assert not any(key.startswith("bet_1_2_") for key in keys)


def _bets_app(monkeypatch, insert=None):
    """The Bets page with its writes faked.

    AppTest reruns the whole script on every interaction, so it cannot show
    that a browser reruns only the fragment the widgets live in.
    """

    @contextmanager
    def connection(dsn):
        yield object()

    insert = insert or Mock(return_value=77)
    monkeypatch.setattr(bets_ui, "open_app_connection", connection)
    monkeypatch.setattr(bets_ui, "app_secrets", lambda: Mock(app_db_url="dsn"))
    monkeypatch.setattr(bets_ui.app_store, "insert_bet", insert)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    return app, insert


def _form_of(app) -> str:
    [stake] = [box for box in app.number_input if box.key.startswith("bet_stake_")]
    return stake.key.removeprefix("bet_stake_")


def test_a_bet_button_opens_the_form_in_the_card_with_the_last_stake(
    bets_page, monkeypatch
):
    app, _ = _bets_app(monkeypatch)
    app.button(key="bet_1_1_1").click().run()
    assert not app.exception
    form = _form_of(app)
    assert form.startswith("1_1_1_")
    assert app.number_input(key=f"bet_stake_{form}").value == 25.0
    assert app.number_input(key=f"bet_line_{form}").value == 230.5
    assert app.number_input(key=f"bet_odds_{form}").value == 1.909
    # Another side replaces it: one form at a time.
    app.button(key="bet_2_1_-1").click().run()
    assert _form_of(app).startswith("2_1_-1_")
    app.button(key=f"bet_cancel_{_form_of(app)}").click().run()
    assert not any(box.key.startswith("bet_stake_") for box in app.number_input)


def test_saving_a_bet_closes_its_form_and_remembers_the_stake(
    bets_page, monkeypatch, today_data
):
    app, insert = _bets_app(monkeypatch)
    app.button(key="bet_1_1_1").click().run()
    form = _form_of(app)
    app.number_input(key=f"bet_odds_{form}").set_value(2.05)
    app.number_input(key=f"bet_line_{form}").set_value(231.0)
    app.number_input(key=f"bet_stake_{form}").set_value(40.0)
    app.button(key=f"bet_save_{form}").click().run()
    assert not app.exception
    bet = insert.call_args.args[1]
    assert (bet["line"], bet["odds_decimal"], bet["stake"]) == (231.0, 2.05, 40.0)
    assert bet["book"] == "bet365"
    assert bet["request_id"] == 5
    offer = bets_ui._offer(
        today_data.games.loc[today_data.games["game_id"].eq(1)].iloc[0],
        _manual_rows(today_data).loc[lambda rows: rows["game_id"].eq(1)],
        bets_ui.MARKET_TOTALS,
        1,
    )
    assert bet["model_build_id"] == offer["followed"]["model_build_id"]
    # The form is gone and the bet is confirmed.
    assert not any(box.key.startswith("bet_stake_") for box in app.number_input)
    assert any("Saved: Over 231" in toast.value for toast in app.toast)
    # The next form starts from the stake just used, without a database read.
    app.button(key="bet_2_1_1").click().run()
    assert app.number_input(key=f"bet_stake_{_form_of(app)}").value == 40.0


def test_a_bet_without_a_line_needs_one_typed(bets_page, monkeypatch, today_data):
    rows = _manual_rows(today_data).assign(
        bet_line_x2=float("nan"),
        bet_price_over_home=float("nan"),
        bet_price_under_away=float("nan"),
    )
    monkeypatch.setattr(bets_ui, "load_manual_predictions", lambda user_id, d: rows)
    app, insert = _bets_app(monkeypatch)
    app.button(key="bet_1_1_1").click().run()
    form = _form_of(app)
    assert app.number_input(key=f"bet_line_{form}").value is None
    assert any("assumed" in caption.value for caption in app.caption)
    app.button(key=f"bet_save_{form}").click().run()
    insert.assert_not_called()
    assert any("Enter the line" in alert.value for alert in app.error)
    # The form stays open with what was typed.
    app.number_input(key=f"bet_line_{form}").set_value(224.5)
    app.button(key=f"bet_save_{form}").click().run()
    assert not app.exception
    bet = insert.call_args.args[1]
    assert (bet["line"], bet["stake"]) == (224.5, 25.0)


def test_a_side_no_model_backs_is_stored_without_a_prediction(
    bets_page, monkeypatch, today_data
):
    # Every prediction backs the Over: nobody backs the Under.
    rows = _manual_rows(today_data).assign(bet_side=1)
    monkeypatch.setattr(bets_ui, "load_manual_predictions", lambda user_id, d: rows)
    app, insert = _bets_app(monkeypatch)
    app.button(key="bet_1_1_-1").click().run()
    assert any("No model" in caption.value for caption in app.caption)
    app.button(key=f"bet_save_{_form_of(app)}").click().run()
    bet = insert.call_args.args[1]
    assert bet["side"] == -1
    assert (bet["model_build_id"], bet["input_id"], bet["request_id"]) == (
        None,
        None,
        None,
    )


def test_a_rejected_bet_is_reported_in_the_form(bets_page, monkeypatch):
    app, insert = _bets_app(monkeypatch, insert=Mock(return_value=None))
    app.button(key="bet_1_1_1").click().run()
    form = _form_of(app)
    app.button(key=f"bet_save_{form}").click().run()
    assert not app.exception
    assert any("no longer" in alert.value for alert in app.error)
    assert _form_of(app) == form


def _tonight_bets(**changes) -> pd.DataFrame:
    """Two bets on game 1 (4 h ahead), one on a game already under way."""
    bets = pd.DataFrame(
        {
            "bet_id": [11, 12, 13],
            "game_id": [1, 1, 2],
            "game_date": NOW.date(),
            "tipoff_at": [
                NOW + pd.Timedelta(hours=4),
                NOW + pd.Timedelta(hours=4),
                NOW - pd.Timedelta(minutes=30),
            ],
            "home_team": ["Boston Celtics", "Boston Celtics", "Los Angeles Lakers"],
            "away_team": [
                "New York Knicks",
                "New York Knicks",
                "Golden State Warriors",
            ],
            "market": [1, 2, 1],
            "side": [1, -1, -1],
            # Stored as the home handicap: the Knicks were backed at +3.5.
            "line": [230.5, -3.5, 228.0],
            "odds_decimal": [1.909, 2.0, 1.909],
            "stake": [10.0, 20.0, 5.0],
            "book": "bet365",
            "placed_at": NOW - pd.Timedelta(hours=1),
            "voided_at": pd.NaT,
            "game_status": 0,
            "result": "pending",
            "profit": float("nan"),
        }
    )
    return bets.assign(**changes)


@pytest.fixture
def tonight(bets_page, monkeypatch):
    @contextmanager
    def connection(dsn):
        yield object()

    update, void = Mock(return_value=True), Mock(return_value=True)
    monkeypatch.setattr(bets_ui, "load_bets", lambda user_id, d: _tonight_bets())
    monkeypatch.setattr(bets_ui, "open_app_connection", connection)
    monkeypatch.setattr(bets_ui, "app_secrets", lambda: Mock(app_db_url="dsn"))
    monkeypatch.setattr(bets_ui.app_store, "update_bet", update)
    monkeypatch.setattr(bets_ui.app_store, "void_bet", void)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    return app, update, void


def test_tonights_bets_are_editable_until_tip_off(tonight):
    app, _, _ = tonight
    assert not app.exception
    requests, editable, started = [frame.value for frame in app.dataframe]
    # The away side reads as its own handicap, +3.5.
    assert editable["Bet"].tolist() == ["Over", "New York Knicks"]
    assert editable["Line"].tolist() == [230.5, 3.5]
    assert editable.index.tolist() == [11, 12]
    assert started["Bet"].tolist() == ["Under 228"]
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Bets"] == "3" and metrics["Staked"] == "35.00 €"
    # Nothing changed: nothing to save.
    assert not any(button.key == "bets_save_changes" for button in app.button)


def test_bet_changes_are_what_the_editor_changed():
    open_bets = _tonight_bets().iloc[:2].set_index("bet_id")
    edited = pd.DataFrame(
        {
            "Line": [231.0, 4.5],
            "Odds": [1.909, 2.0],
            "Stake (€)": [10.0, 20.0],
            "Delete": [False, False],
        },
        index=open_bets.index,
    )
    assert bets_ui._bet_changes(open_bets, edited.assign(Line=[230.5, 3.5])) == []
    changes = bets_ui._bet_changes(open_bets, edited)
    assert [change["bet_id"] for change in changes] == [11, 12]
    # The Knicks' +4.5 is stored as the home handicap -4.5.
    assert [change["line"] for change in changes] == [231.0, -4.5]
    deleted = bets_ui._bet_changes(open_bets, edited.assign(Delete=[True, False]))
    assert deleted[0] == {"bet_id": 11, "delete": True}
    # An emptied cell changes nothing.
    emptied = edited.assign(Line=[float("nan"), 3.5])
    assert bets_ui._bet_changes(open_bets, emptied) == []


def test_saved_changes_update_and_delete_in_one_go(tonight, monkeypatch):
    _, update, void = tonight
    monkeypatch.setattr(bets_ui, "still_allowed", lambda: True)
    monkeypatch.setattr(bets_ui, "clear_bets", Mock())
    state = {}
    monkeypatch.setattr(bets_ui.st, "session_state", state)
    void.return_value = False  # its game tipped off meanwhile
    bets_ui._apply_bet_changes(
        ACTIVE_USER,
        [
            {
                "bet_id": 11,
                "delete": False,
                "line": 231.0,
                "odds_decimal": 1.95,
                "stake": 12.0,
            },
            {"bet_id": 12, "delete": True},
        ],
    )
    assert update.call_args.args[1:] == (11, ACTIVE_USER.user_id)
    assert update.call_args.kwargs == {
        "line": 231.0,
        "odds_decimal": 1.95,
        "stake": 12.0,
    }
    assert void.call_args.args[1:] == (12, ACTIVE_USER.user_id)
    assert state[bets_ui.TOAST_KEY] == "Bets: 1 updated, 1 not changed (game started)"
    assert state[bets_ui.EDITOR_VERSION_KEY] == 1
    bets_ui.clear_bets.assert_called_once()


@pytest.mark.parametrize("failure", [None, ValueError("no url")])
def test_still_allowed_refuses_lost_access(monkeypatch, failure):
    disabled = AppUser(2, "fan@example.com", "Fan", 2, False)
    current = Mock(return_value=disabled, side_effect=failure)
    monkeypatch.setattr(auth, "current_user", current)
    assert not auth.still_allowed()


def test_a_disabled_account_leaves_the_live_sections(bets_page, monkeypatch):
    state = {"disabled": False}
    disabled = AppUser(2, "fan@example.com", "Fan", 2, False)
    monkeypatch.setattr(
        auth, "current_user", lambda: disabled if state["disabled"] else ACTIVE_USER
    )
    # Disabled once the gate has passed: what a fragment run sees.
    monkeypatch.setattr(
        bets_ui, "_new_prediction", lambda *args: state.update(disabled=True)
    )
    candidates = Mock(return_value=pd.DataFrame())
    monkeypatch.setattr(bets_ui, "load_manual_predictions", candidates)
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(BETS_PAGE).run()
    assert not app.exception
    assert titles(app) == ["Access pending"]
    candidates.assert_not_called()


def test_accounts_page_is_for_admins(monkeypatch, today_data):
    loaders.load_today.clear()
    monkeypatch.setattr(loaders, "read_today", Mock(return_value=today_data))
    monkeypatch.setattr(auth, "current_user", lambda: ADMIN_USER)
    monkeypatch.setattr(
        admin_ui,
        "load_users",
        Mock(
            return_value=pd.DataFrame(
                {
                    "user_id": [1, 2],
                    "email": ["admin@example.com", "new@example.com"],
                    "display_name": ["Admin", None],
                    "status": [1, 0],
                    "is_admin": [True, False],
                    "created_at": [NOW, NOW],
                    "last_login_at": [NOW, NOW],
                }
            )
        ),
    )
    app = AppTest.from_file(str(ENTRYPOINT)).run()
    app.switch_page(ADMIN_PAGE).run()
    assert not app.exception
    assert titles(app)[-1] == "Accounts"
    assert app.selectbox(key="admin_account").options == ["new@example.com (Pending)"]
    loaders.load_today.clear()
