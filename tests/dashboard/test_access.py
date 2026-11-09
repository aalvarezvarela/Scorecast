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
    monkeypatch.setattr(bets_ui, "load_bets", lambda user_id: pd.DataFrame())
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
    requests, candidates = (frame.value for frame in app.dataframe)
    assert requests["Status"].tolist() == ["done"]
    assert requests["Model"].tolist() == ["TabPFN intermediate · per game"]
    # One candidate per game, model, dataset type and target: the newest.
    expected = _manual_rows(today_data).drop_duplicates(bets_ui.CANDIDATE_KEY)
    assert len(candidates) == len(expected)
    assert set(candidates["Source"]) == {"Request #5"}


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


def test_a_refreshed_table_drops_the_selected_position(today_data):
    rows = _manual_rows(today_data)
    assert bets_ui._selection_key(rows) == bets_ui._selection_key(rows.copy())
    assert bets_ui._selection_key(rows) != bets_ui._selection_key(rows.iloc[1:])


def _open_bet_dialog(row, user):
    from nba_ou.dashboard.ui.bets import _bet_dialog

    _bet_dialog(row, user, 25.0)


@pytest.fixture
def bet_dialog(monkeypatch, today_data):
    @contextmanager
    def connection(dsn):
        yield object()

    insert = Mock(return_value=77)
    monkeypatch.setattr(bets_ui, "open_app_connection", connection)
    monkeypatch.setattr(bets_ui, "app_secrets", lambda: Mock(app_db_url="dsn"))
    monkeypatch.setattr(bets_ui.app_store, "insert_bet", insert)
    row = _manual_rows(today_data).iloc[0].to_dict()
    form = "_".join(
        str(int(row[name]))
        for name in ("game_id", "target", "input_id", "model_build_id")
    )
    return row, form, insert


def test_a_bet_without_a_line_needs_one_typed(bet_dialog):
    row, form, insert = bet_dialog
    row = {**row, "bet_line_x2": float("nan"), "bet_side": 1}
    row.update(bet_price_over_home=float("nan"), bet_price_under_away=float("nan"))
    app = AppTest.from_function(
        _open_bet_dialog, kwargs={"row": row, "user": ACTIVE_USER}
    ).run()
    assert not app.exception
    line = app.number_input(key=f"bet_line_{form}")
    assert line.value is None
    assert app.button(key=f"bet_save_{form}").disabled
    assert any("assumed" in caption.value for caption in app.caption)

    line.set_value(224.5).run()
    assert not app.exception
    assert not app.button(key=f"bet_save_{form}").disabled
    app.button(key=f"bet_save_{form}").click().run()
    assert not app.exception
    bet = insert.call_args.args[1]
    assert (bet["line"], bet["stake"]) == (224.5, 25.0)


def test_a_bet_starts_from_the_prediction(bet_dialog):
    row, form, insert = bet_dialog
    app = AppTest.from_function(
        _open_bet_dialog, kwargs={"row": row, "user": ACTIVE_USER}
    ).run()
    assert not app.exception
    expected = bets_ui.bet_rules.proposed_line(row["target"], row["bet_line_x2"])
    assert expected is not None
    assert app.number_input(key=f"bet_line_{form}").value == expected
    assert not app.button(key=f"bet_save_{form}").disabled
    assert not any("assumed" in caption.value for caption in app.caption)


def test_a_rejected_bet_is_reported(bet_dialog):
    row, form, insert = bet_dialog
    insert.return_value = None
    app = AppTest.from_function(
        _open_bet_dialog, kwargs={"row": row, "user": ACTIVE_USER}
    ).run()
    app.button(key=f"bet_save_{form}").click().run()
    assert not app.exception
    assert any("no longer" in alert.value for alert in app.error)


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
