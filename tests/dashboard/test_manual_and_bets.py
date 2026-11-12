"""Manual requests, the bet form's defaults and bet grading (no database)."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import yaml
from nba_ou.dashboard.domain import bets as bet_rules
from nba_ou.dashboard.domain import manual
from nba_ou.dashboard.domain.access import Access, AppUser, access_of, admin_emails

from tests.support.paths import REPO_ROOT


def test_default_choice_is_tabpfn_intermediate():
    assert manual.DEFAULT_CHOICES == ("tabpfn_intermediate",)
    choice = manual.CHOICES_BY_KEY["tabpfn_intermediate"]
    assert (choice.kind, choice.dataset_type) == (3, 2)


@pytest.mark.parametrize(
    ("kind", "dataset_type", "horizon", "workflow", "inputs"),
    [
        (1, 1, None, manual.PREDICTOR_WORKFLOW, {"slots": "closing"}),
        (1, 2, None, manual.PREDICTOR_WORKFLOW, {"slots": "nearest"}),
        (1, 2, 360, manual.PREDICTOR_WORKFLOW, {"slots": "360"}),
        (3, 1, None, manual.TABPFN_WORKFLOW, {"line": "closing", "update_db": "false"}),
        (
            3,
            2,
            None,
            manual.TABPFN_WORKFLOW,
            {
                "line": "intermediate",
                "input": "latest",
                "horizon_minutes": "nearest",
                "update_db": "false",
            },
        ),
        (
            3,
            2,
            480,
            manual.TABPFN_WORKFLOW,
            {
                "line": "intermediate",
                "input": "latest",
                "horizon_minutes": "480",
                "update_db": "false",
            },
        ),
    ],
)
def test_dispatch_inputs(kind, dataset_type, horizon, workflow, inputs):
    got_workflow, got = manual.dispatch_inputs(
        12, kind, dataset_type, horizon, ("line_error",)
    )
    assert got_workflow == workflow
    targets = (
        {"targets": "line_error", "update_db": "false"}
        if workflow == manual.PREDICTOR_WORKFLOW
        else {"target": "line_error"}
    )
    assert got == {"request_id": "12", **inputs, **targets}
    # Every value is a string: the dispatch API requires it.
    assert all(isinstance(value, str) for value in got.values())


def test_both_tabpfn_lines_share_one_run():
    assert manual.runs_jointly(["tabpfn_closing", "tabpfn_intermediate"])
    assert not manual.runs_jointly(["tabpfn_intermediate", "xgboost_closing"])
    workflow, inputs = manual.dispatch_inputs(
        12, 3, 2, None, ("line_error",), closing_request_id=13
    )
    assert workflow == manual.TABPFN_WORKFLOW
    assert inputs["request_id"] == "12"
    assert inputs["closing_request_id"] == "13"
    assert inputs["line"] == "intermediate" and inputs["input"] == "latest"
    # Only the TabPFN intermediate run builds what the closing line needs.
    for kind, dataset_type in [(3, 1), (1, 2), (1, 1)]:
        with pytest.raises(ValueError, match="closing line"):
            manual.dispatch_inputs(
                12, kind, dataset_type, None, ("line_error",), closing_request_id=13
            )


def test_dispatch_rejects_bad_requests():
    target = ("line_error",)
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 1, 1, 360, target)
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 3, 2, 999, target)
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 2, 1, None, target)
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 1, 1, None, ())
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 1, 1, None, ("over_under",))


def test_requests_default_to_line_error_in_a_fixed_target_order():
    assert manual.DEFAULT_TARGETS == ("line_error",)
    _, inputs = manual.dispatch_inputs(1, 3, 1, None, ("spread_error", "line_error"))
    assert inputs["target"] == "line_error spread_error"


def test_workflow_inputs_exist():
    for kind, dataset_type, horizon in [(1, 2, None), (3, 2, None), (3, 1, None)]:
        workflow, inputs = manual.dispatch_inputs(
            1, kind, dataset_type, horizon, manual.DEFAULT_TARGETS
        )
        spec = yaml.safe_load((REPO_ROOT / ".github/workflows" / workflow).read_text())
        # An unquoted " #" would cut the expression: GitHub then rejects the file.
        assert spec["run-name"].endswith("}}"), workflow
        declared = spec[True]["workflow_dispatch"]["inputs"]
        assert set(inputs) <= set(declared), workflow
    workflow, inputs = manual.dispatch_inputs(
        1, 3, 2, None, manual.DEFAULT_TARGETS, closing_request_id=2
    )
    spec = yaml.safe_load((REPO_ROOT / ".github/workflows" / workflow).read_text())
    assert set(inputs) <= set(spec[True]["workflow_dispatch"]["inputs"])


def test_queued_requests_expire():
    now = datetime(2026, 11, 6, 20, tzinfo=UTC)
    assert manual.status_label(0, now - timedelta(minutes=5), now) == "queued"
    assert manual.status_label(0, now - timedelta(hours=4), now) == "expired"
    assert manual.status_label(1, now - timedelta(hours=4), now) == "running"
    assert not manual.is_open(2, now, now)


def test_request_labels():
    assert manual.request_label(3, 2, None) == "TabPFN intermediate · per game"
    assert manual.request_label(1, 2, 360) == "XGBoost intermediate · T−6h"
    assert manual.request_label(1, 1, None) == "XGBoost closing"


def test_preview_skips_started_games():
    assert manual.preview_horizons({"a": 250, "b": -5}) == {"a": 240}


def test_access_levels():
    assert access_of(None) == Access.ANONYMOUS
    assert access_of(AppUser(1, "a", None, 0, False)) == Access.PENDING
    assert access_of(AppUser(1, "a", None, 2, True)) == Access.DISABLED
    assert access_of(AppUser(1, "a", None, 1, False)) == Access.ACTIVE
    assert access_of(AppUser(1, "a", None, 1, True)) == Access.ADMIN
    assert admin_emails(" A@x.com, b@y.com ,") == {"a@x.com", "b@y.com"}
    assert admin_emails(["C@z.com"]) == {"c@z.com"}
    assert admin_emails(None) == frozenset()


def _card_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "predicted_at": pd.to_datetime(
                ["2026-11-09 17:00", "2026-11-09 18:00", "2026-11-09 19:00"], utc=True
            ),
            "bet_line_x2": [449.0, 451.0, float("nan")],
            "bet_edge": [1.5, -4.0, -2.0],
            "bet_side": [1, -1, -1],
            "model_build_id": [11, 12, 13],
        }
    )


def test_a_card_offers_its_newest_stored_line():
    assert bet_rules.newest_quote(_card_rows())["bet_line_x2"] == 451.0
    no_line = _card_rows().assign(bet_line_x2=float("nan"))
    assert bet_rules.newest_quote(no_line)["model_build_id"] == 13


def test_a_bet_follows_the_largest_edge_on_its_side():
    rows = _card_rows()
    assert bet_rules.followed_prediction(rows, 1)["model_build_id"] == 11
    assert bet_rules.followed_prediction(rows, -1)["model_build_id"] == 12
    assert bet_rules.followed_prediction(rows.iloc[:1], -1) is None


def test_the_favored_side_is_the_majority():
    rows = _card_rows()
    assert bet_rules.favored_side(rows) == -1
    assert bet_rules.favored_side(rows.iloc[:2]) is None


def test_bet_form_defaults():
    assert bet_rules.market_of_target(1) == bet_rules.MARKET_TOTALS
    assert bet_rules.market_of_target(2) == bet_rules.MARKET_TOTALS
    assert bet_rules.market_of_target(3) == bet_rules.MARKET_SPREAD
    assert bet_rules.proposed_line(1, 449) == 224.5
    # Implied home margin +3.5 is the home -3.5 line.
    assert bet_rules.proposed_line(3, 7) == -3.5
    assert bet_rules.proposed_line(3, None) is None
    assert bet_rules.american_to_decimal(-110) == 1.909
    assert bet_rules.american_to_decimal(150) == 2.5
    assert bet_rules.american_to_decimal(None) is None
    assert bet_rules.proposed_odds(-1, -110, 120) == 2.2
    assert bet_rules.proposed_odds(1, None, None) == bet_rules.DEFAULT_ODDS
    assert bet_rules.stored_odds(1, None, -110) is None
    assert bet_rules.stored_odds(-1, None, -110) == 1.909
    assert bet_rules.default_stake(None) == 10
    assert bet_rules.default_stake(25.0) == 25


def test_describe_bet():
    assert bet_rules.describe_bet(1, -1, 224.5, "BOS", "NYK") == "Under 224.5"
    assert bet_rules.describe_bet(2, 1, -3.5, "BOS", "NYK") == "BOS -3.5"
    assert bet_rules.describe_bet(2, -1, -3.5, "BOS", "NYK") == "NYK +3.5"


def test_grading_and_profit():
    kwargs = dict(home_points=120, away_points=110, game_status=1)
    assert bet_rules.grade(market=1, side=1, line=224.5, **kwargs) == "win"
    assert bet_rules.grade(market=1, side=1, line=230, **kwargs) == "push"
    assert bet_rules.grade(market=2, side=-1, line=-12.5, **kwargs) == "win"
    assert (
        bet_rules.grade(
            market=1,
            side=1,
            line=224.5,
            home_points=None,
            away_points=None,
            game_status=0,
        )
        == "pending"
    )
    assert bet_rules.grade(market=1, side=1, line=1, voided=True, **kwargs) == "void"
    assert bet_rules.profit("win", 10, 1.909) == 9.09
    assert bet_rules.profit("loss", 10, 1.909) == -10
    assert bet_rules.profit("push", 10, 1.909) == 0
    assert bet_rules.profit("pending", 10, 1.909) is None


def test_summary_counts_only_settled_bets():
    bets = pd.DataFrame(
        {
            "result": ["win", "loss", "pending", "void"],
            "stake": [10, 10, 10, 10],
            "profit": [9.09, -10, None, 0],
        }
    )
    totals = bet_rules.summary(bets)
    assert totals["staked"] == 20
    assert totals["profit"] == pytest.approx(-0.91)
    assert totals["pending"] == 1
    assert (totals["wins"], totals["losses"]) == (1, 1)


def _line_rows(*lines_x2, target=1) -> pd.DataFrame:
    return pd.DataFrame({"bet_line_x2": list(lines_x2), "target": target})


def test_a_card_knows_when_its_predictions_used_different_lines():
    assert not bet_rules.lines_differ(_line_rows(449, 449))
    assert bet_rules.lines_differ(_line_rows(449, 451))
    # A prediction without a stored line is not another line.
    assert not bet_rules.lines_differ(_line_rows(449, None))


def test_a_predictions_line_reads_as_the_book_prints_it():
    assert bet_rules.line_text(1, 449) == "224.5"
    assert bet_rules.line_text(2, 450) == "225"
    # Stored as the implied home margin: +3.5 is the home handicap -3.5.
    assert bet_rules.line_text(3, 7) == "-3.5"
    assert bet_rules.line_text(3, -7) == "+3.5"
    assert bet_rules.line_text(1, None) == "—"


def test_the_line_column_appears_only_when_asked_and_marks_other_lines():
    from nba_ou.dashboard.ui.bet_table import (
        BetTableRow,
        bet_line_html,
        bet_predictions_html,
    )

    rows = [
        BetTableRow(
            "TabPFN closing", "Line error", "226.0", "+1.5", "OVER", "red", line="224.5"
        ),
        BetTableRow(
            "TabPFN T−6h",
            "Line error",
            "226.0",
            "+0.5",
            "OVER",
            "red",
            line="225.5",
            line_differs=True,
        ),
    ]
    plain = bet_predictions_html(rows)
    assert "Line</th>" not in plain and "225.5" not in plain
    with_lines = bet_predictions_html(rows, show_line=True)
    assert with_lines.count('<th scope="col">') == 5
    assert "<td>224.5</td>" in with_lines
    assert '<td class="line-differs">225.5</td>' in with_lines
    assert ">224.5<" in bet_line_html("Totals · Bet365 line", "224.5")
    assert "bet-line-missing" in bet_line_html("Totals · Bet365 line", None)


def _upcoming_game(**changes) -> pd.Series:
    now = datetime(2026, 11, 12, 15, 0, tzinfo=UTC)
    game = {
        "status": 0,
        "tipoff_at": now + timedelta(hours=6),
        "home_report_filed": True,
        "away_report_filed": False,
        "reports_read_at": now - timedelta(minutes=20),
    }
    return pd.Series({**game, **changes}), now


def test_an_upcoming_game_is_flagged_while_a_report_is_missing():
    from nba_ou.dashboard.domain.reports import report_flag

    game, now = _upcoming_game()
    flag = report_flag(game, now)
    assert flag.missing == ("away",) and not flag.filed
    assert flag.read_at == pd.Timestamp(now - timedelta(minutes=20))
    game, now = _upcoming_game(away_report_filed=True)
    assert report_flag(game, now).filed


def test_no_report_flag_without_filing_state_or_after_tip_off():
    from nba_ou.dashboard.domain.reports import report_flag

    game, now = _upcoming_game()
    # The injury-report store was not readable: nothing is claimed.
    assert (
        report_flag(game.drop(["home_report_filed", "away_report_filed"]), now) is None
    )
    assert report_flag(_upcoming_game(home_report_filed=None)[0], now) is None
    assert report_flag(_upcoming_game(status=1)[0], now) is None
    assert (
        report_flag(_upcoming_game(tipoff_at=now - timedelta(minutes=1))[0], now)
        is None
    )
