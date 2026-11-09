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
    got_workflow, got = manual.dispatch_inputs(12, kind, dataset_type, horizon)
    assert got_workflow == workflow
    assert got == {"request_id": "12", **inputs}
    # Every value is a string: the dispatch API requires it.
    assert all(isinstance(value, str) for value in got.values())


def test_dispatch_rejects_bad_requests():
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 1, 1, 360)
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 3, 2, 999)
    with pytest.raises(ValueError):
        manual.dispatch_inputs(1, 2, 1, None)


def test_workflow_inputs_exist():
    for kind, dataset_type, horizon in [(1, 2, None), (3, 2, None), (3, 1, None)]:
        workflow, inputs = manual.dispatch_inputs(1, kind, dataset_type, horizon)
        spec = yaml.safe_load((REPO_ROOT / ".github/workflows" / workflow).read_text())
        # An unquoted " #" would cut the expression: GitHub then rejects the file.
        assert spec["run-name"].endswith("}}"), workflow
        declared = spec[True]["workflow_dispatch"]["inputs"]
        assert set(inputs) <= set(declared), workflow


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
