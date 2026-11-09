"""Madrid cutoffs, publication availability, aggregation and shared-line grading."""

from datetime import date

import pandas as pd
import pytest
from nba_ou.dashboard.domain.clock import (
    SOURCES,
    clock_bets,
    clock_members,
    clock_summary,
    hour_label,
    madrid_cutoff,
)

DAY = date(2026, 11, 2)
AS_OF = pd.Timestamp("2026-11-03T08:00:00Z")


def clock_game(game_id=1, **changes):
    return dict(
        game_id=game_id,
        game_date=DAY,
        tipoff_at=pd.Timestamp("2026-11-03T03:00:00Z"),
        home_team="Boston",
        away_team="New York",
        status=1,
        home_points=118,
        away_points=115,
        **changes,
    )


def clock_row(*, game_id=1, horizon=360, kind=1, hour=19, source=SOURCES[0], **changes):
    row = dict(
        **clock_game(game_id),
        kind=kind,
        schema_version="2_5",
        target=1,
        dataset_type=1 if horizon == 0 else 2,
        model_horizon_minutes=horizon,
        variant="main",
        source=source,
        hour=hour,
        cutoff_at=madrid_cutoff(DAY, hour),
        input_at=pd.Timestamp("2026-11-02T17:00:00Z"),
        available_at=pd.Timestamp("2026-11-02T17:20:00Z"),
        selection_at=pd.Timestamp("2026-11-02T17:20:00Z"),
        prediction_value=2.0,
        model_line=230.0,
        run_id=1,
        input_id=1,
        model_build_id=1,
        bet_line_x2=460,
        bet_price_over_home=-110,
        bet_price_under_away=-110,
        quote_input_at=pd.Timestamp("2026-11-02T17:30:00Z"),
        quote_age_minutes=35.0,
    )
    row.update(changes)
    if source == SOURCES[1]:
        row["selection_at"] = row["input_at"]
    return row


def evaluate(rows, *, games=None, source=SOURCES[0], hours=(19,), **kwargs):
    return clock_bets(
        pd.DataFrame(games or [clock_game()]),
        pd.DataFrame(rows),
        source,
        hours,
        AS_OF,
        **kwargs,
    )


def test_latest_per_slot_then_mean_absolute_points_and_one_bet_per_series():
    bets, members = evaluate(
        [
            clock_row(prediction_value=-20),
            clock_row(
                run_id=2,
                available_at=pd.Timestamp("2026-11-02T17:40Z"),
                selection_at=pd.Timestamp("2026-11-02T17:40Z"),
                prediction_value=6,
            ),
            clock_row(horizon=480, model_line=220, prediction_value=20),
            clock_row(horizon=0, prediction_value=5),
            clock_row(horizon=360, kind=3, prediction_value=3),
            clock_row(
                horizon=240,
                kind=3,
                input_at=pd.Timestamp("2026-11-02T17:10Z"),
                prediction_value=4,
            ),
        ]
    )
    assert len(bets) == 3 and len(members) == 4
    result = bets.set_index("series")
    assert result.loc["Intermediate", "predicted_value"] == 238
    assert result.loc["Intermediate", "members"] == 2
    assert result.loc["Closing", "predicted_value"] == 235
    assert result.loc["TabPFN", "predicted_value"] == 234
    assert result["reference_line"].eq(230).all()
    assert result["result"].eq("WIN").all()
    assert result.loc["Intermediate", "unit_profit"] == pytest.approx(100 / 110)


def test_no_future_input_or_late_publication_and_tabpfn_uses_closest_snapshot():
    rows = [
        clock_row(
            kind=3, input_at=pd.Timestamp("2026-11-02T17:40Z"), prediction_value=4
        ),
        clock_row(
            kind=3,
            input_at=pd.Timestamp("2026-11-02T16:00Z"),
            available_at=pd.Timestamp("2026-11-02T17:55Z"),
            selection_at=pd.Timestamp("2026-11-02T17:55Z"),
            prediction_value=99,
        ),
        clock_row(
            kind=3,
            horizon=120,
            input_at=pd.Timestamp("2026-11-02T18:01Z"),
            prediction_value=99,
        ),
        clock_row(
            horizon=0,
            available_at=pd.Timestamp("2026-11-02T18:01Z"),
            prediction_value=99,
        ),
        clock_row(horizon=480, variant="tabpfn_thinking_high", prediction_value=99),
    ]
    bets, members = evaluate(rows)
    assert len(members) == 1
    assert members.iloc[0]["prediction_value"] == 4
    assert bets.set_index("series").loc["TabPFN", "predicted_value"] == 234
    assert bets["result"].value_counts()["MISSING"] == 2
    summary = clock_summary(bets).set_index("series")
    assert summary.loc["Closing", "eligible_games"] == 1
    assert summary.loc["Closing", "coverage"] == 0
    assert summary.loc["TabPFN", "coverage"] == 100


def test_replay_uses_input_time_and_never_fills_live_gaps():
    replay = clock_row(source=SOURCES[1], available_at=AS_OF - pd.Timedelta(hours=1))
    bets, _ = evaluate([replay], source=SOURCES[1])
    assert bets["available"].sum() == 1
    live, members = evaluate([replay], source=SOURCES[0])
    assert not live["available"].any()
    assert members.empty


@pytest.mark.parametrize(
    "day,hour,expected",
    [
        (DAY, 19, "2026-11-02T18:00Z"),
        (DAY, 25, "2026-11-03T00:00Z"),
        (date(2026, 7, 1), 19, "2026-07-01T17:00Z"),
        (date(2026, 10, 24), 26, "2026-10-25T01:00Z"),
        (date(2026, 3, 28), 26, "2026-03-29T01:00Z"),
    ],
)
def test_madrid_midnight_and_dst_match_postgres(day, hour, expected):
    assert madrid_cutoff(day, hour) == pd.Timestamp(expected)
    assert hour_label(25) == "01:00 (+1 day)"


def test_started_games_and_future_betting_times_are_not_opportunities():
    games = [clock_game()]
    games[0]["tipoff_at"] = madrid_cutoff(DAY, 19)
    bets, _ = evaluate([], games=games, hours=(19, 20))
    assert bets.empty
    bets, _ = clock_bets(
        pd.DataFrame([clock_game()]),
        pd.DataFrame(),
        SOURCES[0],
        (19,),
        madrid_cutoff(DAY, 19) - pd.Timedelta(seconds=1),
    )
    assert bets.empty


def test_common_games_require_every_hour_and_series_without_requiring_a_bet():
    rows = [
        clock_row(
            game_id=game, hour=hour, horizon=horizon, kind=kind, prediction_value=0
        )
        for game in (1, 2)
        for hour in (19, 20)
        for kind, horizon in ((1, 0), (1, 360), (3, 240))
        if not (game == 2 and hour == 20 and kind == 3)
    ]
    bets, _ = evaluate(rows, games=[clock_game(1), clock_game(2)], hours=(19, 20))
    assert clock_summary(bets)["eligible_games"].eq(2).all()
    common = clock_summary(bets, common_games=True)
    assert len(common) == 6
    assert common["eligible_games"].eq(1).all()
    assert common["no_bet"].eq(1).all()
    assert common["coverage"].eq(100).all()


def test_push_no_line_and_missing_prices_do_not_create_roi_or_extra_bets():
    bets, _ = evaluate(
        [
            clock_row(horizon=0, bet_line_x2=466, prediction_value=5),
            clock_row(horizon=360, bet_line_x2=None),
            clock_row(kind=3, bet_price_over_home=None, bet_price_under_away=None),
        ]
    )
    summary = clock_summary(bets).set_index("series")
    assert summary.loc["Closing", "pushes"] == 1
    assert summary.loc["Intermediate", "no_line"] == 1
    assert summary.loc["TabPFN", "wins"] == 1
    assert summary.loc["TabPFN", "priced_bets"] == 0
    assert pd.isna(summary.loc["TabPFN", "roi"])


def test_spread_and_total_targets_preserve_their_scales():
    bets, _ = evaluate(
        [
            clock_row(
                horizon=0, target=3, model_line=1.0, prediction_value=4.0, bet_line_x2=4
            )
        ],
        target=3,
    )
    closing = bets.loc[bets["series"].eq("Closing")].iloc[0]
    assert closing["predicted_value"] == 5
    assert closing["signal"] == "HOME" and closing["result"] == "WIN"
    bets, _ = evaluate([clock_row(horizon=0, target=2, prediction_value=240)], target=2)
    assert bets.loc[bets["series"].eq("Closing"), "predicted_value"].item() == 240


def test_family_filter_and_empty_source_keep_missing_coverage_visible():
    bets, _ = evaluate([], kind=3)
    assert bets["series"].tolist() == ["TabPFN"]
    assert bets["result"].tolist() == ["MISSING"]
    assert list(clock_summary(bets)["coverage"]) == [0]
    assert clock_members(pd.DataFrame(), SOURCES[0]).empty
