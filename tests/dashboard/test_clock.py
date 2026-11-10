"""Madrid hours, closest original prediction, its own line and result, coverage."""

from datetime import date

import pandas as pd
import pytest
from nba_ou.dashboard.domain.clock import (
    SOURCES,
    best_cells,
    clock_bets,
    clock_cutoffs,
    clock_summary,
    common_cohort,
    coverage_text,
    hour_label,
    hour_model_table,
    leader_checks,
    madrid_cutoff,
    select_predictions,
    series_names,
)

DAY = date(2026, 11, 2)
AS_OF = pd.Timestamp("2026-11-03T08:00:00Z")
TIPOFF = pd.Timestamp("2026-11-03T03:00:00Z")
# 19:00 in Madrid on DAY, the hour most tests evaluate.
HOUR = pd.Timestamp("2026-11-02T18:00:00Z")
XGB_CLOSE, XGB_6H, XGB_8H = (1, 1, 0), (1, 2, 360), (1, 2, 480)
TAB_CLOSE, TAB_6H, TAB_4H = (3, 1, 0), (3, 2, 360), (3, 2, 240)
XGB_MEAN, XGB_FINAL = "XGBoost · Intermediate", "XGBoost · Closing"
TAB_MEAN, TAB_FINAL = "TabPFN · Intermediate", "TabPFN · Closing"


def clock_game(game_id=1, **changes):
    return dict(
        game_id=game_id,
        game_date=DAY,
        tipoff_at=TIPOFF,
        home_team="Boston",
        away_team="New York",
        status=1,
        home_points=118,
        away_points=115,
        **changes,
    )


def at(minutes):
    """The instant ``minutes`` after 19:00 Madrid."""
    return HOUR + pd.Timedelta(minutes=minutes)


def output(slot=XGB_6H, *, game_id=1, target=1, source=SOURCES[0], **changes):
    kind, dataset, horizon = slot
    row = dict(
        game_id=game_id,
        source=source,
        kind=kind,
        schema_version="2_5",
        target=target,
        dataset_type=dataset,
        model_horizon_minutes=horizon,
        variant="main" if kind == 1 else "tabpfn_v3_5",
        input_at=at(-30),
        predicted_at=at(-20),
        available_at=at(-20),
        prediction_value=2.0,
        model_line=230.0,
        bet_line_x2=460,
        bet_price_over_home=-110,
        bet_price_under_away=-110,
        tipoff_at_prediction=TIPOFF,
        run_id=1,
        input_id=1,
        model_build_id=1,
    )
    row.update(changes)
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


def series(bets, name):
    return bets.set_index("series").loc[name]


def test_the_closest_prediction_is_chosen_and_keeps_its_own_line_and_side():
    bets = evaluate(
        [
            output(
                run_id=1, predicted_at=at(-50), bet_line_x2=500, prediction_value=-9
            ),
            output(
                run_id=2,
                predicted_at=at(-15),
                model_line=225.5,
                bet_line_x2=451,
                prediction_value=1,
            ),
            output(run_id=3, predicted_at=at(40), bet_line_x2=470, prediction_value=-5),
        ]
    )
    chosen = series(bets, XGB_MEAN)
    assert chosen["run_id"] == 2
    assert chosen["reference_line"] == 225.5
    assert chosen["signal"] == "OVER" and chosen["result"] == "WIN"
    assert chosen["minutes_from_hour"] == -15


def test_the_later_prediction_can_be_the_closest_one():
    bets = evaluate(
        [
            output(run_id=1, predicted_at=at(-30), bet_line_x2=460),
            output(run_id=2, predicted_at=at(10), bet_line_x2=470),
        ]
    )
    assert series(bets, XGB_MEAN)["reference_line"] == 235


@pytest.mark.parametrize(
    "minutes,chosen", [(-60, True), (60, True), (-61, False), (61, False)]
)
def test_the_window_is_one_hour_each_side_and_nothing_fills_it(minutes, chosen):
    bets = evaluate([output(predicted_at=at(minutes))])
    row = series(bets, XGB_MEAN)
    assert row["available"] == chosen
    assert row["result"] == ("WIN" if chosen else "MISSING")
    assert clock_summary(bets).set_index("series").loc[XGB_MEAN, "losses"] == 0


def test_a_tie_on_both_sides_of_the_hour_takes_the_earlier_prediction():
    bets = evaluate(
        [
            output(run_id=2, predicted_at=at(20), bet_line_x2=470),
            output(run_id=1, predicted_at=at(-20), bet_line_x2=460),
        ]
    )
    assert series(bets, XGB_MEAN)["run_id"] == 1


def test_models_of_a_group_are_never_averaged_and_one_run_takes_the_nearest_horizon():
    # Tip-off is 9 hours after the hour: the T-480 model is the nearest one.
    bets = evaluate(
        [
            output(XGB_6H, predicted_at=at(0), bet_line_x2=460, prediction_value=-20),
            output(XGB_8H, predicted_at=at(0), bet_line_x2=452, prediction_value=4),
        ]
    )
    chosen = series(bets, XGB_MEAN)
    assert chosen["model_horizon_minutes"] == 480
    assert chosen["reference_line"] == 226 and chosen["predicted_value"] == 234
    assert clock_summary(bets).set_index("series").loc[XGB_MEAN, "bets"] == 1


def test_replay_uses_the_snapshot_time_not_when_the_backfill_ran():
    replay = output(
        source=SOURCES[1],
        input_at=at(-45),
        predicted_at=AS_OF,
        available_at=AS_OF,
    )
    bets = evaluate([replay], source=SOURCES[1])
    assert series(bets, XGB_MEAN)["reference_at"] == at(-45)
    assert series(bets, XGB_MEAN)["available"]
    assert not evaluate([replay], source=SOURCES[0])["available"].any()


def test_live_uses_the_generation_time_not_the_snapshot_or_publication():
    bets = evaluate(
        [
            output(
                TAB_6H,
                input_at=at(-300),
                predicted_at=at(-10),
                available_at=at(90),
            )
        ]
    )
    assert series(bets, TAB_MEAN)["reference_at"] == at(-10)


def test_a_prediction_below_min_edge_stays_chosen_as_no_bet():
    bets = evaluate(
        [
            output(run_id=1, predicted_at=at(-5), prediction_value=0.1),
            output(run_id=2, predicted_at=at(-30), prediction_value=9),
        ],
        thresholds={1: 2.0, 3: 2.0},
    )
    chosen = series(bets, XGB_MEAN)
    assert chosen["run_id"] == 1 and chosen["result"] == "NO BET"
    summary = clock_summary(bets).set_index("series").loc[XGB_MEAN]
    assert summary["covered_games"] == 1 and summary["bets"] == 0


def test_predictions_without_a_line_or_after_tipoff_or_hidden_are_not_candidates():
    cutoffs = clock_cutoffs(pd.DataFrame([clock_game()]), (19,), AS_OF)
    chosen = select_predictions(
        pd.DataFrame(
            [
                output(bet_line_x2=None),
                output(prediction_value=None),
                output(TAB_6H, variant="tabpfn_thinking_high"),
                output(tipoff_at_prediction=at(-40)),
            ]
        ),
        cutoffs,
        SOURCES[0],
    )
    assert chosen.empty


def test_a_closing_snapshot_at_tipoff_is_a_candidate():
    game = clock_game()
    game["tipoff_at"] = at(30)
    bets = evaluate(
        [
            output(
                XGB_CLOSE,
                source=SOURCES[1],
                input_at=at(30),
                tipoff_at_prediction=at(30),
            )
        ],
        games=[game],
        source=SOURCES[1],
    )
    assert series(bets, XGB_FINAL)["available"]


def test_a_prediction_can_serve_two_neighbouring_hours():
    bets = evaluate([output(predicted_at=at(30))], hours=(19, 20, 21))
    chosen = bets.loc[bets["series"].eq(XGB_MEAN)].set_index("hour")
    assert chosen["available"].tolist() == [True, True, False]


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


def test_games_are_opportunities_until_the_window_opens_after_tipoff():
    started = clock_game()
    started["tipoff_at"] = at(30)
    hours = evaluate([], games=[started], hours=(19, 20, 21))["hour"].unique()
    assert hours.tolist() == [19, 20]
    started["tipoff_at"] = at(0)
    assert evaluate([], games=[started], hours=(20,)).empty


def test_future_betting_windows_are_not_opportunities():
    assert clock_bets(
        pd.DataFrame([clock_game()]),
        pd.DataFrame(),
        SOURCES[0],
        (19,),
        at(-61),
    ).empty


def test_coverage_counts_valid_games_over_final_games_without_counting_losses():
    games = [clock_game(game) for game in (1, 2, 3)]
    pending = clock_game(4)
    pending["status"] = 0
    pending["home_points"] = pending["away_points"] = None
    bets = evaluate(
        [
            output(game_id=1),
            output(game_id=2, prediction_value=-2),
            output(game_id=4),
        ],
        games=[*games, pending],
    )
    summary = clock_summary(bets).set_index("series").loc[XGB_MEAN]
    assert summary["eligible_games"] == 3 and summary["covered_games"] == 2
    assert summary["coverage"] == pytest.approx(200 / 3)
    assert (summary["wins"], summary["losses"]) == (1, 1)
    assert summary["win_pct"] == 50
    assert summary["missing"] == 1 and summary["pending"] == 1
    assert coverage_text(94, 120) == "94/120 (78.3%)"


def test_common_games_require_every_hour_and_series_without_requiring_a_bet():
    rows = [
        output(slot, game_id=game, prediction_value=0, predicted_at=at(30))
        for game in (1, 2)
        for slot in (XGB_CLOSE, XGB_6H, TAB_CLOSE, TAB_6H)
        if not (game == 2 and slot == TAB_6H)
    ]
    bets = evaluate(rows, games=[clock_game(1), clock_game(2)], hours=(19, 20))
    assert clock_summary(bets)["eligible_games"].eq(2).all()
    common = clock_summary(common_cohort(bets))
    assert len(common) == 8
    assert common["eligible_games"].eq(1).all()
    assert common["no_bet"].eq(1).all()
    assert common["coverage"].eq(100).all()


def test_push_and_missing_prices_do_not_create_roi_or_extra_bets():
    push = evaluate(
        [output(XGB_CLOSE, bet_line_x2=466, model_line=233, prediction_value=2)]
    )
    closing = clock_summary(push).set_index("series").loc[XGB_FINAL]
    assert closing["pushes"] == 1 and pd.isna(closing["win_pct"])
    unpriced = evaluate(
        [output(TAB_6H, bet_price_over_home=None, bet_price_under_away=None)]
    )
    summary = clock_summary(unpriced).set_index("series").loc[TAB_MEAN]
    assert summary["wins"] == 1 and summary["priced_bets"] == 0
    assert pd.isna(summary["roi"])


def test_each_prediction_uses_its_own_prices():
    bets = evaluate([output(XGB_CLOSE, bet_price_over_home=120)])
    assert series(bets, XGB_FINAL)["unit_profit"] == pytest.approx(1.2)


def test_spread_and_total_targets_preserve_their_scales():
    spread = evaluate(
        [
            output(
                XGB_CLOSE, target=3, model_line=1.0, bet_line_x2=4, prediction_value=4.0
            )
        ],
        target=3,
    )
    closing = series(spread, XGB_FINAL)
    assert closing["predicted_value"] == 5
    assert closing["signal"] == "HOME" and closing["result"] == "WIN"
    total = evaluate([output(XGB_CLOSE, target=2, prediction_value=240)], target=2)
    assert series(total, XGB_FINAL)["predicted_value"] == 240


def test_family_filter_and_empty_source_keep_missing_coverage_visible():
    assert series_names(3) == [TAB_MEAN, TAB_FINAL]
    assert series_names(1) == [XGB_MEAN, XGB_FINAL]
    bets = evaluate([output(XGB_6H)], kind=3)
    assert bets["series"].tolist() == [TAB_MEAN, TAB_FINAL]
    assert set(bets["result"]) == {"MISSING"}
    assert clock_summary(bets)["coverage"].eq(0).all()


def summary_row(hour, series, wins, losses, roi=5.0, eligible=98, covered=None):
    return dict(
        hour=hour,
        hour_label=hour_label(hour),
        series=series,
        wins=wins,
        losses=losses,
        bets=wins + losses,
        eligible_games=eligible,
        covered_games=wins + losses if covered is None else covered,
        win_pct=100 * wins / (wins + losses) if wins + losses else float("nan"),
        roi=roi,
    )


def test_best_cells_rank_by_win_rate_among_enough_settled_bets():
    summary = pd.DataFrame(
        [
            summary_row(18, XGB_MEAN, 40, 20),
            summary_row(19, TAB_MEAN, 21, 9),
            summary_row(20, XGB_FINAL, 20, 9),  # 29 settled bets: too few
            summary_row(21, TAB_FINAL, 33, 27),
            summary_row(22, XGB_MEAN, 30, 30),
            summary_row(23, TAB_MEAN, 25, 35),
            summary_row(12, XGB_FINAL, 0, 0),
        ]
    )
    best = best_cells(summary)
    assert list(zip(best["hour"], best["series"], strict=True)) == [
        (19, TAB_MEAN),
        (18, XGB_MEAN),
        (21, TAB_FINAL),
        (22, XGB_MEAN),
    ]
    assert best_cells(summary.iloc[[2, 6]]).empty


def test_hour_table_has_one_row_per_hour_and_one_column_per_series():
    summary = pd.DataFrame(
        [
            summary_row(19, XGB_MEAN, 53, 39, roi=9.5),
            summary_row(18, XGB_MEAN, 2, 1, roi=float("nan")),
            summary_row(18, TAB_MEAN, 0, 0, roi=float("nan")),
        ]
    )
    table = hour_model_table(summary, [XGB_MEAN, TAB_MEAN])
    assert table.columns.tolist() == ["Hour (Madrid)", XGB_MEAN, TAB_MEAN]
    assert table["Hour (Madrid)"].tolist() == ["18:00", "19:00"]
    assert table[XGB_MEAN].tolist() == [
        "66.7% · 2-1 · 3/98 (3.1%)",
        "57.6% · 53-39 · 92/98 (93.9%)",
    ]
    assert table[TAB_MEAN].tolist() == ["— · 0/98 (0.0%)", "—"]


def test_leader_checks_cover_every_eligible_cell():
    rows = [
        output(XGB_6H, game_id=game, prediction_value=2 if game < 26 else -2)
        for game in range(40)
    ]
    bets = evaluate(
        rows,
        games=[clock_game(game) for game in range(40)],
        hours=(18, 19),
    )
    summary = clock_summary(bets)
    chance, permutation, tried = leader_checks(bets, summary)
    assert tried == 2
    assert 0 <= chance <= 1 and 0 < permutation <= 1
    assert leader_checks(bets, summary.iloc[0:0]) == (
        pytest.approx(float("nan"), nan_ok=True),
        pytest.approx(float("nan"), nan_ok=True),
        0,
    )


def test_hour_table_can_keep_every_hour_row():
    summary = pd.DataFrame([summary_row(19, XGB_MEAN, 3, 1, eligible=5)])
    table = hour_model_table(summary, [XGB_MEAN, TAB_MEAN], (18, 19, 24))
    assert table["Hour (Madrid)"].tolist() == ["18:00", "19:00", "00:00 (+1 day)"]
    assert table[XGB_MEAN].tolist() == ["—", "75.0% · 3-1 · 4/5 (80.0%)", "—"]
    assert table[TAB_MEAN].eq("—").all()
