"""Madrid cutoffs, previous available outputs, series means and shared-quote grading."""

from datetime import date

import pandas as pd
import pytest
from nba_ou.dashboard.domain.clock import (
    SOURCES,
    clock_bets,
    clock_cutoffs,
    clock_members,
    clock_quotes,
    clock_summary,
    hour_label,
    madrid_cutoff,
    series_names,
)

DAY = date(2026, 11, 2)
AS_OF = pd.Timestamp("2026-11-03T08:00:00Z")
TIPOFF = pd.Timestamp("2026-11-03T03:00:00Z")
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
        input_at=pd.Timestamp("2026-11-02T17:00:00Z"),
        available_at=pd.Timestamp("2026-11-02T17:20:00Z"),
        prediction_value=2.0,
        model_line=230.0,
        tipoff_at_prediction=TIPOFF,
        run_id=1,
        input_id=1,
        model_build_id=1,
    )
    row.update(changes)
    return row


def quote(*, game_id=1, source=SOURCES[0], **changes):
    row = dict(
        game_id=game_id,
        source=source,
        input_id=1,
        input_at=pd.Timestamp("2026-11-02T17:30:00Z"),
        available_at=pd.Timestamp("2026-11-02T17:35:00Z"),
        tipoff_at_prediction=TIPOFF,
        bet_line_x2=460,
        bet_price_over_home=-110,
        bet_price_under_away=-110,
        quote_age_at_input=5.0,
    )
    row.update(changes)
    return row


def evaluate(
    rows, *, games=None, quotes=None, source=SOURCES[0], hours=(19,), **kwargs
):
    return clock_bets(
        pd.DataFrame(games or [clock_game()]),
        pd.DataFrame(rows),
        pd.DataFrame([quote(source=source)] if quotes is None else quotes),
        source,
        hours,
        AS_OF,
        **kwargs,
    )


def test_intermediate_series_average_each_models_previous_output():
    bets, members = evaluate(
        [
            output(prediction_value=-20),
            output(
                run_id=2,
                available_at=pd.Timestamp("2026-11-02T17:40Z"),
                prediction_value=6,
            ),
            output(XGB_8H, model_line=220, prediction_value=20),
            output(XGB_CLOSE, prediction_value=5),
            output(TAB_6H, prediction_value=4),
            output(TAB_4H, prediction_value=6),
            output(TAB_CLOSE, prediction_value=3),
        ]
    )
    result = bets.set_index("series")
    assert len(bets) == 4 and len(members) == 6
    assert result.loc[XGB_MEAN, "predicted_value"] == 238
    assert result.loc[XGB_MEAN, "members"] == 2
    assert result.loc[XGB_FINAL, "predicted_value"] == 235
    assert result.loc[TAB_MEAN, "predicted_value"] == 235
    assert result.loc[TAB_MEAN, "members"] == 2
    assert result.loc[TAB_FINAL, "predicted_value"] == 233
    assert result["reference_line"].eq(230).all()
    assert result["result"].eq("WIN").all()
    assert result.loc[XGB_MEAN, "unit_profit"] == pytest.approx(100 / 110)
    assert result.loc[XGB_FINAL, "quote_age_minutes"] == pytest.approx(35)


def test_each_hour_uses_only_what_was_available_before_it():
    bets, _ = evaluate(
        [
            output(prediction_value=1),
            output(
                run_id=2,
                available_at=pd.Timestamp("2026-11-02T18:30Z"),
                prediction_value=9,
            ),
            # This slot appears only after 19:00 Madrid, then joins the mean.
            output(
                XGB_8H,
                input_at=pd.Timestamp("2026-11-02T18:10Z"),
                available_at=pd.Timestamp("2026-11-02T18:20Z"),
                prediction_value=3,
            ),
        ],
        hours=(19, 20),
    )
    series = bets.loc[bets["series"].eq(XGB_MEAN)].set_index("hour")
    assert series.loc[19, "predicted_value"] == 231
    assert series.loc[19, "members"] == 1
    assert series.loc[20, "predicted_value"] == 236
    assert series.loc[20, "members"] == 2


def test_no_future_input_or_late_publication_and_tabpfn_uses_closest_snapshot():
    bets, _ = evaluate(
        [
            output(
                TAB_6H, input_at=pd.Timestamp("2026-11-02T17:40Z"), prediction_value=4
            ),
            # Published later, but of an older snapshot: must not replace it.
            output(
                TAB_6H,
                input_at=pd.Timestamp("2026-11-02T16:00Z"),
                available_at=pd.Timestamp("2026-11-02T17:55Z"),
                prediction_value=99,
            ),
            output(
                XGB_6H, input_at=pd.Timestamp("2026-11-02T18:01Z"), prediction_value=99
            ),
            output(
                XGB_CLOSE,
                available_at=pd.Timestamp("2026-11-02T18:01Z"),
                prediction_value=99,
            ),
        ]
    )
    result = bets.set_index("series")
    assert result.loc[TAB_MEAN, "predicted_value"] == 234
    assert result.loc[[XGB_MEAN, XGB_FINAL, TAB_FINAL], "result"].eq("MISSING").all()
    summary = clock_summary(bets).set_index("series")
    assert summary.loc[XGB_FINAL, "eligible_games"] == 1
    assert summary.loc[XGB_FINAL, "coverage"] == 0
    assert summary.loc[TAB_MEAN, "coverage"] == 100


def test_hidden_variants_and_a_moved_tipoff_never_count():
    cutoffs = clock_cutoffs(pd.DataFrame([clock_game()]), (19,), AS_OF)
    members = clock_members(
        pd.DataFrame(
            [
                output(TAB_6H, variant="tabpfn_thinking_high", prediction_value=99),
                output(
                    tipoff_at_prediction=pd.Timestamp("2026-11-02T17:30Z"),
                    prediction_value=99,
                ),
            ]
        ),
        cutoffs,
        SOURCES[0],
    )
    assert members.empty


def test_replay_uses_input_time_and_never_fills_live_gaps():
    replay = output(source=SOURCES[1], available_at=AS_OF - pd.Timedelta(hours=1))
    bets, _ = evaluate([replay], source=SOURCES[1])
    assert bets.loc[bets["series"].eq(XGB_MEAN), "available"].all()
    live, members = evaluate([replay], source=SOURCES[0])
    assert not live["available"].any()
    assert members.empty


def test_quote_is_the_newest_known_at_the_hour_and_ages_with_time():
    cutoffs = clock_cutoffs(pd.DataFrame([clock_game()]), (19, 20), AS_OF)
    chosen = clock_quotes(
        pd.DataFrame(
            [
                quote(bet_line_x2=460),
                quote(
                    input_id=2,
                    input_at=pd.Timestamp("2026-11-02T18:30Z"),
                    available_at=pd.Timestamp("2026-11-02T18:40Z"),
                    bet_line_x2=470,
                ),
                # A later input known only after 20:00 Madrid stays out.
                quote(
                    input_id=3,
                    input_at=pd.Timestamp("2026-11-02T18:50Z"),
                    available_at=pd.Timestamp("2026-11-02T19:30Z"),
                    bet_line_x2=480,
                ),
                quote(
                    input_id=4,
                    source=SOURCES[1],
                    input_at=pd.Timestamp("2026-11-02T18:55Z"),
                    bet_line_x2=490,
                ),
            ]
        ),
        cutoffs,
        SOURCES[0],
    ).set_index("hour")
    assert chosen.loc[19, "bet_line_x2"] == 460
    assert chosen.loc[20, "bet_line_x2"] == 470
    assert chosen.loc[19, "quote_age_minutes"] == pytest.approx(35)


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
    started = clock_game()
    started["tipoff_at"] = madrid_cutoff(DAY, 19)
    assert evaluate([], games=[started], hours=(19, 20))[0].empty
    bets, _ = clock_bets(
        pd.DataFrame([clock_game()]),
        pd.DataFrame(),
        pd.DataFrame(),
        SOURCES[0],
        (19,),
        madrid_cutoff(DAY, 19) - pd.Timedelta(seconds=1),
    )
    assert bets.empty


def test_common_games_require_every_hour_and_series_without_requiring_a_bet():
    rows = [
        output(slot, game_id=game, prediction_value=0)
        for game in (1, 2)
        for slot in (XGB_CLOSE, XGB_6H, TAB_CLOSE, TAB_6H)
        if not (game == 2 and slot == TAB_6H)
    ]
    bets, _ = evaluate(
        rows,
        games=[clock_game(1), clock_game(2)],
        quotes=[quote(game_id=1), quote(game_id=2)],
        hours=(19, 20),
    )
    assert clock_summary(bets)["eligible_games"].eq(2).all()
    common = clock_summary(bets, common_games=True)
    assert len(common) == 8
    assert common["eligible_games"].eq(1).all()
    assert common["no_bet"].eq(1).all()
    assert common["coverage"].eq(100).all()


def test_push_no_line_and_missing_prices_do_not_create_roi_or_extra_bets():
    push, _ = evaluate(
        [output(XGB_CLOSE, prediction_value=5)], quotes=[quote(bet_line_x2=466)]
    )
    closing = clock_summary(push).set_index("series").loc[XGB_FINAL]
    assert closing["pushes"] == 1
    no_line, _ = evaluate([output()], quotes=[])
    assert clock_summary(no_line).set_index("series").loc[XGB_MEAN, "no_line"] == 1
    unpriced, _ = evaluate(
        [output(TAB_6H)],
        quotes=[quote(bet_price_over_home=None, bet_price_under_away=None)],
    )
    summary = clock_summary(unpriced).set_index("series").loc[TAB_MEAN]
    assert summary["wins"] == 1 and summary["priced_bets"] == 0
    assert pd.isna(summary["roi"])


def test_spread_and_total_targets_preserve_their_scales():
    spread, _ = evaluate(
        [output(XGB_CLOSE, target=3, model_line=1.0, prediction_value=4.0)],
        quotes=[quote(bet_line_x2=4)],
        target=3,
    )
    closing = spread.set_index("series").loc[XGB_FINAL]
    assert closing["predicted_value"] == 5
    assert closing["signal"] == "HOME" and closing["result"] == "WIN"
    total, _ = evaluate([output(XGB_CLOSE, target=2, prediction_value=240)], target=2)
    assert total.set_index("series").loc[XGB_FINAL, "predicted_value"] == 240


def test_family_filter_and_empty_source_keep_missing_coverage_visible():
    assert series_names(3) == [TAB_MEAN, TAB_FINAL]
    assert series_names(1) == [XGB_MEAN, XGB_FINAL]
    bets, members = evaluate([output(XGB_6H)], quotes=[], kind=3)
    assert bets["series"].tolist() == [TAB_MEAN, TAB_FINAL]
    assert set(bets["result"]) == {"MISSING"}
    assert clock_summary(bets)["coverage"].eq(0).all()
    assert members.empty
