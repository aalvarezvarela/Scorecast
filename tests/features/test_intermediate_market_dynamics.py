"""Snapshot injury news, news reaction, cross-market and continuation features.

Every feature at snapshot T may read only reports published strictly before T,
ticks at least T minutes before tip, and labels of games that tipped off
before T.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.config.odds_columns import find_unprefixed_odds_columns
from nba_ou.create_training_data import historical_ridge_movement as ridge
from nba_ou.create_training_data.select_intermediate_columns import is_kept_column
from nba_ou.data_processing.injury_status.news import (
    build_news_timeline,
    news_column,
)
from nba_ou.data_processing.line_history.market_dynamics import (
    LevelReader,
    _league_mean_before,
    _moneyline_margin,
    _signed_news,
    cross_market_features,
    news_reaction_features,
    tick_levels,
)
from nba_ou.data_processing.line_history.walk_forward import (
    walk_forward_least_squares,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)

TIP = pd.Timestamp("2024-01-02 00:00", tz="UTC")
PREV_TIP = pd.Timestamp("2023-12-31 00:00", tz="UTC")
HOME, AWAY = "1610612737", "1610612738"


def before_tip(minutes: float) -> pd.Timestamp:
    return TIP - pd.Timedelta(minutes=minutes)


def _span(game, team, player, status, start, end, reason="injury_illness"):
    tip = TIP if game == "g1" else PREV_TIP
    return {
        "game_id": game,
        "team_id": team,
        "player_id": player,
        "status": status,
        "reason_category": reason,
        "valid_from": start,
        "valid_to": end,
        "game_date": tip.normalize().tz_localize(None),
        "season_year": 2023,
        "tipoff_utc": tip,
    }


def _inputs(extra_spans: list[dict] | None = None):
    spans = [
        # Out last game and Out all along this game, with a reason change:
        # never news.
        _span("g0", HOME, "star", "out", PREV_TIP - pd.Timedelta(hours=20), PREV_TIP),
        _span("g1", HOME, "star", "out", before_tip(1200), before_tip(300)),
        _span("g1", HOME, "star", "out", before_tip(300), TIP),
        # New this game: Questionable (p_out 0.4 x 10 = +4), then Out (+6).
        _span("g1", HOME, "p2", "questionable", before_tip(180), before_tip(90)),
        _span("g1", HOME, "p2", "out", before_tip(90), TIP),
        # A G League assignment is a roster mechanic, never news.
        _span("g1", HOME, "p3", "out", before_tip(100), TIP, reason="g_league"),
        *(extra_spans or []),
    ]
    schedule = pd.DataFrame(
        [
            {"game_id": "g0", "team_id": HOME, "prev_game_id": None},
            {"game_id": "g1", "team_id": HOME, "prev_game_id": "g0"},
            {"game_id": "g1", "team_id": AWAY, "prev_game_id": None},
        ]
    ).assign(game_date=TIP.tz_localize(None), season_year=2023, tipoff_utc=TIP)
    filings = pd.DataFrame(
        [
            {
                "game_id": "g0",
                "team_id": HOME,
                "valid_from": PREV_TIP - pd.Timedelta(hours=24),
                "valid_to": PREV_TIP,
                "submitted": True,
            },
            {
                "game_id": "g1",
                "team_id": HOME,
                "valid_from": before_tip(1440),
                "valid_to": TIP,
                "submitted": True,
            },
        ]
    )
    estimates = pd.DataFrame(
        [
            {"game_id": "g0", "player_id": "star", "FORM_PTS": 20.0},
            {"game_id": "g1", "player_id": "star", "FORM_PTS": 20.0},
            {"game_id": "g1", "player_id": "p2", "FORM_PTS": 10.0},
            {"game_id": "g1", "player_id": "p3", "FORM_PTS": 8.0},
            {"game_id": "g1", "player_id": "p4", "FORM_PTS": 30.0},
        ]
    ).assign(P_PLAY_QUESTIONABLE=0.6, P_PLAY_PROBABLE=0.95)
    return pd.DataFrame(spans), filings, schedule, estimates


def _timeline(extra_spans=None):
    return build_news_timeline(*_inputs(extra_spans))


def _home_features(timeline, minutes_before_tip: float) -> pd.Series:
    ns = np.array([before_tip(minutes_before_tip).value])
    return timeline.features_at(["g1"], [HOME], ns).iloc[0]


# --------------------------------------------------------------------------------
# G1 -- injury news
# --------------------------------------------------------------------------------


def test_news_windows_baseline_carry_and_material_news():
    got = _home_features(_timeline(), 60)
    assert got["NEWS_EXP_PTS_W60"] == pytest.approx(6.0)
    assert got["NEWS_EXP_PTS_W240"] == pytest.approx(10.0)
    # The star was Out last game and is Out now: only p2 is news since then.
    assert got["NEWS_EXP_PTS_SINCE_PREV_GAME"] == pytest.approx(10.0)
    assert got["MAX_PLAYER_NEWS_EXP_PTS_W240"] == pytest.approx(6.0)
    assert got["MIN_SINCE_MATERIAL_NEWS"] == pytest.approx(30.0)
    assert got["HAS_MATERIAL_NEWS"] == 1.0


def test_report_exactly_at_the_snapshot_is_not_visible():
    timeline = _timeline()
    got = _home_features(timeline, 90)
    assert got["NEWS_EXP_PTS_W60"] == pytest.approx(0.0)
    assert got["MIN_SINCE_MATERIAL_NEWS"] == pytest.approx(90.0)
    latest = timeline.latest_material_news(["g1"], [before_tip(90).value], 480)
    assert latest[0] == before_tip(180).value


def test_uncovered_team_is_nan_with_zero_flags():
    ns = np.array([before_tip(60).value])
    got = _timeline().features_at(["g1"], [AWAY], ns).iloc[0]
    assert np.isnan(got["NEWS_EXP_PTS_W60"])
    assert np.isnan(got["NEWS_EXP_PTS_SINCE_PREV_GAME"])
    assert got["HAS_MATERIAL_NEWS"] == 0.0
    assert got["COVERED"] == 0.0


def test_later_reports_do_not_change_features_at_the_snapshot():
    later = [
        _span("g1", HOME, "p4", "out", before_tip(59), TIP),
        _span("g1", HOME, "p2", "available", before_tip(30), TIP),
    ]
    assert _home_features(_timeline(later), 60).equals(_home_features(_timeline(), 60))


def test_news_column_names_pass_the_gate_and_are_not_odds_shaped():
    names = [news_column("NEWS_EXP_PTS_W60", side) for side in ("HOME", "AWAY")]
    assert all(is_kept_column(name) for name in names)
    assert find_unprefixed_odds_columns(names) == []


# --------------------------------------------------------------------------------
# G2 / G3 -- ticks
# --------------------------------------------------------------------------------


def _tick(market, book, minutes, left_line=None, left_price=-110.0, right_price=-110.0):
    right_line = None
    if left_line is not None:
        right_line = -left_line if market == MARKET_SPREAD else left_line
    return {
        "game_id": "g1",
        "season_year": 2023,
        "market": market,
        "book": book,
        "minutes_before_tip": float(minutes),
        "left_line": left_line,
        "right_line": right_line,
        "left_price": left_price,
        "right_price": right_price,
    }


def _ticks(extra=()):
    rows = [
        # Spread level = expected home margin.
        _tick(MARKET_SPREAD, "bet365", 600, 2.0),
        _tick(MARKET_SPREAD, "bet365", 150, 2.5),  # exactly the pre-news anchor
        _tick(MARKET_SPREAD, "bet365", 140, 3.0),  # after it
        _tick(MARKET_SPREAD, "bet365", 70, 4.0),
        _tick(MARKET_SPREAD, "b2", 600, 2.0),
        _tick(MARKET_SPREAD, "b2", 100, 3.5),
        _tick(MARKET_TOTALS, "bet365", 600, 220.0),
        _tick(MARKET_TOTALS, "bet365", 100, 218.0),
        _tick(MARKET_TOTALS, "b2", 600, 220.5),
        _tick(MARKET_MONEYLINE, "bet365", 600, None, 150.0, -170.0),
        _tick(MARKET_MONEYLINE, "bet365", 100, None, 170.0, -200.0),
        *extra,
    ]
    return pd.DataFrame(rows)


def _rows(minutes=60):
    snapshot = before_tip(minutes)
    return pd.DataFrame(
        {
            "GAME_ID": ["g1"],
            "GAME_DATE": [TIP.tz_localize(None)],
            "SEASON_YEAR": [2023],
            "TIME_TO_MATCH_MIN": [minutes],
            "TIPOFF_UTC": [TIP],
            "SNAPSHOT_TS_UTC": [snapshot],
            "HOME_TEAM_ID": [HOME],
            # Both sides read the covered team, so the news is known; its
            # spread sign is then 0 and the residual equals the raw move.
            "AWAY_TEAM_ID": [HOME],
        }
    )


def _reaction(ticks, timeline=None):
    reader = LevelReader(tick_levels(ticks))
    return news_reaction_features(
        reader, _rows(), timeline or _timeline(), anchor="bet365"
    ).iloc[0]


def test_pre_news_anchor_reads_ticks_at_or_before_news_minus_60():
    got = _reaction(_ticks())
    # Latest material news at 90 min before tip -> anchor at 150 min.
    assert got["ODDS_SNAP_NEWS_HAS_RECENT"] == 1.0
    assert got["ODDS_SNAP_NEWS_SPR_BET365_MOVE_SINCE_PRE_NEWS"] == pytest.approx(1.5)
    assert got["ODDS_SNAP_NEWS_SPR_N_BOOKS_MOVED_SINCE_PRE_NEWS"] == 2.0
    # No beta history yet: the residual is the whole move.
    assert got["ODDS_SNAP_NEWS_SPR_BET365_REACTION_RESIDUAL"] == pytest.approx(1.5)


def test_no_recent_material_news_gives_zero_anchored_family_not_nan():
    # At 600 min before tip no report has changed yet, so there is no anchor.
    reader = LevelReader(tick_levels(_ticks()))
    got = news_reaction_features(
        reader, _rows(600), _timeline(), anchor="bet365"
    ).iloc[0]
    assert got["ODDS_SNAP_NEWS_HAS_RECENT"] == 0.0
    anchored = [
        name
        for name in got.index
        if name.endswith(("SINCE_PRE_NEWS", "REACTION_RESIDUAL"))
    ]
    assert len(anchored) == 3 * 4
    assert (got[anchored] == 0.0).all()


def _side_move_ticks(*, spread_move: float, with_moneyline: bool):
    rows = [
        _tick(MARKET_TOTALS, "bet365", 120, 220.0),
        _tick(MARKET_TOTALS, "bet365", 90, 222.0),
        _tick(MARKET_SPREAD, "bet365", 120, 3.0),
        _tick(MARKET_SPREAD, "bet365", 90, 3.0 + spread_move),
    ]
    if with_moneyline:
        rows += [
            _tick(MARKET_MONEYLINE, "bet365", 120, None, 150.0, -170.0),
            _tick(MARKET_MONEYLINE, "bet365", 90, None, 150.0, -170.0),
        ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("spread_move", "with_moneyline", "expected"),
    [
        (0.0, True, 2.0),  # both side markets seen quiet
        (0.0, False, np.nan),  # moneyline unobserved: quietness unknown
        (1.0, False, 0.0),  # an observed side move decides it anyway
        (1.0, True, 0.0),
    ],
)
def test_total_move_without_side_move_does_not_treat_missing_as_quiet(
    spread_move, with_moneyline, expected
):
    ticks = _side_move_ticks(spread_move=spread_move, with_moneyline=with_moneyline)
    got = cross_market_features(
        LevelReader(tick_levels(ticks)), _rows(), anchor="bet365"
    ).iloc[0]["ODDS_SNAP_XMKT_TOTAL_MOVE_WITHOUT_SIDE_MOVE_60"]
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_league_mean_weights_each_game_once_whatever_its_snapshot_count():
    day1, day2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    dates = pd.Series([day1, day1, day1, day1, day2])
    games = np.array(["a", "a", "a", "b", "c"])
    values = np.array([1.0, 1.0, 1.0, 3.0, 0.0])
    got = _league_mean_before(dates, values, games)
    assert np.isnan(got[:4]).all()  # strictly earlier dates only
    # Per game: (1 + 3) / 2, not the per-row (1 + 1 + 1 + 3) / 4.
    assert got[4] == pytest.approx(2.0)


def test_ticks_after_the_snapshot_do_not_change_reaction_or_cross_market():
    late = [
        _tick(MARKET_SPREAD, "bet365", 59, 9.0),
        _tick(MARKET_TOTALS, "bet365", 30, 230.0),
        _tick(MARKET_MONEYLINE, "bet365", 10, None, 400.0, -500.0),
    ]
    assert _reaction(_ticks(late)).equals(_reaction(_ticks()))
    before = cross_market_features(
        LevelReader(tick_levels(_ticks())), _rows(), anchor="bet365"
    )
    after = cross_market_features(
        LevelReader(tick_levels(_ticks(late))), _rows(), anchor="bet365"
    )
    pd.testing.assert_frame_equal(before, after)


def test_orientation_of_margin_and_signed_news():
    assert _moneyline_margin(np.array([0.65]))[0] > 0
    assert np.all(np.diff(_moneyline_margin(np.array([0.3, 0.5, 0.7]))) > 0)
    # The away team losing a scorer makes the home side relatively stronger.
    assert _signed_news(np.array([0.0]), np.array([5.0]), MARKET_SPREAD)[0] > 0
    assert _signed_news(np.array([0.0]), np.array([5.0]), MARKET_MONEYLINE)[0] > 0
    assert _signed_news(np.array([0.0]), np.array([5.0]), MARKET_TOTALS)[0] < 0


def test_cross_market_gap_and_names():
    got = cross_market_features(
        LevelReader(tick_levels(_ticks())), _rows(), anchor="bet365"
    ).iloc[0]
    prefix = "ODDS_SNAP_XMKT_BET365_ML_MARGIN_MINUS_SPREAD"
    fair_home = (200 / 300) / (200 / 300 + 100 / 270)
    assert got[prefix] == pytest.approx(
        _moneyline_margin(np.array([fair_home]))[0] - 4.0
    )
    assert got["ODDS_SNAP_XMKT_ABS_SPREAD_MOVE_60"] >= 0
    assert all(is_kept_column(name) for name in got.index)
    assert find_unprefixed_odds_columns(got.index) == []


# --------------------------------------------------------------------------------
# Walk-forward estimators
# --------------------------------------------------------------------------------


def _walk_forward_frame():
    games = [
        ("a", 2021, "2022-01-01"),
        ("b", 2022, "2023-01-01"),
        ("c", 2023, "2024-01-01"),
        ("d", 2023, "2024-01-02"),
        ("e", 2023, "2024-01-03"),
    ]
    rows = []
    for game, season, day in games:
        tip = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=20)
        for minutes, x in ((120, 1.0), (60, 2.0)):
            rows.append(
                (
                    game,
                    season,
                    tip.value,
                    (tip - pd.Timedelta(minutes=minutes)).value,
                    x,
                )
            )
    frame = pd.DataFrame(rows, columns=["game", "season", "tip", "snap", "x"])
    return frame


def _beta(frame, y):
    return walk_forward_least_squares(
        frame[["x"]].to_numpy(float),
        np.asarray(y, dtype=float),
        game_ids=frame["game"],
        seasons=frame["season"].to_numpy(),
        tip_ns=frame["tip"].to_numpy(),
        snapshot_ns=frame["snap"].to_numpy(),
        label_mask=np.ones(len(frame), dtype=bool),
        predict_mask=np.ones(len(frame), dtype=bool),
        alpha=0.0,
    ).coefficients[:, 0]


def test_beta_ignores_games_not_yet_tipped_and_seasons_before_the_previous():
    frame = _walk_forward_frame()
    y = frame["x"] * 0.03
    base = _beta(frame, y)
    row_d = frame.index[frame["game"].eq("d")][0]
    assert base[row_d] == pytest.approx(0.03)

    later = y.copy()
    later[frame["game"].isin(["d", "e"])] = 9.0
    assert _beta(frame, later)[row_d] == pytest.approx(base[row_d])

    old = y.copy()
    old[frame["game"].eq("a")] = 9.0
    assert _beta(frame, old)[row_d] == pytest.approx(base[row_d])

    previous_season = y.copy()
    previous_season[frame["game"].eq("b")] = 9.0
    assert _beta(frame, previous_season)[row_d] != pytest.approx(base[row_d])


def _ridge_game(market, game, day, season, current, close_delta):
    tip = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=20)
    short = ridge.MARKET_DESIGNS[market].short
    line = ridge.current_line_column(market, "bet365")
    rows = [
        {
            "GAME_ID": game,
            "SEASON_YEAR": season,
            "TIME_TO_MATCH_MIN": horizon,
            "TIPOFF_UTC": tip,
            "SNAPSHOT_TS_UTC": tip - pd.Timedelta(minutes=horizon),
            line: current if horizon else current + close_delta,
            f"ODDS_SNAP_{short}_BET365_MOVE_FROM_OPEN": 1.0,
            f"ODDS_SNAP_{short}_BET365_LINE_AGE_MINUTES": 10.0,
        }
        for horizon in (360, 0)
    ]
    return rows, {"GAME_ID": game, "CLOSING_LINE": current + close_delta}


RIDGE_CASES = [
    (MARKET_TOTALS, 220.0, 2.0),
    (MARKET_SPREAD, 3.0, 1.0),
    (MARKET_MONEYLINE, 0.6, 0.02),
]


@pytest.mark.parametrize(("market", "current", "delta"), RIDGE_CASES)
def test_ridge_uses_only_completed_prior_games(monkeypatch, market, current, delta):
    monkeypatch.setattr(ridge, "MIN_TRAIN_GAMES", 1)
    games = [
        _ridge_game(market, name, day, 2023, current, delta)
        for name, day in (("a", "2024-01-01"), ("b", "2024-01-02"), ("c", "2024-01-03"))
    ]
    frame = pd.DataFrame([row for rows, _ in games for row in rows])
    closes = pd.DataFrame([close for _, close in games])
    column = ridge.EXPECTED_MOVE_COLUMNS[market]
    actual = ridge.add_historical_ridge_movement(
        frame, closes, anchor="bet365", market=market
    )
    assert actual.loc[0, column] == 0.0
    assert actual.loc[1::2, column].eq(0.0).all()
    assert actual.loc[2, column] > 0.0

    changed = closes.copy()
    changed.loc[changed.GAME_ID.eq("b"), "CLOSING_LINE"] = current + 3 * delta
    moved = ridge.add_historical_ridge_movement(
        frame, changed, anchor="bet365", market=market
    )
    assert moved.loc[2, column] == pytest.approx(actual.loc[2, column])
    assert moved.loc[4, column] > actual.loc[4, column]


@pytest.mark.parametrize(("market", "current", "delta"), RIDGE_CASES)
def test_ridge_uses_previous_and_current_season_only(
    monkeypatch, market, current, delta
):
    monkeypatch.setattr(ridge, "MIN_TRAIN_GAMES", 1)
    games = [
        _ridge_game(market, "old", "2022-01-01", 2021, current, delta),
        _ridge_game(market, "previous", "2023-01-01", 2022, current, delta),
        _ridge_game(market, "current", "2024-01-01", 2023, current, delta),
    ]
    frame = pd.DataFrame([row for rows, _ in games for row in rows])
    closes = pd.DataFrame([close for _, close in games])
    column = ridge.EXPECTED_MOVE_COLUMNS[market]
    first = ridge.add_historical_ridge_movement(
        frame, closes, anchor="bet365", market=market
    )
    closes.loc[closes.GAME_ID.eq("old"), "CLOSING_LINE"] = current + 3 * delta
    changed = ridge.add_historical_ridge_movement(
        frame, closes, anchor="bet365", market=market
    )
    assert changed.loc[4, column] == pytest.approx(first.loc[4, column])


def test_ridge_output_names_pass_the_gate():
    assert all(is_kept_column(c) for c in ridge.EXPECTED_MOVE_COLUMNS.values())
    assert find_unprefixed_odds_columns(ridge.EXPECTED_MOVE_COLUMNS.values()) == []


def test_spread_design_reads_cross_market_inputs():
    columns = [
        "ODDS_SNAP_SPR_BET365_MOVE_FROM_OPEN",
        "ODDS_SNAP_SPR_CONSENSUS_N_BOOKS_QUOTING",
        "ODDS_SNAP_XMKT_BET365_ML_MARGIN_MINUS_SPREAD",
        "ODDS_SNAP_TOT_BET365_MOVE_LAST_60",
        "ODDS_SNAP_TOT_BET365_MOVE_FROM_OPEN",
    ]
    assert ridge.ridge_input_columns(columns, MARKET_SPREAD, "bet365") == columns[:4]
    assert ridge.ridge_input_columns(columns, MARKET_TOTALS, "bet365") == columns[3:]


def test_season_parameters_are_plain_ints():
    from nba_ou.postgre_db.injury_report_aiven import fetch

    params = fetch._season_params([2024, np.int64(2023)])
    assert params["season_years"] == [2024, 2023]
    assert all(type(season) is int for season in params["season_years"])


def test_spread_design_adds_nonlinear_gap_terms_only_for_the_spread():
    frame = pd.DataFrame(
        {
            "TIME_TO_MATCH_MIN": [60, 60],
            "ODDS_SNAP_SPR_BET365_MOVE_FROM_OPEN": [0.0, 0.0],
            "ODDS_SNAP_SPR_BET365_LINE_AGE_MINUTES": [0.0, 0.0],
            "ODDS_SNAP_SPR_BET365_LEVEL": [7.0, -7.0],
            "ODDS_SNAP_XMKT_BET365_ML_MARGIN_MINUS_SPREAD": [-2.0, 0.5],
        }
    )
    spread = ridge._design_matrix(frame, anchor="bet365", market=MARKET_SPREAD)
    gap_terms = spread[:, 1 + 9 + 4 : 1 + 9 + 4 + 4]
    np.testing.assert_allclose(gap_terms[0], [2.0, -1.0, -2.0, 1.0])
    np.testing.assert_allclose(gap_terms[1], [0.5, 0.5, 0.5, 1.0])
    totals = frame.rename(columns=lambda c: c.replace("_SPR_", "_TOT_"))
    assert ridge._design_matrix(totals, anchor="bet365").shape[1] == 1 + 9 + 3
