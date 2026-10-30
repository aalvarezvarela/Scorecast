"""Intermediate openers use day-page quotes, normalized exactly as training."""

import pandas as pd
import pytest
from nba_ou.create_training_data.select_intermediate_columns import (
    select_intermediate_training_columns,
)
from nba_ou.data_processing.odds.scheduled_opener import (
    OPENER_COLUMNS,
    merge_scheduled_opener,
)
from nba_ou.data_processing.team.merge_game_df_with_odds_by_game_id import (
    merge_total_spread_moneyline_by_game_id,
)
from nba_ou.postgre_db.odds.merge_odds_data import merge_yahoo_sportsbook_odds

GAME = "0022600145"


def day_quote(game=GAME, line=238.5, over=-110.0, under=-110.0):
    return dict(
        game_id=game,
        total_consensus_opener_line_over=line,
        total_consensus_opener_line_under=line,
        total_consensus_opener_price_over=over,
        total_consensus_opener_price_under=under,
    )


def history():
    return pd.DataFrame(
        [
            {
                **day_quote("earlier", 220.5, 1.91, 1.91),
                "total_bet365_line_over": 221.5,
                "spread_bet365_line_home": -3.5,
                "spread_bet365_line_away": 3.5,
                "ml_bet365_price_home": 1.5,
                "ml_bet365_price_away": 2.6,
                "total_consensus_pct_over": 65.0,
            }
        ]
    )


@pytest.mark.parametrize(
    "line,over,under,normalized",
    [
        (238.5, -110, -110, 238.5),
        (229.5, -115, -105, 230.0),
        (217.5, -115, -105, 218.0),
    ],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_missing_game_gets_the_training_opener_and_reaches_model_feature(
    line, over, under, normalized, normalize
):
    stored = history()
    before = stored.copy(deep=True)
    day = pd.DataFrame([day_quote(line=line, over=over, under=under)])
    training = merge_yahoo_sportsbook_odds(
        pd.DataFrame(), day, normalize_total_lines=normalize
    )
    merged = merge_scheduled_opener(
        stored, day, [GAME], normalize_total_lines=normalize
    )
    actual = merged.set_index("game_id").loc[GAME]
    pd.testing.assert_series_equal(
        actual[list(OPENER_COLUMNS)],
        training.set_index("game_id").loc[GAME, list(OPENER_COLUMNS)],
        check_dtype=False,
    )
    feature = merge_total_spread_moneyline_by_game_id(
        merged,
        pd.DataFrame({"GAME_ID": [GAME], "HOME": [True]}),
        book="bet365",
        total_line_book="bet365",
        debug=False,
    )
    assert feature.ODDS_TOTAL_LINE_consensus_opener.iloc[0] == (
        normalized if normalize else line
    )
    pd.testing.assert_frame_equal(merged.iloc[:1], before)
    pd.testing.assert_frame_equal(stored, before)


def test_current_wagers_closes_scores_and_other_games_are_not_imported():
    stored = history()
    quote = {
        **day_quote(),
        "total_bet365_line_over": 999.0,
        "total_consensus_pct_over": 99.0,
        "total_pct_bets_over": 98.0,
        "total_pct_money_over": 97.0,
        "home_points": 200,
        "total_points": 400,
    }
    day = pd.DataFrame([quote, day_quote("unrelated")])
    merged = merge_scheduled_opener(stored, day, [GAME])
    row = merged.set_index("game_id").loc[GAME]
    assert set(merged.game_id) == {"earlier", GAME}
    assert pd.isna(row.total_bet365_line_over)
    assert pd.isna(row.total_consensus_pct_over)
    assert "total_pct_bets_over" not in merged
    assert "total_pct_money_over" not in merged
    assert "home_points" not in merged and "total_points" not in merged
    pd.testing.assert_frame_equal(merged.iloc[:1], stored)


@pytest.mark.parametrize("empty", [False, True])
def test_unavailable_day_opener_does_not_fall_back_to_a_later_stored_value(empty):
    stored = pd.concat([history(), pd.DataFrame([day_quote()])], ignore_index=True)
    day = pd.DataFrame() if empty else pd.DataFrame([day_quote(line=None)])
    out = merge_scheduled_opener(stored, day, [GAME])
    assert pd.isna(out.set_index("game_id").loc[GAME, OPENER_COLUMNS[0]])


def test_day_quote_overrides_only_scheduled_openers():
    stored = pd.concat(
        [history(), pd.DataFrame([day_quote(line=250)])], ignore_index=True
    )
    day = pd.DataFrame([day_quote(), day_quote("earlier", 999)])
    out = merge_scheduled_opener(stored, day, [GAME])
    assert len(out) == 2
    assert out.set_index("game_id").loc[GAME, OPENER_COLUMNS[0]] == 238.5
    pd.testing.assert_frame_equal(out.iloc[:1], stored.iloc[:1])


def test_ambiguous_day_quotes_fail_instead_of_multiplying_feature_rows():
    day = pd.DataFrame([day_quote(), day_quote(line=240)])
    with pytest.raises(ValueError, match="More than one day-page opener"):
        merge_scheduled_opener(history(), day, [GAME])


def test_intermediate_gate_keeps_historical_wagers_but_drops_current_ones():
    current = [
        "ODDS_total_pct_bets_over",
        "ODDS_total_pct_money_over",
        "ODDS_total_consensus_pct_over",
        "ODDS_spread_consensus_pct_home",
    ]
    historical = [f"{column}_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME" for column in current]
    frame = pd.DataFrame({c: [42.0] for c in current + historical})
    out = select_intermediate_training_columns(frame)
    assert set(out) == set(historical)
    pd.testing.assert_frame_equal(out, frame[historical])


def test_base_pipeline_supplies_normalized_day_opener_before_team_features(monkeypatch):
    from nba_ou.create_training_data import create_base_game_features as base

    stored = history()
    scheduled = pd.DataFrame({"GAME_ID": [GAME]})
    day = pd.DataFrame([day_quote(line=229.5, over=-115, under=-105)])
    monkeypatch.setattr(
        base,
        "load_all_nba_data_from_db",
        lambda **kw: (pd.DataFrame({"GAME_DATE": ["2026-10-28"]}), pd.DataFrame()),
    )
    monkeypatch.setattr(
        base, "filter_by_seasons_with_extra_game_ids", lambda df, **kw: df
    )
    monkeypatch.setattr(
        base, "load_and_merge_odds_yahoo_sportsbookreview", lambda **kw: stored
    )

    class ReachedTeamFeatures(Exception):
        pass

    def team_features(frame, odds, **kwargs):
        assert kwargs["scheduled_games"] is scheduled
        assert odds.set_index("game_id").loc[GAME, OPENER_COLUMNS[0]] == 230.0
        pd.testing.assert_frame_equal(odds.iloc[:1], stored)
        raise ReachedTeamFeatures

    monkeypatch.setattr(base, "process_team_statistics_for_training", team_features)
    with pytest.raises(ReachedTeamFeatures):
        base.create_base_game_features(
            recent_limit_to_include="2026-10-28",
            season_start_date="2026-10-01",
            scheduled_games=scheduled,
            scheduled_odds_sportsbook=day,
            include_all_star=False,
            include_roster_continuity=False,
            verbose=False,
        )
