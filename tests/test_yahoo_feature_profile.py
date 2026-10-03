"""Yahoo inputs must be compact, observed as-is, optional and strictly past."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.config.odds_columns import apply_odds_prefix, total_line_col
from nba_ou.config.yahoo_features import (
    YAHOO_FEATURE_COLUMNS,
    YAHOO_HISTORY_COLUMNS,
    YAHOO_RAW_COLUMNS,
    YAHOO_RAW_SOURCES,
    compact_yahoo_columns,
    is_yahoo_percentage_column,
)
from nba_ou.create_training_data.select_intermediate_columns import (
    select_intermediate_training_columns,
)
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_betting_stats_differences,
)
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from nba_ou.data_processing.odds.yahoo_features import select_yahoo_features
from nba_ou.data_processing.team import rolling
from nba_ou.data_processing.team.merge_game_df_with_odds_by_game_id import (
    merge_odds_percentages_and_prices_by_game_id,
    merge_remaining_odds_by_game_id,
)


@pytest.fixture
def team_history(monkeypatch):
    # Isolate Yahoo and consensus from box-score families. Run the real merge,
    # mean and slope helpers, including the narrow-frame reconstruction.
    for name in (
        "COLS_TO_AVERAGE",
        "COLS_TO_AVERAGE_ODDS",
        "COLS_FOR_WEIGHTED_STATS",
        "COLS_FOR_SEASON_STD",
        "COLS_FOR_SHORT_WINDOWS",
    ):
        monkeypatch.setattr(rolling, name, [])
    monkeypatch.setattr(rolling, "COLS_TO_AVERAGE", ["PTS", "PTS_PER_40"])
    teams = pd.DataFrame(
        [
            {
                "GAME_ID": str(game),
                "TEAM_ID": team,
                "HOME": (game + team) % 2 == 0,
                "GAME_DATE": pd.Timestamp("2025-11-01") + pd.Timedelta(days=game),
                "SEASON_YEAR": 2025,
                "SEASON_TYPE": "Regular Season",
                "PTS": 100.0 + game,
                "PTS_PER_40": 100.0 + game,
            }
            for game in range(8)
            for team in (1, 2)
        ]
    )
    odds = pd.DataFrame({"game_id": [str(g) for g in range(8)]})
    for i, source in enumerate(YAHOO_RAW_SOURCES):
        odds[source] = np.arange(8, dtype=float) * 3 + i
    odds["total_consensus_pct_over"] = np.arange(8, dtype=float) + 30
    return teams, odds


def _build(teams, odds):
    merged = merge_odds_percentages_and_prices_by_game_id(odds, teams)
    history = rolling.compute_all_rolling_statistics(merged)
    home = history[history.HOME].drop(columns="HOME")
    away = history[~history.HOME].drop(columns="HOME")
    game = home.merge(away, on="GAME_ID", suffixes=("_TEAM_HOME", "_TEAM_AWAY"))
    game = merge_remaining_odds_by_game_id(odds, game)
    game = add_betting_stats_differences(game)
    return history, select_yahoo_features(apply_odds_prefix(game))


def test_build_keeps_exactly_36_and_preserves_raw_observations(team_history):
    teams, odds = team_history
    history, result = _build(teams, odds)
    yahoo = [c for c in result if is_yahoo_percentage_column(c)]
    assert len(yahoo) == 36
    assert set(yahoo) == set(YAHOO_FEATURE_COLUMNS)
    for source in YAHOO_RAW_SOURCES:
        actual = result.set_index("GAME_ID")[f"ODDS_{source}"].sort_index()
        expected = odds.set_index("game_id")[source].sort_index()
        np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    derived = [c for c in history if is_yahoo_percentage_column(c) and "BEFORE" in c]
    assert len(derived) == 12  # two statistics per source, before home/away merge
    assert "total_consensus_pct_over_SEASON_BEFORE_AVG" in history
    assert "ODDS_total_consensus_pct_over_LAST_ALL_5_MATCHES_DIFF_BEFORE" in result


def test_history_excludes_the_current_game_and_follows_the_team(team_history):
    teams, odds = team_history
    _, original = _build(teams, odds)
    changed = odds.copy()
    changed.loc[changed.game_id == "6", list(YAHOO_RAW_SOURCES)] = 999.0
    _, updated = _build(teams, changed)
    a = original.set_index("GAME_ID")
    b = updated.set_index("GAME_ID")
    pd.testing.assert_series_equal(
        a.loc["6", list(YAHOO_HISTORY_COLUMNS)],
        b.loc["6", list(YAHOO_HISTORY_COLUMNS)],
    )
    assert b.loc["6", "ODDS_total_pct_bets_over"] == 999.0
    assert (
        a.loc["6", "ODDS_total_pct_bets_over_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"]
        == 9.0
    )
    assert (
        a.loc["6", "ODDS_total_pct_bets_over_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_HOME"]
        == 3.0
    )
    # Home team 2 played home in games 0/2/4 and away in 1/3/5. Its spread
    # history must use the corresponding percentage, not always the home side.
    values = [
        odds.loc[g, "spread_pct_bets_home" if g % 2 == 0 else "spread_pct_bets_away"]
        for g in range(1, 6)
    ]
    assert a.loc[
        "6", "ODDS_spread_pct_bets_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"
    ] == np.mean(values)


def test_feed_outage_keeps_the_schema_and_leaves_raw_values_missing(team_history):
    teams, odds = team_history
    _, result = _build(teams, odds.drop(columns=list(YAHOO_RAW_SOURCES)))
    assert set(compact_yahoo_columns(result)) == set(YAHOO_FEATURE_COLUMNS)
    assert result[list(YAHOO_RAW_COLUMNS)].isna().all().all()


def test_intermediate_gate_never_admits_current_game_percentages(team_history):
    teams, odds = team_history
    _, closing = _build(teams, odds)
    intermediate = select_intermediate_training_columns(closing)
    yahoo = {c for c in intermediate if is_yahoo_percentage_column(c)}
    assert yahoo == set(YAHOO_HISTORY_COLUMNS)
    assert not set(YAHOO_RAW_COLUMNS).intersection(intermediate)


@pytest.mark.parametrize("include_raw", [True, False])
def test_compact_columns_survive_every_cleaning_step_and_optional_na_limits(
    include_raw,
):
    df = pd.DataFrame(
        {
            total_line_col(): [210.0, 220.0, 215.0, 225.0],
            "TOTAL_POINTS": [211.0, 219.0, 213.0, 230.0],
            "SEASON_YEAR": [2024, 2024, 2025, 2025],
        }
    )
    df = select_yahoo_features(df, include_raw=include_raw)
    expected = YAHOO_FEATURE_COLUMNS if include_raw else YAHOO_HISTORY_COLUMNS
    # Duplicates, abs matches, perfect negative correlations, constants,
    # all-null columns and season-gated availability must ALL survive.
    df[expected[0]] = [10.0, 20.0, 30.0, 40.0]
    df[expected[1]] = df[expected[0]]
    df[expected[2]] = -df[expected[0]]
    df[expected[3]] = 100 - df[expected[0]]
    df[expected[4]] = 0.0
    df[expected[5]] = [np.nan, np.nan, 30.0, 40.0]
    before = df[list(expected)].copy()
    cleaned = clean_dataframe_for_training(
        df,
        nan_threshold=0,
        corr_threshold=0.8,
        max_seasonal_nan_spread=10,
        max_na_per_row=0,
        strict_mode=0,
        keep_columns=[total_line_col(), "TOTAL_POINTS", "SEASON_YEAR"],
        verbose=0,
    )
    assert len(cleaned) == len(df)
    pd.testing.assert_frame_equal(cleaned[list(expected)], before)


@pytest.mark.parametrize("include_raw", [True, False])
def test_an_explicit_yahoo_exclusion_removes_the_whole_compact_block(include_raw):
    """Protection is implicit; a "without Yahoo" ablation must still work."""
    df = pd.DataFrame(
        {
            total_line_col(): [210.0, 220.0, 215.0, 225.0],
            "TOTAL_POINTS": [211.0, 219.0, 213.0, 230.0],
        }
    )
    df = select_yahoo_features(df, include_raw=include_raw)
    for i, column in enumerate(YAHOO_FEATURE_COLUMNS):
        if column in df:
            df[column] = np.arange(4, dtype=float) + i

    cleaned = clean_dataframe_for_training(
        df,
        exclude_cols_containing=["pct_bets", "pct_money"],
        max_na_per_row=0,
        verbose=0,
    )

    assert not any(is_yahoo_percentage_column(c) for c in cleaned)
    assert len(cleaned) == len(df)


def test_archived_full_schema_keeps_its_existing_cleaning_policy():
    df = select_yahoo_features(
        pd.DataFrame(
            {
                total_line_col(): [210.0, 220.0],
                "TOTAL_POINTS": [215.0, 225.0],
            }
        )
    )
    df["ODDS_total_pct_bets_over_SEASON_BEFORE_AVG_TEAM_HOME"] = np.nan
    assert compact_yahoo_columns(df.columns) == ()
    cleaned = clean_dataframe_for_training(
        df,
        exclude_cols_containing=["pct_bets", "pct_money"],
        verbose=0,
    )
    assert not any(is_yahoo_percentage_column(c) for c in cleaned)


def test_empty_yahoo_feed_does_not_break_scheduled_odds_validation():
    from nba_ou.data_processing.odds.merge_scheduled_odds import (
        merge_and_validate_scheduled_odds,
    )

    sportsbook = pd.DataFrame(
        {
            "game_id": ["0022500002"],
            "game_date": ["2025-11-02"],
            "season_year": [2025],
            "total_betmgm_line_over": [220.5],
        }
    )
    historical = sportsbook.assign(game_id="0022500001", game_date="2025-11-01")
    historical = historical.assign(**{c: 50.0 for c in YAHOO_RAW_SOURCES})
    result = merge_and_validate_scheduled_odds(
        historical,
        pd.DataFrame(),
        sportsbook,
        strict_mode=0,
        normalize_total_lines=False,
        normalize_spread_lines=False,
    )
    today = result[result.game_id == "0022500002"]
    assert len(today) == 1
    assert today[list(YAHOO_RAW_SOURCES)].isna().all().all()
    assert today.total_betmgm_line_over.iloc[0] == 220.5


@pytest.fixture
def compact_training_config(tmp_path):
    from training_pipeline.config import CleaningConfig, DataConfig, ExperimentConfig

    n = 40
    frame = pd.DataFrame(
        {
            "GAME_ID": [f"00225000{i:02d}" for i in range(n)],
            "GAME_DATE": pd.date_range("2025-11-01", periods=n),
            "SEASON_YEAR": 2025,
            "SEASON_TYPE": "Regular Season",
            total_line_col(): np.linspace(205, 230, n),
            "TOTAL_POINTS": np.linspace(210, 240, n),
        }
    )
    frame = select_yahoo_features(frame)
    for c in YAHOO_RAW_COLUMNS:
        frame[c] = np.linspace(30, 70, n)
    path = tmp_path / "training_data_2_5_test.csv"
    frame.to_csv(path, index=False)
    return ExperimentConfig(
        experiment_name="compact_yahoo",
        prediction_strategy="line_error_regressor",
        data=DataConfig(csv_path=str(path)),
        cleaning=CleaningConfig(verbose=0, max_na_per_row=0),
    ), frame


def test_training_matrix_includes_every_selected_input(compact_training_config):
    from training_pipeline.data import prepare_dataset

    config, frame = compact_training_config
    prepared = prepare_dataset(config)
    assert len(prepared.X) == len(frame)
    assert set(YAHOO_FEATURE_COLUMNS).issubset(prepared.X.columns)
    np.testing.assert_allclose(
        prepared.X[list(YAHOO_RAW_COLUMNS)],
        frame[list(YAHOO_RAW_COLUMNS)],
    )


def test_training_without_yahoo_excludes_every_compact_column(compact_training_config):
    """The ablation configs exclude Yahoo by pattern; nothing may survive."""
    from training_pipeline.data import prepare_dataset

    config, frame = compact_training_config
    config.cleaning.exclude_cols_containing = ["pct_bets", "pct_money"]
    prepared = prepare_dataset(config)
    assert not any(is_yahoo_percentage_column(c) for c in prepared.X.columns)
    assert len(prepared.X) == len(frame)


def test_training_accepts_excluding_yahoo_columns_by_name(compact_training_config):
    from training_pipeline.data import prepare_dataset

    config, _ = compact_training_config
    config.exclude_cols.append(YAHOO_RAW_COLUMNS[0])
    prepared = prepare_dataset(config)
    assert YAHOO_RAW_COLUMNS[0] not in prepared.X.columns
    assert set(YAHOO_FEATURE_COLUMNS[1:]).issubset(prepared.X.columns)


def test_training_fails_when_cleaning_loses_a_yahoo_column(
    compact_training_config, monkeypatch
):
    import training_pipeline.data as data_module

    config, _ = compact_training_config
    real_clean = data_module.clean_for_training

    def losing_clean(df, *args, **kwargs):
        cleaned, report = real_clean(df, *args, **kwargs)
        return cleaned.drop(columns=[YAHOO_HISTORY_COLUMNS[0]]), report

    monkeypatch.setattr(data_module, "clean_for_training", losing_clean)
    with pytest.raises(ValueError, match="lost in cleaning"):
        data_module.prepare_dataset(config)


@pytest.mark.parametrize("drop_raw", [YAHOO_RAW_COLUMNS[:1], YAHOO_RAW_COLUMNS])
def test_closing_training_rejects_missing_raw_columns(compact_training_config, drop_raw):
    from training_pipeline.data import prepare_dataset

    config, frame = compact_training_config
    frame.drop(columns=list(drop_raw)).to_csv(config.data.csv_path, index=False)
    with pytest.raises(ValueError, match="raw percentages"):
        prepare_dataset(config)


def test_prediction_hands_observed_and_missing_raw_inputs_to_the_model(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from nba_ou.prediction import prediction
    from xgboost import XGBRegressor

    rng = np.random.default_rng(42)
    train = pd.DataFrame(rng.uniform(20, 80, (40, 36)), columns=YAHOO_FEATURE_COLUMNS)
    model = XGBRegressor(n_estimators=3, max_depth=2, n_jobs=1)
    model.fit(train, train[YAHOO_RAW_COLUMNS[0]] / 10)
    frame = train.iloc[:2].copy()
    frame.loc[1, list(YAHOO_FEATURE_COLUMNS)] = np.nan
    frame = frame.assign(
        **{
            "GAME_ID": ["0022600001", "0022600002"],
            "GAME_DATE": "2026-09-21",
            "GAME_TIME": "7:00 pm ET",
            "SEASON_TYPE": "Regular Season",
            total_line_col(): 225.0,
            "TEAM_NAME_TEAM_HOME": ["Home A", "Home B"],
            "TEAM_NAME_TEAM_AWAY": ["Away A", "Away B"],
            "MATCHUP_TEAM_HOME": ["A vs. C", "B vs. D"],
        }
    )
    seen = []
    original_predict = model.predict

    def record_inputs(X, **kwargs):
        seen.append(X.copy())
        return original_predict(X, **kwargs)

    monkeypatch.setattr(model, "predict", record_inputs)
    monkeypatch.setattr(prediction, "upload_predictions_to_postgre", lambda df: None)
    result = prediction.load_and_predict_model_for_nba_games(
        frame,
        model,
        model_name="compact_yahoo",
        model_type="line_error_t0000",
        model_version="test",
        required_features=list(YAHOO_FEATURE_COLUMNS),
        total_points_pick_line_col=total_line_col(),
        prediction_datetime=datetime(2026, 9, 21, 18, tzinfo=ZoneInfo("Europe/Madrid")),
        shap_top_n=2,
    )
    assert len(result) == 2
    assert list(seen[0].columns) == list(YAHOO_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(seen[0], frame[list(YAHOO_FEATURE_COLUMNS)])
