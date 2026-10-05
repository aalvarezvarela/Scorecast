"""The 2_6 layer: starter history per side, broadcast onto the parent's rows."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.create_training_data.schema_layers import v2_6
from nba_ou.create_training_data.schema_layers.base import (
    CLOSING_LINE,
    INTERMEDIATE_LINE,
    LayerContext,
)
from nba_ou.create_training_data.schema_layers.registry import apply_layers
from nba_ou.data_processing.injury_status.report_state import InjuryReportState
from nba_ou.data_processing.lineups.features import LINEUP_FEATURE_COLUMNS, RatingBook
from nba_ou.data_processing.players.starter_history import add_starter_history_features

HOME, AWAY = 1610612737, 1610612738


def _box(game_id, date, team, starters, bench="z"):
    rows = [
        {
            "GAME_ID": game_id,
            "TEAM_ID": team,
            "GAME_DATE": date,
            "PLAYER_ID": p,
            "START_POSITION": "F",
            "MIN": 36,
        }
        for p in starters
    ]
    rows.append(
        {
            "GAME_ID": game_id,
            "TEAM_ID": team,
            "GAME_DATE": date,
            "PLAYER_ID": f"{team}{bench}",
            "START_POSITION": "",
            "MIN": 60,
        }
    )
    return rows


@pytest.fixture
def players():
    rows = []
    lineups_home = ["abcde", "abcdf", "abcdf", "ghcdf"]
    lineups_away = ["pqrst", "pqrst", "pqrsu", "vwrsu"]
    for n, date in enumerate(["2025-10-21", "2025-10-23", "2025-10-25", "2025-10-27"]):
        game = f"00225000{n + 1:02d}"
        rows += _box(game, date, HOME, [f"h{c}" for c in lineups_home[n]])
        rows += _box(game, date, AWAY, [f"a{c}" for c in lineups_away[n]])
    return pd.DataFrame(rows)


@pytest.fixture
def closing():
    """As a CSV read leaves it: GAME_ID integers, team ids as text."""
    return pd.DataFrame(
        {
            "GAME_ID": [22500004, 22500003, 22500002],
            "GAME_DATE": ["2025-10-27", "2025-10-25", "2025-10-23"],
            "TEAM_ID_TEAM_HOME": [str(HOME)] * 3,
            "TEAM_ID_TEAM_AWAY": [str(AWAY)] * 3,
            "TOTAL_POINTS": [220.0, 231.0, 210.0],
            "SOME_2_5_FEATURE_BEFORE": [0.1, 0.2, 0.3],
        }
    )


@pytest.fixture
def no_lineups(monkeypatch):
    """The lineup family needs the rating cache and stints; stub it here."""

    def stub(games, players, ctx):
        return pd.DataFrame(
            np.nan, index=games["GAME_ID"], columns=list(LINEUP_FEATURE_COLUMNS)
        )

    monkeypatch.setattr(v2_6, "_lineup", stub)


def _expected(players, game_id, date, team):
    games = pd.DataFrame({"GAME_ID": [game_id], "TEAM_ID": [team], "GAME_DATE": [date]})
    return add_starter_history_features(games, players).iloc[0]


def test_starter_columns_match_the_feature_function_per_side(
    players, closing, no_lineups
):
    out = apply_layers(
        closing,
        to_version="2_6",
        dataset_type=CLOSING_LINE,
        ctx=LayerContext(df_players=players),
    )
    assert list(out.columns) == [
        *closing.columns,
        *v2_6.LAYER_2_6.columns_for(CLOSING_LINE),
    ]
    for row, (game_id, date) in enumerate(
        [("0022500004", "2025-10-27"), ("0022500003", "2025-10-25")]
    ):
        for team, side in ((HOME, "_TEAM_HOME"), (AWAY, "_TEAM_AWAY")):
            expected = _expected(players, game_id, date, team)
            for column in v2_6.STARTER_HISTORY_COLUMNS:
                got = out.loc[row, f"{column}{side}"]
                assert got == pytest.approx(expected[column], nan_ok=True)
    # Spot values: before 2025-10-27 the home side had abcde, abcdf, abcdf.
    assert out.loc[0, "STARTER_UNIQUE_LAST_5_GAMES_BEFORE_TEAM_HOME"] == 6
    assert out.loc[0, "STARTER_REPEAT_RATE_LAST_5_GAMES_BEFORE_TEAM_HOME"] == 0.5


#: 19:30 EDT tip-offs of the fixture's last two games.
TIPOFFS = {
    "0022500003": pd.Timestamp("2025-10-25 23:30", tz="UTC"),
    "0022500004": pd.Timestamp("2025-10-27 23:30", tz="UTC"),
}


@pytest.fixture
def snapshot_ctx(players, monkeypatch):
    """Sidecar times and report states seeded, so nothing reaches a database."""

    def no_database(*args, **kwargs):
        raise AssertionError("the layer must not query a database here")

    from nba_ou.create_training_data.schema_layers import inputs as layer_inputs
    from nba_ou.data_processing.injury_status import report_state

    monkeypatch.setattr(layer_inputs, "_load_tipoffs", no_database)
    monkeypatch.setattr(report_state, "load_snapshot_report_states", no_database)

    def build(snapshots):
        ctx = LayerContext(df_players=players)
        ctx.snapshot_times = snapshots[["GAME_ID", "TIME_TO_MATCH_MIN"]].assign(
            TIPOFF_UTC=snapshots["GAME_ID"].map(TIPOFFS)
        )
        # No team has filed: every LU_* value must be NaN, not league average.
        ctx.snapshot_injury_states = {
            int(h): InjuryReportState.empty()
            for h in snapshots["TIME_TO_MATCH_MIN"].unique()
        }
        ctx.cache["lineup_rating_book"] = RatingBook(
            pd.DataFrame(columns=RATING_COLUMNS)
        )
        ctx.cache["lineup_stints"] = pd.DataFrame()
        return ctx

    return build


RATING_COLUMNS = [
    "as_of_date",
    "player_id",
    "o_rating",
    "d_rating",
    "pace_rating",
    "league_ortg",
    "league_pace",
    "fit_max_game_date",
]


def test_intermediate_rows_share_their_games_starters_and_add_the_lineup_family(
    snapshot_ctx,
):
    snapshots = pd.DataFrame(
        {
            "GAME_ID": ["0022500004"] * 3 + ["0022500003"] * 2,
            "TIME_TO_MATCH_MIN": [720, 60, 0, 60, 0],
            "GAME_DATE": ["2025-10-27"] * 3 + ["2025-10-25"] * 2,
            "TEAM_ID_TEAM_HOME": [HOME] * 5,
            "TEAM_ID_TEAM_AWAY": [AWAY] * 5,
        }
    )
    out = apply_layers(
        snapshots,
        to_version="2_6",
        dataset_type=INTERMEDIATE_LINE,
        ctx=snapshot_ctx(snapshots),
    )
    added = [c for c in out.columns if c not in snapshots.columns]
    assert added == [*v2_6.STARTER_COLUMNS, *LINEUP_FEATURE_COLUMNS]
    # Every cutoff here is after 05:00 ET on game day: one history date a game.
    per_game = out.groupby("GAME_ID")[list(v2_6.STARTER_COLUMNS)].nunique(dropna=False)
    assert (per_game == 1).all().all()
    assert out[list(LINEUP_FEATURE_COLUMNS)].isna().all().all()
    assert out[list(LINEUP_FEATURE_COLUMNS)].dtypes.eq(np.float64).all()


def test_a_snapshot_before_the_previous_day_settled_uses_older_starters(snapshot_ctx):
    # 2550 min before game 4's tip is 01:00 EDT on the 26th: the 25th (game 3)
    # may still be in progress, so starters are as of game 3, not game 4.
    snapshots = pd.DataFrame(
        {
            "GAME_ID": ["0022500004", "0022500004", "0022500003"],
            "TIME_TO_MATCH_MIN": [2550, 0, 0],
            "GAME_DATE": ["2025-10-27", "2025-10-27", "2025-10-25"],
            "TEAM_ID_TEAM_HOME": [HOME] * 3,
            "TEAM_ID_TEAM_AWAY": [AWAY] * 3,
        }
    )
    out = apply_layers(
        snapshots,
        to_version="2_6",
        dataset_type=INTERMEDIATE_LINE,
        ctx=snapshot_ctx(snapshots),
    )
    starters = out[list(v2_6.STARTER_COLUMNS)]
    pd.testing.assert_series_equal(
        starters.iloc[0], starters.iloc[2], check_names=False
    )
    assert not starters.iloc[0].equals(starters.iloc[1])


def test_row_order_of_the_parent_does_not_change_values(players, closing, no_lineups):
    ctx = LayerContext(df_players=players)
    forward = apply_layers(
        closing, to_version="2_6", dataset_type=CLOSING_LINE, ctx=ctx
    )
    shuffled = closing.sample(frac=1, random_state=3)
    backward = apply_layers(
        shuffled, to_version="2_6", dataset_type=CLOSING_LINE, ctx=ctx
    )
    pd.testing.assert_frame_equal(backward.loc[forward.index], forward)


def test_the_lineup_family_receives_one_row_per_game_in_date_order(
    players, closing, monkeypatch
):
    seen = {}

    def spy(games, players, ctx):
        seen["games"] = games.copy()
        return pd.DataFrame(
            0.0, index=games["GAME_ID"], columns=list(LINEUP_FEATURE_COLUMNS)
        )

    monkeypatch.setattr(v2_6, "_lineup", spy)
    apply_layers(
        closing,
        to_version="2_6",
        dataset_type=CLOSING_LINE,
        ctx=LayerContext(df_players=players),
    )
    games = seen["games"]
    assert games["GAME_ID"].tolist() == ["0022500002", "0022500003", "0022500004"]
    assert games["TOTAL_POINTS"].tolist() == [210.0, 231.0, 220.0]
    assert {"GAME_DATE", "TEAM_ID_TEAM_HOME", "TEAM_ID_TEAM_AWAY"} <= set(games.columns)


def test_a_frame_without_totals_still_builds(players, closing, no_lineups):
    out = apply_layers(
        closing.drop(columns="TOTAL_POINTS"),
        to_version="2_6",
        dataset_type=CLOSING_LINE,
        ctx=LayerContext(df_players=players),
    )
    assert "STARTER_UNIQUE_LAST_5_GAMES_BEFORE_TEAM_AWAY" in out.columns
