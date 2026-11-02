"""Stint-graph pretraining dataset: targets, guard provenance, as of the date."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.as_of import (
    CORRECTED_MATCHUP_HISTORY,
    PointInTimeData,
)
from nba_ou.data_processing.player_graph.node_profiles import PROFILE_COLUMNS
from nba_ou.data_processing.player_graph.pretraining_data import (
    GUARD_SOURCES,
    NODE_COLUMNS,
    build_day,
    check_tables,
    guard_arrays,
    same_position_rate,
    stint_table,
    stint_targets,
    to_arrays,
)
from nba_ou.data_processing.player_graph.stint_graph import LABELS

GAME = "0021600001"
DAY = pd.Timestamp("2016-11-10")
HOME = [f"h{k}" for k in range(5)]
AWAY = [f"a{k}" for k in range(5)]
#: Starting position of each starter; h0/a0 are guards, h4/a4 centers.
POSITION = {f"{side}{k}": "GGFFC"[k] for side in "ha" for k in range(5)}


def _stints(**overrides):
    row = {
        "game_id": GAME,
        "season_year": 2016,
        "game_date": DAY,
        "seg_idx": 0,
        "period": 1,
        "seconds": 240.0,
        "home_team_id": "1",
        "away_team_id": "2",
        "home_lineup": np.array(HOME),
        "away_lineup": np.array(AWAY),
        "home_pts": 10,
        "away_pts": 6,
        "home_poss": 8.0,
        "away_poss": 7.0,
        "home_tov": 1,
        "away_tov": 2,
        "home_fga": 8,
        "away_fga": 0,
        "home_fg3a": 4,
        "away_fg3a": 0,
        "home_fta": 2,
        "away_fta": 3,
        "home_oreb": 1,
        "away_oreb": 0,
        "home_dreb": 3,
        "away_dreb": 4,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_targets_carry_their_own_exposures():
    labels, exposure = stint_targets(_stints())
    names = {name: k for k, name in enumerate(LABELS)}
    home, away = 0, 1
    assert labels[0, home, names["pts_per_poss"]] == pytest.approx(10 / 8)
    assert exposure[0, home, names["pts_per_poss"]] == 8.0
    assert labels[0, home, names["fg3a_per_fga"]] == pytest.approx(0.5)
    assert exposure[0, home, names["fg3a_per_fga"]] == 8.0
    # Rebound chances: own OREB + the opponent's DREB.
    assert labels[0, home, names["oreb_rate"]] == pytest.approx(1 / 5)
    assert exposure[0, home, names["oreb_rate"]] == 5.0
    # No FGA for the away side: no 3PA/FGA label and no weight.
    assert np.isnan(labels[0, away, names["fg3a_per_fga"]])
    assert exposure[0, away, names["fg3a_per_fga"]] == 0.0
    # Pace is the stint's (mean possessions), half the seconds on each side.
    pace = names["poss_per_48"]
    assert (
        labels[0, home, pace]
        == labels[0, away, pace]
        == pytest.approx(7.5 * 2880 / 240)
    )
    assert exposure[0, :, pace].tolist() == [120.0, 120.0]


def test_no_possessions_means_no_label_never_a_negative_weight():
    labels, exposure = stint_targets(_stints(home_poss=-0.44, seconds=5.0))
    names = {name: k for k, name in enumerate(LABELS)}
    assert np.isnan(labels[0, 0, names["pts_per_poss"]])
    assert (exposure >= 0).all()


def test_same_position_rate():
    guard, center = np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]])
    assert same_position_rate(guard, guard)[0] == 1.0
    assert same_position_rate(guard, center)[0] == 0.0


def _guard_rows(r_hat=None, structural=None):
    rows = []
    for attackers, defenders in ((HOME, AWAY), (AWAY, HOME)):
        for k, att in enumerate(attackers):
            for l, dfn in enumerate(defenders):  # noqa: E741
                known = r_hat is not None
                rows.append(
                    {
                        "game_id": GAME,
                        "game_date": DAY,
                        "off_player_id": att,
                        "def_player_id": dfn,
                        "r_hat": (0.4 if k == l else 0.1) if known else np.nan,
                        "r_prior": 0.1 if known else np.nan,
                        "prior_weight": 0.5 if known else 1.0,
                        "n_games": 2 if known else 0,
                        "hist_cofloor_seconds_decayed": 300.0 if known else 0.0,
                        "has_pair_history": known,
                        "r_structural": 1.0 if k == l else 0.0,
                        "guard_source": "pair_history" if known else "structural",
                    }
                )
    return pd.DataFrame(rows)


def test_guard_shares_follow_the_provenance_rule():
    stints = _stints()
    known = guard_arrays(stints, _guard_rows(r_hat=True))
    assert known["guard_share"][0, 0] == pytest.approx(0.4 / 0.8)
    assert (known["guard_source"] == GUARD_SOURCES.index("pair_history")).all()

    structural = guard_arrays(stints, _guard_rows())
    assert structural["guard_share"][0, 0] == 1.0  # attacker 0 <- defender 0
    assert structural["guard_share"][0, 1] == 0.0
    assert (structural["guard_source"] == GUARD_SOURCES.index("structural")).all()
    assert (structural["guard_r_hat"] == 0).all()
    assert (structural["guard_prior_weight"] == 1).all()

    uniform = guard_arrays(stints, _guard_rows(), structural=False)
    assert np.allclose(uniform["guard_share"], 0.2)
    assert (uniform["guard_source"] == GUARD_SOURCES.index("uniform")).all()


def test_a_stint_pair_without_a_guard_row_is_an_error():
    rows = _guard_rows(r_hat=True).iloc[1:]
    with pytest.raises(ValueError, match="no guard row"):
        guard_arrays(_stints(), rows)


# --------------------------------------------------------------------------
# One date, as of that date
# --------------------------------------------------------------------------


def _box(dates, game_prefix="00216"):
    rows = []
    for day, date in enumerate(dates):
        game = f"{game_prefix}{day + 100:05d}"
        for player, position in POSITION.items():
            rows.append(
                {
                    "GAME_ID": game,
                    "GAME_DATE": date,
                    "PLAYER_ID": player,
                    "MIN": 30,
                    "START_POSITION": position,
                    "AST": 2,
                    "OREB": 1,
                    "DREB": 3,
                    "BLK": 1,
                    "STL": 1,
                    "FG3A": 3,
                    "FGA": 10,
                    "PF": 2,
                }
            )
    return pd.DataFrame(rows)


def _profiles(players, date=DAY):
    table = pd.DataFrame(
        {column: 0.5 for column in PROFILE_COLUMNS}, index=range(len(players))
    )
    table["player_id"] = list(players)
    table["as_of_date"] = date
    return table


def _data(box):
    data = PointInTimeData.from_frames(pd.DataFrame(), pd.DataFrame(), box)
    return data.without_corrupt_matchups()[0]


def test_build_day_uses_box_starts_and_nothing_from_the_date_on():
    before = pd.date_range("2016-10-26", periods=12, freq="D")
    data = _data(_box(before))
    stints = _stints()
    nodes, guard = build_day(
        data.as_of(DAY), stints, _guard_rows().iloc[:0], _profiles(HOME + AWAY)
    )
    by_player = nodes.set_index("player_id")
    positions = by_player[["p_G", "p_F", "p_C"]]
    assert positions.loc["h0"].idxmax() == "p_G" and positions.loc["h0", "p_G"] > 0.7
    assert positions.loc["a4"].idxmax() == "p_C" and positions.loc["a4", "p_C"] > 0.7
    assert by_player.loc["h0", "position_starts"] == len(before)
    assert set(nodes.columns) == {"as_of_date", "player_id", *NODE_COLUMNS}
    assert guard["guard_source"].eq("structural").all()
    guard = guard.set_index(["off_player_id", "def_player_id"])
    assert (
        guard.loc[("h0", "a0"), "r_structural"]
        > guard.loc[("h0", "a4"), "r_structural"]
    )

    # Box scores on the date and after it, with other positions: no change.
    later = _box(pd.date_range(DAY, periods=5, freq="D"), game_prefix="00217")
    later["START_POSITION"] = "C"
    poisoned = _data(pd.concat([_box(before), later], ignore_index=True))
    nodes_2, guard_2 = build_day(
        poisoned.as_of(DAY), stints, _guard_rows().iloc[:0], _profiles(HOME + AWAY)
    )
    pd.testing.assert_frame_equal(nodes, nodes_2)
    pd.testing.assert_frame_equal(
        guard.reset_index()[guard_2.columns], guard_2, check_like=True
    )


def test_build_day_needs_every_node_profile():
    data = _data(_box(pd.date_range("2016-10-26", periods=3, freq="D")))
    with pytest.raises(ValueError, match="no node profile"):
        build_day(data.as_of(DAY), _stints(), _guard_rows().iloc[:0], _profiles(HOME))


def test_season_builder_refuses_the_original_matchup_history():
    from nba_ou.data_processing.player_graph.pretraining_data import build_season

    data = PointInTimeData.from_frames(_stints(), pd.DataFrame(), _box([DAY]))
    with pytest.raises(ValueError, match="corrected"):
        build_season(data, 2016, _guard_rows(), _profiles(HOME + AWAY))
    assert data.without_corrupt_matchups()[0].matchup_history == (
        CORRECTED_MATCHUP_HISTORY
    )


# --------------------------------------------------------------------------
# Stored tables: checks and the loader
# --------------------------------------------------------------------------


def _tables():
    stints = _stints()
    nodes = _profiles(HOME + AWAY).assign(position_starts=3)
    return {
        "stints": stint_table(stints),
        "nodes": nodes[["as_of_date", "player_id", *NODE_COLUMNS]],
        "guard": _guard_rows(r_hat=True),
    }


def test_valid_tables_pass_and_load():
    tables = _tables()
    summary = check_tables(tables, set())
    assert summary["stints"] == 1 and summary["guard_rows"] == 50
    arrays = to_arrays(tables)
    assert arrays.nodes.shape == (1, 10, len(NODE_COLUMNS))
    assert arrays.players[0].tolist() == HOME + AWAY
    assert arrays.labels.shape == arrays.label_weight.shape == (1, 2, len(LABELS))
    assert np.allclose(arrays.guard_share.reshape(2, 5, 5).sum(axis=2), 1.0)


@pytest.mark.parametrize(
    "break_it, message",
    [
        (lambda t: t["stints"].assign(game_id="0021800927"), "stint-rejected"),
        (lambda t: t["nodes"].iloc[1:], "without features"),
        (lambda t: t["guard"].iloc[1:], "without a guard row"),
        (lambda t: t["guard"].assign(matchup_seconds=1.0), "observed"),
        (lambda t: t["stints"].assign(away_4="h0"), "ten distinct"),
    ],
)
def test_checks_catch_broken_tables(break_it, message):
    tables = _tables()
    broken = break_it(tables)
    name = next(n for n in tables if set(broken.columns) >= set(tables[n].columns))
    if "matchup_seconds" in broken.columns:
        name = "guard"
    tables[name] = broken
    with pytest.raises(ValueError, match=message):
        check_tables(tables, set())


def test_structural_rates_never_fill_a_covered_date():
    tables = _tables()
    guard = tables["guard"].copy()
    guard.loc[0, ["r_hat", "guard_source"]] = [np.nan, "structural"]
    tables["guard"] = guard
    with pytest.raises(ValueError, match="dates with matchup history"):
        check_tables(tables, set())
