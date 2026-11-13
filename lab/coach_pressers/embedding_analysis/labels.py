"""Same-game statistics of each presser, seen from the coach's team.

One row per (game, team): the box score of the team and of its opponent, the
regulars each side missed (`role_probes.regulars_out`), what the team had done
before the game and what the closing line expected. Everything except the
`context` and `market_expectation` groups is known only after the game: these
are labels to read the vectors against, never features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lab.text_to_features import role_probes as rp

#: Box-score columns copied for the team (`target_*`) and its opponent.
TEAM_STATS = (
    "pts",
    "fg_pct",
    "fg3_pct",
    "fg3m",
    "efg_pct",
    "fta",
    "oreb_pct",
    "tm_tov_pct",
    "ast",
    "stl",
    "blk",
    "pf",
    "reb",
    "tov",
)
ADVANCED_STATS = ("pace", "efg_pct", "oreb_pct", "tm_tov_pct")
LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "result": (
        "target_won",
        "target_margin",
        "target_points",
        "opponent_points",
        "total_points",
        "pace",
    ),
    "target_box": (
        "target_efg_pct",
        "target_fg3_pct",
        "target_fta",
        "target_oreb_pct",
        "target_tm_tov_pct",
        "target_ast",
        "target_stl",
        "target_blk",
        "target_pf",
    ),
    "opponent_box": (
        "opponent_efg_pct",
        "opponent_fg3_pct",
        "opponent_fta",
        "opponent_oreb_pct",
        "opponent_tm_tov_pct",
    ),
    "differences": ("reb_diff", "tov_diff", "fg3m_diff"),
    "availability": ("target_regulars_out", "opponent_regulars_out"),
    "context": ("target_is_home", "target_prior_win_pct", "target_prior_games"),
    "market_expectation": ("target_expected_margin", "total_line"),
    "market_surprise": ("target_margin_vs_spread", "total_vs_line"),
}
SPREAD_COLUMN = "ODDS_SPREAD_LINE_HOME_bet365"  # market-implied home margin
TOTAL_COLUMN = "ODDS_TOTAL_LINE_bet365"


def load_box_scores(seasons: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team games of the seasons and the regulars each team missed."""
    from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
        load_games_from_db,
    )
    from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (  # noqa: E501
        get_injury_data_from_db,
    )
    from nba_ou.postgre_db.players.fetch_players_data_from_db import (
        load_players_from_db,
    )

    games = load_games_from_db(seasons=seasons)
    box = load_players_from_db(seasons=seasons)
    inactive = get_injury_data_from_db(seasons=seasons)
    if games is None or box is None or inactive is None:
        raise RuntimeError("Could not load games, box scores or inactive lists")
    games = games.assign(
        game_id=games["game_id"].astype(str), team_id=games["team_id"].astype(str)
    )
    minutes = pd.DataFrame(
        {
            "game_id": box["game_id"].astype(str),
            "team_id": box["team_id"].astype(str),
            "player_id": box["player_id"].astype(str),
            "minutes": box["min"].map(rp.parse_minutes),
        }
    )
    listed = pd.concat(
        [
            minutes[["game_id", "team_id", "player_id"]],
            pd.DataFrame(
                {
                    "game_id": inactive["GAME_ID"].astype(str),
                    "team_id": inactive["TEAM_ID"].astype(str),
                    "player_id": inactive["PLAYER_ID"].astype(str),
                }
            ),
        ]
    )
    order = rp.team_game_order(games)
    return games, rp.regulars_out(order, minutes, listed)


def team_game_labels(games: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team): the labels of `LABEL_GROUPS` except the
    market ones, plus `team` (tricode) and `team_id`."""
    games = games[games["game_id"].str[:3].isin(rp.GAME_TYPES)].copy()
    games = games.drop_duplicates(["game_id", "team_id"])
    games = games.merge(out, on=["team_id", "game_id"], how="left")
    # A few games have no advanced box score: zeros there are missing values.
    no_advanced = games["pace"].fillna(0) <= 0
    games.loc[no_advanced, list(ADVANCED_STATS)] = float("nan")
    games["won"] = (games["wl"] == "W").astype(float)
    games = games.sort_values(["team_id", "season_year", "game_date", "game_id"])
    by_season = games.groupby(["team_id", "season_year"])
    games["prior_games"] = by_season.cumcount()
    wins_before = by_season["won"].cumsum() - games["won"]
    games["prior_win_pct"] = (wins_before / games["prior_games"]).where(
        games["prior_games"] > 0
    )

    columns = ["game_id", "team_id", *TEAM_STATS, "regulars_out"]
    own = games[
        [*columns, "team_abbreviation", "home", "won", "pace"]
        + ["prior_games", "prior_win_pct"]
    ]
    other = games[columns].rename(
        columns={c: f"opp_{c}" for c in columns if c != "game_id"}
    )
    pairs = own.merge(other, on="game_id")
    pairs = pairs[pairs["team_id"] != pairs["opp_team_id"]]

    labels = pd.DataFrame(
        {
            "game_id": pairs["game_id"],
            "team": pairs["team_abbreviation"],
            "team_id": pairs["team_id"],
            "target_won": pairs["won"],
            "target_margin": pairs["pts"] - pairs["opp_pts"],
            "target_points": pairs["pts"],
            "opponent_points": pairs["opp_pts"],
            "total_points": pairs["pts"] + pairs["opp_pts"],
            "pace": pairs["pace"],
            "reb_diff": pairs["reb"] - pairs["opp_reb"],
            "tov_diff": pairs["tov"] - pairs["opp_tov"],
            "fg3m_diff": pairs["fg3m"] - pairs["opp_fg3m"],
            "target_regulars_out": pairs["regulars_out"],
            "opponent_regulars_out": pairs["opp_regulars_out"],
            "target_is_home": pairs["home"].astype(float),
            "target_prior_win_pct": pairs["prior_win_pct"],
            "target_prior_games": pairs["prior_games"],
        }
    )
    for name in LABEL_GROUPS["target_box"]:
        labels[name] = pairs[name.removeprefix("target_")]
    for name in LABEL_GROUPS["opponent_box"]:
        labels[name] = pairs["opp_" + name.removeprefix("opponent_")]
    return labels.reset_index(drop=True)


def add_market(labels: pd.DataFrame, closing_file: Path) -> pd.DataFrame:
    """The closing spread and total of the main book, seen from the team."""
    lines = pd.read_parquet(
        closing_file, columns=["GAME_ID", SPREAD_COLUMN, TOTAL_COLUMN]
    )
    lines = lines.rename(columns={"GAME_ID": "game_id"}).drop_duplicates("game_id")
    lines["game_id"] = lines["game_id"].astype(str)
    frame = labels.merge(lines, on="game_id", how="left")
    side = frame["target_is_home"] * 2 - 1
    frame["target_expected_margin"] = side * frame[SPREAD_COLUMN]
    frame["total_line"] = frame[TOTAL_COLUMN]
    frame["target_margin_vs_spread"] = (
        frame["target_margin"] - frame["target_expected_margin"]
    )
    frame["total_vs_line"] = frame["total_points"] - frame["total_line"]
    return frame.drop(columns=[SPREAD_COLUMN, TOTAL_COLUMN])
