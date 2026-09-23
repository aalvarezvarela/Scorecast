"""Evaluate the phase-F lineup projection against the closing total line.

Regenerates the numbers in ``docs/lineup_projection_plan.md`` sections 7.1d
and 8.4 from the committed code: the projection's MAE, and the slope of
``LINE_ERROR`` on the absence counterfactual per season, with date-clustered
bootstrap intervals and directional accuracy.

The projection is computed exactly as the ``LU_*`` features are
(``nba_ou.data_processing.lineups.features``), so a number here is a number
about the column the model sees.

Example::

    python scripts/lineups/evaluate_game_projection.py \\
        --lines data/train_data/training_data_2_3_20260909.csv \\
        --output-dir data/lineup_ratings/eval_projection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.data_processing.injury_status.report_state import (
    load_injury_report_state,
)
from nba_ou.data_processing.lineups.availability import (
    RECENT_GAMES,
    player_out_probabilities,
)
from nba_ou.data_processing.lineups.features import (
    OFFSET_WINDOW_GAMES,
    load_rating_book,
    project_lineup_games,
    walk_forward_offset,
)
from nba_ou.data_processing.lineups.projection_eval import (
    BREAK_EVEN_110,
    directional_accuracy,
    line_error_slope,
)
from nba_ou.data_processing.players.attach_player_features import (
    _parse_minutes_series,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.players.fetch_players_data_from_db import load_players_from_db

#: The columns tested against ``LINE_ERROR``.
REGRESSORS = ("impact_points", "proj_minus_line")

ACCURACY_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0)


def _seasons(first: int, last: int) -> list[str]:
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(first, last + 1)]


def _game_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game with both team ids and the final total."""
    frame = games.copy()
    frame["GAME_ID"] = frame["GAME_ID"].astype(str).str.zfill(10)
    frame["TEAM_ID"] = frame["TEAM_ID"].astype(str)
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"]).dt.normalize()
    frame["HOME"] = frame["HOME"].astype(str).str.lower().isin(["true", "1"])
    home = frame.loc[frame["HOME"], ["GAME_ID", "GAME_DATE", "TEAM_ID", "PTS"]]
    away = frame.loc[~frame["HOME"], ["GAME_ID", "TEAM_ID", "PTS"]]
    table = home.merge(away, on="GAME_ID", suffixes=("_HOME", "_AWAY"))
    table = table.drop_duplicates("GAME_ID", keep=False)
    return pd.DataFrame(
        {
            "GAME_ID": table["GAME_ID"],
            "GAME_DATE": table["GAME_DATE"],
            "HOME_TEAM_ID": table["TEAM_ID_HOME"],
            "AWAY_TEAM_ID": table["TEAM_ID_AWAY"],
            "TOTAL_POINTS": pd.to_numeric(table["PTS_HOME"], errors="coerce")
            + pd.to_numeric(table["PTS_AWAY"], errors="coerce"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first-season", type=int, default=2018)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument(
        "--ratings",
        type=Path,
        default=None,
        help="Rating cache (default: data/lineup_ratings/player_ratings.parquet)",
    )
    parser.add_argument(
        "--lines",
        type=Path,
        required=True,
        help="A closing-line training CSV carrying GAME_ID and the main total line",
    )
    parser.add_argument("--recent-games", type=int, default=RECENT_GAMES)
    parser.add_argument("--roster-games", type=int)
    parser.add_argument(
        "--holdout-season",
        type=int,
        default=2025,
        help="Season reported separately as untouched by tuning (2025 = 2025-26)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seasons = _seasons(args.first_season, args.last_season)
    games = load_games_from_db(seasons=seasons)
    players = load_players_from_db(seasons=seasons)
    if games is None or players is None:
        raise RuntimeError("Could not load games and players")
    games.columns = games.columns.str.upper()
    players.columns = players.columns.str.upper()
    table = _game_table(games)
    players["GAME_ID"] = players["GAME_ID"].astype(str).str.zfill(10)
    players["MIN"] = _parse_minutes_series(players["MIN"]).fillna(0.0)
    players = players.merge(table[["GAME_ID", "GAME_DATE"]], on="GAME_ID")

    p_out = player_out_probabilities(load_injury_report_state(None).statuses)
    ratings = load_rating_book(args.ratings)

    projected = project_lineup_games(
        table,
        players,
        ratings,
        p_out,
        recent_games=args.recent_games,
        roster_games=args.roster_games,
    )
    result = table.merge(projected, on="GAME_ID").sort_values(["GAME_DATE", "GAME_ID"])
    result["offset"] = walk_forward_offset(
        result["GAME_DATE"], result["raw_total"], result["TOTAL_POINTS"]
    )
    result["proj_total"] = result["raw_total"] + result["offset"]

    line = total_line_col()
    lines = pd.read_csv(args.lines, usecols=["GAME_ID", line])
    lines["GAME_ID"] = lines["GAME_ID"].astype(str).str.zfill(10)
    result = result.merge(lines.drop_duplicates("GAME_ID"), on="GAME_ID", how="left")
    result["LINE_ERROR"] = result["TOTAL_POINTS"] - result[line]
    result["proj_minus_line"] = result["proj_total"] - result[line]
    result["season"] = result["GAME_DATE"].dt.year - (result["GAME_DATE"].dt.month < 8)

    scored = result.dropna(subset=["LINE_ERROR", "proj_total"])
    summary: dict = {
        "games_projected": len(result),
        "games_scored": len(scored),
        "recent_games": args.recent_games,
        "roster_games": args.roster_games,
        "offset_window_games": OFFSET_WINDOW_GAMES,
        "mae_projection": float(
            (scored["TOTAL_POINTS"] - scored["proj_total"]).abs().mean()
        ),
        "mae_line": float(scored["LINE_ERROR"].abs().mean()),
        "break_even": BREAK_EVEN_110,
        "slopes": {},
        "accuracy": {},
    }
    windows = {
        "all": scored,
        "before_holdout": scored.loc[scored["season"] < args.holdout_season],
        "holdout": scored.loc[scored["season"].eq(args.holdout_season)],
    } | {
        f"season_{s}": scored.loc[scored["season"].eq(s)]
        for s in sorted(scored["season"].unique())
    }
    for regressor in REGRESSORS:
        summary["slopes"][regressor] = {
            name: line_error_slope(frame, regressor) for name, frame in windows.items()
        }
    for name in ("before_holdout", "holdout"):
        summary["accuracy"][name] = [
            directional_accuracy(windows[name], "impact_points", threshold)
            for threshold in ACCURACY_THRESHOLDS
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_dir / "projection_games.parquet", index=False)
    (args.output_dir / "projection_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )

    print(
        f"MAE projection {summary['mae_projection']:.3f}, line {summary['mae_line']:.3f} "
        f"on {len(scored):,} games"
    )
    for regressor, rows in summary["slopes"].items():
        print(f"\nLINE_ERROR ~ {regressor}")
        for name, row in rows.items():
            print(
                f"  {name:15s} n={row['n']:5d} slope={row['slope']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            )
    for name, rows in summary["accuracy"].items():
        print(f"\nOVER when impact > 0, {name} (break-even {BREAK_EVEN_110:.2%})")
        for row in rows:
            print(
                f"  |impact|>={row['threshold']:g}  n={row['n']:5d} "
                f"acc={row['accuracy']:.2%} [{row['ci_low']:.2%}, {row['ci_high']:.2%}]"
            )


if __name__ == "__main__":
    main()
