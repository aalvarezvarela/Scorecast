"""Evaluate the phase-F lineup projection against the closing total line.

Regenerates the numbers in ``docs/lineup_projection_plan.md`` sections 7.1d,
8.5 and 8.6 from the committed code: the projection's MAE, the slope of
``LINE_ERROR`` on the absence counterfactual and on each of its channels per
season, with date-clustered bootstrap intervals and directional accuracy, and
the same for the spread (``SPREAD_ERROR`` on the margin columns).

The projection uses the same rating and availability components as ``LU_*``.
Only games with both teams' pre-tip injury filings are evaluated. These are
exploratory historical diagnostics; the revised ridge solver requires a rebuilt
rating cache and a new run before the old reported numbers can be reused.

Example::

    python scripts/lineups/evaluate_game_projection.py \\
        --lines data/train_data/training_data_2_3_20260909.csv \\
        --output-dir data/lineup_ratings/eval_projection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import spread_line_home_col, total_line_col
from nba_ou.data_processing.injury_status.report_state import (
    load_injury_report_state,
)
from nba_ou.data_processing.lineups.availability import (
    RECENT_GAMES,
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import (
    DATA_ROOT,
    OFFSET_MIN_GAMES,
    OFFSET_WINDOW_GAMES,
    game_nights,
    game_phase,
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
REGRESSORS = (
    "impact_points",
    "impact_offense",
    "impact_defense",
    "impact_pace",
    "impact_def_pace",
    "impact_bench_replacement",
    "proj_minus_line",
)

#: And against ``SPREAD_ERROR`` (home margin minus the home spread line).
SPREAD_REGRESSORS = ("impact_margin", "margin_minus_line")

#: Columns whose directional accuracy is reported.
ACCURACY_COLUMNS = ("impact_points", "impact_def_pace")

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
            "HOME_MARGIN": pd.to_numeric(table["PTS_HOME"], errors="coerce")
            - pd.to_numeric(table["PTS_AWAY"], errors="coerce"),
        }
    )


def _style_summary(
    table: pd.DataFrame, nights, scored: pd.DataFrame
) -> tuple[dict, pd.DataFrame]:
    """The three-point matchup columns: do they predict 3PA, and the line?

    Also returns the per-game columns, so they are saved with the projection.
    """
    from nba_ou.data_processing.lineups.stint_store import read_stints
    from nba_ou.data_processing.lineups.style_matchup import (
        build_style_matchup_features,
        stint_directions,
    )

    stints = read_stints(None, local_root=DATA_ROOT)
    style = build_style_matchup_features(stints, table, nights)
    counts = stint_directions(stints).groupby("game_id")[["fg3a", "fga"]].sum()
    frame = style.merge(
        (counts.fg3a / counts.fga).rename("actual_fg3a_rate"),
        left_on="GAME_ID",
        right_index=True,
    )
    neighbor_residual = frame["LU_FG3A_NEIGHBOR_RESIDUAL_BEFORE"]
    projected = (
        frame["LU_PROJ_FG3A_RATE_BEFORE_TEAM_HOME"]
        + frame["LU_PROJ_FG3A_RATE_BEFORE_TEAM_AWAY"]
    ) / 2
    additive_error = frame["actual_fg3a_rate"] - (projected - neighbor_residual)
    lined = scored.merge(style, on="GAME_ID")
    summary = {
        "games": len(frame),
        "corr_neighbor_residual_vs_additive_fg3a_error": float(
            np.corrcoef(neighbor_residual, additive_error)[0, 1]
        ),
        "mae_fg3a_additive": float(additive_error.abs().mean()),
        "mae_fg3a_with_neighbor_residual": float(
            (frame["actual_fg3a_rate"] - projected).abs().mean()
        ),
        "line_error_slopes": {
            column: line_error_slope(lined, column)
            for column in (
                "LU_FG3A_NEIGHBOR_RESIDUAL_BEFORE",
                "LU_ABSENCE_SHIFT_FG3A_RATE_BEFORE",
            )
        },
    }
    return summary, style


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
        "--diagnostic-season",
        "--holdout-season",
        dest="diagnostic_season",
        type=int,
        default=2025,
        help="Season reported separately for diagnostics (2025-26 was inspected during feature selection)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--style",
        action="store_true",
        help="Also build the three-point matchup columns from the stint store "
        "(~3 min) and report how they relate to the game's 3PA rate and the line",
    )
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

    report = load_injury_report_state(None)
    statuses = report.statuses
    covered = {(str(game).zfill(10), str(team)) for game, team in report.covered}
    table = table.loc[
        table.apply(
            lambda row: (
                (row.GAME_ID, row.HOME_TEAM_ID) in covered
                and (row.GAME_ID, row.AWAY_TEAM_ID) in covered
            ),
            axis=1,
        )
    ].copy()
    p_out = player_out_probabilities(statuses)
    ratings = load_rating_book(args.ratings)

    nights = game_nights(
        table,
        players,
        p_out,
        recent_games=args.recent_games,
        roster_games=args.roster_games,
        excluded=roster_exclusions(statuses),
    )
    projected = project_lineup_games(table, players, ratings, p_out, nights=nights)
    result = table.merge(projected, on="GAME_ID").sort_values(["GAME_DATE", "GAME_ID"])
    result["offset"] = walk_forward_offset(
        result["GAME_DATE"],
        result["raw_total"],
        result["TOTAL_POINTS"],
        phases=game_phase(result["GAME_ID"]),
    )
    result["proj_total"] = result["raw_total"] + result["offset"]

    line = total_line_col()
    spread = spread_line_home_col()
    lines = pd.read_csv(args.lines, usecols=["GAME_ID", line, spread])
    lines["GAME_ID"] = lines["GAME_ID"].astype(str).str.zfill(10)
    result = result.merge(lines.drop_duplicates("GAME_ID"), on="GAME_ID", how="left")
    result["LINE_ERROR"] = result["TOTAL_POINTS"] - result[line]
    result["proj_minus_line"] = result["proj_total"] - result[line]
    result["impact_def_pace"] = result["impact_defense"] + result["impact_pace"]
    result["SPREAD_ERROR"] = result["HOME_MARGIN"] - result[spread]
    result["margin_minus_line"] = result["margin"] - result[spread]
    result["season"] = result["GAME_DATE"].dt.year - (result["GAME_DATE"].dt.month < 8)

    scored = result.dropna(subset=["LINE_ERROR", "proj_total"])
    summary: dict = {
        "games_projected": len(result),
        "games_scored": len(scored),
        "recent_games": args.recent_games,
        "roster_games": args.roster_games,
        "offset_window_games": OFFSET_WINDOW_GAMES,
        "offset_min_games": OFFSET_MIN_GAMES,
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
        "before_diagnostic_season": scored.loc[
            scored["season"] < args.diagnostic_season
        ],
        "diagnostic_season": scored.loc[scored["season"].eq(args.diagnostic_season)],
    } | {
        f"season_{s}": scored.loc[scored["season"].eq(s)]
        for s in sorted(scored["season"].unique())
    }
    for regressor in REGRESSORS:
        summary["slopes"][regressor] = {
            name: line_error_slope(frame, regressor) for name, frame in windows.items()
        }
    for regressor in SPREAD_REGRESSORS:
        summary["slopes"][f"spread:{regressor}"] = {
            name: line_error_slope(frame, regressor, target="SPREAD_ERROR")
            for name, frame in windows.items()
        }
    if args.style:
        summary["style"], style = _style_summary(table, nights, windows["all"])
        result = result.merge(style, on="GAME_ID", how="left")
    for column in ACCURACY_COLUMNS:
        for name in ("before_diagnostic_season", "diagnostic_season"):
            summary["accuracy"][f"{column}:{name}"] = [
                directional_accuracy(windows[name], column, threshold)
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
        target, _, name_ = regressor.rpartition(":")
        print(f"\n{'SPREAD_ERROR' if target else 'LINE_ERROR'} ~ {name_}")
        for name, row in rows.items():
            print(
                f"  {name:15s} n={row['n']:5d} slope={row['slope']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
            )
    if "style" in summary:
        style = summary["style"]
        print(
            f"\nThree-point neighbour residual ({style['games']:,} games): "
            f"corr with additive 3PA error {style['corr_neighbor_residual_vs_additive_fg3a_error']:+.4f}; "
            f"3PA-rate MAE {style['mae_fg3a_additive']:.4f} -> "
            f"{style['mae_fg3a_with_neighbor_residual']:.4f}"
        )
        for column, row in style["line_error_slopes"].items():
            print(
                f"  LINE_ERROR ~ {column}: slope={row['slope']:+.3f} "
                f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] (n={row['n']})"
            )
    for name, rows in summary["accuracy"].items():
        column, _, window = name.partition(":")
        print(f"\nOVER when {column} > 0, {window} (break-even {BREAK_EVEN_110:.2%})")
        for row in rows:
            print(
                f"  |x|>={row['threshold']:g}  n={row['n']:5d} "
                f"acc={row['accuracy']:.2%} [{row['ci_low']:.2%}, {row['ci_high']:.2%}]"
            )


if __name__ == "__main__":
    main()
