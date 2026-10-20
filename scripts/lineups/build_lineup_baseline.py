"""Build a reproducible closing-horizon LU feature artifact from current code.

The optional old-ratings run is a diagnostic with the same report-coverage mask.
It does not load the stale cache through the production rating loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from nba_ou.config.odds_columns import spread_line_home_col, total_line_col
from nba_ou.data_processing.injury_status.report_state import load_injury_report_state
from nba_ou.data_processing.lineups.availability import (
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import (
    DEFAULT_RATING_CACHE,
    LINEUP_FEATURE_COLUMNS,
    RatingBook,
    add_lineup_features,
    attach_lineup_features,
    load_rating_book,
)
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.players.attach_player_features import (
    _parse_minutes_series,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.players.fetch_players_data_from_db import load_players_from_db

from scripts.lineups.evaluate_game_projection import _game_table, _seasons

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(frame: pd.DataFrame, report_covered: set[tuple[str, str]]) -> list[dict]:
    pairs = {(str(game).zfill(10), str(team)) for game, team in report_covered}
    covered = [
        (row.GAME_ID, row.TEAM_ID_TEAM_HOME) in pairs
        and (row.GAME_ID, row.TEAM_ID_TEAM_AWAY) in pairs
        for row in frame.itertuples(index=False)
    ]
    count_frame = frame[["GAME_DATE", "LU_PROJ_TOTAL_BEFORE"]].copy()
    count_frame["season"] = (
        count_frame.GAME_DATE.dt.year - count_frame.GAME_DATE.dt.month.lt(8)
    )
    count_frame["both_reports"] = covered
    return [
        {
            "season": int(season),
            "games": int(len(group)),
            "both_reports": int(group.both_reports.sum()),
            "missing_report_coverage": int((~group.both_reports).sum()),
            "valid_lu_projection": int(group.LU_PROJ_TOTAL_BEFORE.notna().sum()),
            "covered_but_no_projection": int(
                (group.both_reports & group.LU_PROJ_TOTAL_BEFORE.isna()).sum()
            ),
        }
        for season, group in count_frame.groupby("season", sort=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=2018)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATING_CACHE)
    parser.add_argument("--old-ratings", type=Path)
    parser.add_argument("--lines", type=Path)
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
    players["GAME_ID"] = players.GAME_ID.astype(str).str.zfill(10)
    players["MIN"] = _parse_minutes_series(players.MIN).fillna(0.0)
    players = players.merge(table[["GAME_ID", "GAME_DATE"]], on="GAME_ID")
    merged = table.rename(
        columns={
            "HOME_TEAM_ID": "TEAM_ID_TEAM_HOME",
            "AWAY_TEAM_ID": "TEAM_ID_TEAM_AWAY",
        }
    )
    report = load_injury_report_state(None)
    stints = read_stints(range(args.first_season, args.last_season + 1))
    corrected = attach_lineup_features(
        merged,
        players,
        enabled=True,
        injury_statuses=report.statuses,
        report_covered=report.covered,
        ratings=load_rating_book(args.ratings),
        stints=stints,
    )
    if args.lines is not None:
        line, spread = total_line_col(), spread_line_home_col()
        lines = pd.read_csv(args.lines, usecols=["GAME_ID", line, spread])
        lines["GAME_ID"] = lines.GAME_ID.astype(str).str.zfill(10)
        corrected = corrected.merge(
            lines.drop_duplicates("GAME_ID"), on="GAME_ID", how="left"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "lineup_features.parquet"
    corrected.to_parquet(feature_path, index=False)
    cache_metadata_path = args.ratings.with_suffix(".metadata.json")
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "horizon": "closing; last injury report strictly before tipoff",
        "rating_cache": str(args.ratings),
        "rating_cache_sha256": _sha256(args.ratings),
        "rating_cache_metadata": json.loads(cache_metadata_path.read_text()),
        "feature_sha256": _sha256(feature_path),
        "feature_columns": list(LINEUP_FEATURE_COLUMNS),
        "stint_seasons": sorted(stints.season_year.unique().astype(int).tolist()),
        "stint_rows": int(len(stints)),
        "stint_games": int(stints.game_id.nunique()),
        "source_games": int(len(table)),
        "source_player_rows": int(len(players)),
        "injury_status_rows": int(len(report.statuses)),
        "injury_covered_team_games": int(len(report.covered)),
        "coverage_by_season": _counts(corrected, report.covered),
        "code_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                ROOT / "src/nba_ou/data_processing/lineups/features.py",
                ROOT / "src/nba_ou/data_processing/lineups/game_projection.py",
                ROOT / "src/nba_ou/data_processing/lineups/player_ratings.py",
                ROOT / "src/nba_ou/data_processing/lineups/style_matchup.py",
                Path(__file__).resolve(),
            )
        },
    }
    if args.lines is not None:
        metadata["lines_path"] = str(args.lines)
        metadata["lines_sha256"] = _sha256(args.lines)
    if args.old_ratings is not None:
        old = add_lineup_features(
            merged,
            players,
            RatingBook(pd.read_parquet(args.old_ratings)),
            player_out_probabilities(report.statuses),
            excluded=roster_exclusions(report.statuses),
            report_covered=report.covered,
        )
        old_path = args.output_dir / "old_ratings_same_coverage.parquet"
        old[["GAME_ID", *[c for c in LINEUP_FEATURE_COLUMNS if c in old]]].to_parquet(
            old_path, index=False
        )
        metadata["old_rating_cache"] = str(args.old_ratings)
        metadata["old_rating_cache_sha256"] = _sha256(args.old_ratings)
        metadata["old_rating_feature_sha256"] = _sha256(old_path)
    (args.output_dir / "lineup_features.metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata["coverage_by_season"], indent=2))
    print(f"Wrote {feature_path}")


if __name__ == "__main__":
    main()
