"""Repair the public-bet splits of existing odds_yahoo rows from a fresh scrape.

Up to fix/yahoo-http-scraper the Yahoo scraper read each market's "% of bets"
and "% of money" by position, while Yahoo (since the 2026 playoffs) lists the
two options in its own order: the splits of many 2026-27 games are swapped.
The lines are not affected. This script re-scrapes the games of a date range
(the scraper now reads every value from its option's label), and rewrites the
twelve split columns of the rows whose re-scraped lines match the stored ones
exactly; a row whose lines differ is left alone and reported.

Yahoo keeps the odds of a game only for a few months, so only recent games
can be repaired.

Dry run (the default) prints what would change and writes the proposals:

    python scripts/clean_databases/repair_yahoo_splits.py \\
        --start-date 2026-10-20 --end-date 2026-10-24

Apply, after a backup of the touched rows to ``--backup-dir``:

    python scripts/clean_databases/repair_yahoo_splits.py \\
        --start-date 2026-10-20 --end-date 2026-10-24 --apply
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    merge_sportsbook_with_games,
)
from nba_ou.fetch_data.odds_yahoo.scrape_yahoo import scrape_yahoo_days
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_yahoo,
)
from nba_ou.postgre_db.odds_yahoo.update_yahoo.update_database_utils import (
    load_games_for_yahoo_update,
)
from nba_ou.postgre_db.odds_yahoo.update_yahoo.update_yahoo_database import (
    _normalize_game_id,
    _normalize_team_names,
)
from psycopg import sql

#: The four split columns of each market; a market is rewritten as a whole.
MARKET_SPLIT_COLUMNS = {
    market: [
        f"{market}_pct_{kind}_{side}" for kind in ("bets", "money") for side in sides
    ]
    for market, sides in (
        ("total", ("over", "under")),
        ("spread", ("away", "home")),
        ("moneyline", ("away", "home")),
    )
}
SPLIT_COLUMNS = [c for columns in MARKET_SPLIT_COLUMNS.values() for c in columns]
LINE_COLUMNS = [
    "spread_home",
    "spread_away",
    "moneyline_home",
    "moneyline_away",
    "total_line",
]
DEFAULT_BACKUP_DIR = Path("data/yahoo_odds/repair_backups")
DEFAULT_PROPOSALS_CSV = Path("/tmp/yahoo_split_repair_proposals.csv")


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return ((a - b).abs() < 1e-6) | (a.isna() & b.isna())


def load_stored_rows(start: date, end: date) -> pd.DataFrame:
    schema = get_schema_name_odds_yahoo()
    query = sql.SQL(
        "SELECT * FROM {}.{} WHERE game_date BETWEEN %s AND %s ORDER BY game_date"
    ).format(sql.Identifier(schema), sql.Identifier(schema))
    conn = connect_nba_db()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (start, end))
            columns = [d.name for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    stored = pd.DataFrame(rows, columns=columns)
    stored["game_id"] = stored["game_id"].astype(str)
    return stored


def scrape_rows(stored: pd.DataFrame) -> pd.DataFrame:
    """Fresh Yahoo rows of the stored games, with their game_id."""
    seasons = sorted(stored["season_year"].astype(int).unique())
    games = pd.concat(
        [load_games_for_yahoo_update(season_year=s) for s in seasons],
        ignore_index=True,
    )
    days = sorted(set(stored["game_date"]))
    fresh = asyncio.run(scrape_yahoo_days(days))
    if fresh.empty:
        return pd.DataFrame(columns=["game_id", *LINE_COLUMNS, *SPLIT_COLUMNS])
    fresh = merge_sportsbook_with_games(
        _normalize_team_names(fresh), _normalize_game_id(games)
    )
    fresh = fresh.dropna(subset=["game_id"])
    fresh["game_id"] = fresh["game_id"].astype(str)
    return fresh


def build_proposals(stored: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """One row per stored game: whether it can be repaired and what changes."""
    merged = stored.merge(
        fresh[["game_id", *LINE_COLUMNS, *SPLIT_COLUMNS]],
        on="game_id",
        how="left",
        suffixes=("", "_new"),
        indicator=True,
    )
    lines_match = pd.Series(True, index=merged.index)
    for column in LINE_COLUMNS:
        lines_match &= _same(merged[column], merged[f"{column}_new"])
    # A market the fresh page gives without all four splits is kept as
    # stored: a missing value is no evidence, and must not overwrite one.
    merged["markets_kept"] = ""
    for market, columns in MARKET_SPLIT_COLUMNS.items():
        incomplete = merged[[f"{c}_new" for c in columns]].isna().any(axis=1)
        for column in columns:
            merged.loc[incomplete, f"{column}_new"] = merged.loc[incomplete, column]
        merged.loc[incomplete, "markets_kept"] += f"{market} "
    merged["markets_kept"] = merged["markets_kept"].str.strip()
    changed = [
        ~_same(merged[column], merged[f"{column}_new"]) for column in SPLIT_COLUMNS
    ]
    merged["n_changed"] = sum(c.astype(int) for c in changed)
    merged["status"] = "unchanged"
    merged.loc[merged["n_changed"] > 0, "status"] = "repair"
    merged.loc[~lines_match, "status"] = "lines differ: skipped"
    merged.loc[merged["_merge"] == "left_only", "status"] = "not re-scraped: skipped"
    return merged.drop(columns="_merge")


def apply_repairs(repairs: pd.DataFrame) -> None:
    schema = get_schema_name_odds_yahoo()
    query = sql.SQL("UPDATE {}.{} SET {} WHERE game_id = %s").format(
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.SQL(", ").join(
            sql.SQL("{} = %s").format(sql.Identifier(c)) for c in SPLIT_COLUMNS
        ),
    )
    values = [
        (
            *(
                None if pd.isna(row[f"{c}_new"]) else float(row[f"{c}_new"])
                for c in SPLIT_COLUMNS
            ),
            row["game_id"],
        )
        for _, row in repairs.iterrows()
    ]
    conn = connect_nba_db()
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(query, values)
    finally:
        conn.close()
    print(f"Updated {len(values)} row(s) in one transaction.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--apply", action="store_true", help="write the repairs")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--proposals-csv", type=Path, default=DEFAULT_PROPOSALS_CSV)
    args = parser.parse_args()

    stored = load_stored_rows(args.start_date, args.end_date)
    print(f"Stored rows {args.start_date} .. {args.end_date}: {len(stored)}")
    if stored.empty:
        return 0
    proposals = build_proposals(stored, scrape_rows(stored))
    proposals.to_csv(args.proposals_csv, index=False)
    print(proposals["status"].value_counts().to_string())
    kept = proposals[
        (proposals["markets_kept"] != "") & ~proposals["status"].str.endswith("skipped")
    ]
    if not kept.empty:
        print(f"{len(kept)} row(s) keep the stored splits of a market Yahoo gave none:")
        print(kept[["game_id", "game_date", "team_away", "team_home", "markets_kept"]])
    print(f"Proposals: {args.proposals_csv}")

    repairs = proposals[proposals["status"] == "repair"]
    for column in SPLIT_COLUMNS:
        n = int((~_same(repairs[column], repairs[f"{column}_new"])).sum())
        print(f"  {column:26s} {n} row(s) change")
    skipped = proposals[proposals["status"].str.endswith("skipped")]
    if not skipped.empty:
        print(skipped[["game_id", "game_date", "team_away", "team_home", "status"]])

    if not args.apply or repairs.empty:
        print("Dry run: nothing written." if not args.apply else "Nothing to do.")
        return 0

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = (
        args.backup_dir
        / f"odds_yahoo_{args.start_date}_{args.end_date}_{stamp}.parquet"
    )
    stored[stored["game_id"].isin(repairs["game_id"])].to_parquet(backup, index=False)
    print(f"Backup of the {len(repairs)} row(s) to repair: {backup}")

    apply_repairs(repairs)

    after = load_stored_rows(args.start_date, args.end_date).set_index("game_id")
    expected = repairs.set_index("game_id")
    wrong = [
        game_id
        for game_id in expected.index
        if not all(
            _same(
                pd.Series([after.at[game_id, c]]),
                pd.Series([expected.at[game_id, f"{c}_new"]]),
            ).iloc[0]
            for c in SPLIT_COLUMNS
        )
    ]
    if wrong:
        print(f"Read-back mismatch for {len(wrong)} row(s): {wrong}")
        return 1
    print(f"Read back: all {len(expected)} repaired row(s) hold the new splits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
