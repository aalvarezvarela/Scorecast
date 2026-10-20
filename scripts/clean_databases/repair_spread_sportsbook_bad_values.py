"""Repair impossible spread values in the wide odds_sportsbook table.

The normal sportsbook updater only inserts missing games. This script targets
existing rows whose spread columns are structurally impossible, re-scrapes SBR
for only the affected dates, and prepares cell-level updates for review.

Dry-run is the default:

    python scripts/clean_databases/repair_spread_sportsbook_bad_values.py --max-dates 3

Apply only after reviewing the generated proposal CSV:

    python scripts/clean_databases/repair_spread_sportsbook_bad_values.py --apply
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd
from nba_ou.config.constants import TEAM_NAME_STANDARDIZATION
from nba_ou.fetch_data.odds_sportsbook.process_spread_data import SPREAD_BOOKS
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    merge_sportsbook_with_games,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    MARKET_SPREAD,
    discover_games_for_date,
    new_session,
    scrape_events,
)
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_sportsbook,
)
from nba_ou.postgre_db.odds_sportsbook.update_sportsbook.update_database_utils import (
    load_games_for_sportsbook_update,
)
from psycopg import sql

DEFAULT_CANDIDATES_CSV = Path("/tmp/spread_impossible_odds_candidates.csv")
DEFAULT_PROPOSALS_CSV = Path("/tmp/spread_sportsbook_repair_proposals.csv")
MIN_VALID_AMERICAN_ODDS_ABS = 90
MAX_REASONABLE_SPREAD_ABS = 60


def _book_prefix(book: str) -> str:
    if book == "consensus_opener":
        return "spread_consensus_opener"
    return f"spread_{book}"


def _line_home_col(book: str) -> str:
    return f"{_book_prefix(book)}_line_home"


def _line_away_col(book: str) -> str:
    return f"{_book_prefix(book)}_line_away"


def _price_home_col(book: str) -> str:
    return f"{_book_prefix(book)}_price_home"


def _price_away_col(book: str) -> str:
    return f"{_book_prefix(book)}_price_away"


def _is_missing(value: object) -> bool:
    return pd.isna(value)


def _is_valid_spread_pair(line_home: object, line_away: object) -> bool:
    if _is_missing(line_home) or _is_missing(line_away):
        return False
    home = float(line_home)
    away = float(line_away)
    return (
        abs(home) <= MAX_REASONABLE_SPREAD_ABS
        and abs(away) <= MAX_REASONABLE_SPREAD_ABS
        and abs(abs(home) - abs(away)) <= 0.001
        and abs(home + away) <= 0.001
    )


def _is_valid_american_price(value: object) -> bool:
    if _is_missing(value):
        return False
    return abs(float(value)) >= MIN_VALID_AMERICAN_ODDS_ABS


def _normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("team_home", "team_away"):
        if col in out:
            out[col] = out[col].map(
                lambda x: TEAM_NAME_STANDARDIZATION.get(x, x) if pd.notna(x) else x
            )
    return out


def _candidate_query(
    *,
    schema: str,
    table: str,
    season_year_min: int,
    books: Iterable[str],
) -> sql.Composed:
    parts: list[sql.Composed] = []
    for book in books:
        lh = _line_home_col(book)
        la = _line_away_col(book)
        ph = _price_home_col(book)
        pa = _price_away_col(book)
        parts.append(
            sql.SQL(
                """
                SELECT
                    game_id,
                    game_date,
                    season_year,
                    team_home,
                    team_away,
                    {book_literal} AS book,
                    {lh}::float8 AS line_home,
                    {la}::float8 AS line_away,
                    {ph}::float8 AS price_home,
                    {pa}::float8 AS price_away,
                    CASE
                      WHEN {lh} IS NOT NULL AND {lh}::text <> 'NaN'
                           AND abs({lh}) > {max_spread}
                           THEN 'home_line_implausible'
                      WHEN {la} IS NOT NULL AND {la}::text <> 'NaN'
                           AND abs({la}) > {max_spread}
                           THEN 'away_line_implausible'
                      WHEN {lh} IS NOT NULL AND {la} IS NOT NULL
                           AND {lh}::text <> 'NaN' AND {la}::text <> 'NaN'
                           AND abs(abs({lh}) - abs({la})) > 0.001
                           THEN 'abs_mismatch'
                      WHEN {lh} IS NOT NULL AND {la} IS NOT NULL
                           AND {lh}::text <> 'NaN' AND {la}::text <> 'NaN'
                           AND abs(({lh} + {la})) > 0.001
                           THEN 'sign_mismatch'
                      WHEN {ph} IS NOT NULL AND {ph}::text <> 'NaN'
                           AND {ph} > -{min_price_abs} AND {ph} < {min_price_abs}
                           THEN 'home_price_implausible'
                      WHEN {pa} IS NOT NULL AND {pa}::text <> 'NaN'
                           AND {pa} > -{min_price_abs} AND {pa} < {min_price_abs}
                           THEN 'away_price_implausible'
                      ELSE 'unknown'
                    END AS reason
                FROM {schema}.{table}
                WHERE season_year >= {season_year_min}
                  AND (
                    ({lh} IS NOT NULL AND {lh}::text <> 'NaN'
                     AND abs({lh}) > {max_spread})
                    OR ({la} IS NOT NULL AND {la}::text <> 'NaN'
                        AND abs({la}) > {max_spread})
                    OR ({lh} IS NOT NULL AND {la} IS NOT NULL
                        AND {lh}::text <> 'NaN' AND {la}::text <> 'NaN'
                        AND abs(abs({lh}) - abs({la})) > 0.001)
                    OR ({lh} IS NOT NULL AND {la} IS NOT NULL
                        AND {lh}::text <> 'NaN' AND {la}::text <> 'NaN'
                        AND abs(({lh} + {la})) > 0.001)
                    OR ({ph} IS NOT NULL AND {ph}::text <> 'NaN'
                        AND {ph} > -{min_price_abs} AND {ph} < {min_price_abs})
                    OR ({pa} IS NOT NULL AND {pa}::text <> 'NaN'
                        AND {pa} > -{min_price_abs} AND {pa} < {min_price_abs})
                  )
                """
            ).format(
                book_literal=sql.Literal(book),
                lh=sql.Identifier(lh),
                la=sql.Identifier(la),
                ph=sql.Identifier(ph),
                pa=sql.Identifier(pa),
                schema=sql.Identifier(schema),
                table=sql.Identifier(table),
                season_year_min=sql.Literal(season_year_min),
                max_spread=sql.Literal(MAX_REASONABLE_SPREAD_ABS),
                min_price_abs=sql.Literal(MIN_VALID_AMERICAN_ODDS_ABS),
            )
        )

    return sql.SQL(" UNION ALL ").join(parts) + sql.SQL(
        " ORDER BY game_date, game_id, book"
    )


def find_candidates(
    *,
    season_year_min: int,
    books: Iterable[str],
) -> pd.DataFrame:
    schema = get_schema_name_odds_sportsbook()
    table = schema
    conn = connect_nba_db()
    try:
        query = _candidate_query(
            schema=schema,
            table=table,
            season_year_min=season_year_min,
            books=books,
        )
        return pd.read_sql_query(query.as_string(conn), conn)
    finally:
        conn.close()


def _team_key(value: object) -> str:
    if pd.isna(value):
        return ""
    name = str(value).strip()
    return TEAM_NAME_STANDARDIZATION.get(name, name)


def _candidate_match_keys(candidates: pd.DataFrame) -> set[tuple[date, str, str]]:
    dates = pd.to_datetime(candidates["game_date"], errors="coerce").dt.date
    return {
        (day, _team_key(row.team_home), _team_key(row.team_away))
        for day, row in zip(dates, candidates.itertuples(index=False), strict=False)
        if day is not None and not pd.isna(day)
    }


def scrape_spread_dates(
    days: list[date], *, books: Iterable[str], candidates: pd.DataFrame
) -> pd.DataFrame:
    if not days:
        return pd.DataFrame()

    wanted_books = set(books) - {"consensus_opener"}
    target_keys = _candidate_match_keys(candidates)
    scraped_games: list[dict[str, object]] = []
    session = new_session()
    try:
        for day in days:
            print(f"Fetching SBR line history for {day.isoformat()}...", flush=True)
            try:
                summaries = discover_games_for_date(session, day)
                summaries = [
                    summary
                    for summary in summaries
                    if (
                        summary.game_date,
                        _team_key(summary.team_home),
                        _team_key(summary.team_away),
                    )
                    in target_keys
                ]
                print(f"  matched candidate games: {len(summaries)}", flush=True)
                games = scrape_events(
                    [summary.event_id for summary in summaries],
                    session=session,
                    markets=[MARKET_SPREAD],
                )
                for game in games:
                    row: dict[str, object] = {
                        "game_date": game.game_date,
                        "game_id": str(game.event_id),
                        "season_year": game.season_year,
                        "team_home": game.team_home,
                        "team_away": game.team_away,
                    }
                    spread_ticks = game.ticks_for(MARKET_SPREAD)
                    for book in wanted_books:
                        ticks = [
                            tick
                            for tick in spread_ticks
                            if tick.book_slug == book and tick.is_pregame
                        ]
                        if not ticks:
                            continue
                        tick = max(ticks, key=lambda t: t.line_ts)
                        row[_line_away_col(book)] = tick.left_line
                        row[_price_away_col(book)] = tick.left_price
                        row[_line_home_col(book)] = tick.right_line
                        row[_price_home_col(book)] = tick.right_price
                    scraped_games.append(row)
            except Exception as exc:
                print(f"  ! {day}: {exc}", flush=True)
    finally:
        session.close()

    if not scraped_games:
        return pd.DataFrame()
    return pd.DataFrame(scraped_games)


def build_proposals(
    candidates: pd.DataFrame,
    refreshed: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty or refreshed.empty:
        return pd.DataFrame()

    refreshed_by_game = refreshed.set_index(refreshed["game_id"].astype(str))
    proposals: list[dict[str, object]] = []

    for row in candidates.itertuples(index=False):
        game_id = str(row.game_id)
        book = str(row.book)
        if game_id not in refreshed_by_game.index:
            continue

        new_row = refreshed_by_game.loc[game_id]
        if isinstance(new_row, pd.DataFrame):
            new_row = new_row.iloc[0]

        line_cols = [_line_home_col(book), _line_away_col(book)]
        price_cols = [_price_home_col(book), _price_away_col(book)]
        requested_cols: list[str] = []

        if "line" in str(row.reason) or "mismatch" in str(row.reason):
            if _is_valid_spread_pair(
                new_row.get(line_cols[0]), new_row.get(line_cols[1])
            ):
                requested_cols.extend(line_cols)

        if "price" in str(row.reason):
            if all(_is_valid_american_price(new_row.get(col)) for col in price_cols):
                requested_cols.extend(price_cols)

        if not requested_cols:
            continue

        for col in dict.fromkeys(requested_cols):
            old_value = getattr(row, _candidate_value_name(col, book), None)
            new_value = new_row.get(col)
            if _is_missing(new_value):
                continue
            if not _is_missing(old_value) and float(old_value) == float(new_value):
                continue
            proposals.append(
                {
                    "game_id": game_id,
                    "game_date": row.game_date,
                    "team_home": row.team_home,
                    "team_away": row.team_away,
                    "book": book,
                    "reason": row.reason,
                    "column_name": col,
                    "old_value": old_value,
                    "new_value": float(new_value),
                }
            )

    return pd.DataFrame(proposals)


def _candidate_value_name(column_name: str, book: str) -> str:
    if column_name == _line_home_col(book):
        return "line_home"
    if column_name == _line_away_col(book):
        return "line_away"
    if column_name == _price_home_col(book):
        return "price_home"
    if column_name == _price_away_col(book):
        return "price_away"
    raise ValueError(f"Unexpected candidate column: {column_name}")


def apply_proposals(proposals: pd.DataFrame) -> int:
    if proposals.empty:
        return 0

    schema = get_schema_name_odds_sportsbook()
    table = schema
    conn = connect_nba_db()
    updated = 0
    try:
        with conn.cursor() as cur:
            for game_id, group in proposals.groupby("game_id", sort=False):
                setters = []
                values = []
                for proposal in group.itertuples(index=False):
                    setters.append(
                        sql.SQL("{} = %s").format(sql.Identifier(proposal.column_name))
                    )
                    values.append(proposal.new_value)
                values.append(str(game_id))
                query = sql.SQL("UPDATE {}.{} SET {} WHERE game_id = %s").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.SQL(", ").join(setters),
                )
                cur.execute(query, values)
                updated += cur.rowcount
        conn.commit()
        return updated
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _parse_books(raw_books: str | None) -> list[str]:
    if not raw_books:
        return list(SPREAD_BOOKS)
    requested = [book.strip() for book in raw_books.split(",") if book.strip()]
    unknown = sorted(set(requested) - set(SPREAD_BOOKS))
    if unknown:
        raise ValueError(f"Unknown spread book(s): {unknown}")
    return requested


def _filter_dates(
    candidates: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    max_dates: int | None,
) -> pd.DataFrame:
    out = candidates.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if start_date:
        out = out[out["game_date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["game_date"] <= pd.Timestamp(end_date)]
    if max_dates is not None:
        keep_dates = sorted(out["game_date"].dropna().dt.date.unique())[:max_dates]
        out = out[out["game_date"].dt.date.isin(keep_dates)]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write proposed updates")
    parser.add_argument("--identify-only", action="store_true")
    parser.add_argument("--season-year-min", type=int, default=2021)
    parser.add_argument("--books", help="Comma-separated subset of spread books")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--max-dates", type=int)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument("--proposals-csv", type=Path, default=DEFAULT_PROPOSALS_CSV)
    args = parser.parse_args()

    books = _parse_books(args.books)
    candidates = find_candidates(season_year_min=args.season_year_min, books=books)
    candidates = _filter_dates(
        candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dates=args.max_dates,
    )
    candidates.to_csv(args.candidates_csv, index=False)
    print(
        f"Candidates: {len(candidates):,} book/game rows, "
        f"{candidates['game_id'].nunique():,} games, "
        f"{candidates['game_date'].nunique():,} dates"
    )
    print(f"Wrote candidates: {args.candidates_csv}")

    if candidates.empty or args.identify_only:
        if args.identify_only:
            print("Identify only: no SBR scrape and no database writes.")
        return 0

    days = sorted(pd.to_datetime(candidates["game_date"]).dt.date.unique())
    scraped = scrape_spread_dates(days, books=books, candidates=candidates)
    if scraped.empty:
        print("No SBR spread rows scraped; no proposals generated.")
        return 1

    games_df = load_games_for_sportsbook_update(season_year=None)
    scraped = _normalize_team_names(scraped)
    games_df["game_id"] = games_df["game_id"].astype(str)
    refreshed = merge_sportsbook_with_games(scraped, games_df)
    refreshed = refreshed.dropna(subset=["game_id"]).copy()
    refreshed["game_id"] = refreshed["game_id"].astype(str)
    refreshed = refreshed[
        refreshed["game_id"].isin(set(candidates["game_id"].astype(str)))
    ]

    proposals = build_proposals(candidates, refreshed)
    proposals.to_csv(args.proposals_csv, index=False)
    print(
        f"Proposals: {len(proposals):,} cell updates across "
        f"{proposals['game_id'].nunique() if not proposals.empty else 0:,} games"
    )
    print(f"Wrote proposals: {args.proposals_csv}")

    if not args.apply:
        print("DRY RUN: no database writes. Re-run with --apply after review.")
        return 0

    updated_rows = apply_proposals(proposals)
    print(f"Updated DB rows: {updated_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
