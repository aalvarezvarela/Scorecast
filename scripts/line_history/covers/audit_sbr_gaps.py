"""List the past games whose SBR line history is missing or incomplete.

The input of the Covers backfill: every game of ``nba_games`` (regular season,
play-in, playoffs, NBA Cup final) from Covers' first history date up to the
season before the current one, checked against the Aiven ``line_history`` store
and the scraped closing table ``odds_sportsbook``. Writes nothing to a database.

A game is flagged with one or more reasons:

* ``no_game``: not in ``lh_game``.
* ``no_ticks``: in ``lh_game`` without any pre-game tick (at least
  ``DEFAULT_MIN_MINUTES_BEFORE_TIP`` before tip) in any book.
* ``no_main_book``: the main book (``get_main_book()``) has no pre-game tick
  in totals or in spread, while other books have.
* ``truncated``: the freshest pre-game tick across every book and market is
  more than ``TRUNCATED_HISTORY_MINUTES`` before tip (the ingest's freeze
  rule). ``freeze_cluster`` marks the ones whose last tick shares the clock
  (within 15 minutes) with another truncated game of the date: a feed freeze,
  not a quiet market.
* ``late_start``: the history starts less than the deepest snapshot horizon
  (``DEFAULT_SNAPSHOT_GRID``, T-1080) before tip, so the early intermediate
  snapshots have no line.
* ``missing_books``: a book that prices at least ``--expected-share`` of the
  season's games in SBR has no pre-game totals or spread tick for this one.
  A book absent from a whole season is not flagged per game; the season
  coverage table shows it.

The per-book gap is not a reason: SBR stores changes only, and bet365's last
change is over an hour before tip in about 40% of games.

Usage:
    python scripts/line_history/covers/audit_sbr_gaps.py --out /tmp/sbr_gaps.csv
    python scripts/line_history/covers/audit_sbr_gaps.py --seasons 2023 2024
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import get_main_book
from nba_ou.data_processing.line_history.snapshots import DEFAULT_SNAPSHOT_GRID
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven.fetch import (
    DEFAULT_MIN_MINUTES_BEFORE_TIP,
    FETCH_ATTEMPTS,
    TRUNCATED_HISTORY_MINUTES,
)
from nba_ou.postgre_db.line_history_aiven.schema import SCHEMA
from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db.fetch_data_from_odds_sportsbook_db import (  # noqa: E501
    load_odds_sportsbook_from_db,
)
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    load_games_for_line_history_creation,
)
from psycopg import sql

#: First date of Covers' line history (docs/odds/covers_line_history.md).
COVERS_HISTORY_START = date(2019, 4, 1)

#: Books stored by the SBR ingest that are not sportsbooks of their own.
IGNORED_BOOKS = ("caesars_odds",)

FREEZE_CLOCK_MINUTES = 15
MARKETS_CHECKED = ("totals", "point_spread")

AGGREGATE = """
    SELECT l.game_id, m.code AS market, b.slug AS book,
           count(*) FILTER (WHERE l.is_pregame AND l.mins_to_tip <= %(gate)s)
               AS n_pregame,
           max(l.mins_to_tip) FILTER (WHERE l.is_pregame AND l.mins_to_tip <= %(gate)s)
               AS last_pregame,
           min(l.mins_to_tip) FILTER (WHERE l.is_pregame AND l.mins_to_tip <= %(gate)s)
               AS first_pregame
    FROM {schema}.lh_line l
    JOIN {schema}.lh_market m USING (market_id)
    JOIN {schema}.lh_book   b USING (book_id)
    WHERE l.season_year = %(season)s
    GROUP BY 1, 2, 3
"""


def _query(query: sql.Composable, params: dict, label: str) -> pd.DataFrame:
    """One read, retried on a fresh connection: Aiven cuts long transfers."""
    for attempt in range(1, FETCH_ATTEMPTS + 1):
        try:
            with connect_line_history_db() as conn, conn.cursor() as cur:
                cur.execute(query, params)
                columns = [d.name for d in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=columns)
        except Exception as exc:  # noqa: BLE001 - connection drops vary
            if attempt == FETCH_ATTEMPTS:
                raise
            print(f"  {label}: {type(exc).__name__}, retry {attempt}", flush=True)
    raise AssertionError("unreachable")


def load_universe(seasons: list[int]) -> pd.DataFrame:
    games = load_games_for_line_history_creation()
    games = games.drop_duplicates("game_id")[
        ["game_id", "game_date", "season_year", "season_type"]
    ].copy()
    games["game_id"] = games["game_id"].astype(str)
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games["season_year"] = pd.to_numeric(games["season_year"]).astype(int)
    games = games[
        games["season_year"].isin(seasons)
        & (games["game_date"] >= COVERS_HISTORY_START)
    ]
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_sbr(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(game, market, book) pre-game aggregates and ``lh_game`` for ``seasons``."""
    schema = sql.Identifier(SCHEMA)
    frames = []
    for season in seasons:
        frame = _query(
            sql.SQL(AGGREGATE).format(schema=schema),
            {"season": season, "gate": -DEFAULT_MIN_MINUTES_BEFORE_TIP},
            f"season {season}",
        )
        if not frame.empty:
            frames.append(frame)
    agg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    agg = agg[(agg["n_pregame"] > 0) & ~agg["book"].isin(IGNORED_BOOKS)].copy()
    agg["game_id"] = agg["game_id"].astype(str)
    lh_game = _query(
        sql.SQL(
            "SELECT game_id, tipoff_utc FROM {}.lh_game WHERE season_year = ANY(%(s)s)"
        ).format(schema),
        {"s": seasons},
        "lh_game",
    )
    lh_game["game_id"] = lh_game["game_id"].astype(str)
    lh_game["tipoff_utc"] = pd.to_datetime(lh_game["tipoff_utc"], utc=True)
    return agg, lh_game


def _freeze_cluster(truncated: pd.DataFrame) -> pd.Series:
    """Truncated games whose last tick shares the clock with another of the date."""
    clock = truncated["last_tick_utc"].astype("int64") // 60_000_000_000
    flags = pd.Series(False, index=truncated.index)
    for _, idx in truncated.groupby("game_date").groups.items():
        values = clock.loc[idx].to_numpy()
        near = np.abs(values[:, None] - values[None, :]) <= FREEZE_CLOCK_MINUTES
        flags.loc[idx] = near.sum(axis=1) >= 2
    return flags


def classify(
    games: pd.DataFrame,
    agg: pd.DataFrame,
    lh_game: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    main_book: str,
    expected_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-game flags, and the season x book share of games priced in SBR."""
    out = games.merge(lh_game, on="game_id", how="left")
    out["in_lh_game"] = out["game_id"].isin(lh_game["game_id"])

    per_game = agg.groupby("game_id").agg(
        last=("last_pregame", "max"), first=("first_pregame", "min")
    )
    out["last_tick_min_before_tip"] = -out["game_id"].map(per_game["last"])
    out["history_start_min_before_tip"] = -out["game_id"].map(per_game["first"])
    out["last_tick_utc"] = out["tipoff_utc"] - pd.to_timedelta(
        out["last_tick_min_before_tip"], unit="m"
    )

    priced = agg[agg["market"].isin(MARKETS_CHECKED)]
    books = priced.groupby("game_id")["book"].agg(lambda s: set(s))
    out["books"] = (
        out["game_id"].map(books).apply(lambda v: v if isinstance(v, set) else set())
    )
    out["n_books"] = out["books"].str.len()
    main_markets = priced[priced["book"] == main_book].groupby("game_id")["market"]
    main_complete = main_markets.nunique() == len(MARKETS_CHECKED)
    out["main_book_complete"] = (
        out["game_id"].map(main_complete).astype("boolean").fillna(False)
    )

    exploded = out[["season_year", "books"]].explode("books").dropna()
    share = (
        exploded.groupby(["season_year", "books"]).size().unstack(fill_value=0)
    ).div(out.groupby("season_year").size(), axis=0)
    expected = {
        season: {book for book, s in row.items() if s >= expected_share}
        for season, row in share.iterrows()
    }
    out["missing_books"] = [
        ",".join(sorted(expected.get(season, set()) - have)) if have else ""
        for season, have in zip(out["season_year"], out["books"], strict=True)
    ]

    closes = closes.drop_duplicates("game_id").set_index("game_id")
    out["in_odds_sportsbook"] = out["game_id"].isin(closes.index)
    main_close = f"total_{main_book}_line_over"
    out["close_main_book_total"] = out["game_id"].map(closes[main_close]).notna()

    has_ticks = out["n_books"] > 0
    truncated = out["last_tick_min_before_tip"] > TRUNCATED_HISTORY_MINUTES
    out["freeze_cluster"] = False
    out.loc[truncated, "freeze_cluster"] = _freeze_cluster(out[truncated])
    deepest = max(DEFAULT_SNAPSHOT_GRID)
    reasons = {
        "no_game": ~out["in_lh_game"],
        "no_ticks": out["in_lh_game"] & ~has_ticks,
        "no_main_book": has_ticks & ~out["main_book_complete"],
        "truncated": truncated,
        "late_start": out["history_start_min_before_tip"] < deepest,
        "missing_books": has_ticks & (out["missing_books"] != ""),
    }
    out["reasons"] = [
        ";".join(name for name, flags in reasons.items() if flags.iloc[i])
        for i in range(len(out))
    ]
    return out, share


def report(out: pd.DataFrame, share: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\nShare of each season's games with a pre-game totals/spread tick in SBR:")
    print(share.round(3).to_string())

    reasons = out["reasons"].str.get_dummies(sep=";")
    reasons["any"] = (out["reasons"] != "").astype(int)
    by_season = reasons.groupby(out["season_year"]).sum()
    by_season.insert(0, "games", out.groupby("season_year").size())
    by_season.loc["total"] = by_season.sum()
    print("\nFlagged games per season (a game can have several reasons):")
    print(by_season.to_string())

    truncated = out[out["reasons"].str.contains("truncated")]
    if not truncated.empty:
        bins = pd.cut(
            truncated["last_tick_min_before_tip"],
            [TRUNCATED_HISTORY_MINUTES, 120, 180, 360, np.inf],
        )
        print("\nTruncated: minutes from the last pre-game tick to tip")
        print(
            pd.crosstab(
                truncated["season_year"], [bins, truncated["freeze_cluster"]]
            ).to_string()
        )

    stored = out.groupby("season_year")["in_lh_game"].any()
    for season in stored.index[~stored]:
        rows = out[out["season_year"] == season]
        print(
            f"\nSeason {season}: no SBR line history at all ({len(rows)} games, "
            f"{rows['game_date'].min()} .. {rows['game_date'].max()}; "
            f"{int(rows['in_odds_sportsbook'].sum())} with an odds_sportsbook close)"
        )
    empty = out[
        out["season_year"].isin(stored.index[stored])
        & out["reasons"].str.contains("no_game|no_ticks|no_main_book")
    ]
    if not empty.empty:
        print("\nGames without SBR lines for the main book:")
        cols = ["game_id", "game_date", "season_type", "reasons", "n_books"]
        cols += ["in_odds_sportsbook", "close_main_book_total"]
        print(empty[cols].to_string(index=False))


def main() -> int:
    current = season_year_for_date(date.today())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(range(COVERS_HISTORY_START.year - 1, current)),
        help="season start years (default: Covers' first season to the last "
        "finished one)",
    )
    parser.add_argument("--expected-share", type=float, default=0.9)
    parser.add_argument("--out", type=Path, help="write flagged games (.csv/.parquet)")
    args = parser.parse_args()
    if any(s >= current for s in args.seasons):
        parser.error(f"season {current} is the current one; it is not audited")

    main_book = get_main_book()
    games = load_universe(args.seasons)
    print(
        f"{len(games)} games, {games['game_date'].min()} .. {games['game_date'].max()}"
    )
    agg, lh_game = load_sbr(args.seasons)
    closes = load_odds_sportsbook_from_db(
        seasons=[f"{s}-{str(s + 1)[-2:]}" for s in args.seasons]
    )
    closes = closes if closes is not None else pd.DataFrame(columns=["game_id"])
    closes["game_id"] = closes["game_id"].astype(str)

    out, share = classify(
        games,
        agg,
        lh_game,
        closes,
        main_book=main_book,
        expected_share=args.expected_share,
    )
    report(out, share)

    if args.out:
        flagged = out[out["reasons"] != ""].drop(columns=["books"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix == ".parquet":
            flagged.to_parquet(args.out, index=False)
        else:
            flagged.to_csv(args.out, index=False)
        print(f"\nwrote {len(flagged)} flagged games to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
