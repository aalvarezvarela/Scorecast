"""Validate the OddsTrader ``lh_*`` tables: coverage, orientation, SBR parity.

Reads the S3 tables (``build_oddstrader_line_history.py``), ``nba_games`` for
realised margins and, where the seasons overlap, the Aiven SBR store. Writes
nothing; prints four reports:

1. **Coverage**: games with at least one pre-game tick, per season and book.
2. **Freshness**: median minutes between a book's last pre-game tick and tip,
   per season and book. Hours mean the feed froze before the close.
3. **Orientation**, checked against outcomes rather than assumed:
   * the std of (home margin - market home margin) at the close, where the
     market home margin is ``-right_line`` (``right`` = HOME handicap), should
     sit near 13.5; the flipped reading gives about 20;
   * the devigged HOME (``right``) moneyline probability should average ~0.55.
4. **Parity with SBR** for the books both carry: share of (game, market, book)
   with the same line (prices for moneyline) at 5, 60, 240 and 720 minutes.

Usage:
    python scripts/line_history/oddstrader/compare_oddstrader_vs_sbr.py
    python scripts/line_history/oddstrader/compare_oddstrader_vs_sbr.py --seasons 2023 2024
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from nba_ou.fetch_data.odds_oddstrader.store import load_ticks, make_store
from nba_ou.postgre_db.line_history_aiven.fetch import (
    available_seasons,
    fetch_pregame_ticks,
)
from nba_ou.postgre_db.odds_sportsbook_line_history.process_sportsbook_line_history_data import (  # noqa: E501
    load_games_for_line_history_creation,
)

HORIZONS = (5, 60, 240, 720)
KEY = ["game_id", "market", "book"]
QUOTE = ["left_line", "left_price", "right_line", "right_price"]


def asof(ticks: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Last quote at least ``minutes`` before tip, per (game, market, book)."""
    seen = ticks[ticks["minutes_before_tip"] >= minutes].sort_values("line_ts")
    return seen.groupby(KEY).tail(1).set_index(KEY)[QUOTE]


def home_margins() -> pd.Series:
    games = load_games_for_line_history_creation()
    games["game_id"] = games["game_id"].astype(str)
    home = games[games["home"].astype(bool)].set_index("game_id")["pts"]
    away = games[~games["home"].astype(bool)].set_index("game_id")["pts"]
    return (home - away).dropna()


def implied(price: pd.Series) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce")
    return np.where(price < 0, -price / (-price + 100.0), 100.0 / (price + 100.0))


def report_orientation(ticks: pd.DataFrame, margins: pd.Series) -> None:
    close = asof(ticks, 5).reset_index()
    spread = close[close["market"] == "point_spread"].dropna(subset=["right_line"])
    spread = spread.join(margins.rename("margin"), on="game_id").dropna(
        subset=["margin"]
    )
    as_stored = (spread["margin"] + spread["right_line"]).std()
    flipped = (spread["margin"] - spread["right_line"]).std()
    print(
        f"  spread: std(margin - (-right_line)) = {as_stored:.2f}, "
        f"flipped {flipped:.2f}  ({len(spread):,} game-books)"
    )
    ml = close[close["market"] == "money_line"].dropna(
        subset=["left_price", "right_price"]
    )
    left, right = implied(ml["left_price"]), implied(ml["right_price"])
    home = right / (left + right)
    print(f"  moneyline: mean devigged HOME (right) probability = {home.mean():.3f}")


def report_parity(ot: pd.DataFrame, sbr: pd.DataFrame) -> None:
    books = sorted(set(ot["book"]) & set(sbr["book"]))
    if not books:
        print("  no shared books")
        return
    ot, sbr = ot[ot["book"].isin(books)], sbr[sbr["book"].isin(books)]
    for minutes in HORIZONS:
        joined = asof(ot, minutes).join(
            asof(sbr, minutes), lsuffix="_ot", rsuffix="_sbr", how="inner"
        )
        for market in ("totals", "point_spread", "money_line"):
            part = joined.xs(market, level="market")
            cols = ("price",) if market == "money_line" else ("line",)
            same = np.ones(len(part), dtype=bool)
            for side in ("left", "right"):
                for col in cols:
                    same &= (
                        part[f"{side}_{col}_ot"] == part[f"{side}_{col}_sbr"]
                    ).to_numpy()
            by_book = (
                pd.Series(same, index=part.index).groupby(level="book").mean().round(2)
            )
            print(
                f"  T-{minutes:<4} {market:<12} n={len(part):>6,}  same={same.mean():.3f}"
                f"  {by_book.to_dict()}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+")
    parser.add_argument(
        "--top-books", type=int, default=15, help="books shown in the tables"
    )
    args = parser.parse_args()

    store = make_store()
    seasons = args.seasons or store.get_manifest()["seasons"]
    margins = home_margins()
    sbr_seasons = set(available_seasons())

    all_ticks = []
    for season in seasons:
        ticks = load_ticks([season], store=store)
        all_ticks.append(ticks)
        print(
            f"\n=== {season}: {ticks['game_id'].nunique():,} games, {len(ticks):,} ticks"
        )
        report_orientation(ticks, margins)
        if season in sbr_seasons:
            sbr = fetch_pregame_ticks([season])
            sbr = sbr[sbr["game_id"].isin(ticks["game_id"].unique())]
            report_parity(ticks, sbr)

    ticks = pd.concat(all_ticks, ignore_index=True)
    top = ticks.groupby("book")["game_id"].nunique().nlargest(args.top_books).index
    ticks = ticks[ticks["book"].isin(top)]
    print("\n=== Coverage: games with a pre-game tick (any market)")
    print(
        ticks.groupby(["season_year", "book"])["game_id"]
        .nunique()
        .unstack(fill_value=0)
        .to_string()
    )
    print("\n=== Freshness: median minutes from last pre-game tick to tip (totals)")
    last = (
        ticks[ticks["market"] == "totals"]
        .groupby(["season_year", "book", "game_id"])["minutes_before_tip"]
        .min()
    )
    print(last.groupby(["season_year", "book"]).median().unstack().round(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
