"""Compare Covers line history with SBR's: coverage, as-of parity and the opener.

Reads Covers ticks either from the Aiven schema ``line_history_covers`` or from
a ``--save-dir`` of ``update_covers_line_history.py --dry-run`` (which is how a
season outside the store is checked), and the SBR ticks of the same games from
the Aiven ``line_history`` store. Writes nothing; prints:

1. **Coverage**: games with a pre-game tick per book, in each source.
2. **Freshness**: median minutes from a book's last pre-game tick to tip.
3. **Parity**: for the books both carry, the share of (game, market, book)
   with the same line (prices for moneyline) at 5, 60, 240 and 720 minutes,
   and the share with the same opener.
4. **Consensus opener**: the rebuilt ``consensus_opener`` against the one SBR
   stored in ``odds_sportsbook``.

Usage:
    python scripts/line_history/covers/compare_covers_vs_sbr.py --seasons 2026
    python scripts/line_history/covers/compare_covers_vs_sbr.py \
        --from-dir /tmp/covers --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.fetch_data.odds_covers.books import COVERS_BOOKS
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven.fetch import (
    DEFAULT_MIN_MINUTES_BEFORE_TIP,
    TICK_COLUMNS,
    _decode_ticks,
    fetch_pregame_ticks,
)
from nba_ou.postgre_db.line_history_aiven.schema import MARKETS
from nba_ou.postgre_db.line_history_covers import store
from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db.fetch_data_from_odds_sportsbook_db import (  # noqa: E501
    load_odds_sportsbook_from_db,
)

HORIZONS = (5, 60, 240, 720)
KEY = ["game_id", "market", "book"]
QUOTE = ["left_line", "left_price", "right_line", "right_price"]
MARKET_VALUE = {"totals": "line", "point_spread": "line", "money_line": "price"}


def ticks_from_dir(directory: Path) -> pd.DataFrame:
    """The dry run's encoded ``lh_line`` rows -> decoded pre-game ticks."""
    rows = pd.read_parquet(directory / "lh_line.parquet")
    rows = rows[
        rows["is_pregame"].astype(bool)
        & (rows["mins_to_tip"] <= -DEFAULT_MIN_MINUTES_BEFORE_TIP)
    ].copy()
    rows["market"] = rows["market_id"].map(dict(MARKETS))
    rows["book"] = rows["book_id"].map(
        {k: slug for k, (slug, _) in COVERS_BOOKS.items()}
    )
    rows["minutes_before_tip"] = rows["mins_to_tip"]  # _decode_ticks flips the sign
    return _decode_ticks(rows[TICK_COLUMNS].reset_index(drop=True))


def asof(ticks: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Last quote at least ``minutes`` before tip, per (game, market, book)."""
    seen = ticks[ticks["minutes_before_tip"] >= minutes].sort_values("line_ts")
    return seen.groupby(KEY).tail(1).set_index(KEY)[QUOTE]


def openers(ticks: pd.DataFrame) -> pd.DataFrame:
    return ticks[ticks["is_opener"]].drop_duplicates(KEY).set_index(KEY)[QUOTE]


def _same(joined: pd.DataFrame, market: str) -> pd.Series:
    col = MARKET_VALUE[market]
    same = np.ones(len(joined), dtype=bool)
    for side in ("left", "right"):
        ours, theirs = joined[f"{side}_{col}_cov"], joined[f"{side}_{col}_sbr"]
        # A line both stores nulled (``implausible_pregame_line``) is the same
        # quote; a value on one side only is not.
        same &= (
            ours.eq(theirs).fillna(False) | (ours.isna() & theirs.isna())
        ).to_numpy(dtype=bool)
    return pd.Series(same, index=joined.index)


def report_coverage(cov: pd.DataFrame, sbr: pd.DataFrame) -> None:
    games = cov["game_id"].nunique()
    table = (
        pd.DataFrame(
            {
                "covers": cov.groupby("book")["game_id"].nunique(),
                "sbr": sbr.groupby("book")["game_id"].nunique(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    print(f"\n=== Coverage: games with a pre-game tick (of {games:,})")
    print(table.to_string())


def report_freshness(cov: pd.DataFrame, sbr: pd.DataFrame) -> None:
    def last(ticks: pd.DataFrame) -> pd.Series:
        totals = ticks[ticks["market"] == "totals"]
        per_game = totals.groupby(["book", "game_id"])["minutes_before_tip"].min()
        return per_game.groupby(level="book").median()

    table = pd.DataFrame({"covers": last(cov), "sbr": last(sbr)}).round(0)
    print("\n=== Freshness: median minutes from last pre-game totals tick to tip")
    print(table.to_string())


def report_parity(cov: pd.DataFrame, sbr: pd.DataFrame) -> None:
    # SBR stores 2026-27 Caesars as "caesars_odds"; Covers keeps the older slug.
    sbr = sbr.assign(book=sbr["book"].replace({"caesars_odds": "caesars"}))
    books = sorted((set(cov["book"]) & set(sbr["book"])) - {"consensus_opener"})
    print(f"\n=== Parity with SBR ({', '.join(books) or 'no shared books'})")
    if not books:
        return
    cov, sbr = cov[cov["book"].isin(books)], sbr[sbr["book"].isin(books)]
    frames = {f"T-{m}": (asof(cov, m), asof(sbr, m)) for m in HORIZONS}
    frames["opener"] = (openers(cov), openers(sbr))
    for label, (left, right) in frames.items():
        joined = left.join(right, lsuffix="_cov", rsuffix="_sbr", how="inner")
        for market in ("totals", "point_spread", "money_line"):
            if market not in joined.index.get_level_values("market"):
                continue
            same = _same(joined.xs(market, level="market"), market)
            by_book = same.groupby(level="book").mean().round(2)
            print(
                f"  {label:<6} {market:<12} n={len(same):>6,}  same={same.mean():.3f}"
                f"  {by_book.to_dict()}"
            )


def report_consensus(cov: pd.DataFrame, seasons: list[int]) -> None:
    """Rebuilt ``consensus_opener`` against SBR's ``odds_sportsbook`` columns."""
    sbr = load_odds_sportsbook_from_db(
        seasons=[f"{s}-{(s + 1) % 100:02d}" for s in seasons]
    )
    if sbr is None or sbr.empty:
        print("\n=== Consensus opener: no odds_sportsbook rows")
        return
    sbr = sbr.assign(game_id=sbr["game_id"].astype(str)).drop_duplicates("game_id")
    sbr = sbr.set_index("game_id")
    rebuilt = cov[cov["book"] == "consensus_opener"].set_index(["game_id", "market"])
    checks = {
        "totals": [("left_line", "total_consensus_opener_line_over")],
        "point_spread": [("right_line", "spread_consensus_opener_line_home")],
        "money_line": [
            ("left_price", "ml_consensus_opener_price_away"),
            ("right_price", "ml_consensus_opener_price_home"),
        ],
    }
    print("\n=== Consensus opener: rebuilt vs odds_sportsbook (same value)")
    for market, pairs in checks.items():
        if market not in rebuilt.index.get_level_values("market"):
            continue
        part = rebuilt.xs(market, level="market")
        part = part[part.index.isin(sbr.index)]
        same = np.ones(len(part), dtype=bool)
        for ours, theirs in pairs:
            target = pd.to_numeric(sbr.loc[part.index, theirs], errors="coerce")
            same &= part[ours].astype(float).to_numpy() == target.to_numpy()
        print(
            f"  {market:<12} n={len(part):>5,}  same={same.mean() if len(part) else float('nan'):.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument(
        "--from-dir",
        type=Path,
        help="read Covers rows from a dry run's --save-dir instead of Aiven",
    )
    args = parser.parse_args()

    if args.from_dir:
        cov = ticks_from_dir(args.from_dir)
    else:
        with connect_line_history_db() as conn:
            cov = store.fetch_pregame_ticks(args.seasons, conn)
    if cov.empty:
        print("No Covers ticks.")
        return 1
    sbr = fetch_pregame_ticks(args.seasons, covers_fallback=False)
    sbr = sbr[sbr["game_id"].isin(cov["game_id"].unique())]
    print(
        f"{cov['game_id'].nunique():,} Covers games ({len(cov):,} ticks); "
        f"SBR has {sbr['game_id'].nunique():,} of them ({len(sbr):,} ticks)"
    )
    report_coverage(cov, sbr)
    report_freshness(cov, sbr)
    report_parity(cov, sbr)
    report_consensus(cov, args.seasons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
