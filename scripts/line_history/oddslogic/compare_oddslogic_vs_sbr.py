"""Compare OddsLogic line history with SBR's: coverage and as-of parity.

Reads OddsLogic ticks either from the Aiven schema ``line_history_oddslogic`` or
from a ``--save-dir`` of ``update_oddslogic_line_history.py --dry-run`` (which is
how a season outside the store is checked), and the SBR ticks of the same games
from the Aiven ``line_history`` store. Writes nothing; prints:

1. **Coverage**: games with a pre-game tick per book, in each source.
2. **Freshness**: median minutes from a book's last pre-game tick to tip.
3. **Parity**: for the books both carry, the share of (game, market, book)
   with the same line (prices for moneyline) at 5, 60, 240 and 720 minutes.

Usage:
    python scripts/line_history/oddslogic/compare_oddslogic_vs_sbr.py --seasons 2026
    python scripts/line_history/oddslogic/compare_oddslogic_vs_sbr.py \
        --from-dir /tmp/oddslogic --seasons 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.fetch_data.odds_oddslogic.books import OL_BOOKS
from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven.fetch import (
    DEFAULT_MIN_MINUTES_BEFORE_TIP,
    TICK_COLUMNS,
    _decode_ticks,
    fetch_pregame_ticks,
)
from nba_ou.postgre_db.line_history_aiven.schema import MARKETS
from nba_ou.postgre_db.line_history_oddslogic import store

HORIZONS = (5, 60, 240, 720)
KEY = ["game_id", "market", "book"]
QUOTE = ["left_line", "left_price", "right_line", "right_price"]


def ticks_from_dir(directory: Path) -> pd.DataFrame:
    """The dry run's encoded ``lh_line`` rows -> decoded pre-game ticks."""
    rows = pd.read_parquet(directory / "lh_line.parquet")
    rows = rows[
        rows["is_pregame"].astype(bool)
        & (rows["mins_to_tip"] <= -DEFAULT_MIN_MINUTES_BEFORE_TIP)
    ].copy()
    rows["market"] = rows["market_id"].map(dict(MARKETS))
    rows["book"] = rows["book_id"].map({k: slug for k, (slug, _) in OL_BOOKS.items()})
    rows["minutes_before_tip"] = rows["mins_to_tip"]  # _decode_ticks flips the sign
    return _decode_ticks(rows[TICK_COLUMNS].reset_index(drop=True))


def asof(ticks: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Last quote at least ``minutes`` before tip, per (game, market, book)."""
    seen = ticks[ticks["minutes_before_tip"] >= minutes].sort_values("line_ts")
    return seen.groupby(KEY).tail(1).set_index(KEY)[QUOTE]


def report_coverage(ol: pd.DataFrame, sbr: pd.DataFrame) -> None:
    games = ol["game_id"].nunique()
    table = (
        pd.DataFrame(
            {
                "oddslogic": ol.groupby("book")["game_id"].nunique(),
                "sbr": sbr.groupby("book")["game_id"].nunique(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    print(f"\n=== Coverage: games with a pre-game tick (of {games:,})")
    print(table.to_string())


def report_freshness(ol: pd.DataFrame, sbr: pd.DataFrame) -> None:
    def last(ticks: pd.DataFrame) -> pd.Series:
        totals = ticks[ticks["market"] == "totals"]
        per_game = totals.groupby(["book", "game_id"])["minutes_before_tip"].min()
        return per_game.groupby(level="book").median()

    table = pd.DataFrame({"oddslogic": last(ol), "sbr": last(sbr)}).round(0)
    print("\n=== Freshness: median minutes from last pre-game totals tick to tip")
    print(table.to_string())


def report_parity(ol: pd.DataFrame, sbr: pd.DataFrame) -> None:
    books = sorted(set(ol["book"]) & set(sbr["book"]))
    print(f"\n=== Parity with SBR ({', '.join(books) or 'no shared books'})")
    if not books:
        return
    ol, sbr = ol[ol["book"].isin(books)], sbr[sbr["book"].isin(books)]
    for minutes in HORIZONS:
        joined = asof(ol, minutes).join(
            asof(sbr, minutes), lsuffix="_ol", rsuffix="_sbr", how="inner"
        )
        for market in ("totals", "point_spread", "money_line"):
            if market not in joined.index.get_level_values("market"):
                continue
            part = joined.xs(market, level="market")
            col = "price" if market == "money_line" else "line"
            same = np.ones(len(part), dtype=bool)
            for side in ("left", "right"):
                # Prices are nullable integers: a missing quote is not a match.
                same &= (
                    part[f"{side}_{col}_ol"]
                    .eq(part[f"{side}_{col}_sbr"])
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )
            by_book = (
                pd.Series(same, index=part.index).groupby(level="book").mean().round(2)
            )
            print(
                f"  T-{minutes:<4} {market:<12} n={len(part):>6,}  same={same.mean():.3f}"
                f"  {by_book.to_dict()}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument(
        "--from-dir",
        type=Path,
        help="read OddsLogic rows from a dry run's --save-dir instead of Aiven",
    )
    args = parser.parse_args()

    if args.from_dir:
        ol = ticks_from_dir(args.from_dir)
    else:
        with connect_line_history_db() as conn:
            ol = store.fetch_pregame_ticks(args.seasons, conn)
    if ol.empty:
        print("No OddsLogic ticks.")
        return 1
    sbr = fetch_pregame_ticks(args.seasons, covers_fallback=False)
    sbr = sbr[sbr["game_id"].isin(ol["game_id"].unique())]
    print(
        f"{ol['game_id'].nunique():,} OddsLogic games ({len(ol):,} ticks); "
        f"SBR has {sbr['game_id'].nunique():,} of them ({len(sbr):,} ticks)"
    )
    report_coverage(ol, sbr)
    report_freshness(ol, sbr)
    report_parity(ol, sbr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
