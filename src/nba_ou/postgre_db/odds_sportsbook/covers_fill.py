"""Fill an ``odds_sportsbook`` row's non-book columns from the Covers store.

The close repair (``repair_closes``) rewrites the per-book closes from line
history, which the Covers fallback now supplies where SBR has none. Two kinds
of column are not per-book closes and so stay as SBR's day page left them:

* the consensus pick percentages (``*_consensus_pct_*``): SBR's are the
  Covers Consensus shares (``parse.PickConsensus``), which the Covers store
  keeps as counts in ``lh_consensus``;
* the consensus opener (``*_consensus_opener_*``): BetMGM's first quote, which
  the Covers store keeps as the synthetic book ``consensus_opener``.

Only what SBR lacks is filled: a percentage cell that is empty, and an opener
market whose prices are not both there (a frozen day page leaves ``0.0``
lines with no price). SBR's own values are never replaced.
"""

from __future__ import annotations

import pandas as pd
import psycopg
from nba_ou.fetch_data.odds_covers.parse import PickConsensus

#: odds_sportsbook column -> ``PickConsensus.sbr_consensus`` key.
PCT_COLUMNS: dict[str, str] = {
    "total_consensus_pct_over": "overPickPercent",
    "total_consensus_pct_under": "underPickPercent",
    "spread_consensus_pct_home": "homeSpreadPickPercent",
    "spread_consensus_pct_away": "awaySpreadPickPercent",
}

#: market -> (prefix, (column suffix, tick field) ...); the left side is OVER /
#: AWAY, as in the line history.
_OPENER_SIDES: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "totals": (
        "total",
        (
            ("line_over", "left_line"),
            ("price_over", "left_price"),
            ("line_under", "right_line"),
            ("price_under", "right_price"),
        ),
    ),
    "point_spread": (
        "spread",
        (
            ("line_away", "left_line"),
            ("price_away", "left_price"),
            ("line_home", "right_line"),
            ("price_home", "right_price"),
        ),
    ),
    "money_line": ("ml", (("price_away", "left_price"), ("price_home", "right_price"))),
}

OPENER_BOOK = "consensus_opener"


def opener_columns(market: str) -> dict[str, str]:
    """odds_sportsbook column -> tick field for one market's opener."""
    prefix, sides = _OPENER_SIDES[market]
    return {f"{prefix}_{OPENER_BOOK}_{suffix}": field for suffix, field in sides}


FILL_COLUMNS: list[str] = list(PCT_COLUMNS) + [
    column for market in _OPENER_SIDES for column in opener_columns(market)
]


def _rendered_pct(value: float) -> float | None:
    """SBR's day-page rendering: a rounded integer, nothing for 0."""
    return None if not value else float(int(float(value) + 0.5))


def pick_percentages(consensus: pd.DataFrame) -> pd.DataFrame:
    """``lh_consensus`` rows -> game_id and SBR's four percentage columns."""
    rows = []
    for row in consensus.itertuples(index=False):
        shares = PickConsensus(
            spread_away=int(row.spread_away_picks),
            spread_home=int(row.spread_home_picks),
            total_over=int(row.total_over_picks),
            total_under=int(row.total_under_picks),
        ).sbr_consensus()
        out = {"game_id": str(row.game_id)}
        for column, key in PCT_COLUMNS.items():
            out[column] = _rendered_pct(shares[key])
        rows.append(out)
    return pd.DataFrame(rows, columns=["game_id", *PCT_COLUMNS])


def opener_quotes(ticks: pd.DataFrame) -> pd.DataFrame:
    """``consensus_opener`` ticks -> game_id and the opener columns, one row a
    game. The earliest tick per market is the opener."""
    if ticks.empty:
        return pd.DataFrame(columns=["game_id"])
    ordered = ticks.sort_values("line_ts", kind="stable")
    first = ordered.drop_duplicates(["game_id", "market"], keep="first")
    out: dict[str, dict[str, object]] = {}
    for tick in first.itertuples(index=False):
        if tick.market not in _OPENER_SIDES:
            continue
        row = out.setdefault(str(tick.game_id), {"game_id": str(tick.game_id)})
        for column, field in opener_columns(tick.market).items():
            row[column] = getattr(tick, field)
    return pd.DataFrame(list(out.values()))


def fill_from_covers(
    rows: pd.DataFrame, percentages: pd.DataFrame, openers: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """``rows`` with what SBR lacks taken from Covers. Pure.

    Returns the rows and, per kind (``pct``, ``opener``), the game ids that got
    at least one Covers value.
    """
    out = rows.copy()
    out["game_id"] = out["game_id"].astype(str)
    filled: dict[str, set[str]] = {"pct": set(), "opener": set()}

    pct = percentages.drop_duplicates("game_id").set_index("game_id")
    for column in PCT_COLUMNS:
        if column not in out.columns or column not in pct.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        covers = out["game_id"].map(pct[column])
        take = values.isna() & covers.notna()
        out[column] = values.where(~take, covers)
        filled["pct"] |= set(out.loc[take, "game_id"])

    opener = openers.drop_duplicates("game_id").set_index("game_id")
    for market in _OPENER_SIDES:
        columns = opener_columns(market)
        if not set(columns) <= set(out.columns) or not set(columns) <= set(
            opener.columns
        ):
            continue
        prices = [c for c, field in columns.items() if field.endswith("_price")]
        sbr_ok = out[prices].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        has_covers = out["game_id"].map(opener[prices[0]]).notna()
        take = ~sbr_ok & has_covers
        for column in columns:
            values = pd.to_numeric(out[column], errors="coerce")
            out[column] = values.where(~take, out["game_id"].map(opener[column]))
        filled["opener"] |= set(out.loc[take, "game_id"])
    return out, {kind: sorted(ids) for kind, ids in filled.items()}


def load_covers_fill(game_ids: list[str], conn: psycopg.Connection | None = None):
    """(percentages, openers) of ``game_ids`` from the Covers store; empty
    frames when the store or its consensus table is not there."""
    from nba_ou.postgre_db.line_history_aiven import fetch as sbr_fetch
    from nba_ou.postgre_db.line_history_covers import schema as covers_schema
    from nba_ou.postgre_db.line_history_covers import store as covers_store

    def read(c):
        if not covers_schema.schema_exists(c):
            return pd.DataFrame(columns=["game_id"]), pd.DataFrame(columns=["game_id"])
        percentages = (
            pick_percentages(covers_store.fetch_consensus(c, game_ids=game_ids))
            if covers_schema.consensus_table_exists(c)
            else pd.DataFrame(columns=["game_id"])
        )
        openers = opener_quotes(covers_store.fetch_book_ticks(c, game_ids, OPENER_BOOK))
        return percentages, openers

    if conn is not None:
        return read(conn)
    with sbr_fetch.connect_line_history_db() as owned:
        return read(owned)
