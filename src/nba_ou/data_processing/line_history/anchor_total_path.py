"""Five compact Bet365 total-line path signals, all as-of the snapshot.

These are exported only for the anchor total. Expanding each to every market
and book would add dozens of mostly redundant columns to the training CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.postgre_db.line_history_aiven.fetch import MARKET_TOTALS

from .cross_book import PEER_GAP_LIMITS
from .snapshots import build_snapshot_panel, resolve_line

PATH_WINDOW_MINUTES = 60
MAX_INTERMOVE_GAP_MINUTES = 720.0


def add_anchor_total_path_features(
    panel: pd.DataFrame, ticks: pd.DataFrame, *, anchor: str
) -> pd.DataFrame:
    """Return one row per anchor total snapshot, with five new features.

    Changes are changes in the raw line level. Repricing at the same level
    leaves the move clock untouched. The 60-minute path sums absolute changes
    and therefore distinguishes a round trip from no activity, even though
    both have zero net ``move_last_60``. The signed streak counts consecutive
    moves in the latest direction *within that window*. A gap of 720 minutes
    means fewer than two prior moves or a longer gap.

    Peer-change state uses the median peer line now and 60 minutes earlier,
    restricted to peers quoted at both instants. It is +1/-1 when that median
    rose/fell by at least half a point while the anchor level did not move.
    """
    prefix = f"ODDS_SNAP_TOT_{anchor.upper()}_"
    feature_names = (
        prefix + "MINUTES_SINCE_LAST_LEVEL_MOVE",
        prefix + "PEERS_MOVED_ANCHOR_STILL_60",
        prefix + "ABS_LEVEL_PATH_60",
        prefix + "SIGNED_MOVE_STREAK_60",
        prefix + "LAST_TWO_LEVEL_MOVES_GAP_MIN",
    )
    keys = ["game_id", "snapshot_minutes"]
    totals = panel[panel["market"].eq(MARKET_TOTALS)].copy()
    if totals.empty:
        return pd.DataFrame(columns=[*keys, *feature_names])
    if "move_last_60" not in totals or "has_window_60" not in totals:
        # Custom --windows grids may omit 60. Resolve it internally for this
        # one feature rather than exporting an entire extra window family.
        earlier = build_snapshot_panel(
            ticks[ticks["market"].eq(MARKET_TOTALS)],
            grid=tuple(
                sorted(set(totals["snapshot_minutes"] + PATH_WINDOW_MINUTES))
            ),
        )
        earlier = earlier[[*keys, "book", "level"]].rename(
            columns={"level": "previous_level"}
        )
        earlier["snapshot_minutes"] -= PATH_WINDOW_MINUTES
        totals = totals.merge(earlier, on=[*keys, "book"], how="left")
        totals["has_window_60"] = totals["previous_level"].notna().astype(int)
        totals["move_last_60"] = (
            totals["level"] - totals["previous_level"]
        ).fillna(0.0)

    anchor_panel = totals[
        totals["book"].eq(anchor) & totals["level"].notna()
    ][keys + ["level", "move_last_60", "has_window_60"]].copy()
    if anchor_panel.empty:
        return pd.DataFrame(columns=[*keys, *feature_names])
    if anchor_panel.duplicated(keys).any():
        raise ValueError("Duplicate anchor total snapshot")

    source = ticks[ticks["market"].eq(MARKET_TOTALS) & ticks["book"].eq(anchor)].copy()
    source["level"] = resolve_line(source)
    source = source[source["level"].notna()].sort_values(
        ["game_id", "minutes_before_tip"], ascending=[True, False], kind="stable"
    )
    tick_groups = source.groupby("game_id", sort=False).indices
    results = []
    for game, positions in anchor_panel.groupby("game_id", sort=False).indices.items():
        game_ticks = source.iloc[tick_groups[game]]
        times = game_ticks["minutes_before_tip"].to_numpy(float)
        values = game_ticks["level"].to_numpy(float)
        deltas = np.diff(values)
        changed = np.flatnonzero(np.isfinite(deltas) & (deltas != 0.0)) + 1
        move_times = times[changed]
        move_deltas = deltas[changed - 1]
        for row_index in positions:
            horizon = float(anchor_panel.iloc[row_index]["snapshot_minutes"])
            seen = move_times >= horizon
            seen_times = move_times[seen]
            seen_deltas = move_deltas[seen]
            age = (seen_times[-1] if len(seen_times) else times[0]) - horizon
            recent = seen_times <= horizon + PATH_WINDOW_MINUTES
            recent_deltas = seen_deltas[recent]
            path = float(np.abs(recent_deltas).sum())
            streak = 0
            if len(recent_deltas):
                direction = int(np.sign(recent_deltas[-1]))
                for delta in recent_deltas[::-1]:
                    if int(np.sign(delta)) != direction:
                        break
                    streak += direction
            gap = (
                min(float(seen_times[-2] - seen_times[-1]), MAX_INTERMOVE_GAP_MINUTES)
                if len(seen_times) >= 2
                else MAX_INTERMOVE_GAP_MINUTES
            )
            results.append((game, int(horizon), age, path, streak, gap))

    path = pd.DataFrame(
        results,
        columns=[
            *keys,
            feature_names[0],
            feature_names[2],
            feature_names[3],
            feature_names[4],
        ],
    )

    peers = totals[totals["book"].ne(anchor)][
        keys + ["level", "move_last_60", "has_window_60"]
    ].copy()
    peers = peers[peers["has_window_60"].eq(1) & peers["level"].notna()]
    peers["previous_level"] = peers["level"] - peers["move_last_60"]
    # A single anomalous peer must not manufacture a "market moved" event.
    now_median = peers.groupby(keys)["level"].transform("median")
    before_median = peers.groupby(keys)["previous_level"].transform("median")
    limit = PEER_GAP_LIMITS[MARKET_TOTALS]
    peers = peers[
        (peers["level"] - now_median).abs().le(limit)
        & (peers["previous_level"] - before_median).abs().le(limit)
    ]
    peer_medians = peers.groupby(keys).agg(
        now=("level", "median"),
        before=("previous_level", "median"),
        count=("level", "count"),
    )
    peer_move = peer_medians["now"] - peer_medians["before"]
    # Signed: the direction the anchor would have to follow to catch up.
    peer_medians["peer_moved"] = np.sign(peer_move).where(
        peer_medians["count"].ge(2) & peer_move.abs().ge(0.5), 0.0
    )
    anchor_panel = anchor_panel.merge(
        peer_medians[["peer_moved"]].reset_index(), on=keys, how="left"
    )
    anchor_still = anchor_panel["has_window_60"].eq(1) & anchor_panel[
        "move_last_60"
    ].eq(0)
    anchor_panel[feature_names[1]] = (
        anchor_panel["peer_moved"].where(anchor_still, 0.0).fillna(0.0).astype(int)
    )
    return path.merge(
        anchor_panel[keys + [feature_names[1]]], on=keys, validate="one_to_one"
    )
