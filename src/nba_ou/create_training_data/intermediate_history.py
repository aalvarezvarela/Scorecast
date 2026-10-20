"""Finished games' snapshot side, kept per season for tonight-only builds.

A tonight-only intermediate build (``create_intermediate_line_df(
target_game_ids=...)``) computes the snapshot side -- panel, movement,
consensus, pivots -- from tonight's ticks alone: every value there is a
function of one game's own ticks. Four later stages also read other games,
and from those games only a few values per row or per game:

* the prior-game line dynamics: each finished game's open-to-close summary
  (``build_game_line_dynamics``), rolled forward per team;
* the walk-forward Ridge move-to-close: the snapshot columns its design reads
  and each game's closing line, over the current and previous season;
* the news beta (G2): each row's 60-minute consensus move;
* the cross-market league reference (G3): each row's moneyline-minus-spread gap.

A ``SeasonSnapshotHistory`` holds exactly those for one season's games, plus
what decides the wide frame's columns and dtypes (``_build_snapshot_side``'s
``schema``). It is computed by the same functions a full build runs, on the
season's ticks, and kept in the history memo keyed on the code, the build
settings and the season's server fingerprint of its ticks, so a season whose
ticks change -- a backfill, a repair, a new day's games -- is rebuilt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_ou.create_training_data import history_memo as memo
from nba_ou.create_training_data.historical_ridge_movement import (
    MARKET_DESIGNS,
    current_line_column,
)
from nba_ou.data_processing.line_history.history_features import (
    build_game_line_dynamics,
)
from nba_ou.data_processing.line_history.market_dynamics import (
    GAP_COLUMN,
    GAP_MOVE_COLUMN,
    LevelReader,
    snapshot_history_inputs,
    tick_levels,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)

MEMO_NAME = "intermediate_snapshot_history"
DYNAMICS_MARKETS = (MARKET_TOTALS, MARKET_SPREAD, MARKET_MONEYLINE)


def ridge_design_columns(anchor: str) -> list[str]:
    """Every snapshot column a Ridge design or its label reads, all markets.

    The cross-market gap columns come from ``snapshot_history_inputs``
    instead (``gap_columns``): they are G3 outputs, not snapshot columns.
    """
    upper = anchor.upper()
    columns: list[str] = ["target_line"]
    for market, design in MARKET_DESIGNS.items():
        book = f"ODDS_SNAP_{design.short}_{upper}_"
        consensus = f"ODDS_SNAP_{design.short}_CONSENSUS_"
        names = [
            consensus + name.removeprefix("CONSENSUS:")
            if name.startswith("CONSENSUS:")
            else book + name
            for name, _ in design.own_scales
        ]
        names += [template.format(anchor=upper) for template, _ in design.cross_scales]
        if design.gap_terms:
            names.append(f"ODDS_SNAP_SPR_{upper}_LEVEL")
        line = current_line_column(market, anchor)
        if market != MARKET_TOTALS:
            names.append(line)
        columns += names
    gap = set(gap_columns(anchor).values())
    return [c for c in dict.fromkeys(columns) if c not in gap]


def gap_columns(anchor: str) -> dict[str, str]:
    """``snapshot_history_inputs`` columns -> the G3 feature names they equal."""
    prefix = f"ODDS_SNAP_XMKT_{anchor.upper()}_ML_MARGIN_MINUS_SPREAD"
    return {GAP_COLUMN: prefix, GAP_MOVE_COLUMN: f"{prefix}_MOVE_60"}


@dataclass
class SeasonSnapshotHistory:
    """One season's games, as the cross-row stages and the schema read them.

    ``rows``: one per (game, output snapshot) of the wide frame, in its order
    -- ``game_id``, ``snapshot_minutes``, ``target_spread_line``, the Ridge
    design columns and the ``snapshot_history_inputs`` columns.
    ``ridge_closing_lines`` and ``dynamics`` are per market; ``combos``,
    ``consensus_markets`` and ``dtypes`` are ``_SnapshotSide``'s.
    """

    rows: pd.DataFrame
    ridge_closing_lines: dict[str, pd.DataFrame]
    dynamics: dict[str, pd.DataFrame]
    combos: pd.DataFrame
    consensus_markets: pd.DataFrame
    dtypes: dict[str, str]


def season_snapshot_history(
    ticks: pd.DataFrame,
    *,
    build_side: Callable[[pd.DataFrame], object],
    anchor: str,
) -> SeasonSnapshotHistory:
    """Compute one season's entry from its ticks, by the full build's code.

    ``build_side`` is the builder's ``_build_snapshot_side`` with the build's
    settings bound; ``ticks`` are the season's, as the full build reads them.
    """
    side = build_side(ticks)
    wide = side.wide
    keep = [
        c for c in ("target_spread_line", *ridge_design_columns(anchor)) if c in wide
    ]
    rows = wide[["game_id", "snapshot_minutes", *keep]].reset_index(drop=True)
    reader = LevelReader(tick_levels(ticks))
    inputs = snapshot_history_inputs(
        reader,
        rows.rename(
            columns={"game_id": "GAME_ID", "snapshot_minutes": "TIME_TO_MATCH_MIN"}
        ),
        anchor=anchor,
    )
    rows = pd.concat([rows, inputs], axis=1)
    return SeasonSnapshotHistory(
        rows=rows,
        ridge_closing_lines=side.ridge_closing_lines,
        dynamics={
            market: build_game_line_dynamics(ticks, market=market)
            for market in DYNAMICS_MARKETS
        },
        combos=side.combos,
        consensus_markets=side.consensus_markets,
        dtypes=side.dtypes,
    )


def settings_key(**settings) -> str:
    """The build settings an entry depends on, as one stable string."""
    return repr(sorted((name, repr(value)) for name, value in settings.items()))


def load_season_histories(
    seasons: list[int],
    *,
    fingerprints: dict[int, list],
    settings: str,
    fetch_season_ticks: Callable[[int], pd.DataFrame],
    build_side: Callable[[pd.DataFrame], object],
    anchor: str,
    verbose: bool = True,
) -> dict[int, SeasonSnapshotHistory]:
    """Every season's entry: from the memo where its digest matches, else built.

    A season's digest covers ``settings`` and the server fingerprint of its
    ticks; the memo's code key covers the builder's code. Without a memo store every season is
    built, which gives the same values, only slower.
    """
    digests = {
        season: hashlib.sha256(
            f"{settings}|{fingerprints[season]!r}".encode()
        ).hexdigest()
        for season in seasons
    }
    backend = memo.default_backend()
    history_memo = (
        memo.HistoryMemo(
            backend,
            digests,
            max(seasons),
            # ``build_side`` and the ticks come from the builder.
            code=memo.code_key(memo.INTERMEDIATE_BUILDER),
        )
        if backend is not None
        else None
    )
    if history_memo is not None:
        history_memo.prefetch([MEMO_NAME])
    out, built = {}, []
    for season in seasons:
        table = history_memo.table(MEMO_NAME, season) if history_memo else None
        entry = table.get("value") if table is not None else None
        if entry is None:
            entry = season_snapshot_history(
                fetch_season_ticks(season), build_side=build_side, anchor=anchor
            )
            built.append(season)
            if table is not None:
                table["value"] = entry
        out[season] = entry
    if history_memo is not None:
        history_memo.flush()
    if verbose:
        print(
            f"Snapshot history: {len(seasons) - len(built)} season(s) stored, "
            f"built {built or 'none'}"
        )
    return out


@dataclass
class SnapshotHistory:
    """The seasons' entries for the games a build holds besides its targets."""

    rows: pd.DataFrame
    ridge_closing_lines: dict[str, pd.DataFrame]
    dynamics: dict[str, pd.DataFrame]
    schema: tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]


def assemble_history(
    histories: dict[int, SeasonSnapshotHistory],
    *,
    keep_game: Callable[[pd.Series], pd.Series],
    history_dtypes: Callable[[list[dict[str, str]]], dict[str, str]],
    by_game_id: bool = False,
) -> SnapshotHistory:
    """Concatenate the seasons, in season order, keeping ``keep_game``'s games.

    The schema's order is the full build's panel order, the order its pivots
    meet the columns in: its ticks come one season after another, or, with
    ``by_game_id``, sorted by game across seasons (a live build's).
    """
    seasons = sorted(histories)

    def concat(frames, column="game_id"):
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        return frame[keep_game(frame[column].astype(str))].reset_index(drop=True)

    rows = concat(histories[s].rows for s in seasons)
    markets = histories[seasons[0]].ridge_closing_lines.keys() if seasons else ()
    closing = {
        market: concat(
            (histories[s].ridge_closing_lines[market] for s in seasons), "GAME_ID"
        )
        for market in markets
    }
    dynamics = {
        market: concat(histories[s].dynamics[market] for s in seasons)
        for market in DYNAMICS_MARKETS
    }
    combos = concat(histories[s].combos for s in seasons)
    consensus_markets = concat(histories[s].consensus_markets for s in seasons)
    if by_game_id:
        combos = combos.sort_values("game_id", kind="stable", ignore_index=True)
        consensus_markets = consensus_markets.sort_values(
            "game_id", kind="stable", ignore_index=True
        )
    schema = (
        combos,
        consensus_markets,
        history_dtypes([histories[s].dtypes for s in seasons]),
    )
    return SnapshotHistory(
        rows=rows, ridge_closing_lines=closing, dynamics=dynamics, schema=schema
    )


def order_key(game_ids: pd.Series, minutes: pd.Series, base_order: dict) -> np.ndarray:
    """The full build's row order: base game order, then snapshot minutes."""
    position = game_ids.astype(str).map(base_order)
    if position.isna().any():
        raise KeyError("rows of games the base does not hold")
    return position.to_numpy("int64") * 100_000 + pd.to_numeric(minutes).to_numpy(
        "int64"
    )
