"""One type for ``player_id`` across the ingest: int64, the store's BIGINT.

Every timeline step keys on ``(game_id, player_id)``. pandas treats ``2544``
and ``"2544"`` as two keys, so a frame mixing them splits one player in two:
the half missing from a report reads as a removal, and the store, where both
halves are the same BIGINT, keeps whichever row it inserts first. The
incremental ingest did exactly that -- spans read back from Aiven are
integers, the rosters in Supabase hold text -- and turned unchanged Out
players into Out / not listed / Out flicker.

Resolution therefore returns int64 (:func:`as_player_ids`) and the steps that
join on the key refuse anything else (:func:`require_player_ids`).
"""

from __future__ import annotations

import pandas as pd

PLAYER_ID_DTYPE = "int64"


def as_player_ids(values) -> pd.Series:
    """NBA person ids as int64; raises on a missing or non-integral id."""
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any():
        raise ValueError("player_id has missing values")
    if not numeric.eq(numeric.round()).all():
        raise ValueError("player_id has non-integral values")
    return numeric.astype(PLAYER_ID_DTYPE)


def require_player_ids(frame: pd.DataFrame, step: str) -> None:
    """Raise unless ``frame.player_id`` is int64 (empty frames pass)."""
    if frame.empty or "player_id" not in frame:
        return
    dtype = frame["player_id"].dtype
    if dtype != PLAYER_ID_DTYPE:
        raise TypeError(
            f"{step}: player_id must be {PLAYER_ID_DTYPE}, got {dtype}; a mixed "
            "key splits one player into two"
        )


def require_unique(frame: pd.DataFrame, keys: list[str], table: str) -> None:
    """Raise if ``frame`` repeats a primary key of ``table``.

    Each writer deletes a game's rows before inserting them, so a repeated key
    within one insert is always a bug; ``ON CONFLICT DO NOTHING`` would keep
    one of the rows silently.
    """
    repeated = frame.duplicated(subset=keys, keep=False)
    if repeated.any():
        sample = frame.loc[repeated, keys].head(5).to_dict("records")
        raise ValueError(
            f"{int(repeated.sum())} rows repeat a {table} primary key: {sample}"
        )
