"""Loader results come out in the same order whatever order the server sends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.postgre_db.row_order import canonical_row_order, final_order_by

from .frame_identity import assert_identical


def _frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "game_date": pd.to_datetime("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 20, n), unit="D"),
            "game_id": [f"00{v:06d}" for v in rng.integers(0, 60, n)],
            "player_id": pd.Series(
                [None if v == 0 else str(v) for v in rng.integers(0, 9, n)],
                dtype=object,
            ),
            "minutes": np.where(rng.random(n) < 0.1, np.nan, rng.normal(20, 8, n)),
            "status": pd.Series(
                rng.choice(["out", "questionable", None], n), dtype=object
            ),
            "listed": rng.random(n) < 0.5,
        }
    )


def _server_order(df: pd.DataFrame, by: tuple[str, ...], seed: int) -> pd.DataFrame:
    """Another order a server could return: sorted by ``by``, ties anywhere."""
    shuffled = df.sample(frac=1.0, random_state=seed)
    if by:
        shuffled = shuffled.sort_values(
            list(by), ascending=[False] * len(by), kind="mergesort"
        )
    return shuffled.reset_index(drop=True)


@pytest.mark.parametrize("by", [(), ("game_date",), ("game_date", "game_id")])
def test_every_server_order_gives_the_same_frame(by):
    df = _frame()
    first = canonical_row_order(_server_order(df, by, seed=0), by)
    for seed in range(1, 6):
        assert_identical(canonical_row_order(_server_order(df, by, seed), by), first)


def test_the_order_by_part_is_kept_as_the_server_sent_it():
    df = _server_order(_frame(), ("game_date",), seed=3)  # descending dates
    out = canonical_row_order(df, ("game_date",))
    assert out["game_date"].tolist() == df["game_date"].tolist()


def test_same_rows_and_dtypes_and_a_fresh_range_index():
    df = _frame()
    df.index = df.index + 1000
    out = canonical_row_order(df)
    assert isinstance(out.index, pd.RangeIndex) and out.index.start == 0
    assert list(out.dtypes) == list(df.dtypes)
    key = lambda f: sorted(map(repr, f.itertuples(index=False)))  # noqa: E731
    assert key(out) == key(df)


def test_identical_rows_are_kept():
    df = pd.concat([_frame(n=50)] * 3, ignore_index=True)
    assert len(canonical_row_order(df)) == 150


def test_small_and_missing_frames():
    assert canonical_row_order(None) is None
    one = _frame(n=1)
    assert_identical(canonical_row_order(one), one)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("SELECT * FROM t", ()),
        ("SELECT * FROM t WHERE x = ANY(%s) ORDER BY game_date DESC", ("game_date",)),
        (
            "SELECT * FROM t ORDER BY game_date DESC, game_id DESC",
            ("game_date", "game_id"),
        ),
        (
            "SELECT a FROM t ORDER BY season_year, fan_rank NULLS LAST LIMIT 5",
            ("season_year", "fan_rank"),
        ),
        ("SELECT l.a, m.b FROM l JOIN m USING (x) ORDER BY l.a, m.b", ("a", "b")),
        # A window ORDER BY is not the result's order.
        ("SELECT a, LAG(a) OVER (ORDER BY ts) FROM t", ()),
        ("WITH c AS (SELECT * FROM t ORDER BY a) SELECT * FROM c", ()),
    ],
)
def test_final_order_by(query, expected):
    assert final_order_by(query) == expected
