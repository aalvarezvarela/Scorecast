"""Alternate-line quotes are dropped from the line history of the filtered seasons."""

from __future__ import annotations

import contextlib

import pandas as pd
import pytest
from nba_ou.postgre_db.line_history_aiven import alt_lines, fetch

GAME = "0022600086"
START = pd.Timestamp("2026-10-21 13:00", tz="UTC")


def _tick(minute, book, *, line, left_price=-110, right_price=-110, **extra):
    """One totals tick; ``line`` is the total, or the away handicap of a spread."""
    market = extra.pop("market", "totals")
    right_line = -line if market == "point_spread" else line
    return {
        "game_id": GAME,
        "season_year": extra.pop("season_year", 2026),
        "market": market,
        "book": book,
        "line_ts": START + pd.Timedelta(minutes=minute),
        "minutes_before_tip": 600 - minute,
        "is_opener": False,
        "left_line": line,
        "left_price": left_price,
        "right_line": right_line,
        "right_price": right_price,
        **extra,
    }


def _market(rows, market="totals", main=238.5, season_year=2026):
    """Three books quoting ``main`` at -110, then ``rows``."""
    if market == "point_spread":
        main = 4.5
    base = [
        _tick(0, book, line=main, market=market, season_year=season_year)
        for book in ("betmgm", "fanduel", "draftkings")
    ]
    return pd.DataFrame(base + rows)[fetch.TICK_COLUMNS]


def _kept(ticks):
    return alt_lines.drop_alternate_lines(ticks)


def test_an_unbalanced_alternate_total_is_dropped_and_the_main_line_kept():
    ticks = _market(
        [
            _tick(1, "bet365", line=238.5),
            _tick(5, "bet365", line=231.5, left_price=-245, right_price=172),
            _tick(9, "bet365", line=238.5, left_price=-112, right_price=-108),
        ]
    )
    kept = _kept(ticks)
    bet365 = kept[kept["book"].eq("bet365")]
    assert bet365["left_line"].tolist() == [238.5, 238.5]
    assert len(kept) == len(ticks) - 1


def test_a_pair_of_prices_from_two_lines_is_dropped():
    ticks = _market(
        [
            _tick(1, "bet365", line=240.5, left_price=108, right_price=137),
            _tick(2, "fanduel", line=230.5, left_price=-260, right_price=-105),
        ]
    )
    assert alt_lines.alternate_line_mask(ticks).tolist() == [False] * 3 + [True] * 2


def test_a_near_alternate_spread_priced_for_its_offset_is_dropped():
    # Away +8 against +4.5 elsewhere: the away side is likelier, and its price
    # leans that way (-150/+125, ~57%) without being unbalanced.
    ticks = _market(
        [
            _tick(
                1,
                "bet365",
                line=8.0,
                left_price=-150,
                right_price=125,
                market="point_spread",
            )
        ],
        market="point_spread",
    )
    assert alt_lines.alternate_line_mask(ticks).tolist() == [False] * 3 + [True]


def test_a_book_leading_a_real_move_at_minus_110_is_kept():
    ticks = _market(
        [
            _tick(1, "bet365", line=242.5),
            # Priced against its own offset: not an alternate's price either.
            _tick(2, "fanduel", line=242.0, left_price=-125, right_price=105),
        ]
    )
    assert not alt_lines.alternate_line_mask(ticks).any()


def test_moneyline_earlier_seasons_and_unreadable_prices_are_left_alone():
    alternate = dict(line=231.5, left_price=-245, right_price=172)
    ticks = pd.concat(
        [
            _market(
                [_tick(1, "bet365", season_year=2025, **alternate)], season_year=2025
            ),
            _market(
                [
                    _tick(
                        1,
                        "bet365",
                        line=None,
                        market="money_line",
                        left_price=-400,
                        right_price=300,
                    ),
                    # The old spread price-bleed: a line read as a price.
                    _tick(2, "bet365", line=231.5, left_price=-13, right_price=-12),
                ]
            ),
        ],
        ignore_index=True,
    )
    assert not alt_lines.alternate_line_mask(ticks).any()
    assert alt_lines.drop_alternate_lines(ticks) is ticks


def test_alternates_quoted_by_several_books_do_not_shield_each_other():
    # Two books on 238.5 and three near alternates above it. With every other
    # book in the median, only the furthest (245.0) stands 3+ points off; once
    # it is out, the other two do too.
    ticks = _market(
        [
            _tick(1, "fanduel", line=242.0, left_price=125, right_price=-150),
            _tick(1, "caesars", line=242.5, left_price=120, right_price=-145),
            _tick(1, "hard_rock_bet", line=245.0, left_price=130, right_price=-160),
        ]
    )
    ticks = ticks[~ticks["book"].eq("fanduel") | ticks["left_line"].ne(238.5)]
    mask = alt_lines.alternate_line_mask(ticks)
    assert ticks.loc[mask, "book"].tolist() == ["fanduel", "caesars", "hard_rock_bet"]


def test_the_frame_keeps_its_attributes():
    ticks = _market([_tick(5, "bet365", line=231.5, left_price=-245, right_price=172)])
    ticks.attrs["season_fingerprints"] = {2026: [1]}
    assert _kept(ticks).attrs == {"season_fingerprints": {2026: [1]}}


def test_only_the_filtered_seasons_fingerprints_carry_the_version():
    marked = alt_lines.filtered_fingerprints({2025: [1, 2], 2026: [3]})
    assert marked == {2025: [1, 2], 2026: [3, alt_lines.FILTER_VERSION]}


class _FakeConn:
    def cursor(self):
        return contextlib.nullcontext(self)

    def execute(self, *args, **kwargs):
        pass


def _install_store(monkeypatch, stored: pd.DataFrame) -> None:
    """Serve ``stored`` as both SBR reads, the closing one cutting its tail
    as the SQL does."""

    def closing_once(*, last_n=None, **kw):
        tail = stored.sort_values("line_ts").groupby(["game_id", "market", "book"])
        return (stored if last_n is None else tail.tail(last_n)).copy()

    monkeypatch.setattr(
        fetch, "_fetch_pregame_ticks_once", lambda season_years, **kw: stored.copy()
    )
    monkeypatch.setattr(fetch, "_fetch_closing_ticks_once", closing_once)
    monkeypatch.setattr(
        fetch, "connect_line_history_db", lambda: contextlib.nullcontext(_FakeConn())
    )
    monkeypatch.setattr(
        fetch, "line_history_season_fingerprint", lambda conn, season: [season]
    )
    monkeypatch.setenv("NBA_OU_S3_SEASONS", "0")


@pytest.fixture
def fake_store(monkeypatch):
    stored = _market([_tick(5, "bet365", line=231.5, left_price=-245, right_price=172)])
    _install_store(monkeypatch, stored)
    return stored


def test_both_store_reads_drop_alternate_lines(fake_store):
    pregame = fetch.fetch_pregame_ticks([2026], covers_fallback=False)
    closing = fetch.fetch_closing_ticks(game_ids=[GAME], covers_fallback=False)
    for ticks in (pregame, closing):
        assert len(ticks) == len(fake_store) - 1
        assert 231.5 not in ticks["left_line"].tolist()


def test_the_pregame_fingerprints_mark_the_filtered_seasons(fake_store):
    fingerprints = fetch.pregame_tick_fingerprints([2025, 2026], covers_fallback=False)
    assert fingerprints == {2025: [2025], 2026: [2026, alt_lines.FILTER_VERSION]}


def test_the_close_keeps_the_main_line_before_a_tail_of_alternates(monkeypatch):
    alternates = [
        _tick(minute, "bet365", line=231.5, left_price=-245, right_price=172)
        for minute in range(6, 12)
    ]
    _install_store(monkeypatch, _market([_tick(1, "bet365", line=238.5), *alternates]))
    closing = fetch.fetch_closing_ticks(game_ids=[GAME], covers_fallback=False)
    bet365 = closing[closing["book"].eq("bet365")]
    assert bet365["left_line"].tolist() == [238.5]
    assert len(closing[closing["book"].ne("bet365")]) == 3


def test_seasons_before_the_filter_keep_the_tail_cut_in_sql(fake_store, monkeypatch):
    calls = []
    real = fetch._fetch_closing_ticks_once

    def spy(**kw):
        calls.append((kw.get("season_years"), kw.get("game_ids"), kw.get("last_n")))
        return real(**kw)

    monkeypatch.setattr(fetch, "_fetch_closing_ticks_once", spy)
    fetch.fetch_closing_ticks(
        game_ids=["0022500001", GAME], last_n=5, covers_fallback=False
    )
    assert calls == [(None, ["0022500001"], 5), (None, [GAME], None)]
