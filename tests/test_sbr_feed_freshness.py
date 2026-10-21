"""SBR feed-freeze detection, run before tonight's predictions.

A freeze stops every tick at one clock time; the site keeps serving that state.
Normal game days are never silent for more than ~2 h after 08:00 ET.
"""

from datetime import date

import pandas as pd
from nba_ou.postgre_db.line_history_aiven.feed_freshness import (
    FEED_STALE_MINUTES,
    assess_feed_freshness,
)

DAY = date(2026, 10, 21)


def et(clock: str, day: date = DAY) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz="America/New_York")


def assess(ticks, tips, as_of):
    return assess_feed_freshness(ticks, tips, game_date=DAY, as_of=as_of)


def test_feed_frozen_since_yesterday_afternoon_is_stale_by_late_morning():
    # 2026-10-20: every tick stopped at ~17:20 ET; tonight's games tip at 19:00+.
    frozen = et("17:21", date(2026, 10, 20))
    result = assess([frozen], [et("19:00"), et("22:30")], as_of=et("11:00"))

    assert result.stale
    assert result.quiet_minutes == 180  # counted from 08:00 ET, not from the tick
    assert result.pending_games == 2


def test_overnight_silence_is_not_stale_before_the_watch_starts():
    frozen = et("17:21", date(2026, 10, 20))
    result = assess([frozen], [et("19:00")], as_of=et("07:30"))

    assert not result.stale
    assert result.quiet_minutes == 0


def test_a_moving_feed_is_fresh():
    result = assess([et("09:00"), et("13:55")], [et("19:00")], as_of=et("14:30"))

    assert not result.stale
    assert result.quiet_minutes == 35


def test_silence_just_past_the_limit_flips_to_stale():
    last = et("12:00")
    at_limit = last + pd.Timedelta(minutes=FEED_STALE_MINUTES)

    assert not assess([last], [et("19:00")], as_of=at_limit).stale
    assert assess([last], [et("19:00")], as_of=at_limit + pd.Timedelta(minutes=1)).stale


def test_nothing_to_check_once_every_game_has_tipped():
    result = assess([et("14:58")], [et("15:00")], as_of=et("23:00"))

    assert result.quiet_minutes is None
    assert not result.stale


def test_ticks_after_as_of_are_ignored():
    # Replaying an earlier moment must not see the future.
    result = assess([et("10:00"), et("18:00")], [et("19:00")], as_of=et("12:00"))

    assert result.latest_tick == et("10:00").tz_convert("UTC")
    assert result.quiet_minutes == 120
