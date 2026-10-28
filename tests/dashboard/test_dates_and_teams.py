from datetime import date

import pandas as pd
from nba_ou.dashboard.domain.dates import madrid_time, slate_date
from nba_ou.dashboard.domain.teams import team_logo_url


def test_madrid_time_observes_winter_summer_and_local_date_rollover():
    assert madrid_time(pd.Timestamp("2026-01-15T23:30Z")) == "16 Jan 2026, 00:30 Madrid"
    assert madrid_time(pd.Timestamp("2026-07-15T23:30Z")) == "16 Jul 2026, 01:30 Madrid"


def test_madrid_fall_dst_change():
    assert madrid_time(pd.Timestamp("2026-10-25T00:30Z")) == "25 Oct 2026, 02:30 Madrid"
    assert madrid_time(pd.Timestamp("2026-10-25T01:30Z")) == "25 Oct 2026, 02:30 Madrid"


def test_nba_slate_is_not_split_at_madrid_midnight():
    assert slate_date(pd.Timestamp("2026-10-29T01:00Z").to_pydatetime()) == date(
        2026, 10, 28
    )


def test_both_team_images_preserve_existing_espn_identity():
    assert team_logo_url("Boston Celtics").endswith("/2.png")
    assert team_logo_url("New York Knicks").endswith("/18.png")
    assert team_logo_url("LA Clippers").endswith("/12.png")
    assert team_logo_url("LAL", 1610612747).endswith("/13.png")
    assert team_logo_url("Unknown team") is None
