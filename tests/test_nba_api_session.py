from nba_api.library.http import NBAHTTP
from nba_api.live.nba.library.http import NBALiveHTTP
from nba_api.stats.library.http import NBAStatsHTTP
from nba_ou.fetch_data.nba_api_session import reset_nba_http_session


def test_reset_replaces_the_session_stats_endpoints_use():
    # Endpoints call get_session through NBAStatsHTTP, which caches on the
    # subclass; resetting only NBAHTTP._session left this one alive.
    before = NBAStatsHTTP().get_session()

    reset_nba_http_session()

    after = NBAStatsHTTP().get_session()
    assert after is not before


def test_reset_covers_live_endpoints_and_base_class():
    live_before = NBALiveHTTP().get_session()
    base_before = NBAHTTP.get_session()

    reset_nba_http_session()

    assert NBALiveHTTP().get_session() is not live_before
    assert NBAHTTP.get_session() is not base_before


def test_reset_closes_the_old_session(monkeypatch):
    session = NBAStatsHTTP().get_session()
    closed = []
    monkeypatch.setattr(session, "close", lambda: closed.append(True))

    reset_nba_http_session()

    assert closed == [True]


def test_reset_is_safe_before_any_request():
    reset_nba_http_session()
    reset_nba_http_session()

    assert NBAStatsHTTP().get_session() is not None
