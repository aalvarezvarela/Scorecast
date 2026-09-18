"""Resetting the HTTP session that ``nba_api`` endpoints share.

``NBAHTTP.get_session`` is a classmethod that caches with ``cls._session = ...``,
so the session lands on whichever subclass asked for it -- ``NBAStatsHTTP`` for
every stats endpoint, ``NBALiveHTTP`` for live ones -- not on ``NBAHTTP``.
Clearing ``NBAHTTP._session`` therefore leaves the endpoints on their old
connection pool and cookies, and only a process restart used to clear them.
"""

from nba_api.library.http import NBAHTTP


def _http_classes(cls: type = NBAHTTP) -> list[type]:
    """``cls`` and every subclass of it currently imported."""
    classes = [cls]
    for subclass in cls.__subclasses__():
        classes.extend(_http_classes(subclass))
    return classes


def reset_nba_http_session() -> None:
    """Close every cached ``nba_api`` session; the next request opens a new one."""
    for cls in _http_classes():
        session = cls.__dict__.get("_session")
        if session is not None:
            session.close()
        # The base class keeps its declared ``None``; subclasses lose the
        # override, so they fall through to it and create a fresh session.
        if cls is NBAHTTP:
            cls._session = None
        elif "_session" in cls.__dict__:
            delattr(cls, "_session")
