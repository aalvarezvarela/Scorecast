"""UTC for calculations; Europe/Madrid for every displayed timestamp."""

from datetime import UTC, date, datetime

import pandas as pd

from nba_ou.dashboard.config import MADRID, SLATE_TIMEZONE


def utc_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def slate_date(now: datetime | None = None) -> date:
    """Match the daily predictor's Pacific-date NBA slate, including after midnight."""
    return (now or datetime.now(UTC)).astimezone(SLATE_TIMEZONE).date()


def madrid_time(value, *, include_date: bool = True) -> str:
    if value is None or pd.isna(value):
        return "Time unavailable"
    timestamp = utc_timestamp(value).tz_convert(MADRID)
    pattern = "%d %b %Y, %H:%M" if include_date else "%H:%M:%S"
    return f"{timestamp.strftime(pattern)} Madrid"


def time_remaining(tipoff, now) -> str:
    minutes = max(
        0, int((utc_timestamp(tipoff) - utc_timestamp(now)).total_seconds() // 60)
    )
    if minutes == 0:
        return "Tip-off imminent"
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"
    return f"Starts in {duration}"


def horizon_label(minutes) -> str:
    if minutes is None or pd.isna(minutes):
        return "Not available"
    hours, remainder = divmod(int(minutes), 60)
    if not hours:
        return f"T−{remainder}m"
    return f"T−{hours}h" + (f" {remainder}m" if remainder else "")
