"""Candidate URL space for the NBA official injury-report archive.

Everything time-related in this package is decided here, because the filename is
the only thing the CDN gives us and it is *not* a faithful record of when the
report was published:

* ``Injury-Report_2024-12-11_08PM.pdf`` is stamped **20:30 ET**, not 20:00. The
  hourly-era label carries an implicit ``+30 min``.
* During the 2020 bubble the same shape meant **:00**, not :30.
* From 2025-12-22 the filename gained a minute field and the cadence went to
  every 15 minutes, and the label is then exact.

So a label is decoded through its *era*, and the resulting ``America/New_York``
wall-clock instant -- not the label -- is what names the object and drives every
join. See ``docs/injury_report_archive_plan.md`` for the measurements behind the
era boundaries.

Two DST rules fall out of the wall-clock model, both verified against the CDN:

* Spring forward: 02:00-02:59 ET does not exist, and no report is published for
  it (2024-03-10 returns 23/24 labels; 2026-03-08 returns 92/96).
* Fall back: the repeated 01:00 hour was published **once**, at the first (EDT)
  occurrence, so naive local times are localized with ``fold=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nba_ou.utils.general_utils import get_season_year_from_date

ET = ZoneInfo("America/New_York")

CDN_BASE = "https://ak-static.cms.nba.com/referee/injury"

#: First report available anywhere (WordPress media library, see ``wp_media.py``).
ARCHIVE_START = Date(2018, 10, 17)
#: First report on the ``/referee/injury/`` path.
CDN_START = Date(2018, 12, 17)
#: The 2020 bubble published on the hour instead of at :30.
BUBBLE_START, BUBBLE_END = Date(2020, 7, 29), Date(2020, 10, 11)
#: 3 reports/day became 24 reports/day.
HOURLY_24_START = Date(2021, 10, 18)
#: The filename gained a minute field and the cadence went to every 15 minutes.
#: This day is a crossover: it carries 9 old-format *and* 60 new-format files.
QUARTER_96_START = Date(2025, 12, 22)

ERA_WP_MEDIA = "wp_media"
ERA_LEGACY_3 = "legacy_3"
ERA_BUBBLE_3 = "bubble_3"
ERA_HOURLY_24 = "hourly_24"
ERA_QUARTER_96 = "quarter_96"

#: Labels for the hourly filename shape, ``_hhAM`` / ``_hhPM`` (24 per day).
HOURLY_LABELS: tuple[str, ...] = tuple(
    f"{h:02d}{ap}" for ap in ("AM", "PM") for h in range(1, 13)
)
#: Labels for the quarter-hourly shape, ``_hh_mmAM`` / ``_hh_mmPM`` (96 per day).
QUARTER_LABELS: tuple[str, ...] = tuple(
    f"{h:02d}_{m:02d}{ap}"
    for ap in ("AM", "PM")
    for h in range(1, 13)
    for m in (0, 15, 30, 45)
)


def season_label(day: Date) -> str:
    """``2024-12-11`` -> ``'2024-25'``."""
    year = get_season_year_from_date(datetime(day.year, day.month, day.day))
    return f"{year}-{str(year + 1)[-2:]}"


def eras_for_date(day: Date) -> tuple[str, ...]:
    """Eras that publish on ``day``.

    Returns two eras only for the 2025-12-22 crossover, which carries both
    filename shapes.
    """
    eras: list[str] = []
    if day < CDN_START:
        return (ERA_WP_MEDIA,)
    if day <= QUARTER_96_START:
        if BUBBLE_START <= day <= BUBBLE_END:
            eras.append(ERA_BUBBLE_3)
        elif day >= HOURLY_24_START:
            eras.append(ERA_HOURLY_24)
        else:
            eras.append(ERA_LEGACY_3)
    if day >= QUARTER_96_START:
        eras.append(ERA_QUARTER_96)
    return tuple(eras)


def _label_to_wall_time(label: str, era: str) -> tuple[int, int]:
    """Decode a filename label into the ET wall-clock ``(hour, minute)`` it means.

    The minute is a property of the *era*, not of the label: the hourly eras
    stamp :30 (or :00 in the bubble) while the label itself carries no minute.
    """
    if era == ERA_QUARTER_96:
        hh, rest = label.split("_")
        minute, meridiem = int(rest[:2]), rest[2:]
        hour12 = int(hh)
    else:
        hour12, meridiem = int(label[:2]), label[2:]
        minute = 0 if era == ERA_BUBBLE_3 else 30

    if meridiem == "AM":
        hour = 0 if hour12 == 12 else hour12
    else:
        hour = 12 if hour12 == 12 else hour12 + 12
    return hour, minute


def et_datetime(day: Date, label: str, era: str) -> datetime | None:
    """Tz-aware ET instant a ``(day, label)`` refers to, or ``None`` if unreal.

    ``None`` means the wall-clock time does not exist -- the spring-forward gap.
    Ambiguous fall-back times resolve to the first (EDT) occurrence, matching
    what the CDN actually publishes.
    """
    hour, minute = _label_to_wall_time(label, era)
    aware = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET, fold=0)
    # A wall time inside the spring-forward gap does not survive a UTC round trip.
    if aware.astimezone(ZoneInfo("UTC")).astimezone(ET).hour != hour:
        return None
    return aware


def canonical_stem(moment: datetime) -> str:
    """``2024-12-11 20:30 EST`` -> ``'2024-12-11T2030-0500'``.

    The UTC offset is part of the stem so the two occurrences of a fall-back
    hour cannot collide. Note this is *not* lexicographically chronological
    across that hour -- no ET-wall-clock-primary name can be, because the clock
    is non-monotonic. Order by ``report_datetime_utc`` from the manifest.
    """
    return f"{moment:%Y-%m-%dT%H%M}{moment:%z}"


def canonical_filename(moment: datetime) -> str:
    return f"injury-report_{canonical_stem(moment)}.pdf"


def s3_key(moment: datetime, *, prefix: str = "injury_reports") -> str:
    """Canonical object key, partitioned by season and ET date."""
    return (
        f"{prefix}/raw/season={season_label(moment.date())}"
        f"/date={moment:%Y-%m-%d}/{canonical_filename(moment)}"
    )


def build_url(day: Date, label: str) -> str:
    return f"{CDN_BASE}/Injury-Report_{day:%Y-%m-%d}_{label}.pdf"


@dataclass(frozen=True)
class Candidate:
    """One candidate URL and everything derivable from it without fetching."""

    report_key: str
    day: Date
    label: str
    era: str
    report_datetime_et: datetime
    url: str

    @property
    def season(self) -> str:
        return season_label(self.day)

    @property
    def report_datetime_utc(self) -> datetime:
        return self.report_datetime_et.astimezone(ZoneInfo("UTC"))

    @property
    def s3_key(self) -> str:
        return s3_key(self.report_datetime_et)


def candidates_for_date(day: Date) -> list[Candidate]:
    """Every plausible URL for ``day``, DST-filtered.

    24 candidates in the hourly eras, 96 in the quarter-hourly era, 120 on the
    single crossover day, 0 before the CDN path begins.
    """
    out: list[Candidate] = []
    for era in eras_for_date(day):
        if era == ERA_WP_MEDIA:
            continue  # listing-driven, see wp_media.py
        labels = QUARTER_LABELS if era == ERA_QUARTER_96 else HOURLY_LABELS
        for label in labels:
            moment = et_datetime(day, label, era)
            if moment is None:
                continue
            out.append(
                Candidate(
                    report_key=canonical_stem(moment),
                    day=day,
                    label=label,
                    era=era,
                    report_datetime_et=moment,
                    url=build_url(day, label),
                )
            )
    return out


def date_range(start: Date, end: Date) -> list[Date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def archive_end(today: Date | None = None) -> Date:
    """Last date worth probing. The feed is live, so this is simply today."""
    return today or Date.today()


def all_seasons(start: Date = CDN_START, end: Date | None = None) -> list[str]:
    """Season labels covered by the archive, oldest first."""
    seen: dict[str, None] = {}
    for day in date_range(start, archive_end(end)):
        seen.setdefault(season_label(day), None)
    return list(seen)


def season_date_range(season: str, *, end: Date | None = None) -> tuple[Date, Date]:
    """Inclusive ET date bounds of ``season`` inside the archive window.

    Derived by walking the calendar rather than hard-coding month cutoffs, so it
    can never disagree with ``season_label`` -- including the delayed 2020-21
    season, which ``get_season_year_from_date`` treats specially.
    """
    days = [
        d for d in date_range(CDN_START, archive_end(end)) if season_label(d) == season
    ]
    if not days:
        raise ValueError(f"No archive dates fall in season {season!r}")
    return days[0], days[-1]
