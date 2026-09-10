"""Temporal-correctness contract for the injury-report archive.

These tests exist because every one of them corresponds to a way the archive can
silently acquire a look-ahead error or lose data. They are all pure -- no network
-- and each pins a fact that was measured against the live CDN and recorded in
``docs/injury_report_archive_plan.md``.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from nba_ou.fetch_data.injury_reports.archive import manifest as mf
from nba_ou.fetch_data.injury_reports.archive import urls as U
from nba_ou.fetch_data.injury_reports.archive.validation import parse_header_datetime

UTC = ZoneInfo("UTC")


# --------------------------------------------------------------------------- #
# Label decoding: the filename hour is not the report time
# --------------------------------------------------------------------------- #


def test_hourly_label_means_half_past():
    """``_08PM`` is stamped 20:30 ET, not 20:00 -- the label hides a +30 min."""
    moment = U.et_datetime(date(2024, 12, 11), "08PM", U.ERA_HOURLY_24)
    assert (moment.hour, moment.minute) == (20, 30)


def test_bubble_label_means_on_the_hour():
    """The 2020 bubble published at :00, so the same shape means something else."""
    moment = U.et_datetime(date(2020, 8, 5), "11AM", U.ERA_BUBBLE_3)
    assert (moment.hour, moment.minute) == (11, 0)


def test_quarter_label_is_exact():
    moment = U.et_datetime(date(2026, 3, 11), "08_15PM", U.ERA_QUARTER_96)
    assert (moment.hour, moment.minute) == (20, 15)


@pytest.mark.parametrize(
    ("label", "era", "expected"),
    [
        ("12AM", U.ERA_HOURLY_24, (0, 30)),
        ("12PM", U.ERA_HOURLY_24, (12, 30)),
        ("12_00AM", U.ERA_QUARTER_96, (0, 0)),
        ("12_45PM", U.ERA_QUARTER_96, (12, 45)),
    ],
)
def test_twelve_oclock_wraps_correctly(label, era, expected):
    moment = U.et_datetime(date(2026, 1, 15), label, era)
    assert (moment.hour, moment.minute) == expected


# --------------------------------------------------------------------------- #
# Daylight saving -- the two ways this archive can gain an hour of leakage
# --------------------------------------------------------------------------- #


def test_spring_forward_hour_does_not_exist():
    """02:00-02:59 ET is not a real wall time, and the CDN publishes nothing."""
    assert U.et_datetime(date(2024, 3, 10), "02AM", U.ERA_HOURLY_24) is None


def test_spring_forward_hourly_day_has_23_candidates():
    assert len(U.candidates_for_date(date(2024, 3, 10))) == 23


def test_spring_forward_quarter_day_has_92_candidates():
    """Measured: 2026-03-08 returns 92/96, the whole 02:00-02:45 band absent."""
    assert len(U.candidates_for_date(date(2026, 3, 8))) == 92


def test_fall_back_resolves_to_the_first_occurrence():
    """The repeated 01:00 is published once, in EDT. ``fold=1`` would be a
    one-hour look-ahead error."""
    moment = U.et_datetime(date(2024, 11, 3), "01AM", U.ERA_HOURLY_24)
    assert moment.utcoffset().total_seconds() == -4 * 3600
    assert moment.astimezone(UTC) == datetime(2024, 11, 3, 5, 30, tzinfo=UTC)


def test_fall_back_second_hour_is_a_distinct_key():
    """Both occurrences must be nameable without colliding."""
    first = datetime(2026, 11, 1, 1, 0, tzinfo=U.ET, fold=0)
    second = datetime(2026, 11, 1, 1, 0, tzinfo=U.ET, fold=1)
    assert U.canonical_stem(first) != U.canonical_stem(second)
    assert U.canonical_stem(first).endswith("-0400")
    assert U.canonical_stem(second).endswith("-0500")


# --------------------------------------------------------------------------- #
# Era boundaries
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2019, 1, 16), (U.ERA_LEGACY_3,)),
        (date(2020, 8, 5), (U.ERA_BUBBLE_3,)),
        (date(2021, 10, 17), (U.ERA_LEGACY_3,)),
        (date(2021, 10, 18), (U.ERA_HOURLY_24,)),
        (date(2025, 12, 21), (U.ERA_HOURLY_24,)),
        (date(2025, 12, 23), (U.ERA_QUARTER_96,)),
        (date(2018, 11, 1), (U.ERA_WP_MEDIA,)),
    ],
)
def test_era_for_date(day, expected):
    assert U.eras_for_date(day) == expected


def test_crossover_day_carries_both_formats():
    """2025-12-22 has 9 old-format and 60 new-format files; probe both spaces."""
    assert U.eras_for_date(date(2025, 12, 22)) == (U.ERA_HOURLY_24, U.ERA_QUARTER_96)
    assert len(U.candidates_for_date(date(2025, 12, 22))) == 24 + 96


def test_ordinary_day_counts():
    assert len(U.candidates_for_date(date(2024, 12, 11))) == 24
    assert len(U.candidates_for_date(date(2026, 3, 11))) == 96


# --------------------------------------------------------------------------- #
# Canonical naming
# --------------------------------------------------------------------------- #


def test_canonical_name_normalises_across_eras():
    """All three eras produce one shape, and the misleading labels are resolved."""
    hourly = U.et_datetime(date(2024, 12, 11), "08PM", U.ERA_HOURLY_24)
    bubble = U.et_datetime(date(2020, 8, 5), "11AM", U.ERA_BUBBLE_3)
    quarter = U.et_datetime(date(2026, 3, 11), "08_15PM", U.ERA_QUARTER_96)
    assert U.canonical_filename(hourly) == "injury-report_2024-12-11T2030-0500.pdf"
    assert U.canonical_filename(bubble) == "injury-report_2020-08-05T1100-0400.pdf"
    assert U.canonical_filename(quarter) == "injury-report_2026-03-11T2015-0400.pdf"


def test_s3_key_partitions_by_season_and_et_date():
    moment = U.et_datetime(date(2024, 12, 11), "08PM", U.ERA_HOURLY_24)
    assert U.s3_key(moment) == (
        "injury_reports/raw/season=2024-25/date=2024-12-11/"
        "injury-report_2024-12-11T2030-0500.pdf"
    )


def test_keys_are_unique_within_a_day():
    keys = [c.report_key for c in U.candidates_for_date(date(2026, 3, 11))]
    assert len(keys) == len(set(keys))


def test_bubble_dates_land_in_the_2019_20_season():
    """``get_season_year_from_date`` puts Aug-Oct 2020 in 2019-20; the key must
    agree or the bubble ends up filed under the wrong season."""
    assert U.season_label(date(2020, 8, 5)) == "2019-20"


# --------------------------------------------------------------------------- #
# Manifest merge semantics
# --------------------------------------------------------------------------- #


def _discovery_row(key: str) -> dict:
    return {
        "report_key": key,
        "season": "2025-26",
        "report_datetime_utc": pd.Timestamp("2026-03-11T05:30:00Z"),
        "original_url": "https://example/x.pdf",
        "nba_available": mf.AVAILABLE_TRUE,
        "download_status": mf.DOWNLOAD_PENDING,
    }


def test_partial_update_does_not_blank_discovery_fields(tmp_path):
    """Regression: a download-phase update used to replace the whole row, wiping
    ``nba_available`` and pushing settled rows back onto the retry queue."""
    store = mf.ManifestStore(root=tmp_path)
    store.upsert("2025-26", [_discovery_row("k1")])
    store.upsert(
        "2025-26",
        [{"report_key": "k1", "download_status": mf.DOWNLOAD_STORED, "sha256": "ab"}],
    )
    row = store.load("2025-26").iloc[0]
    assert row.nba_available == mf.AVAILABLE_TRUE
    assert row.original_url == "https://example/x.pdf"
    assert row.download_status == mf.DOWNLOAD_STORED
    assert row.sha256 == "ab"


def test_resolved_keys_only_returns_terminal_verdicts(tmp_path):
    store = mf.ManifestStore(root=tmp_path)
    rows = [_discovery_row("k1"), _discovery_row("k2"), _discovery_row("k3")]
    rows[1]["nba_available"] = mf.AVAILABLE_FALSE
    rows[2]["nba_available"] = mf.AVAILABLE_UNKNOWN
    store.upsert("2025-26", rows)
    assert store.resolved_keys("2025-26") == {"k1", "k2"}


def test_manifest_round_trips_through_parquet(tmp_path):
    store = mf.ManifestStore(root=tmp_path)
    store.upsert("2025-26", [_discovery_row("k1")])
    store.save("2025-26")
    reloaded = mf.ManifestStore(root=tmp_path).load("2025-26")
    assert len(reloaded) == 1
    assert reloaded.iloc[0].nba_available == mf.AVAILABLE_TRUE


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Injury Report: 12/11/24 08:30 PM", datetime(2024, 12, 11, 20, 30)),
        ("Injury Report: 03/11/26 08:15 PM", datetime(2026, 3, 11, 20, 15)),
        ("Injury Report: 08/05/20 11:00 AM", datetime(2020, 8, 5, 11, 0)),
        ("Injury Report: 12/11/24 12:30 AM", datetime(2024, 12, 11, 0, 30)),
    ],
)
def test_header_parsing(text, expected):
    assert parse_header_datetime(text) == expected


def test_header_parsing_returns_none_when_absent():
    assert parse_header_datetime("no header here") is None


def test_header_matches_derived_time_for_every_era():
    """The check that validates the whole era model at ingest."""
    cases = [
        (
            date(2024, 12, 11),
            "08PM",
            U.ERA_HOURLY_24,
            "Injury Report: 12/11/24 08:30 PM",
        ),
        (date(2020, 8, 5), "11AM", U.ERA_BUBBLE_3, "Injury Report: 08/05/20 11:00 AM"),
        (
            date(2026, 3, 11),
            "08_15PM",
            U.ERA_QUARTER_96,
            "Injury Report: 03/11/26 08:15 PM",
        ),
    ]
    for day, label, era, header in cases:
        derived = U.et_datetime(day, label, era).replace(tzinfo=None)
        assert parse_header_datetime(header) == derived
