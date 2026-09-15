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
    """2025-12-22 has 9 old-format files (00:30-08:30 ET) and 60 new-format files
    (from 09:00 ET); each shape is probed only over its own half of the day."""
    day = date(2025, 12, 22)
    assert U.eras_for_date(day) == (U.ERA_HOURLY_24, U.ERA_QUARTER_96)
    cands = U.candidates_for_date(day)
    hourly = [c for c in cands if c.era == U.ERA_HOURLY_24]
    quarter = [c for c in cands if c.era == U.ERA_QUARTER_96]
    assert (len(hourly), len(quarter)) == (9, 60)
    assert max(c.report_datetime_et.time() for c in hourly).isoformat() == "08:30:00"
    assert min(c.report_datetime_et.time() for c in quarter).isoformat() == "09:00:00"


def test_crossover_day_keys_do_not_collide():
    """Regression: hourly ``12AM`` and quarter ``12_30AM`` both decoded to 00:30.
    The quarter URL's 403 overwrote the real hourly file's manifest row, so the
    nine morning reports were never downloaded."""
    keys = [c.report_key for c in U.candidates_for_date(date(2025, 12, 22))]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("day", U.date_range(date(2025, 12, 15), date(2025, 12, 29)))
def test_keys_are_unique_around_the_crossover(day):
    keys = [c.report_key for c in U.candidates_for_date(day)]
    assert len(keys) == len(set(keys))


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


# --------------------------------------------------------------------------- #
# Offseason skip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "day",
    [
        date(2022, 8, 15),
        date(2023, 8, 1),
        date(2023, 9, 1),
        date(2024, 9, 20),
        date(2026, 8, 10),
    ],
)
def test_deep_offseason_is_skipped(day):
    assert U.is_offseason_gap(day) is True
    assert U.candidates_for_date(day) == []


@pytest.mark.parametrize(
    "day",
    [
        date(2020, 8, 15),  # bubble ran 30 Jul - 11 Oct 2020
        date(2020, 9, 20),
        date(2021, 8, 10),  # delayed calendar: live block 6-15 Aug 2021
    ],
)
def test_covid_years_are_exempt_from_the_skip(day):
    assert U.is_offseason_gap(day) is False
    assert len(U.candidates_for_date(day)) > 0


@pytest.mark.parametrize(
    "day",
    [
        date(2024, 7, 22),  # latest Summer League activity observed
        date(2025, 7, 19),
        date(2019, 9, 29),  # earliest post-summer activity observed
        date(2022, 10, 19),
    ],
)
def test_skip_never_covers_an_observed_active_date(day):
    """The window is set strictly inside the measured active bounds; if someone
    widens it, these dates are the ones that start silently disappearing."""
    assert U.is_offseason_gap(day) is False


def test_skip_boundaries_are_inclusive():
    assert U.is_offseason_gap(date(2023, 7, 24)) is False
    assert U.is_offseason_gap(date(2023, 7, 25)) is True
    assert U.is_offseason_gap(date(2023, 9, 22)) is True
    assert U.is_offseason_gap(date(2023, 9, 23)) is False


def test_skip_can_be_turned_off():
    day = date(2023, 8, 15)
    assert U.candidates_for_date(day) == []
    assert len(U.candidates_for_date(day, skip_offseason=False)) == 24


# --------------------------------------------------------------------------- #
# Header-time tolerance and publication time (2025-12-19 :45 stamping)
# --------------------------------------------------------------------------- #

import pymupdf  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive import discovery as D  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive import download as DL  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive import validation as V  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive.client import (  # noqa: E402
    Probe,
    Verdict,
)


def _report_pdf(header: str) -> bytes:
    """A real PDF whose text carries ``header`` and enough body to not look scanned."""
    doc = pymupdf.open()
    for page_no in range(4):
        page = doc.new_page()
        rows = [
            f"Game Date Game Time Matchup Team Player Name {page_no}-{i:02d}"
            for i in range(30)
        ]
        body = ([header] if page_no == 0 else []) + rows
        page.insert_text((40, 40), "\n".join(body), fontsize=8)
    data = doc.tobytes(deflate=False)
    assert len(data) >= V.MIN_PDF_BYTES
    doc.close()
    return data


def _expected(day, label, era):
    return U.et_datetime(day, label, era)


def test_strict_validation_still_rejects_a_shifted_header():
    data = _report_pdf("Injury Report: 12/19/25 04:45 PM")
    result = V.validate(data, _expected(date(2025, 12, 19), "04PM", U.ERA_HOURLY_24))
    assert result.status == V.MISMATCH and not result.ok


@pytest.mark.parametrize(
    ("header", "day", "label", "era", "status"),
    [
        # hourly file stamped :45 instead of :30 (2025-12-19 16:45 onward)
        (
            "Injury Report: 12/19/25 04:45 PM",
            date(2025, 12, 19),
            "04PM",
            U.ERA_HOURLY_24,
            V.OK_OFFSET,
        ),
        # bubble 05PM stamped 17:30 instead of 17:00
        (
            "Injury Report: 08/05/20 05:30 PM",
            date(2020, 8, 5),
            "05PM",
            U.ERA_BUBBLE_3,
            V.OK_OFFSET,
        ),
        # 2020-21 file 30 minutes early
        (
            "Injury Report: 01/10/21 05:00 PM",
            date(2021, 1, 10),
            "05PM",
            U.ERA_LEGACY_3,
            V.OK_OFFSET,
        ),
        # 85 minutes out is a different report, not a restamp
        (
            "Injury Report: 01/10/21 06:55 PM",
            date(2021, 1, 10),
            "05PM",
            U.ERA_LEGACY_3,
            V.MISMATCH,
        ),
        # quarter-hourly: published a few minutes late
        (
            "Injury Report: 04/30/26 01:20 PM",
            date(2026, 4, 30),
            "01_15PM",
            U.ERA_QUARTER_96,
            V.OK_OFFSET,
        ),
        # quarter-hourly: 15 minutes out would be the next slot
        (
            "Injury Report: 04/30/26 01:30 PM",
            date(2026, 4, 30),
            "01_15PM",
            U.ERA_QUARTER_96,
            V.MISMATCH,
        ),
        # exact match keeps plain ok
        (
            "Injury Report: 12/11/24 08:30 PM",
            date(2024, 12, 11),
            "08PM",
            U.ERA_HOURLY_24,
            V.OK,
        ),
    ],
)
def test_era_tolerance_accepts_restamps_but_not_other_reports(
    header, day, label, era, status
):
    result = V.validate(
        _report_pdf(header),
        _expected(day, label, era),
        max_offset_minutes=U.validation_tolerance_minutes(era),
    )
    assert result.status == status
    assert result.ok is (status != V.MISMATCH)


def test_tolerance_never_reaches_a_neighbouring_slot():
    assert U.validation_tolerance_minutes(U.ERA_HOURLY_24) < 60 / 2 + 30
    assert U.validation_tolerance_minutes(U.ERA_QUARTER_96) < 15
    assert U.validation_tolerance_minutes("unknown_era") == 0


def test_published_utc_prefers_the_header_and_falls_back_to_the_filename():
    df = pd.DataFrame(
        {
            "report_datetime_utc": pd.to_datetime(
                ["2025-12-19T21:30Z", "2024-12-12T01:30Z"]
            ),
            "report_published_utc": pd.to_datetime(
                ["2025-12-19T21:45Z", None], utc=True
            ),
        }
    )
    assert list(mf.published_utc(df)) == [
        pd.Timestamp("2025-12-19T21:45Z"),
        pd.Timestamp("2024-12-12T01:30Z"),
    ]
    assert list(mf.published_utc(df.drop(columns="report_published_utc"))) == list(
        pd.to_datetime(df.report_datetime_utc, utc=True)
    )


class _FakeClient:
    def __init__(
        self, bodies: dict[str, bytes] | None = None, existing: set[str] | None = None
    ):
        self.bodies = bodies or {}
        self.existing = existing or set()
        self.probed: list[str] = []

    def probe(self, url):
        self.probed.append(url)
        if url in self.existing:
            return Probe(Verdict.EXISTS, status=200, content_length=100)
        return Probe(Verdict.MISSING, status=403)

    def fetch(self, url):
        return self.bodies[url], Probe(Verdict.EXISTS, status=200)


class _FakeStorage:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.meta: dict[str, dict] = {}

    def exists(self, key):
        return len(self.objects[key]) if key in self.objects else None

    def get(self, key):
        return self.objects.get(key)

    def put(self, key, data, metadata):
        self.objects[key] = data
        self.meta[key] = metadata

    def describe(self):
        return "fake"


def test_discovery_skips_on_url_so_a_shared_key_cannot_block_a_real_file(tmp_path):
    """A 403 on one URL must not stop another URL with the same key being probed."""
    day = date(2025, 12, 22)
    hourly = next(c for c in U.candidates_for_date(day) if c.label == "12AM")
    store = mf.ManifestStore(root=tmp_path)
    store.upsert(
        "2025-26",
        [
            {
                "report_key": hourly.report_key,  # same derived instant, other URL
                "season": "2025-26",
                "report_datetime_utc": hourly.report_datetime_utc,
                "original_url": U.build_url(day, "12_30AM"),
                "source_era": U.ERA_QUARTER_96,
                "nba_available": mf.AVAILABLE_FALSE,
                "download_status": mf.DOWNLOAD_PENDING,
            }
        ],
    )
    client = _FakeClient(existing={hourly.url})
    D.discover(store=store, client=client, start=day, end=day, verbose=False)

    assert hourly.url in client.probed
    row = store.load("2025-26").set_index("report_key").loc[hourly.report_key]
    assert row.original_url == hourly.url
    assert row.nba_available == mf.AVAILABLE_TRUE


def test_download_stores_a_restamped_report_at_its_header_time(tmp_path):
    day = date(2025, 12, 19)
    cand = next(c for c in U.candidates_for_date(day) if c.label == "04PM")
    store = mf.ManifestStore(root=tmp_path)
    row = D._row(cand, Probe(Verdict.EXISTS, status=200), 1, "injury_reports")
    store.upsert("2025-26", [row])
    body = _report_pdf("Injury Report: 12/19/25 04:45 PM")
    storage = _FakeStorage()

    stats = DL.download_rows(
        rows=store.pending_downloads("2025-26"),
        store=store,
        client=_FakeClient(bodies={cand.url: body}),
        storage=storage,
        verbose=False,
    )

    assert (stats.stored, stats.invalid) == (1, 0)
    saved = store.load("2025-26").iloc[0]
    assert saved.download_status == mf.DOWNLOAD_STORED
    assert saved.validation_status == V.OK_OFFSET
    # 16:45 EST == 21:45 UTC, fifteen minutes after the filename-derived 21:30
    assert saved.report_published_utc == pd.Timestamp("2025-12-19T21:45Z")
    assert saved.report_datetime_utc == pd.Timestamp("2025-12-19T21:30Z")
    assert cand.s3_key in storage.objects
