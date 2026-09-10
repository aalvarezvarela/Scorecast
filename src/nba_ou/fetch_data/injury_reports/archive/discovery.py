"""Phase 1: find out which candidate URLs exist, without downloading bodies.

Writes manifest rows only. A candidate with a terminal verdict is never probed
again, which is the whole of the resume logic -- stop the process at any point
and the next run picks up from what the manifest does not yet know.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import date as Date

from nba_ou.fetch_data.injury_reports.archive import manifest as mf
from nba_ou.fetch_data.injury_reports.archive import urls as U
from nba_ou.fetch_data.injury_reports.archive.client import (
    ArchiveClient,
    ThrottledError,
    Verdict,
)
from tqdm.auto import tqdm


@dataclass
class DiscoveryStats:
    dates: int = 0
    probed: int = 0
    found: int = 0
    missing: int = 0
    unknown: int = 0
    skipped: int = 0
    stopped_early: bool = False
    reason: str = ""
    per_season: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"Dates walked   : {self.dates}",
            f"URLs probed    : {self.probed}  "
            f"(skipped {self.skipped} already resolved)",
            f"  exists       : {self.found}",
            f"  missing      : {self.missing}",
            f"  unknown      : {self.unknown}",
        ]
        if self.per_season:
            lines.append("Found by season:")
            for s in sorted(self.per_season):
                lines.append(f"  {s}: {self.per_season[s]}")
        if self.stopped_early:
            lines.append(f"STOPPED EARLY: {self.reason}")
        return "\n".join(lines)


_VERDICT_TO_AVAILABILITY = {
    Verdict.EXISTS: mf.AVAILABLE_TRUE,
    Verdict.MISSING: mf.AVAILABLE_FALSE,
    Verdict.UNKNOWN: mf.AVAILABLE_UNKNOWN,
}


def _row(candidate: U.Candidate, probe, attempts: int, s3_prefix: str) -> dict:
    return {
        "report_key": candidate.report_key,
        "season": candidate.season,
        "season_year": int(candidate.season.split("-")[0]),
        "report_date_et": candidate.day.isoformat(),
        "report_time_label": candidate.label,
        "source_era": candidate.era,
        "report_datetime_et": candidate.report_datetime_et.isoformat(),
        "report_datetime_utc": candidate.report_datetime_utc,
        "original_url": candidate.url,
        "original_filename": candidate.url.rsplit("/", 1)[-1],
        "nba_available": _VERDICT_TO_AVAILABILITY[probe.verdict],
        "nba_http_status": probe.status,
        "content_type": probe.content_type,
        "content_length": probe.content_length,
        "etag": probe.etag,
        "last_modified": probe.last_modified,
        "discovery_timestamp": datetime.now(UTC),
        "discovery_attempts": attempts,
        "download_status": mf.DOWNLOAD_PENDING,
        "s3_key": U.s3_key(candidate.report_datetime_et, prefix=s3_prefix),
        "notes": probe.note,
    }


def discover(
    *,
    store: mf.ManifestStore,
    client: ArchiveClient,
    start: Date,
    end: Date,
    max_requests: int | None = None,
    checkpoint_every: int = 500,
    retry_unknown: bool = True,
    skip_offseason: bool = True,
    s3_prefix: str = "injury_reports",
    dry_run: bool = False,
    verbose: bool = True,
) -> DiscoveryStats:
    stats = DiscoveryStats()
    pending: dict[str, list[dict]] = {}
    since_checkpoint = 0

    # Count first so the bar has a real total. This is pure arithmetic over the
    # candidate space plus the manifest -- no network -- so it is cheap even for
    # a whole-archive run.
    days = U.date_range(start, end)
    to_probe = 0
    for day in days:
        cands = U.candidates_for_date(day, skip_offseason=skip_offseason)
        if not cands:
            continue
        resolved = store.resolved_keys(U.season_label(day)) if not dry_run else set()
        to_probe += sum(1 for c in cands if c.report_key not in resolved)
    if max_requests is not None:
        to_probe = min(to_probe, max_requests)

    bar = tqdm(
        total=to_probe,
        desc="discover",
        unit="url",
        disable=None if verbose else True,
        smoothing=0.05,
    )

    def flush() -> None:
        if dry_run:
            pending.clear()
            return
        for season, rows in pending.items():
            store.upsert(season, rows)
            store.save(season)
        pending.clear()

    try:
        for day in days:
            candidates = U.candidates_for_date(day, skip_offseason=skip_offseason)
            if not candidates:
                continue
            stats.dates += 1
            season = U.season_label(day)
            resolved = store.resolved_keys(season) if not dry_run else set()
            todo = [c for c in candidates if c.report_key not in resolved]
            stats.skipped += len(candidates) - len(todo)
            if not todo:
                continue

            for cand in todo:
                if max_requests is not None and stats.probed >= max_requests:
                    stats.stopped_early = True
                    stats.reason = f"reached --max-requests={max_requests}"
                    flush()
                    return stats

                probe = client.probe(cand.url)
                stats.probed += 1
                since_checkpoint += 1
                bar.update(1)
                if probe.verdict is Verdict.EXISTS:
                    stats.found += 1
                    stats.per_season[season] = stats.per_season.get(season, 0) + 1
                elif probe.verdict is Verdict.MISSING:
                    stats.missing += 1
                else:
                    stats.unknown += 1

                pending.setdefault(season, []).append(_row(cand, probe, 1, s3_prefix))
                bar.set_postfix_str(
                    f"{day} | found {stats.found} missing {stats.missing}"
                    + (f" unknown {stats.unknown}" if stats.unknown else ""),
                    refresh=False,
                )

                if since_checkpoint >= checkpoint_every:
                    flush()
                    since_checkpoint = 0

    except ThrottledError as exc:
        stats.stopped_early = True
        stats.reason = str(exc)
        flush()
        return stats
    except KeyboardInterrupt:
        stats.stopped_early = True
        stats.reason = "interrupted by user"
        flush()
        return stats
    finally:
        bar.close()

    flush()
    return stats
