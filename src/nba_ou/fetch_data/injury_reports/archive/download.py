"""Phase 2: fetch the confirmed-existing URLs and store them under canonical keys.

Idempotent by construction: the key is a pure function of the report's ET
instant, so a re-run either skips (object already present at the recorded size)
or overwrites byte-identical content. Nothing is ever stored twice under two
names, and a collision between two source URLs is a hard error rather than a
silent overwrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
from nba_ou.fetch_data.injury_reports.archive import manifest as mf
from nba_ou.fetch_data.injury_reports.archive.client import (
    ArchiveClient,
    ThrottledError,
    Verdict,
)
from nba_ou.fetch_data.injury_reports.archive.storage import Storage
from nba_ou.fetch_data.injury_reports.archive.validation import validate
from tqdm.auto import tqdm


@dataclass
class DownloadStats:
    considered: int = 0
    already: int = 0
    stored: int = 0
    failed: int = 0
    invalid: int = 0
    bytes_stored: int = 0
    stopped_early: bool = False
    reason: str = ""
    problems: list[str] = field(default_factory=list)

    def render(self) -> str:
        mb = self.bytes_stored / 1024**2
        lines = [
            f"Considered     : {self.considered}",
            f"  already there: {self.already}",
            f"  stored now   : {self.stored}  ({mb:.1f} MiB)",
            f"  invalid      : {self.invalid}",
            f"  failed       : {self.failed}",
        ]
        if self.problems:
            lines.append(f"Problems ({len(self.problems)}), first 10:")
            lines += [f"  - {p}" for p in self.problems[:10]]
        if self.stopped_early:
            lines.append(f"STOPPED EARLY: {self.reason}")
        return "\n".join(lines)


def _parse_et(value: str) -> datetime:
    return datetime.fromisoformat(value)


def download_rows(
    *,
    rows: pd.DataFrame,
    store: mf.ManifestStore,
    client: ArchiveClient,
    storage: Storage,
    max_files: int | None = None,
    checkpoint_every: int = 100,
    dry_run: bool = False,
    verbose: bool = True,
) -> DownloadStats:
    stats = DownloadStats()
    updates: dict[str, list[dict]] = {}
    since_checkpoint = 0
    seen_keys: dict[str, str] = {}

    total = len(rows) if max_files is None else min(len(rows), max_files)
    bar = tqdm(
        total=total,
        desc="download",
        unit="pdf",
        disable=None if verbose else True,
        smoothing=0.05,
    )

    def flush() -> None:
        if dry_run:
            updates.clear()
            return
        for season, recs in updates.items():
            store.upsert(season, recs)
            store.save(season)
        updates.clear()

    try:
        for row in rows.itertuples(index=False):
            if max_files is not None and stats.stored >= max_files:
                stats.stopped_early = True
                stats.reason = f"reached --max-files={max_files}"
                break

            stats.considered += 1
            bar.update(1)
            key = str(row.s3_key)
            expected_et = _parse_et(str(row.report_datetime_et))

            # A canonical key must identify exactly one report.
            if key in seen_keys and seen_keys[key] != str(row.original_url):
                raise RuntimeError(
                    f"canonical key collision: {key}\n"
                    f"  {seen_keys[key]}\n  {row.original_url}\n"
                    "Two distinct source URLs resolved to one key - the era model "
                    "is wrong. Refusing to overwrite."
                )
            seen_keys[key] = str(row.original_url)

            existing = storage.exists(key)
            if existing is not None and (
                pd.isna(row.content_length) or existing == int(row.content_length)
            ):
                stats.already += 1
                updates.setdefault(str(row.season), []).append(
                    {
                        "report_key": row.report_key,
                        "download_status": mf.DOWNLOAD_STORED,
                        "s3_key": key,
                        "bytes_stored": existing,
                    }
                )
                continue

            if dry_run:
                stats.stored += 1
                continue

            body, probe = client.fetch(str(row.original_url))
            if body is None or probe.verdict is not Verdict.EXISTS:
                stats.failed += 1
                stats.problems.append(f"{row.report_key}: fetch {probe.verdict.value}")
                updates.setdefault(str(row.season), []).append(
                    {
                        "report_key": row.report_key,
                        "download_status": mf.DOWNLOAD_FAILED,
                        "notes": probe.note or probe.verdict.value,
                    }
                )
                continue

            result = validate(body, expected_et)
            update = {
                "report_key": row.report_key,
                "sha256": result.sha256,
                "pdf_page_count": result.page_count,
                "validation_status": result.status,
                "pdf_header_datetime_et": (
                    result.header_datetime_et.isoformat()
                    if result.header_datetime_et
                    else None
                ),
            }

            if not result.ok:
                stats.invalid += 1
                stats.problems.append(
                    f"{row.report_key}: {result.status} {result.detail}"
                )
                update["download_status"] = mf.DOWNLOAD_INVALID
                updates.setdefault(str(row.season), []).append(update)
                continue

            storage.put(
                key,
                body,
                {
                    "original-url": str(row.original_url),
                    "original-filename": str(row.original_filename),
                    "report-datetime-et": str(row.report_datetime_et),
                    "report-datetime-utc": str(row.report_datetime_utc),
                    "source-era": str(row.source_era),
                    "sha256": result.sha256,
                },
            )
            stats.stored += 1
            stats.bytes_stored += len(body)
            update.update(
                {
                    "download_status": mf.DOWNLOAD_STORED,
                    "s3_key": key,
                    "bytes_stored": len(body),
                    "download_timestamp": datetime.now(UTC),
                }
            )
            updates.setdefault(str(row.season), []).append(update)

            since_checkpoint += 1
            bar.set_postfix_str(
                f"{stats.stored} stored, {stats.bytes_stored / 1024**2:.1f} MiB"
                + (f", {stats.invalid} invalid" if stats.invalid else ""),
                refresh=False,
            )
            if since_checkpoint >= checkpoint_every:
                flush()
                since_checkpoint = 0
    except ThrottledError as exc:
        stats.stopped_early = True
        stats.reason = str(exc)
    except KeyboardInterrupt:
        stats.stopped_early = True
        stats.reason = "interrupted by user"
    finally:
        bar.close()

    flush()
    return stats
