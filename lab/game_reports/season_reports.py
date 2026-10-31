"""Resumable, single-season NBA.com recap archives with an S3 manifest.

This module has no network or filesystem side effects when imported. Use the
download_season_reports.py CLI to download or inspect one season. Report objects
are immutable; the manifest uses conditional writes.
"""

from __future__ import annotations

import base64
import fcntl
import gzip
import hashlib
import json
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from botocore.exceptions import ClientError
from nba_api.stats.endpoints import ScheduleLeagueV2

from lab.game_reports.get_game_reports import (
    create_session,
    fetch_page_props,
    parse_report,
    published_at_utc,
)

SCHEMA_VERSION = 1
PHASES = {
    "001": "preseason",
    "002": "regular_season",
    "004": "playoffs",
    "005": "play_in",
    "006": "cup_final",
}
DEFAULT_PHASES = ("regular_season", "play_in", "playoffs", "cup_final")
RETRYABLE = {"pending", "downloaded", "fetch_error", "upload_error"}


class ReportUnavailable(Exception):
    """The public page does not serve a written postgame report."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_bytes(value: dict) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SeasonConfig:
    season_year: int
    local_root: Path
    bucket: str
    prefix: str = "train_data/game_reports/v1"
    phases: tuple[str, ...] = DEFAULT_PHASES
    timeout: int = 30
    workers: int = 4
    request_interval: float = 0.35
    checkpoint_every: int = 10

    def __post_init__(self):
        if not 2000 <= self.season_year <= 2099:
            raise ValueError("season_year must be a four-digit season starting year")
        if not self.phases or set(self.phases) - set(PHASES.values()):
            raise ValueError("Choose at least one supported season phase")
        if self.workers < 1 or self.checkpoint_every < 1 or self.request_interval < 0:
            raise ValueError(
                "Workers/checkpoints must be positive and request_interval nonnegative"
            )
        if not self.bucket or not self.prefix.strip("/"):
            raise ValueError("An S3 bucket and nonempty prefix are required")

    @property
    def season(self) -> str:
        return f"{self.season_year}-{str(self.season_year + 1)[-2:]}"

    @property
    def directory(self) -> Path:
        return Path(self.local_root) / f"season={self.season}"

    @property
    def s3_prefix(self) -> str:
        return f"{self.prefix.strip('/')}/season={self.season}"

    def report_path(self, game_id: str) -> Path:
        return self.directory / "reports" / f"{game_id}.json.gz"

    def report_key(self, game_id: str) -> str:
        return f"{self.s3_prefix}/reports/{game_id}.json.gz"


def season_games(payload: dict, config: SeasonConfig) -> dict:
    schedule = payload.get("leagueSchedule") or {}
    if schedule.get("seasonYear") != config.season:
        raise ValueError("The NBA schedule does not match the requested season")
    games = {}
    for day in schedule.get("gameDates", []):
        for game in day.get("games", []):
            game_id = str(game.get("gameId") or "")
            phase = PHASES.get(game_id[:3])
            if phase not in config.phases:
                continue
            if (
                len(game_id) != 10
                or not game_id.isdigit()
                or game_id[3:5] != str(config.season_year)[-2:]
            ):
                raise ValueError(f"Invalid game ID for this season: {game_id}")
            game_date = date.fromisoformat(game["gameDateEst"][:10]).isoformat()
            away, home = game.get("awayTeam") or {}, game.get("homeTeam") or {}
            away_code, home_code = away.get("teamTricode"), home.get("teamTricode")
            final = game.get("gameStatus") == 3
            if final and (not away_code or not home_code):
                raise ValueError(f"Final game {game_id} has no teams")
            slug = f"{away_code}-vs-{home_code}-{game_id}".lower()
            entry = {
                "game_id": game_id,
                "date": game_date,
                "phase": phase,
                "away": away.get("teamName") or "",
                "home": home.get("teamName") or "",
                "game_status": game.get("gameStatus"),
                "source_url": f"https://www.nba.com/game/{slug}" if final else "",
                "status": "pending" if final else "not_final",
                "attempts": 0,
                "last_attempt_at": None,
                "error": "",
                "s3_key": config.report_key(game_id),
                "sha256": None,
                "stored_bytes": 0,
            }
            if game_id in games and games[game_id] != entry:
                raise ValueError(f"Conflicting schedule entries for {game_id}")
            games[game_id] = entry
    if not games:
        raise ValueError("The schedule has no games in the requested season scope")
    return dict(sorted(games.items(), key=lambda item: (item[1]["date"], item[0])))


def summarize(manifest: dict) -> dict:
    counts = Counter(entry["status"] for entry in manifest["games"].values())
    return {
        "expected_games": len(manifest["games"]),
        "by_status": dict(sorted(counts.items())),
        "uploaded_bytes": sum(
            g["stored_bytes"]
            for g in manifest["games"].values()
            if g["status"] == "uploaded"
        ),
        "missing_game_ids": [
            g["game_id"]
            for g in manifest["games"].values()
            if g["status"] != "uploaded"
        ],
        "all_reports_uploaded": counts.get("uploaded", 0) == len(manifest["games"]),
    }


def checkpoint(config: SeasonConfig, manifest: dict) -> None:
    manifest["updated_at_utc"] = utc_now()
    manifest["summary"] = summarize(manifest)
    atomic_write(config.directory / "manifest.json", json_bytes(manifest))


def validate_manifest(manifest: dict, config: SeasonConfig) -> None:
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("season") != config.season
    ):
        raise ValueError("Manifest schema or season mismatch")
    if set(manifest.get("phases", [])) != set(config.phases):
        raise ValueError(
            "Manifest phase scope differs; use a separate prefix/local root for another scope"
        )
    if (
        manifest.get("bucket") != config.bucket
        or manifest.get("s3_prefix") != config.s3_prefix
    ):
        raise ValueError("Manifest S3 destination differs from the configuration")


def prepare_season(config: SeasonConfig, payload: dict | None = None) -> dict:
    """Refresh the expected-game inventory, retaining same-game checkpoints."""
    if payload is None:
        payload = ScheduleLeagueV2(
            league_id="00", season=config.season, timeout=config.timeout
        ).get_dict()
    games = season_games(payload, config)
    existing_path = config.directory / "manifest.json"
    old = json.loads(existing_path.read_bytes()) if existing_path.is_file() else None
    if old:
        validate_manifest(old, config)
        for game_id, game in games.items():
            previous = old["games"].get(game_id)
            if previous:
                state = {
                    key: previous[key]
                    for key in (
                        "status",
                        "attempts",
                        "last_attempt_at",
                        "error",
                        "sha256",
                        "stored_bytes",
                    )
                }
                if game["game_status"] == 3 and state["status"] == "not_final":
                    state["status"] = "pending"
                game.update(state)
        # A disappearing scheduled game must not silently shrink coverage.
        disappeared = set(old["games"]) - set(games)
        if disappeared:
            raise ValueError(
                f"Schedule lost previously inventoried games: {sorted(disappeared)}"
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "season": config.season,
        "season_year": config.season_year,
        "phases": list(config.phases),
        "bucket": config.bucket,
        "s3_prefix": config.s3_prefix,
        "schedule_source": "NBA ScheduleLeagueV2",
        "schedule_refreshed_at_utc": utc_now(),
        "created_at_utc": old["created_at_utc"] if old else utc_now(),
        "games": games,
    }
    checkpoint(config, manifest)
    return manifest


def load_report(body: bytes) -> dict:
    """Decode a stored report; fills published_at_utc for reports saved without it."""
    report = json.loads(gzip.decompress(body))
    report.setdefault("published_at_utc", published_at_utc(report.get("published_at")))
    return report


def pack_report(report: dict, entry: dict, config: SeasonConfig) -> bytes:
    # The text field already preserves paragraph breaks; avoid duplicating it.
    document = {key: value for key, value in report.items() if key != "paragraphs"}
    document.setdefault(
        "published_at_utc", published_at_utc(document.get("published_at"))
    )
    document.update(
        schema_version=SCHEMA_VERSION, season=config.season, phase=entry["phase"]
    )
    validate_report(document, entry, config)
    return gzip.compress(json_bytes(document), mtime=0)


def validate_report(report: dict, entry: dict, config: SeasonConfig) -> None:
    if report.get("game_id") != entry["game_id"] or report.get("date") != entry["date"]:
        raise ValueError("Report game ID or date mismatch")
    if (
        report.get("season") != config.season
        or report.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Report schema or season mismatch")
    if not isinstance(report.get("text"), str) or not report["text"].strip():
        raise ValueError("Report has no written article text")


class RequestPacer:
    def __init__(self, interval: float):
        self.interval = interval
        self.lock = threading.Lock()
        self.next_request = 0.0

    def wait(self):
        with self.lock:
            pause = max(0, self.next_request - time.monotonic())
            if pause:
                time.sleep(pause)
            self.next_request = time.monotonic() + self.interval


def fetch_report(
    entry: dict, config: SeasonConfig, pacer: RequestPacer
) -> bytes | None:
    # Each worker owns its requests session; sessions are not shared across threads.
    pacer.wait()
    with create_session() as session:
        props = fetch_page_props(session, entry["source_url"], config.timeout)
    if not props.get("game") and "gameCardFeed" in props:
        raise ReportUnavailable(
            "NBA.com returned the games index instead of this game's page"
        )
    report = parse_report(props, entry, date.fromisoformat(entry["date"]))
    if report is None and (props.get("story") or {}).get("storyType") == "preview":
        raise ReportUnavailable(
            "NBA.com only serves a pregame preview, not a postgame report"
        )
    return None if report is None else pack_report(report, entry, config)


def use_local_report(entry: dict, config: SeasonConfig) -> bytes | None:
    """Return a valid local report; set aside an unusable one so it is refetched."""
    path = config.report_path(entry["game_id"])
    if path.is_file():
        body = path.read_bytes()
        try:
            validate_report(json.loads(gzip.decompress(body)), entry, config)
            if (
                entry.get("sha256")
                and hashlib.sha256(body).hexdigest() != entry["sha256"]
            ):
                raise ValueError("checksum mismatch")
            return body
        except (ValueError, OSError, EOFError, zlib.error) as exc:
            # A corrupt file would otherwise fail the same way on every rerun.
            quarantine = path.with_name(path.name + ".corrupt")
            path.replace(quarantine)
            print(
                f"Local report for {entry['game_id']} unusable ({exc}); "
                f"moved to {quarantine.name}, fetching again",
                flush=True,
            )
    # Reuse the daily prototype's JSON without fetching the same article again.
    daily = Path(config.local_root).parent / entry["date"] / f"{entry['game_id']}.json"
    if daily.is_file():
        try:
            return pack_report(json.loads(daily.read_bytes()), entry, config)
        except (ValueError, KeyError) as exc:
            # Leave the prototype's own files alone; just fetch the article.
            print(f"Ignoring daily report {daily} ({exc})", flush=True)
    return None


def record_download(entry: dict, body: bytes, config: SeasonConfig) -> None:
    atomic_write(config.report_path(entry["game_id"]), body)
    digest = hashlib.sha256(body).hexdigest()
    status = (
        "uploaded"
        if entry["status"] == "uploaded" and entry.get("sha256") == digest
        else "downloaded"
    )
    entry.update(
        status=status,
        sha256=digest,
        stored_bytes=len(body),
        error="",
    )


class S3ReportStore:
    def __init__(self, client, config: SeasonConfig):
        self.client, self.config = client, config
        self.manifest_etag = None

    @property
    def manifest_key(self) -> str:
        return f"{self.config.s3_prefix}/manifest.json"

    def load_manifest(self) -> dict | None:
        try:
            response = self.client.get_object(
                Bucket=self.config.bucket, Key=self.manifest_key
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        self.manifest_etag = response["ETag"]
        manifest = json.loads(response["Body"].read())
        validate_manifest(manifest, self.config)
        return manifest

    def inventory(self) -> dict[str, int]:
        objects = {}
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.config.bucket, Prefix=f"{self.config.s3_prefix}/reports/"
        ):
            objects.update(
                {item["Key"]: item["Size"] for item in page.get("Contents", [])}
            )
        return objects

    def read_report(self, entry: dict) -> bytes:
        response = self.client.get_object(
            Bucket=self.config.bucket, Key=entry["s3_key"]
        )
        body = response["Body"].read()
        if hashlib.sha256(body).hexdigest() != response.get("Metadata", {}).get(
            "sha256"
        ):
            raise ValueError(f"S3 checksum metadata mismatch for {entry['game_id']}")
        validate_report(json.loads(gzip.decompress(body)), entry, self.config)
        return body

    def put_report(self, entry: dict, body: bytes) -> None:
        digest = hashlib.sha256(body).digest()
        try:
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=entry["s3_key"],
                Body=body,
                ContentType="application/json",
                ContentEncoding="gzip",
                StorageClass="STANDARD",
                ChecksumSHA256=base64.b64encode(digest).decode(),
                IfNoneMatch="*",
                Metadata={"sha256": digest.hex(), "game-id": entry["game_id"]},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {
                "PreconditionFailed",
                "412",
                "ConditionalRequestConflict",
                "409",
            }:
                raise
            existing = self.client.head_object(
                Bucket=self.config.bucket, Key=entry["s3_key"]
            )
            if existing.get("Metadata", {}).get("sha256") != digest.hex() or existing[
                "ContentLength"
            ] != len(body):
                raise ValueError(
                    f"Refusing to overwrite different S3 content for {entry['game_id']}"
                ) from exc

    def save_manifest(self, manifest: dict) -> None:
        # Only publish at run end/interruption: avoid many full manifest versions.
        condition = (
            {"IfMatch": self.manifest_etag}
            if self.manifest_etag
            else {"IfNoneMatch": "*"}
        )
        response = self.client.put_object(
            Bucket=self.config.bucket,
            Key=self.manifest_key,
            Body=json_bytes(manifest),
            ContentType="application/json",
            StorageClass="STANDARD",
            **condition,
        )
        self.manifest_etag = response["ETag"]


def reconcile(config: SeasonConfig, manifest: dict, store: S3ReportStore) -> dict:
    """Recover S3-only progress and objects uploaded before a lost checkpoint."""
    remote = store.load_manifest()
    inventory = store.inventory()
    for game_id, entry in manifest["games"].items():
        previous = (remote or {}).get("games", {}).get(game_id)
        present = entry["s3_key"] in inventory
        if previous and (
            previous["date"] != entry["date"] or previous["phase"] != entry["phase"]
        ):
            raise ValueError(f"Local/S3 game identity differs for {game_id}")
        if previous and previous["status"] == "uploaded" and present:
            if inventory[entry["s3_key"]] != previous["stored_bytes"]:
                raise ValueError(f"S3 object size mismatch for {game_id}")
            entry.update(
                {
                    key: previous[key]
                    for key in (
                        "status",
                        "sha256",
                        "stored_bytes",
                        "attempts",
                        "last_attempt_at",
                        "error",
                    )
                }
            )
        elif present:
            # An upload may have succeeded before the local/S3 manifest was saved.
            body = store.read_report(entry)
            record_download(entry, body, config)
            entry["status"] = "uploaded"
        elif entry["status"] == "uploaded":
            entry.update(
                status=(
                    "downloaded" if config.report_path(game_id).is_file() else "pending"
                ),
                error="S3 object missing; scheduled for recovery",
            )
        elif (
            previous
            and entry["status"] == "pending"
            and previous["status"] in {"not_available", "fetch_error", "upload_error"}
        ):
            entry.update(
                {
                    key: previous[key]
                    for key in (
                        "status",
                        "attempts",
                        "last_attempt_at",
                        "error",
                        "sha256",
                        "stored_bytes",
                    )
                }
            )
    checkpoint(config, manifest)
    return manifest


def process_game(
    entry: dict, config: SeasonConfig, store: S3ReportStore, pacer: RequestPacer
) -> dict:
    entry = dict(entry)
    try:
        body = use_local_report(entry, config)
        if body is None:
            entry["attempts"] += 1
            entry["last_attempt_at"] = utc_now()
            body = fetch_report(entry, config, pacer)
        if body is None:
            entry.update(
                status="not_available",
                error="No written postgame story in the NBA.com page",
            )
            return entry
        record_download(entry, body, config)
    except ReportUnavailable as exc:
        entry.update(status="not_available", error=str(exc))
        return entry
    except Exception as exc:
        entry.update(status="fetch_error", error=f"{type(exc).__name__}: {exc}")
        return entry
    try:
        store.put_report(entry, body)
        entry.update(status="uploaded", error="")
    except Exception as exc:
        entry.update(status="upload_error", error=f"{type(exc).__name__}: {exc}")
    return entry


def download_season(
    config: SeasonConfig,
    manifest: dict,
    store: S3ReportStore,
    *,
    retry_unavailable: bool = False,
    max_games: int | None = None,
) -> dict:
    """Run one season only; reruns retry failed/missing work and skip uploads."""
    config.directory.mkdir(parents=True, exist_ok=True)
    with (config.directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another local run is already processing this season"
            ) from exc
        reconcile(config, manifest, store)
        eligible = RETRYABLE | ({"not_available"} if retry_unavailable else set())
        entries = [
            g
            for g in manifest["games"].values()
            if g["game_status"] == 3 and g["status"] in eligible
        ]
        if max_games is not None:
            if max_games < 1:
                raise ValueError("max_games must be positive")
            entries = entries[:max_games]
        print(f"Season {config.season}: {len(entries)} reports to process", flush=True)
        pacer = RequestPacer(config.request_interval)
        executor = ThreadPoolExecutor(max_workers=config.workers)
        remaining = iter(entries)
        pending = {}
        completed, consecutive_errors = 0, 0
        failure = None
        try:
            for entry in list(entries[: config.workers]):
                next(remaining)
                pending[executor.submit(process_game, entry, config, store, pacer)] = (
                    entry["game_id"]
                )
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    game_id = pending.pop(future)
                    updated = future.result()
                    manifest["games"][game_id] = updated
                    completed += 1
                    consecutive_errors = (
                        consecutive_errors + 1
                        if updated["status"].endswith("error")
                        else 0
                    )
                    if (
                        completed % config.checkpoint_every == 0
                        or updated["status"] != "uploaded"
                    ):
                        checkpoint(config, manifest)
                        print(
                            f"Processed {completed}/{len(entries)}: {summarize(manifest)['by_status']}",
                            flush=True,
                        )
                    if consecutive_errors >= 12:
                        raise RuntimeError(
                            "Stopped after 12 consecutive failures; inspect manifest errors before retrying"
                        )
                    following = next(remaining, None)
                    if following is not None:
                        pending[
                            executor.submit(
                                process_game, following, config, store, pacer
                            )
                        ] = following["game_id"]
        except BaseException as exc:
            failure = exc
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            # Include uploads that finished while an interrupt/error was handled.
            for future, game_id in pending.items():
                if (
                    future.done()
                    and not future.cancelled()
                    and future.exception() is None
                ):
                    manifest["games"][game_id] = future.result()
            checkpoint(config, manifest)
            try:
                store.save_manifest(manifest)
            except Exception as exc:
                if failure is None:
                    raise
                # Keep the original error visible; the local checkpoint is saved
                # and the next run's reconcile recovers S3 state.
                print(
                    f"Could not publish the S3 manifest ({type(exc).__name__}: {exc})",
                    flush=True,
                )
        summary = dict(manifest["summary"])
        missing = summary.pop("missing_game_ids")
        summary.update(missing_count=len(missing), first_missing_game_ids=missing[:10])
        print(json.dumps(summary, indent=2), flush=True)
        return manifest
