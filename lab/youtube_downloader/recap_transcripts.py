"""Wishper transcripts in the same JSON.gz/manifest layout as written reports.

See docs/pipeline/recap_transcripts.md. Imports do not access the network, AWS,
or the filesystem. Existing audio discovery and encoding stay in recap_audio.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import math
import re
import time
import zlib
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit

import requests

from lab.game_reports import season_reports as reports
from lab.youtube_downloader import recap_audio as audio

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ARTIFACT_TYPE = "audio_transcript"
# A saved transcript belongs to this game; a change here is a different game.
IDENTITY_FIELDS = ("date", "phase", "away", "home", "away_score", "home_score")


@dataclass(frozen=True)
class TranscriptConfig(reports.SeasonConfig):
    prefix: str = "train_data/game_recap_transcripts/v1"
    workers: int = 1
    checkpoint_every: int = 1
    api_base_url: str = "http://127.0.0.1:8000"
    api_key: str = field(default="", repr=False)
    api_timeout: float = 1800
    batch_size: int = 4
    chunk_minutes: int = 15
    busy_retries: int = 2
    profile: str = "wishper-v1"
    objects_dir: ClassVar[str] = "transcripts"

    def __post_init__(self):
        # Allow an empty bucket for local-only runs; retain report validation.
        reports.SeasonConfig(
            self.season_year,
            self.local_root,
            self.bucket or "local",
            prefix=self.prefix,
            phases=self.phases,
            workers=self.workers,
            checkpoint_every=self.checkpoint_every,
        )
        parsed = urlsplit(self.api_base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Use an HTTP(S) API base URL without credentials/query/fragment"
            )
        if not math.isfinite(self.api_timeout) or self.api_timeout <= 0:
            raise ValueError("API timeout must be finite and positive")
        if not 1 <= self.batch_size <= 16 or not 1 <= self.chunk_minutes <= 120:
            raise ValueError("batch_size must be 1–16; chunk_minutes must be 1–120")
        if self.workers != 1 or not 0 <= self.busy_retries <= 10:
            raise ValueError("Use one worker and 0–10 busy retries")
        # The profile is a path segment: each model/settings label is its own archive.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", self.profile):
            raise ValueError("Profile must be letters, digits, '.', '_' or '-'")

    @property
    def directory(self) -> Path:
        return (
            Path(self.local_root) / f"profile={self.profile}" / f"season={self.season}"
        )

    @property
    def s3_prefix(self) -> str:
        return f"{self.prefix.strip('/')}/profile={self.profile}/season={self.season}"

    @property
    def transcription(self) -> dict:
        # Settings that change the text. batch_size only changes throughput, so it
        # is recorded per document but may differ between runs of one archive.
        # Never persist the URL, credentials, or a claim about the server's model.
        return {
            "provider": "wishper",
            "profile": self.profile,
            "chunk_minutes": self.chunk_minutes,
        }


class WishperError(RuntimeError):
    def __init__(self, message: str, *, stop_run: bool = False):
        super().__init__(message)
        self.stop_run = stop_run


class WishperClient:
    def __init__(self, config: TranscriptConfig, session: requests.Session):
        self.config, self.session = config, session

    def transcribe(self, path: Path) -> str:
        size = path.stat().st_size
        if not 0 < size <= MAX_UPLOAD_BYTES:
            raise ValueError(
                "Wishper requires a nonempty audio file no larger than 200 MiB"
            )
        headers = (
            {"Authorization": f"Bearer {self.config.api_key}"}
            if self.config.api_key
            else {}
        )
        url = f"{self.config.api_base_url.rstrip('/')}/transcribe"
        for attempt in range(self.config.busy_retries + 1):
            try:
                # Reopen on every 503 retry: do not resend an exhausted stream.
                with path.open("rb") as source:
                    response = self.session.post(
                        url,
                        headers=headers,
                        files={"file": (path.name, source, "audio/ogg")},
                        data={
                            "batch_size": str(self.config.batch_size),
                            "chunk_minutes": str(self.config.chunk_minutes),
                        },
                        timeout=(10, self.config.api_timeout),
                        allow_redirects=False,
                    )
            except requests.RequestException as exc:
                # A timeout may leave a transcription running. Never auto-retry it.
                raise WishperError(
                    f"Wishper connection failed ({type(exc).__name__}); not retried",
                    stop_run=True,
                ) from None
            try:
                if response.status_code == 503 and attempt < self.config.busy_retries:
                    try:
                        pause = float(response.headers.get("Retry-After", "5"))
                    except ValueError:
                        pause = 5
                    if not math.isfinite(pause) or not 0 <= pause <= 30:
                        raise WishperError(
                            "Wishper is busy; Retry-After exceeds retry budget",
                            stop_run=True,
                        )
                    time.sleep(pause)
                    continue
                if response.status_code != 200:
                    raise WishperError(
                        f"Wishper returned HTTP {response.status_code}",
                        stop_run=response.status_code in {401, 403, 500, 503},
                    )
                try:
                    result = response.json()
                except ValueError:
                    raise WishperError("Wishper returned invalid JSON") from None
                text = result.get("text") if isinstance(result, dict) else None
                if not isinstance(text, str) or not text.strip():
                    raise WishperError("Wishper returned no nonempty transcript text")
                return text
            finally:
                response.close()
        raise AssertionError("unreachable")


def validate_manifest(
    manifest: dict, config: TranscriptConfig, *, promotion=False
) -> None:
    expected = {
        "schema_version": reports.SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "season": config.season,
        "s3_prefix": config.s3_prefix,
        "transcription": config.transcription,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Transcript manifest configuration differs; use another root/prefix"
        )
    if set(manifest.get("phases", [])) != set(config.phases):
        raise ValueError("Transcript manifest phase scope differs")
    if manifest.get("bucket") != config.bucket and not (
        promotion and manifest.get("bucket") == "" and config.bucket
    ):
        raise ValueError("Transcript manifest S3 destination differs")


def summarize(manifest: dict, game_date: date | None = None) -> dict:
    games = [
        g
        for g in manifest["games"].values()
        if game_date is None or g["date"] == game_date.isoformat()
    ]
    complete = {"uploaded"} if manifest["bucket"] else {"downloaded", "uploaded"}
    missing = [g["game_id"] for g in games if g["status"] not in complete]
    return {
        "season": manifest["season"],
        "date": game_date.isoformat() if game_date else None,
        "expected_games": len(games),
        "by_status": dict(sorted(Counter(g["status"] for g in games).items())),
        "missing_game_ids": missing,
        "all_transcripts_saved": not missing,
        "uploaded_bytes": sum(
            g["stored_bytes"] for g in games if g["status"] == "uploaded"
        ),
    }


def checkpoint(config: TranscriptConfig, manifest: dict) -> None:
    manifest["updated_at_utc"] = reports.utc_now()
    manifest["summary"] = summarize(manifest)
    reports.atomic_write(
        config.directory / "manifest.json", reports.json_bytes(manifest)
    )


def keep_transcript_audio(entry: dict, old: dict) -> None:
    """Keep a saved transcript when its game's audio was downloaded again.

    A re-download re-encodes the same recap, so the bytes (and hash) can differ.
    The transcript stays tied to the audio it was made from; S3 refuses to
    overwrite that audio, so it is still the archived copy.
    """
    if old.get("audio_sha256") and old["audio_sha256"] != entry["audio_sha256"]:
        print(
            f"{entry['game_id']}: source audio changed after transcription; "
            "keeping the transcript of the earlier audio",
            flush=True,
        )
        entry["audio_sha256"] = old["audio_sha256"]


def load_audio_manifest(
    config: audio.AudioConfig, store: audio.S3AudioStore | None
) -> dict:
    """Read the local audio manifest, or copy it from S3 when only S3 has it."""
    path = config.directory / "manifest.json"
    if path.is_file():
        return json.loads(path.read_bytes())
    remote = store.load_manifest() if store else None
    if remote is None:
        raise ValueError(
            f"No audio manifest in {config.directory} or S3; run the download stage"
        )
    # Audio files are restored one by one, checksum-verified, as they are needed.
    reports.atomic_write(path, reports.json_bytes(remote))
    return remote


def prepare_manifest(
    config: TranscriptConfig, audio_config: audio.AudioConfig, source: dict
) -> dict:
    audio.validate_manifest(source, audio_config)
    if config.season != audio_config.season or set(config.phases) != set(
        audio_config.phases
    ):
        raise ValueError("Audio and transcript season/phase scopes must match")
    origin = {
        "s3_prefix": audio_config.s3_prefix,
        "language": audio_config.language,
        "channel_id": audio.CHANNEL_ID,
        "audio_profile": audio.PROFILE,
    }
    path = config.directory / "manifest.json"
    previous = json.loads(path.read_bytes()) if path.is_file() else None
    if previous:
        validate_manifest(previous, config, promotion=True)
        if previous.get("audio_source") != origin:
            raise ValueError("Transcript audio source differs; use another root/prefix")
        if set(previous["games"]) - set(source["games"]):
            raise ValueError("Audio inventory lost previously inventoried games")
    games = {}
    for game_id, item in source["games"].items():
        entry = {
            key: item[key]
            for key in (
                "game_id",
                "date",
                "phase",
                "away",
                "home",
                "game_status",
                "away_score",
                "home_score",
            )
        }
        entry.update(
            status="pending" if item["game_status"] == 3 else "not_final",
            source_url=(item.get("recap") or {}).get("url", ""),
            s3_key=config.report_key(game_id),
            local_path=f"{config.objects_dir}/{game_id}.json.gz",
            audio_sha256=item.get("sha256"),
            attempts=0,
            last_attempt_at=None,
            sha256=None,
            stored_bytes=0,
            error="",
        )
        old = (previous or {}).get("games", {}).get(game_id)
        if old:
            if old.get("sha256"):
                if any(old.get(key) != entry[key] for key in IDENTITY_FIELDS):
                    raise ValueError(
                        f"Saved transcript source identity changed for {game_id}"
                    )
                keep_transcript_audio(entry, old)
            entry.update(
                {
                    key: old[key]
                    for key in (
                        "status",
                        "attempts",
                        "last_attempt_at",
                        "sha256",
                        "stored_bytes",
                        "error",
                    )
                }
            )
            if item["game_status"] == 3 and entry["status"] == "not_final":
                entry["status"] = "pending"
        games[game_id] = entry
    manifest = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "season": config.season,
        "season_year": config.season_year,
        "phases": list(config.phases),
        "bucket": config.bucket,
        "s3_prefix": config.s3_prefix,
        "audio_source": origin,
        "transcription": config.transcription,
        "schedule_source": source["schedule_source"],
        "schedule_refreshed_at_utc": source["schedule_refreshed_at_utc"],
        "created_at_utc": previous["created_at_utc"] if previous else reports.utc_now(),
        "games": games,
    }
    checkpoint(config, manifest)
    return manifest


def validate_transcript(body: bytes, entry: dict, config: TranscriptConfig) -> dict:
    document = json.loads(gzip.decompress(body))
    if not isinstance(document, dict) or not isinstance(document.get("audio"), dict):
        raise ValueError("Transcript document or audio provenance has an invalid shape")
    reports.validate_report(document, entry, config)
    transcription = document.get("transcription")
    if isinstance(transcription, dict):
        transcription = {k: v for k, v in transcription.items() if k != "batch_size"}
    if (
        document.get("artifact_type") != ARTIFACT_TYPE
        or transcription != config.transcription
        or (document.get("audio") or {}).get("sha256") != entry.get("audio_sha256")
    ):
        raise ValueError(
            "Transcript provenance does not match this audio/configuration"
        )
    return document


def local_transcript(entry: dict, config: TranscriptConfig) -> bytes | None:
    path = config.report_path(entry["game_id"])
    if not path.is_file():
        return None
    body = path.read_bytes()
    try:
        validate_transcript(body, entry, config)
        if entry.get("sha256") and hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError("Transcript checksum mismatch")
        return body
    except (ValueError, OSError, EOFError, zlib.error):
        path.replace(path.with_name(path.name + ".corrupt"))
        return None


class S3TranscriptStore(reports.S3ReportStore):
    def load_manifest(self) -> dict | None:
        manifest = super().load_manifest()
        if manifest:
            validate_manifest(manifest, self.config)
        return manifest

    def read_report(self, entry: dict) -> bytes:
        body = super().read_report(entry)
        validate_transcript(body, entry, self.config)
        return body

    def put_report(self, entry: dict, body: bytes) -> None:
        validate_transcript(body, entry, self.config)
        super().put_report(entry, body)


def load_audio(
    entry: dict, config: audio.AudioConfig, store: audio.S3AudioStore | None
) -> Path:
    body = audio.local_audio(entry, config)
    if body is None and store:
        body, recap = store.read_audio(entry)
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError("Remote audio differs from its manifest")
        reports.atomic_write(config.audio_path(entry["game_id"]), body)
        entry = {**entry, "recap": recap}
        body = audio.local_audio(entry, config)
    if body is None:
        raise ValueError("Verified audio is missing; rerun the download stage")
    return config.audio_path(entry["game_id"])


def pack_transcript(
    text: str,
    entry: dict,
    source: dict,
    config: TranscriptConfig,
    audio_config: audio.AudioConfig,
) -> bytes:
    recap = source["recap"]
    document = {
        "artifact_type": ARTIFACT_TYPE,
        **{
            key: entry[key]
            for key in ("game_id", "date", "away", "home", "away_score", "home_score")
        },
        "title": recap.get("title", ""),
        "source_url": recap["url"],
        "author": "",
        "bytitle": "",
        "agency": "",
        "published_at": recap.get("published_at_utc") or "",
        "published_at_utc": recap.get("published_at_utc") or "",
        "retrieved_at_utc": reports.utc_now(),
        "text": text,
        "word_count": len(text.split()),
        "recap": recap,
        "transcription": {**config.transcription, "batch_size": config.batch_size},
        "audio": {
            "sha256": source["sha256"],
            "stored_bytes": source["stored_bytes"],
            "bucket": audio_config.bucket,
            "s3_key": source["s3_key"],
            "local_path": source["local_path"],
            "language": audio_config.language,
        },
    }
    return reports.pack_report(document, entry, config)


def run_archive(
    config: TranscriptConfig,
    audio_config: audio.AudioConfig,
    *,
    store: S3TranscriptStore | None = None,
    audio_store: audio.S3AudioStore | None = None,
    game_date: date | None = None,
    max_games: int | None = None,
) -> dict:
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive")
    if bool(config.bucket) != bool(store):
        raise ValueError(
            "An S3 transcript store is required exactly when a bucket is configured"
        )
    if config.directory.resolve() == audio_config.directory.resolve():
        raise ValueError("Audio and transcripts must use separate local roots")
    if config.s3_prefix == audio_config.s3_prefix:
        raise ValueError("Audio and transcripts must use separate S3 prefixes")
    with ExitStack() as stack:
        # Keep the source manifest and audio immutable while this runner reads them.
        for directory in (config.directory, audio_config.directory):
            directory.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((directory / ".run.lock").open("a"))
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "Another run is processing this audio/transcript season"
                ) from exc
        source = load_audio_manifest(audio_config, audio_store)
        manifest = prepare_manifest(config, audio_config, source)
        if store:
            remote = store.load_manifest()
            if remote and remote.get("audio_source") != manifest["audio_source"]:
                raise ValueError("Remote transcript audio source differs")
            for entry in manifest["games"].values():
                old = (remote or {}).get("games", {}).get(entry["game_id"])
                if old and old.get("sha256"):
                    keep_transcript_audio(entry, old)
            # The written-report recovery code checks manifests and orphan uploads.
            reports.reconcile(config, manifest, store)
        session = stack.enter_context(requests.Session())
        client = WishperClient(config, session)
        processed, consecutive_errors, failure = 0, 0, None
        try:
            for game_id, entry in manifest["games"].items():
                if entry["game_status"] != 3 or (
                    game_date and entry["date"] != game_date.isoformat()
                ):
                    continue
                if store and entry["status"] == "uploaded":
                    continue
                body = local_transcript(entry, config)
                if not store and body is not None:
                    reports.record_download(entry, body, config)
                    continue
                item = source["games"][game_id]
                if body is None and (not item.get("sha256") or not item.get("recap")):
                    entry.update(
                        status="waiting_audio", error=f"Audio status: {item['status']}"
                    )
                    continue
                if max_games is not None and processed >= max_games:
                    break
                processed += 1
                stop = False
                try:
                    if body is None:
                        entry["attempts"] += 1
                        entry["last_attempt_at"] = reports.utc_now()
                        path = load_audio(item, audio_config, audio_store)
                        # A new transcript describes the audio it is made from.
                        entry["audio_sha256"] = item["sha256"]
                        body = pack_transcript(
                            client.transcribe(path), entry, item, config, audio_config
                        )
                    reports.record_download(entry, body, config)
                except Exception as exc:
                    entry.update(
                        status="transcription_error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    stop = isinstance(exc, WishperError) and exc.stop_run
                else:
                    if store:
                        try:
                            store.put_report(entry, body)
                            entry.update(status="uploaded", error="")
                        except Exception as exc:
                            entry.update(
                                status="upload_error",
                                error=f"{type(exc).__name__}: {exc}",
                            )
                checkpoint(config, manifest)
                print(
                    f"[{processed}] {entry['date']} {game_id}: {entry['status']}",
                    flush=True,
                )
                consecutive_errors = (
                    consecutive_errors + 1 if entry["status"].endswith("error") else 0
                )
                if stop or consecutive_errors >= 3:
                    raise RuntimeError(f"Transcription run stopped: {entry['error']}")
        except BaseException as exc:
            failure = exc
            raise
        finally:
            checkpoint(config, manifest)
            if store:
                try:
                    store.save_manifest(manifest)
                except Exception as exc:
                    if failure is None:
                        raise
                    print(f"Could not publish transcript manifest: {exc}", flush=True)
        print(json.dumps(summarize(manifest, game_date), indent=2), flush=True)
        return manifest
