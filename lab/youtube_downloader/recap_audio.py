"""Resumable NBA recap audio archives, keyed by official NBA game ID.

Imports have no network or filesystem side effects. The audio profile is fixed
to Opus, 48 kbit/s VBR, mono. YouTube dependencies are loaded only when needed.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from botocore.exceptions import ClientError
from nba_api.stats.endpoints import ScheduleLeagueV2

from lab.game_reports.season_reports import (
    DEFAULT_PHASES,
    RequestPacer,
    SeasonConfig,
    atomic_write,
    json_bytes,
    season_games,
    utc_now,
)

CHANNEL_ID = "UCLd4dSmXdrJykO_hgOzbfPw"
PROFILE = {"codec": "opus", "bitrate_kbps": 48, "channels": 1, "sample_rate": 48000}
STATE_FIELDS = (
    "status",
    "attempts",
    "last_attempt_at",
    "error",
    "sha256",
    "stored_bytes",
    "recap",
)
RETRYABLE = {"pending", "not_found", "downloaded", "fetch_error", "upload_error"}


@dataclass(frozen=True)
class AudioConfig:
    season_year: int
    local_root: Path
    bucket: str = ""
    prefix: str = "train_data/game_recaps/v1"
    phases: tuple[str, ...] = DEFAULT_PHASES
    language: str = "en"
    request_interval: float = 2.0
    search_results: int = 25
    upload_window_days: int = 2
    js_runtime: str = "node"

    def __post_init__(self):
        # Keep season/phase validation consistent with the written archive.
        SeasonConfig(
            self.season_year,
            self.local_root,
            self.bucket or "local",
            phases=self.phases,
        )
        if not self.prefix.strip("/"):
            raise ValueError("S3 prefix must not be empty")
        if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]+)*", self.language):
            raise ValueError("Use a language code such as en or es")
        if not math.isfinite(self.request_interval) or self.request_interval < 0:
            raise ValueError("request_interval must be finite and nonnegative")
        if self.search_results < 1 or self.upload_window_days < 0:
            raise ValueError(
                "search_results must be positive; upload window cannot be negative"
            )
        if self.js_runtime not in {"node", "deno"}:
            raise ValueError("js_runtime must be node or deno")

    @property
    def season(self) -> str:
        return f"{self.season_year}-{(self.season_year + 1) % 100:02d}"

    @property
    def directory(self) -> Path:
        return Path(self.local_root) / f"season={self.season}"

    @property
    def s3_prefix(self) -> str:
        return f"{self.prefix.strip('/')}/season={self.season}"

    def audio_path(self, game_id: str) -> Path:
        return self.directory / "audio" / f"{game_id}.opus"

    def audio_key(self, game_id: str) -> str:
        return f"{self.s3_prefix}/audio/{game_id}.opus"


def validate_manifest(
    manifest: dict, config: AudioConfig, *, allow_local_promotion: bool = False
) -> None:
    expected = {
        "schema_version": 1,
        "season": config.season,
        "audio_profile": PROFILE,
        "channel_id": CHANNEL_ID,
        "language": config.language,
        "bucket": config.bucket,
        "s3_prefix": config.s3_prefix,
    }
    if allow_local_promotion and not manifest.get("bucket") and config.bucket:
        expected["bucket"] = ""
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Manifest season, audio profile, language or destination differs; use a separate local root/prefix"
        )
    if set(manifest.get("phases", [])) != set(config.phases):
        raise ValueError(
            "Manifest phase scope differs; use a separate local root/prefix"
        )


def summarize(manifest: dict, game_date: date | None = None) -> dict:
    games = [
        g
        for g in manifest["games"].values()
        if game_date is None or g["date"] == game_date.isoformat()
    ]
    counts = Counter(g["status"] for g in games)
    complete = {"uploaded"} if manifest["bucket"] else {"downloaded", "uploaded"}
    return {
        "season": manifest["season"],
        "date": game_date.isoformat() if game_date else None,
        "expected_games": len(games),
        "by_status": dict(sorted(counts.items())),
        "stored_bytes": sum(
            g["stored_bytes"] for g in games if g["status"] in complete
        ),
        "missing_game_ids": [
            g["game_id"] for g in games if g["status"] not in complete
        ],
        "all_audio_saved": all(g["status"] in complete for g in games),
    }


def checkpoint(config: AudioConfig, manifest: dict) -> None:
    manifest["updated_at_utc"] = utc_now()
    manifest["summary"] = summarize(manifest)
    atomic_write(config.directory / "manifest.json", json_bytes(manifest))


def prepare_manifest(config: AudioConfig, payload: dict | None = None) -> dict:
    """Refresh the full season inventory even when downloading only one date.

    The caller must hold the season lock until the run has finished.
    """
    if payload is None:
        payload = ScheduleLeagueV2(
            league_id="00", season=config.season, timeout=30
        ).get_dict()
    schedule_config = SeasonConfig(
        config.season_year,
        config.local_root,
        config.bucket or "local",
        phases=config.phases,
    )
    games = season_games(payload, schedule_config)
    raw = {
        str(g["gameId"]): g
        for d in payload["leagueSchedule"]["gameDates"]
        for g in d.get("games", [])
    }
    for game_id, entry in games.items():
        entry["nba_url"] = entry.pop("source_url")
        entry.update(
            s3_key=config.audio_key(game_id),
            local_path=f"audio/{game_id}.opus",
            recap=None,
        )
        for side in ("away", "home"):
            team = raw[game_id].get(f"{side}Team") or {}
            entry[f"{side}_score"] = team.get("score")
            entry[f"{side}_team_id"] = team.get("teamId")
    path = config.directory / "manifest.json"
    old = json.loads(path.read_bytes()) if path.exists() else None
    if old:
        validate_manifest(old, config, allow_local_promotion=True)
        if set(old["games"]) - set(games):
            raise ValueError("Schedule lost previously inventoried games")
        for game_id, entry in games.items():
            previous = old["games"].get(game_id)
            if previous:
                if previous.get("sha256") and any(
                    previous[k] != entry[k]
                    for k in ("date", "away", "home", "away_score", "home_score")
                ):
                    raise ValueError(f"Saved game identity changed for {game_id}")
                entry.update({k: previous[k] for k in STATE_FIELDS})
                if entry["game_status"] == 3 and entry["status"] == "not_final":
                    entry["status"] = "pending"
    manifest = {
        "schema_version": 1,
        "season": config.season,
        "phases": list(config.phases),
        "audio_profile": dict(PROFILE),
        "language": config.language,
        "channel_id": CHANNEL_ID,
        "bucket": config.bucket,
        "s3_prefix": config.s3_prefix,
        "schedule_source": "NBA ScheduleLeagueV2",
        "schedule_refreshed_at_utc": utc_now(),
        "created_at_utc": old["created_at_utc"] if old else utc_now(),
        "games": games,
    }
    checkpoint(config, manifest)
    return manifest


def require_tools(config: AudioConfig) -> None:
    for name in ("ffmpeg", "ffprobe", config.js_runtime):
        if not shutil.which(name):
            raise RuntimeError(f"Install {name} and add it to PATH")
    for name in ("yt_dlp", "yt_dlp_ejs"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(
                "Install the media dependency group: poetry install --with media"
            )


def youtube_options(config: AudioConfig) -> dict:
    return {
        "quiet": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "extractor_retries": 2,
        "js_runtimes": {config.js_runtime: {}},
        "cachedir": str(config.directory / ".yt-dlp-cache"),
    }


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def matches_game(title: str, entry: dict) -> bool:
    text = normalized(title)
    if not text.startswith("game recap "):
        return False
    for side in ("away", "home"):
        team, score = normalized(entry[side]), entry[f"{side}_score"]
        if not team or score is None:
            return False
        if not re.search(rf"\b{re.escape(team)}\s+{int(score)}\b", text):
            return False
    return True


def publication(info: dict) -> tuple[date | None, str | None]:
    timestamp = info.get("release_timestamp") or info.get("timestamp")
    if timestamp is not None:
        instant = datetime.fromtimestamp(timestamp, UTC)
        return (
            instant.astimezone(ZoneInfo("America/New_York")).date(),
            instant.isoformat(),
        )
    try:
        # upload_date has UTC day precision only; never invent an exact instant.
        return datetime.strptime(info.get("upload_date") or "", "%Y%m%d").date(), None
    except ValueError:
        return None, None


def verify_video(info: dict, entry: dict, config: AudioConfig) -> bool:
    uploaded, _ = publication(info)
    day = date.fromisoformat(entry["date"])
    return bool(
        info.get("channel_id") == CHANNEL_ID
        and matches_game(info.get("title", ""), entry)
        and uploaded is not None
        and day <= uploaded <= day + timedelta(days=config.upload_window_days)
    )


def find_recap(entry: dict, config: AudioConfig, pacer: RequestPacer) -> dict | None:
    """Try channel search then broader search, verifying every accepted video."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    query = f'Game Recap {entry["away"]} {entry["away_score"]} {entry["home"]} {entry["home_score"]}'
    searches = [
        f"https://www.youtube.com/channel/{CHANNEL_ID}/search?"
        + urlencode({"query": query}),
        f"ytsearch{config.search_results}:{query} Motion Station",
    ]
    errors, seen = [], set()
    for search in searches:
        try:
            pacer.wait()
            with YoutubeDL(
                {
                    **youtube_options(config),
                    "extract_flat": True,
                    "skip_download": True,
                    "playlistend": config.search_results,
                }
            ) as ydl:
                result = ydl.extract_info(search, download=False)
            if not isinstance(result, dict) or "entries" not in result:
                raise RuntimeError("YouTube returned no search response")
            candidates = [
                e
                for e in result["entries"]
                if e
                and e.get("id")
                and matches_game(e.get("title", ""), entry)
                and e.get("channel_id") in (None, CHANNEL_ID)
            ]
            candidates.sort(
                key=lambda e: (
                    "extended" in normalized(e.get("title", "")),
                    -int(e.get("view_count") or 0),
                )
            )
            verified = []
            for candidate in candidates:
                video_id = candidate["id"]
                if video_id in seen:
                    continue
                pacer.wait()
                try:
                    with YoutubeDL(
                        {
                            **youtube_options(config),
                            "skip_download": True,
                            "ignore_no_formats_error": True,
                        }
                    ) as ydl:
                        info = ydl.extract_info(
                            f"https://www.youtube.com/watch?v={video_id}",
                            download=False,
                        )
                except DownloadError as exc:
                    errors.append(str(exc))
                    continue
                seen.add(video_id)
                if info and verify_video(info, entry, config):
                    verified.append(info)
            standard = [i for i in verified if "extended" not in normalized(i["title"])]
            choices = standard or verified
            if len(choices) > 1:
                raise ValueError(f"Ambiguous recap match: {[i['id'] for i in choices]}")
            if choices:
                return choices[0]
        except DownloadError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(f"YouTube search/verification failed: {errors[-1]}")
    return None


def select_audio(info: dict, language: str) -> dict:
    """Select an audio-only track; do not silently substitute another language."""
    formats = [
        f
        for f in info.get("formats", [])
        if f.get("vcodec") == "none"
        and f.get("acodec") not in (None, "none")
        and f.get("url")
    ]
    matching = [
        f
        for f in formats
        if (f.get("language") or "").lower().split("-")[0]
        == language.lower().split("-")[0]
    ]
    # Older videos may have no per-track language tags. Only accept untagged
    # tracks if the video's language agrees, or no language is known anywhere.
    if (
        not matching
        and (
            not info.get("language")
            or info["language"].split("-")[0] == language.split("-")[0]
        )
        and not any(f.get("language") for f in formats)
    ):
        matching = formats
    if not matching:
        raise RuntimeError(f"No audio-only track for language {language}")
    return max(
        matching,
        key=lambda f: (
            f.get("language_preference") or 0,
            f.get("quality") or 0,
            f.get("abr") or f.get("tbr") or 0,
        ),
    )


def inspect_audio(
    path: Path, expected_duration: float | None = None, *, decode: bool = False
) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,codec_type,channels,sample_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    duration = float(info["format"]["duration"])
    if (
        len(streams) != 1
        or streams[0].get("codec_name") != "opus"
        or streams[0].get("channels") != 1
        or streams[0].get("sample_rate") != "48000"
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("Audio does not match the Opus mono 48 kHz profile")
    if expected_duration and abs(duration - expected_duration) > max(
        2, expected_duration * 0.02
    ):
        raise ValueError("Audio duration differs from the recap")
    if decode:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=300,
        )
    return duration


def encode_audio(
    source: Path, destination: Path, expected_duration: float | None = None
) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".encode-", dir=destination.parent) as tmp:
        temporary = Path(tmp) / "audio.opus"
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-map_metadata",
                "-1",
                "-ac",
                "1",
                "-ar",
                "48000",
                "-c:a",
                "libopus",
                "-b:a",
                "48k",
                "-vbr",
                "on",
                str(temporary),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode:
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr.strip()}")
        duration = inspect_audio(temporary, expected_duration, decode=True)
        temporary.replace(destination)
    return duration


def fetch_audio(entry: dict, config: AudioConfig, pacer: RequestPacer) -> bool:
    from yt_dlp import YoutubeDL

    info = find_recap(entry, config, pacer)
    if info is None:
        return False
    selected = {}

    def choose_audio(context):
        track = select_audio(context, config.language)
        selected.update(track)
        yield track

    def check_match(metadata, *, incomplete=False):
        if not incomplete and not verify_video(metadata, entry, config):
            return "Recap metadata no longer matches the expected game"
        return None

    with tempfile.TemporaryDirectory(prefix=".download-", dir=config.directory) as tmp:
        options = {
            **youtube_options(config),
            "format": choose_audio,
            "match_filter": check_match,
            "outtmpl": str(Path(tmp) / "source.%(ext)s"),
            "skip_download": False,
            "noprogress": True,
        }
        pacer.wait()
        with YoutubeDL(options) as ydl:
            # Extract and download in the same session. Reusing signed URLs from
            # discovery can produce HTTP 403. Select by language again because
            # numeric format IDs can change between extractions.
            downloaded = ydl.extract_info(
                f"https://www.youtube.com/watch?v={info['id']}", download=True
            )
        if (
            not downloaded
            or not selected
            or not verify_video(downloaded, entry, config)
        ):
            raise RuntimeError("Downloaded recap metadata does not match the game")
        recap = {
            "video_id": downloaded["id"],
            "url": f"https://www.youtube.com/watch?v={downloaded['id']}",
            "title": downloaded["title"],
            "channel_id": CHANNEL_ID,
            "upload_date": downloaded.get("upload_date"),
            "published_at_utc": publication(downloaded)[1],
            "retrieved_at_utc": utc_now(),
            "language": selected.get("language") or downloaded.get("language"),
            "format_id": selected["format_id"],
            "source_codec": selected["acodec"],
            "source_duration_seconds": downloaded.get("duration"),
        }
        files = [
            p for p in Path(tmp).glob("source.*") if p.suffix not in {".part", ".ytdl"}
        ]
        if len(files) != 1:
            raise RuntimeError("YouTube did not produce one complete source audio file")
        recap["duration_seconds"] = encode_audio(
            files[0], config.audio_path(entry["game_id"]), downloaded.get("duration")
        )
    entry["recap"] = recap
    return True


def record_audio(entry: dict, config: AudioConfig) -> bytes:
    path = config.audio_path(entry["game_id"])
    body = path.read_bytes()
    entry.update(
        status="downloaded",
        sha256=hashlib.sha256(body).hexdigest(),
        stored_bytes=len(body),
        error="",
    )
    return body


def local_audio(entry: dict, config: AudioConfig) -> bytes | None:
    path = config.audio_path(entry["game_id"])
    if not path.is_file():
        return None
    try:
        body = path.read_bytes()
        if not entry.get("recap") or not entry.get("sha256"):
            raise ValueError("Local audio has no matching checkpoint metadata")
        if hashlib.sha256(body).hexdigest() != entry["sha256"]:
            raise ValueError("Local audio checksum mismatch")
        inspect_audio(path, entry["recap"].get("duration_seconds"))
        return body
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as exc:
        path.replace(path.with_name(path.name + ".corrupt"))
        print(f"Quarantined {path.name}: {exc}", flush=True)
        return None


class S3AudioStore:
    def __init__(self, client, config: AudioConfig):
        self.client, self.config, self.manifest_etag = client, config, None

    def load_manifest(self) -> dict | None:
        try:
            response = self.client.get_object(
                Bucket=self.config.bucket, Key=f"{self.config.s3_prefix}/manifest.json"
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise
        self.manifest_etag = response["ETag"]
        manifest = json.loads(response["Body"].read())
        validate_manifest(manifest, self.config)
        return manifest

    def inventory(self) -> dict:
        return {
            obj["Key"]: obj["Size"]
            for page in self.client.get_paginator("list_objects_v2").paginate(
                Bucket=self.config.bucket, Prefix=f"{self.config.s3_prefix}/audio/"
            )
            for obj in page.get("Contents", [])
        }

    def read_audio(self, entry: dict) -> tuple[bytes, dict]:
        response = self.client.get_object(
            Bucket=self.config.bucket, Key=entry["s3_key"]
        )
        body, metadata = response["Body"].read(), response.get("Metadata", {})
        if (
            metadata.get("sha256") != hashlib.sha256(body).hexdigest()
            or metadata.get("game-id") != entry["game_id"]
            or metadata.get("date") != entry["date"]
            or metadata.get("profile") != "opus-mono-48k-v1"
            or metadata.get("language") != self.config.language
        ):
            raise ValueError(f"S3 identity/checksum mismatch for {entry['game_id']}")
        recap = json.loads(base64.b64decode(metadata["recap"]))
        if recap.get("channel_id") != CHANNEL_ID:
            raise ValueError("S3 recap channel mismatch")
        return body, recap

    def put_audio(self, entry: dict, body: bytes) -> None:
        digest = hashlib.sha256(body).digest()
        metadata = {
            "sha256": digest.hex(),
            "game-id": entry["game_id"],
            "date": entry["date"],
            "profile": "opus-mono-48k-v1",
            "language": self.config.language,
            "recap": base64.b64encode(json_bytes(entry["recap"])).decode(),
        }
        try:
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=entry["s3_key"],
                Body=body,
                ContentType="audio/ogg",
                StorageClass="STANDARD",
                ChecksumSHA256=base64.b64encode(digest).decode(),
                IfNoneMatch="*",
                Metadata=metadata,
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
                    f"Refusing to overwrite different S3 audio for {entry['game_id']}"
                ) from exc

    def save_manifest(self, manifest: dict) -> None:
        condition = (
            {"IfMatch": self.manifest_etag}
            if self.manifest_etag
            else {"IfNoneMatch": "*"}
        )
        response = self.client.put_object(
            Bucket=self.config.bucket,
            Key=f"{self.config.s3_prefix}/manifest.json",
            Body=json_bytes(manifest),
            ContentType="application/json",
            StorageClass="STANDARD",
            **condition,
        )
        self.manifest_etag = response["ETag"]


def reconcile(config: AudioConfig, manifest: dict, store: S3AudioStore) -> None:
    remote, inventory = store.load_manifest(), store.inventory()
    for game_id, entry in manifest["games"].items():
        previous = (remote or {}).get("games", {}).get(game_id)
        if previous and previous.get("sha256") and any(
            previous[k] != entry[k]
            for k in ("date", "phase", "away", "home", "away_score", "home_score")
        ):
            raise ValueError(f"Local/S3 game identity differs for {game_id}")
        if entry["s3_key"] in inventory:
            trusted = (
                previous if previous and previous["status"] == "uploaded" else entry
            )
            if trusted.get("sha256") and trusted.get("recap"):
                if inventory[entry["s3_key"]] != trusted["stored_bytes"]:
                    raise ValueError(f"S3 audio size mismatch for {game_id}")
                entry.update({k: trusted[k] for k in STATE_FIELDS})
            else:
                # Recover an upload whose manifest checkpoint was lost.
                body, recap = store.read_audio(entry)
                with tempfile.TemporaryDirectory(dir=config.directory) as tmp:
                    temporary = Path(tmp) / "recovered.opus"
                    temporary.write_bytes(body)
                    inspect_audio(temporary, recap.get("duration_seconds"), decode=True)
                atomic_write(config.audio_path(game_id), body)
                entry["recap"] = recap
                record_audio(entry, config)
            entry.update(status="uploaded", error="")
        elif entry["status"] == "uploaded":
            entry.update(
                status=(
                    "downloaded" if config.audio_path(game_id).exists() else "pending"
                ),
                error="Remote audio missing; scheduled for recovery",
            )
        elif (
            previous
            and entry["status"] == "pending"
            and previous["status"] not in {"uploaded", "not_final", "downloaded"}
        ):
            entry.update({k: previous[k] for k in STATE_FIELDS})


def process_game(
    entry: dict, config: AudioConfig, store: S3AudioStore | None, pacer: RequestPacer
) -> None:
    try:
        body = local_audio(entry, config)
        if body is None:
            entry["attempts"] += 1
            entry["last_attempt_at"] = utc_now()
            if not fetch_audio(entry, config, pacer):
                entry.update(
                    status="not_found",
                    error="No verified recap found; search misses are retried on the next run",
                )
                return
        body = record_audio(entry, config)
    except Exception as exc:
        entry.update(status="fetch_error", error=f"{type(exc).__name__}: {exc}")
        return
    if store:
        try:
            store.put_audio(entry, body)
            entry.update(status="uploaded", error="")
        except Exception as exc:
            entry.update(status="upload_error", error=f"{type(exc).__name__}: {exc}")


def run_archive(
    config: AudioConfig,
    *,
    game_date: date | None = None,
    max_games: int | None = None,
    payload: dict | None = None,
    store: S3AudioStore | None = None,
) -> dict:
    """Process one season or one US game date, with a checkpoint after each game."""
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive")
    if bool(config.bucket) != bool(store):
        raise ValueError("An S3 store is required exactly when a bucket is configured")
    config.directory.mkdir(parents=True, exist_ok=True)
    with (config.directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another download is already processing this season"
            ) from exc
        manifest = prepare_manifest(config, payload)
        if store:
            reconcile(config, manifest, store)
        entries = [
            g
            for g in manifest["games"].values()
            if g["game_status"] == 3
            and (game_date is None or g["date"] == game_date.isoformat())
        ]
        if store:
            entries = [g for g in entries if g["status"] in RETRYABLE]
        else:
            # Check local completed files too: detect corruption or deleted files.
            entries = [g for g in entries if g["status"] != "uploaded"]
        pacer, consecutive_errors, processed = (
            RequestPacer(config.request_interval),
            0,
            0,
        )
        failure = None
        try:
            for entry in entries:
                if (
                    not store
                    and entry["status"] == "downloaded"
                    and local_audio(entry, config) is not None
                ):
                    continue
                if max_games is not None and processed >= max_games:
                    break
                print(
                    f"[{processed + 1}] {entry['date']} {entry['away']} @ {entry['home']} ({entry['game_id']})",
                    flush=True,
                )
                process_game(entry, config, store, pacer)
                processed += 1
                checkpoint(config, manifest)
                print(
                    f"  {entry['status']}: {entry['error'] or entry['local_path']}",
                    flush=True,
                )
                consecutive_errors = (
                    consecutive_errors + 1 if entry["status"].endswith("error") else 0
                )
                if consecutive_errors >= 5:
                    raise RuntimeError(
                        "Stopped after 5 consecutive failures; inspect manifest errors before retrying"
                    )
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
                    print(f"Could not publish S3 manifest: {exc}", flush=True)
        print(json.dumps(summarize(manifest, game_date), indent=2), flush=True)
        return manifest
