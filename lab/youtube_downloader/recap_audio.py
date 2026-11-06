"""Resumable NBA recap audio archives, keyed by official NBA game ID.

Imports have no network or filesystem side effects. The audio profile is fixed
to Opus, 48 kbit/s VBR, mono. YouTube dependencies are loaded only when needed.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from botocore.exceptions import ClientError
from nba_api.stats.endpoints import ScheduleLeagueV2
from tqdm.auto import tqdm

from lab.game_reports.season_reports import (
    DEFAULT_PHASES,
    RequestPacer,
    SeasonConfig,
    atomic_write,
    json_bytes,
    season_games,
    utc_now,
)
from lab.youtube_downloader import youtube_media as media
from lab.youtube_downloader.youtube_media import (  # noqa: F401 - re-exported
    encode_audio,
    inspect_audio,
    publication,
    select_audio,
)

CHANNEL_ID = "UCLd4dSmXdrJykO_hgOzbfPw"
PROFILE = media.PROFILE
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
    diagnostics: bool = False
    use_channel_index: bool = True

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

    @property
    def index_path(self) -> Path:
        # One listing of the whole channel serves every season under this root.
        return Path(self.local_root) / "channel_index.json"

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
    media.require_tools(config.js_runtime)


def youtube_options(config: AudioConfig) -> dict:
    return media.youtube_options(config.directory, config.js_runtime)


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


def verify_video(info: dict, entry: dict, config: AudioConfig) -> bool:
    uploaded, _ = publication(info)
    day = date.fromisoformat(entry["date"])
    return bool(
        info.get("channel_id") == CHANNEL_ID
        and matches_game(info.get("title", ""), entry)
        and uploaded is not None
        and day <= uploaded <= day + timedelta(days=config.upload_window_days)
    )


def list_channel(config: AudioConfig, pacer: RequestPacer) -> list[dict]:
    """Flat listing (ID and title) of every upload on the channel."""
    videos = media.list_tab(
        f"https://www.youtube.com/channel/{CHANNEL_ID}/videos",
        youtube_options(config),
        pacer,
    )
    if not videos:
        raise RuntimeError("YouTube returned no channel listing")
    return [{"id": v["id"], "title": v["title"]} for v in videos]


def index_covers(index: dict, entry: dict, config: AudioConfig) -> bool:
    """True if the game's upload window had closed when the channel was listed."""
    listed = (
        datetime.fromisoformat(index["listed_at_utc"])
        .astimezone(ZoneInfo("America/New_York"))
        .date()
    )
    window_end = date.fromisoformat(entry["date"]) + timedelta(
        days=config.upload_window_days
    )
    return listed > window_end


def index_matches(index: dict, entry: dict) -> list[dict]:
    return [v for v in index["videos"] if matches_game(v["title"], entry)]


def channel_index(
    config: AudioConfig, pacer: RequestPacer, entries: list[dict]
) -> dict:
    """Load the cached channel listing, relisting only when it would decide more games.

    Searching game by game cannot tell an absent recap from a search miss and
    costs two searches per game; the listing answers every closed upload window
    at once (Motion Station has no recaps for most of Oct 2025-Feb 2026).
    """
    path = config.index_path
    cached = json.loads(path.read_bytes()) if path.is_file() else None
    if cached and cached.get("channel_id") != CHANNEL_ID:
        cached = None
    now = {"listed_at_utc": utc_now()}
    if cached and not any(
        index_covers(now, e, config) and not index_covers(cached, e, config)
        for e in entries
    ):
        return cached
    print("Listing every channel upload (takes a few minutes)...", flush=True)
    # The start time is conservative: uploads during the listing may be absent.
    listed_at = utc_now()
    videos = list_channel(config, pacer)
    if cached and len(videos) < 0.99 * len(cached["videos"]):
        raise RuntimeError(
            f"Channel listing returned {len(videos)} videos, fewer than the "
            f"{len(cached['videos'])} cached; YouTube may have truncated it, retry later"
        )
    index = {"channel_id": CHANNEL_ID, "listed_at_utc": listed_at, "videos": videos}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json_bytes(index))
    print(f"  {len(videos)} uploads listed", flush=True)
    return index


def verify_candidates(
    candidates: list[dict],
    entry: dict,
    config: AudioConfig,
    pacer: RequestPacer,
    seen: set,
    errors: list,
) -> dict | None:
    """Fetch each candidate's metadata and return the single verified recap."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    candidates = sorted(
        candidates,
        key=lambda e: (
            "extended" in normalized(e.get("title", "")),
            -int(e.get("view_count") or 0),
        ),
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
        elif config.diagnostics:
            print(
                f"    Rejected {video_id}: channel={info.get('channel_id') if info else None}, "
                f"publication={publication(info)[0] if info else None}, expected game date={entry['date']}",
                flush=True,
            )
    standard = [i for i in verified if "extended" not in normalized(i["title"])]
    choices = standard or verified
    if len(choices) > 1:
        raise ValueError(f"Ambiguous recap match: {[i['id'] for i in choices]}")
    return choices[0] if choices else None


def find_recap(
    entry: dict, config: AudioConfig, pacer: RequestPacer, index: dict | None = None
) -> dict | None:
    """Use the channel listing, else channel then broader search; verify every video.

    Once the listing covers the game's upload window, it is the only source.
    """
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    errors, seen = [], set()
    if index is not None:
        candidates = index_matches(index, entry)
        if config.diagnostics:
            print(f"  Channel listing: {len(candidates)} candidates", flush=True)
        info = verify_candidates(candidates, entry, config, pacer, seen, errors)
        if info or index_covers(index, entry, config):
            if not info and errors:
                raise RuntimeError(f"YouTube verification failed: {errors[-1]}")
            return info
    query = f'Game Recap {entry["away"]} {entry["away_score"]} {entry["home"]} {entry["home_score"]}'
    searches = [
        f"https://www.youtube.com/channel/{CHANNEL_ID}/search?"
        + urlencode({"query": query}),
        f"ytsearch{config.search_results}:{query} Motion Station",
    ]
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
            if config.diagnostics:
                entries = result["entries"]
                print(
                    f"  Search: {search} | {len(entries)} results, "
                    f"{len(candidates)} matching title/score/channel candidates",
                    flush=True,
                )
                for item in entries[:3]:
                    if item:
                        print(
                            f"    Result: {item.get('title')} ({item.get('id')})",
                            flush=True,
                        )
            info = verify_candidates(candidates, entry, config, pacer, seen, errors)
            if info:
                return info
        except DownloadError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(f"YouTube search/verification failed: {errors[-1]}")
    return None


def fetch_audio(
    entry: dict, config: AudioConfig, pacer: RequestPacer, index: dict | None = None
) -> bool:
    from yt_dlp import YoutubeDL

    info = find_recap(entry, config, pacer, index)
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
        media.put_immutable(
            self.client,
            self.config.bucket,
            entry["s3_key"],
            body,
            content_type="audio/ogg",
            metadata={
                "game-id": entry["game_id"],
                "date": entry["date"],
                "profile": media.PROFILE_LABEL,
                "language": self.config.language,
                "recap": base64.b64encode(json_bytes(entry["recap"])).decode(),
            },
            label=f"audio for {entry['game_id']}",
        )

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
        if (
            previous
            and previous.get("sha256")
            and any(
                previous[k] != entry[k]
                for k in ("date", "phase", "away", "home", "away_score", "home_score")
            )
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
    entry: dict,
    config: AudioConfig,
    store: S3AudioStore | None,
    pacer: RequestPacer,
    index: dict | None = None,
) -> None:
    try:
        body = local_audio(entry, config)
        if body is None:
            entry["attempts"] += 1
            entry["last_attempt_at"] = utc_now()
            if not fetch_audio(entry, config, pacer, index):
                error = (
                    "No verified recap found; search misses are retried on the next run"
                )
                if index and index_covers(index, entry, config):
                    error = (
                        "Listed titles match the score but not the channel and "
                        "publication window (another season's game)"
                    )
                entry.update(status="not_found", error=error)
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
        consecutive_missing = 0
        index, absent = None, 0
        failure = None
        progress = tqdm(
            entries,
            desc=f"Audio {config.season}",
            unit="game",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}]{postfix}",
            dynamic_ncols=True,
            disable=None,
        )
        try:
            if config.use_channel_index and entries:
                index = channel_index(config, pacer, entries)
            for entry in progress:
                progress.set_postfix_str(f"{entry['game_id']} {entry['status']}")
                if (
                    not store
                    and entry["status"] == "downloaded"
                    and local_audio(entry, config) is not None
                ):
                    continue
                if (
                    index
                    and entry["status"] in {"pending", "not_found"}
                    and not config.audio_path(entry["game_id"]).is_file()
                    and index_covers(index, entry, config)
                    and not index_matches(index, entry)
                ):
                    # Decided without network: neither an attempt nor a search miss.
                    entry.update(
                        status="not_found",
                        error=f"Not in the channel listing of {index['listed_at_utc']}",
                    )
                    absent += 1
                    continue
                if max_games is not None and processed >= max_games:
                    break
                progress.write(
                    f"[{processed + 1}] {entry['date']} {entry['away']} @ {entry['home']} ({entry['game_id']})",
                )
                process_game(entry, config, store, pacer, index)
                processed += 1
                checkpoint(config, manifest)
                progress.set_postfix_str(f"{entry['game_id']} {entry['status']}")
                progress.write(
                    f"  {entry['status']}: {entry['error'] or entry['local_path']}",
                )
                consecutive_errors = (
                    consecutive_errors + 1 if entry["status"].endswith("error") else 0
                )
                # The stop detects broken search; a listed game whose only title
                # match is another season's same score is a definite answer.
                if entry["status"] != "not_found":
                    consecutive_missing = 0
                elif not (index and index_covers(index, entry, config)):
                    consecutive_missing += 1
                if consecutive_missing >= 5:
                    raise RuntimeError(
                        "Stopped after 5 consecutive recap misses among listed or recent games. "
                        "Use --diagnostics to inspect results. Missing games remain retryable."
                    )
                if consecutive_errors >= 5:
                    raise RuntimeError(
                        "Stopped after 5 consecutive failures; inspect manifest errors before retrying"
                    )
        except BaseException as exc:
            failure = exc
            raise
        finally:
            progress.close()
            checkpoint(config, manifest)
            if store:
                try:
                    store.save_manifest(manifest)
                except Exception as exc:
                    if failure is None:
                        raise
                    print(f"Could not publish S3 manifest: {exc}", flush=True)
        if absent:
            print(f"{absent} games have no recap in the channel listing", flush=True)
        print(json.dumps(summarize(manifest, game_date), indent=2), flush=True)
        return manifest
