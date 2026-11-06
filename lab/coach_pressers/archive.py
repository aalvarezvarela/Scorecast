"""Steps 9-10: audio, coach cut, captions and info JSON into S3; state backup.

For each selected presser, newest first: download the audio-only track in the
video's language (and the English auto-captions) in one yt-dlp session,
encode Opus 48 kbit/s mono with the video's chapters embedded, encode the
coach segment of a combined video when its chapter is known, and upload
everything with create-only, checksummed puts. The manifest is checkpointed
after every entry; caches, reports and the manifest are copied to S3 at the
end of the run (ETag-conditional), so another machine can resume.

S3 object metadata must be ASCII, so the identity (with names such as
Rajaković) travels as base64 JSON in ``archive``.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import logging
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from lab.coach_pressers.config import PresserConfig, object_stem
from lab.coach_pressers.manifest import ARCHIVABLE, checkpoint, selected_entries
from lab.game_reports.season_reports import atomic_write, json_bytes, utc_now
from lab.youtube_downloader import youtube_media as media

log = logging.getLogger("coach_pressers")

UNAVAILABLE_HINTS = (
    "private video",
    "video unavailable",
    "has been removed",
    "members-only",
    "not available",
    "account associated with this video has been terminated",
)
INFO_DROP = (
    "formats",
    "requested_formats",
    "requested_downloads",
    "requested_subtitles",
    "url",
    "http_headers",
    "automatic_captions",
    "subtitles",
    "thumbnails",
    "fragments",
    "manifest_url",
)
MAX_CONSECUTIVE_ERRORS = 5


class Unavailable(RuntimeError):
    """The video is removed, private or otherwise not served."""


# ------------------------------------------------------------------ download


def default_downloader(config: PresserConfig) -> Callable[[str, Path], dict]:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    def download(video_id: str, directory: Path) -> dict:
        selected: dict = {}

        def choose(context):
            track = media.select_audio(context, config.language)
            selected.update(track)
            yield track

        options = {
            **media.youtube_options(config.root, config.js_runtime),
            "format": choose,
            "outtmpl": str(directory / "source.%(ext)s"),
            "skip_download": not config.audio,
            "noprogress": True,
            "writeautomaticsub": config.captions,
            "subtitleslangs": [config.language],
            "subtitlesformat": "vtt",
            "retries": 10,
            "sleep_interval_requests": 1,
        }
        try:
            with YoutubeDL(options) as ydl:
                # Extract and download in one session: signed URLs from an
                # earlier extraction can fail with HTTP 403.
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=True
                )
        except DownloadError as exc:
            if any(h in str(exc).lower() for h in UNAVAILABLE_HINTS):
                raise Unavailable(str(exc)[:300]) from exc
            raise
        sources = [
            p
            for p in directory.glob("source.*")
            if p.suffix not in {".part", ".ytdl", ".vtt"}
        ]
        captions = sorted(directory.glob("source*.vtt"))
        return {
            "info": info,
            "source": sources[0] if len(sources) == 1 else None,
            "captions": captions[0] if captions else None,
            "format": selected,
        }

    return download


def trimmed_info(info: dict) -> bytes:
    kept = {k: v for k, v in info.items() if k not in INFO_DROP}
    return gzip.compress(
        json.dumps(kept, ensure_ascii=False, sort_keys=True, default=str).encode(),
        mtime=0,
    )


# ------------------------------------------------------------------- storage


@dataclass
class Objects:
    """Where one entry's files go, locally and in S3."""

    config: PresserConfig
    entry: dict

    @property
    def stem(self) -> str:
        return object_stem(
            self.entry["game_id"],
            self.entry["team"],
            self.entry["selection"]["video_id"],
        )

    def names(self) -> dict[str, tuple[str, str]]:
        return {
            "audio": ("audio", f"{self.stem}.opus"),
            "coach": ("audio", f"{self.stem}.coach.opus"),
            "captions": ("captions", f"{self.stem}.{self.config.language}.vtt"),
            "info": ("info", f"{self.stem}.info.json.gz"),
        }

    def key(self, part: str) -> str:
        kind, name = self.names()[part]
        return self.config.object_key(kind, name)

    def local(self, part: str) -> Path:
        kind, name = self.names()[part]
        return self.config.local_object(kind, name)


CONTENT_TYPES = {
    "audio": "audio/ogg",
    "coach": "audio/ogg",
    "captions": "text/vtt",
    "info": "application/gzip",
}


def identity(entry: dict) -> dict:
    selection = entry["selection"]
    return {
        "game_id": entry["game_id"],
        "team": entry["team"],
        "opponent": entry["opponent"],
        "date": entry["date"],
        "video_id": selection["video_id"],
        "coach": selection.get("coach_actual"),
        "kind": selection.get("kind"),
    }


def object_metadata(entry: dict) -> dict[str, str]:
    return {
        "game-id": entry["game_id"],
        "team": entry["team"],
        "video-id": entry["selection"]["video_id"],
        "profile": media.PROFILE_LABEL,
        "archive": base64.b64encode(json_bytes(identity(entry))).decode(),
    }


def store(config, client, objects: Objects, part: str, body: bytes) -> dict:
    """Upload (or save locally) one part; return its archive fields."""
    if client is not None:
        media.put_immutable(
            client,
            config.bucket,
            objects.key(part),
            body,
            content_type=CONTENT_TYPES[part],
            metadata=object_metadata(objects.entry),
            label=f"{part} for {objects.stem}",
        )
    if client is None or config.keep_local:
        atomic_write(objects.local(part), body)
    return {
        "key": objects.key(part) if client is not None else "",
        "local_path": (
            str(objects.local(part).relative_to(config.directory))
            if client is None or config.keep_local
            else ""
        ),
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
    }


# -------------------------------------------------------------------- entries


def promote_local(config: PresserConfig, entry: dict, client) -> bool:
    """Upload a --local-only archive to S3 without downloading it again."""
    archived = entry.get("archive") or {}
    if client is None or entry["status"] != "downloaded" or not archived.get("audio"):
        return False
    objects = Objects(config, entry)
    bodies = {}
    for part in ("audio", "coach", "captions", "info"):
        if not archived.get(part):
            continue
        path = config.directory / archived[part]["local_path"]
        if not path.is_file():
            return False
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != archived[part]["sha256"]:
            return False
        bodies[part] = body
    for part, body in bodies.items():
        archived[part] = {
            **archived[part],
            **store(config, client, objects, part, body),
        }
    entry.update(archive=archived, status="uploaded", error="")
    return True


def archive_entry(
    config: PresserConfig, entry: dict, client, downloader: Callable[[str, Path], dict]
) -> None:
    selection = entry["selection"]
    objects = Objects(config, entry)
    config.tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="presser-", dir=config.tmp_dir) as tmp:
        directory = Path(tmp)
        result = downloader(selection["video_id"], directory)
        info = result["info"] or {}
        if info.get("id") != selection["video_id"]:
            raise RuntimeError("Downloaded video id differs from the selection")
        if selection.get("channel_id") and info.get("channel_id") not in (
            None,
            selection["channel_id"],
        ):
            raise RuntimeError("Downloaded video belongs to another channel")
        archived = dict(entry.get("archive") or {})
        archived.update(
            retrieved_at_utc=utc_now(),
            source_codec=result["format"].get("acodec"),
            format_id=result["format"].get("format_id"),
            language=result["format"].get("language") or info.get("language"),
            source_duration_s=info.get("duration"),
        )
        chapters = selection.get("chapters") or []
        if config.audio:
            if result["source"] is None:
                raise RuntimeError("YouTube did not produce one complete audio file")
            full = directory / "full.opus"
            archived["duration_s"] = media.encode_audio(
                result["source"], full, info.get("duration"), chapters=chapters or None
            )
            archived["chapters_embedded"] = bool(chapters) and len(
                media.probe_chapters(full)
            ) == len([c for c in chapters if c["end_time"] > c["start_time"]])
            archived["audio"] = store(
                config, client, objects, "audio", full.read_bytes()
            )
            if selection.get("trim_start_s") is not None:
                archived["coach"] = cut_coach(
                    config, client, objects, result["source"], selection
                )
        if result["captions"] is not None:
            archived["captions"] = store(
                config, client, objects, "captions", result["captions"].read_bytes()
            )
        archived["info"] = store(config, client, objects, "info", trimmed_info(info))
    entry["archive"] = archived
    entry["error"] = ""
    if config.audio:
        entry["status"] = "uploaded" if client is not None else "downloaded"


def cut_coach(config, client, objects: Objects, source: Path, selection: dict) -> dict:
    start, end = float(selection["trim_start_s"]), float(selection["trim_end_s"])
    chapters = media.shift_chapters(selection.get("chapters") or [], start, end)
    out = source.parent / "coach.opus"
    duration = media.encode_audio(
        source, out, end - start, start=start, end=end, chapters=chapters or None
    )
    stored = store(config, client, objects, "coach", out.read_bytes())
    return {**stored, "duration_s": duration, "start_s": start, "end_s": end}


def late_trim(config: PresserConfig, entry: dict, client) -> bool:
    """Cut the coach segment of an archived combined video, from the archive."""
    selection, archived = entry["selection"], entry.get("archive") or {}
    if (
        selection.get("trim_start_s") is None
        or archived.get("coach")
        or not archived.get("audio")
    ):
        return False
    objects = Objects(config, entry)
    config.tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trim-", dir=config.tmp_dir) as tmp:
        source = Path(tmp) / "full.opus"
        if client is not None:
            body, _ = media.get_verified(client, config.bucket, objects.key("audio"))
        else:
            body = (config.directory / archived["audio"]["local_path"]).read_bytes()
        source.write_bytes(body)
        archived["coach"] = cut_coach(config, client, objects, source, selection)
    entry["archive"] = archived
    return True


def reconcile(config: PresserConfig, manifest: dict, client) -> None:
    """Align the manifest with what S3 holds (lost checkpoints, lost objects)."""
    inventory = media.inventory(
        client, config.bucket, f"{config.s3_season_prefix}/audio/"
    )
    for entry in manifest["entries"].values():
        if not entry.get("selection"):
            continue
        key = Objects(config, entry).key("audio")
        archived = entry.get("archive") or {}
        if entry["status"] == "uploaded" and key not in inventory and config.audio:
            entry.update(status="selected", error="Remote audio missing; re-archiving")
        elif entry["status"] in ARCHIVABLE and key in inventory:
            head = client.head_object(Bucket=config.bucket, Key=key)
            metadata = head.get("Metadata", {})
            if (
                metadata.get("game-id") == entry["game_id"]
                and metadata.get("video-id") == entry["selection"]["video_id"]
            ):
                archived["audio"] = {
                    **archived.get("audio", {}),
                    "key": key,
                    "sha256": metadata.get("sha256"),
                    "bytes": head["ContentLength"],
                }
                coach_key = Objects(config, entry).key("coach")
                if coach_key in inventory and not archived.get("coach"):
                    selection = entry["selection"]
                    archived["coach"] = {
                        "key": coach_key,
                        "bytes": inventory[coach_key],
                        "start_s": selection.get("trim_start_s"),
                        "end_s": selection.get("trim_end_s"),
                    }
                entry.update(archive=archived, status="uploaded", error="")


def archive_season(
    config: PresserConfig,
    manifest: dict,
    coaches: dict,
    *,
    client=None,
    downloader: Callable[[str, Path], dict] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    if downloader is None:
        media.require_tools(config.js_runtime)
    downloader = downloader or default_downloader(config)
    if client is not None:
        reconcile(config, manifest, client)
    statuses = ARCHIVABLE if client is not None else ARCHIVABLE - {"downloaded"}
    entries = selected_entries(manifest, config, statuses)
    if not config.audio:  # --captions-only: skip entries that have them
        entries = [e for e in entries if not (e.get("archive") or {}).get("captions")]
    stats = {"archived": 0, "errors": 0, "unavailable": 0, "late_trims": 0}
    consecutive = 0
    for done, entry in enumerate(entries):
        if done:
            low, high = config.download_sleep
            sleep(random.uniform(low, high))
            if config.pause_every and done % config.pause_every == 0:
                log.info(
                    "step 9: pausing %.0f s after %d downloads",
                    config.pause_seconds,
                    done,
                )
                sleep(config.pause_seconds)
        entry["attempts"] += 1
        entry["last_attempt_at"] = utc_now()
        try:
            if not promote_local(config, entry, client):
                archive_entry(config, entry, client, downloader)
            stats["archived"] += 1
            consecutive = 0
        except Unavailable as exc:
            entry.update(status="unavailable", error=str(exc))
            stats["unavailable"] += 1
            consecutive = 0
        except Exception as exc:
            upload = "Refusing to overwrite" in str(exc) or "AccessDenied" in str(exc)
            entry.update(
                status="upload_error" if upload else "fetch_error",
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
            stats["errors"] += 1
            consecutive += 1
        checkpoint(config, manifest)
        log.info(
            "step 9 %s %s %s: %s %s",
            config.season,
            entry["date"],
            entry["team"],
            entry["status"],
            entry["error"],
        )
        if consecutive >= MAX_CONSECUTIVE_ERRORS:
            stats["stopped"] = f"{MAX_CONSECUTIVE_ERRORS} consecutive failures"
            log.error("step 9: stopped after %s", stats["stopped"])
            break
    if config.audio:
        for entry in manifest["entries"].values():
            if entry["status"] in {"uploaded", "downloaded"} and entry.get("selection"):
                try:
                    if late_trim(config, entry, client):
                        stats["late_trims"] += 1
                        checkpoint(config, manifest)
                except Exception as exc:
                    log.warning("step 10 %s: %s", entry["game_id"], exc)
    log.info("step 9-10 %s: %s", config.season, json.dumps(stats))
    return stats


# ---------------------------------------------------------- state in S3


@dataclass
class RemoteState:
    etags: dict[str, str | None] = field(default_factory=dict)


def _state_files(config: PresserConfig) -> list[tuple[Path, str, str]]:
    """(local path, S3 key, merge mode) of everything worth keeping in S3."""
    files = [
        (config.manifest_path, config.season_file_key("manifest.json"), "replace"),
        (config.coaches_path, config.season_file_key("coaches.json"), "replace"),
        (config.candidates_path, config.season_file_key("candidates.json"), "replace"),
        (config.llm_cache_path, config.shared_key("llm_decisions.jsonl"), "lines"),
    ]
    for name in (
        "review_decisions.csv",
        "coverage.csv",
        "missing.csv",
        "review.csv",
        "coach_changes.csv",
        "proposed_overrides.csv",
        "coach_timeline_issues.csv",
        "trim_manual.csv",
    ):
        mode = "lines" if name == "review_decisions.csv" else "replace"
        files.append(
            (config.reports_dir / name, config.season_file_key(f"reports/{name}"), mode)
        )
    if config.channels_dir.is_dir():
        for directory in sorted(config.channels_dir.iterdir()):
            files.append(
                (
                    directory / "index.json",
                    config.shared_key(f"channels/{directory.name}/index.json"),
                    "replace",
                )
            )
            files.append(
                (
                    directory / "videos.jsonl.gz",
                    config.shared_key(f"channels/{directory.name}/videos.jsonl.gz"),
                    "gzip_lines",
                )
            )
    return files


def _lines(body: bytes, mode: str) -> list[str]:
    if mode == "gzip_lines":
        body = gzip.decompress(body)
    return [line for line in body.decode("utf-8").splitlines() if line.strip()]


def _merge(local: bytes, remote: bytes, mode: str) -> bytes:
    seen, merged = set(), []
    header = None
    for source in (remote, local):
        lines = _lines(source, mode)
        if mode == "lines" and lines and lines[0].startswith("decided_at_utc,"):
            header, lines = lines[0], lines[1:]
        for line in lines:
            if line not in seen:
                seen.add(line)
                merged.append(line)
    text = "\n".join(([header] if header else []) + merged) + "\n"
    return (
        gzip.compress(text.encode(), mtime=0) if mode == "gzip_lines" else text.encode()
    )


def restore_state(config: PresserConfig, client) -> RemoteState:
    """Bring S3 state to this machine before a run (missing or newer lines)."""
    state = RemoteState()
    remote_channels = media.inventory(
        client, config.bucket, config.shared_key("channels/")
    )
    for key in remote_channels:
        relative = key[len(config.shared_key("channels/")) :]
        path = config.channels_dir / relative
        if not path.exists():
            body, etag = media.load_object(client, config.bucket, key)
            atomic_write(path, body)
    for path, key, mode in _state_files(config):
        body, etag = media.load_object(client, config.bucket, key)
        state.etags[key] = etag
        if body is None:
            continue
        if not path.exists() or (mode == "replace" and _remote_is_newer(path, body)):
            atomic_write(path, body)
        elif mode != "replace":
            atomic_write(path, _merge(path.read_bytes(), body, mode))
    return state


def _remote_is_newer(path: Path, remote: bytes) -> bool:
    """For JSON state with ``updated_at_utc`` (the manifest), keep the newer copy."""
    if path.suffix != ".json":
        return False
    try:
        local_time = json.loads(path.read_bytes()).get("updated_at_utc") or ""
        remote_time = json.loads(remote).get("updated_at_utc") or ""
    except (ValueError, AttributeError):
        return False
    return remote_time > local_time


def publish_state(
    config: PresserConfig, manifest: dict, client, state: RemoteState | None = None
) -> None:
    """Copy manifest, caches and reports to S3 (ETag-conditional)."""
    state = state or RemoteState()
    published = 0
    for path, key, mode in _state_files(config):
        if not path.is_file():
            continue
        if key not in state.etags:
            remote, state.etags[key] = media.load_object(client, config.bucket, key)
            if remote is not None and mode != "replace":
                atomic_write(path, _merge(path.read_bytes(), remote, mode))
        body = path.read_bytes()
        state.etags[key] = media.save_object(
            client,
            config.bucket,
            key,
            body,
            state.etags.get(key),
            content_type="application/gzip" if path.suffix == ".gz" else "text/plain",
        )
        published += 1
    log.info("state %s: %d files published to S3", config.season, published)
