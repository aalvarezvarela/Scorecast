"""Channel-agnostic YouTube audio and S3 helpers shared by the audio archives.

Used by the recap archive (``recap_audio.py``) and the coach presser archive
(``lab/coach_pressers/``). Imports have no network or filesystem side effects;
yt-dlp is imported only when a listing is requested.

The archived audio profile is fixed: Opus, 48 kbit/s VBR, mono, 48 kHz.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from botocore.exceptions import ClientError

PROFILE = {"codec": "opus", "bitrate_kbps": 48, "channels": 1, "sample_rate": 48000}
PROFILE_LABEL = "opus-mono-48k-v1"
CONFLICT_CODES = {"PreconditionFailed", "412", "ConditionalRequestConflict", "409"}
MISSING_CODES = {"NoSuchKey", "404", "NotFound"}


def youtube_options(cache_dir: Path, js_runtime: str) -> dict:
    return {
        "quiet": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "extractor_retries": 2,
        "js_runtimes": {js_runtime: {}},
        "cachedir": str(cache_dir / ".yt-dlp-cache"),
    }


def require_tools(js_runtime: str) -> None:
    for name in ("ffmpeg", "ffprobe", js_runtime):
        if not shutil.which(name):
            raise RuntimeError(f"Install {name} and add it to PATH")
    for name in ("yt_dlp", "yt_dlp_ejs"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(
                "Install the media dependency group: poetry install --with media"
            )


def publication(info: dict) -> tuple[date | None, str | None]:
    """US Eastern publication date and UTC instant, when YouTube exposes one."""
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


def list_tab(url: str, options: dict, pacer) -> list[dict]:
    """Flat listing of one channel tab: id, title and duration of every entry."""
    from yt_dlp import YoutubeDL

    pacer.wait()
    with YoutubeDL({**options, "extract_flat": True, "skip_download": True}) as ydl:
        result = ydl.extract_info(url, download=False)
    if not isinstance(result, dict) or result.get("entries") is None:
        raise RuntimeError(f"YouTube returned no listing for {url}")
    return [
        {"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration")}
        for e in result["entries"]
        if e and e.get("id")
    ]


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
    """Check the archive profile (and optionally a full decode); return seconds."""
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
        raise ValueError("Audio duration differs from the source video")
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


def _escape_metadata(value: str) -> str:
    out = []
    for char in value:
        if char in "=;#\\":
            out.append("\\" + char)
        elif char == "\n":
            out.append("\\\n")
        else:
            out.append(char)
    return "".join(out)


def ffmetadata_chapters(chapters: list[dict]) -> str:
    """An FFmpeg metadata file with one [CHAPTER] per chapter (millisecond base)."""
    lines = [";FFMETADATA1"]
    for chapter in chapters:
        start = int(round(float(chapter["start_time"]) * 1000))
        end = int(round(float(chapter["end_time"]) * 1000))
        if end <= start:
            continue
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start}",
            f"END={end}",
            f"title={_escape_metadata(str(chapter.get('title') or ''))}",
        ]
    return "\n".join(lines) + "\n"


def shift_chapters(chapters: list[dict], start: float, end: float) -> list[dict]:
    """Chapters that overlap [start, end], clipped and moved to start at 0."""
    shifted = []
    for chapter in chapters:
        lo = max(float(chapter["start_time"]), start)
        hi = min(float(chapter["end_time"]), end)
        if hi - lo >= 1:
            shifted.append(
                {
                    "start_time": round(lo - start, 3),
                    "end_time": round(hi - start, 3),
                    "title": chapter.get("title") or "",
                }
            )
    return shifted


def encode_command(
    source: Path,
    destination: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    metadata_file: Path | None = None,
) -> list[str]:
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        # Input seeking is sample-accurate when the audio is re-encoded.
        command += ["-ss", f"{start:.3f}"]
    if end is not None:
        command += ["-to", f"{end:.3f}"]
    command += ["-i", str(source)]
    if metadata_file is not None:
        command += ["-f", "ffmetadata", "-i", str(metadata_file)]
    command += ["-map", "0:a:0", "-vn"]
    if metadata_file is None:
        command += ["-map_metadata", "-1"]
    else:
        # A plain "-map_metadata -1" also blanks the chapter titles; strip only
        # global and stream metadata and take the chapters from the second input.
        command += [
            "-map_metadata:g",
            "-1",
            "-map_metadata:s:a",
            "-1",
            "-map_chapters",
            "1",
        ]
    command += [
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
        str(destination),
    ]
    return command


def encode_audio(
    source: Path,
    destination: Path,
    expected_duration: float | None = None,
    *,
    start: float | None = None,
    end: float | None = None,
    chapters: list[dict] | None = None,
) -> float:
    """Encode to the archive profile, verify it, and move it into place.

    ``start``/``end`` cut a segment (seconds of the source); ``chapters`` are
    embedded relative to the output. Without them the command is the one the
    recap archive has always used.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".encode-", dir=destination.parent) as tmp:
        temporary = Path(tmp) / "audio.opus"
        metadata_file = None
        if chapters:
            metadata_file = Path(tmp) / "chapters.txt"
            metadata_file.write_text(ffmetadata_chapters(chapters), encoding="utf-8")
        result = subprocess.run(
            encode_command(
                source,
                temporary,
                start=start,
                end=end,
                metadata_file=metadata_file,
            ),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode:
            raise RuntimeError(f"FFmpeg encoding failed: {result.stderr.strip()}")
        duration = inspect_audio(temporary, expected_duration, decode=True)
        temporary.replace(destination)
    return duration


def probe_chapters(path: Path) -> list[dict]:
    """Chapters FFprobe reads back from a file, as start/end seconds and title."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return [
        {
            "start_time": float(c["start_time"]),
            "end_time": float(c["end_time"]),
            "title": (c.get("tags") or {}).get("title", ""),
        }
        for c in json.loads(result.stdout).get("chapters", [])
    ]


# --------------------------------------------------------------------------- S3


def error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def put_immutable(
    client,
    bucket: str,
    key: str,
    body: bytes,
    *,
    content_type: str,
    metadata: dict[str, str],
    label: str,
) -> None:
    """Create-only upload with a SHA-256 checksum.

    An existing object with the same checksum and size is accepted; different
    content at the key raises instead of being overwritten.
    """
    digest = hashlib.sha256(body).digest()
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType=content_type,
            StorageClass="STANDARD",
            ChecksumSHA256=base64.b64encode(digest).decode(),
            IfNoneMatch="*",
            Metadata={**metadata, "sha256": digest.hex()},
        )
    except ClientError as exc:
        if error_code(exc) not in CONFLICT_CODES:
            raise
        existing = client.head_object(Bucket=bucket, Key=key)
        if existing.get("Metadata", {}).get("sha256") != digest.hex() or existing[
            "ContentLength"
        ] != len(body):
            raise ValueError(f"Refusing to overwrite different S3 {label}") from exc


def get_verified(client, bucket: str, key: str) -> tuple[bytes, dict]:
    """Body and metadata of an object, checked against its stored SHA-256."""
    response = client.get_object(Bucket=bucket, Key=key)
    body, metadata = response["Body"].read(), response.get("Metadata", {})
    if metadata.get("sha256") != hashlib.sha256(body).hexdigest():
        raise ValueError(f"S3 checksum mismatch for {key}")
    return body, metadata


def inventory(client, bucket: str, prefix: str) -> dict[str, int]:
    return {
        obj["Key"]: obj["Size"]
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        )
        for obj in page.get("Contents", [])
    }


def load_object(client, bucket: str, key: str) -> tuple[bytes | None, str | None]:
    """Body and ETag of a mutable object (manifest, cache), or (None, None)."""
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if error_code(exc) in MISSING_CODES:
            return None, None
        raise
    return response["Body"].read(), response["ETag"]


def save_object(
    client,
    bucket: str,
    key: str,
    body: bytes,
    etag: str | None,
    *,
    content_type: str = "application/json",
) -> str:
    """Write a mutable object only if nobody changed it since it was read."""
    condition = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
    response = client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        StorageClass="STANDARD",
        **condition,
    )
    return response["ETag"]
