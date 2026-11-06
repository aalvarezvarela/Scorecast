"""Shared YouTube audio and S3 helpers (lab/youtube_downloader/youtube_media.py)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from lab.youtube_downloader import youtube_media as media
from tests.support.conditional_s3 import ConditionalFakeS3

LEGACY_COMMAND = [
    "ffmpeg",
    "-nostdin",
    "-hide_banner",
    "-loglevel",
    "error",
    "-i",
    "in.m4a",
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
    "out.opus",
]

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def test_default_encode_command_is_the_recap_command():
    assert media.encode_command(Path("in.m4a"), Path("out.opus")) == LEGACY_COMMAND


def test_cut_and_chapters_change_only_the_needed_arguments():
    command = media.encode_command(
        Path("in.m4a"),
        Path("out.opus"),
        start=12.5,
        end=300,
        metadata_file=Path("meta.txt"),
    )
    assert command[5:9] == ["-ss", "12.500", "-to", "300.000"]
    assert command[command.index("-f") + 1] == "ffmetadata"
    assert "-map_chapters" in command and "-map_metadata:g" in command
    assert command.count("-i") == 2


def test_ffmetadata_escapes_titles_and_skips_empty_chapters():
    text = media.ffmetadata_chapters(
        [
            {"start_time": 0, "end_time": 61.25, "title": "Coach = Darko; #1"},
            {"start_time": 70, "end_time": 70, "title": "empty"},
        ]
    )
    assert text.startswith(";FFMETADATA1\n")
    assert "START=0\nEND=61250\n" in text
    assert "title=Coach \\= Darko\\; \\#1" in text
    assert "empty" not in text


def test_shift_chapters_clips_to_the_cut():
    chapters = [
        {"start_time": 0, "end_time": 300, "title": "Darko Rajakovic"},
        {"start_time": 300, "end_time": 600, "title": "Scottie Barnes"},
        {"start_time": 600, "end_time": 900, "title": "RJ Barrett"},
    ]
    assert media.shift_chapters(chapters, 120, 400) == [
        {"start_time": 0, "end_time": 180, "title": "Darko Rajakovic"},
        {"start_time": 180, "end_time": 280, "title": "Scottie Barnes"},
    ]


@pytest.mark.skipif(not HAS_FFMPEG, reason="FFmpeg is optional system tooling")
def test_real_cut_keeps_chapter_titles(tmp_path):
    source = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=30",
            "-ac",
            "2",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    chapters = media.shift_chapters(
        [
            {"start_time": 0, "end_time": 10, "title": "Coach"},
            {"start_time": 10, "end_time": 30, "title": "Player"},
        ],
        5,
        20,
    )
    out = tmp_path / "cut.opus"
    duration = media.encode_audio(source, out, 15, start=5, end=20, chapters=chapters)
    assert abs(duration - 15) < 0.1
    probed = media.probe_chapters(out)
    assert [c["title"] for c in probed] == ["Coach", "Player"]
    assert abs(probed[1]["start_time"] - 5) < 0.01


def test_put_immutable_accepts_same_content_and_refuses_different():
    s3 = ConditionalFakeS3()
    put = dict(content_type="audio/ogg", metadata={"game-id": "1"}, label="audio x")
    media.put_immutable(s3, "b", "k", b"abc", **put)
    assert s3.objects["k"]["Metadata"]["sha256"]
    assert s3.objects["k"]["IfNoneMatch"] == "*"
    media.put_immutable(s3, "b", "k", b"abc", **put)
    with pytest.raises(ValueError, match="Refusing to overwrite different S3 audio x"):
        media.put_immutable(s3, "b", "k", b"abd", **put)
    body, metadata = media.get_verified(s3, "b", "k")
    assert body == b"abc" and metadata["game-id"] == "1"


def test_mutable_objects_use_etags():
    s3 = ConditionalFakeS3()
    assert media.load_object(s3, "b", "m.json") == (None, None)
    etag = media.save_object(s3, "b", "m.json", b"{}", None)
    body, loaded = media.load_object(s3, "b", "m.json")
    assert body == b"{}" and loaded == etag
    media.save_object(s3, "b", "m.json", b'{"a": 1}', etag)
    with pytest.raises(Exception, match="PreconditionFailed"):
        media.save_object(s3, "b", "m.json", b"{}", etag)
    assert media.inventory(s3, "b", "m") == {"m.json": 8}
