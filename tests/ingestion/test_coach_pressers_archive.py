"""Steps 9-10 of the coach presser archive: S3 objects, recovery and state."""

from __future__ import annotations

import base64
import gzip
import json
import shutil
import subprocess

import pytest

from lab.coach_pressers import archive, catalog
from lab.youtube_downloader import youtube_media as media
from tests.support.coach_pressers import config_2021, game, schedule
from tests.support.conditional_s3 import ConditionalFakeS3

GAMES = [
    game("0022100010", "2021-11-18", "PHX", "SAC"),
    game("0022100020", "2021-11-20", "SAC", "PHX"),
]
CHAPTERS = [
    {"start_time": 0.0, "end_time": 300.0, "title": "Alvin Gentry"},
    {"start_time": 300.0, "end_time": 600.0, "title": "De'Aaron Fox"},
]


def selected_manifest(config, *, combined=True):
    manifest = catalog.prepare_manifest(config, schedule("2021-22", GAMES))
    for number, key in enumerate(["0022100010:SAC", "0022100020:SAC"]):
        manifest["entries"][key].update(
            status="selected",
            selection={
                "video_id": f"vid{number}",
                "channel_id": "UC1",
                "title": "t",
                "published_at_utc": None,
                "duration_s": 600,
                "kind": "combined" if combined else "coach",
                "coach_actual": "Darko Rajaković",
                "decided_by": "rules",
                "confidence": None,
                "reason": "",
                "coach_chapter": "Alvin Gentry" if combined else None,
                "trim_start_s": 0.0 if combined else None,
                "trim_end_s": 300.0 if combined else None,
                "needs_trim": False,
                "chapters": CHAPTERS,
            },
        )
    return manifest


class FakeDownloader:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail or {}

    def __call__(self, video_id, directory):
        self.calls.append(video_id)
        if video_id in self.fail:
            raise self.fail[video_id]
        source = directory / "source.webm"
        source.write_bytes(b"source-" + video_id.encode())
        captions = directory / "source.en.vtt"
        captions.write_text("WEBVTT\n")
        return {
            "info": {
                "id": video_id,
                "channel_id": "UC1",
                "duration": 600,
                "title": "t",
                "formats": [{"url": "https://signed"}],
                "automatic_captions": {"en": [{"url": "https://signed"}]},
                "chapters": CHAPTERS,
            },
            "source": source,
            "captions": captions,
            "format": {"format_id": "251", "acodec": "opus", "language": "en"},
        }


@pytest.fixture
def fake_encoder(monkeypatch):
    def encode(
        source, destination, expected=None, *, start=None, end=None, chapters=None
    ):
        destination.write_bytes(f"opus {source.name} {start} {end}".encode())
        return (end - start) if start is not None else 600.0

    monkeypatch.setattr(media, "encode_audio", encode)
    monkeypatch.setattr(media, "probe_chapters", lambda path: CHAPTERS)


def run(config, manifest, client, downloader, **kwargs):
    return archive.archive_season(
        config,
        manifest,
        {},
        client=client,
        downloader=downloader,
        sleep=lambda s: None,
        **kwargs,
    )


def test_uploads_audio_cut_captions_and_info(tmp_path, fake_encoder):
    config = config_2021(tmp_path, bucket="b", teams=("SAC",))
    manifest = selected_manifest(config)
    s3 = ConditionalFakeS3()
    stats = run(config, manifest, s3, FakeDownloader())
    assert stats["archived"] == 2
    entry = manifest["entries"]["0022100020:SAC"]
    assert entry["status"] == "uploaded"
    prefix = "train_data/coach_pressers/v1/season=2021-22"
    stem = "0022100020_SAC_vid1"
    expected = {
        f"{prefix}/audio/{stem}.opus",
        f"{prefix}/audio/{stem}.coach.opus",
        f"{prefix}/captions/{stem}.en.vtt",
        f"{prefix}/info/{stem}.info.json.gz",
    }
    assert expected <= set(s3.objects)
    audio = s3.objects[f"{prefix}/audio/{stem}.opus"]
    assert audio["ContentType"] == "audio/ogg" and audio["IfNoneMatch"] == "*"
    identity = json.loads(base64.b64decode(audio["Metadata"]["archive"]))
    assert identity["coach"] == "Darko Rajaković"
    assert all(v.isascii() for v in audio["Metadata"].values())
    info = json.loads(
        gzip.decompress(s3.objects[f"{prefix}/info/{stem}.info.json.gz"]["Body"])
    )
    assert "formats" not in info and "automatic_captions" not in info
    assert info["chapters"] == CHAPTERS
    archived = entry["archive"]
    assert archived["coach"]["start_s"] == 0.0 and archived["coach"]["end_s"] == 300.0
    assert archived["chapters_embedded"] is True
    assert archived["format_id"] == "251"
    assert not (config.directory / "audio").exists()  # no local copy by default


def test_local_only_then_promotion_without_download(tmp_path, fake_encoder):
    local = config_2021(tmp_path, teams=("SAC",))
    manifest = selected_manifest(local)
    downloader = FakeDownloader()
    run(local, manifest, None, downloader)
    assert {e["status"] for k, e in manifest["entries"].items() if "SAC" in k} == {
        "downloaded"
    }
    assert (local.directory / "audio" / "0022100010_SAC_vid0.opus").is_file()

    remote = config_2021(tmp_path, bucket="b", teams=("SAC",))
    s3 = ConditionalFakeS3()
    run(remote, manifest, s3, downloader)
    assert len(downloader.calls) == 2  # nothing downloaded again
    assert manifest["entries"]["0022100010:SAC"]["status"] == "uploaded"
    assert any(k.endswith("0022100010_SAC_vid0.coach.opus") for k in s3.objects)


def test_errors_unavailable_and_overwrite_refusal(tmp_path, fake_encoder):
    config = config_2021(tmp_path, bucket="b", teams=("SAC",))
    manifest = selected_manifest(config)
    s3 = ConditionalFakeS3()
    key = config.object_key("audio", "0022100010_SAC_vid0.opus")
    s3.put_object(Bucket="b", Key=key, Body=b"different", Metadata={"sha256": "x"})
    downloader = FakeDownloader(fail={"vid1": archive.Unavailable("Private video")})
    run(config, manifest, s3, downloader)
    assert manifest["entries"]["0022100020:SAC"]["status"] == "unavailable"
    first = manifest["entries"]["0022100010:SAC"]
    assert first["status"] == "upload_error" and "Refusing" in first["error"]


def test_five_consecutive_failures_stop(tmp_path, fake_encoder):
    config = config_2021(tmp_path, bucket="b", teams=("SAC",))
    manifest = selected_manifest(config)
    for i in range(6):
        manifest["entries"][f"00221000{50 + i}:SAC"] = {
            **manifest["entries"]["0022100010:SAC"],
            "game_id": f"00221000{50 + i}",
            "date": f"2021-12-0{i + 1}",
            "selection": {
                **manifest["entries"]["0022100010:SAC"]["selection"],
                "video_id": f"bad{i}",
            },
        }
    failing = {f"bad{i}": RuntimeError("HTTP 403") for i in range(6)}
    stats = run(config, manifest, ConditionalFakeS3(), FakeDownloader(fail=failing))
    assert stats["stopped"] == "5 consecutive failures"
    assert stats["errors"] == 5


def test_reconcile_recovers_an_orphan_upload(tmp_path, fake_encoder):
    config = config_2021(tmp_path, bucket="b", teams=("SAC",))
    manifest = selected_manifest(config)
    s3 = ConditionalFakeS3()
    run(config, manifest, s3, FakeDownloader())
    lost = selected_manifest(config)  # the checkpoint after the uploads was lost
    downloader = FakeDownloader()
    run(config, lost, s3, downloader)
    assert downloader.calls == []
    assert lost["entries"]["0022100010:SAC"]["status"] == "uploaded"


def test_late_trim_from_s3(tmp_path, fake_encoder):
    config = config_2021(tmp_path, bucket="b", teams=("SAC",))
    manifest = selected_manifest(config, combined=False)
    s3 = ConditionalFakeS3()
    run(config, manifest, s3, FakeDownloader())
    entry = manifest["entries"]["0022100010:SAC"]
    assert "coach" not in entry["archive"]
    entry["selection"].update(kind="combined", trim_start_s=0.0, trim_end_s=300.0)
    stats = run(config, manifest, s3, FakeDownloader())
    assert stats["late_trims"] == 1
    assert entry["archive"]["coach"]["key"].endswith(".coach.opus")


def test_state_round_trip_merges_line_caches(tmp_path, fake_encoder):
    config = config_2021(tmp_path / "a", bucket="b", teams=("SAC",))
    manifest = selected_manifest(config)
    from lab.coach_pressers.manifest import checkpoint

    checkpoint(config, manifest)
    config.llm_cache_path.write_text('{"input_hash": "1"}\n')
    s3 = ConditionalFakeS3()
    archive.publish_state(config, manifest, s3, archive.restore_state(config, s3))

    other = config_2021(tmp_path / "b", bucket="b", teams=("SAC",))
    other.llm_cache_path.parent.mkdir(parents=True)
    other.llm_cache_path.write_text('{"input_hash": "2"}\n')
    state = archive.restore_state(other, s3)
    assert other.manifest_path.is_file()
    lines = other.llm_cache_path.read_text().splitlines()
    assert sorted(lines) == ['{"input_hash": "1"}', '{"input_hash": "2"}']
    archive.publish_state(other, manifest, s3, state)
    key = other.shared_key("llm_decisions.jsonl")
    assert len(s3.objects[key]["Body"].decode().splitlines()) == 2


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is optional system tooling",
)
def test_real_encoding_with_chapters_and_cut(tmp_path):
    config = config_2021(tmp_path, teams=("SAC",))
    manifest = selected_manifest(config)

    class Downloader(FakeDownloader):
        def __call__(self, video_id, directory):
            result = super().__call__(video_id, directory)
            source = directory / "source.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=600",
                    "-ac",
                    "2",
                    str(source),
                ],
                check=True,
                capture_output=True,
            )
            result["source"].unlink()
            return {**result, "source": source}

    run(config, manifest, None, Downloader())
    entry = manifest["entries"]["0022100010:SAC"]
    assert entry["status"] == "downloaded"
    full = config.directory / entry["archive"]["audio"]["local_path"]
    cut = config.directory / entry["archive"]["coach"]["local_path"]
    assert [c["title"] for c in media.probe_chapters(full)] == [
        "Alvin Gentry",
        "De'Aaron Fox",
    ]
    assert abs(media.inspect_audio(cut) - 300) < 0.5
    assert entry["archive"]["chapters_embedded"] is True
