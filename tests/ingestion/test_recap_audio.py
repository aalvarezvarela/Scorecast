"""Audio archive coverage, language selection, recovery and encoding contracts."""

import hashlib
import io
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from lab.youtube_downloader import download_recaps as cli
from lab.youtube_downloader import recap_audio as audio


def client_error(code):
    return ClientError({"Error": {"Code": code}}, "test")


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail_keys = set()

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if key in self.fail_keys:
            raise client_error("AccessDenied")
        old = self.objects.get(key)
        if kwargs.get("IfNoneMatch") == "*" and old:
            raise client_error("PreconditionFailed")
        if "IfMatch" in kwargs and (not old or old["ETag"] != kwargs["IfMatch"]):
            raise client_error("PreconditionFailed")
        self.objects[key] = {
            **kwargs,
            "ETag": hashlib.sha256(kwargs["Body"]).hexdigest(),
        }
        return {"ETag": self.objects[key]["ETag"]}

    def get_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise client_error("NoSuchKey")
        item = self.objects[kwargs["Key"]]
        return {**item, "Body": io.BytesIO(item["Body"])}

    def head_object(self, **kwargs):
        item = self.objects[kwargs["Key"]]
        return {"Metadata": item["Metadata"], "ContentLength": len(item["Body"])}

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        yield {
            "Contents": [
                {"Key": k, "Size": len(v["Body"])}
                for k, v in self.objects.items()
                if k.startswith(kwargs["Prefix"])
            ]
        }


@pytest.fixture
def config(tmp_path):
    return audio.AudioConfig(2026, tmp_path / "audio", request_interval=0)


@pytest.fixture
def payload():
    def game(game_id, day, status=3):
        return {
            "gameId": game_id,
            "gameDateEst": f"2026-10-{day}T00:00:00Z",
            "gameStatus": status,
            "awayTeam": {
                "teamId": 1,
                "teamName": "Heat",
                "teamTricode": "MIA",
                "score": 129,
            },
            "homeTeam": {
                "teamId": 2,
                "teamName": "Hawks",
                "teamTricode": "ATL",
                "score": 117,
            },
        }

    return {
        "leagueSchedule": {
            "seasonYear": "2026-27",
            "gameDates": [
                {
                    "games": [
                        game("0022600001", "28"),
                        game("0022600002", "29"),
                        game("0052600001", "30", 1),
                        game("0012600001", "28"),
                    ]
                }
            ],
        }
    }


def recap():
    return {
        "video_id": "X03mj26ELHE",
        "url": "https://www.youtube.com/watch?v=X03mj26ELHE",
        "title": "Game Recap: Heat 129, Hawks 117",
        "channel_id": audio.CHANNEL_ID,
        "duration_seconds": 109,
        "language": "en-US",
    }


@pytest.fixture
def fake_media(monkeypatch):
    calls = []

    def fetch(entry, config, pacer):
        calls.append(entry["game_id"])
        path = config.audio_path(entry["game_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fictional verified opus audio " + entry["game_id"].encode())
        entry["recap"] = recap()
        return True

    monkeypatch.setattr(audio, "fetch_audio", fetch)
    monkeypatch.setattr(audio, "inspect_audio", lambda *args, **kwargs: 109)
    return calls


def test_date_then_season_share_inventory_and_skip_completed(
    config, payload, fake_media
):
    first = audio.run_archive(config, payload=payload, game_date=date(2026, 10, 28))
    assert len(first["games"]) == 3
    assert audio.summarize(first, date(2026, 10, 28))["all_audio_saved"]
    assert first["games"]["0022600002"]["status"] == "pending"
    result = audio.run_archive(config, payload=payload)
    assert fake_media == ["0022600001", "0022600002"]
    assert audio.summarize(result)["by_status"] == {"downloaded": 2, "not_final": 1}


def test_pilot_does_not_shrink_inventory(config, payload, fake_media):
    manifest = audio.run_archive(config, payload=payload, max_games=1)
    assert len(manifest["games"]) == 3
    assert len(fake_media) == 1
    audio.run_archive(config, payload=payload, max_games=1)
    assert len(fake_media) == 2


def test_not_found_is_retryable(config, payload, fake_media, monkeypatch):
    fetch = audio.fetch_audio
    monkeypatch.setattr(audio, "fetch_audio", lambda *args: False)
    manifest = audio.run_archive(config, payload=payload, max_games=1)
    assert manifest["games"]["0022600001"]["status"] == "not_found"
    monkeypatch.setattr(audio, "fetch_audio", fetch)
    manifest = audio.run_archive(config, payload=payload, max_games=1)
    assert manifest["games"]["0022600001"]["status"] == "downloaded"
    assert manifest["games"]["0022600001"]["attempts"] == 2


@pytest.mark.parametrize("damage", ["deleted", "corrupt"])
def test_local_media_is_repaired(config, payload, fake_media, damage):
    audio.run_archive(config, payload=payload, max_games=1)
    path = config.audio_path("0022600001")
    if damage == "deleted":
        path.unlink()
    else:
        path.write_bytes(b"damaged")
    audio.run_archive(config, payload=payload, max_games=1)
    assert fake_media == ["0022600001", "0022600001"]
    if damage == "corrupt":
        assert path.with_name(path.name + ".corrupt").exists()


def test_s3_resume_reuploads_missing_object_without_refetch(
    config, payload, fake_media
):
    config = replace(config, bucket="test")
    s3 = FakeS3()
    audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    key = config.audio_key("0022600001")
    obj = s3.objects.pop(key)
    assert obj["ContentType"] == "audio/ogg"
    assert "ContentEncoding" not in obj
    result = audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    assert fake_media == ["0022600001"]
    assert s3.objects[key]["Body"] == obj["Body"]
    assert result["games"]["0022600001"]["status"] == "uploaded"


def test_remote_only_resume_recovers_metadata(config, payload, fake_media):
    config = replace(config, bucket="test")
    s3 = FakeS3()
    audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    other = replace(config, local_root=config.local_root / "another-machine")
    result = audio.run_archive(
        other,
        payload=payload,
        store=audio.S3AudioStore(s3, other),
        max_games=1,
        game_date=date(2026, 10, 28),
    )
    assert result["games"]["0022600001"]["recap"] == recap()
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert fake_media == ["0022600001"]


def test_orphan_upload_recovers_from_object_metadata(config, payload, fake_media):
    config = replace(config, bucket="test")
    s3 = FakeS3()
    manifest = audio.prepare_manifest(config, payload)
    entry = manifest["games"]["0022600001"]
    audio.fetch_audio(entry, config, audio.RequestPacer(0))
    store = audio.S3AudioStore(s3, config)
    store.put_audio(entry, audio.record_audio(entry, config))
    # Simulate a lost local checkpoint and no remote manifest.
    shutil.rmtree(config.directory)
    result = audio.run_archive(
        config, payload=payload, store=store, game_date=date(2026, 10, 28)
    )
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert config.audio_path("0022600001").exists()
    assert fake_media == ["0022600001"]


def test_upload_failure_keeps_local_audio_for_retry(config, payload, fake_media):
    config = replace(config, bucket="test")
    s3 = FakeS3()
    key = config.audio_key("0022600001")
    s3.fail_keys.add(key)
    result = audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    assert result["games"]["0022600001"]["status"] == "upload_error"
    s3.fail_keys.clear()
    result = audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert fake_media == ["0022600001"]


def test_s3_media_and_manifest_are_not_silently_overwritten(
    config, payload, fake_media
):
    config = replace(config, bucket="test")
    s3 = FakeS3()
    manifest = audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    entry = manifest["games"]["0022600001"]
    store = audio.S3AudioStore(s3, config)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        store.put_audio(entry, b"different encoding")
    store.load_manifest()
    other = audio.S3AudioStore(s3, config)
    other.load_manifest()
    manifest["marker"] = "new version"
    store.save_manifest(manifest)
    with pytest.raises(ClientError):
        other.save_manifest(manifest)


def test_manifest_refuses_language_scope_and_identity_changes(
    config, payload, fake_media
):
    audio.run_archive(config, payload=payload, max_games=1)
    with pytest.raises(ValueError, match="language"):
        audio.prepare_manifest(replace(config, language="es"), payload)
    with pytest.raises(ValueError, match="phase"):
        audio.prepare_manifest(replace(config, phases=("regular_season",)), payload)
    payload["leagueSchedule"]["gameDates"][0]["games"][0]["awayTeam"]["score"] = 130
    with pytest.raises(ValueError, match="identity changed"):
        audio.prepare_manifest(config, payload)


def test_season_lock_covers_schedule_refresh(config, payload):
    import fcntl

    config.directory.mkdir(parents=True)
    with (config.directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another download"):
            audio.run_archive(config, payload=payload)
    assert not (config.directory / "manifest.json").exists()


def test_interrupt_checkpoints_attempt(config, payload, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(audio, "fetch_audio", interrupt)
    with pytest.raises(KeyboardInterrupt):
        audio.run_archive(config, payload=payload)
    manifest = json.loads((config.directory / "manifest.json").read_bytes())
    assert manifest["games"]["0022600001"]["attempts"] == 1


def test_language_selection_never_falls_back_to_video_or_wrong_language():
    def track(name, lang, bitrate, preference=0):
        return {
            "format_id": name,
            "vcodec": "none",
            "acodec": "opus",
            "language": lang,
            "abr": bitrate,
            "language_preference": preference,
            "url": "https://example.invalid/audio",
        }

    original = track("en-original", "en-US", 64, 10)
    info = {
        "formats": [
            track("es", "es-US", 128),
            original,
            {**original, "format_id": "video", "vcodec": "h264", "abr": 200},
        ]
    }
    assert audio.select_audio(info, "en")["format_id"] == "en-original"
    with pytest.raises(RuntimeError, match="No audio-only"):
        audio.select_audio(info, "fr")
    assert (
        audio.select_audio({"formats": [track("old", None, 64)]}, "en")["format_id"]
        == "old"
    )


def test_match_requires_channel_scores_and_publication_window(config, payload):
    entry = audio.prepare_manifest(config, payload)["games"]["0022600001"]
    info = {
        "channel_id": audio.CHANNEL_ID,
        "title": recap()["title"],
        "upload_date": "20261029",
    }
    assert audio.verify_video(info, entry, config)
    assert not audio.verify_video({**info, "channel_id": "other"}, entry, config)
    assert not audio.verify_video(
        {**info, "title": "Game Recap: Heat 128, Hawks 117"}, entry, config
    )
    assert not audio.verify_video({**info, "upload_date": "20251029"}, entry, config)
    assert not audio.verify_video({**info, "upload_date": None}, entry, config)


def test_search_fallback_verifies_channel_and_avoids_false_missing(
    config, payload, monkeypatch
):
    entry = audio.prepare_manifest(config, payload)["games"]["0022600001"]
    calls = []

    class YoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            calls.append(url)
            if "/channel/" in url:
                return {"entries": []}
            if url.startswith("ytsearch"):
                return {"entries": [{"id": "correct", "title": recap()["title"]}]}
            return {
                "id": "correct",
                "title": recap()["title"],
                "channel_id": audio.CHANNEL_ID,
                "upload_date": "20261029",
            }

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=YoutubeDL))
    monkeypatch.setitem(
        sys.modules, "yt_dlp.utils", SimpleNamespace(DownloadError=RuntimeError)
    )
    assert audio.find_recap(entry, config, audio.RequestPacer(0))["id"] == "correct"
    assert len(calls) == 3


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is optional system tooling",
)
def test_real_encoding_preserves_duration_and_produces_decodable_mono_opus(tmp_path):
    source, destination = tmp_path / "source.wav", tmp_path / "archive" / "recap.opus"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-ac",
            "2",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    before = source.read_bytes()
    duration = audio.encode_audio(source, destination, 2)
    assert abs(duration - 2) < 0.1
    assert source.read_bytes() == before
    assert destination.stat().st_size < source.stat().st_size
    assert list(destination.parent.glob(".encode-*")) == []
    with pytest.raises(ValueError, match="duration"):
        audio.inspect_audio(destination, 20)


@pytest.mark.parametrize(
    "args,year,day",
    [
        (["download", "2025-26"], 2025, None),
        (["download", "--date", "2026-10-28"], 2026, date(2026, 10, 28)),
        (["download", "--date", "2020-09-01"], 2019, date(2020, 9, 1)),
    ],
)
def test_cli_season_and_date_selection(args, year, day, monkeypatch, tmp_path):
    captured = {}

    def run(config, **kwargs):
        captured.update(year=config.season_year, day=kwargs["game_date"])
        return {"season": config.season, "bucket": "", "games": {}}

    monkeypatch.setattr(cli, "run_archive", run)
    monkeypatch.setattr(cli, "require_tools", lambda config: None)
    assert cli.main([*args, "--local-only", "--local-root", str(tmp_path)]) == 0
    assert captured == {"year": year, "day": day}


def test_cli_status_is_local_and_date_filtered(config, payload, fake_media, capsys):
    audio.run_archive(config, payload=payload, max_games=1)
    capsys.readouterr()
    assert (
        cli.main(
            ["status", "--date", "2026-10-28", "--local-root", str(config.local_root)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["expected_games"] == 1


def test_cli_requires_selection():
    with pytest.raises(SystemExit) as exc:
        cli.main(["download", "--local-only"])
    assert exc.value.code == 2


def test_local_archive_can_be_uploaded_later(config, payload, fake_media):
    audio.run_archive(config, payload=payload, max_games=1)
    uploaded = replace(config, bucket="test")
    s3 = FakeS3()
    result = audio.run_archive(
        uploaded,
        payload=payload,
        store=audio.S3AudioStore(s3, uploaded),
        game_date=date(2026, 10, 28),
    )
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert fake_media == ["0022600001"]
    with pytest.raises(ValueError, match="destination"):
        audio.prepare_manifest(config, payload)


def test_fetch_extracts_fresh_and_selects_current_language_track(
    config, payload, monkeypatch
):
    entry = audio.prepare_manifest(config, payload)["games"]["0022600001"]
    discovery = {
        "id": "video",
        "format_id": "old-en",
        "title": recap()["title"],
        "channel_id": audio.CHANNEL_ID,
        "upload_date": "20261029",
        "duration": 109,
    }
    monkeypatch.setattr(audio, "find_recap", lambda *args: discovery)
    calls = []

    class YoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download=False):
            calls.append(url)
            assert download
            formats = [
                {
                    "format_id": "old-en",
                    "vcodec": "none",
                    "acodec": "opus",
                    "language": "es-US",
                    "url": "https://example.invalid/es",
                },
                {
                    "format_id": "new-en",
                    "vcodec": "none",
                    "acodec": "opus",
                    "language": "en-US",
                    "url": "https://example.invalid/en",
                },
            ]
            context = {**discovery, "formats": formats}
            assert self.options["match_filter"](context) is None
            chosen = list(self.options["format"](context))
            assert chosen[0]["format_id"] == "new-en"
            Path(self.options["outtmpl"].replace("%(ext)s", "webm")).write_bytes(
                b"source audio"
            )
            return context

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=YoutubeDL))

    def encode(source, destination, duration):
        assert source.read_bytes() == b"source audio"
        return 109

    monkeypatch.setattr(audio, "encode_audio", encode)
    assert audio.fetch_audio(entry, config, audio.RequestPacer(0))
    assert entry["recap"]["language"] == "en-US"
    assert entry["recap"]["format_id"] == "new-en"
    assert calls == ["https://www.youtube.com/watch?v=video"]
    assert list(config.directory.glob(".download-*")) == []


def test_error_stop_checkpoints_and_keeps_original_error_if_manifest_fails(
    config, payload, monkeypatch, capsys
):
    config = replace(config, bucket="test")
    games = payload["leagueSchedule"]["gameDates"][0]["games"]
    for number in range(3, 10):
        games.append({**games[0], "gameId": f"00226{number:05d}"})

    def fail(*args):
        raise RuntimeError("source changed")

    monkeypatch.setattr(audio, "fetch_audio", fail)
    store = audio.S3AudioStore(FakeS3(), config)
    monkeypatch.setattr(store, "save_manifest", fail)
    with pytest.raises(RuntimeError, match="5 consecutive"):
        audio.run_archive(config, payload=payload, store=store)
    saved = json.loads((config.directory / "manifest.json").read_bytes())
    assert saved["summary"]["by_status"]["fetch_error"] == 5
    assert "Could not publish S3 manifest" in capsys.readouterr().out


def test_orphan_audio_cannot_be_adopted_under_another_language(
    config, payload, fake_media
):
    config = replace(config, bucket="test")
    s3 = FakeS3()
    manifest = audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    spanish = audio.S3AudioStore(s3, replace(config, language="es"))
    with pytest.raises(ValueError, match="identity/checksum"):
        spanish.read_audio(manifest["games"]["0022600001"])


def test_current_season_refresh_accepts_newly_final_scores(config, payload, fake_media):
    config = replace(config, bucket="test")
    s3 = FakeS3()
    scheduled = payload["leagueSchedule"]["gameDates"][0]["games"][2]
    scheduled["awayTeam"] = {**scheduled["awayTeam"], "score": 0}
    scheduled["homeTeam"] = {**scheduled["homeTeam"], "score": 0}
    audio.run_archive(
        config, payload=payload, store=audio.S3AudioStore(s3, config), max_games=1
    )
    scheduled["gameStatus"] = 3
    scheduled["awayTeam"]["score"] = 129
    scheduled["homeTeam"]["score"] = 117
    result = audio.run_archive(
        config,
        payload=payload,
        store=audio.S3AudioStore(s3, config),
        game_date=date(2026, 10, 30),
    )
    assert result["games"]["0052600001"]["status"] == "uploaded"
    assert fake_media == ["0022600001", "0052600001"]
