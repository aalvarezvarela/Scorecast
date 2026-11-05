"""Wishper multipart contract and resumable archives, without external services."""

import fcntl
import gzip
import hashlib
import json
import shutil
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest
import requests

from lab.game_reports import season_reports as reports
from lab.youtube_downloader import recap_audio as audio
from lab.youtube_downloader import recap_transcripts as archive
from lab.youtube_downloader import transcribe_recaps as cli
from tests.support.conditional_s3 import ConditionalFakeS3


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "inspect_audio", lambda *args, **kwargs: 109)
    audio_config = audio.AudioConfig(2026, tmp_path / "audio", request_interval=0)
    config = archive.TranscriptConfig(2026, tmp_path / "transcripts", "")
    games = []
    for index, day in enumerate((28, 29, 30, 31), start=1):
        games.append(
            {
                "gameId": f"002260000{index}",
                "gameDateEst": f"2026-10-{day}T00:00:00Z",
                "gameStatus": 3 if index < 4 else 1,
                "awayTeam": {"teamName": "Heat", "teamTricode": "MIA", "score": 129},
                "homeTeam": {"teamName": "Hawks", "teamTricode": "ATL", "score": 117},
            }
        )
    payload = {
        "leagueSchedule": {"seasonYear": "2026-27", "gameDates": [{"games": games}]}
    }
    manifest = audio.prepare_manifest(audio_config, payload)
    for entry in manifest["games"].values():
        if entry["game_status"] != 3:
            continue
        path = audio_config.audio_path(entry["game_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fictional opus " + entry["game_id"].encode())
        entry["recap"] = {
            "video_id": "X03mj26ELHE",
            "channel_id": audio.CHANNEL_ID,
            "url": "https://www.youtube.com/watch?v=X03mj26ELHE",
            "title": "Game Recap: Heat 129, Hawks 117",
            "duration_seconds": 109,
            "published_at_utc": "2026-10-29T02:30:00+00:00",
            "language": "en",
        }
        audio.record_audio(entry, audio_config)
    audio.checkpoint(audio_config, manifest)
    return SimpleNamespace(config=config, audio_config=audio_config, manifest=manifest)


@pytest.fixture
def wire(monkeypatch):
    state = SimpleNamespace(calls=[], replies=[], sleeps=[])

    def post(session, url, **kwargs):
        files = kwargs["files"]
        request = requests.Request(
            "POST", url, headers=kwargs["headers"], files=files, data=kwargs["data"]
        ).prepare()
        state.calls.append({"request": request, **kwargs})
        result = (
            state.replies.pop(0)
            if state.replies
            else (200, {"text": "- A full transcript.\n\n- Another paragraph."})
        )
        if isinstance(result, Exception):
            raise result
        status, payload = result
        response = requests.Response()
        response.status_code = status
        response._content = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )
        response._content_consumed = True
        response.headers["Retry-After"] = "0"
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    monkeypatch.setattr(archive.time, "sleep", state.sleeps.append)
    return state


def run(source, **kwargs):
    return archive.run_archive(source.config, source.audio_config, **kwargs)


def enable_s3(source):
    source.audio_config = replace(source.audio_config, bucket="test-bucket")
    source.config = replace(source.config, bucket="test-bucket")
    source.manifest["bucket"] = "test-bucket"
    audio.checkpoint(source.audio_config, source.manifest)
    s3 = ConditionalFakeS3()
    return s3, archive.S3TranscriptStore(s3, source.config)


def test_local_schema_matches_written_reports_and_preserves_publication(source, wire):
    manifest = run(source, max_games=1)
    assert len(manifest["games"]) == 4
    entry = manifest["games"]["0022600001"]
    body = source.config.report_path(entry["game_id"]).read_bytes()
    report = reports.load_report(body)
    reports.validate_report(report, entry, source.config)
    assert report["artifact_type"] == "audio_transcript"
    assert report["season"] == "2026-27" and report["phase"] == "regular_season"
    assert report["text"] == "- A full transcript.\n\n- Another paragraph."
    assert report["published_at_utc"] == "2026-10-29T02:30:00+00:00"
    assert (
        report["audio"]["sha256"]
        == source.manifest["games"][entry["game_id"]]["sha256"]
    )
    assert "paragraphs" not in report
    assert entry["sha256"] == hashlib.sha256(body).hexdigest()
    assert entry["status"] == "downloaded"


def test_actual_multipart_headers_fields_and_timeouts(source, wire):
    source.config = replace(
        source.config, api_key="test-secret", batch_size=2, chunk_minutes=3
    )
    run(source, max_games=1)
    call = wire.calls[0]
    request = call["request"]
    assert request.url == "http://127.0.0.1:8000/transcribe"
    assert request.headers["Authorization"] == "Bearer test-secret"
    assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="0022600001.opus"' in request.body
    assert b"fictional opus 0022600001" in request.body
    assert call["data"] == {"batch_size": "2", "chunk_minutes": "3"}
    assert call["timeout"] == (10, 1800)
    assert not call["allow_redirects"]
    assert "test-secret" not in repr(source.config)
    assert (
        b"test-secret" not in (source.config.directory / "manifest.json").read_bytes()
    )
    assert b"test-secret" not in gzip.decompress(
        source.config.report_path("0022600001").read_bytes()
    )


def test_date_then_season_and_pilot_skip_completed(source, wire):
    first = run(source, game_date=date(2026, 10, 28))
    assert archive.summarize(first, date(2026, 10, 28))["all_transcripts_saved"]
    assert first["games"]["0022600002"]["status"] == "pending"
    run(source, max_games=1)
    assert len(wire.calls) == 2
    manifest = run(source)
    assert len(wire.calls) == 3
    assert archive.summarize(manifest)["by_status"] == {"downloaded": 3, "not_final": 1}
    run(source)
    assert len(wire.calls) == 3


def test_missing_audio_visible_and_retryable(source, wire):
    item = source.manifest["games"]["0022600001"]
    saved = dict(item)
    item.update(status="not_found", sha256=None, recap=None)
    audio.checkpoint(source.audio_config, source.manifest)
    manifest = run(source, game_date=date(2026, 10, 28))
    assert manifest["games"][item["game_id"]]["status"] == "waiting_audio"
    assert not wire.calls
    item.update(saved)
    audio.checkpoint(source.audio_config, source.manifest)
    assert run(source, max_games=1)["games"][item["game_id"]]["status"] == "downloaded"


def test_corrupt_local_transcript_is_quarantined_and_regenerated(source, wire):
    run(source, max_games=1)
    path = source.config.report_path("0022600001")
    path.write_bytes(b"damaged")
    run(source, max_games=1)
    assert len(wire.calls) == 2
    assert path.with_name(path.name + ".corrupt").exists()


def test_failed_upload_reuses_local_transcript_without_calling_wishper(source, wire):
    s3, store = enable_s3(source)
    key = source.config.report_key("0022600001")
    s3.fail_keys.add(key)
    manifest = run(source, store=store, max_games=1)
    assert manifest["games"]["0022600001"]["status"] == "upload_error"
    s3.fail_keys.clear()
    manifest = run(source, store=store, max_games=1)
    assert manifest["games"]["0022600001"]["status"] == "uploaded"
    assert len(wire.calls) == 1
    assert s3.objects[key]["ContentType"] == "application/json"
    assert s3.objects[key]["ContentEncoding"] == "gzip"
    assert s3.objects[key]["IfNoneMatch"] == "*"


def test_missing_remote_object_is_reuploaded_without_transcription(source, wire):
    s3, store = enable_s3(source)
    run(source, store=store, max_games=1)
    key = source.config.report_key("0022600001")
    body = s3.objects.pop(key)["Body"]
    run(source, store=store, max_games=1)
    assert s3.objects[key]["Body"] == body
    assert len(wire.calls) == 1


@pytest.mark.parametrize("orphan", [False, True])
def test_remote_only_and_orphan_uploads_recover_without_api(source, wire, orphan):
    s3, store = enable_s3(source)
    run(source, store=store, game_date=date(2026, 10, 28))
    shutil.rmtree(source.config.directory)
    if orphan:
        del s3.objects[store.manifest_key]
    result = run(
        source,
        store=archive.S3TranscriptStore(s3, source.config),
        game_date=date(2026, 10, 28),
    )
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert len(wire.calls) == 1


def test_missing_local_audio_is_restored_from_verified_s3(source, wire):
    s3, store = enable_s3(source)
    audio_store = audio.S3AudioStore(s3, source.audio_config)
    entry = source.manifest["games"]["0022600001"]
    path = source.audio_config.audio_path(entry["game_id"])
    audio_store.put_audio(entry, path.read_bytes())
    path.unlink()
    result = run(source, store=store, audio_store=audio_store, max_games=1)
    assert result["games"][entry["game_id"]]["status"] == "uploaded"
    assert path.is_file() and len(wire.calls) == 1


def test_immutable_report_content_cannot_be_overwritten(source, wire):
    s3, store = enable_s3(source)
    result = run(source, store=store, max_games=1)
    entry = result["games"]["0022600001"]
    original = s3.objects[entry["s3_key"]]["Body"]
    doc = json.loads(gzip.decompress(original))
    doc["text"] = "Different text"
    different = reports.pack_report(doc, entry, source.config)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        store.put_report(entry, different)
    assert s3.objects[entry["s3_key"]]["Body"] == original


def test_local_archive_promotes_to_s3_without_transcribing_again(source, wire):
    run(source, max_games=1)
    s3, store = enable_s3(source)
    result = run(source, store=store, max_games=1)
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert len(wire.calls) == 1


def test_profile_and_season_partition_the_layout(source):
    config = replace(source.config, bucket="test-bucket")
    prefix = "train_data/game_recap_transcripts/v1/profile=wishper-v1/season=2026-27"
    assert config.s3_prefix == prefix
    assert config.report_key("0022600001") == f"{prefix}/transcripts/0022600001.json.gz"
    assert config.report_path("0022600001") == (
        config.local_root
        / "profile=wishper-v1"
        / "season=2026-27"
        / "transcripts"
        / "0022600001.json.gz"
    )


def test_new_profile_is_a_separate_archive(source, wire):
    run(source, max_games=1)
    other = replace(source.config, profile="another-model")
    manifest = archive.run_archive(other, source.audio_config, max_games=1)
    assert manifest["games"]["0022600001"]["status"] == "downloaded"
    assert other.directory != source.config.directory
    assert len(wire.calls) == 2


def test_batch_size_change_reuses_the_archive_but_chunks_do_not(source, wire):
    run(source, max_games=1)
    source.config = replace(source.config, batch_size=2)
    manifest = run(source, max_games=1)
    assert manifest["games"]["0022600001"]["status"] == "downloaded"
    assert manifest["games"]["0022600002"]["status"] == "downloaded"
    assert len(wire.calls) == 2
    doc = reports.load_report(source.config.report_path("0022600002").read_bytes())
    assert doc["transcription"]["batch_size"] == 2
    with pytest.raises(ValueError, match="configuration differs"):
        archive.run_archive(
            replace(source.config, chunk_minutes=5), source.audio_config
        )
    assert len(wire.calls) == 2


def test_redownloaded_audio_keeps_the_saved_transcript(source, wire):
    run(source, max_games=1)
    first = source.config.report_path("0022600001").read_bytes()
    transcribed_from = source.manifest["games"]["0022600001"]["sha256"]
    source.manifest["games"]["0022600001"]["sha256"] = "re-encoded-audio"
    audio.checkpoint(source.audio_config, source.manifest)
    manifest = run(source, game_date=date(2026, 10, 28))
    entry = manifest["games"]["0022600001"]
    assert entry["status"] == "downloaded"
    assert entry["audio_sha256"] == transcribed_from
    assert source.config.report_path("0022600001").read_bytes() == first
    assert len(wire.calls) == 1


def test_game_identity_change_is_rejected(source, wire):
    run(source, max_games=1)
    source.manifest["games"]["0022600001"]["home_score"] = 118
    audio.checkpoint(source.audio_config, source.manifest)
    with pytest.raises(ValueError, match="source identity changed"):
        run(source)
    assert len(wire.calls) == 1


@pytest.mark.parametrize("lock_source", [False, True])
def test_season_locks_prevent_simultaneous_read_write(source, wire, lock_source):
    directory = (
        source.audio_config.directory if lock_source else source.config.directory
    )
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another run"):
            run(source)
    assert not wire.calls


@pytest.mark.parametrize(
    "reply", [(401, {}), (500, {}), requests.Timeout("secret-url")]
)
def test_fatal_api_error_stops_and_checkpoints_without_automatic_retry(
    source, wire, reply
):
    wire.replies = [reply]
    with pytest.raises(RuntimeError, match="run stopped"):
        run(source)
    assert len(wire.calls) == 1
    saved = json.loads((source.config.directory / "manifest.json").read_bytes())
    assert saved["games"]["0022600001"]["status"] == "transcription_error"
    assert "secret-url" not in json.dumps(saved)
    assert saved["games"]["0022600002"]["status"] == "pending"
    assert run(source, max_games=1)["games"]["0022600001"]["status"] == "downloaded"


def test_503_retry_reopens_audio_and_has_a_fixed_budget(source, wire):
    wire.replies = [(503, {}), (200, {"text": "recovered"})]
    run(source, max_games=1)
    assert len(wire.calls) == 2 and wire.sleeps == [0]
    for call in wire.calls:
        assert b"fictional opus 0022600001" in call["request"].body
    wire.replies = [(503, {})] * 3
    with pytest.raises(RuntimeError, match="HTTP 503"):
        run(source, max_games=1)
    assert len(wire.calls) == 5


@pytest.mark.parametrize(
    "reply",
    [
        (200, {}),
        (200, {"text": ""}),
        (200, {"text": 7}),
        (200, b"not JSON"),
        (302, {}),
        (422, {}),
    ],
)
def test_invalid_response_does_not_save_a_report(source, wire, reply):
    wire.replies = [reply]
    result = run(source, max_games=1)
    assert result["games"]["0022600001"]["status"] == "transcription_error"
    assert not source.config.report_path("0022600001").exists()


def test_missing_publication_is_not_replaced_with_game_date(source, wire):
    source.manifest["games"]["0022600001"]["recap"]["published_at_utc"] = ""
    audio.checkpoint(source.audio_config, source.manifest)
    run(source, max_games=1)
    doc = reports.load_report(source.config.report_path("0022600001").read_bytes())
    assert doc["published_at_utc"] == ""


@pytest.mark.parametrize(
    "overrides",
    [
        dict(api_timeout=0),
        dict(api_timeout=float("nan")),
        dict(api_base_url="file:///tmp/audio"),
        dict(api_base_url="http://user:secret@localhost"),
        dict(batch_size=17),
        dict(chunk_minutes=0),
        dict(busy_retries=11),
        dict(workers=2),
        dict(profile=""),
        dict(profile="../escape"),
        dict(profile="a/b"),
    ],
)
def test_config_rejects_invalid_values(source, overrides):
    with pytest.raises(ValueError):
        replace(source.config, **overrides)


def test_cli_skip_download_status_and_date_are_local(source, wire, monkeypatch, capsys):
    monkeypatch.setattr(
        audio, "require_tools", lambda *_: pytest.fail("No download needed")
    )
    monkeypatch.setattr(
        audio, "run_archive", lambda *a, **kw: pytest.fail("No network needed")
    )
    common = [
        "--local-only",
        "--audio-root",
        str(source.audio_config.local_root),
        "--local-root",
        str(source.config.local_root),
    ]
    assert (
        cli.main(["download", "--date", "2026-10-28", "--skip-download", *common]) == 0
    )
    assert len(wire.calls) == 1
    assert cli.main(["status", "2026", "--date", "2026-10-28", *common]) == 0
    assert '"all_transcripts_saved": true' in capsys.readouterr().out


def test_cli_full_pipeline_runs_downloader_before_transcription(
    source, wire, monkeypatch
):
    calls = []
    monkeypatch.setattr(audio, "require_tools", lambda *_: None)

    def download(config, **kwargs):
        assert not wire.calls
        calls.append(kwargs)

    monkeypatch.setattr(audio, "run_archive", download)
    assert (
        cli.main(
            [
                "download",
                "2026-27",
                "--local-only",
                "--max-games",
                "1",
                "--audio-root",
                str(source.audio_config.local_root),
                "--local-root",
                str(source.config.local_root),
            ]
        )
        == 0
    )
    assert calls[0]["max_games"] == 1 and len(wire.calls) == 1


def test_cli_requires_season_or_date():
    with pytest.raises(SystemExit):
        cli.main(["download", "--local-only"])


@pytest.mark.parametrize("bad_document", [[], {"audio": []}])
def test_valid_gzip_with_invalid_document_shape_is_quarantined(
    source, wire, bad_document
):
    run(source, max_games=1)
    path = source.config.report_path("0022600001")
    path.write_bytes(gzip.compress(json.dumps(bad_document).encode(), mtime=0))
    run(source, max_games=1)
    assert len(wire.calls) == 2
    assert path.with_name(path.name + ".corrupt").exists()


def test_failed_manifest_publication_resumes_without_retranscribing(source, wire):
    s3, store = enable_s3(source)
    s3.fail_keys.add(store.manifest_key)
    with pytest.raises(Exception, match="AccessDenied"):
        run(source, store=store, max_games=1)
    assert source.config.report_path("0022600001").is_file()
    s3.fail_keys.clear()
    result = run(source, store=store, game_date=date(2026, 10, 28))
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert len(wire.calls) == 1


def test_remote_transcript_of_earlier_audio_is_recovered_not_redone(source, wire):
    s3, store = enable_s3(source)
    run(source, store=store, max_games=1)
    shutil.rmtree(source.config.directory)
    transcribed_from = source.manifest["games"]["0022600001"]["sha256"]
    source.manifest["games"]["0022600001"]["sha256"] = "re-encoded-audio"
    audio.checkpoint(source.audio_config, source.manifest)
    result = run(source, store=store, game_date=date(2026, 10, 28))
    entry = result["games"]["0022600001"]
    assert entry["status"] == "uploaded"
    assert entry["audio_sha256"] == transcribed_from
    assert len(wire.calls) == 1


def test_skip_download_restores_audio_manifest_and_files_from_s3(source, wire):
    s3, store = enable_s3(source)
    audio_store = audio.S3AudioStore(s3, source.audio_config)
    for entry in source.manifest["games"].values():
        if entry.get("sha256"):
            path = source.audio_config.audio_path(entry["game_id"])
            audio_store.put_audio(entry, path.read_bytes())
            entry["status"] = "uploaded"
    audio_store.save_manifest(source.manifest)
    shutil.rmtree(source.audio_config.directory)
    result = run(
        source, store=store, audio_store=audio_store, game_date=date(2026, 10, 28)
    )
    assert result["games"]["0022600001"]["status"] == "uploaded"
    assert (source.audio_config.directory / "manifest.json").is_file()
    assert source.audio_config.audio_path("0022600001").is_file()
    assert len(wire.calls) == 1


def test_missing_audio_manifest_everywhere_is_a_clear_error(source, wire):
    shutil.rmtree(source.audio_config.directory)
    with pytest.raises(ValueError, match="run the download stage"):
        run(source)
    assert not wire.calls


def test_server_upload_limit_is_checked_before_sending(source, wire):
    path = source.audio_config.audio_path("0022600001")
    with path.open("wb") as stream:
        stream.truncate(archive.MAX_UPLOAD_BYTES + 1)
    with requests.Session() as session:
        client = archive.WishperClient(source.config, session)
        with pytest.raises(ValueError, match="200 MiB"):
            client.transcribe(path)
    assert not wire.calls


def test_interruption_checkpoints_and_publishes_partial_progress(
    source, wire, monkeypatch
):
    s3, store = enable_s3(source)
    calls = []

    def transcribe(self, path):
        calls.append(path.name)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return "Completed first game"

    monkeypatch.setattr(archive.WishperClient, "transcribe", transcribe)
    with pytest.raises(KeyboardInterrupt):
        run(source, store=store)
    manifest = json.loads((source.config.directory / "manifest.json").read_bytes())
    remote = json.loads(s3.objects[store.manifest_key]["Body"])
    assert manifest == remote
    assert manifest["games"]["0022600001"]["status"] == "uploaded"
    assert manifest["games"]["0022600002"]["attempts"] == 1
    assert manifest["games"]["0022600003"]["attempts"] == 0
