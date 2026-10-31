"""Archive invariants: expected-game coverage, immutable uploads, and recovery."""

import gzip
import hashlib
import io
import json
import zlib
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from lab.game_reports import season_reports as archive


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "test")


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.fail_keys = set()
        self.puts = []

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if key in self.fail_keys:
            raise client_error("AccessDenied")
        previous = self.objects.get(key)
        if kwargs.get("IfNoneMatch") == "*" and previous:
            raise client_error("PreconditionFailed")
        if "IfMatch" in kwargs and (
            not previous or previous["ETag"] != kwargs["IfMatch"]
        ):
            raise client_error("PreconditionFailed")
        body = kwargs["Body"]
        self.objects[key] = {
            "Body": body,
            "Metadata": kwargs.get("Metadata", {}),
            "ETag": hashlib.sha256(body).hexdigest(),
        }
        self.puts.append(key)
        return {"ETag": self.objects[key]["ETag"]}

    def get_object(self, **kwargs):
        if kwargs["Key"] not in self.objects:
            raise client_error("NoSuchKey")
        stored = self.objects[kwargs["Key"]]
        return {**stored, "Body": io.BytesIO(stored["Body"])}

    def head_object(self, **kwargs):
        stored = self.objects[kwargs["Key"]]
        return {"ContentLength": len(stored["Body"]), "Metadata": stored["Metadata"]}

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        yield {
            "Contents": [
                {"Key": key, "Size": len(obj["Body"])}
                for key, obj in self.objects.items()
                if key.startswith(kwargs["Prefix"])
            ]
        }


@pytest.fixture
def config(tmp_path):
    return archive.SeasonConfig(
        2025,
        tmp_path / "seasons",
        "test-bucket",
        workers=2,
        request_interval=0,
        checkpoint_every=1,
    )


@pytest.fixture
def payload():
    def game(game_id, status=3):
        return {
            "gameId": game_id,
            "gameDateEst": "2025-11-28T00:00:00Z",
            "gameStatus": status,
            "awayTeam": {"teamTricode": "SAS", "teamName": "Spurs"},
            "homeTeam": {"teamTricode": "UTA", "teamName": "Jazz"},
        }

    return {
        "leagueSchedule": {
            "seasonYear": "2025-26",
            "gameDates": [
                {
                    "games": [
                        game("0022500001"),
                        game("0042500001"),
                        game("0052500001", 1),
                        game("0012500001"),
                        game("0032500001"),
                    ]
                }
            ],
        }
    }


@pytest.fixture
def manifest(config, payload):
    return archive.prepare_season(config, payload)


def report_body(entry, config, text="A fictional written recap."):
    return archive.pack_report(
        {
            "game_id": entry["game_id"],
            "date": entry["date"],
            "text": text,
            "paragraphs": [text],
        },
        entry,
        config,
    )


def test_authoritative_inventory_excludes_exhibitions_and_keeps_unfinished(
    config, manifest
):
    assert len(manifest["games"]) == 3
    assert manifest["games"]["0052500001"]["status"] == "not_final"
    assert manifest["games"]["0022500001"]["game_id"].startswith("00")
    assert set(archive.summarize(manifest)["missing_game_ids"]) == set(
        manifest["games"]
    )


def test_wrong_season_and_disappearing_games_are_rejected(config, payload, manifest):
    payload["leagueSchedule"]["seasonYear"] = "2024-25"
    with pytest.raises(ValueError, match="requested season"):
        archive.prepare_season(config, payload)
    payload["leagueSchedule"]["seasonYear"] = "2025-26"
    payload["leagueSchedule"]["gameDates"][0]["games"] = payload["leagueSchedule"][
        "gameDates"
    ][0]["games"][1:]
    with pytest.raises(ValueError, match="lost previously"):
        archive.prepare_season(config, payload)


def test_json_gzip_is_deterministic_without_duplicate_text(config, manifest):
    entry = next(iter(manifest["games"].values()))
    body = report_body(entry, config)
    assert body == report_body(entry, config)
    document = json.loads(gzip.decompress(body))
    assert document["text"] and "paragraphs" not in document
    assert document["season"] == "2025-26"


def test_failed_upload_retries_local_file_without_refetch(
    config, manifest, monkeypatch
):
    s3 = FakeS3()
    failing = config.report_key("0042500001")
    s3.fail_keys.add(failing)
    fetched = []

    def fetch(entry, cfg, pacer):
        fetched.append(entry["game_id"])
        return report_body(entry, cfg)

    monkeypatch.setattr(archive, "fetch_report", fetch)
    archive.download_season(config, manifest, archive.S3ReportStore(s3, config))
    assert manifest["games"]["0042500001"]["status"] == "upload_error"
    assert manifest["games"]["0022500001"]["status"] == "uploaded"
    assert len(fetched) == 2
    s3.fail_keys.clear()
    archive.download_season(config, manifest, archive.S3ReportStore(s3, config))
    assert len(fetched) == 2
    assert manifest["games"]["0042500001"]["status"] == "uploaded"
    assert not manifest["summary"][
        "all_reports_uploaded"
    ]  # Unfinished game is still missing.


def test_orphan_upload_is_recovered_without_source_request(
    config, manifest, monkeypatch
):
    s3 = FakeS3()
    entry = manifest["games"]["0022500001"]
    store = archive.S3ReportStore(s3, config)
    store.put_report(entry, report_body(entry, config))
    monkeypatch.setattr(archive, "fetch_report", lambda *args: None)
    archive.download_season(config, manifest, store)
    assert entry["status"] == "uploaded"
    assert config.report_path(entry["game_id"]).is_file()
    assert s3.puts.count(entry["s3_key"]) == 1


def test_missing_s3_object_is_reuploaded_from_local(config, manifest, monkeypatch):
    s3 = FakeS3()
    monkeypatch.setattr(
        archive, "fetch_report", lambda entry, cfg, pacer: report_body(entry, cfg)
    )
    archive.download_season(config, manifest, archive.S3ReportStore(s3, config))
    key = config.report_key("0022500001")
    del s3.objects[key]
    monkeypatch.setattr(
        archive,
        "fetch_report",
        lambda *args: pytest.fail("Source should not be fetched again"),
    )
    archive.download_season(config, manifest, archive.S3ReportStore(s3, config))
    assert key in s3.objects
    assert manifest["games"]["0022500001"]["status"] == "uploaded"


def test_existing_different_s3_report_is_never_overwritten(config, manifest):
    s3 = FakeS3()
    store = archive.S3ReportStore(s3, config)
    entry = manifest["games"]["0022500001"]
    body = report_body(entry, config)
    store.put_report(entry, body)
    store.put_report(entry, body)  # Same bytes are safe to retry.
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        store.put_report(entry, report_body(entry, config, "Different article text."))
    assert s3.objects[entry["s3_key"]]["Body"] == body


def test_manifest_conditional_write_rejects_stale_writer(config, manifest):
    s3 = FakeS3()
    first, second = archive.S3ReportStore(s3, config), archive.S3ReportStore(s3, config)
    first.save_manifest(manifest)
    second.load_manifest()
    manifest["marker"] = "new data"
    first.save_manifest(manifest)
    with pytest.raises(ClientError):
        second.save_manifest(manifest)


def test_remote_only_resume_preserves_unavailable_status(
    config, payload, manifest, monkeypatch, tmp_path
):
    s3 = FakeS3()
    monkeypatch.setattr(archive, "fetch_report", lambda *args: None)
    archive.download_season(config, manifest, archive.S3ReportStore(s3, config))
    other = replace(config, local_root=tmp_path / "another-machine")
    fresh = archive.prepare_season(other, payload)
    monkeypatch.setattr(
        archive,
        "fetch_report",
        lambda *args: pytest.fail("Unavailable should be skipped by default"),
    )
    archive.download_season(other, fresh, archive.S3ReportStore(s3, other))
    assert fresh["games"]["0022500001"]["status"] == "not_available"


@pytest.mark.parametrize("damage", ["checksum", "truncated", "deflate"])
def test_unusable_local_report_is_set_aside_and_refetched(
    config, manifest, monkeypatch, damage
):
    entry = manifest["games"]["0022500001"]
    body = report_body(entry, config)
    archive.record_download(entry, body, config)
    path = config.report_path(entry["game_id"])
    if damage == "checksum":
        damaged = report_body(entry, config, "Tampered text.")
    elif damage == "truncated":
        damaged = body[:20]
    else:
        damaged = bytearray(body)
        damaged[10] ^= 0xFF  # Damage the deflate stream, keeping the gzip header.
        with pytest.raises(zlib.error):
            gzip.decompress(damaged)
    path.write_bytes(damaged)
    assert archive.use_local_report(entry, config) is None
    assert not path.exists()
    assert path.with_name(path.name + ".corrupt").is_file()

    entry["status"] = "fetch_error"
    monkeypatch.setattr(
        archive, "fetch_report", lambda entry, cfg, pacer: report_body(entry, cfg)
    )
    s3 = FakeS3()
    archive.download_season(config, manifest, archive.S3ReportStore(s3, config))
    assert manifest["games"]["0022500001"]["status"] == "uploaded"
    assert archive.use_local_report(manifest["games"]["0022500001"], config) == body


def test_legacy_estimate_metadata_is_not_required_to_resume(config, manifest, payload):
    manifest["estimate"] = {"sample_reports": 8}
    entry = manifest["games"]["0022500001"]
    entry.update(status="not_available", attempts=2, error="No recap")
    archive.checkpoint(config, manifest)

    refreshed = archive.prepare_season(config, payload)

    assert "estimate" not in refreshed
    assert refreshed["games"][entry["game_id"]]["status"] == "not_available"
    assert refreshed["games"][entry["game_id"]]["attempts"] == 2


def test_stored_reports_expose_utc_publication_time(config, manifest):
    entry = manifest["games"]["0022500001"]
    report = {
        "game_id": entry["game_id"],
        "date": entry["date"],
        "text": "A fictional written recap.",
        "published_at": "2025-11-28T23:30:00",
    }
    packed = archive.load_report(archive.pack_report(report, entry, config))
    assert packed["published_at_utc"] == "2025-11-29T04:30:00+00:00"
    # Reports archived before the field existed get it when read.
    legacy = gzip.compress(archive.json_bytes(report))
    assert archive.load_report(legacy)["published_at_utc"] == packed["published_at_utc"]


def test_manifest_publish_failure_does_not_hide_the_run_error(
    config, manifest, monkeypatch, capsys
):
    s3 = FakeS3()
    monkeypatch.setattr(
        archive,
        "fetch_report",
        lambda *args: (_ for _ in ()).throw(RuntimeError("layout changed")),
    )
    # Twelve consecutive fetch errors stop the run; give it enough games.
    for number in range(2, 15):
        game_id = f"00225{number:05d}"
        manifest["games"][game_id] = {
            **manifest["games"]["0022500001"],
            "game_id": game_id,
            "s3_key": config.report_key(game_id),
        }
    store = archive.S3ReportStore(s3, config)
    monkeypatch.setattr(
        store,
        "save_manifest",
        lambda manifest: (_ for _ in ()).throw(client_error("ExpiredToken")),
    )
    with pytest.raises(RuntimeError, match="12 consecutive"):
        archive.download_season(config, manifest, store)
    assert "Could not publish the S3 manifest" in capsys.readouterr().out
    assert (config.directory / "manifest.json").is_file()


@pytest.mark.parametrize("redirected", [True, False])
def test_missing_game_index_is_distinguished_from_wrong_game_data(
    config, manifest, monkeypatch, redirected
):
    entry = manifest["games"]["0022500001"]
    monkeypatch.setattr(archive, "create_session", MagicMock())
    props = (
        {"gameCardFeed": {"modules": []}}
        if redirected
        else {"game": {"gameId": "0022500999"}}
    )
    monkeypatch.setattr(archive, "fetch_page_props", lambda *args: props)
    result = archive.process_game(
        entry, config, archive.S3ReportStore(FakeS3(), config), archive.RequestPacer(0)
    )
    assert result["status"] == ("not_available" if redirected else "fetch_error")
    assert result["error"]


def test_pregame_preview_is_explicitly_recorded_as_unavailable(
    config, manifest, monkeypatch
):
    entry = manifest["games"]["0022500001"]
    props = {
        "game": {
            "gameId": entry["game_id"],
            "gameEt": entry["date"] + "T20:00:00Z",
            "gameStatus": 3,
        },
        "story": {"storyType": "preview", "content": ["A pregame preview."]},
    }
    monkeypatch.setattr(archive, "create_session", MagicMock())
    monkeypatch.setattr(archive, "fetch_page_props", lambda *args: props)
    result = archive.process_game(
        entry, config, archive.S3ReportStore(FakeS3(), config), archive.RequestPacer(0)
    )
    assert result["status"] == "not_available"
    assert "pregame preview" in result["error"]
