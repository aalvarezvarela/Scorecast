import json

from nba_ou.fetch_data.nba_lineups.archive import RawArchive
from nba_ou.fetch_data.nba_lineups.client import EmptyResponse
from nba_ou.fetch_data.nba_lineups.manifest import Manifest


def test_manifest_resumes_only_successful_archived_responses(tmp_path):
    # Import the thin script's callable without contacting the database.
    from scripts.lineups.backfill_lineup_raw import backfill

    archive = RawArchive(root=tmp_path)
    manifest = Manifest(tmp_path / "manifest")

    class Client:
        calls = []

        def fetch(self, endpoint, game_id):
            self.calls.append(endpoint)
            if endpoint == "playbyplayv3" and self.calls.count(endpoint) == 1:
                raise EmptyResponse("temporary empty response")
            return json.dumps({"endpoint": endpoint}).encode()

    client = Client()
    game = [(2018, "0021800001")]
    first = backfill(game, archive=archive, manifest=manifest, client=client)
    assert first == {"ok": 1, "empty": 1, "failed": 0, "skipped": 0}
    second = backfill(game, archive=archive, manifest=Manifest(tmp_path / "manifest"), client=client)
    assert second == {"ok": 1, "empty": 0, "failed": 0, "skipped": 1}
    assert client.calls == ["gamerotation", "playbyplayv3", "playbyplayv3"]
    assert archive.get("playbyplayv3", 2018, "0021800001") == b'{"endpoint": "playbyplayv3"}'


def test_lost_object_is_fetched_even_if_manifest_says_ok(tmp_path):
    from scripts.lineups.backfill_lineup_raw import backfill

    archive = RawArchive(root=tmp_path)
    manifest = Manifest(tmp_path / "manifest")
    manifest.record(2018, "0021800001", "gamerotation", "ok", 100)

    class Client:
        calls = []

        def fetch(self, endpoint, game_id):
            self.calls.append(endpoint)
            return b"{}"

    client = Client()
    counts = backfill([(2018, "0021800001")], archive=archive, manifest=manifest,
                      client=client, limit=1)
    assert counts["ok"] == 1
    assert client.calls == ["gamerotation"]


def test_the_progress_total_counts_only_calls_still_owed(tmp_path):
    """The bar's rate and ETA must describe API calls, not archive hits."""
    from scripts.lineups.backfill_lineup_raw import pending_calls

    archive = RawArchive(root=tmp_path)
    manifest = Manifest(tmp_path / "manifest")
    archive.put("gamerotation", 2018, "0021800001", b"{}")
    manifest.record(2018, "0021800001", "gamerotation", "ok", 2)
    # Recorded ok, but its object is gone: still owed.
    manifest.record(2018, "0021800002", "gamerotation", "ok", 2)

    pending, skipped = pending_calls(
        [(2018, "0021800001"), (2018, "0021800002")],
        archive=archive,
        manifest=manifest,
    )
    assert skipped == 1
    assert pending == [
        (2018, "0021800001", "playbyplayv3"),
        (2018, "0021800002", "gamerotation"),
        (2018, "0021800002", "playbyplayv3"),
    ]


def test_the_backfill_can_be_restricted_to_the_endpoints_still_owed():
    """Once the play-by-play is imported, only gamerotation should be fetched."""
    from scripts.lineups.backfill_lineup_raw import pending_calls

    class Archive:
        def exists(self, endpoint, season, game_id):
            return False

    class NothingArchived:
        def is_ok(self, season, game_id, endpoint):
            return False

    pending, skipped = pending_calls(
        [(2018, "0021800001")],
        archive=Archive(),
        manifest=NothingArchived(),
        endpoints=("gamerotation",),
    )
    assert pending == [(2018, "0021800001", "gamerotation")]
    assert skipped == 0
