"""The per-season manifest: one entry per expected presser (game x team).

The manifest is the archive's record of truth. It is written atomically after
every processed entry and published to S3 at the end of a run with an
ETag-conditional write (see archive.py).
"""

from __future__ import annotations

import fcntl
import json
from collections import Counter
from contextlib import contextmanager

from lab.coach_pressers.config import SCHEMA_VERSION, PresserConfig
from lab.game_reports.season_reports import atomic_write, json_bytes, utc_now

STATUSES = (
    "not_final",  # the schedule does not mark the game final
    "pending",  # final, not matched yet
    "not_found",  # matched, no presser among the channels' videos
    "review",  # candidates the rules and the LLM could not settle
    "selected",  # a video is chosen; not archived yet
    "uploaded",  # audio (and extras) archived in S3
    "downloaded",  # archived locally only (--local-only)
    "fetch_error",  # download, integrity or encoding failed
    "upload_error",  # local files are fine, the S3 upload failed
    "unavailable",  # the chosen video is removed, private or blocked
)
# Entries in these states are matched again on every run.
MATCHABLE = {"pending", "not_found", "review"}
# Entries in these states are (re)tried by the archive step.
ARCHIVABLE = {"selected", "fetch_error", "upload_error", "downloaded"}
IDENTITY = ("game_id", "team", "opponent", "is_home", "date", "phase")
SCORES = ("team_score", "opponent_score")
REFRESHED = (*IDENTITY, *SCORES, "tip_utc", "playoff_game_no", "game_status")
STATE_DEFAULTS = {
    "coach_expected": None,
    "status": "pending",
    "selection": None,
    "archive": None,
    "attempts": 0,
    "last_attempt_at": None,
    "error": "",
}


def entry_key(game_id: str, team: str) -> str:
    return f"{game_id}:{team}"


def new_entry(game: dict) -> dict:
    entry = {k: game[k] for k in REFRESHED}
    entry.update(json.loads(json.dumps(STATE_DEFAULTS)))
    if game["game_status"] != 3:
        entry["status"] = "not_final"
    return entry


def merge_entries(
    old: dict | None, games: dict[str, dict], config: PresserConfig
) -> dict[str, dict]:
    """Fresh schedule rows with the saved state of the same (game, team)."""
    entries = {key: new_entry(game) for key, game in games.items()}
    if not old:
        return entries
    lost = set(old["entries"]) - set(entries)
    if lost:
        raise ValueError(
            f"Schedule lost {len(lost)} inventoried pressers, e.g. {sorted(lost)[:3]}"
        )
    for key, entry in entries.items():
        previous = old["entries"].get(key)
        if not previous:
            continue
        if previous.get("selection") and any(
            previous[k] != entry[k] for k in (*IDENTITY, *SCORES)
        ):
            raise ValueError(f"Saved game identity changed for {key}")
        entry.update({k: previous[k] for k in STATE_DEFAULTS})
        if entry["game_status"] == 3 and entry["status"] == "not_final":
            entry["status"] = "pending"
    return entries


def new_manifest(config: PresserConfig, entries: dict, old: dict | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "season": config.season,
        "phases": list(config.phases),
        "bucket": config.bucket,
        "s3_prefix": config.s3_season_prefix,
        "schedule_source": "NBA ScheduleLeagueV2",
        "created_at_utc": old["created_at_utc"] if old else utc_now(),
        "schedule_refreshed_at_utc": utc_now(),
        "entries": entries,
    }


def validate_manifest(
    manifest: dict, config: PresserConfig, *, allow_local_promotion: bool = True
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "season": config.season,
        "bucket": config.bucket,
        "s3_prefix": config.s3_season_prefix,
    }
    if allow_local_promotion and not manifest.get("bucket") and config.bucket:
        expected["bucket"] = ""
    differing = [k for k, v in expected.items() if manifest.get(k) != v]
    if differing:
        raise ValueError(
            f"Manifest {', '.join(differing)} differs from this run; "
            "use a separate --local-root"
        )
    if set(manifest.get("phases", [])) != set(config.phases):
        raise ValueError("Manifest phase scope differs; use a separate --local-root")


def load_manifest(config: PresserConfig) -> dict | None:
    path = config.manifest_path
    if not path.is_file():
        return None
    manifest = json.loads(path.read_bytes())
    validate_manifest(manifest, config)
    return manifest


def summarize(manifest: dict) -> dict:
    entries = list(manifest["entries"].values())
    final = [e for e in entries if e["status"] != "not_final"]
    found = [e for e in final if e.get("selection")]
    by_team = Counter(e["team"] for e in final)
    found_by_team = Counter(e["team"] for e in found)
    return {
        "season": manifest["season"],
        "expected_pressers": len(final),
        "found": len(found),
        "found_pct": round(100 * len(found) / len(final), 1) if final else 0.0,
        "by_status": dict(sorted(Counter(e["status"] for e in entries).items())),
        "found_pct_by_team": {
            team: round(100 * found_by_team[team] / n, 1)
            for team, n in sorted(by_team.items())
        },
        "stored_bytes": sum(
            (e.get("archive") or {}).get("stored_bytes", 0)
            + (e.get("archive") or {}).get("coach_bytes", 0)
            for e in entries
        ),
    }


def checkpoint(config: PresserConfig, manifest: dict) -> None:
    manifest["updated_at_utc"] = utc_now()
    manifest["summary"] = summarize(manifest)
    atomic_write(config.manifest_path, json_bytes(manifest))


@contextmanager
def season_lock(config: PresserConfig):
    """One run per season directory at a time."""
    config.directory.mkdir(parents=True, exist_ok=True)
    with (config.directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another run is already processing this season") from exc
        yield


def selected_entries(manifest: dict, config: PresserConfig, statuses: set) -> list:
    """Entries in ``statuses`` within the --team filter, newest games first."""
    entries = [
        e
        for e in manifest["entries"].values()
        if e["status"] in statuses and (not config.teams or e["team"] in config.teams)
    ]
    return sorted(entries, key=lambda e: (e["date"], e["game_id"]), reverse=True)
