"""One season, end to end: the steps of docs/pipeline/coach_pressers.md in order."""

from __future__ import annotations

import json
import logging

from lab.coach_pressers import catalog, coaches, crawl, llm_judge, match, report
from lab.coach_pressers.channels import read_channels
from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.manifest import checkpoint, season_lock, summarize

log = logging.getLogger("coach_pressers")


def _log_timeline(config: PresserConfig, manifest: dict, built: dict) -> None:
    missing = sum(
        e["coach_expected"] is None
        for e in manifest["entries"].values()
        if e["status"] != "not_final"
    )
    log.info(
        "step 2 %s: %d in-season changes, %d unparsed, %d pressers without an "
        "expected coach",
        config.season,
        len(built["in_season_changes"]),
        len(built["unparsed_changes"]),
        missing,
    )
    for team, data in built["teams"].items():
        for warning in data["warnings"]:
            log.warning("step 2 %s %s: %s", config.season, team, warning)
    for warning in built["warnings"]:
        log.warning("step 2 %s: %s", config.season, warning)


def run_season(
    config: PresserConfig,
    *,
    dry_run: bool = True,
    until_step: int = 10,
    client=None,
    payload: dict | None = None,
    wikipedia=None,
    players_loader=None,
    lister=None,
    extractor=None,
    judge=None,
    downloader=None,
) -> dict:
    """Run steps 1..``until_step`` for one season and return the summary.

    The keyword hooks replace the network clients in tests.
    """
    if bool(config.bucket) != bool(client):
        raise ValueError("An S3 client is required exactly when a bucket is set")
    result: dict = {}
    with season_lock(config):
        if client is not None:
            from lab.coach_pressers import archive

            archive.restore_state(config, client)
        manifest = catalog.prepare_manifest(config, payload)
        checkpoint(config, manifest)
        log.info(
            "step 1 %s: %d expected pressers, final games %s",
            config.season,
            sum(e["status"] != "not_final" for e in manifest["entries"].values()),
            manifest["final_games_by_phase"],
        )
        for warning in manifest["warnings"]:
            log.warning("step 1 %s: %s", config.season, warning)
        if until_step >= 2:
            built = coaches.build_coaches(
                config, manifest, wikipedia=wikipedia, players_loader=players_loader
            )
            checkpoint(config, manifest)
            _log_timeline(config, manifest, built)
        if until_step >= 4:
            rows = [
                r for r in read_channels(config.channels_csv) if r["verified_by_human"]
            ]
            if not rows:
                log.warning(
                    "step 3: no channel has verified_by_human=true in %s; "
                    "checkpoint 1 is pending, stopping after step 2",
                    config.channels_csv,
                )
                result["stopped"] = "checkpoint 1: channels not verified"
                until_step = 2
        if until_step >= 4:
            previous = (
                json.loads(config.candidates_path.read_bytes())
                if config.candidates_path.is_file()
                else {}
            )
            report.apply_review(config, manifest, previous)
            checkpoint(config, manifest)
            videos = crawl.crawl_season(
                config, manifest, built, rows, lister=lister, extractor=extractor
            )
        if until_step >= 6:
            to_judge = match.match_season(config, manifest, built, videos)
            checkpoint(config, manifest)
        if until_step >= 7:
            result["llm"] = llm_judge.judge_season(
                config, manifest, built, to_judge, judge=judge
            )
            checkpoint(config, manifest)
        if until_step >= 8:
            candidates = json.loads(config.candidates_path.read_bytes())
            result["reports"] = report.write_reports(
                config, manifest, built, candidates
            )
        if until_step >= 9 and not dry_run:
            from lab.coach_pressers import archive

            result["archive"] = archive.archive_season(
                config, manifest, built, client=client, downloader=downloader
            )
            if until_step >= 8:
                report.write_reports(config, manifest, built, candidates)
        if client is not None:
            from lab.coach_pressers import archive

            archive.publish_state(config, manifest, client)
        result["summary"] = summarize(manifest)
        log.info("summary %s %s", config.season, json.dumps(result["summary"]))
        return result
