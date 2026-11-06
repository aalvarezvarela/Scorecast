"""One season, end to end: the steps of docs/pipeline/coach_pressers.md in order."""

from __future__ import annotations

import json
import logging

from lab.coach_pressers import catalog, coaches
from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.manifest import checkpoint, season_lock, summarize

log = logging.getLogger("coach_pressers")


def run_season(
    config: PresserConfig,
    *,
    dry_run: bool = True,
    until_step: int = 10,
    client=None,
    payload: dict | None = None,
    wikipedia=None,
    players_loader=None,
) -> dict:
    """Run steps 1..``until_step`` for one season and return the summary."""
    if bool(config.bucket) != bool(client):
        raise ValueError("An S3 client is required exactly when a bucket is set")
    with season_lock(config):
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
            missing = sum(
                e["coach_expected"] is None
                for e in manifest["entries"].values()
                if e["status"] != "not_final"
            )
            log.info(
                "step 2 %s: %d in-season changes, %d unparsed, %d pressers "
                "without an expected coach",
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
        summary = summarize(manifest)
        log.info("summary %s", json.dumps(summary))
        return summary
