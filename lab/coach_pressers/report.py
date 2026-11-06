"""Step 8: per-season reports, and the human review loop.

``review.csv`` has one row per candidate of every entry in review. Put ``yes``
in ``accept`` on the right video (optionally correct ``kind``,
``coach_actual`` or ``coach_chapter``) or ``none`` on any row of an entry that
has no presser; the next run applies the decision (``decided_by = human``) and
appends it to ``review_decisions.csv``, which is kept in S3 because it cannot
be recreated.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.match import selection_from, team_games
from lab.coach_pressers.names import normalize
from lab.game_reports.season_reports import utc_now

log = logging.getLogger("coach_pressers")

REVIEW_FIELDS = (
    "accept",
    "game_id",
    "team",
    "date",
    "opponent",
    "coach_expected",
    "video_id",
    "url",
    "title",
    "published_at_utc",
    "duration_s",
    "class",
    "score",
    "match_method",
    "chapters",
    "llm_choice",
    "llm_confidence",
    "llm_reason",
    "kind",
    "coach_actual",
    "coach_chapter",
)
DECISION_FIELDS = (
    "decided_at_utc",
    "game_id",
    "team",
    "video_id",
    "kind",
    "coach_actual",
    "coach_chapter",
)


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------- review loop


def apply_review(config: PresserConfig, manifest: dict, candidates: dict) -> int:
    """Apply accepted rows of review.csv; return how many entries were decided."""
    rows = read_csv(config.reports_dir / "review.csv")
    decisions = read_csv(config.reports_dir / "review_decisions.csv")
    applied = 0
    by_entry = defaultdict(list)
    for row in rows:
        if row.get("accept", "").strip():
            by_entry[f"{row['game_id']}:{row['team']}"].append(row)
    for key, accepted in by_entry.items():
        entry = manifest["entries"].get(key)
        if entry is None or entry["status"] != "review":
            continue
        choice = accepted[0]
        verdict = choice["accept"].strip().lower()
        if verdict == "none":
            entry.update(status="not_found", selection=None, error="human: no presser")
            video_id = ""
        elif verdict in {"yes", "y", "x", "1"}:
            found = {c["video_id"]: c for c in candidates.get(key, [])}
            chosen = found.get(choice["video_id"])
            if chosen is None:
                log.warning(
                    "review: %s is not a candidate of %s", choice["video_id"], key
                )
                continue
            kind = (choice.get("kind") or chosen["class"]).strip()
            if kind not in {"coach", "combined"}:
                kind = "combined" if chosen["class"] == "combined" else "coach"
            title = (choice.get("coach_chapter") or "").strip()
            chapter = next((c for c in chosen["chapters"] if c["title"] == title), None)
            entry.update(
                status="selected",
                selection=selection_from(
                    chosen,
                    decided_by="human",
                    coach_actual=(choice.get("coach_actual") or "").strip()
                    or chosen.get("coach_guess")
                    or entry.get("coach_expected"),
                    kind=kind,
                    chapter=chapter,
                ),
                error="",
            )
            video_id = chosen["video_id"]
        else:
            continue
        selection = entry.get("selection") or {}
        decisions.append(
            {
                "decided_at_utc": utc_now(),
                "game_id": entry["game_id"],
                "team": entry["team"],
                "video_id": video_id,
                "kind": selection.get("kind", "none"),
                "coach_actual": selection.get("coach_actual", ""),
                "coach_chapter": selection.get("coach_chapter") or "",
            }
        )
        applied += 1
    if applied:
        write_csv(
            config.reports_dir / "review_decisions.csv", DECISION_FIELDS, decisions
        )
        log.info("review %s: applied %d human decisions", config.season, applied)
    return applied


def review_rows(manifest: dict, candidates: dict) -> list[dict]:
    rows = []
    for key, entry in sorted(manifest["entries"].items()):
        if entry["status"] != "review":
            continue
        answer = (entry.get("llm") or {}).get("answer") or {}
        for found in candidates.get(key, []):
            rows.append(
                {
                    "accept": "",
                    "game_id": entry["game_id"],
                    "team": entry["team"],
                    "date": entry["date"],
                    "opponent": entry["opponent"],
                    "coach_expected": entry.get("coach_expected") or "",
                    "video_id": found["video_id"],
                    "url": f"https://www.youtube.com/watch?v={found['video_id']}",
                    "title": found["title"],
                    "published_at_utc": found["published_at_utc"] or "",
                    "duration_s": found["duration_s"] or "",
                    "class": found["class"],
                    "score": found["score"],
                    "match_method": found["match_method"],
                    "chapters": " | ".join(c["title"] for c in found["chapters"]),
                    "llm_choice": answer.get("video_id") or answer.get("kind") or "",
                    "llm_confidence": answer.get("confidence", ""),
                    "llm_reason": answer.get("reason", "")
                    or (entry.get("llm") or {}).get("error")
                    or "",
                    "kind": "",
                    "coach_actual": "",
                    "coach_chapter": "",
                }
            )
    return rows


# ------------------------------------------------------------------- reports


def coverage_rows(manifest: dict) -> list[dict]:
    groups = defaultdict(list)
    for entry in manifest["entries"].values():
        if entry["status"] == "not_final":
            continue
        groups[(entry["team"], entry["phase"])].append(entry)
        groups[(entry["team"], "all")].append(entry)
        groups[("ALL", "all")].append(entry)
    rows = []
    for (team, phase), entries in sorted(groups.items()):
        found = [e for e in entries if e.get("selection")]
        decided = Counter(e["selection"]["decided_by"] for e in found)
        archived = sum(e["status"] in {"uploaded", "downloaded"} for e in entries)
        chapters = sum(bool(e["selection"].get("chapters")) for e in found)
        rows.append(
            {
                "team": team,
                "phase": phase,
                "expected": len(entries),
                "found": len(found),
                "found_pct": round(100 * len(found) / len(entries), 1),
                "rules": decided["rules"],
                "llm": decided["llm"],
                "human": decided["human"],
                "review": sum(e["status"] == "review" for e in entries),
                "archived": archived,
                "with_chapters_pct": (
                    round(100 * chapters / len(found), 1) if found else ""
                ),
            }
        )
    return rows


def missing_rows(manifest: dict, candidates: dict) -> list[dict]:
    return [
        {
            "game_id": e["game_id"],
            "date": e["date"],
            "team": e["team"],
            "opponent": e["opponent"],
            "phase": e["phase"],
            "coach_expected": e.get("coach_expected") or "",
            "status": e["status"],
            "candidates": len(candidates.get(key, [])),
        }
        for key, e in sorted(manifest["entries"].items(), key=lambda kv: kv[1]["date"])
        if e["status"] in {"pending", "not_found", "review", "unavailable"}
    ]


def coach_change_rows(manifest: dict) -> tuple[list[dict], list[dict]]:
    """Games whose speaker differs from the timeline, and proposed overrides."""
    changes, proposals = [], []
    teams = sorted({e["team"] for e in manifest["entries"].values()})
    for team in teams:
        run: list[dict] = []
        games = [g for g in team_games(manifest, team) if g.get("selection")]
        for game in [*games, None]:
            differs = game is not None and normalize(
                game["selection"].get("coach_actual") or ""
            ) not in {"", normalize(game.get("coach_expected") or "")}
            if differs and (
                not run
                or normalize(run[-1]["selection"]["coach_actual"])
                == normalize(game["selection"]["coach_actual"])
            ):
                run.append(game)
                continue
            if run:
                coach = run[0]["selection"]["coach_actual"]
                for item in run:
                    changes.append(
                        {
                            "team": team,
                            "date": item["date"],
                            "game_id": item["game_id"],
                            "coach_expected": item.get("coach_expected") or "",
                            "coach_actual": item["selection"]["coach_actual"],
                            "run_length": len(run),
                        }
                    )
                if len(run) >= 3:
                    proposals.append(
                        {
                            "team": team,
                            "coach": coach,
                            "start_date": run[0]["date"],
                            "end_date": run[-1]["date"],
                            "kind": "acting",
                            "note": f"detected from {len(run)} pressers",
                        }
                    )
            run = [game] if differs else []
    return changes, proposals


def trim_rows(manifest: dict) -> list[dict]:
    return [
        {
            "game_id": e["game_id"],
            "team": e["team"],
            "date": e["date"],
            "video_id": e["selection"]["video_id"],
            "url": f"https://www.youtube.com/watch?v={e['selection']['video_id']}",
            "title": e["selection"]["title"],
            "duration_s": e["selection"]["duration_s"],
        }
        for e in manifest["entries"].values()
        if (e.get("selection") or {}).get("needs_trim")
    ]


def write_reports(
    config: PresserConfig, manifest: dict, coaches: dict, candidates: dict
) -> dict:
    directory = config.reports_dir
    coverage = coverage_rows(manifest)
    write_csv(
        directory / "coverage.csv",
        list(coverage[0]) if coverage else ["team"],
        coverage,
    )
    missing = missing_rows(manifest, candidates)
    write_csv(
        directory / "missing.csv",
        (
            "game_id",
            "date",
            "team",
            "opponent",
            "phase",
            "coach_expected",
            "status",
            "candidates",
        ),
        missing,
    )
    review = review_rows(manifest, candidates)
    write_csv(directory / "review.csv", REVIEW_FIELDS, review)
    changes, proposals = coach_change_rows(manifest)
    write_csv(
        directory / "coach_changes.csv",
        ("team", "date", "game_id", "coach_expected", "coach_actual", "run_length"),
        changes,
    )
    write_csv(
        directory / "proposed_overrides.csv",
        ("team", "coach", "start_date", "end_date", "kind", "note"),
        proposals,
    )
    unparsed = [{"sentence": s} for s in coaches.get("unparsed_changes", [])]
    for team, data in coaches.get("teams", {}).items():
        unparsed += [{"sentence": f"{team}: {w}"} for w in data.get("warnings", [])]
    write_csv(directory / "coach_timeline_issues.csv", ("sentence",), unparsed)
    trims = trim_rows(manifest)
    write_csv(
        directory / "trim_manual.csv",
        ("game_id", "team", "date", "video_id", "url", "title", "duration_s"),
        trims,
    )
    total = next((r for r in coverage if r["team"] == "ALL"), {})
    summary = {
        "found_pct": total.get("found_pct"),
        "missing": len(missing),
        "review_entries": len({(r["game_id"], r["team"]) for r in review}),
        "coach_changes": len(changes),
        "proposed_overrides": len(proposals),
        "timeline_issues": len(unparsed),
        "needs_trim": len(trims),
    }
    log.info("step 8 %s: %s", config.season, json.dumps(summary))
    return summary
