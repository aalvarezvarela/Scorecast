"""Step 8 of the coach presser archive: reports and the human review loop."""

from __future__ import annotations

import csv

from lab.coach_pressers import catalog, report
from tests.support.coach_pressers import config_2021, game, schedule

GAMES = [
    game("0022100010", "2021-11-18", "PHX", "SAC", tip="2021-11-19T03:00:00Z"),
    game("0022100020", "2021-11-20", "SAC", "PHX", tip="2021-11-21T02:00:00Z"),
    game("0022100030", "2021-11-22", "PHX", "SAC", tip="2021-11-23T03:00:00Z"),
    game("0022100040", "2021-11-24", "SAC", "PHX", tip="2021-11-25T03:00:00Z"),
]


def candidate(vid, cls="coach", chapters=()):
    return {
        "video_id": vid,
        "channel_id": "UC",
        "channel_priority": 1,
        "title": f"title {vid}",
        "description": "",
        "published_at_utc": "2021-11-19T06:00:00+00:00",
        "published_date": "2021-11-19",
        "duration_s": 600,
        "chapters": list(chapters),
        "class": cls,
        "score": 0.7,
        "coach_guess": "Alvin Gentry",
        "match_method": "window_instant",
        "coach_chapter": None,
    }


def setup(tmp_path):
    config = config_2021(tmp_path)
    manifest = catalog.prepare_manifest(config, schedule("2021-22", GAMES))
    for entry in manifest["entries"].values():
        entry["coach_expected"] = "Alvin Gentry"
    return config, manifest


def test_review_round_trip(tmp_path):
    config, manifest = setup(tmp_path)
    key, other = "0022100010:SAC", "0022100020:SAC"
    manifest["entries"][key]["status"] = "review"
    manifest["entries"][other]["status"] = "review"
    candidates = {
        key: [
            candidate("a"),
            candidate(
                "b", "combined", [{"start_time": 0, "end_time": 300, "title": "Gentry"}]
            ),
        ],
        other: [candidate("c")],
    }
    report.write_reports(config, manifest, {}, candidates)
    path = config.reports_dir / "review.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["video_id"] for r in rows] == ["a", "b", "c"]
    rows[1].update(accept="yes", kind="combined", coach_chapter="Gentry")
    rows[2].update(accept="none")
    report.write_csv(path, report.REVIEW_FIELDS, rows)

    assert report.apply_review(config, manifest, candidates) == 2
    chosen = manifest["entries"][key]
    assert chosen["status"] == "selected"
    assert chosen["selection"]["decided_by"] == "human"
    assert chosen["selection"]["video_id"] == "b"
    assert chosen["selection"]["trim_end_s"] == 300
    assert manifest["entries"][other]["status"] == "not_found"
    decisions = report.read_csv(config.reports_dir / "review_decisions.csv")
    assert [(d["video_id"], d["kind"]) for d in decisions] == [
        ("b", "combined"),
        ("", "none"),
    ]
    # Applying the same file again changes nothing.
    assert report.apply_review(config, manifest, candidates) == 0


def test_coach_changes_and_proposed_overrides(tmp_path):
    config, manifest = setup(tmp_path)
    speakers = ["Alvin Gentry", "Doug Christie", "Doug Christie", "Doug Christie"]
    for game_row, speaker in zip(GAMES, speakers, strict=True):
        entry = manifest["entries"][f"{game_row['gameId']}:SAC"]
        entry["selection"] = {"coach_actual": speaker, "decided_by": "rules"}
    entry = manifest["entries"]["0022100010:PHX"]
    entry["selection"] = {"coach_actual": "Someone Else", "decided_by": "llm"}
    changes, proposals = report.coach_change_rows(manifest)
    sac = [c for c in changes if c["team"] == "SAC"]
    assert [c["date"] for c in sac] == ["2021-11-20", "2021-11-22", "2021-11-24"]
    assert proposals == [
        {
            "team": "SAC",
            "coach": "Doug Christie",
            "start_date": "2021-11-20",
            "end_date": "2021-11-24",
            "kind": "acting",
            "note": "detected from 3 pressers",
        }
    ]
    assert [c["run_length"] for c in changes if c["team"] == "PHX"] == [1]


def test_coverage_counts(tmp_path):
    config, manifest = setup(tmp_path)
    entry = manifest["entries"]["0022100010:SAC"]
    entry.update(
        status="uploaded",
        selection={"decided_by": "rules", "chapters": [{"title": "x"}]},
    )
    rows = {(r["team"], r["phase"]): r for r in report.coverage_rows(manifest)}
    sac = rows[("SAC", "all")]
    assert (sac["expected"], sac["found"], sac["rules"], sac["archived"]) == (
        4,
        1,
        1,
        1,
    )
    assert sac["with_chapters_pct"] == 100.0
    assert rows[("ALL", "all")]["expected"] == 8
