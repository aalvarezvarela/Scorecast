"""Step 3: compare YouTube captions with Wishper transcripts, video by video.

    python lab/coach_pressers/caption_vs_whisper/compare_transcripts.py

Both texts are normalized (Whisper's English normalizer: case, punctuation,
numbers, contractions, "uh"/"um") and aligned word by word. Neither side is
ground truth, so the word error rate only says how much they disagree; the
useful outputs are the kinds of disagreement:

- ``sub``: aligned words that differ (mishearings on one side).
- ``yt_only`` / ``w_only``: words only one side has; ``yt_only_disfluency``
  counts the YouTube-only words that are repeats or discourse markers
  ("we we", "you know"), which Wishper leaves out on purpose.
- ``*_runs``: runs of at least ``--run-words`` consecutive words only one side
  has. A YouTube-only run is speech Wishper dropped; a Wishper-only run is
  speech YouTube dropped, or text Wishper invented.
- with human-made subtitles, the error rate of each side against them.

Writes ``summary.csv``, ``summary.json``, ``diffs.txt`` (every disagreement
with context) and ``runs.txt`` (the long one-sided runs) under the probe
folder, plus plain-text transcripts next to each video.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import jiwer
from common import DEFAULT_ROOT
from whisper_normalizer.english import EnglishTextNormalizer

normalize = EnglishTextNormalizer()
# Discourse markers: a one-sided chunk made only of these carries no content.
MARKERS = {"you", "know", "i", "mean", "like", "kind", "sort", "of", "well", "so",
           "and", "but", "yeah", "right", "just"}  # fmt: skip


def caption_text(path: Path) -> str:
    events = json.loads(path.read_text()).get("events", [])
    parts = ["".join(seg.get("utf8", "") for seg in e.get("segs", [])) for e in events]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def clean(text: str) -> list[str]:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|>>+", " ", text)  # [Music], speaker marks
    text = text.replace("\n- ", " ").lstrip("- ")  # Wishper paragraph bullets
    return normalize(text).split()


def align(reference: list[str], hypothesis: list[str]):
    """(kind, reference words, hypothesis words, reference index) per difference.

    Also returns how many hypothesis-only words are disfluencies: a repeat of
    the neighbouring words ("we we", "to try to try") or discourse markers.
    """
    out = jiwer.process_words(" ".join(reference), " ".join(hypothesis))
    diffs = [
        (
            chunk.type,
            reference[chunk.ref_start_idx : chunk.ref_end_idx],
            hypothesis[chunk.hyp_start_idx : chunk.hyp_end_idx],
            chunk.ref_start_idx,
        )
        for chunk in out.alignments[0]
        if chunk.type != "equal"
    ]
    disfluent = 0
    for chunk in out.alignments[0]:
        if chunk.type != "insert":
            continue
        start, end = chunk.hyp_start_idx, chunk.hyp_end_idx
        part, size = hypothesis[start:end], end - start
        repeat = part in (
            hypothesis[end : end + size],
            hypothesis[start - size : start],
        )
        if repeat or set(part) <= MARKERS:
            disfluent += size
    return out, diffs, disfluent


def error_rate(reference: list[str], hypothesis: list[str]) -> float:
    return round(100 * jiwer.wer(" ".join(reference), " ".join(hypothesis)), 1)


def max_repeat(words: list[str], size: int = 6) -> int:
    """Most repetitions of one ``size``-word sequence (a hallucination loop)."""
    grams = Counter(tuple(words[i : i + size]) for i in range(len(words) - size + 1))
    return max(grams.values(), default=0)


def compare(video: dict, folder: Path, run_words: int):
    wishper = json.loads((folder / "wishper.json").read_text())
    auto_path = next(
        (
            folder / f"source.{lang}.json3"
            for lang in ("en-orig", "en")
            if (folder / f"source.{lang}.json3").exists()
        ),
        None,
    )
    if auto_path is None:
        return None
    raw_auto = caption_text(auto_path)
    (folder / "youtube_auto.txt").write_text(raw_auto)
    (folder / "wishper.txt").write_text(wishper["text"])
    w_words, yt_words = clean(wishper["text"]), clean(raw_auto)
    out, diffs, disfluent = align(w_words, yt_words)

    runs = []
    for _kind, w_part, yt_part, at in diffs:
        position = round(100 * at / max(1, len(w_words)))
        if len(yt_part) - len(w_part) >= run_words:
            runs.append(("yt_only", position, w_part, yt_part))
        if len(w_part) - len(yt_part) >= run_words:
            runs.append(("w_only", position, w_part, yt_part))
    yt_run_words = sum(len(y) - len(w) for k, _, w, y in runs if k == "yt_only")
    w_run_words = sum(len(w) - len(y) for k, _, w, y in runs if k == "w_only")

    row = {
        "season": video["season"],
        "id": video["id"],
        "date": video["upload_date"],
        "channel": video["channel"],
        "minutes": round(video["duration"] / 60, 1),
        "w_words": len(w_words),
        "yt_words": len(yt_words),
        "disagree_pct": round(100 * out.wer, 1),
        "sub": out.substitutions,
        "yt_only": out.insertions,
        "yt_only_disfluency": disfluent,
        "w_only": out.deletions,
        "yt_only_runs": sum(k == "yt_only" for k, *_ in runs),
        "yt_only_run_words": yt_run_words,
        "w_only_runs": sum(k == "w_only" for k, *_ in runs),
        "w_only_run_words": w_run_words,
        "w_max_repeat": max_repeat(w_words),
        "yt_punct_per_100w": round(
            100 * len(re.findall(r"[.,?!]", raw_auto)) / max(1, len(raw_auto.split())),
            1,
        ),
        "yt_masked": raw_auto.count("[ __ ]") + raw_auto.count("[ __ ]"),
        "wishper_seconds": wishper["seconds"],
        "manual_vs_wishper": "",
        "manual_vs_youtube": "",
    }
    manual = [p for p in folder.glob("source.en-*.json3") if "orig" not in p.name]
    if manual:
        raw_manual = caption_text(manual[0])
        (folder / "youtube_manual.txt").write_text(raw_manual)
        manual_words = clean(raw_manual)
        row["manual_vs_wishper"] = error_rate(manual_words, w_words)
        row["manual_vs_youtube"] = error_rate(manual_words, yt_words)
    return row, diffs, runs, w_words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-words", type=int, default=6)
    args = parser.parse_args()

    videos = json.loads((args.root / "videos.json").read_text())
    rows, diff_lines, run_lines = [], [], []
    for video in videos:
        folder = args.root / "videos" / video["id"]
        if not (folder / "wishper.json").exists():
            continue
        result = compare(video, folder, args.run_words)
        if result is None:
            continue
        row, diffs, runs, w_words = result
        rows.append(row)
        header = f"\n===== {video['id']} {video['upload_date']} {video['title']}"
        diff_lines.append(header)
        for kind, w_part, yt_part, at in diffs:
            context = " ".join(w_words[max(0, at - 6) : at])
            diff_lines.append(
                f"[{kind[:3]}] …{context} | W: {' '.join(w_part) or '∅'}"
                f" | YT: {' '.join(yt_part) or '∅'}"
            )
        run_lines.append(header)
        for kind, position, w_part, yt_part in runs:
            run_lines.append(
                f"[{kind} at {position}%] W: {' '.join(w_part) or '∅'}"
                f" | YT: {' '.join(yt_part) or '∅'}"
            )

    (args.root / "summary.json").write_text(json.dumps(rows, indent=1))
    with (args.root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.root / "diffs.txt").write_text("\n".join(diff_lines))
    (args.root / "runs.txt").write_text("\n".join(run_lines))

    columns = ["season", "id", "minutes", "w_words", "yt_words", "disagree_pct", "sub",
               "yt_only", "yt_only_disfluency", "w_only", "yt_only_run_words", "w_only_run_words",
               "w_max_repeat", "manual_vs_wishper", "manual_vs_youtube"]  # fmt: skip
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row[c]) for c in columns))
    total = sum(r["w_words"] for r in rows)
    for key in (
        "sub",
        "yt_only",
        "yt_only_disfluency",
        "w_only",
        "yt_only_run_words",
        "w_only_run_words",
    ):
        print(f"{key}: {round(100 * sum(r[key] for r in rows) / total, 1)}% of words")


if __name__ == "__main__":
    main()
