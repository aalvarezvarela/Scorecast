"""Step 4: check whether the speech Wishper dropped is really in the audio.

    python lab/coach_pressers/caption_vs_whisper/verify_dropped_runs.py

For each YouTube-only run in ``runs.txt`` (words the captions have and the
Wishper transcript lacks) it finds the run in the timed captions, cuts that
moment of ``audio.opus`` with some margin, and transcribes the clip on its
own. A run whose words come back was audible speech that the full-file
transcription skipped; a run that does not come back is unclear audio or a
caption error. Writes ``dropped_runs_check.json`` under the probe folder.
Resumable: checked runs are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from common import DEFAULT_ROOT
from compare_transcripts import clean
from fetch_and_transcribe import transcribe

RUN = re.compile(r"^\[yt_only at \d+%\] W: (.*) \| YT: (.*)$")


def timed_words(path: Path) -> list[tuple[str, float]]:
    """(normalized word, start second) for every caption word."""
    words = []
    for event in json.loads(path.read_text()).get("events", []):
        start = event.get("tStartMs", 0)
        for seg in event.get("segs") or []:
            second = (start + seg.get("tOffsetMs", 0)) / 1000
            words.extend((word, second) for word in clean(seg.get("utf8", "")))
    return words


def locate(
    words: list[tuple[str, float]], run: list[str]
) -> tuple[float, float] | None:
    """Start and end second of the run, matched on its longest findable prefix."""
    plain = [word for word, _ in words]
    for size in range(min(len(run), 8), 3, -1):
        for offset in range(len(run) - size + 1):
            target = run[offset : offset + size]
            for i in range(len(plain) - size + 1):
                if plain[i : i + size] == target:
                    first = max(0, i - offset)
                    last = min(len(words) - 1, first + len(run))
                    return words[first][1], words[last][1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--margin", type=float, default=8.0, help="seconds each side")
    parser.add_argument(
        "--api", default=os.getenv("WISHPER_BASE_URL", "http://127.0.0.1:8000")
    )
    args = parser.parse_args()

    key = os.getenv("WISHPER_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    out = args.root / "dropped_runs_check.json"
    checked = json.loads(out.read_text()) if out.exists() else []
    done = {(row["id"], row["run"]) for row in checked}

    video_id = None
    for line in (args.root / "runs.txt").read_text().splitlines():
        if line.startswith("====="):
            video_id = line.split()[1]
            continue
        match = RUN.match(line)
        if not match or video_id is None:
            continue
        w_part = [] if match[1] == "∅" else match[1].split()
        run = match[2].split()
        if (video_id, match[2]) in done:
            continue
        folder = args.root / "videos" / video_id
        captions = next(
            folder / f"source.{lang}.json3"
            for lang in ("en-orig", "en")
            if (folder / f"source.{lang}.json3").exists()
        )
        span = locate(timed_words(captions), run)
        row = {"id": video_id, "run": match[2], "wishper_had": " ".join(w_part)}
        if span is None:
            row["status"] = "not_located"
        else:
            start = max(0.0, span[0] - args.margin)
            length = span[1] - span[0] + 2 * args.margin
            with tempfile.TemporaryDirectory() as tmp:
                clip = Path(tmp) / "clip.opus"
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}",
                     "-t", f"{length:.2f}", "-i", str(folder / "audio.opus"),
                     "-c:a", "libopus", "-b:a", "48k", str(clip)],
                    check=True,
                )  # fmt: skip
                text = transcribe(clip, args.api, headers)["text"]
            heard = set(clean(text))
            new = [word for word in run if word not in w_part]
            found = sum(word in heard for word in new)
            row.update(
                start=round(span[0], 1),
                clip_text=text,
                recovered_pct=round(100 * found / max(1, len(new))),
                status="recovered" if found >= 0.7 * len(new) else "not_recovered",
            )
        checked.append(row)
        out.write_text(json.dumps(checked, ensure_ascii=False, indent=1))
        print(row["id"], row["status"], row.get("recovered_pct"), "|", row["run"][:70])

    counts: dict[str, int] = {}
    for row in checked:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
