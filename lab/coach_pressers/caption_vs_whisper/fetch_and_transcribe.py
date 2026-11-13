"""Step 2: download audio and YouTube captions, transcribe the audio with Wishper.

    python lab/coach_pressers/caption_vs_whisper/fetch_and_transcribe.py

For every video in ``videos.json`` it stores, under ``videos/<id>/``:
``audio.opus`` (the archive profile: Opus 48 kbit/s mono), the YouTube
captions as ``source.<lang>.json3`` and the Wishper transcript as
``wishper.json``. Resumable: a step is skipped when its output exists.

Wishper must be running (``uvicorn wishper.api:app`` in the Wishper
repository); it serves one transcription at a time, so run one copy only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import requests
import yt_dlp
from common import DEFAULT_ROOT, youtube_options


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def download(video: dict, folder: Path, base: dict) -> None:
    options = {
        **base,
        "format": "bestaudio/best",
        "outtmpl": str(folder / "source.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en-orig", "en", *video["manual_subs"]],
        "subtitlesformat": "json3",
        "sleep_interval_subtitles": 2,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([video["id"]])
    source = next(
        p for p in folder.glob("source.*") if p.suffix not in {".json3", ".part"}
    )
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-vn", "-ac", "1",
         "-ar", "48000", "-c:a", "libopus", "-b:a", "48k", str(folder / "audio.opus")],
        check=True,
    )  # fmt: skip
    source.unlink()


def transcribe(audio: Path, api: str, headers: dict) -> dict:
    for _ in range(20):
        started = time.time()
        with audio.open("rb") as handle:
            response = requests.post(
                f"{api}/transcribe",
                files={"file": (audio.name, handle, "audio/ogg")},
                data={"batch_size": "4", "chunk_minutes": "15"},
                headers=headers,
                timeout=(10, 1800),
                allow_redirects=False,
            )
        if response.status_code == 503:  # another transcription is running
            time.sleep(10)
            continue
        response.raise_for_status()
        seconds = round(time.time() - started, 1)
        return {"text": response.json()["text"], "seconds": seconds}
    raise RuntimeError("Wishper stayed busy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--js-runtime", default="deno", help="name on PATH or path")
    parser.add_argument(
        "--api", default=os.getenv("WISHPER_BASE_URL", "http://127.0.0.1:8000")
    )
    args = parser.parse_args()

    key = os.getenv("WISHPER_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    base = youtube_options(args.root, args.js_runtime)
    videos = json.loads((args.root / "videos.json").read_text())
    for index, video in enumerate(videos, 1):
        folder = args.root / "videos" / video["id"]
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if not (folder / "audio.opus").exists():
                download(video, folder, base)
                log(index, video["id"], "downloaded")
                time.sleep(5)
            out = folder / "wishper.json"
            if not out.exists():
                result = transcribe(folder / "audio.opus", args.api, headers)
                out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
                log(index, video["id"], f"{video['duration']}s in {result['seconds']}s")
        except Exception as error:  # noqa: BLE001
            log(index, video["id"], "FAILED", repr(error)[:300])
    log("done")


if __name__ == "__main__":
    main()
