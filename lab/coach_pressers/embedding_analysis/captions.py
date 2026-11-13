"""YouTube automatic captions (WebVTT) of a coach presser as plain text.

YouTube's automatic VTT is a rolling display: every phrase appears in a cue
that shows it being spoken (the last line, with inline word times) and again
in a 10 ms cue that only repeats it. The text is the last line of every real
cue, in order.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

CUE_TIMES = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d{3}) --> (\d+):(\d\d):(\d\d)\.(\d{3})")
TAG = re.compile(r"<[^>]+>")
#: "[laughter]", "[music]", "[snorts]": sounds, not speech.
SOUND = re.compile(r"\[[^\]\n]{0,40}\]")
#: Speaker change in captions from 2024-25 on.
SPEAKER_CHANGE = re.compile(r"\s*>>\s*")
#: Shorter cues only repeat the previous line.
MIN_CUE_SECONDS = 0.02
PUNCTUATION = re.compile(r"[.,?!]")


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_vtt(vtt: str) -> list[Cue]:
    """The spoken phrases with their times, without the repeating cues."""
    cues = []
    for block in re.split(r"\n{2,}", vtt.replace("\r", "")):
        lines = block.split("\n")
        for index, line in enumerate(lines):
            match = CUE_TIMES.match(line)
            if not match:
                continue
            start = _seconds(*match.groups()[:4])
            end = _seconds(*match.groups()[4:])
            body = [TAG.sub("", item).strip() for item in lines[index + 1 :]]
            body = [item for item in body if item]
            if body and end - start > MIN_CUE_SECONDS:
                cues.append(Cue(start, end, html.unescape(body[-1])))
            break
    return cues


def caption_text(
    vtt: str, start_s: float | None = None, end_s: float | None = None
) -> str:
    """The captions as one text; ``start_s`` / ``end_s`` keep the phrases that
    start in that part of the video (the coach's part of a combined video).

    Sound tags are dropped and a speaker change starts a new line."""
    phrases = [
        cue.text
        for cue in parse_vtt(vtt)
        if (start_s is None or cue.start >= start_s)
        and (end_s is None or cue.start < end_s)
    ]
    text = SOUND.sub(" ", " ".join(phrases))
    text = SPEAKER_CHANGE.sub("\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def punctuation_per_100_words(text: str) -> float:
    """Captions before 2024-25 have no punctuation; later ones about 12."""
    words = len(text.split())
    return 100 * len(PUNCTUATION.findall(text)) / words if words else 0.0
