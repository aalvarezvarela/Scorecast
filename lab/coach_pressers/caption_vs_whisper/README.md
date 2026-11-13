# YouTube captions vs Wishper on coach pressers

Probe to decide which transcript to trust for the coach postgame pressers:
YouTube's automatic English captions or our own Wishper transcription
(faster-whisper `large-v3`, the HTTP API of the Wishper repository).

## Run

The scripts need `yt-dlp`, `yt-dlp-ejs`, `jiwer`, `whisper-normalizer` and
`requests` (the last three steps do not touch YouTube), FFmpeg with `libopus`,
a JavaScript runtime for yt-dlp (`--js-runtime deno`, or a path to the
binary), and Wishper listening on `http://127.0.0.1:8000`
(`WISHPER_BASE_URL`, `WISHPER_API_KEY` override it). `jiwer` and
`whisper-normalizer` are not in `pyproject.toml`; install them in the
environment you run this from.

```bash
cd lab/coach_pressers/caption_vs_whisper
python discover_videos.py        # 1. videos.json: ~3 captioned pressers per season
python fetch_and_transcribe.py   # 2. audio.opus, captions, wishper.json per video
python compare_transcripts.py    # 3. summary.csv, diffs.txt, runs.txt
python verify_dropped_runs.py    # 4. dropped_runs_check.json
```

Every step is resumable. Output goes to
`data/coach_pressers/caption_vs_whisper_probe/` (gitignored; `--root` changes
it). Wishper serves one transcription at a time: run one copy of steps 2 and 4.

| File | Content |
| --- | --- |
| `common.py` | probe folder and yt-dlp options |
| `discover_videos.py` | YouTube search, one presser per coach query, captions required |
| `fetch_and_transcribe.py` | audio in the archive profile (Opus 48 kbit/s mono), captions as json3, Wishper transcript |
| `compare_transcripts.py` | normalizes both texts, aligns them word by word, classifies the differences |
| `verify_dropped_runs.py` | re-transcribes the moments Wishper skipped, to see if the speech is there |

## Result (2026-11-13)

19 pressers, 2019-20 to 2025-26, 4 to 16 minutes, 156 minutes and 28,579
Wishper words in total. 21 were selected; two downloads failed with HTTP 403.
Wishper ran with the server defaults (`batch_size=4`, `chunk_minutes=15`,
VAD on) at 6.6x real time on the GTX 1060.

Neither side is ground truth, so the figures are disagreement, as a share of
Wishper's words:

| Kind | Share | What it is |
| --- | --- | --- |
| All differences | 11.6% | 6.4% to 28.3% per video, no trend by season |
| Different word | 2.5% | mishearings on either side |
| Only in YouTube | 7.7% | 2.8% are repeats and discourse markers ("we we", "you know") |
| Only in YouTube, runs of 6+ words | 2.5% | 59 passages, 3 per video on average |
| Only in Wishper | 1.5% | 0.3% in runs of 6+ words (7 passages, 5 in one video) |

- **Wishper skips speech.** Step 4 cut each of the 59 passages out of the
  audio and transcribed it alone: 47 came back with the caption's words, so
  the speech is audible and the full-file transcription left it out. The 12
  that did not come back are mostly reporters' questions away from the
  microphone. The skipped passages include coach sentences ("but I thought we
  were a ton more physical"), not only questions. No hallucination loops
  (no 6-word sequence repeats more than 3 times).
- **Wishper cleans up.** It drops "uh", repeated words and false starts, and
  writes punctuation, capitals and paragraphs. YouTube is verbatim, and its
  captions have no punctuation or capitals until 2024-25 (0 marks per 100
  words in 15 of 19 videos, 11 to 15 in the four newest team-channel videos).
- **Different words**: most are short function words that cannot be judged
  from text ("and"/"in", "them"/"him"). Where the context decides, Wishper is
  more often right on vocabulary and names ("game in this playoffs" vs "games
  gaming display house", "Toronto Star" vs "staff", "Miami" vs "my"), but
  YouTube wins some ("Suns" vs "sons", "scheme went" vs "is being white").
  This is a reading of about 70 sampled differences, not a count.
- **YouTube drops speech too, less often**: one video (Nets, 2021) lacks 61
  words that Wishper has, including the first question.
- The two videos with uploader subtitles do not settle it: those tracks are
  verbatim machine transcripts (they keep "um" and repeats), so they sit
  closer to YouTube (8.6% and 10.5%) than to Wishper (12.3% and 18.2%).

Not tested: whether the skipped passages come from the VAD filter or from the
batched inference, and whether other Wishper settings avoid them. Wishper
returns text without timestamps, so a combined video cannot be cut to the
coach's part from its transcript; the captions carry word times.
