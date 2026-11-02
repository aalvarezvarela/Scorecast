# NBA recap audio transcripts

Status: Reference — implemented; HTTP and S3 recovery are tested with simulated services. A live Wishper/CUDA run and S3 upload have not been performed for this change.

`lab/youtube_downloader/transcribe_recaps.py` downloads verified NBA recap audio
using the existing Motion Station collector, sends each Opus file to Wishper,
and stores one compressed JSON per game. The official NBA game ID, season
partition, JSON fields, gzip representation, and S3 recovery behavior follow the
[written-report archive](../../lab/game_reports/README.md).

## Setup

Use this repository's existing environment and AWS configuration. No Whisper,
CUDA, or inference dependencies are added here: those belong to the separate
Wishper server. The existing `requests` dependency handles multipart uploads.
For the download stage, install the optional media dependencies:

```bash
poetry install --with media
```

Alternatively, use the existing exported requirements:

```bash
.venv/bin/python -m pip install -r requirements/recap-audio.txt
.venv/bin/python -m pip install --no-deps -e .
```

Downloading requires FFmpeg/FFprobe with libopus and Node >=22 (or Deno using
`--js-runtime deno`). `--skip-download` does not need yt-dlp or a JavaScript
runtime, but audio integrity checks still use FFprobe.

Start Wishper in its own terminal, on the machine with the model:

```bash
cd /home/panchojasen/Projects/Wishper
.venv/bin/python -m uvicorn wishper.api:app --host 127.0.0.1 --port 8000 --workers 1
```

In the NBA repository terminal, export the client settings:

```bash
export WISHPER_BASE_URL=http://127.0.0.1:8000
# Only if the Wishper server has API_KEY configured:
# export WISHPER_API_KEY='the-same-secret-as-the-server'
```

The CLI reads these environment variables directly; it does not load them from
a `.env` file. `--api-base-url` overrides the URL. Keep keys out of command
arguments and source control; neither the key nor the server URL is saved in
the archive. On another machine/container, replace localhost with an address
that reaches Wishper. Wishper must listen on an accessible interface.

The client calls `POST /transcribe` with `file`, `batch_size`, and `chunk_minutes`
as multipart form fields, and consumes `{"text": "..."}`. Authentication uses
`Authorization: Bearer <key>`. Redirects are rejected rather than followed.
The client uses Requests' [multipart and timeout support](https://requests.readthedocs.io/en/latest/user/quickstart/).

## Run

From the root of `NBA_over_under_predictor`:

```bash
# Download audio, archive audio in S3, transcribe, and archive transcripts in S3.
.venv/bin/python lab/youtube_downloader/transcribe_recaps.py download --date 2026-10-28

# One explicitly selected season; rerun the same command to resume.
.venv/bin/python lab/youtube_downloader/transcribe_recaps.py download 2025-26

# Transcribe already archived audio without requesting NBA or YouTube.
.venv/bin/python lab/youtube_downloader/transcribe_recaps.py download 2025-26 --skip-download

# Inspect transcript coverage; no AWS, YouTube, or Wishper requests.
.venv/bin/python lab/youtube_downloader/transcribe_recaps.py status 2025-26
.venv/bin/python lab/youtube_downloader/transcribe_recaps.py status --date 2026-10-28
```

For a bounded local pilot using separate directories and no AWS access:

```bash
.venv/bin/python lab/youtube_downloader/transcribe_recaps.py download \
  --date 2026-10-28 --local-only --max-games 2 \
  --audio-root data/nba_recap_transcript_pilot/audio \
  --local-root data/nba_recap_transcript_pilot/transcripts
```

`status` reads `profile=<--transcription-profile>/season=<season>/` under
`--local-root`; pass the same profile you ran with (default `wishper-v1`).

Interactive terminals show a progress bar for each stage, with a game counter,
percentage, elapsed time, estimated remaining time, and the current game/status.
Bars track final games in the selected season/date, including checks of cached
files and unavailable audio. A `--max-games` pilot can stop before 100% because
that option limits new processing rather than the selected inventory. Wishper
returns the completed text only, so the bar advances between games rather than
showing progress within a single audio file. Bars are hidden when output is not
a terminal; per-game messages and the final JSON summary remain available.

`--max-games` limits processing separately in each stage; it does not shrink
either manifest. Completed transcripts do not consume the transcript-stage
budget. A date run shares the full season inventory and only processes the
selected US NBA game date. Starting years such as `2025` are also accepted.
Default phases match the crónicas: regular season, play-in, playoffs and Cup
final. Preseason is optional; All-Star exhibitions are excluded.

Without `--local-only`, both archives use `SETTINGS.s3_bucket`,
`SETTINGS.s3_aws_region`, and `SETTINGS.s3_aws_profile`, just like the written
reports. The script uses the existing AWS credential chain. A full run can
promote a local-only archive to S3 without retranscribing saved files. Switching
to a different bucket, back to local-only, or to another phase/language/profile
requires separate destinations.

The audio stage finishes before transcription begins. If audio collection
stops early, its checkpoints remain usable; inspect them and run with
`--skip-download` to transcribe the audio already collected. On a machine
without the local audio archive, `--skip-download` copies the audio manifest
from S3 and restores each Opus file, checksum-verified, when it is transcribed. Missing source
recaps stay visible rather than being replaced with unrelated videos.
See the [audio collector documentation](../../lab/youtube_downloader/README.md)
for source matching, historical coverage, encoding, and discovery flags.

## Storage and document format

```text
data/nba_game_recaps/seasons/season=2025-26/
  audio/0022500001.opus
  manifest.json

data/nba_game_recap_transcripts/profile=wishper-v1/season=2025-26/
  transcripts/0022500001.json.gz
  manifest.json

s3://<configured-bucket>/train_data/game_recaps/v1/season=2025-26/
  audio/0022500001.opus
  manifest.json

s3://<configured-bucket>/train_data/game_recap_transcripts/v1/profile=wishper-v1/season=2025-26/
  transcripts/0022500001.json.gz
  manifest.json
```

The written-report prefix `train_data/game_reports/v1` is preserved. Transcripts
can be joined to written reports and audio using the official game ID, including
its leading zeros. `v1` is the document schema; `profile=` is the transcription
model/settings label, so each Wishper configuration is a sibling archive of the
same games and two profiles can be compared directly. The profile is added to
both the local root and the S3 prefix; override the parts above it with
`--local-root` and `--s3-prefix`. Audio destinations use `--audio-root` and
`--audio-s3-prefix`. Audio and transcripts must use separate roots and prefixes.

Transcript files use the existing deterministic gzip/JSON serializer, without a
duplicate `paragraphs` array. Common written-report fields are preserved:
`schema_version`, `season`, `phase`, `game_id`, `date`, teams, scores, `title`,
`source_url`, publication/retrieval timestamps, `text`, and `word_count`.
Author/byline/agency fields are empty because the source is a narrated video,
not an attributed article. Additional fields identify the artifact and source:

```json
{
  "artifact_type": "audio_transcript",
  "game_id": "0022500001",
  "season": "2025-26",
  "text": "- The full transcript, preserving paragraph breaks.",
  "transcription": {
    "provider": "wishper",
    "profile": "wishper-v1",
    "chunk_minutes": 15,
    "batch_size": 4
  },
  "audio": {
    "sha256": "<checksum-of-the-source-opus>",
    "bucket": "<source-bucket-or-empty-for-local-only>",
    "s3_key": "train_data/game_recaps/v1/season=2025-26/audio/0022500001.opus"
  }
}
```

The example shows selected fields; full documents also retain recap metadata
(video/channel IDs, URL, title, language, duration, and source timestamps) and
audio size/path information. Do not treat the text as a generated summary: it
is the narration transcribed by Wishper.

`provider`, `profile` and `chunk_minutes` identify the archive: they change the
text, and a run with different values is rejected. `batch_size` only changes
throughput, so it is recorded per document and may differ between runs of one
archive (for example, lowered to fit GPU memory).

`published_at_utc` is the source video's known publication timestamp, not the
transcription time or game date. It remains empty if unavailable. Recap features
must only be used after publication when building historical prediction inputs;
never use a game's postgame transcript to predict that same game. The current
pipeline archives data only and does not add it to model features.

Read a transcript using the same helper as written reports:

```python
from lab.game_reports.season_reports import load_report

response = s3.get_object(Bucket=bucket, Key=transcript_key)
document = load_report(response["Body"].read())
text = document["text"]
```

## Recovery and errors

Each transcript manifest inventories all games in the source audio manifest,
including pending/unplayed games and unavailable audio. It records attempts,
last attempt, errors, source-audio hash, compressed transcript hash, bytes, paths,
and status. Local checkpoints are atomic after each processed game and on a
handled interruption. The S3 manifest is published once at run end/interruption.

| Status | Meaning |
| --- | --- |
| `pending` | Final game waiting to be processed. |
| `not_final` | The source schedule does not mark the game final. |
| `waiting_audio` | No verified audio/recap metadata is available yet; retried next run. |
| `downloaded` | Valid local transcript saved; complete locally or awaiting S3 upload. |
| `uploaded` | Transcript is reconciled with S3. |
| `transcription_error` | Source audio integrity, API request/response, or local write failed. |
| `upload_error` | Local transcript is valid, but its S3 upload failed. |

Reruns reuse completed local/S3 transcripts. Upload retries do not call Wishper
again. Remote-only and orphan transcript uploads are recovered, and missing
remote objects are restored from local files. Missing local audio can be restored
from S3 after checksum/identity checks. Corrupt local transcripts are quarantined
as `*.json.gz.corrupt` before regeneration. A different `chunk_minutes` is
rejected rather than mixed into an existing archive; a different profile writes
to its own `profile=` archive. A changed game identity (date, phase, teams or
score) stops the run.

When a game's audio is downloaded again after it was transcribed, the re-encode
can produce a different hash. The saved transcript is kept and stays tied to the
audio it was made from (its `audio_sha256`); the run prints a warning and goes
on. S3 refuses to overwrite the archived audio, so that copy is the one the
transcript describes. If the transcript itself is lost, it is regenerated from
the current audio and records that hash.

S3 transcript puts reuse written-report SHA-256 checksums and create-only
conditional writes. Different content is never silently overwritten. Manifest
updates use ETags to reject stale writers. Season locks cover both the source
audio and transcript directories, preventing simultaneous local download/read
and concurrent transcript runs.

Only one request is sent to Wishper at a time. Defaults are batch size 4,
15-minute chunks, a 10-second connection timeout, and a 1,800-second read timeout.
Use `--batch-size`, `--chunk-minutes`, and `--api-timeout` to adjust them. The
200 MiB file limit is checked before upload. HTTP 503 is retried at most twice
(`--busy-retries`), reopening the file each time and honoring a numeric
`Retry-After` up to 30 seconds. Longer waits stop the run. Network failures and
timeouts are never retried automatically because the server may still be working.
Authentication errors, HTTP 500, or exhausted 503 retries stop and checkpoint the
run. Other errors are recorded per game; three consecutive errors stop the run.

An empty/malformed transcript is recorded as an error rather than saved as a
successful written-report-compatible document. A processing/upload error returns
exit code 1; an interruption returns 130. Missing source recaps and unfinished
games remain visible but are not processing errors.

`--transcription-profile` is an operator-supplied label, not a model identifier
reported by Wishper: its API returns text only. Configure the intended model,
language, and formatting on the Wishper server and pass a new profile label when
those settings change; the new label gets its own `profile=` archive. Labels use
letters, digits, `.`, `_` and `-`. Keep Wishper at one worker per GPU and
avoid concurrent inference through Streamlit on that GPU.

## Validation

```bash
.venv/bin/pytest -q tests/ingestion/test_recap_transcripts.py \
  tests/ingestion/test_recap_audio.py tests/ingestion/test_game_reports.py \
  tests/ingestion/test_season_reports.py tests/ingestion/test_download_season_reports.py
```

Tests substitute HTTP and S3, generate the actual multipart request in memory,
and cover document compatibility, retry budgets, credentials, error checkpoints,
date/season selection, source locks, checksums, immutable puts, and recovery.
They do not contact YouTube, Wishper, or AWS or verify model accuracy.
