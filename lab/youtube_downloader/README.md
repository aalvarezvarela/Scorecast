# NBA recap audio archive

`download_recaps.py` downloads Motion Station's YouTube game recaps as **Opus,
48 kbit/s VBR, mono, 48 kHz**. Select one season or one US calendar date. The
runner uses the same official NBA schedule, game IDs, phase scope and S3 settings
as the [written-report archive](../game_reports/README.md).

## Run

From the repository root, using the project's existing environment:

```bash
# One explicitly selected season, with local files and an S3 copy.
.venv/bin/python lab/youtube_downloader/download_recaps.py download 2025-26

# One US game date; its season is inferred automatically.
.venv/bin/python lab/youtube_downloader/download_recaps.py download --date 2026-10-28

# Coverage summaries: local only, no network or AWS credentials needed.
.venv/bin/python lab/youtube_downloader/download_recaps.py status 2026-27
.venv/bin/python lab/youtube_downloader/download_recaps.py status --date 2026-10-28

# Bounded pilot, without S3 access.
.venv/bin/python lab/youtube_downloader/download_recaps.py download 2025-26 \
  --local-only --max-games 3
```

Starting years such as `2025` are also accepted. A date run inventories the
**whole season** and processes only the selected date. A later season run shares
the same files and manifest, skipping completed work. Re-run the same download
command to resume missing or failed work. `--max-games` limits new processing,
not the expected-game inventory; verified completed local files do not consume
that budget.

Dates are NBA game dates in the US, not the local European date after midnight.
The existing repository season-boundary helper also handles the 2020 bubble.
An explicit season can accompany `--date` if needed. A date with no scheduled
games prints an empty selected-date summary.

By default, downloads use `SETTINGS.s3_bucket`, AWS region and profile, matching
the written archive. `--local-only` avoids AWS access and settings. A local-only
archive can later be uploaded by re-running without `--local-only`: saved audio
is reused. Switching an S3 archive to another bucket or back to local-only in
the same directory is refused; use a separate `--local-root` for that scope.

Options include `--local-root`, `--s3-prefix`, `--language` (default `en`),
`--js-runtime node|deno`, `--search-results` (25), `--request-interval` (2 seconds),
`--upload-window-days` (2), and `--phases`; see `--help`. The default phases are
regular season, play-in, playoffs and the Cup final. Preseason is opt-in;
All-Star exhibitions are excluded. Another language or phase scope requires a
separate local root and S3 prefix to avoid mixing archive configurations.

Use `--diagnostics --max-games 1` to inspect search counts, sample titles and
rejected publication dates when discovery fails. Five consecutive `not_found`
results among games that needed YouTube requests stop and checkpoint the run;
games absent from the channel listing never trigger this stop (see
[Matching](#matching-audio-selection-and-recovery)). Misses remain retryable.

## Install

The optional Poetry `media` group pins yt-dlp and its EJS challenge solver:

```bash
poetry install --with media
```

Alternatively, the generated manual-job requirements include main, scrape and
media, without XGBoost or dashboard dependencies:

```bash
.venv/bin/python -m pip install -r requirements/recap-audio.txt
.venv/bin/python -m pip install --no-deps -e .
```

Install FFmpeg/FFprobe with the `libopus` encoder and Node >=22 on the system
PATH. The runner enables Node explicitly. Deno is supported through
`--js-runtime deno` if installed instead. No browser, login cookies or YouTube
Data API key is needed for the public recaps used in the pilot.
See the [upstream EJS setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

When changing dependencies, follow [requirements/README.md](../../requirements/README.md):
edit `pyproject.toml`, run `poetry lock`, then `scripts/export_requirements.py`.

## Files and coverage

```text
data/nba_game_recaps/seasons/season=2025-26/
  audio/0022500001.opus
  ...
  manifest.json

s3://<configured-bucket>/train_data/game_recaps/v1/season=2025-26/
  audio/0022500001.opus
  ...
  manifest.json
```

The official NBA game ID joins an audio recap to its written report. Only the
final audio is retained; source audio downloads are temporary, and videos are
not downloaded. The Opus file is already compressed and is not wrapped in gzip.

The manifest inventories every expected game, including unfinished games and
search misses. Each entry contains teams, scores, game date and phase, relative
local audio path, S3 key, attempts, last error, SHA-256 and stored bytes. Successful
entries also record YouTube video/channel IDs, canonical URL, title, source and
final durations, original audio codec, format ID, language, retrieval time,
upload date and publication time where exposed. Signed media URLs are not
persisted. The exact output profile is recorded at manifest level.

| Status | Meaning |
| --- | --- |
| `pending` | Scheduled final game, not processed yet. |
| `not_final` | Official schedule does not mark the game final. |
| `not_found` | No verified recap was found, or none is in the channel listing (`error` says which); re-checked on the next run. |
| `downloaded` | Verified local Opus exists; complete for local-only runs, otherwise awaiting upload. |
| `uploaded` | Audio exists in S3 and is reconciled with the manifest. |
| `fetch_error` | Search, matching, language, download, integrity or encoding failed. |
| `upload_error` | Local audio exists but S3 upload failed. |

`status` reports all gaps and optionally filters the summary by date. A download
exits 1 if errors remain in its selected scope; unfinished games and search
misses remain visible but are not treated as processing errors.

## Matching, audio selection and recovery

Discovery starts from a flat listing (ID and title) of every Motion Station
upload, cached in `<local-root>/channel_index.json` and shared by all seasons.
The runner lists the channel again only when a pending game's upload window
(game date + `--upload-window-days`, New York time) has closed after the cached
listing was taken; one listing of about 31,000 uploads takes a few minutes. For
a game whose window had closed before the listing:

- a matching title is verified as below and downloaded;
- no matching title marks it `not_found` without any YouTube request. Such games
  do not count as attempts, toward `--max-games` or toward the five-miss stop.

Games whose window was still open at listing time (yesterday's games, for
example) fall back to the per-game search: Motion Station's channel, then a
broader YouTube search. `--no-channel-index` uses only that per-game search.
A listing more than 1% shorter than the cached one is refused as a probable
truncated response.

Every accepted candidate must match the configured channel,
both team names, final scores and publication window. Standard recaps are
preferred over extended versions. Multiple matching candidates of the preferred
type produce an explicit error instead of selecting a potentially wrong game.
The channel search can omit existing videos, so misses are retryable. This was
observed in the old 2026-10-28 CSV: Thunder/Mavericks was marked missing although
its [recap exists](https://www.youtube.com/watch?v=lnpLTsZP9O4).

Audio selection only accepts audio-only formats in the requested language and
prefers the original track where available. Older untagged tracks are accepted
only if the video's language agrees or no language information is available.
Other-language tracks and combined video/audio formats are never a silent
fallback. Discovery and media download use separate extractions; the download
selects its own current language track in the same session, avoiding reuse of
signed URLs or format IDs from discovery.

FFmpeg explicitly encodes to 48 kbit/s mono, even if the source is already
Opus. yt-dlp's extract-audio quality setting alone can stream-copy an existing
Opus track and would not enforce that profile. The output must pass FFprobe
codec/channel/sample-rate/duration checks and a full decode before an atomic
move into its final location. A local checksum mismatch quarantines the file as
`*.opus.corrupt`; subsequent processing fetches it again.

The season lock covers schedule refresh, processing and checkpoints. The
manifest is checkpointed atomically after every processed game and on handled
interruption. Five consecutive processing errors stop the run. S3 uploads have
SHA-256 checksums and create-only conditional writes; different audio at the
same key is never silently overwritten. The remote manifest uses ETag-based
conditional writes and is published once per run or handled interruption.
Reconciliation recovers remote-only progress and orphan uploads, and reuploads
a missing remote object from a valid local file. Object metadata preserves the
recap identity for recovery after a lost manifest checkpoint.

Some videos may be unavailable or published outside the default window. Inspect
the coverage manifest before concluding a season is complete. If these recaps
later become model inputs, use their publication timestamps: a postgame recap
cannot be used to predict that same game.

### Channel coverage by season (measured 2026-11-01)

A full flat listing returned 31,322 uploads. Matching `Game Recap` titles to the
official schedule by team name and final score (regular season, play-in,
playoffs, Cup final) gives:

| Season | Final games | With recap | Coverage |
| --- | ---: | ---: | ---: |
| 2015-16 to 2019-20 | ~1,300 each | ~10 each | ~1% (score coincidences with later seasons) |
| 2020-21 | 1,171 | 1,108 | 94.6% |
| 2021-22 | 1,323 | 1,307 | 98.8% |
| 2022-23 | 1,320 | 1,296 | 98.2% |
| 2023-24 | 1,319 | 1,297 | 98.3% |
| 2024-25 | 1,321 | 1,295 | 98.0% |
| 2025-26 | 1,322 | 446 | 33.7% |
| 2026-27 (to 2026-10-31) | 88 | 84 | 95.5% |

2025-26 has almost no recaps from October 2025 to February 2026 (30 of 896
games, none of 198 in December) and ~98% from March 2026 on. The listing does
not say whether the missing videos were never published, removed or hidden.
A few titles are truncated (`Game Recap: Rockets 119, Thunder`) and cannot be
matched. Before the channel index, a 2025-26 season run stopped at its first
five games and never reached March.

The first two official recaps were located on NBA.com through each game's
`gameRecap` metadata. Anonymous playback options were obtainable, but the
investigated playback path reported Widevine; a downloadable unencrypted audio
track has not been verified. No NBA.com fallback is implemented or promised by
this runner. Historical season completeness remains a source-availability
limitation, not an audio encoding guarantee.

## Validation and measurements

The live local pilot on 2026-10-31 downloaded Heat/Hawks from 2026-10-28
(`0022600132`, YouTube `X03mj26ELHE`) through the new CLI, producing 649,963 bytes
of verified, decodable mono Opus. Its artifacts are under
`data/nba_highlights/archive_pilot/season=2026-27/`. S3 recovery and conditional
writes are covered by simulated-service tests; this pilot did not upload to S3.

The earlier compression comparison used 11 existing MP3 recaps, with a mean
duration of about 112 seconds. MB and GB below are decimal:

| Format | Total for 11 games | Mean per game | Reduction from MP3 | Projection for 1,322 similar games |
| --- | ---: | ---: | ---: | ---: |
| Existing MP3, 192 kbit/s stereo | 29.545 MB | 2.686 MB | — | 3.551 GB |
| Opus, 32 kbit/s mono | 5.097 MB | 0.463 MB | 82.7% | 0.613 GB |
| **Opus, 48 kbit/s mono** | **7.587 MB** | **0.690 MB** | **74.3%** | **0.912 GB** |
| Opus, 64 kbit/s mono | 10.108 MB | 0.919 MB | 65.8% | 1.215 GB |

VBR size depends on duration and content; historical/extended recaps may be much
longer. The comparison transcoded existing MP3 files, whereas the new runner
encodes directly from YouTube's audio track. Comparison files and detailed
measurements remain in `data/nba_highlights/compression_probe/`.

```bash
.venv/bin/pytest -q tests/ingestion/test_recap_audio.py \
  tests/ingestion/test_game_reports.py tests/ingestion/test_season_reports.py \
  tests/ingestion/test_download_season_reports.py
```

`get_game_recaps.py` remains the earlier daily exploratory script, including its
video and MP3 modes. Use **`download_recaps.py`** for the resumable Opus archive.
