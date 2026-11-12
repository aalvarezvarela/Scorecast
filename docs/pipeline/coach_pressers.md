# NBA coach postgame presser archive

Status: Partial — code implemented 2026-11-06 in `lab/coach_pressers/` (run
guide: its `README.md`; tests: `tests/ingestion/test_coach_pressers_*.py`).
No live run beyond checks yet: checkpoint 1 (the channel list), the pilot and
the season runs are pending. The sections below are the design; where the
implementation differs from the first plan it says so.

One resumable command, run per season, finds the head coach's postgame press
conference for both teams of every NBA game from 2019-20 on, archives its audio
in S3, and reports exactly which pressers are missing. It is an archive only:
nothing here becomes a model feature. If it ever does, use each video's
publication time; a postgame presser can never be an input for the same game.

## Goal, scope and ground rules

**Unit of work:** one *expected presser* = one (game, team) pair. Phases are the
recap archive's defaults: regular season, play-in, playoffs and the Cup final
(game ID prefixes `002`, `005`, `004`, `006`). About 2,640 expected pressers per
full season, roughly 18,000 for 2019-20 to 2025-26.

**Primary source:** the 30 official team YouTube channels. Secondary sources
(local TV channels, independent outlets) are added only to fill gaps, after the
coverage report (step 8b).

**Deliverables:**

1. A per-season manifest (local and S3) listing every expected presser with its
   status, the chosen video, who spoke and where the audio is.
2. A coverage report (season × team, % found) plus `missing.csv` and
   `review.csv`.
3. Audio in S3, one file per matched presser, plus the coach-only cut for
   combined videos, a trimmed `.info.json.gz` and the English auto-captions.
   The video's YouTube chapters (segment labels), where it has any, are kept
   in the manifest and embedded in the audio file.
4. Optional, later: transcripts through Wishper, as for the recaps.

**Ground rules:**

- Run on the user's own computer (home internet), not a cloud server or the
  GitHub runner. YouTube's bot checks hit data-centre IPs much harder.
- Personal research use only. The S3 copy lives in the private bucket
  (`SETTINGS.s3_bucket`); never make objects public, never hand out presigned
  URLs, never re-upload the audio anywhere.
- Do not bypass access controls: no VPN or proxy to dodge geo-blocks, no DRM
  circumvention, no logging into paid services. NBA.com is out of scope.
- Rate-limit every request (see Operations).
- Stop at each human checkpoint (Milestones) and wait for approval.

## What the research found

YouTube coverage is strong from 2022-23 onward and patchy before it. Plan for
that; early gaps are not bugs. These figures come from small hand samples
(November 2026), so treat them as rough expectations.

| Period | Coach pressers found on YouTube (sample) | Notes |
| --- | --- | --- |
| 2023-24 to 2025-26 | 9 of 10 | Nearly every team channel posts its coach. |
| 2021-22 regular season | 3 of 10 | Suns, Cavs, Pistons posted; Celtics, Warriors, Hawks, Bulls, Hornets did not show up. |
| 2020-21 regular season | 1 of 6 | Magic posted; Celtics and Clippers did not. |
| 2020 bubble playoffs | 2 of 2 | Nuggets and Clippers both posted. |
| 2019-20 regular season, 2021 playoffs | 0 of 8 | Pressers happened (quoted in news) but no video found by search. |

The gaps before 2022 may partly be a search limitation. The channel inventory
(step 4) is the real test, and the coverage report settles it.

**Facts the design depends on:**

- Titles are inconsistent across teams and over time. Some never say
  "postgame" or "press conference" (e.g. "Steve Kerr Recaps Warriors Loss
  Against San Antonio Spurs | April 9, 2025"). Matching must use the coach's
  name and the publication time, not keywords alone.
- Some teams post one combined video with the coach and players together
  (Suns, Pistons, Nuggets, Raptors). These need the coach's segment cut out,
  using YouTube chapters or description timestamps where they exist.
- Fox Sports game pages list each game's YouTube videos from several channels:
  a possible discovery index for 2020 onward, but pages for 2019 and some 2021
  games list nothing. Not used by the pipeline.
- NBA.com hosts a full coach presser for both teams from about October 2022,
  but playback is restricted (the recap investigation found Widevine) and
  yt-dlp's NBA extractors are marked broken. Not a usable source.
- Local TV websites (NBC Sports Bay Area, NBC Sports Boston) host some pre-2022
  pressers for teams missing on YouTube. Whether they can be downloaded is
  untested.
- Team websites published written postgame quotes in some seasons (Nets,
  Celtics). That is a text-only fallback (open question at the end).

## What already exists and is reused

The recap archive solved most of the infrastructure. Reuse it; do not write a
second copy.

| Need | Existing code | Notes |
| --- | --- | --- |
| Season games, phases, game IDs, US dates, scores | `lab/game_reports/season_reports.py` (`ScheduleLeagueV2`, `PHASES`, `season_games`) | Same inventory as recaps and written reports, so the official game ID joins all three archives. Tip-off instant from `gameDateTimeUTC`, already read in `fetch_data/nba_schedule/fetch_nba_schedule.py`. |
| Request pacing, atomic checkpoints, season locks | `season_reports.py` (`RequestPacer`, `atomic_write`, `checkpoint`) | |
| yt-dlp options with the JS challenge solver | `lab/youtube_downloader/recap_audio.py` (`youtube_options`, `require_tools`) | yt-dlp and yt-dlp-ejs are pinned in the Poetry `media` group; Node >=22 or Deno required. |
| Publication instant of a video | `recap_audio.publication` | Uses `release_timestamp`/`timestamp`; never invents an instant from `upload_date`. |
| Flat channel listing with a cached index | `recap_audio.list_channel`, `channel_index` | Cached listing shared by all seasons; refused if >1% shorter than the cached one. |
| Audio track choice, integrity check, Opus encoding | `recap_audio.select_audio`, `inspect_audio`, `encode_audio` | |
| S3 writes | `recap_audio.S3AudioStore`, `reconcile` | SHA-256 checksums, create-only puts (`IfNoneMatch="*"`), ETag-conditional manifest, recovery of orphan or missing objects. |
| Transcription | Wishper client, `lab/youtube_downloader/recap_transcripts.py` | Replaces `faster-whisper` in the first draft. |

`recap_audio.py` hard-codes the Motion Station channel and its `AudioConfig`.
**Step 0** of the implementation moves the generic pieces (options,
publication, listing of any channel, audio selection, encoding, the S3 store
with a configurable prefix) into `lab/youtube_downloader/youtube_media.py`,
leaves `recap_audio.py` importing them, and keeps
`tests/ingestion/test_recap_audio.py` green without changes to its assertions.

## Environment and project layout

Poetry environment with `poetry install --with media`; the `media` group
gained `openai` (step 7). `requests`, `pandas`, `lxml` and `nba_api` were
already installed. No YAML, `typer`, `python-dateutil`, `rapidfuzz` or
`faster-whisper`: the CLI uses `argparse` like the other collectors, small
reference tables are CSV, title dates are parsed with explicit regexes (fuzzy
date parsing misreads titles like "Game 1"), and names are matched as whole
words after normalisation (accents, punctuation, initials), which the 24
fixture titles need and which keeps the tests runnable in CI (CI does not
install the `media` group; `openai` and `yt_dlp` are imported lazily).

System tools: FFmpeg/FFprobe with `libopus`, Node >=22 (or Deno with
`--js-runtime deno`). yt-dlp is pinned; bump the pin (and the lock) when
YouTube breaks it, rather than `pip install -U` outside Poetry.

The code lives under `lab/`, next to the recap archive, because it is a manual
archive, not part of training or serving. If pressers ever feed features, the
loader moves to `src/nba_ou/data_processing/` like `game_texts/`.

```
lab/coach_pressers/
  README.md              # run guide
  get_pressers.py        # CLI: run / status / channels / llm-check
  pipeline.py            # the steps of one season in order
  config.py              # PresserConfig, local and S3 layout
  manifest.py            # the season manifest, statuses, season lock
  teams.py, names.py     # tricodes and Wikipedia names; name normalisation
  channels.csv           # step 3 (written by `channels --resolve`)
  coaches_overrides.csv  # team, coach, start_date, end_date, kind, note
  catalog.py             # step 1: expected pressers from the season schedule
  coaches.py             # step 2: coach spells, staff and player names
  channels.py            # step 3: resolving and checking handles
  crawl.py               # step 4: channel index and video metadata cache
  classify.py            # step 5: is this a coach presser?
  title_dates.py         # dates written in titles
  match.py               # step 6: video -> (game, team)
  llm_judge.py           # step 7: LLM picks among candidates
  llm_cases.py           # 20 hand-labelled judge inputs (tests and llm-check)
  report.py              # step 8: reports and the review loop
  archive.py             # steps 9-10: audio, cut, captions, S3, state backup
lab/youtube_downloader/youtube_media.py  # helpers shared with the recap archive
tests/ingestion/
  test_coach_pressers_{catalog,coaches,channels,crawl,classify,dates,match,
                       llm_judge,report,archive,pipeline}.py
  test_youtube_media.py
tests/support/coach_pressers.py          # schedule, Wikipedia and config builders
```

CLI examples, from the repository root:

```bash
# Whole season, metadata only (steps 1-8), no downloads.
.venv/bin/python lab/coach_pressers/get_pressers.py run 2023-24 --dry-run
# Several seasons, metadata only.
.venv/bin/python lab/coach_pressers/get_pressers.py run 2019-20:2025-26 --dry-run
# One team, for pilots and debugging; --local-only skips AWS entirely.
.venv/bin/python lab/coach_pressers/get_pressers.py run 2021-22 --team BOS --dry-run --local-only
# Download and archive to S3.
.venv/bin/python lab/coach_pressers/get_pressers.py run 2025-26
# Coverage summary from the local manifest; no network.
.venv/bin/python lab/coach_pressers/get_pressers.py status 2025-26
```

Every run is idempotent and resumable: re-running a season skips work already
recorded in its manifest and caches.

## Data layout

No database. Each stage writes plain files that the next stage reads, following
the recap archive: one JSON manifest per season as the record of truth, JSON
caches for anything fetched from the network, CSV for reports and for the small
hand-edited tables. Local files are written atomically (temporary file +
rename) and checkpointed after every processed item.

```
data/coach_pressers/                       # gitignored (data/)
  raw/schedule/season=2025-26.json         # ScheduleLeagueV2 response
  raw/wikipedia/<article title>.html       # league- and team-season articles
  raw/players/season=2025-26.json          # box-score player names per team
  channels/<channel_id>/index.json         # flat listing: id, title, duration, tab
  channels/<channel_id>/videos.jsonl.gz    # full metadata of pre-filtered videos
  llm_decisions.jsonl                      # cache keyed by input hash
  seasons/season=2025-26/
    coaches.json                           # coach spells + staff + roster names
    candidates.json                        # per (game, team): videos, class, score, evidence
    manifest.json                          # one entry per expected presser
    reports/coverage.csv, missing.csv, review.csv, review_decisions.csv,
            coach_changes.csv, proposed_overrides.csv,
            coach_timeline_issues.csv, trim_manual.csv
  tmp/                                     # source audio while encoding; emptied after upload
  logs/<utc>.log

s3://<SETTINGS.s3_bucket>/train_data/coach_pressers/v1/
  channels/<channel_id>/index.json, videos.jsonl.gz
  llm_decisions.jsonl
  season=2025-26/
    manifest.json
    coaches.json, candidates.json
    reports/*.csv                          # review_decisions.csv cannot be recreated
    audio/<game_id>_<TEAM>_<video_id>.opus
    audio/<game_id>_<TEAM>_<video_id>.coach.opus   # coach segment of a combined video
    captions/<game_id>_<TEAM>_<video_id>.en.vtt
    info/<game_id>_<TEAM>_<video_id>.info.json.gz
```

Object keys use stable identifiers only: the official game ID (with leading
zeros), the team tricode and the video ID. The coach and opponent go in the
manifest and in S3 object metadata, not in the key, because a review can change
who spoke and S3 objects are create-only. A different video chosen later gets a
new key; the old object stays and the manifest points to the new one.

**Manifest entry** (one per game × team; keys `"<game_id>:<TEAM>"`):

| Field | Content |
| --- | --- |
| identity | `game_id`, `team`, `opponent`, `is_home`, `date` (US Eastern), `tip_utc`, `phase`, `playoff_game_no`, scores |
| `coach_expected` | from the coach spells (step 2) |
| `status` | `pending`, `not_final`, `not_found`, `review`, `selected`, `uploaded`, `downloaded` (local only), `fetch_error`, `upload_error`, `unavailable` |
| `selection` | `video_id`, `channel_id`, `title`, `published_at_utc`, `duration_s`, `kind` (coach, combined), `coach_actual`, `decided_by` (rules, llm, human), `confidence`, `reason`, `coach_chapter`, `trim_start_s`, `trim_end_s`, `needs_trim`, `chapters` |
| `chapters` | inside `selection`: the video's chapters as yt-dlp returns them, `[{start_time, end_time, title}]` in seconds of the full video, or `[]`; the coach cut gets the chapters inside its range, shifted to start at 0 |
| `llm` | the judge's raw answer, model, input hash and error, when step 7 ran |
| `archive` | `audio`, `coach`, `captions`, `info`, each `{key, local_path, sha256, bytes}` (`coach` adds `start_s`, `end_s`, `duration_s`); `duration_s`, `chapters_embedded`, `source_codec`, `format_id`, `language`, `source_duration_s`, `retrieved_at_utc` |
| bookkeeping | `attempts`, `last_attempt_at`, `error` |

Validation follows `season_reports.validate_manifest`: a changed game identity
stops the run, and a scheduled game that disappears from the schedule is an
error, not a silent coverage drop. The S3 manifest is published once at the
end of each run (or on a handled interruption) with an ETag-conditional write.

Team codes are the NBA tricodes from the schedule (BOS, GSW, PHX, …). Dates are
ISO strings; instants are UTC ISO strings.

## Pipeline steps

`get_pressers.py run` executes the steps in order for the seasons given. Steps
1-8 only read metadata and are cheap; steps 9-10 download and upload, and are
skipped with `--dry-run`. Deterministic rules decide whenever they can; the LLM
(step 7) only chooses among candidates the rules could not separate.

1. **Expected pressers** (`catalog.py`)
   - Load the season with `ScheduleLeagueV2` through `season_games` (cache the
     raw response), keep the four default phases, add `tip_utc` and the playoff
     game number, and write two manifest entries per final game.
   - Checks against the recap inventory (final games, all four phases): 1,171
     (2020-21), 1,323 (2021-22), 1,320 (2022-23), 1,319 (2023-24), 1,321
     (2024-25), 1,322 (2025-26). Regular-season games: 1,059 (2019-20 incl.
     bubble seeding games), 1,080 (2020-21) and 1,230 later; the Cup final is
     not one of the 1,230.
2. **Coach timeline** (`coaches.py`): coach spells per team and the names that
   can appear in titles. Full rules in the coach-change section below.
3. **Channel registry** (`channels.csv`, built once, shared by all seasons)
   - One official channel per team, given as an `@handle` URL. Do not guess
     handles: resolve each with yt-dlp (`extract_info(url, download=False,
     process=False)`), store the `channel_id`, and confirm that
     `channel`/`uploader` names the team.
   - Every row starts with `verified_by_human = false` until checkpoint 1;
     step 4 only reads rows the user set to `true`, so a season run stops
     after step 2 until the list is confirmed.
   - Secondary channels start empty and are added only in step 8b.
4. **Video inventory** (`crawl.py`, cached across seasons)
   - For each channel, a flat listing of the `/videos` and `/streams` tabs
     (skip `/shorts`), cached in `channels/<channel_id>/index.json` as in the
     recap channel index. Flat entries carry id, title and usually duration,
     but no reliable date, so the whole channel is listed once rather than
     paged to a date cut-off. Re-list only when a pending game's window has
     closed after the cached listing; refuse a listing more than 1% shorter
     than the cached one.
   - Pre-filter on the title alone with the classifier (step 5), using that
     season's staff and player names. Fetch full
     metadata (`extract_info(video_url, download=False)`) only for titles that
     come out `coach`, `combined`, a postgame cue with an unknown name, or
     `requires_llm` (a plausible presser with a warning keyword, no postgame
     cue, or unusual duration). A cached listing also admits newly eligible
     titles on rerun; no cache deletion is needed.
     Store id, title, description, `timestamp`/`release_timestamp`,
     `upload_date`, duration, chapters, `live_status` and `availability` in
     `videos.jsonl.gz`; never store format lists or signed URLs. Re-runs fetch
     only ids not yet cached.
   - Each tab is ordered by publication, newest first, so the survivors that
     belong to one season are found with a binary search on their listing
     position (each probe is a cached metadata request), plus a margin of five
     on each side. A season run costs its own uploads plus a few dozen probes
     per channel, not the channel's whole history; 2019-26 is still on the
     order of 10,000-20,000 extractions in total.
5. **Classify** (`classify.py`)
   - Class `coach`, `combined`, `player` or `other`, with a score from 0 to 1.
     Rules and test titles are in the classifier section.
   - Names: whole-word match of the normalised title against every coach and
     staff name for that team-season (full name, joined initials, "coach
     <last>", or the last name alone when no player shares it) and against the
     team's player names.
   - Warning terms and a coach name without a postgame cue set `requires_llm`
     on plausible pressers instead of dropping them. Duration under 45 seconds
     or over 25 minutes (coach) / 45 minutes (combined) also requires the judge.
     These referrals survive the minimum-score filter, even after a length
     penalty, and are never auto-accepted.
6. **Match to a game** (`match.py`)
   - A video from team T's channel can only belong to one of T's games. The
     window is by instant, not by date: published at or after `tip_utc` + 90
     minutes and before the tip-off of T's next game (at most 3 days). This
     separates back-to-backs, where a late western game is published after
     midnight UTC on the day of the next game. If a video has only
     `upload_date`, fall back to the dates `[upload_date − 2 days,
     upload_date]` and treat any back-to-back in that span as ambiguous.
   - Order of evidence: a date written in the title (formats seen: `2-26-26`,
     `02.20.2026`, `4.9.2025`, `11.05.23`, `Nov. 5, 2023`, `April 9, 2025`,
     `March 11, 2024`); then the opponent named in the title; then a playoff
     `Game N`; then the publication window.
   - A video outside the window whose title date or opponent names one of T's
     games is still a candidate (`match_method = title_only`), for late
     uploads; it never auto-accepts.
   - **Auto-accept** when exactly one candidate for a (game, team) has class
     `coach` or `combined`, score 0.8 or higher, and a title date or opponent
     that agrees with the window, and no `requires_llm` flag. The entry becomes `selected` with
     `decided_by = rules`.
   - Everything else goes to step 7: two or more candidates, a best score
     between 0.4 and 0.8, a name on the staff that is not `coach_expected`, an
     unknown name with a postgame cue, an ambiguous back-to-back, or an
     explicit `requires_llm` referral regardless of its score.
   - **Gap fill (step 6b, `gap_fill.py`).** A game left `not_found` (no
     candidate at all; not one a person closed in `review.csv`) gets a second
     look: every upload of the team's official channels published from the
     game date to 3 days later gets metadata, whatever its title says, and
     every one the window assigns to the game goes to step 7 with
     `requires_llm` and the note `gap_fill` (at most 15, coach-like titles
     first). Outlet channels get no new requests: their pressers nearly always
     name the coach, so step 4 finds them, and their step 4 videos join the
     pool at no cost. The window is located
     from the dates already cached for the channel, plus a few binary-search
     probes, so the cost is the uploads of the gap games' own windows. Skipped
     with `--no-llm` or `--no-gap-fill`.
7. **LLM tie-break** (`llm_judge.py`)
   - **Model:** GPT-6 Luna (`gpt-6-luna`) through the OpenAI Responses API
     (`client.responses.create`, `store=False`) with Structured Outputs:
     `text.format` is a strict `json_schema` built per call, so `video_id` is
     an enum of the candidates (or null) and `coach_chapter` an enum of safe
     references such as `v0_ch0`. Original chapter titles stay in the input,
     outside the schema, so quotes and other free-form text do not invalidate
     the strict enum. References are checked against the selected video and
     decoded back to original titles before validation, caching and trimming.
     Existing cached decisions remain reusable. Reasoning effort is `none`
     (`--llm-reasoning`; the model
     accepts none, low, medium, high, xhigh, max, not minimal) and is part of
     the cache key. Model and spending cap are CLI options (`--llm-model`,
     `--llm-budget`); the key is `OPENAI_API_KEY` (env var, or `[OpenAI]` in
     `config.secrets.ini`) and is never logged. Confirm the model ID and price
     with `get_pressers.py llm-check` once the key exists.
   - **Input, one call per (game, team):** compact JSON with the game (date,
     tip-off, home/away, opponent, score, phase, playoff game number), the
     team's other games within 3 days, `coach_expected`, the team's staff list,
     and each candidate: video_id, channel, title, publication instant,
     duration, up to 400 characters of the description after removing URLs,
     common promotional lines and redundant whitespace, the chapter
     titles, `requires_llm` and `rule_flags` explaining the classifier's doubts.
   - **Instruction:** pick the video in which this team's head coach (or the
     acting coach that night) speaks to the media after this game, or answer
     none. Say whether it is coach-only or combined, and if combined, which
     chapter is the coach. Warning words can describe postgame discussion
     topics (the locker room, a quarter, tonight's win, the next opponent);
     neither those words, missing postgame keywords nor unusual duration alone
     disqualifies a video. Actual pregame/practice/player-only content is
     still rejected based on all available context.
     Step 6b candidates (`gap_fill`) are all the window's uploads: pick one
     only if its title, description, chapters, duration or timing show the
     coach after this game; a player-titled video can be the full
     availability with the coach in a chapter.
   - **Output** (strict JSON schema / structured outputs): `video_id` or null,
     `kind` (coach, combined, none), `coach_name`, `coach_chapter` or null,
     `confidence` 0-1. No explanation or `reason` is requested. The instruction
     is shortened and input JSON omits optional whitespace. Legacy cached
     answers with `reason` remain readable; new selections leave it empty.
   - **Guardrails:** reject any `video_id` not among the candidates;
     `confidence` under 0.6 goes to `review.csv`, not to `selected`; append
     every answer to `llm_decisions.jsonl` keyed by a hash of the input (with
     model, tokens and cost), so re-runs cost nothing; stop and report at the
     spending cap. `--no-llm` sends every ambiguous case to `review.csv`, so
     the pipeline works before the LLM is wired in.
   - **Gaps:** a step 6b game the judge answers "none" for stays
     `not_found` (the answer is kept in the entry's `llm`); a pick becomes
     `selected` like any other, and an error or a pick under 0.6 confidence
     goes to `review.csv`. Gaps the spending cap leaves unjudged stay
     `not_found` and are retried on the next run.
   - **Cost:** about 1,500 input and 100 output tokens per call; at the listed
     $0.10 per million input and $0.50 per million output tokens, 10,000 calls
     cost about $2. [source](https://llmgateway.io/models/gpt-6-luna)
   - **Accuracy check:** in the pilot, compare 50 LLM decisions with hand
     labels before trusting it for whole seasons.
8. **Coverage report** (`report.py`, per season in `reports/`)
   - `coverage.csv`: % of expected pressers found per team and phase, and how
     each was decided (rules, LLM, human), % of found pressers with chapters;
     after step 9, also % archived in S3.
   - `missing.csv`: every unmatched (game, team) with date, opponent and coach.
   - `review.csv`: low-confidence LLM answers and anything marked none despite
     candidates. The user fills an `accept` column; the next run reads it into
     the manifest (`decided_by = human`) and copies the file to S3 as
     `review_decisions.csv`.
   - `coach_changes.csv`: coach switches found in the data that the timeline
     did not expect, and `proposed_overrides.csv` for runs of 3 or more games
     (coach section).
   - `coach_timeline_issues.csv`: Wikipedia change sentences that could not be
     parsed and infobox coaches without a dated change; fix them with
     `coaches_overrides.csv`.
   - **8b. Gap filling (optional, after checkpoint 2).** For teams and seasons
     with low coverage, propose secondary channels: CLNS Media (Celtics),
     LetsGoWarriors (Warriors), and the NBC Sports Bay Area and Boston websites
     (test first whether yt-dlp can read them). Add them to `channels.csv` with
     a lower priority and repeat steps 4-8.
9. **Archive** (`archive.py`, skipped with `--dry-run`)
   - Only `selected` entries, newest games first. For each one:
     1. Extract the video again in a new session and download the audio-only
        track in the video's language (`select_audio`) to `tmp/`. Captions:
        `writeautomaticsub`, `subtitleslangs: ['en']`, `subtitlesformat:
        'vtt'`.
     2. Check it with `inspect_audio` and encode Opus 48 kbit/s mono (the recap
        profile, `encode_audio`). For a combined video with a known chapter or
        description timestamp (`0:00 Darko Rajakovic`), also encode the coach
        segment with `-ss`/`-to`; re-encoding makes the cut exact.
        Embed the chapters in both files (an FFmpeg metadata file as a second
        input, `-map_chapters 1`; the cut gets `coach_chapters`) and check
        them with `ffprobe -show_chapters` before upload. If the Ogg muxer
        drops them, upload without them: the manifest and the info JSON stay
        the record.
     3. Upload audio, coach cut, captions and the trimmed info JSON (no
        `formats`, `requested_formats`, `url` or `http_headers`; `chapters`
        and `description` kept) with SHA-256
        checksums and create-only puts. Object metadata: game ID, team, video
        ID, coach.
     4. Mark the entry `uploaded`, checkpoint the manifest, delete the local
        files (`--keep-local` keeps them).
   - `--captions-only` fetches the auto-captions without audio, as cheap
     transcripts. `--local-only` writes everything under `data/coach_pressers/`
     and never touches AWS; a later run without it uploads what is there.
   - yt-dlp options: `retries: 10`, `sleep_interval_requests: 1`, and the
     download pacing below. The manifest replaces yt-dlp's `download_archive`.
   - **Storage.** Opus 48 kbit/s mono is about 0.36 MB per minute: ~3 MB for an
     8-minute presser, ~11 MB for a 30-minute combined video. A recent season
     with ~2,400 pressers is about 6-10 GB; 2019-26 is about 40-50 GB, roughly
     $1 a month in S3 Standard in eu-west-1. Keeping the original AAC stream
     instead costs about 2.7× the space and makes cuts keyframe-bound; decide at
     checkpoint 3.
10. **Late trims** (`archive.py`)
    - When a combined video's coach segment is identified after its audio was
      archived (review or LLM), restore the audio from S3 (checksum verified),
      cut and encode the segment, and upload the `.coach.opus`.
    - With neither chapter nor timestamps, keep only the full file, leave
      `needs_trim = true`, and list it in `trim_manual.csv`.

## Mid-season coach changes

The coach is tracked per game, not per season. Each game gets its expected
coach from a dated timeline, and the videos are then used to check the timeline
and catch what it missed.

| Kind of change | Example | How it is caught |
| --- | --- | --- |
| Firing; an interim coach takes over | Nuggets fired Michael Malone days before the 2025 playoffs | Wikipedia's in-season coaching changes |
| Long absence; an acting coach for weeks or months | Mitch Johnson acting for the Spurs in 2024-25 | Wikipedia's in-season coaching changes, plus `coaches_overrides.csv` if missing |
| Short absence of 1-10 games: illness, suspension, health-and-safety protocols, family leave | Many acting coaches during the Dec 2021 to Jan 2022 protocol wave | Rarely listed anywhere: accept any staff name, then detect from the videos |

**How `coaches.py` builds and checks the timeline** (output: `coaches.json` per
season, with `spells`, `staff` and `players` per team):

1. **Staff from Wikipedia.** The team-season article ("2021–22 Sacramento
   Kings season") lists the season's head coaches in order in its infobox
   ("Luke Walton (fired)", "Alvin Gentry (interim)") and the head coach and
   assistants at season end in its roster template. Each name gets aliases
   (full name, joined initials, last name, "Coach \<last name>"). Any of them
   can be at the podium; assistants of the previous season's roster are added
   too. stats.nba.com `CommonTeamRoster` was the first plan, but for past
   seasons it returns a later staff (2021-22 Kings: Mike Brown's staff, hired
   in 2022), so it is not used.
2. **Player names.** Every player who appeared for the team that season, from
   the box scores in the database (`load_players_from_db`). The classifier
   needs them to tell a player video from a coach video. Without database
   access the run goes on with a warning and no player names.
3. **Dated changes from Wikipedia.** The league-season article ("2021–22 NBA
   season", MediaWiki `action=parse`) lists in-season changes as sentences
   ("On November 21, 2021, the Sacramento Kings fired head coach Luke Walton
   and named Alvin Gentry as their interim head coach."). Each parsed sentence
   gives the team, the date and the incoming coach; the outgoing spell ends the
   day before. A sentence without a team name ("Gregg Popovich suffered a mild
   stroke...") is placed through the coaches each team lists. Anything still
   unparsed goes to `coach_timeline_issues.csv`, never guessed. Some seasons
   (2022-23) have no such section; the infobox then shows whether a change
   needs an override.
4. **Manual overrides last.** `coaches_overrides.csv` (team, coach, start_date,
   end_date, kind, note) always wins over the first sources.
5. **Expected coach per game.** `coach_expected` = the spell covering the game
   date; with no change that season, simply the infobox's head coach.
6. **Accept, don't force.** The classifier and the LLM accept any staff name,
   not only `coach_expected`. Whoever actually speaks is stored as
   `coach_actual`.
7. **Detect what the timeline missed.** After matching, walk each team's games
   in date order and compare `coach_actual` with `coach_expected`. A run of 1-2
   games is a normal short absence: keep it and list it in
   `coach_changes.csv`. A run of 3 or more is probably a missing spell: write a
   proposed override line for the user to approve.
8. **Re-run after approval.** Approved lines go into `coaches_overrides.csv`
   and the season runs again. `coach_expected` is part of the LLM input hash,
   so only the affected games go to the LLM again.

The manifest and the S3 object metadata always say who actually spoke, so an
analysis per coach stays correct across a mid-season change.

## Title classifier rules and test fixtures

The classifier is rule-based and tested against real titles. A coach name plus
a postgame cue is the strongest signal; keywords alone are not enough.

**Rules, applied in order:**

1. Identify staff names, players and postgame cues before considering warning
   words. Unrelated titles and known player-only videos are not candidates.
2. **Coach** if a coach alias for that team-season is present, plus one cue:
   postgame, post game, post-game, presser, press conference, media
   availability, availability, postgame sound, postgame interview. A coach name
   with "Recaps" or "Reacts" counts too, at score 0.7.
3. **Combined** if a postgame cue is present with no person name before it, or
   with several names (`Williams, Paul, Ayton & Bridges`), or words like "Full
   Postgame Media", "ALL ... Postgame Media" or "Media Availability |
   Postgame".
4. **Player** if a postgame cue is present with a name from the team's player
   list. A person-like name that is neither staff nor player, with a postgame
   cue, is `player` with score at most 0.5 and the note `unknown_name`, so an
   acting coach missing from the staff still reaches step 7.
5. A staff name without a postgame cue is **other** with `requires_llm = true`:
   it still gets metadata and reaches the judge when matched to a game.
6. Former exclusion terms (pregame, shootaround, practice, mic'd up, locker
   room, speech, highlights, full game, game recap, game rewind, quarter
   references, shorts, previews, `Tonight's`, and other offseason terms) are
   warnings. For a plausible presser, retain it as **other** with
   `requires_llm = true` and `keyword:<term>` notes; they cannot veto it or
   allow automatic acceptance. Without presser evidence it remains filtered.
7. Very short or long plausible pressers also get `requires_llm = true` and
   `too_short` / `long` notes. Everything else is **other**. The judge decides
   the flagged cases; `--no-llm` or an exhausted budget sends them to review.

Minimum fixtures for `tests/ingestion/test_coach_pressers_classify.py`, all seen
on team channels during research:

| Title | Channel team | Expected class |
| --- | --- | --- |
| Hornets vs Cavaliers: Coach Charles Lee Postgame Media Availability | CHA | coach |
| Cavs at Hornets \| Kenny Atkinson Post Game \| 02.20.2026 | CLE | coach |
| Head Coach Jordan Ott Postgame Availability \| Phoenix Suns \| 2-26-26 | PHX | coach |
| Mitch Johnson Postgame vs Golden State Warriors \| 4.9.2025 | SAS | coach |
| Steve Kerr Recaps Warriors Loss Against San Antonio Spurs \| April 9, 2025 | GSW | coach (score 0.7) |
| Coach Malone Full Post Game Press Conference vs. Raptors | DEN | coach |
| Raptors Post Game: Darko Rajakovic - March 11, 2024 | TOR | coach |
| Chauncey Billups Press Conference \| Portland Trail Blazers 100, Memphis Grizzlies 112 \| Nov. 5, 2023 | POR | coach |
| Grizzlies vs Trail Blazers: Coach Taylor Jenkins press conference 11.05.23 | MEM | coach |
| Doc Rivers Post-Game Press Conference \| Clippers vs. Nuggets Game One | LAC | coach |
| Nuggets vs. Clippers Game 1: Michael Malone postgame interview | DEN | coach |
| Steve Clifford Postgame Sound vs. Golden State Warriors \| Orlando Magic | ORL | coach |
| Cavs vs. Bulls Postgame: J.B. Bickerstaff | CLE | coach |
| Suns vs Warriors Postgame Media Availability: Williams, Paul, Ayton & Bridges \| Suns | PHX | combined |
| Detroit Pistons vs. Atlanta Hawks \| Full Postgame Media Availability | DET | combined |
| Toronto Raptors Media Availability \| Postgame at Denver Nuggets \| March 11, 2024 | TOR | combined |
| ALL Denver Nuggets Postgame Media vs. UTA | DEN | combined |
| Hornets vs Cavaliers: Brandon Miller Postgame Media Availability | CHA | player |
| Royce O'Neale Postgame Availability \| Phoenix Suns \| 2-26-26 | PHX | player |
| Mitch Johnson Pregame vs Golden State Warriors \| 4.9.2025 | SAS | other |
| Toronto Raptors Media Availability \| Pregame at Denver Nuggets \| March 11, 2024 | TOR | other |
| Oso Ighodaro Shootaround Media Availability \| Phoenix Suns \| 2-26-26 | PHX | other |
| Coby White Practice Media Availability | CHA | other |
| CAVALIERS at HORNETS \| FULL GAME HIGHLIGHTS \| February 20, 2026 | CLE | other |

`tests/ingestion/test_coach_pressers_dates.py` parses each date written in
these titles to an ISO date. Two-digit years are 20xx; `11.05.23` is
month.day.year (US order). `test_coach_pressers_match.py` covers a back-to-back
where the first game's presser is published after midnight UTC on the day of
the second game, and a late upload outside the window that only the title
places.

## Milestones, checkpoints and acceptance criteria

Seven milestones and three human checkpoints. Nothing is downloaded until
checkpoint 3.

0. **M0: Shared YouTube helpers.** Step 0 above.
   - Done when: `recap_audio.py` uses `youtube_media.py` and the recap tests
     pass unchanged.
1. **M1: Expected pressers and coach timeline.** Steps 1-2 for all seasons.
   - Done when: game counts match the checks in step 1, every manifest entry
     has a `coach_expected`, and every in-season change on Wikipedia appears in
     the spells.
2. **M2: Channel registry.** Step 3.
   - **Checkpoint 1:** show the user the 30 resolved channels (handle, channel
     name, upload count). Wait for confirmation.
3. **M3: Pilot on two team-seasons.** `run --dry-run --local-only` for PHX
   2025-26 (expected high coverage) and BOS 2021-22 (expected low coverage).
   - Done when: the classifier, date, match and LLM tests pass, and a hand
     check of 30 matched games per pilot finds no wrong game and no wrong coach.
   - Report precision and recall separately for rule and LLM decisions, and
     how many entries fell to step 7. The LLM must agree with hand labels on at
     least 45 of 50 cases.
4. **M4: Dry run for all seasons.** `run 2019-20:<current> --dry-run`.
   - **Checkpoint 2:** send each season's `coverage.csv`, `missing.csv`,
     `review.csv` and `coach_changes.csv` with a short summary: coverage by
     season, the teams with the biggest gaps, LLM spend so far, proposed coach
     overrides. The user approves overrides and decides on 8b.
5. **M5: Archive, one season at a time.** `run <season>`, newest season first.
   - **Checkpoint 3:** before the first season, confirm the audio format (Opus
     48 kbit/s mono or the original stream), the S3 size estimate, and whether
     to fetch audio, captions or both.
   - Done per season when: every `selected` entry is `uploaded` or has a
     recorded error, the S3 inventory under the season prefix matches the
     manifest, and 20 random files restored from S3 have the right length and
     the right coach (listen to the first 20 seconds).
6. **M6: Trims and optional transcripts.** Step 10, then optionally Wishper
   transcripts in the recap transcript format under
   `train_data/coach_presser_transcripts/v1/profile=<profile>/`.
   - Done when: `trim_manual.csv` lists every combined video that could not be
     cut automatically.

## Operations

The main operational risk is YouTube blocking the run: go slowly, keep yt-dlp
current, and keep every step resumable.

- **Rate limits.** At least 1 s between metadata requests (`RequestPacer`),
  5-15 s (random) between downloads, no parallel downloads, a 10-minute pause
  every 200 downloads. Spread the full archive over several days.
- **"Sign in to confirm you're not a bot".** First bump yt-dlp and slow down. If
  it continues, pass the user's own browser cookies
  (`cookiesfrombrowser: ('firefox',)` or `('chrome',)`), with the user's
  permission. A PO-token provider plugin is the last resort. No proxies or VPNs.
- **HTTP 403 or format errors.** Bump yt-dlp and yt-dlp-ejs, retry once, then
  record the error on the entry (`fetch_error`) and move on. Five consecutive
  processing errors stop and checkpoint the run, as in the recap archive.
- **Removed or private videos.** Mark the entry `unavailable` with the reason;
  it counts as missing in the report. Once archived, S3 is the only copy: a
  video removed later cannot be fetched again.
- **Logging.** One log file per run in `data/coach_pressers/logs/`, a one-line
  summary per step (items processed, new, failed), and the final JSON summary
  on stdout like the recap CLI.
- **Incremental updates.** For the current season, a weekly `run <current>`
  picks up new games; cached listings, metadata and LLM answers mean only the
  new week costs anything. The manifest and create-only S3 writes stop anything
  from being fetched or stored twice.
- **Recovery.** Caches, `coaches.json`, `candidates.json`, the manifest and the
  review decisions are copied to S3 at the end of each run. On another machine
  the run restores them from S3 before doing any network work, as the recap
  transcript run restores its audio manifest.

## Out of scope

Excluded on purpose; do not attempt without a new decision from the user.

- **NBA.com video.** Restricted and its yt-dlp extractors are broken. Do not
  scrape its player, look for hidden stream URLs, or work around its region or
  login checks.
- **Geo-blocked or paywalled sources.** No VPNs, proxies or shared logins.
- **Redistribution.** Audio stays in the private bucket and on the user's
  machine; nothing is re-uploaded, published or shared by link.
- **Speaker diarization** (separating the coach from players in combined videos
  without chapters). Possible later with `pyannote`, not part of this plan.
- **Model features.** This is an archive; any feature use is a separate plan
  and must respect publication times.
- **Pressers before 2019-20**, pregame pressers, practice availabilities and
  player pressers.

Open question for the user: if coverage for 2019-2022 stays low after gap
filling, should the written postgame quotes on team websites be collected as a
text-only fallback?
