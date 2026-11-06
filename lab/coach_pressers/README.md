# Coach postgame presser archive

Finds the head coach's postgame press conference for both teams of every NBA
game from 2019-20 on, on the official team YouTube channels, and archives its
audio (Opus 48 kbit/s mono, with the video's chapters), English auto-captions
and a trimmed info JSON in S3. Design, decisions and data layout:
[docs/pipeline/coach_pressers.md](../../docs/pipeline/coach_pressers.md).

Personal research use only. Run it on your own computer (home internet), keep
the bucket private, and never share or re-upload the audio.

## Setup

```bash
poetry install --with media        # yt-dlp, yt-dlp-ejs, openai
```

System tools: FFmpeg/FFprobe with `libopus`, and Node >= 22 (or Deno with
`--js-runtime deno`) for YouTube's JavaScript challenges. The archive uses the
configured S3 bucket and AWS profile (`[S3]` in `src/nba_ou/config.ini`);
`--local-only` never touches AWS. Player names come from the box scores in the
database; without database access the run continues with a warning.

The LLM tie-break needs an OpenAI API key, either as an environment variable
or in `src/nba_ou/config.secrets.ini`:

```ini
[OpenAI]
OPENAI_API_KEY = sk-...
```

Without a key, run with `--no-llm`: ambiguous cases go to `review.csv`.

## Run, in checkpoint order

```bash
# M1: expected pressers and coach timeline (stops after step 2 until
# checkpoint 1; Wikipedia and stats.nba.com only).
.venv/bin/python lab/coach_pressers/get_pressers.py run 2019-20:2025-26 --dry-run

# M2 / checkpoint 1: resolve the 30 team channels, check every row, then set
# verified_by_human=true in lab/coach_pressers/channels.csv.
.venv/bin/python lab/coach_pressers/get_pressers.py channels --resolve

# Quick LLM check on 20 labelled cases (tokens and cost are printed).
.venv/bin/python lab/coach_pressers/get_pressers.py llm-check

# M3: pilot, metadata only, one team-season each.
.venv/bin/python lab/coach_pressers/get_pressers.py run 2025-26 --team PHX --dry-run
.venv/bin/python lab/coach_pressers/get_pressers.py run 2021-22 --team BOS --dry-run

# M4 / checkpoint 2: every season, metadata only; read the reports.
.venv/bin/python lab/coach_pressers/get_pressers.py run 2019-20:2025-26 --dry-run

# M5 / checkpoint 3: archive one season at a time, newest first.
.venv/bin/python lab/coach_pressers/get_pressers.py run 2025-26

# Coverage from the local manifests; no network.
.venv/bin/python lab/coach_pressers/get_pressers.py status 2019-20:2025-26
```

Other options: `--team` (repeatable), `--until-step N`, `--phases`,
`--llm-model`, `--llm-budget` (USD per run, default 5), `--captions-only`,
`--no-captions`, `--keep-local`, `--local-root`, `--verbose`. Every run is
resumable: re-running a season skips what its manifest and caches already hold.
A run that ends with `--local-only` can be promoted later by the same command
without the flag; the saved files are uploaded without downloading again.

## Files

Local state is under `data/coach_pressers/` (gitignored): raw schedule,
Wikipedia and player caches, channel listings and video metadata, the LLM
decision cache, and per season `manifest.json`, `coaches.json`,
`candidates.json` and `reports/`. A real (non-dry) run restores this state
from `s3://<bucket>/train_data/coach_pressers/v1/` first and publishes it back
at the end, so another machine can continue.

## Reports (per season, `reports/`)

| File | Use |
| --- | --- |
| `coverage.csv` | % found per team and phase; rules / LLM / human; archived; % with chapters |
| `missing.csv` | every expected presser without a video |
| `review.csv` | one row per candidate of each undecided presser: write `yes` in `accept` on the right video (optionally fix `kind`, `coach_actual`, `coach_chapter`) or `none`; the next run applies it |
| `review_decisions.csv` | every human decision applied (kept in S3) |
| `coach_changes.csv`, `proposed_overrides.csv` | speakers that differ from the timeline; runs of 3+ games as override lines to approve into `coaches_overrides.csv` |
| `coach_timeline_issues.csv` | Wikipedia sentences that could not be parsed, infobox coaches without a date |
| `trim_manual.csv` | combined videos without a coach chapter |

## Tests

```bash
.venv/bin/python -m pytest -q tests/ingestion/test_coach_pressers_*.py \
  tests/ingestion/test_youtube_media.py
```

They fake YouTube, Wikipedia, OpenAI and S3; only the encoding tests call the
local FFmpeg (skipped without it).
