# NBA.com written-report archive

`download_season_reports.py` archives **one season at a time**,
named on the command line as a starting year (`2025`) or a label (`2025-26`).
From the repository root:

```bash
.venv/bin/python lab/game_reports/download_season_reports.py download 2025-26
.venv/bin/python lab/game_reports/download_season_reports.py status 2025
```

1. `download` fetches the official season schedule and starts downloading its
   available reports immediately. It uses the existing repository S3 bucket, AWS
   region, and profile settings. It exits 1 if any game ends in a fetch or
   upload error.
2. Re-run `download` to resume failed or missing work. `status` prints a
   local-only coverage summary; `manifest.json` has the details.
3. Select the next season explicitly; the runner never loops over seasons.

Options: `--retry-unavailable`, `--max-games N` (pilot run), `--workers`,
`--request-interval`, `--phases`, `--local-root`, `--s3-prefix`; see `--help`.

## Storage

One compressed JSON per game preserves the complete extracted article text,
title, byline, agency, original publication timestamp, source URL, scores,
retrieval time, official game ID, and season/phase metadata. The redundant
`paragraphs` array from the daily prototype is omitted: `text` already preserves
paragraph breaks.

`published_at` is NBA.com's story date verbatim: a US Eastern wall-clock time
(`America/New_York`, daylight saving included) with no offset. `published_at_utc`
is the same instant in UTC. Reports archived before 2026-10-31 do not carry that
field; read them through `season_reports.load_report()`, which fills it in. The
story date may be the last edit rather than the first publication: a few
reports are dated more than a day after the game.

Measured on 2025-26 (all 1,315 reports against the official tip-off; exact end
from the page's game duration on a 120-game sample), the story date falls a
median of about 40 minutes after the final buzzer (18 minutes at the earliest in
the sample). About 71% fall within an hour of the end and 87% within two hours,
but the mean is about 1 h 45 min because of a long tail (0.6% more than a day
later). The reports are postgame by construction: a feature built from one can
only describe games played after its `published_at_utc`. No TXT, CSV, HTML, or
video copies are created for the season archive.

```text
data/nba_game_reports/seasons/season=2025-26/
  reports/0022500001.json.gz
  ...
  manifest.json

s3://<configured-bucket>/train_data/game_reports/v1/season=2025-26/
  reports/0022500001.json.gz
  ...
  manifest.json
```

The authoritative NBA `ScheduleLeagueV2` feed supplies the expected game IDs.
Default coverage includes regular season, play-in, playoffs, and the Cup final;
it excludes preseason and All-Star exhibitions. `PHASES` can select another
scope, but an existing manifest refuses a scope change in place. Use a separate
prefix/local root for an alternate scope. A partial/current season retains
unplayed games as `not_final` and refreshes their state from the next schedule.

Source HTML pages are discarded after extracting the article. The remote
manifest is published only at the end of a run or a handled interruption,
avoiding a full new S3 manifest version for every game.

The initial 2025–26 execution archived 1,315 reports and recorded seven unavailable
source reports: three game URLs resolve to the games index, and four pages only
serve pregame previews. All 1,315 S3 reports were read back and checked against
their manifest hashes and local bytes. Current objects occupied 2,259,637 bytes
(approximately 2.26 MB), including the manifest; retained older versions are
outside that measurement.

## Coverage and recovery

The manifest inventories **all** expected games, including failures, rather than
listing only successful downloads. Each entry has its date, phase, source URL,
S3 key, attempt count, last attempt time, status, error, compressed byte size,
and SHA-256. The summary lists every game not yet uploaded.

| Status | Meaning |
| --- | --- |
| `pending` | Not fetched yet. |
| `downloaded` | Local JSON exists; an S3 upload is still needed. |
| `uploaded` | S3 accepted the report, or existing S3 progress was recovered. |
| `not_final` | The official schedule does not mark the game final. |
| `not_available` | The game page is missing, has no written recap, or only serves a pregame preview. |
| `fetch_error` | Request, parsing, or local-integrity failure; see `error`. |
| `upload_error` | Local JSON was saved, but S3 upload failed. |

Failed downloads/uploads are retried on the next run. Pass
`--retry-unavailable` to also recheck source articles previously absent.
Existing daily-prototype JSON files are reused when the game/date match. A local
season report that is corrupt or no longer matches its manifest hash is renamed
to `*.json.gz.corrupt` and fetched again.

Local checkpoints are written atomically every ten processed games and on errors
or shutdown. Each report file is also written atomically before upload. S3 puts
validate a SHA-256 checksum and use create-only conditional writes; different
content at the same report key is never silently replaced. A resume reconciles
the remote manifest and object inventory, recovers uploads made before a lost
checkpoint, and reuploads missing S3 objects from local files when possible.
The manifest uses its S3 ETag for conditional updates, and a local lock prevents
two download runs from processing the same season simultaneously.

`--max-games` bounds a pilot run without shrinking the expected-game inventory.
After twelve consecutive processing errors, the downloader stops and checkpoints
so that a changed site layout or lost credentials do not trigger an entire
season of failing requests. If the S3 manifest cannot be published after such a
stop, the original error is still raised; the local checkpoint is saved and the
next run reconciles S3.

To read a stored report:

```python
from lab.game_reports.season_reports import load_report

response = s3.get_object(Bucket=bucket, Key=report_key)
report = load_report(response["Body"].read())
article_text, published = report["text"], report["published_at_utc"]
```

Checks:

```bash
.venv/bin/pytest -q tests/ingestion/test_game_reports.py \
  tests/ingestion/test_season_reports.py tests/ingestion/test_download_season_reports.py
```
