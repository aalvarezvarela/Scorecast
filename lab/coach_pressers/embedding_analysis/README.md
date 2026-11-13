# Coach presser embeddings against the game statistics

Probe: what do Qwen3-Embedding-8B vectors of the coach postgame pressers
(YouTube automatic captions) contain, and how do their principal components
relate to the statistics of the same game? Same question as the report
embeddings ([`lab/text_to_features/README.md`](../../text_to_features/README.md)),
on a sample.

## Views

Each presser is embedded three times (`presser_inputs.VIEWS`):

| View | Text | Instruction |
| --- | --- | --- |
| `raw` | captions as they are | none |
| `instructed` | captions as they are | "Represent this NBA head coach's postgame press conference: how his team and the opponent played (result, offense, defense, effort, pace) and which players were hurt or unavailable." |
| `target_view` | coach's team → `Target Team`, the other → `Opponent Team`, player notes | the same, "from the Target Team's point of view", with the labels |

`raw` → `instructed` isolates the instruction, `instructed` → `target_view`
the rewrite. One rewrite per presser: the coach belongs to the Target Team.
Differences from the report rewrite:

- The opening sentence gives the sides, not the franchises ("In this text,
  the head coach of the Target Team (home team) answers reporters after the
  game against the Opponent Team (visiting team).").
- A player gets his note at the first mention and again after 80 words, not
  once per sentence: captions before 2024-25 have no punctuation.

Captions: the last line of every real VTT cue; sound tags dropped; a speaker
change (`>>`) starts a new line. A combined video is cut to the coach's part
with the manifest's `trim_start_s` / `trim_end_s`; combined videos without it
are left out, as are parts under 60 s or over 40 min and texts under 150 words.

## Run

```bash
.venv/bin/python -m lab.coach_pressers.embedding_analysis.run_embeddings --dry-run  # tokens, cost
.venv/bin/python -m lab.coach_pressers.embedding_analysis.run_embeddings
```

Needs S3 (presser manifests and captions), the database (players, box scores),
`[Nebius] NEBIUS_API_KEY` and a `closing_line_data_2_5_*.parquet` for the
market labels. Output in `artifacts/coach_pressers/embedding_analysis/`
(`sample.parquet`, `texts.parquet`, `labels.parquet`, `vectors/<view>.npy`,
`run.json`); a step whose file exists is skipped. `--pressers-per-season`
(default 400) and `--seed` set the sample: whole games in random order.

[`presser_embeddings.ipynb`](presser_embeddings.ipynb) reads those files (no
API, no database) and draws everything below.

| File | Content |
| --- | --- |
| `captions.py` | VTT → text |
| `presser_inputs.py` | the three views, the rewrite |
| `labels.py` | same-game statistics seen from the coach's team |
| `run_embeddings.py` | sample, texts, labels, vectors |
| `analysis.py` | PCA, correlations, out-of-fold ridge (used by the notebooks) |
| `run_variants.py` | chunks, instructions, masked text, anchors, report vectors |
| `anchors.py` | anchor statements of the named axes |
| `intangibles.py` | controls, residuals, agreement measures, anchor scores |

## Normalize before embedding (OpenAI)

`normalize.py` rewrites automatic transcripts into reports with a fixed order
of topics. It defaults to `gpt-6-luna`, `reasoning.effort=low`, and strict
Structured Outputs on `/v1/responses`. All schema fields are required and
nullable; null topics are omitted from the rendered text. No player lists
are sent. The key is `SETTINGS.openai_api_key`, from `[OpenAI] OPENAI_API_KEY`
in `config.secrets.ini` or the environment.

The Batch workflow is split into explicit commands; none has been run as
part of implementing this change:

```bash
# Local files only; --all uses all 1,200 sampled pressers (default: --n 4).
.venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode prepare --all
# Upload and submit; this starts paid processing.
.venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode submit
.venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode status
# Download available results once the batch is terminal, including partial results.
.venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode collect
# Alternatively, poll every minute and collect automatically (up to 25 hours).
.venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode watch
```

Batch offers [50% lower token costs, with a 24-hour processing window](https://developers.openai.com/api/docs/guides/batch).
The default mode is `prepare`, which does not construct an API client. Local
requests and the manifest live in `normalized/gpt-6-luna/batch/`. Choose an
empty `--batch-dir` for another batch and use it for all four commands.
Repeated submission of a directory with a saved batch ID does not resubmit it.

Collection matches `custom_id` to the presser, checks completed responses and
refusals, validates field names/types, and writes `.model.json`, `.json`,
`.txt` and `.meta.json` per report under `normalized/<model>/`. Metadata holds
token usage, the request hash and rule warnings. Raw output/error JSONL files
and `collection.json` retain failures and warnings. Structured Outputs enforce
the JSON shape; word counts and numerical-detail rules are checked locally.
Batch does not automatically retry failures or repair rule warnings.

`--mode sync --n 4` is also available, with one correction attempt for local
rule violations. Normalization does not generate embeddings; the existing
embedding experiments remain separate.

Cost planning for the 1,200 pressers (2026-11-13; estimates, no requests run):
the original texts used 1.55M Qwen tokens. Allow roughly 3.2M OpenAI input
tokens including the repeated prompt and schema; this is a planning proxy,
not an exact OpenAI token count. Assuming 500 JSON output tokens per report,
and 1,000 reasoning tokens per report, Luna Batch costs approximately
`3.2 * $0.05 + (0.6 + 1.2) * $0.25 = $0.61`.
[Luna rates](https://developers.openai.com/api/docs/models/gpt-6-luna)
are $0.10/M input and $0.50/M output before the 50% Batch discount;
[reasoning tokens are billed as output](https://developers.openai.com/api/docs/guides/reasoning).
No cache savings are assumed.

At [Nebius's published Qwen3-Embedding-8B rate](https://tokenfactory.nebius.com/api/public/models_info)
of $0.01/M input tokens, embedding the reports once adds roughly $0.005
(assuming 0.5M tokens). Total: about $0.62 with the reasoning assumption above,
or $0.92 at 2,000 reasoning tokens per report. Using the configured 4,096
output-token cap in every response gives about $1.40 for one Batch pass plus
embeddings at the estimated input volume. This cap includes reasoning and
visible output; repeated runs/repairs and taxes are not included. Synchronous
normalization at the 1,000-reasoning-token assumption is about $1.22 plus
embeddings, before any correction attempts.

## Result (2026-11-13)

1,200 pressers of 798 games (402 games with both coaches), 2023-24 to
2025-26, median 1,091 words. 4.8M tokens, $0.05, about one minute per view.
One sample, one fold split; same-game content only, nothing here tests a
prediction.

- **PC1 is the result, in every view** (6% of the variance): r with
  `target_won` 0.60 (`raw`), 0.71 (`instructed`), 0.71 (`target_view`);
  transcript, season and game statistics explain 43% / 58% / 59% of it out of
  fold (the first two alone 4–6%). It follows the
  margin against the closing spread as closely as the margin itself (0.63).
- **After PC1 the components are not box score.** From PC2 on, the
  statistics add at most 14 points of R2, usually under 5. Several follow
  the transcript: length (r up to 0.45 on `target_view` PC3) and the season/punctuation of the captions.
- **Little of the whole vector is explained**: transcript properties, season,
  statistics, team, coach and opponent together reach 10–11% of the variance
  (statistics about 4 points, team about 4). In the report vectors the box
  score explained 13% and team identity 34% more: a presser vector is far
  less "which teams played".
- **The instruction sharpens PC1 and little else.** Probes from the full
  vector read the same from the three views, within the intervals:
  `target_won` AUC 0.975–0.980, margin r 0.72–0.73, eFG% 0.50–0.51, the team's
  earlier win rate 0.60–0.61, regulars out 0.33–0.34 (own team) and 0.14–0.18
  (opponent).
- **The rewrite adds the side**: `target_is_home` AUC 0.69 → 0.97, which is
  the opening sentence. A coach names the teams less than once per presser
  ("we", "they"), so there is little else to rewrite: about five player notes.
- The views stay close: cosine 0.93 `raw`/`instructed`, 0.91
  `instructed`/`target_view`, 0.85 `raw`/`target_view`.

## Beyond the known numbers (2026-11-13)

Second question: after removing what already exists as numbers (transcript
length and season, box score, regulars out, record, closing line), is anything
repeatable left, and can an instruction point the vector at intangibles
(style, effort, mood)? [`presser_intangibles.ipynb`](presser_intangibles.ipynb),
vectors from `run_variants.py` (13.8M tokens, $0.14, same 1,200 pressers):

```bash
.venv/bin/python -m lab.coach_pressers.embedding_analysis.run_variants [--dry-run]
```

- every presser cut into chunks of about 220 words (median 5), each embedded
  under five instructions (`run_variants.INSTRUCTIONS`: `none`, `basketball`,
  `intangibles`, `style`, `mood`), and a `masked` text (teams and recognized
  players replaced by "our team", "their player") under `intangibles`;
- 64 anchor statements for eight named axes (`anchors.py`);
- the stored NBA.com report vector (`target_view`) of the same game and team.

There is no label for intangibles, so what is left is judged by agreement
(`intangibles.py`), as an AUC (0.5 = nothing): `halves`, odd against even
chunks of one presser, and `game`, the two coaches of the same game (402
games), after removing the controls by least squares.

- **Length is the amount of text embedded, not content.** It can be read from
  one vector of the whole text (r 0.95) and from the mean of all chunks (0.85),
  but not from the mean of the first two chunks (0.34, about what length
  really has to do with the game: winners talk longer, r 0.24). The result is
  read as well from two chunks (0.79 against 0.83). In variance it is small
  (transcript properties explain 2% of any representation).
- **Instructions about intangibles point the vector at the result.** The
  share the known numbers explain goes up, not down: 3.8% (`none`), 5.9%
  (`intangibles`, whole text), 7.8% (`intangibles`, chunks), 20.9% (`mood`,
  where PC1 alone is 34% of the variance). Effort, focus and mood are how a
  coach describes winning and losing.
- **The anchors are one axis.** Effort, physicality, ball movement, defensive
  focus, composure and satisfaction correlate 0.94–0.98 with each other, 0.68–
  0.70 with winning and 0.66–0.68 with the margin against the spread; one
  component holds 81% of the eight after removing the known numbers. What is
  left is repeatable inside a presser (split-half r 0.74) but the two coaches
  do not share it (r −0.04 to +0.04): it is the coach's tone, not the game.
  Only tempo stands apart, weakly (split-half r 0.39).
- **The two coaches do agree beyond the numbers**: `game` AUC 0.70 [0.67,
  0.73] (`none`) and 0.71 [0.69, 0.74] (`basketball`) after removing the known
  numbers, 0.65 after team, opponent and coach too. Every intangible-oriented
  instruction lowers it (0.69, 0.63 with `style`, 0.57 with `mood`). Masking
  names costs it only through identity (0.63 with or without the names once
  identity is removed).
- **That agreement is already in the news.** Removing the report vector of
  the game as well takes the `game` AUC to 0.42–0.45, below chance (what is
  left of two coaches of one game is opposite, not shared). The presser and
  the report of the same game recognize each other after every control (AUC
  0.85 [0.83, 0.86]).
- **Against the report vectors**: a report is 9% known numbers and 36%
  identity; a presser 4–8% and 2–4%. The report reads almost every statistic
  better (won r 0.98 against 0.82, total points 0.66 against 0.33, pace 0.53
  against 0.28); the presser matches it on the margin against the spread (0.71
  / 0.70) and is ahead only on turnovers (0.33 against 0.24). Each adds little
  to the other beyond the known part (+2 points of presser variance from the
  report, +0.5 the other way).

Reading: what a presser vector holds about the game is the result relative to
expectation, and the game's story, which the report has in more detail. No
instruction, masking or anchor tried here isolates a team trait or a game
quality that two observers share and the numbers and news lack. Team identity
(a persistent style would live there) is 2–4% of the variance.

Not done: the full archive (4,210 usable captioned pressers), Wishper
transcripts instead of captions, removing transcript length and season before
the PCA, any forward test.
