# Game-report embeddings

Status: Implemented — every archived NBA.com report (2017-18 on) is embedded
and stored in S3. No feature reads the vectors yet; the daily job does not run
it yet. Two research campaigns found no value as line_error features (see
"Feature experiments"), so the feature work is closed.

The reasons for each choice (model, instruction, team rewrites) are in
[`lab/text_to_features/README.md`](../../lab/text_to_features/README.md).
The report archive is described in
[`lab/game_reports/README.md`](../../lab/game_reports/README.md).

## What is embedded

Three inputs per report (`embedding_inputs.VIEWS`), embedded with
Qwen3-Embedding-8B through the Nebius Token Factory API (4096 dimensions):

| `view` | `side` | `team_id` | Input |
| --- | --- | --- | --- |
| `raw` | — | — | Title and body without AP dateline, footers and URL lines; no instruction. The control: the raw text does not say which team it is about. |
| `target_view` | `home` | home team | The `home_target` rewrite (intro sentence naming both teams, team names → `Target Team` / `Opponent Team`, player notes with `label_rule="any"`) under the instruction "Represent this NBA game recap from the Target Team's point of view." |
| `target_view` | `away` | away team | The `away_target` rewrite, same instruction. |

The player notes need each game's players (box score and inactive list),
which are read from the database. A team's vectors are the `target_view` rows
with its `team_id`, whichever side it was on.

## Code

`src/nba_ou/data_processing/game_texts/`:

| Module | Role |
| --- | --- |
| `team_perspective.py`, `players.py` | The rewrites (moved from `lab/text_to_features/`, unchanged). |
| `documents.py` | Reads the archived reports from S3, and the games' players from the database. Fills `published_at_utc` for reports archived before 2026-10-31. |
| `embedding_inputs.py` | The exact model input of each view, and its sha256. |
| `nebius.py` | API client: batching, retries, token check against the bill, refuses inputs over 32k tokens and responses float32 cannot hold exactly. |
| `vector_codec.py` | Lossless 16-bit form of a vector (below). |
| `embedding_store.py` | S3 store, one Parquet file per season plus a manifest. |
| `embed_reports.py` | `embed_documents(reports, embedder)` (rows and vectors of given reports: the unit for a daily job) and `update_season` (incremental, resumable). |

CLI, from the repository root:

```bash
python scripts/text_embeddings/embed_game_reports.py status            # S3 only
python scripts/text_embeddings/embed_game_reports.py update --dry-run  # tokens and cost, no API
python scripts/text_embeddings/embed_game_reports.py update [--season 2025-26 ...]
python scripts/text_embeddings/embed_game_reports.py verify
```

Needs `[Nebius] NEBIUS_API_KEY` in `config.secrets.ini`, S3 and the database.

## Incremental updates

`update_season` rebuilds every input of the season and compares its
`input_sha256` with the stored rows (key `game_id`, `view`, `side`). It embeds
only inputs that are new or whose text changed: an edited report, players
corrected in the database, or a change to the rewrite rules. It writes after
every chunk of 1,200 inputs, so a stopped run resumes where it stopped. A
changed row keeps its old vector until the replacement is written. Rows of
reports no longer in the archive are kept and counted as `orphans`.

## Storage

```text
s3://<bucket>/train_data/text_embeddings/v1/source=report/model=qwen3-embedding-8b/
  season=2025-26.parquet
  ...
  manifest.json
```

The stored vectors are the API's values, bit for bit (not renormalized; the
norm the API returned is in `api_norm`, 1 to within float32 rounding).

- Every value Nebius returns is exactly a float32. This is checked on every
  response, and a response that fails the check raises.
- Each vector is, bit for bit, `float32(q / d)`: `q` is a bfloat16 vector (the
  model runs in bfloat16) and `d` is one float32 divisor (the L2 norm). A
  vector has about 850 distinct values out of 4096. `vector_codec.encode`
  finds `q` and `d` and returns them only when decoding reproduces every bit.
  The row is then stored as `vector_bf16` (uint16 codes) and `vector_divisor`,
  which is half the size of float32. Otherwise it is stored as `vector_f32`,
  and `encoding` says which form a row uses. Every vector seen so far took the
  compact form.
- Parquet with zstd (lossless). Before uploading, a write decodes what it
  serialized and refuses to upload anything that differs by a bit. Every read
  checks the file's sha256 against the manifest.
- The manifest records the model, endpoint, tokenizer revision, instructions,
  rewrite options and vector dimension. A write made with different settings
  is refused, so vectors of different inputs are never mixed. Use another
  prefix for another model or instruction.

Row columns: `game_id`, `season`, `game_date`, `view`, `side`, `team_id`,
`published_at` (NBA.com's Eastern story time, verbatim), `published_at_utc`,
`report_sha256`, `input_sha256`, `n_players`, `n_tokens`, `api_norm`,
`embedded_at_utc`.

Reading: `EmbeddingStore(s3_client, bucket).read_season("2025-26")` returns an
`EmbeddingTable` (`rows` DataFrame, `vectors` float32 array). Normalize the
vectors before using them as features.

## Measurements

- **Parity with the role probes.** The new code reproduces exactly the
  rewrites the 8B probes embedded (250 of 250 reports). Re-embedding 12 of them
  (36 inputs) gives cosine ≥ 0.9998 with the probe vectors, and 69% are
  bit-identical. Nebius is not bitwise deterministic: the same input can come
  back slightly different, probably because of how bf16 requests are batched.
  A re-embedded vector is therefore not an exact check of a stored one.
- **Full run, 2026-11-03.** 11,646 reports (2017-18 to 2026-27, 107 of them
  2026-27), 34,938 vectors. Billed: 32.5M tokens, $0.33 at $0.01 per million;
  the bill matched the local token count on every request. Reports before
  2024-25 are about twice as long as later ones (≈1,100 against ≈530 tokens per
  input). About 5–8 minutes of API time per season. With 8 workers Nebius
  answers some requests with 429 (token quota); the one-second retry succeeds.
  Every game had its players in the database.
- **Size.** All 34,938 vectors took the bf16 + divisor form; none needed
  float32. The store holds 225 MB, about 6.4 KB per vector against 16 KB as raw
  float32. `verify` passes on every season.

## Feature experiments

Two campaigns tested the vectors as line_error features, with research
columns built outside the schema layers (each research Parquet has a
`.provenance.json`). Per team, they used the newest reports published strictly
before the snapshot (`published_at_utc`). The vectors were reduced by PCA, or
by PCA of a residual from which the game's box score, both franchises and the
season running means were removed first.

- [`report_text_last_game_2026_11`](../../experiments/report_text_last_game_2026_11/README.md):
  the last game's report at the close, T-480 and T-720, four reductions, at the
  production hyperparameters. No arm passed. The 64-dimension arms were worse
  in CV, and the 16-dimension residual arms were neutral.
- [`report_text_optuna_2026_11`](../../experiments/report_text_optuna_2026_11/README.md):
  the last game plus the mean of the last 5 reports (residual, 16 dimensions
  each), at T-480 and T-720, with every arm re-tuned by Optuna. In the holdout
  the text arm was worse than both controls at both horizons. Two tunings of
  the same control differed by 0.07–0.09 MAE.

Earlier probes found that a previous-game vector does not predict the next
game's LINE_ERROR or SPREAD_ERROR, at any reduction size.

## Not done

- Features: the campaigns above found no value, so no schema layer was built.
- A daily job: the report download is not scheduled either.
- Recap transcripts (`documents.SOURCES["transcript"]`): the inputs work for
  them, but the store and CLI are for reports only.
