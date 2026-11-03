# Text to features

Exploration code for turning NBA news and game texts into model features,
including embeddings. Currently, it loads archived reports and recap transcripts,
cleans the text, and prepares team and player perspectives for a later text model.

## Conclusions and recommendation

Read this first; the sections after it describe the rewrites, the method and
the evidence. Status 2026-11-03: decodability probes on Qwen3-Embedding 0.6B,
4B and 8B are done. **No walk-forward ablation yet**, so nothing here shows
that a text feature improves a prediction.

**The question.** Can the embedding of a game text (NBA.com report or recap
transcript) become a feature of one team, i.e. does the vector know which team
won, scored, shot well or was missing regulars, and can a task instruction
point it at one of the two teams?

**What was tested** (details in "Role probes" below). 500 documents, each
rewritten twice with the team names replaced by `Target Team` / `Opponent
Team` (one rewrite per side), embedded under no instruction, a neutral
instruction of similar length (control) and five role instructions. Linear
probes, out of fold and grouped by game, predict box-score labels seen from
the Target Team. The metric is the **pair accuracy**: within a game, the
rewrite whose Target Team has the higher label gets the higher prediction
(50% = blind to the roles). Two more checks: synthetic minimal pairs, and a
transfer test (probes fitted on `target_view` applied to `opponent_view`).

| Check | 0.6B (local) | 4B (local, NF4) | 8B (Nebius API) |
| --- | ---: | ---: | ---: |
| who won, no instruction | 0.916 | 0.972 | 0.988 |
| who won, `target_view` | 0.964 | 1.000 | 1.000 |
| regulars out by side (mean of both sides), `neutral` | 0.539 | 0.577 | 0.556 |
| regulars out by side, `target_view` | 0.541 | 0.585 | **0.614** |
| `target_view` − `neutral` on it, pp [95% CI] | +0.2 [−3.7, +4.4] | +0.9 [−2.9, +4.3] | **+5.8 [+2.2, +9.4]** |
| minimal pairs in context, `target_view` | fail | 0.77 | 0.82 |
| `opponent_view` flips the `target_view` probes | — | pending | 99% of games |

What follows from it:

1. **The rewrite is what makes a vector team-specific.** The raw text has real
   names and no dateline, so a probe cannot tell which side it is about
   (56% = the home win rate). From the rewrites, even without an instruction,
   every model reads who won (92–99%).
2. **Result and points are easy; they are also in the box score.** 4B and 8B
   are at 98–100% with or without an instruction, so instructions cannot
   help there and these numbers do not separate the models.
3. **Availability by side is the part that matters** (the box score does not
   say who was missing) **and only 8B with `target_view` (or
   `target_focus`) gains on it.** The availability labels of the two sides are
   mirror images, so an instruction can raise the Target score just by putting
   "missing players" on the Target Team; it then loses the same amount on the
   opponent. That is what 0.6B and 4B do (Target up 6–11 pp, opponent down).
   The fair measure is the mean of both sides, and only 8B improves it.
   It is still modest (61% against 99% for the result), and the label is our
   rule (24+ mpg regulars listed without minutes), not what the text says.
4. **The instruction really sets the team, not just a prefix (8B).** A probe
   fitted on `target_view` reads `opponent_view` vectors backwards in 99% of
   games, and `opponent_view` of one rewrite is closer to `target_view` of the
   other rewrite than to its own text in 98.8% of documents.
5. **Minimal pairs are a weak tool.** They fail everywhere with 0.6B and pass
   with the Target-oriented instructions on 4B/8B; use them as a sanity check,
   not to choose.

**Recommendation.**

- **Model: Qwen3-Embedding-8B through Nebius** ([`nebius.py`](nebius.py)): the
  only one that gains on availability by side, and fast and cheap (the whole
  7-condition sweep, about 9,300 texts: 8 minutes, 5.08M tokens, about $0.05). 4B
  on the local GPU is the fallback if the API is not acceptable in serving
  (about 2 s per text). Whichever is chosen must be used for training and
  serving alike: the vector spaces differ (4096 vs 2560 dimensions) and cannot
  be mixed.
- **Instruction: `target_view`** ("Represent this NBA game recap from the
  Target Team's point of view."): the shortest, the best on availability,
  within 1 pp of the best on result and points, and its mirror behaves as one.
- **Two embeddings per text, one per team.** Rewrite each text with each team
  as the Target Team (`document_perspectives(..., label_rule="any")`: the
  `home_target` and `away_target` versions) and embed both with `target_view`.
  Store them keyed by `(game_id, source, target team_id)`. For a team's
  features, take the vectors of its previous games in which **it** was the
  Target Team, selected by `team_id`, never by whether it was home or away in
  that game. The opponent's features come from the opponent's own vectors.
- **Do not embed `opponent_view` or the raw text for features.** `opponent_view`
  of one rewrite ≈ `target_view` of the other, so it adds nothing; the raw text
  does not carry the side.
- **Into XGBoost** (not done yet): keep the full vectors; reduce inside each
  training window only (PCA on the residual after the box score, or Matryoshka
  truncation `v[:k]` renormalized, done after embedding); feed lags or EWMAs
  of the team's previous games as `_BEFORE` columns, in a new schema layer;
  judge it by a walk-forward ablation (start at T-720, where fundamentals still
  matter) following the `experiments` skill.

Open: 4B's `opponent_view` transfer (running); one sample and one seed; the
probe CIs do not correct for the number of comparisons.

## Team-perspective game texts

Turns one game text into three, so that a later text model reads it from either
side of the game.

| Version | Content |
| --- | --- |
| `raw` | The stored title and text, without AP datelines and footers. |
| `home_target` | Home team → `Target Team`, away team → `Opponent Team`. |
| `away_target` | Away team → `Target Team`, home team → `Opponent Team`. |

Each version is the title, a blank line and the text; both carry the same
replacement. A rewrite opens with a sentence naming what each label stands
for:

```text
In this text, Target Team is the Minnesota Timberwolves (visiting team) and Opponent Team is the Los Angeles Lakers (home team).

Doncic scores 38 as Opponent Team beat Target Team 118-110

Luka Doncic scored 38 points and the Opponent Team beat the Target Team ...
```

The sentence is added after the replacement, so it keeps the real names.
`name_teams=False` leaves only the sides ("Target Team is the home team and
Opponent Team is the visiting team"); `intro=False` drops it. "Home" is the
designated home team, also at neutral sites.

It applies to both archived text sources, which share the same `text`, `home`
and `away` fields: NBA.com written reports
([`lab/game_reports/`](../game_reports/README.md)) and recap-video transcripts
([`docs/pipeline/recap_transcripts.md`](../../docs/pipeline/recap_transcripts.md)).

```python
from lab.text_to_features.team_perspective import (
    document_perspectives,
    perspective_texts,
    replace_teams,
)

texts = document_perspectives(doc)  # a stored report or transcript
texts = perspective_texts(text, home, away, title=title)
rewrite = replace_teams(text, target="Lakers", opponent="Timberwolves")
rewrite.text, rewrite.mentions  # every matched span with its role
```

## Rules

No model is involved. `team_perspective.py` holds an alias table for the 30
franchises, keyed by the schedule nickname stored with each document:

- **Forms**: nickname, city or region, every city + nickname combination
  ("Los Angeles Lakers", "L.A. Lakers"), singulars ("Laker"), and common short
  forms ("Cavs", "Sixers", "T-Wolves", "OKC", "Philly", "Purple and Gold").
- **Only the two teams of the game are replaced**, cities included. Every
  other team stays as written ("Washington" in a game without the Wizards);
  other teams are matched only so that their full names stay whole ("LA
  Clippers" in a Lakers game is not cut to "Target Team Clippers").
- **Case**: nicknames that are ordinary words (Heat, Magic, Kings, Thunder,
  Jazz, Nets, Wolves, ...) and aliases of three letters or fewer (LA, OKC)
  match only as written or in capitals; the rest match in any case, which
  covers transcripts that lowercase names.
- **Shared cities**: a city both teams carry (Los Angeles / LA / L.A. in a
  Lakers–Clippers game) is left as written.
- Every version, `raw` included, drops the AP dateline ("LOS ANGELES (AP) —",
  the venue in capitals, which would otherwise become a team label), AP footers
  and URL lines, and the leading "- " of Wishper transcripts. `strip=False`
  keeps them.

The labels are parameters (`target_label`, `opponent_label`). The defaults
never occur in NBA prose, so they can be found again unambiguously.

## Players

With the game's players, the rewrites also say which side each player is on:

```python
from lab.text_to_features.documents import load_game_players

players = load_game_players([doc["game_id"]])[doc["game_id"]]
texts = document_perspectives(doc, players=players, label_rule="any")
```

```text
The Opponent Team played without Tyler Herro (right big toe contusion).
Brunson (Target Team player) scored 30 and Adebayo (Opponent Team player) had 12.
```

- **Who**: every player the database lists for the game, from the box score
  (played, DNP, inactive) and the NBA inactive list (`nba_injuries`), so
  injured players are found too. The team comes from `team_id`. Names come from
  the box score, or from `nba_api`'s static list by `player_id` when the box
  score only has "S. Castle" (2024-25) or a single full-name field (before).
- **Forms**: full name (with and without "Jr."/"III"), first name, last name
  ("Jones Jr." too), "TJ"/"T.J.", "OG"/"O.G."; accents are ignored ("Doncic"
  finds "Dončić"). The name is kept and the note goes after it, after a
  possessive ("Anunoby's (Opponent Team player) defense").
- **Ambiguity**: a form that players of both teams carry ("Jalen" in a
  Thunder–Knicks game) is not used. A lone first or last name next to another
  capitalized name is somebody else ("Kobe Bryant", coach "Monty Williams")
  unless that name is close to the player's other half, a misspelling
  ("Austin Reeves" for Reaves).
- **`label_rule`** (`players.LABEL_RULES`), checked on the text before any note
  is added; only the first mention of each player in a sentence gets a note:

  | Value | No note when the sentence contains |
  | --- | --- |
  | `"any"` (default) | either team label |
  | `"own"` | the label of the player's own team |
  | `"never"` | — (always a note) |

On 490 sampled documents (400 reports, 90 transcripts), `"any"` adds a note to
39% of report mentions and 69% of transcript mentions: AP sentences usually
name a team already, transcripts rarely do. `"own"` adds 44% / 73%. About 1.5%
of mentions are taken as somebody else; the misses seen there are nicknames
("Nick Batum"), former names ("Enes Kanter") and badly misheard names
("Assure Thompson").

## Known limits

- No phrase is protected: a city of a playing team used as a place ("a loss
  in Washington last week") and names that contain an alias of a playing team
  ("P.J. Washington" in a Wizards game, "Magic Johnson" in a Magic game) are
  replaced too. Accepted for now; player tagging is the next step.
- Transcription errors are not corrected: a misheard team name stays as heard.
- Grammar follows the source: "the Target Team are", "eight Target Team scored".

## Notebook

[`team_perspective_demo.ipynb`](team_perspective_demo.ipynb) draws random
documents from S3 (two per source by default; `SEASON` and `SEED` select),
loads their players from the database, and shows the raw text with the matched
spans coloured next to both rewrites, with a count of player outcomes.
`LABEL_RULE` sets the player rule.

Checks: `.venv/bin/pytest -q tests/features/test_team_perspective.py tests/features/test_text_players.py`

## Embedding similarity experiment

The notebook also embeds each document's `raw`, `home_target` and `away_target`
versions using [`embeddings.py`](embeddings.py) and
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B).
It displays the three pairwise cosine similarities per document as a table and
an annotated heatmap. The sample uses `SEED = 42` (two reports and two transcripts).
`LABEL_RULE` controls both the displayed rewrites and the embedding inputs.

Additional local dependencies:

```bash
.venv/bin/pip install bitsandbytes==0.48.2 nbformat nbclient
```

The GTX 1060 has 6 GiB VRAM; the model's 4,021,774,336 parameters require at
least 7.49 GiB for FP16 weights alone. The loader selects NF4 with double
quantization, using FP32 compute and unquantized layers on Pascal. All layers
stay on the GPU. It uses the pinned model revision recorded in `embeddings.py`,
last-token pooling, L2 normalization, and no retrieval instruction. Each text
is processed separately and in full; overlength inputs raise an error instead
of silently truncating. The model cache is under `artifacts/huggingface/`.

Tables and heatmaps are displayed directly in the notebook. Embeddings,
transformed inputs and run metadata stay in memory; the experiment does not
export result files.

This compares the **combined** rewrite (team aliases, introduction and player
notes). `raw` is already cleaned with the same boilerplate removal. High cosine
means nearby representations; lower cosine shows sensitivity to the rewrite,
not necessarily a change in meaning. Four documents are an exploratory sample,
and NF4 results have not been validated against an unquantized reference.

Measured on the four-document sample (seed 42, `label_rule="any"`):

| Pair | Mean cosine | Range |
| --- | --- | --- |
| `raw` / `home_target` | 0.8171 | 0.7942–0.8518 |
| `raw` / `away_target` | 0.8297 | 0.8129–0.8625 |
| `home_target` / `away_target` | 0.9869 | 0.9780–0.9934 |

The rewrite shifts the embeddings relative to the cleaned original; switching
the target side changes them much less. The loaded model occupied 3.14 GiB;
peak allocated GPU memory was 3.76 GiB (4.40 GiB reserved by PyTorch), and the
12 embeddings took 32.6 seconds, excluding model loading and download.

## Team-role instruction experiment

The next notebook section repeats the same sample with an editable English
`TEAM_ROLE_INSTRUCTION`. It asks the model to distinguish facts about the
Target Team from facts about the Opponent Team: scoring, offense, defense,
pace, player availability and game outcome. It follows the explicit mapping
already present in the rewritten texts and does not invent one for `raw`.

`compare_perspectives(..., instruction=TEAM_ROLE_INSTRUCTION)` prefixes every
version with Qwen's `Instruct: {instruction}\nQuery:{text}` format. The same
instruction is applied to all three versions; text transformations, model,
quantization, pooling and sample stay the same. `instruction=None` retains the
original experiment's inputs. The in-memory metadata records the instruction,
and the returned document records include the exact model inputs with prefixes.

The instructed heatmap is displayed once, followed by a table comparing each
pair against the experiment without an instruction. Positive cosine deltas
mean the pair became closer; negative deltas mean it separated. Both plots use
the same color scale. The comparison is displayed in the notebook without
exporting files. Results describe this particular prompt and
sample; cosine alone does not establish correct understanding of team roles.

Measured on the same four documents with the English instruction:

| Pair | Without instruction | With instruction | Delta |
| --- | --- | --- | --- |
| `raw` / `home_target` | 0.8171 | 0.9238 | +0.1067 |
| `raw` / `away_target` | 0.8297 | 0.9235 | +0.0938 |
| `home_target` / `away_target` | 0.9869 | 0.9658 | −0.0210 |

The two role perspectives separate slightly more with this instruction, while
both move closer to the neutral original. A common prompt can also shift the
similarities, so this is not proof of correct role reasoning. Encoding took
35.0 seconds with a peak of 3.81 GiB allocated GPU memory.

## Role probes: does an instruction carry the roles?

The cosines above say that an instruction moves the embeddings, not that the
moved vectors know which team did what. [`role_probes.py`](role_probes.py)
measures that directly, on a larger sample, for several instructions
(`role_probes.INSTRUCTIONS`):

| Name | Kind | Instruction |
| --- | --- | --- |
| `none` | control | — |
| `neutral` | control | archive-storage task of similar length, no roles |
| `current` | candidate | the `TEAM_ROLE_INSTRUCTION` above |
| `target_focus` | candidate | how the Target Team played: result, scoring, defense, pace, missing players |
| `target_view` | candidate | "Represent this NBA game recap from the Target Team's point of view." |
| `opponent_view` | mirror | `target_view` with "Opponent Team": if the roles matter, it reverses them |
| `next_game` | candidate | what helps predict the Target Team's points for and against next game |

A candidate earns credit only for what it gains over `neutral`: a shared
prefix alone moves every vector.

```bash
.venv/bin/python -m lab.text_to_features.run_role_probes --model 0.6B
.venv/bin/python -m lab.text_to_features.run_role_probes --model 4B --conditions none neutral current
.venv/bin/python -m lab.text_to_features.run_role_probes --model 8B
```

0.6B and 4B run on the local GPU; 8B (Qwen3-Embedding-8B, 4096 dimensions)
through the Nebius Token Factory API ([`nebius.py`](nebius.py), key in
`[Nebius] NEBIUS_API_KEY`, smoke test in `nebius_embedding_api_test.ipynb`).
The API returns L2-normalized vectors; the instruction is applied locally in the
same format, and the billed tokens equal the local tokenizer's count. The full
8B sweep (seven conditions) took 8 minutes and 5.08M tokens (about $0.05 at
$0.01/M).

The first run draws the sample and stores it, with its labels and texts, under
`artifacts/text_to_features/role_probes/`; every later run, any model, reuses
it. Embeddings go to `<model>/`, one `.npy` per condition; existing files are
skipped, so a stopped run resumes. The notebook section "Does an instruction
make the embedding know which team did what?" reads those files (no GPU).
Speed on the GTX 1060: 0.6B (FP32, unquantized) ~0.4 s per document, the
whole sweep ~50 min; 4B (NF4) ~2.1 s per document.

**Minimal pairs.** 44 synthetic triplets in seven facets: an anchor ("Target
Team beat Opponent Team 112-104."), the same roles reworded ("Opponent Team
lost to Target Team 112-104.") and the roles swapped ("Opponent Team beat
Target Team 112-104."), alone and inside a neutral paragraph. Accuracy = the
rewording is closer than the swap; the swap ratio grades how far the swap
moves the anchor per unit of the rewording's move.

**Linear probes.** 500 documents (250 reports, 250 transcripts, 2023-24 to
2025-26; seed 2026; games where both teams had played 10 season games), both
rewrites. Labels from the box score seen from the Target Team: won, margin,
points for and against, FG%, 3PM/rebound/turnover differences, and regulars
out (players who averaged 24+ minutes in 5+ of the team's previous 10 games,
listed for the game without minutes). Ridge probe, out of fold, five folds
grouped by game. **Pair accuracy**: within a game, the home rewrite gets the
higher prediction exactly when the home team has the higher label; 50% means
blind to the roles. Baselines: `side only` (the home rate) and `raw + side`
(the raw text's embedding and its product with the side).

### Results, Qwen3-Embedding-0.6B

Pair accuracy (95% Wilson intervals about ±2.5 pp for the first rows, ±4 pp for
regulars out, where 41% of the games have equal counts and drop out):

| Label | side only | raw + side | none | neutral | current | target_focus | target_view | next_game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target_won | 0.562 | 0.562 | 0.916 | 0.934 | 0.956 | 0.926 | **0.964** | 0.924 |
| target_margin | 0.562 | 0.554 | 0.900 | 0.920 | 0.946 | 0.914 | **0.956** | 0.898 |
| target_points | 0.562 | 0.520 | 0.894 | 0.922 | **0.950** | 0.900 | 0.942 | 0.880 |
| opponent_points | 0.562 | 0.520 | 0.892 | 0.876 | **0.930** | 0.900 | 0.910 | 0.884 |
| target_fg_pct | 0.569 | 0.546 | 0.708 | 0.726 | 0.748 | 0.718 | **0.754** | 0.690 |
| reb_diff | 0.525 | 0.523 | 0.692 | 0.688 | **0.711** | 0.694 | 0.709 | 0.674 |
| target_regulars_out | 0.527 | 0.493 | 0.582 | 0.548 | 0.585 | 0.554 | 0.660 | 0.599 |
| opponent_regulars_out | 0.527 | 0.493 | 0.476 | 0.531 | 0.527 | 0.531 | 0.422 | 0.449 |

Paired differences over `neutral` (same documents, 95% bootstrap):

| | target_won | target_margin | target_points | opponent_points |
| --- | --- | --- | --- | --- |
| `current` | +2.2 [+0.2, +4.2] | +2.6 [+0.6, +4.6] | +2.8 [+0.8, +4.8] | +5.4 [+3.0, +8.0] |
| `target_view` | +3.0 [+1.2, +5.0] | +3.6 [+1.8, +5.6] | +2.0 [−0.2, +4.2] | +3.4 [+0.6, +6.2] |
| `target_focus` | −0.8 [−2.8, +1.4] | −0.6 [−2.6, +1.6] | −2.2 [−4.4, 0.0] | +2.4 [+0.2, +4.8] |
| `next_game` | −1.0 [−3.2, +1.2] | −2.2 [−4.4, 0.0] | −4.2 [−6.6, −1.8] | +0.8 [−2.0, +3.6] |

What this says:

- **The rewrite carries the roles; the instruction refines them.** Without
  any instruction a probe reads who won from the rewrites in 92% of games. From
  the raw text it cannot (56% = the home win rate): with real names and no
  dateline, the raw text does not say which team is home. The shared controls
  (`total_points` r 0.69–0.75, `pace`) are read as well from the rewrites as
  from the raw text, so the rewrite loses no shared information.
- **`current` and `target_view` add 2–5 pp over both controls** on result
  and margin, with intervals above zero; `current` also on both point totals,
  `target_view` not on every one. `target_focus` and `next_game` do not. The gain is in the reports (target_won 0.89 → 0.96–0.97); the
  transcripts are at 0.94 already without an instruction and stay there,
  likely because most of their player mentions carry a side note.
- **Availability by side is not captured, with any instruction.** Each
  instruction's two availability rows add up to 1.05–1.11 (`target_view`
  0.66 + 0.42): the probe finds "regulars were missing" (r ≈ 0.27 for both
  sides), not on which side. Even with `target_view` the average is 54%. The
  label is a rule of ours; a recap may name other absences. This is the information the box
  score does not already give, so it is the part that matters most for
  features, and the instructions do not deliver it.
- **The minimal pairs do not predict the probes.** No condition passes them
  (the swap is closer than the rewording in 89–100% of triplets), and their
  ranking does not follow the probes' (`none` has the highest short-sentence
  swap ratio). Keep them as a sanity check, not for choosing an instruction.

These are decodability checks on one sample and one seed, with several
comparisons; none of them says whether a text feature helps predict the next
game, which only a walk-forward ablation can show.

### Results, Qwen3-Embedding-4B (NF4, local)

Same sample, labels and folds; `none`, `neutral`, `current` and `target_view`
(`opponent_view` is still running, so no transfer test yet). Pair accuracy:

| Label | none | neutral | current | target_view |
| --- | ---: | ---: | ---: | ---: |
| target_won | 0.972 | 0.996 | **1.000** | **1.000** |
| target_margin | 0.960 | **0.996** | **0.996** | 0.994 |
| target_points | 0.958 | 0.988 | **0.990** | 0.972 |
| opponent_points | 0.956 | 0.990 | **0.994** | 0.978 |
| target_fg_pct | 0.774 | 0.800 | **0.808** | 0.806 |
| reb_diff | **0.721** | 0.717 | **0.721** | 0.719 |
| target_regulars_out | 0.558 | 0.602 | 0.653 | **0.663** |
| opponent_regulars_out | 0.554 | 0.551 | **0.565** | 0.507 |

- Like 8B, the result is at the ceiling from `neutral` on (0.996); the
  instructions add at most 0.4 pp there. Over `none`, `current` and
  `target_view` add about 3 pp, so most of that is the prefix, not the roles.
- `target_view` raises `target_regulars_out` by +6.1 pp [+0.7, +11.6] over
  `neutral` but lowers `opponent_regulars_out` by 4.4 pp: the mean of both
  sides does not move (see "Availability by side" below).
- Minimal pairs in context: `target_view` 0.77, `current` 0.52, `neutral`
  0.23, `none` 0.21.
- Speed on the GTX 1060: about 2.1 s per document text, 35–50 minutes per
  condition.

### Results, Qwen3-Embedding-8B (Nebius)

Same sample, labels and folds. Pair accuracy:

| Label | none | neutral | current | target_focus | target_view | opponent_view | next_game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target_won | 0.988 | 0.992 | 1.000 | 0.998 | **1.000** | 0.996 | 1.000 |
| target_margin | 0.984 | 0.988 | 0.996 | **0.998** | 0.994 | 0.992 | 0.992 |
| target_points | 0.978 | 0.986 | 0.986 | **0.992** | 0.982 | 0.992 | 0.970 |
| opponent_points | 0.976 | 0.984 | **0.998** | 0.994 | 0.992 | 0.980 | 0.974 |
| target_fg_pct | 0.778 | 0.792 | **0.808** | 0.804 | 0.804 | 0.792 | 0.790 |
| reb_diff | 0.717 | 0.717 | 0.721 | 0.721 | 0.719 | 0.719 | **0.725** |
| target_regulars_out | 0.602 | 0.605 | 0.660 | 0.663 | **0.677** | 0.510 | 0.673 |
| opponent_regulars_out | 0.483 | 0.507 | 0.473 | 0.548 | 0.551 | **0.629** | 0.490 |

Paired differences over `neutral`, regulars out (pp, 95% bootstrap):

| | target_regulars_out | opponent_regulars_out |
| --- | --- | --- |
| `current` | +5.4 [+0.3, +10.2] | −3.4 [−8.5, +1.7] |
| `target_focus` | +5.8 [+1.0, +10.5] | +4.1 [−1.4, +9.5] |
| `target_view` | +7.1 [+2.0, +12.2] | +4.4 [−1.0, +9.5] |
| `next_game` | +6.8 [+1.4, +12.2] | −1.7 [−7.5, +4.1] |
| `opponent_view` | −9.5 [−15.0, −4.4] | +12.2 [+5.4, +19.4] |

**Transfer.** Probes fitted on one condition and applied to the vectors of
another (same folds). Pair accuracy of the `target_view` probes:

| Label | none | neutral | current | target_focus | target_view | opponent_view | next_game |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target_won | 0.976 | 0.996 | 0.998 | 0.996 | 1.000 | **0.010** | 0.994 |
| target_points | 0.896 | 0.946 | 0.980 | 0.978 | 0.982 | **0.144** | 0.980 |
| opponent_points | 0.912 | 0.982 | 0.990 | 0.986 | 0.992 | **0.090** | 0.956 |
| target_regulars_out | 0.609 | 0.602 | 0.670 | 0.633 | 0.677 | **0.364** | 0.650 |

Per document, the cosine between the home-minus-away rewrite difference under
`target_view` and under each condition: `none` +0.40, `neutral` +0.54,
`current` +0.72, `target_focus` +0.73, `next_game` +0.68, **`opponent_view`
−0.57** (negative in 100% of documents, every other condition in at most 0.2%).

What changes from 0.6B:

- **Result and points are at the ceiling without any instruction** (98–99%
  for `none` and `neutral`). The rewrite alone carries them; the instructions
  add about 1 pp, with intervals that just clear zero on `target_won`.
- **Availability by side is captured, with `target_view` and
  `target_focus`.** Every Target-oriented instruction raises
  `target_regulars_out` by 5–7 pp over both controls, and `opponent_view`
  moves the gain to `opponent_regulars_out` (+12 pp) while
  `target_regulars_out` drops to 0.51. Only `target_view` and `target_focus`
  also keep the opponent side, so the mean of both sides improves (next
  section): the vector says on which side the regulars were missing. Still far
  from the result's 99%, and the label is our rule, not what the recap says.
- **`opponent_view` really turns the roles around.** The `target_view`
  probes read it with 1% pair accuracy on `target_won`: 99% of games flipped,
  not just noise. `opponent_view` of the home rewrite is closer to
  `target_view` of the away rewrite (cosine 0.955, both about the away team)
  than to `target_view` of the same text (0.895) in 98.8% of documents: the
  instruction outweighs the text. `current`, `target_focus` and `next_game`
  read the roles like `target_view`; `none` and `neutral` lean the same way
  (the rewrites name the Target Team first).
- **Minimal pairs work for the Target-oriented instructions** (accuracy in
  context: `next_game` 0.86, `target_view` 0.82, `current` 0.80,
  `target_focus` 0.73, against `none` 0.25, `neutral` 0.39 and
  `opponent_view` 0.36). With 8B they agree with the probes on which
  instructions carry roles.

`target_view` is the instruction to keep for 8B: the shortest, within 1 pp of
the best on result, margin and points, the best on `target_regulars_out`, and
its mirror behaves as a mirror should.

### Availability by side, all models

The two availability labels are mirror images (`target_regulars_out` of the
home rewrite is `opponent_regulars_out` of the away rewrite), and the games
where both sides have the same count drop out of both (294 of 500 documents
remain). A probe can raise one of them by reading "missing players" as the
Target Team's, at the cost of the other. The measure that rewards knowing
the side is the per-document mean of the two pair outcomes:

| Model | Condition | Target | Opponent | Mean | Mean − `neutral` [95% CI] |
| --- | --- | ---: | ---: | ---: | --- |
| 0.6B | neutral | 0.548 | 0.531 | 0.539 | |
| 0.6B | current | 0.585 | 0.527 | 0.556 | +1.7 [−1.7, +5.3] |
| 0.6B | target_view | 0.660 | 0.422 | 0.541 | +0.2 [−3.7, +4.4] |
| 4B | neutral | 0.602 | 0.551 | 0.577 | |
| 4B | current | 0.653 | 0.565 | 0.609 | +3.2 [−0.2, +6.5] |
| 4B | target_view | 0.663 | 0.507 | 0.585 | +0.9 [−2.9, +4.3] |
| 8B | neutral | 0.605 | 0.507 | 0.556 | |
| 8B | current | 0.660 | 0.473 | 0.566 | +1.0 [−2.6, +4.4] |
| 8B | target_focus | 0.663 | 0.548 | 0.605 | +4.9 [+1.4, +8.3] |
| 8B | target_view | 0.677 | 0.551 | **0.614** | **+5.8 [+2.2, +9.4]** |
| 8B | next_game | 0.673 | 0.490 | 0.582 | +2.6 [−1.4, +6.3] |
| 8B | opponent_view | 0.510 | 0.629 | 0.570 | +1.4 [−2.7, +5.3] |

Only 8B with `target_view` or `target_focus` has an interval above zero. This is
the main reason for recommending 8B over 4B: on result and points they are
equal.

