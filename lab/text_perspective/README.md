# Team-perspective game texts

Exploration code: turns one game text into three, so that a later text model
reads it from either side of the game.

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
from lab.text_perspective.team_perspective import (
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
from lab.text_perspective.documents import load_game_players

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
