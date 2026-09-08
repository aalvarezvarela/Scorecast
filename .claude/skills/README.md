# Skills

`experiments` is repo-specific: it describes `training_pipeline/`,
`experiments/` and `artifacts/experiments/` in this repository.

The other four are **portable**. They were distilled from this NBA system but
are written as sport-agnostic decisions with the NBA implementation as
evidence. The architecture and feature skills also include explicit MLB
translations:

- `odds-data-architecture` — SBR ingestion, tick storage, snapshots
- `sports-data-architecture` — entities, identity, availability, context data
- `feature-engineering` — temporal rules and the eleven feature families
- `sports-model-pipeline` — cleaning, training, temporal CV, Optuna selection,
  walk-forward holdout, uncertainty, and reproducibility

To bootstrap another sport, copy those four directories into the new repo's
`.claude/skills/` unchanged. Each provides an ordered workflow or build
checklist. Port
`experiments` too once the new repo has a training pipeline — the evaluation
discipline in it (seed noise before ranking, CV vs holdout, the silent-no-op
table) is sport-independent even though the file paths are not.
