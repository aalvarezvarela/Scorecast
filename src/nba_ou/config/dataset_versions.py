"""Schema versions for the generated training datasets.

Bumped whenever a regenerated CSV gains or loses columns, so a new build lands
beside the old one instead of overwriting it. Both training pipelines pin their
``expected_checksum`` against a specific file, and an in-place overwrite turns
that guard into a failure at the *next* run rather than a clean, dated artifact.

History
-------
``2_0``
    Totals only. ``TOTAL_POINTS`` is the sole outcome column; per-team final
    scores are dropped by the selection gates.

``2_1``
    Adds the spread market and moneyline data readiness:

    * ``PTS_TEAM_HOME`` / ``PTS_TEAM_AWAY`` / ``HOME_MARGIN`` survive selection
      (outcome columns, blocked from every feature matrix).
    * ``ODDS_SPREAD_LINE_HOME_<book>`` -- every book's spread normalised to the
      implied-home-margin convention.
    * Cross-book spread consensus features (median-based).
    * ``ODDS_ML_PRICE_HOME`` / ``_AWAY`` and de-vigged probabilities.
    * Intermediate dataset only: ``SPREAD_ERROR``, derived per snapshot against
      the Bet365 spread as of THAT snapshot.

    Totals columns and semantics are unchanged, so a 2_1 file trains the existing
    totals strategies identically to a 2_0 file built from the same games.

``2_2``
    Keeps the 2_1 columns but changes closing spread semantics to match totals:
    asymmetrically priced spread quotes are centered to their estimated
    -110/-110 equivalent, and implausibly extreme spread prices are nulled
    before centering. Intermediate spread snapshots already used centered
    ``norm_line`` values; 2_2 applies the same extreme-price guard there too.

``2_3``
    Rebuilds player availability and the availability-effect family. Columns and
    values both move, so a 2_3 file is not comparable to a 2_2 file even for the
    totals strategies.

    * The injured/available split is decided by the ABSENCE REASON, not by
      target-game minutes. The inferred "expected-rotation DNP" rule is gone --
      it read ``MIN`` from the game being predicted, the same class of leak
      recorded in ``nba_ou.config.leakage``. In its place the box-score
      ``COMMENT`` filter widened from injury wording alone to every reason the
      pre-game NBA injury report would also have carried: Injury/Illness, Rest,
      Personal Reasons, suspensions and Trade Pending. ``DNP - Coach's
      Decision`` and ``NWT - G League`` stay available. This changes the
      membership behind every ``TOP*_INJURED_PLAYER_*`` and
      ``TOP*_PLAYER_*`` column.
    * Availability-effect columns: ``WIN_RATE_DIFF`` dropped, and the
      ``BONFERRONI_PVALUE_*`` family replaced by ``MEAN_SE_*``. The compact
      schema goes from 22 to 18 columns per call.
    * Effect values change again through shrinkage: the fixed ``k = 10`` is
      replaced by an empirical-Bayes weight fitted per metric on strictly
      earlier games.
"""

from __future__ import annotations

#: Current schema version for both generated training datasets.
TRAINING_DATA_SCHEMA_VERSION = "2_3"
