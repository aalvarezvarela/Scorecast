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

``2_4``
    Closing-line dataset only; the intermediate dataset's content is unchanged.
    Reads the last injury report before each tipoff from Aiven
    ``injury_report`` (``nba_ou.data_processing.injury_status``). See
    ``docs/injury_status_tiers_plan.md``.

    * Pre-game out set = Out ∪ Doubtful ∪ Questionable on that report, for every
      team-game whose team had filed. It replaces the inactive-list/comment set
      behind ``TOP*_INJURED_*``, ``N_INJURED_PLAYERS``, the available side,
      ``ALL_STAR_*_INJURED_*`` and the current game of the injured streak.
      Team-games without a report (2017-18, most of 2018-19, the odd unfiled
      team) keep the 2_3 set. History -- earlier games of streaks, the
      availability-effect present/absent map, roster continuity -- still uses
      realized absences.
    * New per side: ``INJURY_REPORT_COVERED``, ``LAST_STATUS_REPORT_AGE_MIN``,
      ``N_REPORT_{OUT,DOUBTFUL,QUESTIONABLE,PROBABLE}_PLAYERS`` (G League
      listings not counted), ``N_AVAILABLE_ROSTER_PLAYERS``.
    * New listed-status block per side, for Questionable (top 2), Probable
      (top 1) and Doubtful (top 1), ranked by minutes form for the out-set
      statuses and by points form for Probable, skipping G League listings
      exactly as the counters do: per player
      ``FORM_{MIN,PTS,PACE_PER40}``, their own pre-game form. Questionable and
      Probable add ``P_PLAY``, ``N_HISTORY`` and
      ``EFFECT_{MIN,PTS,PACE_PER40}``, plus
      ``SUM_<STATUS>_EXP_{PLAYERS,MIN,PTS}``, ``MEAN_<STATUS>_P_PLAY`` and
      ``LEAGUE_<STATUS>_P_PLAY``. Doubtful gets form only: it plays ~2% of the
      time at the last report, so a probability or effect fitted for it is
      noise, while the size of the player being lost is a fact. Empty slots on a
      covered team-game take the neutral values (``P_PLAY`` 1, everything else
      0); uncovered team-games are NaN.

    ``create_train_data.py --no-injury-report-features`` is the control: it
    writes ``training_data_2_3_<date>_rebuild.csv`` with the 2_3 columns and
    out sets, and none of the above.

``2_5``
    Closing-line dataset only. Splits the report into **three availability
    groups** instead of folding Questionable into the out set, and gives the new
    group the same feature families the other two have.

    * Pre-game groups on a covered team-game: **injured** = Out ∪ Doubtful,
      **questionable** = Questionable, **available** = everyone else on the
      roster. The three partition the roster, so a Questionable player is
      neither subtracted as an absence (2_4 did that, and those players play 62%
      of the time) nor counted among the available.
    * ``TOP*_INJURED_*``, ``N_INJURED_PLAYERS``, ``AVG_INJURED_*``,
      ``TOTAL_INJURED_PLAYER_*``, ``ALL_STAR_*_INJURED_*`` and
      ``TOP3_INJURED_AVAILABILITY_EFFECT_*`` therefore change value: they now
      describe Out ∪ Doubtful only. The available-side columns are unchanged
      from 2_4.
    * New per side, the questionable group mirroring the injured one, for every
      statistic in ``stats``: ``TOP{1,2}_QUESTIONABLE_PLAYER_<stat>``,
      ``AVG_QUESTIONABLE_<stat>``, ``TOTAL_QUESTIONABLE_PLAYER_<stat>``,
      plus ``N_QUESTIONABLE_PLAYERS``, ``TOP{1,2}_QUESTIONABLE_STREAK_PTS``,
      ``ALL_STAR_{MAX_QUESTIONABLE_FAN_VOTE_SHARE,MIN_QUESTIONABLE_SCORE}`` and
      the ``TOP2_QUESTIONABLE_AVAILABILITY_EFFECT_*`` block. A team-game the
      report does not cover has no questionable group, so every one of these is
      NaN there -- "nobody Questionable" and "no report" are different facts.
      The effect builder is coverage-blind (it fills a missing aggregate with
      the shrinkage limit, 0), so that block is masked by
      ``INJURY_REPORT_COVERED`` per side after it runs.
    * ``ALL_STAR_MIN_QUESTIONABLE_SCORE`` is the one column with a neutral
      value on a covered but empty group: 1000, a stand-in for the +infinity a
      minimum over an empty set really is, above every score the voting table
      produces. Its injured twin keeps NaN there, because the 2_3 control build
      has to reproduce that column value for value.
    * The 2_4 status block (``P_PLAY``, ``EFFECT_*``, ``FORM_*``, the
      ``SUM_/MEAN_/LEAGUE_`` aggregates) is unchanged and stays separate: it
      says what a *status* implies for a player, not what a group contributes to
      a team. A Questionable player carries both.
"""

from __future__ import annotations

#: Current schema version for both generated training datasets.
TRAINING_DATA_SCHEMA_VERSION = "2_5"
