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

    ``create_train_data.py --no-injury-report-features`` omits report-derived
    availability. In the current build it retains the other 2_5 features and
    writes a separately suffixed 2_5 file.

``2_5``
    Splits closing-line report availability into **three groups** instead of
    folding Questionable into the out set. It also adds referee and player
    feature changes to both generated datasets.

    * Pre-game groups on a covered team-game: **injured** = Out ∪ Doubtful,
      **questionable** = Questionable, **available** = everyone else on the
      roster. The three partition the roster, so a Questionable player is
      neither subtracted as an absence (2_4 did that, and those players play 62%
      of the time) nor counted among the available.
    * ``TOP*_INJURED_*``, ``N_INJURED_PLAYERS``, ``AVG_INJURED_*``,
      ``TOTAL_INJURED_PLAYER_*``, ``ALL_STAR_*_INJURED_*`` and
      ``TOP3_INJURED_AVAILABILITY_EFFECT_*`` therefore change value: they now
      describe Out ∪ Doubtful only. The available side follows the reduced
      player profile described below.
    * New per side, the questionable group mirroring the injured one under the
      reduced player profile: PTS/MIN per-player values and aggregates,
      ``N_QUESTIONABLE_PLAYERS``, ``TOP{1,2}_QUESTIONABLE_STREAK_{PTS,MIN}``,
      and minutes-weighted rate aggregates. Per-slot rate columns and sums of
      rates are omitted by the same rules as the injured group. The group also
      has ``ALL_STAR_{MAX_QUESTIONABLE_FAN_VOTE_SHARE,MIN_QUESTIONABLE_SCORE}``
      and ``TOP2_QUESTIONABLE_AVAILABILITY_EFFECT_*``. A team-game the
      report does not cover has no questionable group, so every one of these is
      NaN there -- "nobody Questionable" and "no report" are different facts.
      The effect builder is coverage-blind (it fills a missing aggregate with
      the shrinkage limit, 0), so that block is masked by
      ``INJURY_REPORT_COVERED`` per side after it runs.
    * ``ALL_STAR_MIN_QUESTIONABLE_SCORE`` is the one column with a neutral
      value on a covered but empty group: 1000, a stand-in for the +infinity a
      minimum over an empty set really is, above every score the voting table
      produces. Its injured twin keeps NaN there.
    * The 2_4 status block (``P_PLAY``, ``EFFECT_*``, ``FORM_*``, the
      ``SUM_/MEAN_/LEAGUE_`` aggregates) is unchanged and stays separate: it
      says what a *status* implies for a player, not what a group contributes to
      a team. A Questionable player carries both.

    The same 2_5 build also adds the referee crew tendency family, fresh
    absence interactions and the reduced player feature profile. The referee
    features are available for both closing and intermediate datasets.

    * ``REF_CREW_*_TENDENCY_BEFORE`` for free throws, fouls, possessions and
      line error (totals track) and for spread error, favourite spread error,
      home free-throw edge and home foul edge (spread track), plus
      ``REF_CREW_MIN_PRIOR_GAMES_BEFORE`` and ``REF_CREW_UNKNOWN_COUNT_BEFORE``.
    * ``REF_CREW_*_X_*_BEFORE`` combinations with expected free throws,
      absolute spread and expected home free-throw-rate edge.
    * Optional ``REF_CREW_SS_*`` same-season-only variants when built with
      ``include_same_season_referee_variants=True``.
    * The legacy ``REF_AVG/STD/SUM_*`` columns stay for comparison.

    Intermediate-line builds only, added later under the same 2_5 label:

    * The per-book ``DEVIATION_FROM_CONSENSUS``, ``ABS_DEVIATION_FROM_CONSENSUS``,
      ``DEVIATION_Z`` and ``IS_OUTLIER_BOOK`` columns keep their names but now
      measure against the leave-one-book-out peer median, with grossly
      discrepant peers removed and the gap capped. **Intermediate 2_5 files
      built before and after this change are not directly comparable** on
      those columns; rebuild rather than mixing them.
    * New anchor-total path columns ``ODDS_SNAP_TOT_<ANCHOR>_*``:
      ``MINUTES_SINCE_LAST_LEVEL_MOVE``, ``PEERS_MOVED_ANCHOR_STILL_60``,
      ``ABS_LEVEL_PATH_60``, ``SIGNED_MOVE_STREAK_60`` and
      ``LAST_TWO_LEVEL_MOVES_GAP_MIN``, plus the walk-forward Ridge
      ``ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE``.
"""

from __future__ import annotations

#: Current schema version for both generated training datasets.
TRAINING_DATA_SCHEMA_VERSION = "2_5"
