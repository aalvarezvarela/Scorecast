"""Which player/availability columns the pipeline emits.

The presence/absence block grew to 216 columns (12% of the dataset) by taking
the cross product of 6 statistics x 4-6 top-N slots x 2 sides, for both the
available and the injured roster, and then adding per-team averages and sums on
top. Most of that cross product is redundant, and the redundancy is measurable:

- **Summing a rate is meaningless**, and the data says so:
  ``TOTAL_INJURED_PLAYER_DEF_RATING`` correlates **0.98** with
  ``N_INJURED_PLAYERS``. Summing OFF_RATING / DEF_RATING / PACE_PER40 / TS_PCT
  over players re-counts how many are out, with extra steps. Sums survive only
  for the counting statistics, PTS and MIN.
- **Per-slot rate statistics repeat each other.** ``TOP2_PLAYER_PACE_PER40``
  and ``TOP3_PLAYER_PACE_PER40`` correlate **0.97**: ranking rotation players by
  a rate returns nearly the same number whichever slot you read. They are
  replaced by one minutes-weighted aggregate per side, which is the quantity
  those columns were reaching for and which does not degrade when a slot is
  empty.
- **Deep slots are mostly padding.** Among injured players, slot 3 is zero or
  missing in 27% of games and slot 4 in 50%.
- **The existing cleaning step already deleted most of this.** At the
  ``corr_threshold: 0.95`` the training pipeline uses, 216 generated columns
  become 96 -- and only **4 of the 72** top-active-player columns and **1 of 12**
  ``AVG_INJURED`` columns survive. Generating them costs build time and reader
  attention to produce something the model never sees.

Gain importance says the block as a whole pays its way (11.4% of columns, 12.7%
of gain), so this is a redundancy cut, not a "these are useless" cut. The
families kept are the ones that carry information no other column does.

Two constraints shape what could NOT be cut:

1. ``add_top3_availability_effect_features_for_columns`` locates players through
   the **id** columns ``TOP1-3_PLAYER_ID_PTS`` and ``TOP1_PLAYER_ID_MIN`` (and
   the injured equivalents). Those ids must keep being produced or the 36
   availability-effect columns break. They are bookkeeping and are dropped from
   the frame at the end of the pipeline either way.
2. The fresh-absence features (``players/fresh_absence.py``) sum over **all four**
   injured slots of PTS and MIN. Those slots therefore keep their value columns;
   the saving comes from dropping the four rate statistics across every slot
   instead, which lands in the same place without silently redefining a feature.

**Switching back.** ``ACTIVE_PROFILE`` selects the column set. Serving reads the
feature schema stored in each model bundle and *raises* on a missing feature
(``prediction.py``), so a model trained on the legacy set cannot score a reduced
frame. Pin ``ACTIVE_PROFILE = LEGACY_PROFILE`` to keep the daily prediction job
running until the six production models have been retrained and promoted on a
regenerated dataset.
"""

from dataclasses import dataclass, field

#: Statistics the pipeline computes player averages for, in rank order.
ALL_STAT_COLS = ("PTS", "PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT", "MIN")

#: Rate statistics: an average over players is meaningful, a sum is not.
RATE_STAT_COLS = ("PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT")

#: Counting statistics: both an average and a sum are meaningful.
COUNTING_STAT_COLS = ("PTS", "MIN")


@dataclass(frozen=True)
class PlayerFeatureProfile:
    """Which player columns to emit. See module docstring for the measurements."""

    #: Statistics emitting a per-slot VALUE column for available players.
    active_value_stats: tuple[str, ...]
    #: How many top-N slots get a value column, per statistic above.
    active_value_slots: int
    #: The same, for injured players.
    injured_value_stats: tuple[str, ...]
    injured_value_slots: int
    #: Statistics emitting ``AVG_INJURED_<stat>`` / ``TOTAL_INJURED_PLAYER_<stat>``.
    avg_injured_stats: tuple[str, ...]
    total_injured_stats: tuple[str, ...]
    #: Rate statistics collapsed into one minutes-weighted aggregate per side,
    #: for the available and the injured set respectively.
    weighted_rate_stats: tuple[str, ...]
    #: Bench columns to keep.
    bench_cols: tuple[str, ...]
    #: Slots whose ID/NAME bookkeeping columns are produced, per statistic.
    #: Consumed by the availability-effect features, then dropped.
    id_slots_by_stat: dict[str, int] = field(default_factory=dict)

    def emits_value(self, stat_col: str, slot: int, injured: bool) -> bool:
        stats = self.injured_value_stats if injured else self.active_value_stats
        slots = self.injured_value_slots if injured else self.active_value_slots
        return stat_col in stats and slot <= slots

    def emits_identifier(self, stat_col: str, slot: int) -> bool:
        return slot <= self.id_slots_by_stat.get(stat_col, 0)


#: What the pipeline produced before the reduction. Kept so a dataset can be
#: regenerated to match a model bundle trained on it.
LEGACY_PROFILE = PlayerFeatureProfile(
    active_value_stats=ALL_STAT_COLS,
    active_value_slots=6,
    injured_value_stats=ALL_STAT_COLS,
    injured_value_slots=4,
    avg_injured_stats=ALL_STAT_COLS,
    total_injured_stats=ALL_STAT_COLS,
    weighted_rate_stats=(),
    bench_cols=(
        "BENCH_AVG_PTS_PER_MIN",
        "BENCH_MAX_PTS_PER_MIN",
        "BENCH_AVG_PACE_PER40",
        "BENCH_MAX_PACE_PER40",
        "N_BENCH_PLAYERS",
    ),
    id_slots_by_stat={stat: 6 for stat in ALL_STAT_COLS},
)

#: The reduced set: 216 presence/absence columns -> 90.
REDUCED_PROFILE = PlayerFeatureProfile(
    # Slots 4-6 of the available roster are deep-rotation players already
    # described by the team-level rolling stats.
    active_value_stats=COUNTING_STAT_COLS,
    active_value_slots=3,
    # All four injured slots, because fresh_absence.py sums over them.
    injured_value_stats=COUNTING_STAT_COLS,
    injured_value_slots=4,
    avg_injured_stats=COUNTING_STAT_COLS,
    total_injured_stats=COUNTING_STAT_COLS,
    weighted_rate_stats=("OFF_RATING", "PACE_PER40"),
    # One average and one count; the MAX variants tracked the same rotations.
    bench_cols=("BENCH_AVG_PTS_PER_MIN", "N_BENCH_PLAYERS"),
    # Ids/names for the PTS ranking and the top MIN player only -- dropping the
    # other four statistics' ids is where the saving is. The PTS ranking keeps
    # its full depth although the availability-effect features read just the top
    # three: these columns never reach a model (``drop_player_identifier_columns``
    # removes them at the end of the pipeline), so the only thing trimming them
    # further would buy is losing the surface the roster-classification tests
    # and any future debugging read the split through.
    id_slots_by_stat={"PTS": 6, "MIN": 1},
)

#: The profile in force. See "Switching back" in the module docstring before
#: changing this: serving raises on a feature a model bundle expects and the
#: frame no longer has.
ACTIVE_PROFILE = REDUCED_PROFILE
