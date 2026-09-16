"""Pre-game injury-report statuses as closing-line features.

See ``docs/injury_status_tiers_plan.md``. Two modules:

* :mod:`.report_state` -- the last report before each tipoff: the three
  availability groups it splits a roster into (injured = Out ∪ Doubtful,
  questionable, available), per-category counters, coverage.
* :mod:`.status_history` -- per-player columns for the players listed
  Questionable, Probable or Doubtful: their own form in minutes, points and
  pace, and for the first two the chance of playing and the change in those
  stats when they do.
"""
