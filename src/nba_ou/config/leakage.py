"""Column families that are functions of the game they are attached to.

Separate from the outcome columns named in ``training_pipeline.config``
(``TOTAL_POINTS``, ``HOME_MARGIN``, ...). Those are obviously the result. The
families here *look* like ordinary pre-game features -- they carry the
``_BEFORE_`` suffix, their values are correctly lagged EWMAs -- and only the
MEMBERSHIP of the set they aggregate over is post-game. That is why they went
unnoticed, and why they are named centrally rather than left to a config field.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Column-name prefixes whose value depends on WHICH PLAYERS APPEARED in the
#: game being predicted.
#:
#: ``get_top_n_averages_with_names`` resolves the non-injured player set as
#: ``df[df["GAME_DATE"] == date]`` whenever the game has already been played
#: (falling back to prior games only for a scheduled game), and the frame it
#: reads has already been filtered to ``MIN > 0``. So the set is "players who
#: logged minutes tonight", and anything counting or summing over it is a
#: readout of the rotation the coach actually used.
#:
#: Measured on ``training_data_2_2_20260901.csv`` (8,935 games, 2019-2025):
#:
#:   * ``corr(N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME, |HOME_MARGIN|) = +0.55``.
#:     Mean count runs 9.74 for games decided by <=5 points up to 12.31 for
#:     games decided by >30 -- bench-emptying in a blowout, read backwards.
#:     It is also LOWER in overtime games (9.78 vs 10.65), which are close and
#:     rotate short.
#:   * Home and away both correlate POSITIVELY with the spread residual
#:     (+0.116 / +0.085). A real availability effect must be opposite-signed
#:     between the two teams; the shared sign is the artifact showing through.
#:   * Removing these 14 columns (of 1,421 features) moved a spread_error
#:     regressor from 66.7% to 52.4% against the closing spread on a 609-game
#:     holdout. Removing either family ALONE changed almost nothing (65.8% /
#:     67.1%) because each reconstructs the other -- which is why a one-at-a-
#:     time ablation misses this.
#:
#: The totals market is untouched by it (``corr`` with ``TOTAL_POINTS`` is
#: +0.003), which is exactly why only the spread runs looked spectacular.
#:
#: Prefix matching, because each family fans out over stat and team
#: (``TOTAL_NON_INJURED_PLAYER_TS_PCT_BEFORE_TEAM_AWAY`` and eleven siblings).
#: The prefixes are anchored at the start of the name, so the legitimate
#: ``TOTAL_INJURED_PLAYER_*`` family -- built from each player's last game
#: STRICTLY BEFORE this one -- is not matched by ``TOTAL_NON_INJURED_``.
ROTATION_LEAK_COLUMN_PREFIXES: tuple[str, ...] = (
    "N_ACTIVE_PLAYERS",
    "TOTAL_NON_INJURED_PLAYER_",
)


def rotation_leak_columns(columns: Iterable[str]) -> list[str]:
    """Which of ``columns`` are rotation-depth leaks. Sorted, for stable reports."""
    return sorted(
        column for column in columns if column.startswith(ROTATION_LEAK_COLUMN_PREFIXES)
    )
