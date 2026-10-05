"""The lineup family at intermediate snapshots: cutoffs, parity and leakage.

* **Cutoffs.** History is read before the earlier of the game date and the
  cutoff's Eastern date, and a cutoff before ``DAY_SETTLED_HOUR_ET`` belongs to
  the previous date. Snapshot UTC cutoffs resolve from embedded timestamps or
  the scoring sidecar and must agree with ``TIPOFF_UTC - horizon``.
* **Parity.** A tip-off snapshot reproduces the closing projection.
* **Leakage.** A horizon reads only the report state at its own cutoff; box
  scores, results and ratings dated on or after its history date never reach it.
* **Provenance.** A build records where its cutoffs came from and a digest of
  the report states it read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.create_training_data.schema_layers.base import LayerContext
from nba_ou.create_training_data.schema_layers.inputs import resolve_snapshot_cutoffs
from nba_ou.data_processing.injury_status.report_state import (
    InjuryReportState,
    report_state_digest,
)
from nba_ou.data_processing.lineups import features
from nba_ou.data_processing.lineups.availability import (
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import (
    RatingBook,
    add_lineup_features,
    walk_forward_offset,
)
from nba_ou.data_processing.lineups.intermediate_features import (
    snapshot_history_dates,
    snapshot_lineup_features,
)

from tests.test_lineup_leakage import (
    DATES,
    PROJECTION_COLUMNS,
    ROSTERS,
    TARGET,
    TARGET_DATE,
    A,
    H,
    _world,
)

EASTERN = "America/New_York"


@pytest.fixture(autouse=True)
def calibrate_from_few_games(monkeypatch):
    """This world has a handful of games; production needs OFFSET_MIN_GAMES."""
    monkeypatch.setattr(features, "OFFSET_MIN_GAMES", 1)


def _eastern(local: str) -> pd.Timestamp:
    return pd.Timestamp(local).tz_localize(EASTERN).tz_convert("UTC")


# ---------------------------------------------------------------------------
# History dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cutoff_et", "history"),
    [
        ("2025-01-15 19:30", "2025-01-15"),  # tip-off
        ("2025-01-15 09:30", "2025-01-15"),  # morning: yesterday has settled
        ("2025-01-15 05:00", "2025-01-15"),
        ("2025-01-15 04:59", "2025-01-14"),  # late games may still be running
        ("2025-01-15 01:30", "2025-01-14"),  # T-1080 of a 19:30 tip
        ("2025-01-14 19:00", "2025-01-14"),  # previous evening
        ("2025-01-14 02:00", "2025-01-13"),
    ],
)
def test_history_date_holds_back_the_cutoff_day_and_an_unsettled_previous_day(
    cutoff_et, history
):
    rows = pd.DataFrame(
        {"GAME_DATE": ["2025-01-15"], "SNAPSHOT_TS_UTC": [_eastern(cutoff_et)]}
    )
    assert snapshot_history_dates(rows).iloc[0] == pd.Timestamp(history)


def test_history_date_requires_a_valid_cutoff():
    rows = pd.DataFrame({"GAME_DATE": ["2025-01-15"], "SNAPSHOT_TS_UTC": [None]})
    with pytest.raises(ValueError, match="UTC cutoffs"):
        snapshot_history_dates(rows)


def test_offset_reads_only_results_dated_before_the_cutoff_date():
    dates = pd.Series(pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15"]))
    raw = pd.Series([200.0, 200.0, 200.0])
    actual = pd.Series([210.0, 220.0, 230.0])
    cutoffs = pd.Series(pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-14"]))
    offset = walk_forward_offset(dates, raw, actual, cutoff_dates=cutoffs, min_games=1)
    # The 15th's snapshot was taken on the 14th: the 14th's result is unknown.
    assert np.isnan(offset.iloc[0])
    assert offset.iloc[1:].tolist() == [10.0, 10.0]


# ---------------------------------------------------------------------------
# Cutoff resolution
# ---------------------------------------------------------------------------

TIP = pd.Timestamp("2025-01-16 00:30", tz="UTC")


def _keys(**extra):
    return pd.DataFrame(
        {"GAME_ID": [22400001, 22400001], "TIME_TO_MATCH_MIN": [0, 600], **extra}
    )


def test_cutoffs_resolve_from_embedded_tipoffs():
    cutoffs = resolve_snapshot_cutoffs(_keys(TIPOFF_UTC=[TIP, TIP]))
    assert cutoffs["game_id"].tolist() == ["0022400001"] * 2
    assert cutoffs["snapshot_minutes"].tolist() == [0, 600]
    assert cutoffs["as_of"].tolist() == [TIP, TIP - pd.Timedelta(minutes=600)]


def test_cutoffs_resolve_from_the_sidecar_whatever_its_row_order():
    sidecar = pd.DataFrame(
        {
            "GAME_ID": ["0022400001", "0022400001"],
            "TIME_TO_MATCH_MIN": [600, 0],
            "TIPOFF_UTC": [TIP, TIP],
            "SNAPSHOT_TS_UTC": [TIP - pd.Timedelta(minutes=600), TIP],
        }
    )
    cutoffs = resolve_snapshot_cutoffs(_keys(), sidecar)
    assert cutoffs["as_of"].tolist() == [TIP, TIP - pd.Timedelta(minutes=600)]


def test_a_row_missing_from_the_sidecar_is_refused():
    sidecar = pd.DataFrame(
        {"GAME_ID": ["0022400001"], "TIME_TO_MATCH_MIN": [0], "TIPOFF_UTC": [TIP]}
    )
    with pytest.raises(ValueError, match="Missing or invalid TIPOFF_UTC"):
        resolve_snapshot_cutoffs(_keys(), sidecar)


def test_a_snapshot_time_that_is_not_tipoff_minus_horizon_is_refused():
    frame = _keys(TIPOFF_UTC=[TIP, TIP], SNAPSHOT_TS_UTC=[TIP, TIP])
    with pytest.raises(ValueError, match="tipoff minus horizon"):
        resolve_snapshot_cutoffs(frame)


def test_input_and_sidecar_must_agree():
    sidecar = _keys(TIPOFF_UTC=[TIP, TIP])
    frame = _keys(TIPOFF_UTC=[TIP, TIP + pd.Timedelta(minutes=1)])
    with pytest.raises(ValueError, match="disagree on TIPOFF_UTC"):
        resolve_snapshot_cutoffs(frame, sidecar)


def test_snapshots_implying_two_tipoffs_for_one_game_are_refused():
    frame = _keys(SNAPSHOT_TS_UTC=[TIP, TIP])
    with pytest.raises(ValueError, match="conflicting tipoff"):
        resolve_snapshot_cutoffs(frame)


@pytest.mark.parametrize("horizons", [[0, 0], [0, -30], [0, 30.5]])
def test_horizons_must_be_unique_nonnegative_integers(horizons):
    frame = pd.DataFrame(
        {"GAME_ID": ["0022400001"] * 2, "TIME_TO_MATCH_MIN": horizons}
    ).assign(TIPOFF_UTC=TIP)
    with pytest.raises(ValueError, match="Snapshot"):
        resolve_snapshot_cutoffs(frame)


# ---------------------------------------------------------------------------
# The family at snapshots
# ---------------------------------------------------------------------------

#: T-0 and T-60 have game-day history; T-1080 of a 19:30 tip is 01:30 ET, so
#: its history ends the day before yesterday.
HORIZONS = (0, 60, 1080)
PREVIOUS_DATE = DATES[-2]


def _snapshots(merged: pd.DataFrame) -> pd.DataFrame:
    tips = pd.to_datetime(merged["GAME_DATE"]).map(
        lambda date: _eastern(f"{date:%Y-%m-%d} 19:30")
    )
    rows = [
        merged.assign(
            TIME_TO_MATCH_MIN=horizon,
            SNAPSHOT_TS_UTC=tips - pd.Timedelta(minutes=horizon),
        )
        for horizon in HORIZONS
    ]
    return pd.concat(rows, ignore_index=True)


def _status(game, team, player, status="out", category="injury"):
    return {
        "game_id": game,
        "team_id": team,
        "player_id": player,
        "status": status,
        "reason_category": category,
        "reason_detail": "",
        "game_date": pd.Timestamp(TARGET_DATE),
    }


def _state(merged, statuses, unfiled=()) -> InjuryReportState:
    empty = InjuryReportState.empty()
    filings = pd.DataFrame(
        [
            {
                "game_id": row.GAME_ID,
                "team_id": team,
                "submitted": (row.GAME_ID, team) not in unfiled,
            }
            for row in merged.itertuples(index=False)
            for team in (row.TEAM_ID_TEAM_HOME, row.TEAM_ID_TEAM_AWAY)
        ]
    )
    return InjuryReportState(
        statuses=pd.DataFrame(statuses, columns=empty.statuses.columns),
        filings=filings,
        report_age=empty.report_age,
        status_events=empty.status_events,
        listed_pairs=empty.listed_pairs,
    )


#: The star is ruled out only on the tip-off report.
EARLY = [_status(TARGET, A, ROSTERS[A][1], status="questionable")]
LATE = [_status(TARGET, H, ROSTERS[H][0]), _status(TARGET, A, ROSTERS[A][1])]


def _states(merged, late=LATE, unfiled_at_60=()):
    return {
        0: _state(merged, late),
        60: _state(merged, EARLY, unfiled=unfiled_at_60),
        1080: _state(merged, EARLY),
    }


def _target(box, merged, ratings, states) -> pd.DataFrame:
    snapshots = _snapshots(merged)
    out = snapshot_lineup_features(snapshots, box, RatingBook(ratings), states)
    rows = snapshots["GAME_ID"].eq(TARGET).to_numpy()
    return out.loc[rows, PROJECTION_COLUMNS].set_axis(
        snapshots.loc[rows, "TIME_TO_MATCH_MIN"]
    )


def test_a_tipoff_snapshot_reproduces_the_closing_projection():
    box, merged, ratings = _world()
    statuses = _state(merged, LATE).statuses
    closing = add_lineup_features(
        merged,
        box,
        RatingBook(ratings),
        player_out_probabilities(statuses),
        excluded=roster_exclusions(statuses),
        report_covered=_state(merged, LATE).covered,
    )
    snapshots = _snapshots(merged)
    out = snapshot_lineup_features(snapshots, box, RatingBook(ratings), _states(merged))
    at_tip = snapshots["TIME_TO_MATCH_MIN"].eq(0).to_numpy()
    pd.testing.assert_frame_equal(
        out.loc[at_tip, PROJECTION_COLUMNS].reset_index(drop=True),
        closing[PROJECTION_COLUMNS].reset_index(drop=True),
    )


def test_each_horizon_reads_only_its_own_report():
    box, merged, ratings = _world()
    values = _target(box, merged, ratings, _states(merged))
    assert values.notna().all().all()
    assert values.loc[0, "LU_ABSENCE_IMPACT_PTS_BEFORE"] != pytest.approx(
        values.loc[60, "LU_ABSENCE_IMPACT_PTS_BEFORE"]
    )
    # A different tip-off report cannot move an earlier snapshot.
    other = _target(box, merged, ratings, _states(merged, late=EARLY))
    pd.testing.assert_series_equal(other.loc[60], values.loc[60])
    pd.testing.assert_series_equal(other.loc[1080], values.loc[1080])
    assert not other.loc[0].equals(values.loc[0])


def test_a_team_without_a_filed_report_blanks_only_that_snapshot():
    box, merged, ratings = _world()
    values = _target(box, merged, ratings, _states(merged, unfiled_at_60={(TARGET, A)}))
    assert values.loc[60].isna().all()
    assert values.loc[[0, 1080]].notna().all().all()


def test_nothing_from_the_unsettled_previous_day_reaches_a_small_hours_snapshot():
    box, merged, ratings = _world()
    states = _states(merged)
    baseline = _target(box, merged, ratings, states)

    previous = box["GAME_DATE"].eq(pd.Timestamp(PREVIOUS_DATE))
    box.loc[previous, "MIN"] = box.loc[previous, "MIN"][::-1].to_numpy() * 3.0
    merged.loc[merged["GAME_DATE"].eq(pd.Timestamp(PREVIOUS_DATE)), "TOTAL_POINTS"] = (
        400.0
    )
    # Fits dated after the previous date are the first to include its games.
    late_fits = ratings["as_of_date"].gt(pd.Timestamp(PREVIOUS_DATE))
    ratings.loc[late_fits, "o_rating"] += 5.0
    perturbed = _target(box, merged, ratings, states)

    pd.testing.assert_series_equal(perturbed.loc[1080], baseline.loc[1080])
    # The same perturbation is visible to a game-day snapshot.
    assert not perturbed.loc[0].equals(baseline.loc[0])


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_the_report_digest_ignores_row_order_but_not_content():
    _, merged, _ = _world()
    state = _state(merged, LATE)
    shuffled = _state(merged, LATE[::-1])
    shuffled.filings = shuffled.filings.iloc[::-1]
    assert report_state_digest({0: state}) == report_state_digest({0: shuffled})
    assert report_state_digest({0: state}) != report_state_digest(
        {0: _state(merged, EARLY)}
    )
    assert report_state_digest({0: state}) != report_state_digest({60: state})


def test_the_context_records_where_cutoffs_and_reports_came_from():
    _, merged, _ = _world()
    snapshots = _snapshots(merged)
    ctx = LayerContext(snapshot_injury_states=_states(merged))
    cutoffs = ctx.snapshot_cutoffs(snapshots)
    ctx.snapshot_reports(cutoffs)
    assert ctx.provenance["snapshot_time_source"] == "embedded"
    assert ctx.provenance["snapshot_report_states_sha256"] == report_state_digest(
        _states(merged)
    )


def test_a_file_build_context_refuses_the_live_schedule():
    ctx = LayerContext(allow_schedule_tipoffs=False)
    keys = pd.DataFrame({"GAME_ID": ["0022400001"], "TIME_TO_MATCH_MIN": [0]})
    with pytest.raises(ValueError, match="scoring sidecar"):
        ctx.snapshot_cutoffs(keys)
    assert "snapshot_time_source" not in ctx.provenance
