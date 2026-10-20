"""Pin which referee columns each referee_tendencies_2026_09 cell trains on.

The cells differ only by ``cleaning.exclude_cols_containing`` substrings. A
typo there does not error -- the column simply stays in -- so the design is
checked against the real matcher and the real feature names.
"""

from pathlib import Path

import pandas as pd
import pytest
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    _get_cols_matching_patterns,
)
from nba_ou.data_processing.referees.referee_tendencies import (
    DEFAULT_REFEREE_TENDENCY_SPECS,
    FTA_X_ABS_SPREAD_FEATURE,
    FTA_X_EXPECTED_TOTAL_FTA_FEATURE,
    FTA_X_HOME_FTA_RATE_EDGE_FEATURE,
    POSS_X_ABS_SPREAD_FEATURE,
    PRIOR_GAMES_FEATURE,
    TRACK_SPREAD,
    TRACK_TOTALS,
    UNKNOWN_COUNT_FEATURE,
)

from training_pipeline.cli import load_config

CAMPAIGN = (
    Path(__file__).resolve().parents[1] / "experiments" / "referee_tendencies_2026_09"
)

LEGACY = {
    f"REF_{agg}_{metric}_DIFF_BEFORE"
    for agg in ("AVG", "STD", "SUM")
    for metric in ("TOTAL_POINTS", "DIFF_FROM_LINE", "TOTAL_PF")
} | {"STYLE_FTA_REFEREE_INTERACTION_BEFORE"}
TOTALS = {s.feature for s in DEFAULT_REFEREE_TENDENCY_SPECS if s.track == TRACK_TOTALS}
SPREAD = {s.feature for s in DEFAULT_REFEREE_TENDENCY_SPECS if s.track == TRACK_SPREAD}
SS_TOTALS = {
    s.same_season_feature()
    for s in DEFAULT_REFEREE_TENDENCY_SPECS
    if s.track == TRACK_TOTALS
}
SS_SPREAD = {
    s.same_season_feature()
    for s in DEFAULT_REFEREE_TENDENCY_SPECS
    if s.track == TRACK_SPREAD
}
COUNTS = {PRIOR_GAMES_FEATURE, UNKNOWN_COUNT_FEATURE}
X_TOTALS = {FTA_X_EXPECTED_TOTAL_FTA_FEATURE, FTA_X_ABS_SPREAD_FEATURE}
X_SPREAD = {POSS_X_ABS_SPREAD_FEATURE, FTA_X_HOME_FTA_RATE_EDGE_FEATURE}
UNIVERSE = (
    LEGACY | TOTALS | SPREAD | SS_TOTALS | SS_SPREAD | COUNTS | X_TOTALS | X_SPREAD
)
# Non-referee columns whose names must survive every cell.
BYSTANDERS = {
    "STYLE_EXPECTED_TOTAL_FTA_BEFORE",
    "ODDS_SPREAD_bet365",
    "TOTAL_POINTS",
    "SPREAD_ERROR",
}

EXPECTED_KEPT = {
    "t_a_line_error_no_referee": set(),
    "t_b_line_error_legacy_referee": LEGACY,
    "t_c_line_error_crew_same_season": SS_TOTALS | COUNTS,
    "t_d_line_error_crew_decay": TOTALS | COUNTS,
    "t_e_line_error_crew_decay_interactions": TOTALS | COUNTS | X_TOTALS,
    "t_f_total_points_no_referee": set(),
    "t_g_total_points_crew_decay": TOTALS | COUNTS,
    "s_a_spread_no_referee": set(),
    "s_b_spread_legacy_referee": LEGACY,
    "s_c_spread_totals_track_only": TOTALS | COUNTS,
    "s_d_spread_both_tracks": TOTALS | SPREAD | COUNTS,
    "s_e_spread_both_tracks_interactions": TOTALS
    | SPREAD
    | COUNTS
    | X_TOTALS
    | X_SPREAD,
}


def test_every_campaign_file_is_pinned():
    assert {p.stem for p in CAMPAIGN.glob("*.yaml")} == set(EXPECTED_KEPT)


@pytest.mark.parametrize("stem", sorted(EXPECTED_KEPT))
def test_cell_keeps_exactly_its_referee_columns(stem):
    config = load_config(str(CAMPAIGN / f"{stem}.yaml"))
    frame = pd.DataFrame(columns=sorted(UNIVERSE | BYSTANDERS))
    dropped = _get_cols_matching_patterns(
        frame, config.cleaning.exclude_cols_containing
    )

    assert not (dropped & BYSTANDERS)
    assert UNIVERSE - dropped == EXPECTED_KEPT[stem]


def test_cells_share_everything_but_the_referee_exclusions():
    def reduced(stem):
        dump = load_config(str(CAMPAIGN / f"{stem}.yaml")).model_dump(mode="json")
        dump["cleaning"].pop("exclude_cols_containing")
        for label in ("experiment_name", "hypothesis"):
            dump.pop(label)
        return dump

    by_strategy = {}
    for stem in EXPECTED_KEPT:
        dump = reduced(stem)
        by_strategy.setdefault(dump["prediction_strategy"], []).append(dump)
    for dumps in by_strategy.values():
        assert all(d == dumps[0] for d in dumps[1:])
