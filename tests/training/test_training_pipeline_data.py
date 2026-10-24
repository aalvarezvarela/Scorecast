import pandas as pd
import pytest
from nba_ou.config.odds_columns import total_line_col

from training_pipeline.config import DEFAULT_ALLOWED_SEASON_TYPES, BaselineConfig
from training_pipeline.data import (
    BOOKMAKER_MEDIAN_LINE_COL,
    ensure_line_error_column,
    filter_allowed_season_types,
    resolve_baseline_line_col,
    resolve_season_type,
)


def _season_type_frame() -> pd.DataFrame:
    """One row per competition type, with GAME_ID prefixes per SEASON_TYPE_MAP.

    The SEASON_TYPE text column deliberately mislabels the play-in game as
    "Playoffs", mirroring the real training data.
    """
    return pd.DataFrame(
        {
            "GAME_ID": ["0022500001", "0042400301", "0052400101", "0062300001"],
            "SEASON_TYPE": ["Regular Season", "Playoffs", "Playoffs", "In-Season Final Game"],
            "TOTAL_POINTS": [225.0, 214.0, 220.0, 216.0],
        }
    )


def test_resolve_season_type_reads_the_game_id_prefix():
    resolved = resolve_season_type(_season_type_frame())
    assert list(resolved) == [
        "Regular Season",
        "Playoffs",
        "Play-In Tournament",
        "In-Season Final Game",
    ]


def test_resolve_season_type_requires_the_game_id_column():
    with pytest.raises(KeyError, match="exclude_playoffs"):
        resolve_season_type(pd.DataFrame({"TOTAL_POINTS": [220.0]}))


def test_default_filter_keeps_regular_season_and_play_in_only():
    kept = filter_allowed_season_types(
        _season_type_frame(), allowed_season_types=DEFAULT_ALLOWED_SEASON_TYPES
    )
    assert list(resolve_season_type(kept)) == ["Regular Season", "Play-In Tournament"]


def test_play_in_games_survive_despite_being_mislabelled_as_playoffs():
    """Regression: the SEASON_TYPE text column calls play-in games "Playoffs",
    so filtering on that column would drop exactly the games we mean to keep.
    Season type must come from the GAME_ID prefix.
    """
    df = _season_type_frame()
    play_in = df[df["GAME_ID"].str.startswith("005")]
    assert play_in["SEASON_TYPE"].iloc[0] == "Playoffs"  # the trap

    kept = filter_allowed_season_types(
        df, allowed_season_types=DEFAULT_ALLOWED_SEASON_TYPES
    )
    assert "0052400101" in set(kept["GAME_ID"])


def test_filter_drops_rows_with_unknown_game_id_prefix():
    df = pd.DataFrame({"GAME_ID": ["0022500001", "9992500001"], "TOTAL_POINTS": [1.0, 2.0]})
    kept = filter_allowed_season_types(
        df, allowed_season_types=DEFAULT_ALLOWED_SEASON_TYPES
    )
    assert list(kept["GAME_ID"]) == ["0022500001"]


def test_filter_can_be_widened_to_include_playoffs():
    kept = filter_allowed_season_types(
        _season_type_frame(),
        allowed_season_types=("Regular Season", "Play-In Tournament", "Playoffs"),
    )
    assert len(kept) == 3


def test_ensure_line_error_column_matches_total_points_minus_line():
    line_col = total_line_col()
    df = pd.DataFrame({"TOTAL_POINTS": [210.0, 220.0], line_col: [205.0, 225.0]})

    result = ensure_line_error_column(df)

    assert "LINE_ERROR" in result.columns
    assert result["LINE_ERROR"].tolist() == pytest.approx([5.0, -5.0])


def test_ensure_line_error_column_is_idempotent_when_already_present():
    df = pd.DataFrame({"TOTAL_POINTS": [210.0], "LINE_ERROR": [1.0]})
    result = ensure_line_error_column(df)
    assert result["LINE_ERROR"].tolist() == [1.0]


def test_ensure_line_error_column_is_a_no_op_when_line_column_missing():
    df = pd.DataFrame({"TOTAL_POINTS": [210.0]})
    result = ensure_line_error_column(df)
    assert "LINE_ERROR" not in result.columns


def test_resolve_baseline_line_col_prefers_explicit_override():
    df = pd.DataFrame({"MY_LINE": [1.0], BOOKMAKER_MEDIAN_LINE_COL: [2.0]})
    col = resolve_baseline_line_col(df, BaselineConfig(line_col="MY_LINE"))
    assert col == "MY_LINE"


def test_resolve_baseline_line_col_raises_when_explicit_override_missing():
    df = pd.DataFrame({total_line_col(): [1.0]})
    with pytest.raises(KeyError):
        resolve_baseline_line_col(df, BaselineConfig(line_col="MISSING_COL"))


def test_resolve_baseline_line_col_prefers_main_total_line_over_cross_book_median_by_default():
    """The cross-book median column is never a silent default: it was found
    empirically (against a real archived training CSV) to sometimes hold
    values that are not points-scale and don't correlate with TOTAL_POINTS.
    It's only used when a caller explicitly opts in via line_col.
    """
    df = pd.DataFrame({BOOKMAKER_MEDIAN_LINE_COL: [2.0], total_line_col(): [1.0]})
    col = resolve_baseline_line_col(df, BaselineConfig())
    assert col == total_line_col()


def test_resolve_baseline_line_col_uses_cross_book_median_only_via_explicit_opt_in():
    df = pd.DataFrame({BOOKMAKER_MEDIAN_LINE_COL: [2.0], total_line_col(): [1.0]})
    col = resolve_baseline_line_col(
        df, BaselineConfig(line_col=BOOKMAKER_MEDIAN_LINE_COL)
    )
    assert col == BOOKMAKER_MEDIAN_LINE_COL


def test_resolve_baseline_line_col_falls_back_to_main_total_line_col():
    df = pd.DataFrame({total_line_col(): [1.0]})
    col = resolve_baseline_line_col(df, BaselineConfig())
    assert col == total_line_col()


def test_resolve_baseline_line_col_supports_reduced_schema_column():
    """The reduced-feature CSV family uses a single unsuffixed line column
    (e.g. TOTAL_OVER_UNDER_LINE) instead of per-book TOTAL_LINE_<book>
    columns -- it must be selectable via an explicit override.
    """
    df = pd.DataFrame({"TOTAL_OVER_UNDER_LINE": [1.0]})
    col = resolve_baseline_line_col(df, BaselineConfig(line_col="TOTAL_OVER_UNDER_LINE"))
    assert col == "TOTAL_OVER_UNDER_LINE"


def test_resolve_baseline_line_col_raises_when_nothing_resolvable():
    df = pd.DataFrame({"SOME_OTHER_COLUMN": [1.0]})
    with pytest.raises(KeyError):
        resolve_baseline_line_col(df, BaselineConfig())
