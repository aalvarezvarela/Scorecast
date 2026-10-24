import pandas as pd
import pytest
from nba_ou.postgre_db.line_history_aiven import transform as tf

TIPOFF = pd.Timestamp("2024-11-15 00:30:00", tz="UTC")  # 2024-11-14 19:30 ET


def _identity(name):
    return name


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022400123"],
            "game_date": [pd.Timestamp("2024-11-14").date()],
            "team_home": ["Indiana Pacers"],
            "team_away": ["Miami Heat"],
        }
    )


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022400123"],
            "game_date": [pd.Timestamp("2024-11-14").date()],
            "tipoff_utc": [TIPOFF],
            "team_home": ["IND"],
            "team_away": ["MIA"],
        }
    )


def _raw(**overrides) -> pd.DataFrame:
    row = {
        "game_date": pd.Timestamp("2024-11-14").date(),
        "season_year": 2024,
        "event_id": 315429,
        "team_home": "Indiana Pacers",
        "team_away": "Miami Heat",
        "bookmaker_slug": "betmgm",
        "market": "totals",
        "row_kind": "history",
        # 20:00 Madrid on 2024-11-14 == 19:00 UTC == 90 min before tipoff
        "timestamp": pd.Timestamp("2024-11-14 20:00:00"),
        "left_line": 224.5,
        "left_price": -110,
        "right_line": 224.5,
        "right_price": -110,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _run(raw: pd.DataFrame):
    return tf.transform_season(
        raw,
        season_year=2024,
        games=_games(),
        schedule=_schedule(),
        book_ids={"betmgm": 2},
        market_ids={"totals": 1, "point_spread": 2, "money_line": 3},
        normalize_team=_identity,
    )


def test_lines_are_stored_as_doubled_half_points():
    rows, _dim, _stats = _run(_raw())

    assert rows.loc[0, "left_line"] == 449  # 224.5 * 2
    assert rows.loc[0, "right_line"] == 449


def test_moneyline_rows_keep_null_lines():
    rows, _dim, _stats = _run(
        _raw(market="money_line", left_line=None, right_line=None)
    )

    assert pd.isna(rows.loc[0, "left_line"])
    assert rows.loc[0, "left_price"] == -110
    assert rows.loc[0, "market_id"] == 3


def test_off_the_board_sentinel_becomes_null():
    rows, _dim, _stats = _run(_raw(left_price=tf.NO_PRICE_SENTINEL))

    assert pd.isna(rows.loc[0, "left_price"])
    assert rows.loc[0, "right_price"] == -110


def test_mins_to_tip_uses_madrid_not_utc():
    """20:00 Madrid on 14 Nov is 19:00 UTC, 330 min before a 00:30 UTC tipoff.

    Reading the same naive value as UTC would give -270, so this pins the zone.
    """
    rows, _dim, _stats = _run(_raw())

    assert rows.loc[0, "mins_to_tip"] == -330
    assert bool(rows.loc[0, "is_pregame"]) is True


def test_in_play_rows_are_kept_but_flagged():
    # 02:00 Madrid on the 15th == 01:00 UTC == 30 min after tipoff
    rows, _dim, _stats = _run(_raw(timestamp=pd.Timestamp("2024-11-15 02:00:00")))

    assert rows.loc[0, "mins_to_tip"] == 30
    assert bool(rows.loc[0, "is_pregame"]) is False


def test_opener_is_flagged():
    rows, _dim, _stats = _run(_raw(row_kind="opener"))

    assert bool(rows.loc[0, "is_opener"]) is True


def test_preseason_rows_are_dropped_with_a_reason():
    raw = _raw(team_home="Los Angeles Lakers")  # not in the games table
    rows, _dim, stats = _run(raw)

    assert rows.empty
    assert stats.dropped["preseason_or_unmatched_game"] == 1


def test_duplicate_timepoints_collapse():
    raw = pd.concat([_raw(), _raw()], ignore_index=True)
    rows, _dim, stats = _run(raw)

    assert len(rows) == 1
    assert stats.dropped["duplicate_timepoint"] == 1


def test_rows_in_the_dst_repeated_hour_are_dropped_not_guessed():
    # 02:30 on 2024-10-27 occurs twice in Europe/Madrid.
    raw = _raw(
        game_date=pd.Timestamp("2024-10-27").date(),
        timestamp=pd.Timestamp("2024-10-27 02:30:00"),
    )
    games = _games().assign(game_date=[pd.Timestamp("2024-10-27").date()])
    schedule = _schedule().assign(game_date=[pd.Timestamp("2024-10-27").date()])

    rows, _dim, stats = tf.transform_season(
        raw,
        season_year=2024,
        games=games,
        schedule=schedule,
        book_ids={"betmgm": 2},
        market_ids={"totals": 1},
        normalize_team=_identity,
    )

    assert rows.empty
    assert stats.dropped["dst_ambiguous_or_nonexistent"] == 1


def test_game_dimension_pairs_game_id_with_its_own_event_id():
    rows, dim, _stats = _run(_raw())

    assert len(dim) == 1
    assert dim.loc[0, "game_id"] == "0022400123"
    assert dim.loc[0, "event_id"] == 315429
    assert dim.loc[0, "tipoff_utc"] == TIPOFF
    assert set(rows["game_id"]) == set(dim["game_id"])


def test_pickem_spread_price_is_not_stored_as_a_line():
    """SBR renders a pick'em as a bare price; the scraper read it as the spread."""
    rows, _dim, stats = _run(
        _raw(
            market="point_spread",
            left_line=-110,
            left_price=None,
            right_line=-110,
            right_price=None,
        )
    )

    assert pd.isna(rows.loc[0, "left_line"])
    assert pd.isna(rows.loc[0, "right_line"])
    assert rows.loc[0, "left_price"] == -110
    assert rows.loc[0, "right_price"] == -110
    assert stats.repaired["spread_price_bleed"] == 1


def test_complementary_pickem_prices_are_relabelled():
    rows, _dim, _stats = _run(
        _raw(
            market="point_spread",
            left_line=-115,
            left_price=None,
            right_line=-105,
            right_price=None,
        )
    )

    assert rows.loc[0, "left_price"] == -115
    assert rows.loc[0, "right_price"] == -105


def test_genuine_mirrored_spread_is_left_alone():
    rows, _dim, stats = _run(
        _raw(
            market="point_spread",
            left_line=-5.5,
            left_price=-110,
            right_line=5.5,
            right_price=-110,
        )
    )

    assert rows.loc[0, "left_line"] == -11  # -5.5 * 2
    assert rows.loc[0, "right_line"] == 11
    assert rows.loc[0, "left_price"] == -110
    assert "spread_price_bleed" not in stats.repaired


def test_totals_are_never_treated_as_price_bleed():
    """Totals quote the same number on both sides, so the mirror test must not fire."""
    rows, _dim, stats = _run(_raw(market="totals", left_line=224.5, right_line=224.5))

    assert rows.loc[0, "left_line"] == 449
    assert "spread_price_bleed" not in stats.repaired


def test_impossible_pregame_total_is_cleared_but_prices_kept():
    """2285 is 228.5 with the decimal lost; it must not reach a feature.

    Prices survive so the timepoint itself is not lost.
    """
    rows, _dim, stats = _run(_raw(left_line=2285, right_line=2285))

    assert pd.isna(rows.loc[0, "left_line"])
    assert rows.loc[0, "left_price"] == -110
    assert stats.repaired["implausible_pregame_line"] == 1


def test_in_play_blowout_spread_is_not_clipped():
    """A live spread past 30 is real, so the plausibility guard must skip it."""
    rows, _dim, stats = _run(
        _raw(
            market="point_spread",
            timestamp=pd.Timestamp("2024-11-15 02:00:00"),  # after tipoff
            left_line=-34.5,
            right_line=34.5,
        )
    )

    assert bool(rows.loc[0, "is_pregame"]) is False
    assert rows.loc[0, "left_line"] == -69  # -34.5 * 2, untouched
    assert "implausible_pregame_line" not in stats.repaired


def test_normal_pregame_total_is_untouched():
    rows, _dim, stats = _run(_raw(left_line=224.5, right_line=224.5))

    assert rows.loc[0, "left_line"] == 449
    assert "implausible_pregame_line" not in stats.repaired


def test_empty_csv_frames_do_not_break_reading(tmp_path):
    season = tmp_path / "2018-19" / "line_history"
    season.mkdir(parents=True)
    (season / "empty.csv").write_text(",".join(tf.READ_COLUMNS) + "\n")

    assert tf.read_season_csvs(tmp_path, "2018-19").empty


def test_low_confidence_seasons_are_marked():
    assert tf.season_timezone(2020)[1] == "low"
    assert tf.season_timezone(2024) == (tf.MADRID, "high")


@pytest.mark.parametrize("value", [-32769 / 2, 32768 / 2])
def test_out_of_range_lines_become_null(value):
    rows, _dim, _stats = _run(_raw(left_line=value))

    assert pd.isna(rows.loc[0, "left_line"])
