import pandas as pd
import pytest
from nba_ou.config.constants import SEASON_TYPE_MAP
from nba_ou.utils.general_utils import (
    get_nba_season_nullable_from_date,
    get_season_year_from_date,
)
from nba_ou.utils.seasons import (
    classify_season_type,
    get_all_seasons_from_2006,
    get_season_start_date_n_seasons_back,
    get_seasons_between_dates,
)


@pytest.mark.parametrize(
    ("date", "expected_season_year"),
    [
        ("2026-01-01", 2025),
        ("2026-07-31", 2025),
        ("2026-08-01", 2026),
        ("2026-12-31", 2026),
        ("2020-10-31", 2019),
        ("2020-11-01", 2020),
    ],
)
def test_season_date_helpers_share_the_canonical_boundary(
    date,
    expected_season_year,
):
    expected_label = (
        f"{expected_season_year}-{str(expected_season_year + 1)[-2:]}"
    )

    assert get_season_year_from_date(date) == expected_season_year
    assert get_nba_season_nullable_from_date(date) == expected_label


def test_two_season_july_context_does_not_include_future_season():
    reference_date = pd.Timestamp("2026-07-23")
    current_season_year = get_season_year_from_date(reference_date)
    start_date = pd.Timestamp(
        year=current_season_year - 1,
        month=10,
        day=1,
    )

    assert get_seasons_between_dates(start_date, reference_date) == [
        "2024-25",
        "2025-26",
    ]


def test_all_seasons_and_start_date_use_the_same_july_boundary():
    reference_date = pd.Timestamp("2026-07-23")

    assert get_all_seasons_from_2006(reference_date)[-2:] == [
        "2024-25",
        "2025-26",
    ]
    assert get_season_start_date_n_seasons_back(
        2, reference_date
    ) == pd.Timestamp("2024-10-01")


@pytest.mark.parametrize(
    ("prefix", "expected_type"),
    list(SEASON_TYPE_MAP.items()),
)
def test_season_type_classifier_uses_canonical_mapping(prefix, expected_type):
    assert classify_season_type(f"{prefix}2500001") == expected_type


def test_season_type_classifier_returns_unknown_for_unmapped_prefix():
    assert classify_season_type("9992500001") == "Unknown"

