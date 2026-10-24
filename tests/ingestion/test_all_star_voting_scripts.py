import pandas as pd

from scripts.all_star_voting.manage_all_star_voting import (
    build_available_input_csv,
    find_available_season_csvs,
    prepare_from_available_data,
)


def test_build_available_input_csv_uses_available_season_files(tmp_path):
    data_dir = tmp_path / "all_star_voting"
    for year, player in [(2026, "Player One"), (2027, "Player Two")]:
        season_dir = data_dir / str(year)
        season_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "conference": ["Western Conference"],
                "position": ["All Positions"],
                "season": [f"{year - 1}-{str(year)[-2:]}"],
                "player_name": [player],
                "fan_votes": [1],
                "fan_rank": [1],
                "player_votes": [0],
                "player_rank": [1],
                "media_votes": [0],
                "media_rank": [1],
                "score": [1.0],
            }
        ).to_csv(season_dir / "all_conferences.csv", index=False)

    output_csv = tmp_path / "combined.csv"
    result_path = build_available_input_csv(
        data_dir=data_dir,
        output_csv=output_csv,
        overwrite=True,
    )

    assert result_path == output_csv
    assert find_available_season_csvs(data_dir, start_year=2027) == [
        data_dir / "2027/all_conferences.csv"
    ]

    combined = pd.read_csv(output_csv)
    assert combined["player_name"].tolist() == ["Player One", "Player Two"]


def test_prepare_from_available_data_reuses_existing_combined_csv(tmp_path):
    output_csv = tmp_path / "combined.csv"
    output_csv.write_text("conference,position,season,player_name\n")

    result = prepare_from_available_data(
        input_csv=None,
        data_dir=tmp_path / "missing_data_dir",
        output_csv=output_csv,
        start_year=None,
        end_year=None,
        overwrite_output=False,
    )

    assert result == output_csv
