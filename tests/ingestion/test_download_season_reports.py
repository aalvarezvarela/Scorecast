"""Command-line selection of the season to archive."""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lab.game_reports import download_season_reports as cli


@pytest.mark.parametrize("value", ["2025", "2025-26"])
def test_season_accepts_starting_year_or_label(value):
    assert cli.season_year(value) == 2025


@pytest.mark.parametrize("value", ["2025-27", "25", "2025/26", "1999-00x"])
def test_season_rejects_anything_but_one_season(value):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.season_year(value)


def test_century_rollover_label():
    assert cli.season_year("2099-00") == 2099


def test_parser_defaults_and_mode():
    args = cli.build_parser().parse_args(["download", "2024-25", "--max-games", "5"])
    assert (args.mode, args.season, args.max_games) == ("download", 2024, 5)
    assert tuple(args.phases) == cli.DEFAULT_PHASES
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["download", "2024", "--max-games", "0"])


@pytest.mark.parametrize(
    "argv", [["estimate", "2024"], ["download", "2024", "--sample-size", "2"]]
)
def test_removed_estimate_options_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_download_starts_without_an_estimate(monkeypatch, tmp_path):
    from nba_ou.config import settings
    from nba_ou.utils import s3_models

    monkeypatch.setattr(
        settings,
        "SETTINGS",
        SimpleNamespace(
            s3_bucket="test-bucket",
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
        ),
    )
    manifest = {"summary": {"by_status": {"uploaded": 1}}}
    prepare = MagicMock(return_value=manifest)
    download = MagicMock(return_value=manifest)
    client, store = object(), object()
    monkeypatch.setattr(cli, "prepare_season", prepare)
    monkeypatch.setattr(cli, "download_season", download)
    monkeypatch.setattr(s3_models, "make_s3_client", MagicMock(return_value=client))
    monkeypatch.setattr(cli, "S3ReportStore", lambda s3, config: store)

    assert cli.main(["download", "2024", "--local-root", str(tmp_path)]) == 0

    config = prepare.call_args.args[0]
    assert config.season_year == 2024
    download.assert_called_once_with(
        config, manifest, store, retry_unavailable=False, max_games=None
    )
