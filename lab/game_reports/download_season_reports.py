"""Download or inspect the NBA.com written-report archive of one season.

``download`` starts or resumes the selected season; ``status`` is local-only.
No TXT or CSV copies are produced.

    python lab/game_reports/download_season_reports.py download 2025-26
    python lab/game_reports/download_season_reports.py status 2025
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab.game_reports.season_reports import (  # noqa: E402
    DEFAULT_PHASES,
    PHASES,
    S3ReportStore,
    SeasonConfig,
    download_season,
    prepare_season,
    summarize,
)

LOCAL_ROOT = PROJECT_ROOT / "data" / "nba_game_reports" / "seasons"
S3_PREFIX = "train_data/game_reports/v1"


def season_year(value: str) -> int:
    """Accept a starting year (2025) or a season label (2025-26)."""
    match = re.fullmatch(r"(\d{4})(?:-(\d{2}))?", value)
    if not match:
        raise argparse.ArgumentTypeError("use a starting year (2025) or 2025-26")
    year = int(match.group(1))
    if match.group(2) and match.group(2) != f"{(year + 1) % 100:02d}":
        raise argparse.ArgumentTypeError(f"{value} is not one season")
    return year


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("download", "status"))
    parser.add_argument(
        "season", type=season_year, help="starting year (2025) or season (2025-26)"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=sorted(PHASES.values()),
        default=list(DEFAULT_PHASES),
        help="game phases to archive (an existing manifest refuses a change)",
    )
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.35,
        help="minimum seconds between NBA.com requests across workers",
    )
    parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help="also recheck games whose written report was absent earlier",
    )
    parser.add_argument(
        "--max-games", type=positive_int, help="bounded pilot run (default: all)"
    )
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--s3-prefix", default=S3_PREFIX)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Imported here so --help works without the repository settings.
    from nba_ou.config.settings import SETTINGS

    config = SeasonConfig(
        season_year=args.season,
        local_root=args.local_root,
        bucket=SETTINGS.s3_bucket,
        prefix=args.s3_prefix,
        phases=tuple(args.phases),
        workers=args.workers,
        request_interval=args.request_interval,
    )
    if args.mode == "status":
        path = config.directory / "manifest.json"
        if not path.is_file():
            print(f"No local manifest for {config.season}: {path}", file=sys.stderr)
            return 1
        print(json.dumps(summarize(json.loads(path.read_bytes())), indent=2))
        return 0

    manifest = prepare_season(config)
    from nba_ou.utils.s3_models import make_s3_client

    client = make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    )
    manifest = download_season(
        config,
        manifest,
        S3ReportStore(client, config),
        retry_unavailable=args.retry_unavailable,
        max_games=args.max_games,
    )
    # Unavailable and unfinished games are expected; failed work is not.
    failed = [
        status
        for status in manifest["summary"]["by_status"]
        if status.endswith("error")
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
