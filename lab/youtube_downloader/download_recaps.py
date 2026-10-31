"""Download NBA recap audio for one season or US calendar date.

    python lab/youtube_downloader/download_recaps.py download 2025-26
    python lab/youtube_downloader/download_recaps.py download --date 2026-10-28
    python lab/youtube_downloader/download_recaps.py status 2026-27

Audio is always Opus, mono, 48 kbit/s VBR. Downloads use the configured S3 bucket
unless --local-only is supplied. A date run shares the full season manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nba_ou.utils.general_utils import get_season_year_from_date  # noqa: E402

from lab.game_reports.download_season_reports import (  # noqa: E402
    positive_int,
    season_year,
)
from lab.game_reports.season_reports import DEFAULT_PHASES, PHASES  # noqa: E402
from lab.youtube_downloader.recap_audio import (  # noqa: E402
    AudioConfig,
    S3AudioStore,
    require_tools,
    run_archive,
    summarize,
)

LOCAL_ROOT = PROJECT_ROOT / "data" / "nba_game_recaps" / "seasons"


def calendar_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "use a US game date in YYYY-MM-DD format"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("download", "status"))
    parser.add_argument(
        "season",
        nargs="?",
        type=season_year,
        help="starting year (2025) or label (2025-26); inferred from --date if omitted",
    )
    parser.add_argument(
        "--date",
        type=calendar_date,
        help="only process this US game date, keeping the full season inventory",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="save locally without AWS settings or S3 access",
    )
    parser.add_argument(
        "--max-games",
        type=positive_int,
        help="limit games processed, without shrinking the manifest",
    )
    parser.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
    parser.add_argument("--s3-prefix", default="train_data/game_recaps/v1")
    parser.add_argument(
        "--language",
        default="en",
        help="audio language (default: en); another language requires a separate destination",
    )
    parser.add_argument("--js-runtime", choices=("node", "deno"), default="node")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=2.0,
        help="minimum seconds between YouTube operations (default: 2)",
    )
    parser.add_argument("--search-results", type=positive_int, default=25)
    parser.add_argument(
        "--upload-window-days",
        type=int,
        default=2,
        help="accept uploads up to this many days after the game (default: 2)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=sorted(PHASES.values()),
        default=list(DEFAULT_PHASES),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.season is None and args.date is None:
        parser.error("select a season or --date YYYY-MM-DD")
    year = (
        args.season if args.season is not None else get_season_year_from_date(args.date)
    )
    season = f"{year}-{(year + 1) % 100:02d}"
    if args.mode == "status":
        path = args.local_root / f"season={season}" / "manifest.json"
        if not path.is_file():
            print(f"No local manifest: {path}", file=sys.stderr)
            return 1
        print(json.dumps(summarize(json.loads(path.read_bytes()), args.date), indent=2))
        return 0

    # Local-only mode and status work without loading credentials.
    bucket, client = "", None
    if not args.local_only:
        from nba_ou.config.settings import SETTINGS
        from nba_ou.utils.s3_models import make_s3_client

        bucket = SETTINGS.s3_bucket
        if not bucket:
            parser.error("configure an S3 bucket or use --local-only")
        client = make_s3_client(
            profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        )
    try:
        config = AudioConfig(
            year,
            args.local_root,
            bucket,
            prefix=args.s3_prefix,
            phases=tuple(args.phases),
            language=args.language,
            request_interval=args.request_interval,
            search_results=args.search_results,
            upload_window_days=args.upload_window_days,
            js_runtime=args.js_runtime,
        )
        require_tools(config)
        manifest = run_archive(
            config,
            game_date=args.date,
            max_games=args.max_games,
            store=S3AudioStore(client, config) if client else None,
        )
        return int(
            any(
                status.endswith("error")
                for status in summarize(manifest, args.date)["by_status"]
            )
        )
    except KeyboardInterrupt:
        print(
            "Interrupted; progress was checkpointed. Re-run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
