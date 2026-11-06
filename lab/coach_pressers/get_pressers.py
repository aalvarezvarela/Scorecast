"""Find, match and archive NBA head-coach postgame pressers.

    python lab/coach_pressers/get_pressers.py run 2023-24 --dry-run
    python lab/coach_pressers/get_pressers.py run 2019-20:2025-26 --dry-run
    python lab/coach_pressers/get_pressers.py run 2021-22 --team BOS --dry-run --local-only
    python lab/coach_pressers/get_pressers.py run 2025-26
    python lab/coach_pressers/get_pressers.py status 2025-26
    python lab/coach_pressers/get_pressers.py channels --resolve
    python lab/coach_pressers/get_pressers.py llm-check

Without --local-only the archive uses the configured S3 bucket. --dry-run runs
every metadata step (1-8) and downloads nothing. See
docs/pipeline/coach_pressers.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab.coach_pressers.config import (  # noqa: E402
    CHANNELS_CSV,
    LOCAL_ROOT,
    S3_PREFIX,
    PresserConfig,
)
from lab.coach_pressers.teams import TEAMS  # noqa: E402
from lab.game_reports.download_season_reports import season_year  # noqa: E402
from lab.game_reports.season_reports import DEFAULT_PHASES, PHASES  # noqa: E402


def season_range(value: str) -> list[int]:
    """2023-24, 2023, or an inclusive range 2019-20:2025-26."""
    if ":" not in value:
        return [season_year(value)]
    start, end = (season_year(part) for part in value.split(":", 1))
    if end < start:
        raise argparse.ArgumentTypeError("the range must go forward in time")
    return list(range(start, end + 1))


def team_code(value: str) -> str:
    code = value.upper()
    if code not in TEAMS:
        raise argparse.ArgumentTypeError(f"unknown team {value}")
    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--local-root", type=Path, default=LOCAL_ROOT)
        p.add_argument("--s3-prefix", default=S3_PREFIX)
        p.add_argument("--local-only", action="store_true", help="never touch AWS")
        p.add_argument("--verbose", action="store_true")

    run = sub.add_parser("run", help="run the pipeline for one or more seasons")
    run.add_argument("seasons", type=season_range)
    common(run)
    run.add_argument("--team", type=team_code, action="append", default=[])
    run.add_argument("--dry-run", action="store_true", help="steps 1-8 only")
    run.add_argument(
        "--until-step", type=int, default=10, choices=range(1, 11), metavar="N"
    )
    run.add_argument(
        "--phases",
        nargs="+",
        choices=sorted(PHASES.values()),
        default=list(DEFAULT_PHASES),
    )
    run.add_argument("--no-llm", action="store_true", help="ambiguous cases -> review")
    run.add_argument("--llm-model", default="gpt-6-luna")
    run.add_argument("--llm-budget", type=float, default=5.0, help="USD per run")
    run.add_argument("--captions-only", action="store_true")
    run.add_argument("--no-captions", action="store_true")
    run.add_argument("--keep-local", action="store_true")
    run.add_argument("--js-runtime", choices=("node", "deno"), default="node")

    status = sub.add_parser("status", help="coverage summary from the local manifest")
    status.add_argument("seasons", type=season_range)
    common(status)

    channels = sub.add_parser("channels", help="show or resolve the channel registry")
    channels.add_argument("--resolve", action="store_true")
    channels.add_argument(
        "--refresh", action="store_true", help="re-resolve unverified rows"
    )
    channels.add_argument("--csv", type=Path, default=CHANNELS_CSV)
    common(channels)
    channels.add_argument("--js-runtime", choices=("node", "deno"), default="node")

    check = sub.add_parser("llm-check", help="live LLM check on the labelled cases")
    common(check)
    check.add_argument("--llm-model", default="gpt-6-luna")
    return parser


def setup_logging(root: Path, verbose: bool) -> Path:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
    handlers = [logging.FileHandler(path), logging.StreamHandler(sys.stderr)]
    handlers[1].setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    for noisy in ("urllib3", "botocore", "boto3", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return path


def s3_client(parser, local_only: bool):
    if local_only:
        return "", None
    from nba_ou.config.settings import SETTINGS
    from nba_ou.utils.s3_models import make_s3_client

    if not SETTINGS.s3_bucket:
        parser.error("configure an S3 bucket or use --local-only")
    return SETTINGS.s3_bucket, make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "status":
        from lab.coach_pressers.manifest import load_manifest, summarize

        result = {}
        for year in args.seasons:
            config = PresserConfig(year, args.local_root, prefix=args.s3_prefix)
            try:
                manifest = load_manifest(config)
            except ValueError:
                manifest = json.loads(config.manifest_path.read_bytes())
            result[config.season] = summarize(manifest) if manifest else None
        print(json.dumps(result, indent=2))
        return 0

    log_path = setup_logging(args.local_root, args.verbose)
    logging.getLogger("coach_pressers").info("log file %s", log_path)

    if args.mode == "channels":
        from lab.coach_pressers import channels

        return channels.main(args)

    if args.mode == "llm-check":
        from lab.coach_pressers import llm_judge

        return llm_judge.check_main(args)

    from lab.coach_pressers.pipeline import run_season

    bucket, client = ("", None) if args.dry_run else s3_client(parser, args.local_only)
    summaries, failed = {}, False
    for year in args.seasons:
        config = PresserConfig(
            year,
            args.local_root,
            bucket,
            prefix=args.s3_prefix,
            phases=tuple(args.phases),
            teams=tuple(args.team),
            js_runtime=args.js_runtime,
            use_llm=not args.no_llm,
            llm_model=args.llm_model,
            llm_budget_usd=args.llm_budget,
            captions=not args.no_captions,
            audio=not args.captions_only,
            keep_local=args.keep_local,
        )
        try:
            summaries[config.season] = run_season(
                config,
                dry_run=args.dry_run,
                until_step=min(args.until_step, 8) if args.dry_run else args.until_step,
                client=client,
            )
        except KeyboardInterrupt:
            print("Interrupted; progress is checkpointed", file=sys.stderr)
            return 130
        except Exception as exc:
            logging.getLogger("coach_pressers").exception("%s failed", config.season)
            summaries[config.season] = {"error": f"{type(exc).__name__}: {exc}"}
            failed = True
    print(json.dumps(summaries, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
