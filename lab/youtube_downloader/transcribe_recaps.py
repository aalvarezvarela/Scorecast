"""Download NBA recap audio, transcribe it with Wishper, and archive JSON.gz.

    python lab/youtube_downloader/transcribe_recaps.py download --date 2026-10-28
    python lab/youtube_downloader/transcribe_recaps.py download 2025-26 --skip-download
    python lab/youtube_downloader/transcribe_recaps.py status 2025-26

Uses the existing audio downloader and S3 settings. One season/date per run.
See docs/pipeline/recap_transcripts.md for the contract and recovery behavior.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from botocore.exceptions import BotoCoreError, ClientError  # noqa: E402

from lab.youtube_downloader import download_recaps, recap_audio  # noqa: E402
from lab.youtube_downloader.recap_transcripts import (  # noqa: E402
    S3TranscriptStore,
    TranscriptConfig,
    run_archive,
    summarize,
)

# Archives live under profile=<label>/season=<season>/, locally and in S3.
LOCAL_ROOT = PROJECT_ROOT / "data" / "nba_game_recap_transcripts"


def build_parser():
    # Keep season/date selection and all source-discovery flags identical.
    parser = download_recaps.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        local_root=LOCAL_ROOT, s3_prefix="train_data/game_recap_transcripts/v1"
    )
    parser.add_argument("--audio-root", type=Path, default=download_recaps.LOCAL_ROOT)
    parser.add_argument("--audio-s3-prefix", default="train_data/game_recaps/v1")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use the existing audio manifest; do not query NBA or YouTube",
    )
    parser.add_argument(
        "--api-base-url", default=os.getenv("WISHPER_BASE_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=1800,
        help="Wishper read timeout in seconds (default: 1800)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-minutes", type=int, default=15)
    parser.add_argument(
        "--busy-retries",
        type=int,
        default=2,
        help="extra attempts for HTTP 503 only (default: 2)",
    )
    parser.add_argument(
        "--transcription-profile",
        default="wishper-v1",
        help="operator label for the server model/settings; each label is its own "
        "profile=<label> archive (letters, digits, '.', '_', '-')",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.season is None and args.date is None:
        parser.error("select a season or --date YYYY-MM-DD")
    year = (
        args.season
        if args.season is not None
        else download_recaps.get_season_year_from_date(args.date)
    )
    if args.mode == "status":
        try:
            directory = TranscriptConfig(
                year,
                args.local_root,
                "",
                phases=tuple(args.phases),
                profile=args.transcription_profile,
            ).directory
        except ValueError as exc:
            parser.error(str(exc))
        path = directory / "manifest.json"
        if not path.is_file():
            print(f"No local transcript manifest: {path}", file=sys.stderr)
            return 1
        print(json.dumps(summarize(json.loads(path.read_bytes()), args.date), indent=2))
        return 0

    try:
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
        config = TranscriptConfig(
            year,
            args.local_root,
            bucket,
            prefix=args.s3_prefix,
            phases=tuple(args.phases),
            api_base_url=args.api_base_url,
            api_key=os.getenv("WISHPER_API_KEY", ""),
            api_timeout=args.api_timeout,
            batch_size=args.batch_size,
            chunk_minutes=args.chunk_minutes,
            busy_retries=args.busy_retries,
            profile=args.transcription_profile,
        )
        audio_config = recap_audio.AudioConfig(
            year,
            args.audio_root,
            bucket,
            prefix=args.audio_s3_prefix,
            phases=tuple(args.phases),
            language=args.language,
            request_interval=args.request_interval,
            search_results=args.search_results,
            upload_window_days=args.upload_window_days,
            js_runtime=args.js_runtime,
            diagnostics=args.diagnostics,
            use_channel_index=not args.no_channel_index,
        )
        # Reject colliding destinations before the download stage writes anything.
        if (
            config.directory.resolve() == audio_config.directory.resolve()
            or config.s3_prefix == audio_config.s3_prefix
        ):
            raise ValueError(
                "Audio and transcripts need separate roots and S3 prefixes"
            )
        audio_store = recap_audio.S3AudioStore(client, audio_config) if client else None
        if not args.skip_download:
            recap_audio.require_tools(audio_config)
            recap_audio.run_archive(
                audio_config,
                game_date=args.date,
                max_games=args.max_games,
                store=audio_store,
            )
        manifest = run_archive(
            config,
            audio_config,
            game_date=args.date,
            max_games=args.max_games,
            store=S3TranscriptStore(client, config) if client else None,
            audio_store=audio_store,
        )
        transcript_errors = any(
            status.endswith("error")
            for status in summarize(manifest, args.date)["by_status"]
        )
        source = json.loads((audio_config.directory / "manifest.json").read_bytes())
        audio_errors = any(
            status.endswith("error")
            for status in recap_audio.summarize(source, args.date)["by_status"]
        )
        return int(transcript_errors or audio_errors)
    except KeyboardInterrupt:
        print(
            "Interrupted; progress was checkpointed. Re-run the same command to resume.",
            file=sys.stderr,
        )
        return 130
    except (ValueError, RuntimeError, OSError, ClientError, BotoCoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
