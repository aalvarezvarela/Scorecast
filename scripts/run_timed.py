"""Run a workflow operation with progress, timing and a job summary.

    python scripts/run_timed.py "Update line history" -- python scripts/update_databases/update_line_history_database.py

The child's output streams directly to the Actions log. Arguments are passed
without a shell and a failed command keeps its exit code.
"""

import argparse
import subprocess

from nba_ou.utils.phase_timer import PhaseTimer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("title")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after the title")

    timer = PhaseTimer()
    try:
        with timer.report(args.title), timer.phase(args.title):
            subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
