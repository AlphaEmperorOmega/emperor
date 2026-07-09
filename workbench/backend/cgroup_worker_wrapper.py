"""Join a training cgroup before executing the real worker command."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cgroup", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("cgroup worker wrapper requires a command.", file=sys.stderr)
        return 125

    try:
        (Path(args.cgroup) / "cgroup.procs").write_text(
            f"{os.getpid()}\n",
            encoding="utf-8",
        )
        Path(args.ready).write_text(f"{os.getpid()}\n", encoding="utf-8")
    except OSError as exc:
        print(
            f"Failed to join training cgroup '{args.cgroup}': {exc}",
            file=sys.stderr,
        )
        return 125

    os.execvpe(command[0], command, os.environ)
    return 125


if __name__ == "__main__":
    raise SystemExit(main())
