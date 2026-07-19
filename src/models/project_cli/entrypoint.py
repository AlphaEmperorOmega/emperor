from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "test":
        from models.project_cli.runner import run_tests

        return run_tests(arguments[1:])
    if arguments and arguments[0] in {"logs:archive", "logs-archive"}:
        from models.project_cli.logs_archive import archive_logs

        return archive_logs(arguments[1:])

    from models.project_cli.main import main as project_main

    return project_main(arguments)


__all__ = ["main"]
