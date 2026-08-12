"""Create portable log archives from the project-owned CLI."""

from __future__ import annotations

import re
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

USAGE = """Usage: mise run logs:archive -- [log_entry] [output_zip]

Run this from the project directory. Create a zip archive for downloading data
from ./logs, or from one folder inside ./logs. A selected folder is preserved
as the archive prefix so imports restore it under the same experiment path.

Arguments:
  log_entry    Optional folder inside ./logs to archive. Defaults to all
               contents of ./logs. Accepts either my_experiment or
               logs/my_experiment.
  output_zip   Zip path to create. Defaults to ./<folder>_<timestamp>.zip.

Examples:
  mise run logs:archive --
  mise run logs:archive -- my_experiment
  mise run logs:archive -- logs/my_experiment
  mise run logs:archive -- logs emperor_logs.zip"""


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    if sys.platform != "win32":
        return False
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _error(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    print(file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 1


class _ArchivePlanningError(ValueError):
    """The requested archive cannot be planned safely."""


@dataclass(frozen=True, slots=True)
class _ArchivePlan:
    log_root: Path
    source: Path
    output: Path


def _resolve_project_roots(repository_root: Path | None) -> tuple[Path, Path]:
    root = (repository_root or Path.cwd()).resolve()
    if (
        not (root / "pyproject.toml").is_file()
        or not (root / "src" / "models").is_dir()
    ):
        raise _ArchivePlanningError("run this command from the project root.")
    log_root = (root / "logs").resolve()
    if not log_root.is_dir():
        raise _ArchivePlanningError(
            "./logs not found. Run this command from the project directory."
        )
    if _is_link_like(root / "logs"):
        raise _ArchivePlanningError(
            "./logs must not be a symlink, junction, or reparse point."
        )
    return root, log_root


def _resolve_archive_source(
    selected: str,
    *,
    root: Path,
    log_root: Path,
) -> Path:
    if selected in {"", "logs", "logs/", "logs\\"}:
        source = log_root
    else:
        normalized = selected.replace("\\", "/")
        if normalized.startswith("logs/"):
            normalized = normalized[5:]
        candidate = Path(normalized)
        if candidate.is_absolute() or any(
            part in {"", ".", ".."} for part in candidate.parts
        ):
            raise _ArchivePlanningError(
                f"log entry must be a folder inside ./logs: {selected}"
            )
        source = (log_root / candidate).resolve()
        try:
            source.relative_to(log_root)
        except ValueError as exc:
            raise _ArchivePlanningError(
                f"log entry must be a folder inside ./logs: {selected}"
            ) from exc
    if not source.is_dir():
        raise _ArchivePlanningError(
            f"log folder not found under ./logs: {source.relative_to(root)}"
        )
    return source


def _resolve_archive_output(
    arguments: Sequence[str],
    *,
    root: Path,
    source: Path,
) -> Path:
    if len(arguments) >= 2:
        output = Path(arguments[1]).expanduser()
        return output.resolve() if output.is_absolute() else (root / output).resolve()
    archive_name = re.sub(r"[^A-Za-z0-9._-]", "_", source.name) or "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{archive_name}_{timestamp}.zip"


def _plan_archive(
    arguments: Sequence[str],
    *,
    repository_root: Path | None,
) -> _ArchivePlan:
    root, log_root = _resolve_project_roots(repository_root)
    selected = arguments[0] if arguments else ""
    source = _resolve_archive_source(selected, root=root, log_root=log_root)
    output = _resolve_archive_output(arguments, root=root, source=source)
    return _ArchivePlan(
        log_root=log_root,
        source=source,
        output=output,
    )


def _write_archive(plan: _ArchivePlan) -> int:
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with ZipFile(plan.output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(plan.source.rglob("*")):
            if _is_link_like(path):
                raise SystemExit(
                    "Error: refusing symlink, junction, or reparse point in "
                    f"log archive: {path}"
                )
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(plan.log_root)
            except ValueError as exc:
                raise SystemExit(
                    f"Error: log archive entry resolves outside {plan.log_root}: {path}"
                ) from exc
            if resolved == plan.output:
                continue
            archive.write(path, relative.as_posix())
            if path.is_file():
                file_count += 1
    return file_count


def archive_logs(argv: Sequence[str], *, repository_root: Path | None = None) -> int:
    arguments = list(argv)
    if arguments and arguments[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    try:
        plan = _plan_archive(arguments, repository_root=repository_root)
    except _ArchivePlanningError as exc:
        return _error(str(exc))
    file_count = _write_archive(plan)
    size_mb = plan.output.stat().st_size / (1024 * 1024)
    print(f"Created archive: {plan.output}")
    print(f"Included files: {file_count}")
    print(f"Archive size: {size_mb:.2f} MiB")
    return 0


__all__ = ["archive_logs"]
