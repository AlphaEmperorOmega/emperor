from __future__ import annotations

import io
import os
import stat
import tempfile
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO

from emperor_workbench.failures import FailureKind
from emperor_workbench.filesystem import (
    apply_owner_only_permissions,
    reject_link_like,
    resolve_root,
    resolve_under_root,
)
from emperor_workbench.run_history._errors import RunHistoryFailure
from emperor_workbench.run_history._records import LogArchiveImportResult

ZIP_CHUNK_SIZE = 1024 * 1024
_DESCRIPTOR_RELATIVE_FILESYSTEM_SUPPORTED = os.name == "posix" and all(
    function in os.supports_dir_fd
    for function in (os.open, os.mkdir, os.rename, os.stat)
)


@dataclass(frozen=True, slots=True)
class LogArchiveImportPlanEntry:
    info: zipfile.ZipInfo
    relative_parts: tuple[str, ...]
    target: Path
    target_identity: tuple[int, int, int, int] | None


def _too_large_error(limit: int) -> RunHistoryFailure:
    return RunHistoryFailure(
        f"Log archive upload exceeds the {limit} byte limit.",
        kind=FailureKind.TOO_LARGE,
    )


def _detail_text(value: object) -> str:
    return str(value) if value is not None else ""


def _zip_error(detail: str = "Invalid zip archive.") -> RunHistoryFailure:
    return RunHistoryFailure(detail)


def _is_unsafe_zip_mode(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type == 0:
        return False
    if file_type in {
        stat.S_IFLNK,
        stat.S_IFCHR,
        stat.S_IFBLK,
        stat.S_IFIFO,
        stat.S_IFSOCK,
    }:
        return True
    expected_file_type = stat.S_IFDIR if info.is_dir() else stat.S_IFREG
    return file_type != expected_file_type


def _archive_relative_parts(
    name: str,
    *,
    is_dir: bool,
) -> tuple[str, ...] | None:
    if "\x00" in name:
        raise RunHistoryFailure(f"Unsafe archive path contains a null byte: {name!r}")
    if "\\" in name:
        raise RunHistoryFailure(f"Unsafe archive path uses backslashes: {name}")

    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise RunHistoryFailure(f"Unsafe archive path is absolute: {name}")

    parts = tuple(part for part in path.parts if part not in {"", "."})
    if any(part == ".." for part in parts):
        raise RunHistoryFailure(f"Unsafe archive path contains traversal: {name}")
    if not parts:
        if is_dir:
            return None
        raise RunHistoryFailure(
            f"Unsafe archive path resolves to the logs root: {name}"
        )
    if parts[0] == "logs":
        raise RunHistoryFailure(
            "Log archives must be rooted directly at experiment paths; "
            "a top-level logs/ wrapper is not supported."
        )
    return parts


def _resolved_under_root(path: Path, root: Path, original_name: str) -> Path:
    try:
        return resolve_under_root(root, path)
    except (OSError, RuntimeError, ValueError):
        raise RunHistoryFailure(
            f"Unsafe archive path escapes the logs root: {original_name}"
        ) from None


def _validate_existing_ancestors(target: Path, root: Path, original_name: str) -> None:
    relative_parent = target.parent.relative_to(root)
    current = root
    for part in relative_parent.parts:
        current = current / part
        try:
            reject_link_like(current, "archive destination ancestor")
        except ValueError:
            raise RunHistoryFailure(
                f"Unsafe archive path uses symlink destination: {original_name}"
            ) from None
        if current.exists() and not current.is_dir():
            raise RunHistoryFailure(
                f"Archive path parent is not a directory: {original_name}"
            )


def _validate_existing_target(
    target: Path,
    original_name: str,
) -> tuple[int, int, int, int] | None:
    try:
        reject_link_like(target, "archive destination")
    except ValueError:
        raise RunHistoryFailure(
            f"Unsafe archive path uses symlink destination: {original_name}"
        ) from None
    if target.exists() and not target.is_file():
        raise RunHistoryFailure(
            f"Archive path target is not a regular file: {original_name}"
        )
    if not target.exists():
        return None
    target_stat = target.stat()
    return (
        int(target_stat.st_dev),
        int(target_stat.st_ino),
        int(target_stat.st_size),
        int(target_stat.st_mtime_ns),
    )


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise _zip_error("Encrypted log archives are not supported.")
    if _is_unsafe_zip_mode(info):
        raise RunHistoryFailure(
            f"Unsafe archive entry is a symlink or special file: {info.filename}"
        )
    if info.file_size < 0:
        raise _zip_error()


def _validate_archive_budgets(
    infos: list[zipfile.ZipInfo],
    *,
    max_member_count: int,
    max_path_bytes: int,
) -> None:
    if len(infos) > max_member_count:
        raise RunHistoryFailure(
            f"Log archive exceeds the {max_member_count} member limit.",
            kind=FailureKind.TOO_LARGE,
        )

    total_path_bytes = 0
    for info in infos:
        total_path_bytes += len(info.filename.encode("utf-8"))
        if total_path_bytes > max_path_bytes:
            raise RunHistoryFailure(
                (
                    "Log archive member paths exceed the "
                    f"{max_path_bytes} byte path limit."
                ),
                kind=FailureKind.TOO_LARGE,
            )


def _plan_import(
    zip_file: zipfile.ZipFile,
    *,
    logs_root: Path,
    max_extracted_size: int | None,
    max_member_count: int,
    max_path_bytes: int,
) -> tuple[list[LogArchiveImportPlanEntry], int]:
    plan: list[LogArchiveImportPlanEntry] = []
    skipped_count = 0
    total_size = 0
    planned_files: set[tuple[str, ...]] = set()
    planned_directories: set[tuple[str, ...]] = set()
    infos = zip_file.infolist()
    _validate_archive_budgets(
        infos,
        max_member_count=max_member_count,
        max_path_bytes=max_path_bytes,
    )
    for info in infos:
        _validate_zip_info(info)
        relative_parts = _archive_relative_parts(
            info.filename,
            is_dir=info.is_dir(),
        )
        if relative_parts is None or info.is_dir():
            continue

        total_size += info.file_size
        if max_extracted_size is not None and total_size > max_extracted_size:
            raise RunHistoryFailure(
                (
                    "Log archive extracted files exceed the "
                    f"{max_extracted_size} byte limit."
                ),
                kind=FailureKind.TOO_LARGE,
            )

        if relative_parts in planned_files:
            skipped_count += 1
            continue
        if relative_parts in planned_directories:
            raise RunHistoryFailure(
                f"Archive file conflicts with directory path: {info.filename}"
            )
        for index in range(1, len(relative_parts)):
            if relative_parts[:index] in planned_files:
                raise RunHistoryFailure(
                    f"Archive file conflicts with directory path: {info.filename}"
                )
        planned_files.add(relative_parts)
        planned_directories.update(
            relative_parts[:index] for index in range(1, len(relative_parts))
        )

        target = logs_root.joinpath(*relative_parts)
        _validate_existing_ancestors(target, logs_root, info.filename)
        target_identity = _validate_existing_target(target, info.filename)
        resolved_target = _resolved_under_root(target, logs_root, info.filename)
        plan.append(
            LogArchiveImportPlanEntry(
                info=info,
                relative_parts=relative_parts,
                target=resolved_target,
                target_identity=target_identity,
            )
        )

    if not plan and skipped_count == 0:
        raise RunHistoryFailure("Log archive does not contain files to import.")
    return plan, skipped_count


def _archive_experiments(
    zip_file: zipfile.ZipFile,
    *,
    max_extracted_size: int | None,
    max_member_count: int,
    max_path_bytes: int,
) -> tuple[str, ...]:
    infos = zip_file.infolist()
    _validate_archive_budgets(
        infos,
        max_member_count=max_member_count,
        max_path_bytes=max_path_bytes,
    )
    experiments: set[str] = set()
    total_size = 0
    for info in infos:
        _validate_zip_info(info)
        relative_parts = _archive_relative_parts(
            info.filename,
            is_dir=info.is_dir(),
        )
        if relative_parts is None or info.is_dir():
            continue
        total_size += info.file_size
        if max_extracted_size is not None and total_size > max_extracted_size:
            raise RunHistoryFailure(
                (
                    "Log archive extracted files exceed the "
                    f"{max_extracted_size} byte limit."
                ),
                kind=FailureKind.TOO_LARGE,
            )
        experiments.add(relative_parts[0])
    if not experiments:
        raise RunHistoryFailure("Log archive does not contain files to import.")
    return tuple(sorted(experiments))


def _extract_archive_to_stage(
    zip_file: zipfile.ZipFile,
    plan: list[LogArchiveImportPlanEntry],
    stage_root: Path,
) -> None:
    """Decompress each validated member exactly once into private staging."""

    try:
        for entry in plan:
            staged_path = stage_root.joinpath(*entry.relative_parts)
            staged_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            extracted_size = 0
            with zip_file.open(entry.info) as source, staged_path.open("xb") as output:
                if hasattr(os, "fchmod"):
                    os.fchmod(output.fileno(), 0o600)
                while True:
                    chunk = source.read(ZIP_CHUNK_SIZE)
                    if not chunk:
                        break
                    extracted_size += len(chunk)
                    output.write(chunk)
            apply_owner_only_permissions(staged_path)
            if extracted_size != entry.info.file_size:
                raise _zip_error(f"Invalid zip archive member: {entry.info.filename}")
    except (RuntimeError, zipfile.BadZipFile, zlib.error, OSError) as exc:
        if isinstance(exc, RunHistoryFailure):
            raise
        raise _zip_error(f"Invalid zip archive: {_detail_text(exc)}") from exc


def _require_descriptor_relative_filesystem() -> None:
    if not _DESCRIPTOR_RELATIVE_FILESYSTEM_SUPPORTED:
        raise RunHistoryFailure(
            "Safe log archive import requires POSIX descriptor-relative "
            "filesystem operations."
        )
    if not getattr(os, "O_NOFOLLOW", 0) or not getattr(os, "O_DIRECTORY", 0):
        raise RunHistoryFailure(
            "Safe log archive import requires no-follow directory operations."
        )


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


@contextmanager
def _relative_directory_fd(
    root_fd: int,
    parts: tuple[str, ...],
    *,
    create: bool,
) -> Iterator[int]:
    current_fd = os.dup(root_fd)
    try:
        for part in parts:
            try:
                next_fd = os.open(part, _directory_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise RunHistoryFailure(
                        "Archive staging directory disappeared during import: "
                        + "/".join(parts)
                    ) from None
                try:
                    os.mkdir(part, 0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise RunHistoryFailure(
                        "Unsafe archive destination ancestor changed during "
                        "import: " + "/".join(parts)
                    ) from exc
                try:
                    next_fd = os.open(
                        part,
                        _directory_flags(),
                        dir_fd=current_fd,
                    )
                except OSError as exc:
                    raise RunHistoryFailure(
                        "Unsafe archive destination ancestor changed during "
                        "import: " + "/".join(parts)
                    ) from exc
            except OSError as exc:
                raise RunHistoryFailure(
                    "Unsafe archive destination ancestor changed during import: "
                    + "/".join(parts)
                ) from exc
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd
    finally:
        os.close(current_fd)


def _descriptor_target_identity(
    parent_fd: int,
    filename: str,
) -> tuple[int, int, int, int] | None:
    try:
        target_stat = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(target_stat.st_mode):
        raise RunHistoryFailure(
            f"Unsafe archive path target changed during import: {filename}"
        )
    return (
        int(target_stat.st_dev),
        int(target_stat.st_ino),
        int(target_stat.st_size),
        int(target_stat.st_mtime_ns),
    )


def _descriptor_rename(
    filename: str,
    *,
    source_parent: int,
    target_parent: int,
) -> None:
    os.rename(
        filename,
        filename,
        src_dir_fd=source_parent,
        dst_dir_fd=target_parent,
    )


def _commit_staged_entry(
    *,
    root_fd: int,
    stage_fd: int,
    entry: LogArchiveImportPlanEntry,
) -> bool:
    parent_parts = entry.relative_parts[:-1]
    filename = entry.relative_parts[-1]
    with (
        _relative_directory_fd(stage_fd, parent_parts, create=False) as source_parent,
        _relative_directory_fd(root_fd, parent_parts, create=True) as target_parent,
    ):
        observed_identity = _descriptor_target_identity(target_parent, filename)
        if entry.target_identity is None and observed_identity is not None:
            return False
        if (
            entry.target_identity is not None
            and observed_identity != entry.target_identity
        ):
            raise RunHistoryFailure(
                f"Archive destination changed during import: {entry.info.filename}"
            )
        try:
            _descriptor_rename(
                filename,
                source_parent=source_parent,
                target_parent=target_parent,
            )
        except FileExistsError:
            return False
        except OSError as exc:
            raise RunHistoryFailure(
                f"Could not commit archive member: {entry.info.filename}"
            ) from exc
    return True


def _commit_staged_import(
    *,
    root: Path,
    stage_root: Path,
    plan: list[LogArchiveImportPlanEntry],
) -> tuple[int, int]:
    if os.name == "nt":
        return _commit_staged_import_windows(
            root=root,
            stage_root=stage_root,
            plan=plan,
        )
    _require_descriptor_relative_filesystem()
    root_fd = os.open(root, _directory_flags())
    stage_fd = os.open(stage_root, _directory_flags())
    extracted_count = 0
    skipped_count = 0
    try:
        for entry in plan:
            if _commit_staged_entry(
                root_fd=root_fd,
                stage_fd=stage_fd,
                entry=entry,
            ):
                extracted_count += 1
            else:
                skipped_count += 1
    finally:
        os.close(stage_fd)
        os.close(root_fd)
    return extracted_count, skipped_count


def _windows_destination_parent(root: Path, parts: tuple[str, ...]) -> Path:
    current = root
    for part in parts:
        current = current / part
        try:
            if current.exists():
                reject_link_like(current, "archive destination ancestor")
                if not current.is_dir():
                    raise RunHistoryFailure(
                        "Archive path parent is not a directory: " + "/".join(parts)
                    )
            else:
                current.mkdir()
                apply_owner_only_permissions(current)
            resolve_under_root(root, current)
        except (OSError, ValueError) as exc:
            raise RunHistoryFailure(
                "Unsafe archive destination ancestor changed during import: "
                + "/".join(parts)
            ) from exc
    return current


def _commit_staged_import_windows(
    *,
    root: Path,
    stage_root: Path,
    plan: list[LogArchiveImportPlanEntry],
) -> tuple[int, int]:
    extracted_count = 0
    skipped_count = 0
    for entry in plan:
        source = stage_root.joinpath(*entry.relative_parts)
        try:
            source = resolve_under_root(stage_root, source)
            target_parent = _windows_destination_parent(
                root,
                entry.relative_parts[:-1],
            )
            target = resolve_under_root(root, target_parent / entry.relative_parts[-1])
            observed_identity = _validate_existing_target(
                target,
                entry.info.filename,
            )
        except (OSError, ValueError) as exc:
            raise RunHistoryFailure(
                "Unsafe archive destination changed during import: "
                f"{entry.info.filename}"
            ) from exc
        if entry.target_identity is None and observed_identity is not None:
            skipped_count += 1
            continue
        if (
            entry.target_identity is not None
            and observed_identity != entry.target_identity
        ):
            raise RunHistoryFailure(
                f"Archive destination changed during import: {entry.info.filename}"
            )
        try:
            if observed_identity is None:
                source.rename(target)
            else:
                source.replace(target)
            apply_owner_only_permissions(target)
        except FileExistsError:
            skipped_count += 1
            continue
        except OSError as exc:
            raise RunHistoryFailure(
                f"Could not commit archive member: {entry.info.filename}"
            ) from exc
        extracted_count += 1
    return extracted_count, skipped_count


def _seekable_archive(
    archive: bytes | bytearray | memoryview | BinaryIO,
) -> tuple[BinaryIO, int]:
    if isinstance(archive, (bytes, bytearray, memoryview)):
        return io.BytesIO(archive), len(archive)

    archive.seek(0, io.SEEK_END)
    size = archive.tell()
    archive.seek(0)
    return archive, size


def import_log_archive(
    *,
    archive: bytes | bytearray | memoryview | BinaryIO,
    filename: str,
    logs_root: Path | str,
    max_upload_size: int | None,
    max_extracted_size: int | None,
    max_member_count: int,
    max_path_bytes: int,
) -> LogArchiveImportResult:
    archive_stream, archive_size = _seekable_archive(archive)
    if max_upload_size is not None and archive_size > max_upload_size:
        raise _too_large_error(max_upload_size)
    if not filename.lower().endswith(".zip"):
        raise RunHistoryFailure("Uploaded log archive must be a .zip file.")

    root = resolve_root(Path(logs_root))
    try:
        with zipfile.ZipFile(archive_stream) as zip_file:
            plan, skipped_count = _plan_import(
                zip_file,
                logs_root=root,
                max_extracted_size=max_extracted_size,
                max_member_count=max_member_count,
                max_path_bytes=max_path_bytes,
            )
            root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".workbench-import-",
                dir=root,
            ) as stage_directory:
                stage_root = Path(stage_directory)
                apply_owner_only_permissions(stage_root)
                _extract_archive_to_stage(zip_file, plan, stage_root)
                extracted_count, commit_skipped_count = _commit_staged_import(
                    root=root,
                    stage_root=stage_root,
                    plan=plan,
                )
                skipped_count += commit_skipped_count
    except zipfile.BadZipFile as exc:
        raise _zip_error() from exc

    return LogArchiveImportResult(
        extracted_file_count=extracted_count,
        skipped_file_count=skipped_count,
        destination_root=root.as_posix(),
    )


def inspect_log_archive_experiments(
    *,
    archive: bytes | bytearray | memoryview | BinaryIO,
    filename: str,
    max_upload_size: int | None,
    max_extracted_size: int | None,
    max_member_count: int,
    max_path_bytes: int,
) -> tuple[str, ...]:
    """Preflight an archive and return its affected top-level identities."""

    archive_stream, archive_size = _seekable_archive(archive)
    if max_upload_size is not None and archive_size > max_upload_size:
        raise _too_large_error(max_upload_size)
    if not filename.lower().endswith(".zip"):
        raise RunHistoryFailure("Uploaded log archive must be a .zip file.")

    try:
        try:
            with zipfile.ZipFile(archive_stream) as zip_file:
                return _archive_experiments(
                    zip_file,
                    max_extracted_size=max_extracted_size,
                    max_member_count=max_member_count,
                    max_path_bytes=max_path_bytes,
                )
        except zipfile.BadZipFile as exc:
            raise _zip_error() from exc
    finally:
        archive_stream.seek(0)


__all__ = [
    "import_log_archive",
    "inspect_log_archive_experiments",
]
