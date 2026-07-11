"""Safe extraction for uploaded Workbench log archives."""

from __future__ import annotations

import io
import stat
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO

from workbench.backend.core.errors import ApiError
from workbench.backend.core.limits import (
    DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
    DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
)
from workbench.backend.storage.local_files import resolve_root, resolve_under_root

ZIP_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LogArchiveImportPlanEntry:
    info: zipfile.ZipInfo
    relative_parts: tuple[str, ...]
    target: Path


def _too_large_error(limit: int) -> ApiError:
    return ApiError(
        f"Log archive upload exceeds the {limit} byte limit.",
        status_code=413,
    )


def _detail_text(value: object) -> str:
    return str(value) if value is not None else ""


def _zip_error(detail: str = "Invalid zip archive.") -> ApiError:
    return ApiError(detail)


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
    return not (
        file_type == stat.S_IFDIR
        if info.is_dir()
        else file_type == stat.S_IFREG
    )


def _archive_relative_parts(
    name: str,
    *,
    is_dir: bool,
    strip_top_level_logs: bool = False,
) -> tuple[str, ...] | None:
    if "\x00" in name:
        raise ApiError(f"Unsafe archive path contains a null byte: {name!r}")
    if "\\" in name:
        raise ApiError(f"Unsafe archive path uses backslashes: {name}")

    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ApiError(f"Unsafe archive path is absolute: {name}")

    parts = tuple(part for part in path.parts if part not in {"", "."})
    if any(part == ".." for part in parts):
        raise ApiError(f"Unsafe archive path contains traversal: {name}")
    if strip_top_level_logs and parts and parts[0] == "logs":
        parts = parts[1:]
    if not parts:
        if is_dir:
            return None
        raise ApiError(f"Unsafe archive path resolves to the logs root: {name}")
    return parts


def _resolved_under_root(path: Path, root: Path, original_name: str) -> Path:
    try:
        return resolve_under_root(root, path)
    except (OSError, RuntimeError, ValueError):
        raise ApiError(
            f"Unsafe archive path escapes the logs root: {original_name}"
        ) from None


def _validate_existing_ancestors(target: Path, root: Path, original_name: str) -> None:
    relative_parent = target.parent.relative_to(root)
    current = root
    for part in relative_parent.parts:
        current = current / part
        if current.is_symlink():
            raise ApiError(
                f"Unsafe archive path uses symlink destination: {original_name}"
            )
        if current.exists() and not current.is_dir():
            raise ApiError(
                f"Archive path parent is not a directory: {original_name}"
            )


def _validate_existing_target(target: Path, original_name: str) -> None:
    if target.is_symlink():
        raise ApiError(
            f"Unsafe archive path uses symlink destination: {original_name}"
        )
    if target.exists() and not target.is_file():
        raise ApiError(f"Archive path target is not a regular file: {original_name}")


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise _zip_error("Encrypted log archives are not supported.")
    if _is_unsafe_zip_mode(info):
        raise ApiError(
            f"Unsafe archive entry is a symlink or special file: {info.filename}"
        )
    if info.file_size < 0:
        raise _zip_error()


def _has_top_level_logs_wrapper(infos: list[zipfile.ZipInfo]) -> bool:
    saw_wrapped_file = False

    for info in infos:
        relative_parts = _archive_relative_parts(
            info.filename,
            is_dir=info.is_dir(),
        )
        if relative_parts is None:
            continue
        if relative_parts[0] != "logs":
            return False
        if not info.is_dir() and len(relative_parts) > 1:
            saw_wrapped_file = True

    return saw_wrapped_file


def _validate_archive_budgets(
    infos: list[zipfile.ZipInfo],
    *,
    max_member_count: int,
    max_path_bytes: int,
) -> None:
    if len(infos) > max_member_count:
        raise ApiError(
            f"Log archive exceeds the {max_member_count} member limit.",
            status_code=413,
        )

    total_path_bytes = 0
    for info in infos:
        total_path_bytes += len(info.filename.encode("utf-8"))
        if total_path_bytes > max_path_bytes:
            raise ApiError(
                (
                    "Log archive member paths exceed the "
                    f"{max_path_bytes} byte path limit."
                ),
                status_code=413,
            )


def _plan_import(
    zip_file: zipfile.ZipFile,
    *,
    logs_root: Path,
    max_extracted_size: int | None,
    max_member_count: int = DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    max_path_bytes: int = DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
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
    strip_top_level_logs = _has_top_level_logs_wrapper(infos)

    for info in infos:
        _validate_zip_info(info)
        relative_parts = _archive_relative_parts(
            info.filename,
            is_dir=info.is_dir(),
            strip_top_level_logs=strip_top_level_logs,
        )
        if relative_parts is None or info.is_dir():
            continue

        total_size += info.file_size
        if max_extracted_size is not None and total_size > max_extracted_size:
            raise ApiError(
                (
                    "Log archive extracted files exceed the "
                    f"{max_extracted_size} byte limit."
                ),
                status_code=413,
            )

        if relative_parts in planned_files:
            skipped_count += 1
            continue
        if relative_parts in planned_directories:
            raise ApiError(
                f"Archive file conflicts with directory path: {info.filename}"
            )
        for index in range(1, len(relative_parts)):
            if relative_parts[:index] in planned_files:
                raise ApiError(
                    f"Archive file conflicts with directory path: {info.filename}"
                )
        planned_files.add(relative_parts)
        planned_directories.update(
            relative_parts[:index] for index in range(1, len(relative_parts))
        )

        target = logs_root.joinpath(*relative_parts)
        _validate_existing_ancestors(target, logs_root, info.filename)
        _validate_existing_target(target, info.filename)
        resolved_target = _resolved_under_root(target, logs_root, info.filename)
        plan.append(
            LogArchiveImportPlanEntry(
                info=info,
                relative_parts=relative_parts,
                target=resolved_target,
            )
        )

    if not plan and skipped_count == 0:
        raise ApiError("Log archive does not contain files to import.")
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
    strip_top_level_logs = _has_top_level_logs_wrapper(infos)
    experiments: set[str] = set()
    total_size = 0
    for info in infos:
        _validate_zip_info(info)
        relative_parts = _archive_relative_parts(
            info.filename,
            is_dir=info.is_dir(),
            strip_top_level_logs=strip_top_level_logs,
        )
        if relative_parts is None or info.is_dir():
            continue
        total_size += info.file_size
        if max_extracted_size is not None and total_size > max_extracted_size:
            raise ApiError(
                (
                    "Log archive extracted files exceed the "
                    f"{max_extracted_size} byte limit."
                ),
                status_code=413,
            )
        experiments.add(relative_parts[0])
    if not experiments:
        raise ApiError("Log archive does not contain files to import.")
    return tuple(sorted(experiments))


def _validate_zip_payload(zip_file: zipfile.ZipFile) -> None:
    try:
        corrupt_name = zip_file.testzip()
    except (RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        raise _zip_error(f"Invalid zip archive: {_detail_text(exc)}") from exc
    if corrupt_name is not None:
        raise _zip_error(f"Invalid zip archive member: {corrupt_name}")


def _copy_zip_entry(
    zip_file: zipfile.ZipFile,
    entry: LogArchiveImportPlanEntry,
) -> bool:
    temp_path: Path | None = None
    try:
        entry.target.parent.mkdir(parents=True, exist_ok=True)
        with zip_file.open(entry.info) as source:
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=entry.target.parent,
                prefix=f".{entry.target.name}.",
                suffix=".tmp",
            ) as output:
                temp_path = Path(output.name)
                while True:
                    chunk = source.read(ZIP_CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
        temp_path.replace(entry.target)
    except FileExistsError:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False
    except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise _zip_error(f"Invalid zip archive: {_detail_text(exc)}") from exc
    return True


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
    max_member_count: int = DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    max_path_bytes: int = DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
) -> dict[str, object]:
    archive_stream, archive_size = _seekable_archive(archive)
    if max_upload_size is not None and archive_size > max_upload_size:
        raise _too_large_error(max_upload_size)
    if not filename.lower().endswith(".zip"):
        raise ApiError("Uploaded log archive must be a .zip file.")

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
            _validate_zip_payload(zip_file)
            root.mkdir(parents=True, exist_ok=True)
            extracted_count = 0
            for entry in plan:
                if _copy_zip_entry(zip_file, entry) > 0:
                    extracted_count += 1
                else:
                    skipped_count += 1
    except zipfile.BadZipFile as exc:
        raise _zip_error() from exc

    return {
        "extractedFileCount": extracted_count,
        "skippedFileCount": skipped_count,
        "destinationRoot": root.as_posix(),
    }


def inspect_log_archive_experiments(
    *,
    archive: bytes | bytearray | memoryview | BinaryIO,
    filename: str,
    max_upload_size: int | None,
    max_extracted_size: int | None,
    max_member_count: int = DEFAULT_MAX_LOG_ARCHIVE_MEMBER_COUNT,
    max_path_bytes: int = DEFAULT_MAX_LOG_ARCHIVE_PATH_BYTES,
) -> tuple[str, ...]:
    """Preflight an archive and return its affected top-level identities."""

    archive_stream, archive_size = _seekable_archive(archive)
    if max_upload_size is not None and archive_size > max_upload_size:
        raise _too_large_error(max_upload_size)
    if not filename.lower().endswith(".zip"):
        raise ApiError("Uploaded log archive must be a .zip file.")

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
    "DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE",
    "import_log_archive",
    "inspect_log_archive_experiments",
]
