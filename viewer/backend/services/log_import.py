"""Safe extraction for uploaded Viewer log archives."""

from __future__ import annotations

import io
import stat
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath, PureWindowsPath

from viewer.backend.core.errors import ApiError
from viewer.backend.core.limits import (
    DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE,
)
from viewer.backend.storage.local_files import resolve_root, resolve_under_root

UPLOAD_FIELD_NAMES = {"archive", "file", "logs"}
ZIP_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class LogArchiveUpload:
    filename: str
    content: bytes


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


def parse_multipart_log_archive_upload(
    *,
    content_type: str,
    body: bytes,
    max_upload_size: int | None,
) -> LogArchiveUpload:
    """Extract the first zip file part from a bounded multipart/form-data body."""

    if max_upload_size is not None and len(body) > max_upload_size:
        raise _too_large_error(max_upload_size)
    if not content_type.lower().startswith("multipart/form-data"):
        raise ApiError("Expected multipart form data upload.")

    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: "
        + content_type.encode("latin-1", errors="ignore")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart():
        raise ApiError("Expected multipart form data upload.")

    fallback_upload: LogArchiveUpload | None = None
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        filename = part.get_filename()
        if not filename:
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        upload = LogArchiveUpload(filename=filename, content=payload)
        field_name = part.get_param("name", header="content-disposition")
        if field_name in UPLOAD_FIELD_NAMES:
            return upload
        fallback_upload = fallback_upload or upload

    if fallback_upload is not None:
        return fallback_upload
    raise ApiError("Log archive upload is required.")


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


def _has_top_level_logs_wrapper(zip_file: zipfile.ZipFile) -> bool:
    saw_wrapped_file = False

    for info in zip_file.infolist():
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


def _plan_import(
    zip_file: zipfile.ZipFile,
    *,
    logs_root: Path,
    max_extracted_size: int,
) -> tuple[list[LogArchiveImportPlanEntry], int]:
    plan: list[LogArchiveImportPlanEntry] = []
    skipped_count = 0
    total_size = 0
    planned_parts: set[tuple[str, ...]] = set()
    strip_top_level_logs = _has_top_level_logs_wrapper(zip_file)

    for info in zip_file.infolist():
        _validate_zip_info(info)
        relative_parts = _archive_relative_parts(
            info.filename,
            is_dir=info.is_dir(),
            strip_top_level_logs=strip_top_level_logs,
        )
        if relative_parts is None or info.is_dir():
            continue

        total_size += info.file_size
        if total_size > max_extracted_size:
            raise ApiError(
                (
                    "Log archive extracted files exceed the "
                    f"{max_extracted_size} byte limit."
                ),
                status_code=413,
            )

        for index in range(1, len(relative_parts)):
            if relative_parts[:index] in planned_parts:
                raise ApiError(
                    f"Archive file conflicts with directory path: {info.filename}"
                )
        if any(
            len(existing_parts) > len(relative_parts)
            and existing_parts[: len(relative_parts)] == relative_parts
            for existing_parts in planned_parts
        ):
            raise ApiError(
                f"Archive file conflicts with directory path: {info.filename}"
            )

        if relative_parts in planned_parts:
            skipped_count += 1
            continue
        planned_parts.add(relative_parts)

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
        return False
    except (RuntimeError, zipfile.BadZipFile, OSError) as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise _zip_error(f"Invalid zip archive: {_detail_text(exc)}") from exc
    return True


def import_log_archive(
    *,
    archive: bytes,
    filename: str,
    logs_root: Path | str,
    max_upload_size: int | None,
    max_extracted_size: int,
) -> dict[str, object]:
    if max_upload_size is not None and len(archive) > max_upload_size:
        raise _too_large_error(max_upload_size)
    if not filename.lower().endswith(".zip"):
        raise ApiError("Uploaded log archive must be a .zip file.")

    root = resolve_root(Path(logs_root))
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
            plan, skipped_count = _plan_import(
                zip_file,
                logs_root=root,
                max_extracted_size=max_extracted_size,
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


__all__ = [
    "DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE",
    "LogArchiveUpload",
    "import_log_archive",
    "parse_multipart_log_archive_upload",
]
