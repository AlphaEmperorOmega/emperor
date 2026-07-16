from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def resolve_root(root: Path) -> Path:
    """Return an absolute, resolved storage root."""

    candidate = Path(root)
    if sys.platform == "win32":
        _reject_windows_reparse_components(candidate)
    return candidate.resolve()


def _is_windows_reparse_point(path: Path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _reject_windows_reparse_components(path: Path) -> None:
    """Reject every existing junction/reparse component before path resolution."""

    if sys.platform != "win32":
        return
    absolute = Path(os.path.abspath(path))
    pending = [absolute, *absolute.parents]
    for component in reversed(pending):
        if component.exists() and _is_windows_reparse_point(component):
            raise ValueError(f"Refusing Windows reparse-point path: {component}")


def _strip_windows_device_prefix(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def _windows_final_path(path: Path) -> Path:
    """Resolve an existing path through a Win32 handle, not string traversal."""

    if sys.platform != "win32" or not path.exists():
        return path.resolve()
    try:
        import win32con
        import win32file
    except ImportError as exc:  # pragma: no cover - Windows dependency contract
        raise ValueError("Secure Windows paths require pywin32.") from exc
    handle = win32file.CreateFile(
        str(path),
        win32con.FILE_READ_ATTRIBUTES,
        win32con.FILE_SHARE_READ
        | win32con.FILE_SHARE_WRITE
        | win32con.FILE_SHARE_DELETE,
        None,
        win32con.OPEN_EXISTING,
        win32con.FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    try:
        return Path(
            _strip_windows_device_prefix(win32file.GetFinalPathNameByHandle(handle, 0))
        )
    finally:
        handle.Close()


@contextmanager
def windows_regular_file_descriptor(
    path: Path,
    *,
    trusted_root: Path,
) -> Iterator[int]:
    """Open one Windows regular file by handle and prove its final root."""

    if sys.platform != "win32":
        raise OSError("Windows file-handle validation requires Windows.")
    try:
        import msvcrt

        import win32api
        import win32con
        import win32file
    except ImportError as exc:  # pragma: no cover - Windows dependency contract
        raise OSError("Secure Windows paths require pywin32.") from exc

    root = Path(trusted_root)
    candidate = Path(path)
    _reject_windows_reparse_components(root)
    _reject_windows_reparse_components(candidate)
    root_final = _windows_final_path(root)
    handle = win32file.CreateFile(
        str(candidate),
        win32con.GENERIC_READ,
        win32con.FILE_SHARE_READ
        | win32con.FILE_SHARE_WRITE
        | win32con.FILE_SHARE_DELETE,
        None,
        win32con.OPEN_EXISTING,
        win32con.FILE_FLAG_OPEN_REPARSE_POINT
        | getattr(win32con, "FILE_FLAG_SEQUENTIAL_SCAN", 0),
        None,
    )
    descriptor: int | None = None
    handle_detached = False
    try:
        information = win32file.GetFileInformationByHandle(handle)
        attributes = int(information.get("FileAttributes", 0))
        if attributes & (
            stat.FILE_ATTRIBUTE_REPARSE_POINT | stat.FILE_ATTRIBUTE_DIRECTORY
        ):
            raise ValueError(f"Refusing non-regular Windows path: {candidate}")
        final_path = Path(
            _strip_windows_device_prefix(win32file.GetFinalPathNameByHandle(handle, 0))
        )
        normalized_root = Path(os.path.normcase(os.path.abspath(root_final)))
        normalized_final = Path(os.path.normcase(os.path.abspath(final_path)))
        try:
            normalized_final.relative_to(normalized_root)
        except ValueError as exc:
            raise ValueError(f"Path is outside allowed root: {candidate}") from exc
        raw_handle = int(handle.Detach())
        handle_detached = True
        try:
            descriptor = msvcrt.open_osfhandle(
                raw_handle,
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
        except Exception:
            win32api.CloseHandle(raw_handle)
            raise
        yield descriptor
    finally:
        if descriptor is not None:
            os.close(descriptor)
        elif not handle_detached:
            handle.Close()


def reject_link_like(path: Path, label: str = "path") -> None:
    """Reject POSIX symlinks and Windows reparse points/junctions."""

    candidate = Path(path)
    if candidate.is_symlink() or _is_windows_reparse_point(candidate):
        raise ValueError(f"Refusing to use link-like {label}: {candidate}")


def resolve_under_root(root: Path, path: Path) -> Path:
    """Resolve ``path`` and require it to stay under ``root``.

    Relative paths are interpreted as children of ``root``. Absolute paths are
    checked directly against ``root``.
    """

    resolved_root = resolve_root(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    try:
        if sys.platform == "win32":
            _reject_windows_reparse_components(candidate)
            resolved_candidate = _windows_final_path(candidate)
        else:
            resolved_candidate = candidate.resolve()
        resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Path is outside allowed root: {path}") from exc
    return resolved_candidate


def safe_child_path(root: Path, relative_path: str | Path) -> Path:
    """Build a child path under ``root`` from user-controlled relative input."""

    path_text = str(relative_path)
    child = Path(path_text)
    if child.is_absolute():
        raise ValueError(f"Path must be relative: {relative_path}")
    if not path_text or any(part in {"", ".", ".."} for part in path_text.split("/")):
        raise ValueError(f"Path contains unsafe components: {relative_path}")
    if "\\" in path_text:
        raise ValueError(f"Path contains unsafe separators: {relative_path}")
    return resolve_under_root(root, child)


def require_safe_name(name: str, label: str = "name") -> str:
    """Validate a single filesystem name component."""

    path = Path(name)
    if not name:
        raise ValueError(f"{label} is required")
    if path.is_absolute() or len(path.parts) != 1:
        raise ValueError(f"{label} must be a single path component")
    if name in {".", ".."} or "\\" in name:
        raise ValueError(f"{label} contains unsafe path characters")
    return name
