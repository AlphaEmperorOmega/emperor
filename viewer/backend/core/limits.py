"""Shared Viewer backend size limits."""

from __future__ import annotations

DEFAULT_MAX_UPLOAD_SIZE = 64 * 1024 * 1024
DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE = 512 * 1024 * 1024

__all__ = [
    "DEFAULT_MAX_LOG_ARCHIVE_EXTRACTED_SIZE",
    "DEFAULT_MAX_UPLOAD_SIZE",
]
