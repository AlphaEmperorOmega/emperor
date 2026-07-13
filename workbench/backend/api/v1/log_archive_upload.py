from __future__ import annotations

import io
from dataclasses import dataclass
from email import policy
from email.parser import BytesHeaderParser
from typing import BinaryIO

from workbench.backend.core.errors import ApiError

UPLOAD_FIELD_NAMES = {"archive", "file", "logs"}
MULTIPART_HEADER_LIMIT = 64 * 1024
MULTIPART_BOUNDARY_LIMIT = 200
MULTIPART_SCAN_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class LogArchiveUpload:
    """Borrowed seekable view into the HTTP request spool."""

    filename: str
    content: BinaryIO
    size: int


class _ArchiveSlice(io.RawIOBase):
    """Seekable view over one archive part in a shared request spool."""

    def __init__(self, source: BinaryIO, *, start: int, size: int) -> None:
        super().__init__()
        self._source = source
        self._start = start
        self._size = size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise OSError("Negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed archive view")
        remaining = max(0, self._size - self._position)
        read_size = remaining if size is None or size < 0 else min(size, remaining)
        if read_size == 0:
            return b""
        self._source.seek(self._start + self._position)
        chunk = self._source.read(read_size)
        self._position += len(chunk)
        return chunk


def _too_large_error(limit: int) -> ApiError:
    return ApiError(
        f"Log archive upload exceeds the {limit} byte limit.",
        status_code=413,
    )


def _multipart_boundary(content_type: str) -> bytes:
    header = BytesHeaderParser(policy=policy.default).parsebytes(
        b"Content-Type: "
        + content_type.encode("latin-1", errors="ignore")
        + b"\r\n\r\n"
    )
    if header.get_content_type() != "multipart/form-data":
        raise ApiError("Expected multipart form data upload.")
    boundary = header.get_boundary()
    if not boundary:
        raise ApiError("Expected multipart form data upload with a boundary.")
    try:
        encoded = boundary.encode("ascii")
    except UnicodeEncodeError:
        raise ApiError("Invalid multipart form data boundary.") from None
    if len(encoded) > MULTIPART_BOUNDARY_LIMIT or b"\r" in encoded or b"\n" in encoded:
        raise ApiError("Invalid multipart form data boundary.")
    return encoded


def _limited_multipart_line(body: BinaryIO, *, remaining: int) -> bytes:
    if remaining < 1:
        raise ApiError("Multipart form data headers are too large.")
    line = body.readline(remaining + 1)
    if len(line) > remaining:
        raise ApiError("Multipart form data headers are too large.")
    return line


def _seek_first_multipart_boundary(body: BinaryIO, boundary: bytes) -> bool:
    delimiter = b"--" + boundary
    consumed = 0
    while consumed <= MULTIPART_HEADER_LIMIT:
        line = _limited_multipart_line(
            body,
            remaining=MULTIPART_HEADER_LIMIT - consumed,
        )
        if not line:
            raise ApiError("Expected multipart form data upload.")
        consumed += len(line)
        marker = line.rstrip(b"\r\n")
        if marker == delimiter:
            return False
        if marker == delimiter + b"--":
            return True
    raise ApiError("Expected multipart form data upload.")


def _multipart_part_headers(body: BinaryIO):
    lines: list[bytes] = []
    consumed = 0
    while consumed <= MULTIPART_HEADER_LIMIT:
        line = _limited_multipart_line(
            body,
            remaining=MULTIPART_HEADER_LIMIT - consumed,
        )
        if not line:
            raise ApiError("Expected multipart form data upload.")
        consumed += len(line)
        if line in {b"\r\n", b"\n"}:
            return BytesHeaderParser(policy=policy.default).parsebytes(
                b"".join(lines) + b"\r\n"
            )
        lines.append(line)
    raise ApiError("Multipart form data headers are too large.")


def _next_multipart_boundary(
    body: BinaryIO,
    boundary: bytes,
) -> tuple[int, bool]:
    marker = b"\r\n--" + boundary
    tail = b""
    while True:
        chunk_start = body.tell()
        chunk = body.read(MULTIPART_SCAN_CHUNK_SIZE)
        if not chunk:
            raise ApiError("Expected multipart form data upload.")
        data = tail + chunk
        data_start = chunk_start - len(tail)
        search_from = 0
        incomplete_at: int | None = None
        while True:
            marker_at = data.find(marker, search_from)
            if marker_at < 0:
                break
            suffix_at = marker_at + len(marker)
            if len(data) < suffix_at + 2:
                incomplete_at = marker_at
                break
            suffix = data[suffix_at : suffix_at + 2]
            if suffix in {b"\r\n", b"--"}:
                body.seek(data_start + suffix_at + 2)
                return data_start + marker_at, suffix == b"--"
            search_from = marker_at + 1

        if incomplete_at is not None:
            tail = data[incomplete_at:]
        else:
            tail_size = min(len(data), len(marker) + 1)
            tail = data[-tail_size:]


def parse_multipart_log_archive_upload(
    *,
    content_type: str,
    body: BinaryIO,
    max_upload_size: int | None,
) -> LogArchiveUpload:
    """Locate the first ZIP part without copying it out of the request spool."""

    boundary = _multipart_boundary(content_type)
    body.seek(0)
    is_final = _seek_first_multipart_boundary(body, boundary)
    fallback_upload: LogArchiveUpload | None = None
    while not is_final:
        headers = _multipart_part_headers(body)
        content_start = body.tell()
        content_end, is_final = _next_multipart_boundary(body, boundary)
        if headers.get_content_disposition() != "form-data":
            continue
        filename = headers.get_filename()
        if not filename:
            continue
        content_size = content_end - content_start
        if max_upload_size is not None and content_size > max_upload_size:
            raise _too_large_error(max_upload_size)
        upload = LogArchiveUpload(
            filename=filename,
            content=_ArchiveSlice(
                body,
                start=content_start,
                size=content_size,
            ),
            size=content_size,
        )
        field_name = headers.get_param("name", header="content-disposition")
        if field_name in UPLOAD_FIELD_NAMES:
            return upload
        fallback_upload = fallback_upload or upload

    if fallback_upload is not None:
        return fallback_upload
    raise ApiError("Log archive upload is required.")


__all__ = ["LogArchiveUpload", "parse_multipart_log_archive_upload"]
