import gzip
import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

from emperor.datasets.text.translation._manifest import Multi30kFile

DownloadFunction = Callable[[str, Path], object]


@contextmanager
def _exclusive_file_lock(
    path: str | Path,
    *,
    timeout: float = -1,
) -> Iterator[Path]:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=timeout)
    with lock:
        yield lock_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    """Download *url* to *destination* without importing TorchText."""

    request = urllib.request.Request(url, headers={"User-Agent": "emperor/0.1"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


class _DownloadSupport:
    def _prepare_file(self, file_spec: Multi30kFile) -> None:
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.archive_dir / file_spec.filename
        if not self._archive_is_valid(archive_path, file_spec.sha256):
            self._download_verified_archive(file_spec, archive_path)

        text_path = self.corpus_dir / file_spec.text_filename
        if not self._text_file_is_valid(text_path, file_spec.line_count):
            self._decompress_atomically(archive_path, text_path, file_spec.line_count)

    def _archive_is_valid(self, path: Path, expected_sha256: str) -> bool:
        return path.is_file() and _sha256(path) == expected_sha256

    def _download_verified_archive(
        self,
        file_spec: Multi30kFile,
        archive_path: Path,
    ) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{file_spec.filename}.",
            suffix=".tmp",
            dir=self.archive_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            result = self._downloader(file_spec.url, temporary_path)
            if isinstance(result, (bytes, bytearray)):
                temporary_path.write_bytes(result)
            actual_hash = _sha256(temporary_path)
            if actual_hash != file_spec.sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {file_spec.filename}: expected "
                    f"{file_spec.sha256}, got {actual_hash}."
                )
            os.replace(temporary_path, archive_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _text_file_is_valid(self, path: Path, expected_lines: int) -> bool:
        if not path.is_file():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                return sum(1 for _ in handle) == expected_lines
        except (OSError, UnicodeDecodeError):
            return False

    def _decompress_atomically(
        self,
        archive_path: Path,
        text_path: Path,
        expected_lines: int,
    ) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{text_path.name}.",
            suffix=".tmp",
            dir=self.corpus_dir,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with (
                gzip.open(archive_path, "rb") as source,
                temporary_path.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination)
            if not self._text_file_is_valid(temporary_path, expected_lines):
                raise RuntimeError(
                    f"Prepared {text_path.name} does not contain "
                    f"{expected_lines} UTF-8 lines."
                )
            os.replace(temporary_path, text_path)
        finally:
            temporary_path.unlink(missing_ok=True)
