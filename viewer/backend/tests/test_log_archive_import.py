from __future__ import annotations

import asyncio
import io
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

import httpx

from viewer.backend.api import ViewerApiSettings, create_app


def zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zip_file.writestr(name, data)
    return archive.getvalue()


def zip_with_info(entries: list[tuple[zipfile.ZipInfo, bytes | str]]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zip_file:
        for info, content in entries:
            data = content.encode("utf-8") if isinstance(content, str) else content
            zip_file.writestr(info, data)
    return archive.getvalue()


async def post_archive(
    *,
    logs_root: Path,
    archive: bytes,
    filename: str = "logs.zip",
    allow_unsafe_local_mutations: bool = True,
    max_upload_size: int | None = None,
    max_extracted_size: int | None = None,
) -> httpx.Response:
    app = create_app(
        ViewerApiSettings(
            logs_root=str(logs_root),
            allow_unsafe_local_mutations=allow_unsafe_local_mutations,
            **(
                {"max_upload_size": max_upload_size}
                if max_upload_size is not None
                else {}
            ),
            **(
                {"max_log_archive_extracted_size": max_extracted_size}
                if max_extracted_size is not None
                else {}
            ),
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/logs/import",
            files={"archive": (filename, archive, "application/zip")},
        )


async def post_multipart_body(
    *,
    logs_root: Path,
    body: bytes,
    content_type: str,
    max_upload_size: int | None = None,
) -> httpx.Response:
    app = create_app(
        ViewerApiSettings(
            logs_root=str(logs_root),
            allow_unsafe_local_mutations=True,
            **(
                {"max_upload_size": max_upload_size}
                if max_upload_size is not None
                else {}
            ),
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/logs/import",
            content=body,
            headers={"content-type": content_type},
        )


async def post_streaming_multipart_body(
    *,
    logs_root: Path,
    chunks: list[bytes],
    content_type: str,
    max_upload_size: int,
) -> tuple[httpx.Response, int]:
    app = create_app(
        ViewerApiSettings(
            logs_root=str(logs_root),
            allow_unsafe_local_mutations=True,
            max_upload_size=max_upload_size,
        )
    )
    yielded_count = 0

    async def content_stream():
        nonlocal yielded_count
        for chunk in chunks:
            yielded_count += 1
            yield chunk

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/logs/import",
            content=content_stream(),
            headers={"content-type": content_type},
        )
    return response, yielded_count


class LogArchiveImportApiTests(unittest.TestCase):
    def test_successful_import_extracts_zip_contents_into_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes(
                {
                    "my_experiment/version_0/events.out.tfevents": "events",
                    "my_experiment/version_0/hparams.yaml": "batch_size: 4\n",
                }
            )

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["extractedFileCount"], 2)
            self.assertEqual(response.json()["skippedFileCount"], 0)
            self.assertEqual(
                (logs_root / "my_experiment/version_0/hparams.yaml").read_text(
                    encoding="utf-8"
                ),
                "batch_size: 4\n",
            )

    def test_whole_archive_top_level_logs_wrapper_is_stripped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_with_info(
                [
                    (zipfile.ZipInfo("logs/"), b""),
                    (
                        zipfile.ZipInfo(
                            "logs/linear_adaptive_long_test/version_0/result.json"
                        ),
                        "{}",
                    ),
                    (
                        zipfile.ZipInfo(
                            "logs/linear_adaptive_long_test/version_0/hparams.yaml"
                        ),
                        "",
                    ),
                ]
            )

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(
                (
                    logs_root
                    / "linear_adaptive_long_test/version_0/result.json"
                ).is_file()
            )
            self.assertFalse((logs_root / "logs").exists())

    def test_whole_archive_logs_wrapper_without_directory_is_stripped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes(
                {
                    "logs/linear/version_0/result.json": "{}",
                    "logs/linear/version_0/hparams.yaml": "",
                }
            )

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue((logs_root / "linear/version_0/result.json").is_file())
            self.assertTrue((logs_root / "linear/version_0/hparams.yaml").is_file())
            self.assertFalse((logs_root / "logs").exists())

    def test_partial_logs_prefix_preserves_logs_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes(
                {
                    "logs/version_0/result.json": "{}",
                    "other_experiment/version_0/result.json": "{}",
                }
            )

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue((logs_root / "logs/version_0/result.json").is_file())
            self.assertTrue(
                (logs_root / "other_experiment/version_0/result.json").is_file()
            )

    def test_path_traversal_entries_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            archive = zip_bytes(
                {
                    "safe/version_0/result.json": "{}",
                    "../escaped.txt": "nope",
                }
            )

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("traversal", response.json()["detail"])
            self.assertFalse((logs_root / "safe").exists())
            self.assertFalse((root / "escaped.txt").exists())

    def test_absolute_path_entries_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            archive = zip_bytes(
                {
                    "safe/version_0/result.json": "{}",
                    "/tmp/escaped.txt": "nope",
                }
            )

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("absolute", response.json()["detail"])
            self.assertFalse((logs_root / "safe").exists())

    def test_existing_files_are_skipped_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            existing = logs_root / "my_experiment/version_0/result.json"
            existing.parent.mkdir(parents=True)
            existing.write_text("existing", encoding="utf-8")
            archive = zip_bytes(
                {
                    "my_experiment/version_0/result.json": "new",
                    "my_experiment/version_0/hparams.yaml": "batch_size: 4\n",
                }
            )

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["extractedFileCount"], 1)
            self.assertEqual(response.json()["skippedFileCount"], 1)
            self.assertEqual(existing.read_text(encoding="utf-8"), "existing")
            self.assertTrue(
                (logs_root / "my_experiment/version_0/hparams.yaml").is_file()
            )

    def test_zero_byte_files_are_counted_as_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes({"my_experiment/version_0/empty.txt": ""})

            response = asyncio.run(
                post_archive(logs_root=logs_root, archive=archive)
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["extractedFileCount"], 1)
            self.assertEqual(response.json()["skippedFileCount"], 0)
            self.assertTrue((logs_root / "my_experiment/version_0/empty.txt").is_file())

    def test_invalid_non_zip_upload_returns_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(
                post_archive(logs_root=Path(tmp) / "logs", archive=b"not a zip")
            )

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"], "Invalid zip archive.")

    def test_disabled_local_mutation_capability_blocks_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(
                post_archive(
                    logs_root=Path(tmp) / "logs",
                    archive=zip_bytes({"my_experiment/file.txt": "data"}),
                    allow_unsafe_local_mutations=False,
                )
            )

            self.assertEqual(response.status_code, 403)
            self.assertEqual(
                response.json()["detail"],
                "Local mutation endpoints are disabled",
            )

    def test_non_zip_filename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(
                post_archive(
                    logs_root=Path(tmp) / "logs",
                    archive=zip_bytes({"my_experiment/file.txt": "data"}),
                    filename="logs.tar",
                )
            )

            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json()["detail"],
                "Uploaded log archive must be a .zip file.",
            )

    def test_upload_size_limit_rejects_request_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes({"my_experiment/file.txt": "data"}),
                    max_upload_size=64,
                )
            )

            self.assertEqual(response.status_code, 413)
            self.assertIn("64 byte limit", response.json()["detail"])
            self.assertFalse(logs_root.exists())

    def test_streaming_upload_size_limit_stops_reading_before_full_body(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            chunks = [
                b"12345678",
                b"abcdefgh",
                b"this chunk should not be consumed",
            ]

            response, yielded_count = asyncio.run(
                post_streaming_multipart_body(
                    logs_root=logs_root,
                    chunks=chunks,
                    content_type="multipart/form-data; boundary=limit",
                    max_upload_size=12,
                )
            )

            self.assertEqual(response.status_code, 413)
            self.assertIn("12 byte limit", response.json()["detail"])
            self.assertLess(yielded_count, len(chunks))
            self.assertFalse(logs_root.exists())

    def test_extracted_size_limit_rejects_archive_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes({"my_experiment/file.txt": "data"}),
                    max_extracted_size=2,
                )
            )

            self.assertEqual(response.status_code, 413)
            self.assertIn("2 byte limit", response.json()["detail"])
            self.assertFalse(logs_root.exists())

    def test_symlink_entries_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            symlink_info = zipfile.ZipInfo("my_experiment/link")
            symlink_info.external_attr = (stat.S_IFLNK | 0o777) << 16

            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_with_info(
                        [
                            (
                                zipfile.ZipInfo(
                                    "my_experiment/version_0/result.json"
                                ),
                                "{}",
                            ),
                            (symlink_info, "target"),
                        ]
                    ),
                )
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("symlink or special file", response.json()["detail"])
            self.assertFalse(logs_root.exists())

    def test_backslash_paths_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes(
                        {
                            "my_experiment/version_0/result.json": "{}",
                            r"my_experiment\bad.txt": "bad",
                        }
                    ),
                )
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("backslashes", response.json()["detail"])
            self.assertFalse(logs_root.exists())

    def test_duplicate_file_directory_conflicts_are_rejected_before_writing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes(
                        {
                            "my_experiment/conflict": "file",
                            "my_experiment/conflict/result.json": "{}",
                        }
                    ),
                )
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("conflicts with directory path", response.json()["detail"])
            self.assertFalse(logs_root.exists())

    def test_malformed_multipart_upload_returns_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(
                post_multipart_body(
                    logs_root=Path(tmp) / "logs",
                    body=b"not actually multipart",
                    content_type="multipart/form-data; boundary=missing",
                )
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("multipart form data", response.json()["detail"])

    def test_import_clears_run_listing_cache_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes(
                {
                    (
                        "imported_exp/linears/linear/BASELINE/Mnist/"
                        "aaa_20260601_010203/version_0/result.json"
                    ): "{}",
                    (
                        "imported_exp/linears/linear/BASELINE/Mnist/"
                        "aaa_20260601_010203/version_0/hparams.yaml"
                    ): "batch_size: 4\n",
                }
            )
            app = create_app(
                ViewerApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                )
            )

            async def call_sequence() -> tuple[
                httpx.Response,
                httpx.Response,
                httpx.Response,
            ]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    before = await client.get("/logs/runs")
                    imported = await client.post(
                        "/logs/import",
                        files={"archive": ("logs.zip", archive, "application/zip")},
                    )
                    after = await client.get("/logs/runs")
                    return before, imported, after

            before, imported, after = asyncio.run(call_sequence())

            self.assertEqual(before.status_code, 200, before.text)
            self.assertEqual(before.json()["runs"], [])
            self.assertEqual(imported.status_code, 200, imported.text)
            self.assertEqual(after.status_code, 200, after.text)
            self.assertEqual(after.json()["total"], 1)
            self.assertEqual(after.json()["runs"][0]["experiment"], "imported_exp")


if __name__ == "__main__":
    unittest.main()
