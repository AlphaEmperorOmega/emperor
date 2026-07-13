from __future__ import annotations

import asyncio
import io
import os
import stat
import tempfile
import threading
import time
import unittest
import uuid
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx

from workbench.backend.api import WorkbenchApiSettings, create_app
from workbench.backend.api.v1.routers import logs as logs_router
from workbench.backend.core.security import (
    LOCAL_MUTATION_DISABLED_DETAIL,
    MUTATION_PROOF_REQUIRED_DETAIL,
)
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import RunHistoryService
from workbench.backend.run_history import archive as log_archive
from workbench.backend.run_history.errors import RunHistoryFailure

MUTATION_HEADER_NAME = "X-Workbench-Mutation"
MUTATION_HEADER_VALUE = "true"
TRUSTED_FRONTEND_ORIGIN = "http://localhost:9000"
UNTRUSTED_FRONTEND_ORIGIN = "https://evil.example"


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
    allow_log_imports: bool | None = True,
    max_upload_size: int | None = None,
    max_extracted_size: int | None = None,
    max_member_count: int | None = None,
    max_path_bytes: int | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    app = create_app(
        WorkbenchApiSettings(
            logs_root=str(logs_root),
            allow_unsafe_local_mutations=allow_unsafe_local_mutations,
            **(
                {"allow_log_imports": allow_log_imports}
                if allow_log_imports is not None
                else {}
            ),
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
            **(
                {"max_log_archive_member_count": max_member_count}
                if max_member_count is not None
                else {}
            ),
            **(
                {"max_log_archive_path_bytes": max_path_bytes}
                if max_path_bytes is not None
                else {}
            ),
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost",
    ) as client:
        return await client.post(
            "/logs/import",
            headers=(
                {
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                }
                if headers is None
                else headers
            ),
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
        WorkbenchApiSettings(
            logs_root=str(logs_root),
            allow_unsafe_local_mutations=True,
            allow_log_imports=True,
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
        base_url="http://localhost",
    ) as client:
        return await client.post(
            "/logs/import",
            content=body,
            headers={
                "content-type": content_type,
                MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                "Idempotency-Key": uuid.uuid4().hex,
            },
        )


async def post_streaming_multipart_body(
    *,
    logs_root: Path,
    chunks: list[bytes],
    content_type: str,
    max_upload_size: int,
    headers: dict[str, str] | None = None,
    allow_log_imports: bool = True,
) -> tuple[httpx.Response, int]:
    app = create_app(
        WorkbenchApiSettings(
            logs_root=str(logs_root),
            allow_unsafe_local_mutations=True,
            allow_log_imports=allow_log_imports,
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
        base_url="http://localhost",
    ) as client:
        response = await client.post(
            "/logs/import",
            content=content_stream(),
            headers=(
                {
                    "content-type": content_type,
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                }
                if headers is None
                else {"content-type": content_type, **headers}
            ),
        )
    return response, yielded_count


class LogArchiveImportApiTests(unittest.TestCase):
    def test_archive_members_are_decompressed_once_into_private_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = io.BytesIO(
                zip_bytes(
                    {
                        "experiment/one.json": "one",
                        "experiment/two.json": "two",
                    }
                )
            )
            service = RunHistoryService(
                logs_root=logs_root,
                mutation_coordinator=LogExperimentMutationCoordinator(),
                active_log_writers=lambda: (),
            )
            original_open = zipfile.ZipFile.open
            opened: list[str] = []

            def tracked_open(zip_file, member, *args, **kwargs):
                name = (
                    member.filename
                    if isinstance(member, zipfile.ZipInfo)
                    else member
                )
                opened.append(str(name))
                return original_open(zip_file, member, *args, **kwargs)

            with patch.object(zipfile.ZipFile, "open", tracked_open):
                service.import_archive(
                    archive=archive,
                    filename="logs.zip",
                    max_upload_size=None,
                    max_extracted_size=None,
                )

            self.assertEqual(
                opened,
                ["experiment/one.json", "experiment/two.json"],
            )

    def test_destination_ancestor_swap_after_staging_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            outside = root / "outside"
            outside.mkdir()
            archive = io.BytesIO(
                zip_bytes({"experiment/nested/result.json": "secret"})
            )
            service = RunHistoryService(
                logs_root=logs_root,
                mutation_coordinator=LogExperimentMutationCoordinator(),
                active_log_writers=lambda: (),
            )
            original_extract = log_archive._extract_archive_to_stage

            def extract_then_swap(*args, **kwargs):
                result = original_extract(*args, **kwargs)
                (logs_root / "experiment").symlink_to(
                    outside,
                    target_is_directory=True,
                )
                return result

            with (
                patch.object(
                    log_archive,
                    "_extract_archive_to_stage",
                    extract_then_swap,
                ),
                self.assertRaises(RunHistoryFailure),
            ):
                service.import_archive(
                    archive=archive,
                    filename="logs.zip",
                    max_upload_size=None,
                    max_extracted_size=None,
                )

            self.assertFalse((outside / "nested" / "result.json").exists())

    def test_default_settings_disable_log_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            relative_path = "default_disabled/result.json"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes({relative_path: "{}"}),
                    allow_unsafe_local_mutations=False,
                    allow_log_imports=None,
                )
            )

            self.assertEqual(response.status_code, 403, response.text)
            self.assertFalse((logs_root / relative_path).exists())

    def test_browser_import_requires_mutation_proof_and_trusted_context(
        self,
    ) -> None:
        cases = (
            ("absent-origin-without-proof", {}, 403),
            (
                "absent-origin-with-proof",
                {
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                200,
            ),
            (
                "trusted-origin-with-proof",
                {
                    "Origin": TRUSTED_FRONTEND_ORIGIN,
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                200,
            ),
            ("trusted-origin-without-proof", {"Origin": TRUSTED_FRONTEND_ORIGIN}, 403),
            (
                "untrusted-origin-with-proof",
                {
                    "Origin": UNTRUSTED_FRONTEND_ORIGIN,
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                403,
            ),
            (
                "trusted-cross-site-fetch-with-proof",
                {
                    "Origin": TRUSTED_FRONTEND_ORIGIN,
                    "Sec-Fetch-Site": "cross-site",
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                200,
            ),
            (
                "cross-site-fetch-without-origin",
                {
                    "Sec-Fetch-Site": "cross-site",
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                403,
            ),
            (
                "same-origin-fetch-with-proof",
                {
                    "Origin": "http://localhost",
                    "Sec-Fetch-Site": "same-origin",
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                200,
            ),
            (
                "trusted-same-site-fetch-with-proof",
                {
                    "Origin": TRUSTED_FRONTEND_ORIGIN,
                    "Sec-Fetch-Site": "same-site",
                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                    "Idempotency-Key": uuid.uuid4().hex,
                },
                200,
            ),
        )

        for index, (label, headers, expected_status) in enumerate(cases):
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                logs_root = Path(tmp) / "logs"
                relative_path = f"{index}_{label}/result.json"
                response = asyncio.run(
                    post_archive(
                        logs_root=logs_root,
                        archive=zip_bytes({relative_path: "{}"}),
                        headers=headers,
                    )
                )

                self.assertEqual(response.status_code, expected_status, response.text)
                self.assertEqual(
                    (logs_root / relative_path).is_file(),
                    expected_status == 200,
                )
                if headers.get("Origin") == TRUSTED_FRONTEND_ORIGIN:
                    self.assertEqual(
                        response.headers.get("access-control-allow-origin"),
                        TRUSTED_FRONTEND_ORIGIN,
                    )

    def test_cross_site_import_is_rejected_before_streaming_the_request_body(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response, yielded_count = asyncio.run(
                post_streaming_multipart_body(
                    logs_root=logs_root,
                    chunks=[b"untrusted", b" multipart", b" body"],
                    content_type="multipart/form-data; boundary=probe",
                    max_upload_size=1024,
                    headers={
                        "Origin": UNTRUSTED_FRONTEND_ORIGIN,
                        "Sec-Fetch-Site": "cross-site",
                    },
                )
            )

            self.assertEqual(response.status_code, 403, response.text)
            self.assertEqual(yielded_count, 0)
            self.assertFalse(logs_root.exists())

    def test_missing_proof_is_rejected_before_streaming_the_request_body(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response, yielded_count = asyncio.run(
                post_streaming_multipart_body(
                    logs_root=logs_root,
                    chunks=[b"missing", b" mutation", b" proof"],
                    content_type="multipart/form-data; boundary=probe",
                    max_upload_size=1024,
                    headers={},
                )
            )

            self.assertEqual(response.status_code, 403, response.text)
            self.assertEqual(
                response.json(),
                {"detail": MUTATION_PROOF_REQUIRED_DETAIL},
            )
            self.assertEqual(yielded_count, 0)
            self.assertFalse(logs_root.exists())

    def test_disabled_import_is_rejected_before_streaming_the_request_body(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response, yielded_count = asyncio.run(
                post_streaming_multipart_body(
                    logs_root=logs_root,
                    chunks=[b"disabled", b" import", b" body"],
                    content_type="multipart/form-data; boundary=probe",
                    max_upload_size=1024,
                    headers={
                        MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                    allow_log_imports=False,
                )
            )

            self.assertEqual(response.status_code, 403, response.text)
            self.assertEqual(
                response.json(),
                {"detail": LOCAL_MUTATION_DISABLED_DETAIL},
            )
            self.assertEqual(yielded_count, 0)
            self.assertFalse(logs_root.exists())

    def test_successful_import_extracts_zip_contents_into_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes(
                {
                    "my_experiment/version_0/events.out.tfevents": "events",
                    "my_experiment/version_0/hparams.yaml": "batch_size: 4\n",
                }
            )

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

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

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

            self.assertEqual(response.status_code, 200)
            self.assertTrue(
                (
                    logs_root / "linear_adaptive_long_test/version_0/result.json"
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

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

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

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

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

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

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

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

            self.assertEqual(response.status_code, 400)
            self.assertIn("absolute", response.json()["detail"])
            self.assertFalse((logs_root / "safe").exists())

    def test_existing_files_are_overwritten(self) -> None:
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

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["extractedFileCount"], 2)
            self.assertEqual(response.json()["skippedFileCount"], 0)
            self.assertEqual(existing.read_text(encoding="utf-8"), "new")
            self.assertTrue(
                (logs_root / "my_experiment/version_0/hparams.yaml").is_file()
            )

    def test_existing_symlink_targets_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            existing = logs_root / "my_experiment/version_0/result.json"
            existing.parent.mkdir(parents=True)
            existing.symlink_to(root / "escaped.txt")
            archive = zip_bytes(
                {
                    "my_experiment/version_0/result.json": "new",
                    "my_experiment/version_0/hparams.yaml": "batch_size: 4\n",
                }
            )

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

            self.assertEqual(response.status_code, 400)
            self.assertIn("symlink destination", response.json()["detail"])
            self.assertTrue(existing.is_symlink())
            self.assertFalse((root / "escaped.txt").exists())
            self.assertFalse(
                (logs_root / "my_experiment/version_0/hparams.yaml").exists()
            )

    def test_zero_byte_files_are_counted_as_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes({"my_experiment/version_0/empty.txt": ""})

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

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

    def test_explicit_log_import_does_not_require_broad_mutation_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes({"my_experiment/file.txt": "data"}),
                    allow_unsafe_local_mutations=False,
                )
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["extractedFileCount"], 1)
            self.assertTrue((logs_root / "my_experiment/file.txt").is_file())

    def test_disabled_log_import_capability_blocks_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = asyncio.run(
                post_archive(
                    logs_root=Path(tmp) / "logs",
                    archive=zip_bytes({"my_experiment/file.txt": "data"}),
                    allow_log_imports=False,
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

    def test_default_upload_size_allows_archives_larger_than_small_cap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            archive = zip_bytes({"my_experiment/file.txt": "x" * 256})
            self.assertGreater(len(archive), 64)

            response = asyncio.run(post_archive(logs_root=logs_root, archive=archive))

            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue((logs_root / "my_experiment/file.txt").is_file())

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

    def test_member_count_limit_rejects_archive_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes(
                        {
                            "experiment/version_0/first.txt": "first",
                            "experiment/version_0/second.txt": "second",
                            "experiment/version_0/third.txt": "third",
                        }
                    ),
                    max_member_count=2,
                )
            )

            self.assertEqual(response.status_code, 413, response.text)
            self.assertIn("2 member limit", response.json()["detail"])
            self.assertFalse(logs_root.exists())

    def test_cumulative_path_budget_rejects_archive_before_writing(self) -> None:
        names = (
            "experiment/version_0/first.txt",
            "experiment/version_0/second.txt",
        )
        path_bytes = sum(len(name.encode("utf-8")) for name in names)

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes({name: "data" for name in names}),
                    max_path_bytes=path_bytes - 1,
                )
            )

            self.assertEqual(response.status_code, 413, response.text)
            self.assertIn(
                f"{path_bytes - 1} byte path limit",
                response.json()["detail"],
            )
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
                                zipfile.ZipInfo("my_experiment/version_0/result.json"),
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

    def test_special_file_entries_are_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            fifo_info = zipfile.ZipInfo("my_experiment/fifo")
            fifo_info.external_attr = (stat.S_IFIFO | 0o644) << 16

            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_with_info(
                        [
                            (
                                zipfile.ZipInfo("my_experiment/version_0/result.json"),
                                "{}",
                            ),
                            (fifo_info, "special"),
                        ]
                    ),
                )
            )

            self.assertEqual(response.status_code, 400, response.text)
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

    def test_file_directory_conflicts_are_rejected_in_inverse_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=zip_bytes(
                        {
                            "my_experiment/conflict/result.json": "{}",
                            "my_experiment/conflict": "file",
                        }
                    ),
                )
            )

            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("conflicts with directory path", response.json()["detail"])
            self.assertFalse(logs_root.exists())

    def test_duplicate_archive_files_keep_first_entry_and_are_counted(self) -> None:
        duplicate_name = "my_experiment/version_0/result.json"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive = zip_with_info(
                [
                    (zipfile.ZipInfo(duplicate_name), "first"),
                    (zipfile.ZipInfo(duplicate_name), "second"),
                ]
            )
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_archive(
                    logs_root=logs_root,
                    archive=archive,
                )
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["extractedFileCount"], 1)
            self.assertEqual(response.json()["skippedFileCount"], 1)
            self.assertEqual(
                (logs_root / duplicate_name).read_text(encoding="utf-8"),
                "first",
            )

    def test_boundary_like_archive_bytes_do_not_truncate_multipart_part(self) -> None:
        boundary = b"workbench-boundary"
        archived_content = b"before\r\n--workbench-boundaryXafter"
        archive = zip_bytes({"my_experiment/version_0/payload.bin": archived_content})
        body = (
            b"--"
            + boundary
            + b'\r\nContent-Disposition: form-data; name="archive"; '
            + b'filename="logs.zip"\r\nContent-Type: application/zip\r\n\r\n'
            + archive
            + b"\r\n--"
            + boundary
            + b"--\r\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            response = asyncio.run(
                post_multipart_body(
                    logs_root=logs_root,
                    body=body,
                    content_type=("multipart/form-data; boundary=workbench-boundary"),
                )
            )

            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                (logs_root / "my_experiment/version_0/payload.bin").read_bytes(),
                archived_content,
            )

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

    def test_multipart_parsing_is_spooled_and_does_not_block_health(self) -> None:
        original_parser = logs_router.parse_multipart_log_archive_upload
        observed_body_is_seekable = False

        def slow_parser(**kwargs):
            nonlocal observed_body_is_seekable
            observed_body_is_seekable = callable(getattr(kwargs["body"], "seek", None))
            time.sleep(0.2)
            return original_parser(**kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(Path(tmp) / "logs"),
                    allow_log_imports=True,
                )
            )
            archive = zip_bytes({"experiment/version_0/result.json": "{}"})

            async def call_api() -> tuple[httpx.Response, httpx.Response, float]:
                transport = httpx.ASGITransport(app=app)
                async with (
                    httpx.AsyncClient(
                        transport=transport,
                        base_url="http://localhost",
                    ) as upload_client,
                    httpx.AsyncClient(
                        transport=transport,
                        base_url="http://localhost",
                    ) as health_client,
                ):
                    started = time.perf_counter()
                    upload_task = asyncio.create_task(
                        upload_client.post(
                            "/logs/import",
                            headers={
                                MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                                "Idempotency-Key": uuid.uuid4().hex,
                            },
                            files={
                                "archive": (
                                    "logs.zip",
                                    archive,
                                    "application/zip",
                                )
                            },
                        )
                    )
                    await asyncio.sleep(0)
                    health_response = await health_client.get("/health")
                    health_elapsed = time.perf_counter() - started
                    return health_response, await upload_task, health_elapsed

            with patch.object(
                logs_router,
                "parse_multipart_log_archive_upload",
                side_effect=slow_parser,
            ):
                health_response, upload_response, health_elapsed = asyncio.run(
                    call_api()
                )

        self.assertTrue(observed_body_is_seekable)
        self.assertLess(health_elapsed, 0.15)
        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(upload_response.status_code, 200, upload_response.text)

    def test_upload_specific_admission_limit_serializes_archive_work(self) -> None:
        original_import = RunHistoryService.import_archive
        active_imports = 0
        peak_active_imports = 0
        counter_lock = threading.Lock()

        def slow_import(service, **kwargs):
            nonlocal active_imports, peak_active_imports
            with counter_lock:
                active_imports += 1
                peak_active_imports = max(peak_active_imports, active_imports)
            try:
                time.sleep(0.1)
                return original_import(service, **kwargs)
            finally:
                with counter_lock:
                    active_imports -= 1

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(Path(tmp) / "logs"),
                    allow_log_imports=True,
                    log_archive_upload_concurrency=1,
                )
            )
            archives = [
                zip_bytes({f"experiment_{index}/version_0/result.json": "{}"})
                for index in range(3)
            ]

            async def call_api() -> list[httpx.Response]:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                ) as client:
                    return await asyncio.gather(
                        *(
                            client.post(
                                "/logs/import",
                                headers={
                                    MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                                    "Idempotency-Key": uuid.uuid4().hex,
                                },
                                files={
                                    "archive": (
                                        f"logs-{index}.zip",
                                        archive,
                                        "application/zip",
                                    )
                                },
                            )
                            for index, archive in enumerate(archives)
                        )
                    )

            with patch.object(RunHistoryService, "import_archive", new=slow_import):
                responses = asyncio.run(call_api())

        self.assertEqual(peak_active_imports, 1)
        self.assertTrue(
            all(response.status_code == 200 for response in responses),
            [response.text for response in responses],
        )

    def test_upload_admission_limit_applies_before_consuming_request_body(
        self,
    ) -> None:
        def multipart_body(boundary: bytes, archive: bytes) -> bytes:
            return (
                b"--"
                + boundary
                + b'\r\nContent-Disposition: form-data; name="archive"; '
                + b'filename="logs.zip"\r\nContent-Type: application/zip\r\n\r\n'
                + archive
                + b"\r\n--"
                + boundary
                + b"--\r\n"
            )

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(
                WorkbenchApiSettings(
                    logs_root=str(Path(tmp) / "logs"),
                    allow_log_imports=True,
                    log_archive_upload_concurrency=1,
                )
            )
            first_body = multipart_body(
                b"first-boundary",
                zip_bytes({"first/version_0/result.json": "{}"}),
            )
            second_body = multipart_body(
                b"second-boundary",
                zip_bytes({"second/version_0/result.json": "{}"}),
            )

            async def call_api() -> tuple[list[httpx.Response], bool]:
                first_started = asyncio.Event()
                release_first = asyncio.Event()
                second_consumed = asyncio.Event()

                async def first_content():
                    first_started.set()
                    yield first_body[:32]
                    await release_first.wait()
                    yield first_body[32:]

                async def second_content():
                    second_consumed.set()
                    yield second_body

                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://localhost",
                ) as client:
                    first_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            content=first_content(),
                            headers={
                                "content-type": (
                                    "multipart/form-data; boundary=first-boundary"
                                ),
                                MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                                "Idempotency-Key": uuid.uuid4().hex,
                            },
                        )
                    )
                    await first_started.wait()
                    second_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            content=second_content(),
                            headers={
                                "content-type": (
                                    "multipart/form-data; boundary=second-boundary"
                                ),
                                MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                                "Idempotency-Key": uuid.uuid4().hex,
                            },
                        )
                    )
                    await asyncio.sleep(0.05)
                    consumed_before_release = second_consumed.is_set()
                    release_first.set()
                    return (
                        list(await asyncio.gather(first_task, second_task)),
                        consumed_before_release,
                    )

            responses, consumed_before_release = asyncio.run(call_api())

        self.assertFalse(consumed_before_release)
        self.assertTrue(
            all(response.status_code == 200 for response in responses),
            [response.text for response in responses],
        )

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
                WorkbenchApiSettings(
                    logs_root=str(logs_root),
                    allow_unsafe_local_mutations=True,
                    allow_log_imports=True,
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
                    base_url="http://localhost",
                ) as client:
                    before = await client.get("/logs/runs")
                    imported = await client.post(
                        "/logs/import",
                        headers={
                            MUTATION_HEADER_NAME: MUTATION_HEADER_VALUE,
                            "Idempotency-Key": uuid.uuid4().hex,
                        },
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

    @unittest.skipUnless(os.name == "posix", "descriptor commits require POSIX")
    def test_partial_write_failure_keeps_prior_replacement_cleans_temp_and_invalidates(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            relative_run = Path(
                "partial_exp/linear/BASELINE/Mnist/run_20260711_040506/version_0"
            )
            run_dir = logs_root / relative_run
            run_dir.mkdir(parents=True)
            (run_dir / "result.json").write_text(
                '{"metrics":{"score":1},"params":{"batch_size":4}}',
                encoding="utf-8",
            )
            service = RunHistoryService(
                logs_root=logs_root,
                mutation_coordinator=LogExperimentMutationCoordinator(),
                active_log_writers=lambda: (),
            )
            before = service.list_runs(limit=10, offset=0)
            run_id = before.runs[0].id
            self.assertEqual(
                service.artifacts_for_run(run_id).params,
                {"batch_size": 4},
            )
            archive = io.BytesIO(
                zip_bytes(
                    {
                        f"{relative_run.as_posix()}/result.json": (
                            '{"metrics":{"score":2},"params":{"batch_size":8}}'
                        ),
                        f"{relative_run.as_posix()}/second.json": "second",
                    }
                )
            )
            original_rename = log_archive._descriptor_rename

            def fail_second_rename(
                filename: str,
                *,
                source_parent: int,
                target_parent: int,
            ):
                if filename == "second.json":
                    raise OSError("forced second-entry write failure")
                return original_rename(
                    filename,
                    source_parent=source_parent,
                    target_parent=target_parent,
                )

            with (
                patch.object(log_archive, "_descriptor_rename", fail_second_rename),
                self.assertRaises(RunHistoryFailure),
            ):
                service.import_archive(
                    archive=archive,
                    filename="logs.zip",
                    max_upload_size=None,
                    max_extracted_size=None,
                )

            self.assertFalse((run_dir / "second.json").exists())
            self.assertEqual(list(run_dir.glob(".*.tmp")), [])
            after = service.list_runs(limit=10, offset=0)
            self.assertEqual(after.runs[0].metrics, {"score": 2})
            self.assertEqual(
                service.artifacts_for_run(run_id).params,
                {"batch_size": 8},
            )

    @unittest.skipUnless(os.name == "posix", "descriptor commits require POSIX")
    def test_file_exists_during_atomic_replace_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            service = RunHistoryService(
                logs_root=logs_root,
                mutation_coordinator=LogExperimentMutationCoordinator(),
                active_log_writers=lambda: (),
            )
            archive = io.BytesIO(zip_bytes({"new_exp/nested/result.json": "{}"}))

            with patch.object(
                log_archive,
                "_descriptor_rename",
                side_effect=FileExistsError("forced replacement race"),
            ):
                result = service.import_archive(
                    archive=archive,
                    filename="logs.zip",
                    max_upload_size=None,
                    max_extracted_size=None,
                )

            self.assertEqual(result.extracted_file_count, 0)
            self.assertEqual(result.skipped_file_count, 1)
            self.assertFalse((logs_root / "new_exp/nested/result.json").exists())
            self.assertEqual(list(logs_root.rglob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows commit race requires Windows")
    def test_windows_file_exists_during_commit_is_counted_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            service = RunHistoryService(
                logs_root=logs_root,
                mutation_coordinator=LogExperimentMutationCoordinator(),
                active_log_writers=lambda: (),
            )
            archive = io.BytesIO(zip_bytes({"new_exp/nested/result.json": "{}"}))

            with patch.object(
                Path,
                "rename",
                side_effect=FileExistsError("forced replacement race"),
            ):
                result = service.import_archive(
                    archive=archive,
                    filename="logs.zip",
                    max_upload_size=None,
                    max_extracted_size=None,
                )

            self.assertEqual(result.extracted_file_count, 0)
            self.assertEqual(result.skipped_file_count, 1)
            self.assertFalse((logs_root / "new_exp/nested/result.json").exists())

    def test_import_preserves_safe_legacy_experiment_names_outside_ui_grammar(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            service = RunHistoryService(
                logs_root=logs_root,
                mutation_coordinator=LogExperimentMutationCoordinator(),
                active_log_writers=lambda: (),
            )
            archive = io.BytesIO(
                zip_bytes(
                    {
                        (
                            "legacy-name/linear/BASELINE/Mnist/"
                            "run_20260711_070809/version_0/result.json"
                        ): "{}"
                    }
                )
            )

            service.import_archive(
                archive=archive,
                filename="logs.zip",
                max_upload_size=None,
                max_extracted_size=None,
            )

            runs = service.list_runs(limit=10, offset=0)
            experiments = service.list_experiments(limit=10, offset=0)
            self.assertEqual(runs.runs[0].experiment, "legacy-name")
            self.assertEqual(experiments.experiments, ())


if __name__ == "__main__":
    unittest.main()
