from __future__ import annotations

import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from emperor_workbench.log_experiments import LogExperimentMutationCoordinator
from emperor_workbench.run_history import RunHistoryFailure, RunHistoryService
from emperor_workbench.run_history import _archive as log_archive
from tests.support.model_packages import model_identity_resolver

TEST_ARCHIVE_MEMBER_LIMIT = 32_000
TEST_ARCHIVE_PATH_BYTE_LIMIT = 4 * 1024 * 1024


def zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zip_file.writestr(name, data)
    return archive.getvalue()


def run_history_service(logs_root: Path) -> RunHistoryService:
    return RunHistoryService(
        logs_root=logs_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: (),
        model_identity_resolver=model_identity_resolver(),
    )


class LogArchiveImportTests(unittest.TestCase):
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
            service = run_history_service(logs_root)
            original_open = zipfile.ZipFile.open
            opened: list[str] = []

            def tracked_open(zip_file, member, *args, **kwargs):
                name = (
                    member.filename if isinstance(member, zipfile.ZipInfo) else member
                )
                opened.append(str(name))
                return original_open(zip_file, member, *args, **kwargs)

            with patch.object(zipfile.ZipFile, "open", tracked_open):
                service.import_archive(
                    archive=archive,
                    filename="logs.zip",
                    max_upload_size=None,
                    max_extracted_size=None,
                    max_member_count=TEST_ARCHIVE_MEMBER_LIMIT,
                    max_path_bytes=TEST_ARCHIVE_PATH_BYTE_LIMIT,
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
            archive = io.BytesIO(zip_bytes({"experiment/nested/result.json": "secret"}))
            service = run_history_service(logs_root)
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
                    max_member_count=TEST_ARCHIVE_MEMBER_LIMIT,
                    max_path_bytes=TEST_ARCHIVE_PATH_BYTE_LIMIT,
                )

            self.assertFalse((outside / "nested" / "result.json").exists())

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
            service = run_history_service(logs_root)
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
                    max_member_count=TEST_ARCHIVE_MEMBER_LIMIT,
                    max_path_bytes=TEST_ARCHIVE_PATH_BYTE_LIMIT,
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
            service = run_history_service(logs_root)
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
                    max_member_count=TEST_ARCHIVE_MEMBER_LIMIT,
                    max_path_bytes=TEST_ARCHIVE_PATH_BYTE_LIMIT,
                )

            self.assertEqual(result.extracted_file_count, 0)
            self.assertEqual(result.skipped_file_count, 1)
            self.assertFalse((logs_root / "new_exp/nested/result.json").exists())
            self.assertEqual(list(logs_root.rglob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows commit race requires Windows")
    def test_windows_file_exists_during_commit_is_counted_as_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            service = run_history_service(logs_root)
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
                    max_member_count=TEST_ARCHIVE_MEMBER_LIMIT,
                    max_path_bytes=TEST_ARCHIVE_PATH_BYTE_LIMIT,
                )

            self.assertEqual(result.extracted_file_count, 0)
            self.assertEqual(result.skipped_file_count, 1)
            self.assertFalse((logs_root / "new_exp/nested/result.json").exists())

    def test_import_preserves_safe_legacy_experiment_names_outside_ui_grammar(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            service = run_history_service(logs_root)
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
                max_member_count=TEST_ARCHIVE_MEMBER_LIMIT,
                max_path_bytes=TEST_ARCHIVE_PATH_BYTE_LIMIT,
            )

            runs = service.list_runs(limit=10, offset=0)
            experiments = service.list_experiments(limit=10, offset=0)
            self.assertEqual(runs.runs[0].experiment, "legacy-name")
            self.assertEqual(experiments.experiments, ())


if __name__ == "__main__":
    unittest.main()
