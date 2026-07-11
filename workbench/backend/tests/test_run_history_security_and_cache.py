from __future__ import annotations

import io
import tempfile
import threading
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import RunHistoryService
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tests.helpers import write_tensorboard_run


def _service(logs_root: Path) -> RunHistoryService:
    return RunHistoryService(
        logs_root=logs_root,
        mutation_coordinator=LogExperimentMutationCoordinator(),
        active_log_writers=lambda: (),
    )


class _ScalarEvent:
    step = 1
    value = 123.0
    wall_time = 1.0


class _ImageEvent:
    step = 1
    wall_time = 1.0
    encoded_image_string = b"outside"


class _TensorProto:
    string_val = [b"outside"]


class _TensorEvent:
    step = 1
    wall_time = 1.0
    tensor_proto = _TensorProto()


class _OutsideAccumulator:
    def Tags(self) -> dict[str, list[str]]:
        return {
            "scalars": [
                "secret/value",
                "main.layer/weights/delta_norm",
            ],
            "histograms": [],
            "images": ["secret/image"],
            "tensors": ["secret/text_summary"],
        }

    def Scalars(self, _tag: str) -> list[_ScalarEvent]:
        return [_ScalarEvent()]

    def Images(self, _tag: str) -> list[_ImageEvent]:
        return [_ImageEvent()]

    def Tensors(self, _tag: str) -> list[_TensorEvent]:
        return [_TensorEvent()]


class _VersionedScalarEvent:
    def __init__(self, step: int, value: float) -> None:
        self.step = step
        self.value = value
        self.wall_time = float(step)


class _VersionedImageEvent:
    def __init__(self, value: str) -> None:
        self.step = 2
        self.wall_time = 2.0
        self.encoded_image_string = value.encode()


class _VersionedTensorProto:
    def __init__(self, value: str) -> None:
        self.string_val = [value.encode()]


class _VersionedTensorEvent:
    def __init__(self, value: str) -> None:
        self.step = 2
        self.wall_time = 2.0
        self.tensor_proto = _VersionedTensorProto(value)


class _VersionedAccumulator:
    def __init__(self, version: str) -> None:
        self.version = version

    def Tags(self) -> dict[str, list[str]]:
        return {
            "scalars": [
                f"{self.version}/value",
                "main.layer/weights/delta_norm",
            ],
            "histograms": [],
            "images": [f"{self.version}/image"],
            "tensors": [f"{self.version}/text_summary"],
        }

    def Scalars(self, tag: str) -> list[_VersionedScalarEvent]:
        value = 0.0 if self.version == "old" else 2.0
        if tag == "main.layer/weights/delta_norm":
            return [
                _VersionedScalarEvent(1, 0.0),
                _VersionedScalarEvent(2, value),
            ]
        return [_VersionedScalarEvent(1, value)]

    def Images(self, _tag: str) -> list[_VersionedImageEvent]:
        return [_VersionedImageEvent(self.version)]

    def Tensors(self, _tag: str) -> list[_VersionedTensorEvent]:
        return [_VersionedTensorEvent(self.version)]


class RunHistorySecurityAndFreshnessTests(unittest.TestCase):
    def test_in_flight_scan_cannot_republish_after_delete_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "run_20260711_020304",
                    "version_0",
                ],
            )
            service = _service(logs_root)
            scan_captured = threading.Event()
            release_scan = threading.Event()
            first_call_lock = threading.Lock()
            first_call = True
            original_scan = LogRunScanner._version_dirs_and_fingerprint

            def paused_first_scan(
                scanner: LogRunScanner,
                root: Path,
            ):
                nonlocal first_call
                result = original_scan(scanner, root)
                with first_call_lock:
                    should_pause = first_call
                    first_call = False
                if should_pause:
                    scan_captured.set()
                    if not release_scan.wait(timeout=5):
                        raise AssertionError("Timed out releasing catalog scan")
                return result

            with (
                patch.object(
                    LogRunScanner,
                    "_version_dirs_and_fingerprint",
                    paused_first_scan,
                ),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                stale_reader = executor.submit(
                    service.list_runs,
                    limit=10,
                    offset=0,
                )
                self.assertTrue(scan_captured.wait(timeout=5))
                service.delete_experiment("test_model")
                release_scan.set()
                self.assertEqual(stale_reader.result(timeout=5)["total"], 1)

            self.assertEqual(
                service.list_runs(limit=10, offset=0)["total"],
                0,
            )

    def test_mixed_event_directory_with_escape_is_ignored_by_public_queries(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            run_dir = logs_root.joinpath(
                "test_model",
                "linear",
                "BASELINE",
                "Mnist",
                "run_20260711_030405",
                "version_0",
            )
            run_dir.mkdir(parents=True)
            (run_dir / "events.out.tfevents.safe").write_bytes(b"safe")
            outside_event = root / "events.out.tfevents.outside"
            outside_event.write_bytes(b"outside secret")
            (run_dir / "events.out.tfevents.escape").symlink_to(outside_event)
            service = _service(logs_root)
            run = service.list_runs(limit=1, offset=0)["runs"][0]
            run_id = str(run["id"])
            loads: list[Path] = []

            def load_outside(event_dir: Path, **_kwargs: Any):
                loads.append(event_dir)
                return _OutsideAccumulator()

            with (
                patch(
                    "workbench.backend.run_history.query.load_event_accumulator",
                    load_outside,
                ),
                patch(
                    "workbench.backend.tensorboard.readers.load_event_accumulator",
                    load_outside,
                ),
            ):
                tags = service.tags_for_runs([run_id])
                scalars = service.scalars_for_runs(
                    run_ids=[run_id],
                    tags=["secret/value"],
                    max_points=10,
                    sampling="tail",
                )
                media = service.media_for_runs(
                    run_ids=[run_id],
                    image_tags=["secret/image"],
                    text_tags=["secret/text_summary"],
                )
                monitor = service.monitor_data_for_run(
                    run_id,
                    node_path="main.layer",
                )
                parameters = service.parameter_status_for_runs([run_id])

            self.assertEqual(loads, [])
            self.assertEqual(tags[0]["scalarTags"], [])
            self.assertEqual(tags[0]["imageTags"], [])
            self.assertEqual(tags[0]["textTags"], [])
            self.assertEqual(scalars, [])
            self.assertEqual(media["images"], [])
            self.assertEqual(media["texts"], [])
            self.assertEqual(monitor["scalarSeries"], [])
            self.assertEqual(parameters[0]["nodes"], [])

    def test_archive_overwrite_invalidates_every_public_read_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            relative_run = Path(
                "cache_exp/linear/BASELINE/Mnist/"
                "run_20260711_060708/version_0"
            )
            run_dir = logs_root / relative_run
            run_dir.mkdir(parents=True)
            event_name = "events.out.tfevents.fixed"
            (run_dir / event_name).write_text("old", encoding="utf-8")
            service = _service(logs_root)
            run_id = str(service.list_runs(limit=10, offset=0)["runs"][0]["id"])
            fixed_fingerprint = (("fixed", 1, 1),)

            def load_version(event_dir: Path, **_kwargs: Any):
                version = (event_dir / event_name).read_text(encoding="utf-8")
                return _VersionedAccumulator(version)

            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr(
                    f"{relative_run.as_posix()}/{event_name}",
                    "new",
                )
            archive.seek(0)

            with (
                patch(
                    "workbench.backend.run_history.query._event_file_fingerprint",
                    return_value=fixed_fingerprint,
                ),
                patch(
                    "workbench.backend.run_history.query.event_file_fingerprint",
                    return_value=fixed_fingerprint,
                ),
                patch(
                    "workbench.backend.tensorboard.readers.event_file_fingerprint",
                    return_value=fixed_fingerprint,
                ),
                patch(
                    "workbench.backend.run_history.query.load_event_accumulator",
                    load_version,
                ),
                patch(
                    "workbench.backend.tensorboard.readers.load_event_accumulator",
                    load_version,
                ),
            ):
                before_tags = service.tags_for_runs([run_id])[0]
                before_scalars = service.scalars_for_runs(
                    run_ids=[run_id],
                    tags=["old/value"],
                    max_points=10,
                    sampling="tail",
                )
                before_media = service.media_for_runs(
                    run_ids=[run_id],
                    image_tags=["old/image"],
                    text_tags=["old/text_summary"],
                )
                before_monitor = service.monitor_data_for_run(
                    run_id,
                    node_path="main.layer",
                )
                before_parameters = service.parameter_status_for_runs([run_id])

                service.import_archive(
                    archive=archive,
                    filename="logs.zip",
                    max_upload_size=None,
                    max_extracted_size=None,
                )

                after_listing = service.list_runs(limit=10, offset=0)
                after_tags = service.tags_for_runs([run_id])[0]
                after_scalars = service.scalars_for_runs(
                    run_ids=[run_id],
                    tags=["new/value"],
                    max_points=10,
                    sampling="tail",
                )
                after_media = service.media_for_runs(
                    run_ids=[run_id],
                    image_tags=["new/image"],
                    text_tags=["new/text_summary"],
                )
                after_monitor = service.monitor_data_for_run(
                    run_id,
                    node_path="main.layer",
                )
                after_parameters = service.parameter_status_for_runs([run_id])

            self.assertIn("old/value", before_tags["scalarTags"])
            self.assertEqual(before_scalars[0]["points"][0]["value"], 0.0)
            self.assertIn("old", before_media["texts"][0]["text"])
            self.assertEqual(
                before_monitor["scalarSeries"][0]["points"][-1]["value"],
                0.0,
            )
            self.assertEqual(
                before_parameters[0]["nodes"][0]["weights"]["status"],
                "unchanged",
            )
            self.assertEqual(after_listing["total"], 1)
            self.assertIn("new/value", after_tags["scalarTags"])
            self.assertNotIn("old/value", after_tags["scalarTags"])
            self.assertEqual(after_scalars[0]["points"][0]["value"], 2.0)
            self.assertIn("new", after_media["texts"][0]["text"])
            self.assertEqual(
                after_monitor["scalarSeries"][0]["points"][-1]["value"],
                2.0,
            )
            self.assertEqual(
                after_parameters[0]["nodes"][0]["weights"]["status"],
                "updated",
            )


if __name__ == "__main__":
    unittest.main()
