from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from workbench.backend.inspector.errors import InspectorError
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tensorboard.readers import (
    TensorBoardMonitorReader,
    TensorBoardParameterStatusReader,
)
from workbench.backend.tests.helpers import (
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
)


class TagsFailureAccumulator:
    def Tags(self):
        raise RuntimeError("broken tags")


class ReadFailureAccumulator:
    def Tags(self):
        return {
            "scalars": ["main_model.0.model/output/mean"],
            "histograms": ["main_model.0.model/histogram/usage_fraction"],
            "images": ["main_model.0.model/heatmap/usage_fraction"],
        }

    def Scalars(self, tag):
        raise RuntimeError(f"broken scalar read: {tag}")

    def Histograms(self, tag):
        raise RuntimeError(f"broken histogram read: {tag}")

    def Images(self, tag):
        raise RuntimeError(f"broken image read: {tag}")


class FakeScalarEvent:
    def __init__(self, step: int, value: float) -> None:
        self.step = step
        self.value = value
        self.wall_time = float(step)


class NoMatchingMonitorAccumulator:
    def Tags(self):
        return {
            "scalars": ["other_node/output/mean"],
            "histograms": [],
            "images": [],
        }


class ParameterStatusAccumulator:
    def Tags(self):
        return {
            "scalars": ["main_model.0.model/weights/relative_delta_norm"],
        }

    def Scalars(self, tag):
        if tag != "main_model.0.model/weights/relative_delta_norm":
            raise KeyError(tag)
        return [FakeScalarEvent(1, 0.0), FakeScalarEvent(2, 0.5)]


class LargeParameterStatusAccumulator:
    def Tags(self):
        return {
            "scalars": [
                "main_model.0.model/weights/relative_delta_norm",
                "main_model.0.model/bias/l2_norm",
            ],
        }

    def Scalars(self, tag):
        if tag == "main_model.0.model/weights/relative_delta_norm":
            return [
                FakeScalarEvent(1, 0.75),
                FakeScalarEvent(2, 0.0),
                FakeScalarEvent(3, 0.0),
                FakeScalarEvent(4, 0.0),
                FakeScalarEvent(5, 0.0),
            ]
        if tag == "main_model.0.model/bias/l2_norm":
            return [
                FakeScalarEvent(1, 1.0),
                FakeScalarEvent(2, 2.0),
                FakeScalarEvent(3, 2.0),
                FakeScalarEvent(4, 2.0),
                FakeScalarEvent(5, 2.0),
            ]
        raise KeyError(tag)


class TensorBoardMonitorReaderFailureTests(unittest.TestCase):
    def read_with_accumulators(
        self,
        log_dir: Path,
        accumulators: list[object | None],
    ) -> dict:
        run_dirs = [log_dir / f"run-{index}" for index, _ in enumerate(accumulators)]
        for index, run_dir in enumerate(run_dirs):
            run_dir.mkdir()
            run_dir.joinpath(f"events.out.tfevents.{index}").write_bytes(b"events")
        with patch(
            "workbench.backend.tensorboard.events.load_event_accumulator",
            side_effect=accumulators,
        ):
            return TensorBoardMonitorReader().read(
                job_id="job-1",
                node_path="main_model.0.model",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

    def assert_empty_monitor_payload(self, data: dict, log_dir: Path | None) -> None:
        self.assertEqual(
            data,
            {
                "jobId": "job-1",
                "nodePath": "main_model.0.model",
                "dataset": "Mnist",
                "logDir": str(log_dir) if log_dir is not None else None,
                "scalarSeries": [],
                "histograms": [],
                "images": [],
            },
        )

    def test_missing_or_nonexistent_log_dir_returns_empty_payload(self) -> None:
        reader = TensorBoardMonitorReader()

        missing_data = reader.read(
            job_id="job-1",
            node_path="main_model.0.model",
            dataset="Mnist",
            log_dir=None,
        )
        self.assert_empty_monitor_payload(missing_data, None)

        with tempfile.TemporaryDirectory() as tmp:
            nonexistent_log_dir = Path(tmp) / "missing-run"
            nonexistent_data = reader.read(
                job_id="job-1",
                node_path="main_model.0.model",
                dataset="Mnist",
                log_dir=str(nonexistent_log_dir),
            )

        self.assert_empty_monitor_payload(nonexistent_data, nonexistent_log_dir)

    def test_accumulator_load_failure_returns_empty_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            data = self.read_with_accumulators(log_dir, [None])

        self.assert_empty_monitor_payload(data, log_dir)

    def test_oversized_event_files_skip_monitor_tensorboard_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.large"
            event_file.write_text("large-event-payload", encoding="utf-8")
            reader = TensorBoardMonitorReader(max_event_bytes=4)

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator"
            ) as load:
                data = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(
            data,
            {
                "jobId": "job-1",
                "nodePath": "main_model.0.model",
                "dataset": "Mnist",
                "logDir": str(log_dir),
                "scalarSeries": [],
                "histograms": [],
                "images": [],
                "eventBytes": len("large-event-payload"),
                "skippedEventFiles": 1,
                "truncated": True,
                "truncationReason": (
                    "event files skipped: 19 bytes exceeds 4 byte read cap"
                ),
                "sourceItemCount": 1,
                "returnedItemCount": 0,
            },
        )
        load.assert_not_called()

    def test_tags_failure_returns_empty_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            data = self.read_with_accumulators(log_dir, [TagsFailureAccumulator()])

        self.assert_empty_monitor_payload(data, log_dir)

    def test_scalar_histogram_and_image_failures_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            data = self.read_with_accumulators(log_dir, [ReadFailureAccumulator()])

        self.assert_empty_monitor_payload(data, log_dir)

    def test_monitor_reader_matches_legacy_layer_path_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar("main_model.0.model/weights/mean", 0.25, 1)
            writer.flush()
            writer.close()

            data = TensorBoardMonitorReader().read(
                job_id="job-1",
                node_path="main_model.layers.0.model",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        self.assertEqual(data["nodePath"], "main_model.layers.0.model")
        self.assertEqual(
            [(series["tag"], series["label"]) for series in data["scalarSeries"]],
            [("main_model.0.model/weights/mean", "weights/mean")],
        )

    def test_negative_monitor_results_are_cached_until_event_files_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.cache"
            event_file.write_text("first", encoding="utf-8")
            reader = TensorBoardMonitorReader()

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                return_value=NoMatchingMonitorAccumulator(),
            ) as load:
                first = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                second = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                event_file.write_text("first-second", encoding="utf-8")
                changed = reader.read(
                    job_id="job-1",
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(first, second)
        self.assertEqual(first, changed)
        self.assertEqual(load.call_count, 2)


class TensorBoardParameterStatusReaderTests(unittest.TestCase):
    def write_scalars(
        self,
        log_dir: Path,
        scalars: dict[str, list[tuple[int, float]]],
    ) -> None:
        writer = SummaryWriter(log_dir=str(log_dir))
        for tag, points in scalars.items():
            for step, value in points:
                writer.add_scalar(tag, value, step)
        writer.flush()
        writer.close()

    def test_classifies_delta_statuses_and_missing_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            self.write_scalars(
                log_dir,
                {
                    "main_model.0.model/weights/relative_delta_norm": [
                        (2, 0.0),
                        (3, 1e-6),
                    ],
                    "main_model.0.model/bias/delta_norm": [(2, 0.0), (3, 0.0)],
                    "main_model.1.model/weights/delta_norm": [(2, 0.0), (3, 0.0)],
                    "main_model.2.model/weights/l2_norm": [(1, 4.0)],
                    "main_model.3.model/weights/delta_norm": [(2, 0.0), (3, 0.25)],
                },
            )

            data = TensorBoardParameterStatusReader().read(
                source_id="run-1",
                preset="baseline",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        nodes = {node["nodePath"]: node for node in data["nodes"]}
        self.assertEqual(data["sourceId"], "run-1")
        self.assertEqual(data["preset"], "baseline")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertEqual(nodes["main_model.0.model"]["weights"]["status"], "updated")
        self.assertEqual(
            nodes["main_model.0.model"]["weights"]["metric"],
            "main_model.0.model/weights/relative_delta_norm",
        )
        self.assertEqual(nodes["main_model.0.model"]["weights"]["lastStep"], 3)
        self.assertEqual(nodes["main_model.0.model"]["weights"]["observedPoints"], 2)
        self.assertEqual(nodes["main_model.0.model"]["bias"]["status"], "unchanged")
        self.assertEqual(nodes["main_model.0.model"]["bias"]["observedPoints"], 2)
        self.assertEqual(nodes["main_model.1.model"]["weights"]["status"], "unchanged")
        self.assertEqual(nodes["main_model.1.model"]["bias"]["status"], "missing")
        self.assertEqual(nodes["main_model.2.model"]["weights"]["status"], "unknown")
        self.assertEqual(nodes["main_model.3.model"]["weights"]["status"], "updated")
        self.assertEqual(
            nodes["main_model.3.model"]["weights"]["metric"],
            "main_model.3.model/weights/delta_norm",
        )

    def test_single_zero_delta_point_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            self.write_scalars(
                log_dir,
                {
                    "main_model.0.model/weights/delta_norm": [(2, 0.0)],
                },
            )

            data = TensorBoardParameterStatusReader().read(
                source_id="run-1",
                preset="baseline",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        node = data["nodes"][0]
        self.assertEqual(node["nodePath"], "main_model.0.model")
        self.assertEqual(node["weights"]["status"], "unknown")
        self.assertEqual(
            node["weights"]["metric"], "main_model.0.model/weights/delta_norm"
        )
        self.assertEqual(node["weights"]["lastStep"], 2)
        self.assertEqual(node["weights"]["observedPoints"], 1)
        self.assertEqual(node["bias"]["status"], "missing")

    def test_old_logs_without_delta_metrics_use_value_stat_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "run"
            self.write_scalars(
                log_dir,
                {
                    "main_model.0.model/weights/l2_norm": [(1, 4.0), (2, 4.25)],
                    "main_model.0.model/bias/mean": [(1, 0.5), (2, 0.5)],
                },
            )

            data = TensorBoardParameterStatusReader().read(
                source_id="run-1",
                preset="baseline",
                dataset="Mnist",
                log_dir=str(log_dir),
            )

        node = data["nodes"][0]
        self.assertEqual(node["nodePath"], "main_model.0.model")
        self.assertEqual(node["weights"]["status"], "updated")
        self.assertEqual(
            node["weights"]["metric"], "main_model.0.model/weights/l2_norm"
        )
        self.assertEqual(node["bias"]["status"], "unchanged")
        self.assertEqual(node["bias"]["metric"], "main_model.0.model/bias/mean")

    def test_missing_or_nonexistent_log_dir_returns_empty_status_payload(self) -> None:
        reader = TensorBoardParameterStatusReader()

        missing_data = reader.read(
            source_id="job-1",
            preset=None,
            dataset="Mnist",
            log_dir=None,
        )

        with tempfile.TemporaryDirectory() as tmp:
            nonexistent_log_dir = Path(tmp) / "missing-run"
            nonexistent_data = reader.read(
                source_id="job-1",
                preset=None,
                dataset="Mnist",
                log_dir=str(nonexistent_log_dir),
            )

        self.assertEqual(
            missing_data,
            {
                "sourceId": "job-1",
                "preset": None,
                "dataset": "Mnist",
                "logDir": None,
                "nodes": [],
            },
        )
        self.assertEqual(
            nonexistent_data,
            {
                "sourceId": "job-1",
                "preset": None,
                "dataset": "Mnist",
                "logDir": str(nonexistent_log_dir),
                "nodes": [],
            },
        )

    def test_oversized_event_files_skip_parameter_status_tensorboard_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.large"
            event_file.write_text("large-event-payload", encoding="utf-8")
            reader = TensorBoardParameterStatusReader(max_event_bytes=4)

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator"
            ) as load:
                data = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(
            data,
            {
                "sourceId": "job-1",
                "preset": None,
                "dataset": "Mnist",
                "logDir": str(log_dir),
                "nodes": [],
                "eventBytes": len("large-event-payload"),
                "skippedEventFiles": 1,
                "truncated": True,
                "truncationReason": (
                    "event files skipped: 19 bytes exceeds 4 byte read cap"
                ),
                "sourceItemCount": 1,
                "returnedItemCount": 0,
            },
        )
        load.assert_not_called()

    def test_parameter_status_is_cached_until_event_files_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            event_file = log_dir / "events.out.tfevents.cache"
            event_file.write_text("first", encoding="utf-8")
            reader = TensorBoardParameterStatusReader()

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                return_value=ParameterStatusAccumulator(),
            ) as load:
                first = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                second = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )
                event_file.write_text("first-second", encoding="utf-8")
                changed = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        self.assertEqual(first, second)
        self.assertEqual(first, changed)
        self.assertEqual(load.call_count, 2)
        node = first["nodes"][0]
        self.assertEqual(node["nodePath"], "main_model.0.model")
        self.assertEqual(node["weights"]["status"], "updated")

    def test_parameter_status_uses_custom_scalar_size_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "events.out.tfevents.cache").write_text(
                "events",
                encoding="utf-8",
            )
            reader = TensorBoardParameterStatusReader(scalar_point_limit=7)

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                return_value=ParameterStatusAccumulator(),
            ) as load:
                reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        size_guidance = load.call_args.kwargs["size_guidance"]
        self.assertEqual(size_guidance[event_accumulator.SCALARS], 7)

    def test_parameter_status_classification_uses_bounded_scalar_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            (log_dir / "events.out.tfevents.cache").write_text(
                "events",
                encoding="utf-8",
            )
            reader = TensorBoardParameterStatusReader(scalar_point_limit=3)

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                return_value=LargeParameterStatusAccumulator(),
            ):
                data = reader.read(
                    source_id="job-1",
                    preset=None,
                    dataset="Mnist",
                    log_dir=str(log_dir),
                )

        node = data["nodes"][0]
        self.assertEqual(node["weights"]["status"], "unchanged")
        self.assertEqual(node["weights"]["observedPoints"], 3)
        self.assertEqual(node["weights"]["lastStep"], 5)
        self.assertEqual(node["bias"]["status"], "unchanged")
        self.assertEqual(node["bias"]["observedPoints"], 3)
        self.assertEqual(node["bias"]["lastStep"], 5)


class HistoricalMonitorDataFailureTests(unittest.TestCase):
    def write_historical_run(self, logs_root: Path) -> tuple[str, Path]:
        run_dir = logs_root.joinpath(
            "test_model",
            "linear",
            "baseline",
            "Mnist",
            "historical_20260601_010203",
            "version_0",
        )
        run_dir.mkdir(parents=True)
        (run_dir / "events.out.tfevents.test").write_text("broken", encoding="utf-8")
        run = LogRunScanner(logs_root=logs_root).list_runs()[0]
        return run.id, run_dir

    def test_broken_historical_tags_return_empty_monitor_response_shape(self) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings, create_app

        async def call_api(logs_root: Path, run_id: str) -> httpx.Response:
            transport = httpx.ASGITransport(
                app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
            )
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get(
                    f"/logs/runs/{run_id}/monitor-data",
                    params={"nodePath": "main_model.0.model"},
                )

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_id, run_dir = self.write_historical_run(logs_root)

            with patch(
                "workbench.backend.tensorboard.events.load_event_accumulator",
                return_value=TagsFailureAccumulator(),
            ):
                response = asyncio.run(call_api(logs_root, run_id))

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            response.json(),
            {
                "jobId": run_id,
                "nodePath": "main_model.0.model",
                "preset": None,
                "dataset": "Mnist",
                "logDir": str(run_dir),
                "scalarSeries": [],
                "histograms": [],
                "images": [],
            },
        )


class TrainingMonitorDataTests(unittest.TestCase):
    def assert_empty_live_monitor_payload(
        self,
        data: dict[str, object],
        *,
        job_id: str,
        node_path: str,
        dataset: str | None,
        preset: str | None,
        log_dir: str | None = None,
    ) -> None:
        self.assertEqual(
            data,
            {
                "jobId": job_id,
                "nodePath": node_path,
                "dataset": dataset,
                "logDir": log_dir,
                "scalarSeries": [],
                "histograms": [],
                "images": [],
                "preset": preset,
            },
        )

    def test_training_job_monitor_data_without_log_dir_returns_empty_payload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                },
            )

            data = manager.get_monitor_data(
                payload["id"],
                node_path="main_model.0.model",
                dataset="Mnist",
            )

        self.assert_empty_live_monitor_payload(
            data,
            job_id=payload["id"],
            node_path="main_model.0.model",
            dataset="Mnist",
            preset=None,
        )

    def test_training_job_monitor_data_rejects_unknown_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )

            with self.assertRaises(InspectorError) as caught:
                manager.get_monitor_data(
                    payload["id"],
                    node_path="main_model.0.model",
                    dataset="Cifar10",
                )

        self.assertEqual(
            str(caught.exception),
            f"Unknown dataset 'Cifar10' for training job '{payload['id']}'.",
        )

    def test_training_job_monitor_data_rejects_unknown_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )

            with self.assertRaises(InspectorError) as caught:
                manager.get_monitor_data(
                    payload["id"],
                    node_path="main_model.0.model",
                    dataset="Mnist",
                    preset="gating",
                )

        self.assertEqual(
            str(caught.exception),
            f"Unknown preset 'gating' for training job '{payload['id']}'.",
        )

    def test_training_job_monitor_data_without_matching_log_dir_returns_empty_payload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist", "Cifar10"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Cifar10",
                    "logDir": str(Path(tmp) / "logs" / "cifar10"),
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "gating",
                    "dataset": "Mnist",
                },
            )

            data = manager.get_monitor_data(
                payload["id"],
                node_path="main_model.0.model",
                dataset="Mnist",
                preset="gating",
            )

        self.assert_empty_live_monitor_payload(
            data,
            job_id=payload["id"],
            node_path="main_model.0.model",
            dataset="Mnist",
            preset="gating",
        )

    def test_training_job_monitor_data_rejects_log_dirs_outside_job_experiment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            outside_log_dir = root / "outside-run"
            other_experiment = logs_root / "other_model" / "run"
            symlink_escape = logs_root / "test_model" / "linked"
            outside_log_dir.mkdir()
            other_experiment.mkdir(parents=True)
            symlink_escape.parent.mkdir(parents=True)
            symlink_escape.symlink_to(outside_log_dir, target_is_directory=True)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            for untrusted_log_dir in (
                outside_log_dir,
                other_experiment,
                logs_root / "test_model" / ".." / "other_model" / "run",
                symlink_escape,
            ):
                with self.subTest(log_dir=untrusted_log_dir):
                    manager.runtime._write_event(
                        job,
                        {
                            "type": "dataset_started",
                            "status": "running",
                            "preset": "baseline",
                            "dataset": "Mnist",
                            "logDir": str(untrusted_log_dir),
                        },
                    )

                    with self.assertRaisesRegex(
                        InspectorError,
                        "outside this Training Job's Log Experiment",
                    ):
                        manager.get_monitor_data(
                            payload["id"],
                            node_path="main_model.0.model",
                            dataset="Mnist",
                            preset="baseline",
                        )

    def test_training_job_monitor_data_filters_tensorboard_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            log_dir = Path(tmp) / "logs" / "test_model" / "run"
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar("main_model.0.model/output/mean", 0.12, 100)
            writer.add_scalar("main_model.1.model/output/mean", 0.99, 100)
            writer.add_histogram(
                "main_model.0.model/histogram/usage_fraction",
                torch.tensor([0.05, 0.15, 0.2]),
                100,
            )
            writer.add_image(
                "main_model.0.model/heatmap/usage_fraction",
                torch.ones(1, 2, 2),
                100,
                dataformats="CHW",
            )
            writer.flush()
            writer.close()

            manager = TrainingJobServiceHarness(
                root=root,
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "dataset": "Mnist",
                    "logDir": str(log_dir),
                },
            )

            data = manager.get_monitor_data(
                payload["id"],
                node_path="main_model.0.model",
                dataset="Mnist",
            )
            unmatched = manager.get_monitor_data(
                payload["id"],
                node_path="missing",
                dataset="Mnist",
            )

        self.assertEqual(data["jobId"], payload["id"])
        self.assertEqual(data["nodePath"], "main_model.0.model")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertEqual(
            data["scalarSeries"][0]["tag"], "main_model.0.model/output/mean"
        )
        self.assertEqual(data["scalarSeries"][0]["label"], "output/mean")
        self.assertEqual(data["scalarSeries"][0]["points"][0]["step"], 100)
        self.assertEqual(len(data["histograms"]), 1)
        self.assertEqual(len(data["images"]), 1)
        self.assertTrue(
            data["images"][0]["dataUrl"].startswith("data:image/png;base64,")
        )
        self.assertEqual(unmatched["scalarSeries"], [])
        self.assertEqual(unmatched["histograms"], [])
        self.assertEqual(unmatched["images"], [])

    def test_training_job_monitor_data_filters_by_preset_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            baseline_dir = Path(tmp) / "logs" / "test_model" / "baseline"
            gating_dir = Path(tmp) / "logs" / "test_model" / "gating"
            baseline_writer = SummaryWriter(log_dir=str(baseline_dir))
            baseline_writer.add_scalar("main_model.0.model/output/mean", 0.12, 100)
            baseline_writer.flush()
            baseline_writer.close()
            gating_writer = SummaryWriter(log_dir=str(gating_dir))
            gating_writer.add_scalar("main_model.0.model/output/mean", 0.88, 100)
            gating_writer.flush()
            gating_writer.close()

            manager = TrainingJobServiceHarness(
                root=root,
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(baseline_dir),
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "gating",
                    "dataset": "Mnist",
                    "logDir": str(gating_dir),
                },
            )

            data = manager.get_monitor_data(
                payload["id"],
                node_path="main_model.0.model",
                preset="gating",
                dataset="Mnist",
            )

        self.assertEqual(data["preset"], "gating")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertAlmostEqual(data["scalarSeries"][0]["points"][0]["value"], 0.88)

    def test_training_job_parameter_status_endpoint_filters_by_preset_and_dataset(
        self,
    ) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings

        async def call_api(app, job_id: str) -> httpx.Response:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get(
                    f"/training/jobs/{job_id}/monitor-parameter-status",
                    params={"preset": "gating", "dataset": "Mnist"},
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            logs_root = Path(tmp) / "logs"
            baseline_dir = logs_root / "test_model" / "baseline"
            gating_dir = logs_root / "test_model" / "gating"

            baseline_writer = SummaryWriter(log_dir=str(baseline_dir))
            baseline_writer.add_scalar(
                "main_model.0.model/weights/relative_delta_norm",
                0.0,
                10,
            )
            baseline_writer.flush()
            baseline_writer.close()
            gating_writer = SummaryWriter(log_dir=str(gating_dir))
            gating_writer.add_scalar(
                "main_model.0.model/weights/relative_delta_norm",
                1e-6,
                20,
            )
            gating_writer.flush()
            gating_writer.close()

            manager = TrainingJobServiceHarness(
                root=root,
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(baseline_dir),
                },
            )
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "gating",
                    "dataset": "Mnist",
                    "logDir": str(gating_dir),
                },
            )
            app = create_app_with_training_service(
                WorkbenchApiSettings(logs_root=str(logs_root)),
                manager,
            )

            response = asyncio.run(call_api(app, payload["id"]))

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["sourceId"], payload["id"])
        self.assertEqual(data["preset"], "gating")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertEqual(data["logDir"], str(gating_dir))
        self.assertEqual(data["nodes"][0]["weights"]["status"], "updated")
        self.assertEqual(data["nodes"][0]["weights"]["lastStep"], 20)

    def test_training_job_parameter_status_rejects_untrusted_event_log_dir(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=root / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager.runtime._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(root / "outside-run"),
                },
            )

            with self.assertRaisesRegex(
                InspectorError,
                "outside this Training Job's Log Experiment",
            ):
                manager.get_parameter_status(
                    payload["id"],
                    dataset="Mnist",
                    preset="baseline",
                )

    def test_log_run_monitor_data_filters_tensorboard_tags(self) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            log_dir = logs_root.joinpath(
                "test_model",
                "linear",
                "BASELINE",
                "Mnist",
                "historical_20260601_010203",
                "version_0",
            )
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_scalar("main_model.0.model/output/mean", 0.12, 100)
            writer.add_scalar("main_model.1.model/output/mean", 0.99, 100)
            writer.add_histogram(
                "main_model.0.model/histogram/usage_fraction",
                torch.tensor([0.05, 0.15, 0.2]),
                100,
            )
            writer.add_image(
                "main_model.0.model/heatmap/usage_fraction",
                torch.ones(1, 2, 2),
                100,
                dataformats="CHW",
            )
            writer.flush()
            writer.close()

            run_id = LogRunScanner(logs_root=logs_root).list_runs()[0].id

            async def call_api() -> tuple[
                httpx.Response, httpx.Response, httpx.Response
            ]:
                transport = httpx.ASGITransport(
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    data_response = await client.get(
                        f"/logs/runs/{run_id}/monitor-data",
                        params={"nodePath": "main_model.0.model"},
                    )
                    unmatched_response = await client.get(
                        f"/logs/runs/{run_id}/monitor-data",
                        params={"nodePath": "missing"},
                    )
                    unknown_response = await client.get(
                        "/logs/runs/not-a-run/monitor-data",
                        params={"nodePath": "main_model.0.model"},
                    )
                    return data_response, unmatched_response, unknown_response

            data_response, unmatched_response, unknown_response = asyncio.run(
                call_api()
            )

        self.assertEqual(data_response.status_code, 200)
        data = data_response.json()
        self.assertEqual(data["jobId"], run_id)
        self.assertEqual(data["nodePath"], "main_model.0.model")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertEqual(
            data["scalarSeries"][0]["tag"], "main_model.0.model/output/mean"
        )
        self.assertEqual(data["scalarSeries"][0]["label"], "output/mean")
        self.assertEqual(data["scalarSeries"][0]["points"][0]["step"], 100)
        self.assertEqual(len(data["histograms"]), 1)
        self.assertEqual(len(data["images"]), 1)
        self.assertTrue(
            data["images"][0]["dataUrl"].startswith("data:image/png;base64,")
        )

        self.assertEqual(unmatched_response.status_code, 200)
        unmatched = unmatched_response.json()
        self.assertEqual(unmatched["jobId"], run_id)
        self.assertEqual(unmatched["dataset"], "Mnist")
        self.assertEqual(unmatched["scalarSeries"], [])
        self.assertEqual(unmatched["histograms"], [])
        self.assertEqual(unmatched["images"], [])

        self.assertEqual(unknown_response.status_code, 400)
        self.assertIn("Unknown log run id", unknown_response.json()["detail"])

    def test_log_parameter_status_endpoint_resolves_runs_and_rejects_unknown_ids(
        self,
    ) -> None:
        import httpx

        from workbench.backend.api import WorkbenchApiSettings, create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            first_dir = logs_root.joinpath(
                "test_model",
                "linear",
                "baseline",
                "Mnist",
                "historical_20260601_010203",
                "version_0",
            )
            second_dir = logs_root.joinpath(
                "test_model",
                "linear",
                "gating",
                "Mnist",
                "historical_20260601_020304",
                "version_0",
            )
            first_writer = SummaryWriter(log_dir=str(first_dir))
            first_writer.add_scalar(
                "main_model.0.model/weights/relative_delta_norm",
                0.0,
                10,
            )
            first_writer.add_scalar(
                "main_model.0.model/weights/relative_delta_norm",
                0.0,
                11,
            )
            first_writer.flush()
            first_writer.close()
            second_writer = SummaryWriter(log_dir=str(second_dir))
            second_writer.add_scalar(
                "main_model.0.model/weights/relative_delta_norm",
                1e-6,
                20,
            )
            second_writer.flush()
            second_writer.close()

            runs_by_preset = {
                run.preset: run
                for run in LogRunScanner(logs_root=logs_root).list_runs()
            }
            run_ids = [
                runs_by_preset["baseline"].id,
                runs_by_preset["gating"].id,
            ]

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                transport = httpx.ASGITransport(
                    app=create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                )
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    status_response = await client.post(
                        "/logs/parameter-status",
                        json={"runIds": run_ids},
                    )
                    unknown_response = await client.post(
                        "/logs/parameter-status",
                        json={"runIds": ["not-a-run"]},
                    )
                    return status_response, unknown_response

            status_response, unknown_response = asyncio.run(call_api())

        self.assertEqual(status_response.status_code, 200, status_response.text)
        payload = status_response.json()
        self.assertEqual([run["sourceId"] for run in payload["runs"]], run_ids)
        self.assertEqual(payload["runs"][0]["preset"], "baseline")
        self.assertEqual(
            payload["runs"][0]["nodes"][0]["weights"]["status"],
            "unchanged",
        )
        self.assertEqual(payload["runs"][1]["preset"], "gating")
        self.assertEqual(
            payload["runs"][1]["nodes"][0]["weights"]["status"],
            "updated",
        )
        self.assertEqual(unknown_response.status_code, 400)
        self.assertIn("Unknown log run id", unknown_response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
