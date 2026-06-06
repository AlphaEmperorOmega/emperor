from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.utils.tensorboard import SummaryWriter

from viewer.backend.inspector.errors import InspectorError
from viewer.backend.log_runs import LogRunIndex
from viewer.backend.monitor_data import TensorBoardMonitorReader
from viewer.backend.tests.helpers import FakeRunner
from viewer.backend.training_jobs import TrainingJobManager


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


class TensorBoardMonitorReaderFailureTests(unittest.TestCase):
    def read_with_accumulators(
        self,
        log_dir: Path,
        accumulators: list[object | None],
    ) -> dict:
        run_dirs = [log_dir / f"run-{index}" for index, _ in enumerate(accumulators)]
        with (
            patch("viewer.backend.monitor_data.event_dirs", return_value=run_dirs),
            patch(
                "viewer.backend.monitor_data.load_event_accumulator",
                side_effect=accumulators,
            ),
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
        run = LogRunIndex(logs_root=logs_root).list_runs()[0]
        return run.id, run_dir

    def test_broken_historical_tags_return_empty_monitor_response_shape(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

        async def call_api(logs_root: Path, run_id: str) -> httpx.Response:
            transport = httpx.ASGITransport(
                app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
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
                "viewer.backend.monitor_data.load_event_accumulator",
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
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
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
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
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
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
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
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist", "Cifar10"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Cifar10",
                    "logDir": str(Path(tmp) / "logs" / "cifar10"),
                },
            )
            manager._write_event(
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

    def test_training_job_monitor_data_trusts_existing_log_dir_outside_logs_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outside_log_dir = Path(tmp) / "outside-run"
            writer = SummaryWriter(log_dir=str(outside_log_dir))
            writer.add_scalar("main_model.0.model/output/mean", 0.42, 7)
            writer.flush()
            writer.close()

            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(outside_log_dir),
                },
            )

            data = manager.get_monitor_data(
                payload["id"],
                node_path="main_model.0.model",
                dataset="Mnist",
                preset="baseline",
            )

        self.assertEqual(data["logDir"], str(outside_log_dir))
        self.assertEqual(data["preset"], "baseline")
        self.assertEqual(data["dataset"], "Mnist")
        self.assertEqual(
            data["scalarSeries"][0]["tag"], "main_model.0.model/output/mean"
        )
        self.assertEqual(data["scalarSeries"][0]["points"][0]["step"], 7)
        self.assertAlmostEqual(data["scalarSeries"][0]["points"][0]["value"], 0.42)

    def test_training_job_monitor_data_filters_tensorboard_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            log_dir = Path(tmp) / "logs" / "run"
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

            manager = TrainingJobManager(
                root=root,
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
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
            baseline_dir = Path(tmp) / "logs" / "baseline"
            gating_dir = Path(tmp) / "logs" / "gating"
            baseline_writer = SummaryWriter(log_dir=str(baseline_dir))
            baseline_writer.add_scalar("main_model.0.model/output/mean", 0.12, 100)
            baseline_writer.flush()
            baseline_writer.close()
            gating_writer = SummaryWriter(log_dir=str(gating_dir))
            gating_writer.add_scalar("main_model.0.model/output/mean", 0.88, 100)
            gating_writer.flush()
            gating_writer.close()

            manager = TrainingJobManager(
                root=root,
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            payload = manager.create_job(
                model="linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=["linear"],
            )
            job = manager.jobs[payload["id"]]
            manager._write_event(
                job,
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(baseline_dir),
                },
            )
            manager._write_event(
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

    def test_log_run_monitor_data_filters_tensorboard_tags(self) -> None:
        import httpx
        from viewer.backend.api import ViewerApiSettings, create_app

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

            run_id = LogRunIndex(logs_root=logs_root).list_runs()[0].id

            async def call_api() -> (
                tuple[httpx.Response, httpx.Response, httpx.Response]
            ):
                transport = httpx.ASGITransport(
                    app=create_app(ViewerApiSettings(logs_root=str(logs_root)))
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

if __name__ == "__main__":
    unittest.main()
