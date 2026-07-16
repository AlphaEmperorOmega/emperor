from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.utils.tensorboard import SummaryWriter

from emperor_workbench.training_jobs import TrainingJobFailure
from tests.support import lifespan_client
from tests.support.model_packages import list_log_runs
from tests.support.training_jobs import (
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
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
            manager.append_progress_event(
                str(payload["id"]),
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

            with self.assertRaises(TrainingJobFailure) as caught:
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

            with self.assertRaises(TrainingJobFailure) as caught:
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
            manager.append_progress_event(
                str(payload["id"]),
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Cifar10",
                    "logDir": str(Path(tmp) / "logs" / "cifar10"),
                },
            )
            manager.append_progress_event(
                str(payload["id"]),
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
            for untrusted_log_dir in (
                outside_log_dir,
                other_experiment,
                logs_root / "test_model" / ".." / "other_model" / "run",
                symlink_escape,
            ):
                with self.subTest(log_dir=untrusted_log_dir):
                    manager.append_progress_event(
                        str(payload["id"]),
                        {
                            "type": "dataset_started",
                            "status": "running",
                            "preset": "baseline",
                            "dataset": "Mnist",
                            "logDir": str(untrusted_log_dir),
                        },
                    )

                    with self.assertRaisesRegex(
                        TrainingJobFailure,
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
            manager.append_progress_event(
                str(payload["id"]),
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
            manager.append_progress_event(
                str(payload["id"]),
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(baseline_dir),
                },
            )
            manager.append_progress_event(
                str(payload["id"]),
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

        from emperor_workbench.settings import WorkbenchApiSettings

        async def call_api(app, job_id: str) -> httpx.Response:
            async with lifespan_client(app) as client:
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
            manager.append_progress_event(
                str(payload["id"]),
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(baseline_dir),
                },
            )
            manager.append_progress_event(
                str(payload["id"]),
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
            manager.append_progress_event(
                str(payload["id"]),
                {
                    "type": "dataset_started",
                    "status": "running",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "logDir": str(root / "outside-run"),
                },
            )

            with self.assertRaisesRegex(
                TrainingJobFailure,
                "outside this Training Job's Log Experiment",
            ):
                manager.get_parameter_status(
                    payload["id"],
                    dataset="Mnist",
                    preset="baseline",
                )

    def test_log_run_monitor_data_filters_tensorboard_tags(self) -> None:
        import httpx

        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

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

            run_id = list_log_runs(logs_root=logs_root)[0].id

            async def call_api() -> tuple[
                httpx.Response, httpx.Response, httpx.Response
            ]:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(app) as client:
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

        from emperor_workbench.api import create_app
        from emperor_workbench.settings import WorkbenchApiSettings

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
                run.preset: run for run in list_log_runs(logs_root=logs_root)
            }
            run_ids = [
                runs_by_preset["baseline"].id,
                runs_by_preset["gating"].id,
            ]

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(app) as client:
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
