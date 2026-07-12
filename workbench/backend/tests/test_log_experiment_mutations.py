from __future__ import annotations

import asyncio
import io
import shutil
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx

from workbench.backend.api import WorkbenchApiSettings
from workbench.backend.core.errors import ApiError
from workbench.backend.log_experiments import (
    LogExperimentMutationCoordinator,
)
from workbench.backend.run_history import service as run_history_service_module
from workbench.backend.run_history.scanner import LogRunScanner
from workbench.backend.tests.helpers import (
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
    write_tensorboard_run,
)


def _zip_bytes(entries: dict[str, str]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name, content in entries.items():
            zip_file.writestr(name, content)
    return archive.getvalue()


async def _wait_for_thread_event(event: threading.Event, label: str) -> None:
    for _attempt in range(500):
        if event.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"Timed out waiting for {label}")


class LogExperimentMutationApiTests(unittest.TestCase):
    @staticmethod
    def _create_app(root: Path):
        logs_root = root / "logs"
        write_tensorboard_run(
            logs_root,
            [
                "shared_experiment",
                "linear",
                "BASELINE",
                "Mnist",
                "run_20260710_120000",
                "version_0",
            ],
        )
        manager = TrainingJobServiceHarness(
            root=root / "jobs",
            logs_root=logs_root,
            runner=FakeRunner(),
        )
        app = create_app_with_training_service(
            WorkbenchApiSettings(
                logs_root=str(logs_root),
                allow_unsafe_local_mutations=True,
                allow_log_imports=True,
            ),
            manager,
        )
        return app, manager, logs_root

    @staticmethod
    def _training_request() -> dict[str, object]:
        return {
            "modelType": "linears",
            "model": "linear",
            "preset": "baseline",
            "datasets": ["Mnist"],
            "overrides": {},
            "logFolder": "shared_experiment",
            "monitors": [],
        }

    @staticmethod
    def _filtered_delete_request(logs_root: Path) -> dict[str, object]:
        run = LogRunScanner(logs_root=logs_root).list_runs()[0]
        return {
            "experiments": [run.experiment],
            "datasets": [run.dataset],
            "models": [{"modelType": "linears", "model": "linear"}],
            "presets": [run.preset],
            "runIds": [run.id],
        }

    def test_delete_snapshot_serializes_training_job_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app, _manager, logs_root = self._create_app(root)
            training_service = app.state.workbench_services.training_jobs
            original_active_jobs = training_service.active_jobs
            snapshot_taken = threading.Event()
            release_snapshot = threading.Event()

            def paused_active_jobs():
                snapshot = original_active_jobs()
                snapshot_taken.set()
                if not release_snapshot.wait(timeout=5):
                    raise AssertionError(
                        "Timed out waiting to release blocker snapshot"
                    )
                return snapshot

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    delete_task = asyncio.create_task(
                        client.delete("/logs/experiments/shared_experiment")
                    )
                    await _wait_for_thread_event(
                        snapshot_taken,
                        "delete blocker snapshot",
                    )
                    start_task = asyncio.create_task(
                        client.post(
                            "/training/jobs",
                            json=self._training_request(),
                        )
                    )
                    await asyncio.sleep(0.1)
                    start_completed_before_delete = start_task.done()
                    release_snapshot.set()
                    return (
                        await delete_task,
                        await start_task,
                        start_completed_before_delete,
                    )

            with patch.object(
                training_service,
                "active_jobs",
                side_effect=paused_active_jobs,
            ):
                delete_response, start_response, start_completed = asyncio.run(
                    race_requests()
                )
            experiment_exists = logs_root.joinpath("shared_experiment").is_dir()

        self.assertFalse(
            start_completed,
            "Training Job started after the delete blocker snapshot was captured",
        )
        self.assertEqual(delete_response.status_code, 200, delete_response.text)
        self.assertEqual(start_response.status_code, 200, start_response.text)
        self.assertTrue(experiment_exists)

    def test_training_job_start_serializes_experiment_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, manager, logs_root = self._create_app(Path(tmp))
            original_create_job = manager.runtime.create_job_from_command
            start_entered = threading.Event()
            release_start = threading.Event()

            def paused_create_job(command):
                start_entered.set()
                if not release_start.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release job start")
                return original_create_job(command)

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    start_task = asyncio.create_task(
                        client.post(
                            "/training/jobs",
                            json=self._training_request(),
                        )
                    )
                    await _wait_for_thread_event(start_entered, "Training Job start")
                    delete_task = asyncio.create_task(
                        client.delete("/logs/experiments/shared_experiment")
                    )
                    await asyncio.sleep(0.1)
                    delete_completed_before_start = delete_task.done()
                    release_start.set()
                    return (
                        await start_task,
                        await delete_task,
                        delete_completed_before_start,
                    )

            with patch.object(
                manager.runtime,
                "create_job_from_command",
                side_effect=paused_create_job,
            ):
                start_response, delete_response, delete_completed = asyncio.run(
                    race_requests()
                )
            experiment_exists = logs_root.joinpath("shared_experiment").is_dir()

        self.assertFalse(delete_completed)
        self.assertEqual(start_response.status_code, 200, start_response.text)
        self.assertEqual(delete_response.status_code, 400, delete_response.text)
        self.assertIn("training job is still writing", delete_response.text.lower())
        self.assertTrue(experiment_exists)

    def test_training_job_start_serializes_filtered_run_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, manager, logs_root = self._create_app(Path(tmp))
            delete_request = self._filtered_delete_request(logs_root)
            original_create_job = manager.runtime.create_job_from_command
            start_entered = threading.Event()
            release_start = threading.Event()

            def paused_create_job(command):
                start_entered.set()
                if not release_start.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release job start")
                return original_create_job(command)

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    start_task = asyncio.create_task(
                        client.post("/training/jobs", json=self._training_request())
                    )
                    await _wait_for_thread_event(start_entered, "Training Job start")
                    delete_task = asyncio.create_task(
                        client.post("/logs/runs/delete", json=delete_request)
                    )
                    await asyncio.sleep(0.1)
                    delete_completed_before_start = delete_task.done()
                    release_start.set()
                    return (
                        await start_task,
                        await delete_task,
                        delete_completed_before_start,
                    )

            with patch.object(
                manager.runtime,
                "create_job_from_command",
                side_effect=paused_create_job,
            ):
                start_response, delete_response, delete_completed = asyncio.run(
                    race_requests()
                )

        self.assertFalse(delete_completed)
        self.assertEqual(start_response.status_code, 200, start_response.text)
        self.assertEqual(delete_response.status_code, 400, delete_response.text)
        self.assertIn("training job is still writing", delete_response.text.lower())

    def test_filtered_run_delete_serializes_training_job_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _manager, logs_root = self._create_app(Path(tmp))
            delete_request = self._filtered_delete_request(logs_root)
            original_rmtree = shutil.rmtree
            delete_entered = threading.Event()
            release_delete = threading.Event()

            def paused_rmtree(path: Path):
                delete_entered.set()
                if not release_delete.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release Run delete")
                return original_rmtree(path)

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    delete_task = asyncio.create_task(
                        client.post("/logs/runs/delete", json=delete_request)
                    )
                    await _wait_for_thread_event(delete_entered, "filtered Run delete")
                    start_task = asyncio.create_task(
                        client.post("/training/jobs", json=self._training_request())
                    )
                    await asyncio.sleep(0.1)
                    start_completed_before_delete = start_task.done()
                    release_delete.set()
                    return (
                        await delete_task,
                        await start_task,
                        start_completed_before_delete,
                    )

            with patch(
                "workbench.backend.run_history.deletion.shutil.rmtree",
                side_effect=paused_rmtree,
            ):
                delete_response, start_response, start_completed = asyncio.run(
                    race_requests()
                )

        self.assertFalse(start_completed)
        self.assertEqual(delete_response.status_code, 200, delete_response.text)
        self.assertEqual(start_response.status_code, 200, start_response.text)

    def test_archive_import_serializes_training_job_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _manager, logs_root = self._create_app(Path(tmp))
            original_import_archive = run_history_service_module.import_log_archive
            import_entered = threading.Event()
            release_import = threading.Event()
            archive = _zip_bytes(
                {"shared_experiment/imported/result.json": "imported"}
            )

            def paused_import_archive(**kwargs):
                import_entered.set()
                if not release_import.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release log import")
                return original_import_archive(**kwargs)

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    import_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            files={
                                "archive": ("logs.zip", archive, "application/zip")
                            },
                        )
                    )
                    await _wait_for_thread_event(import_entered, "log import")
                    start_task = asyncio.create_task(
                        client.post(
                            "/training/jobs",
                            json=self._training_request(),
                        )
                    )
                    await asyncio.sleep(0.1)
                    start_completed_before_import = start_task.done()
                    release_import.set()
                    return (
                        await import_task,
                        await start_task,
                        start_completed_before_import,
                    )

            with patch.object(
                run_history_service_module,
                "import_log_archive",
                side_effect=paused_import_archive,
            ):
                import_response, start_response, start_completed = asyncio.run(
                    race_requests()
                )
            imported = logs_root.joinpath(
                "shared_experiment/imported/result.json"
            ).read_text(encoding="utf-8")

        self.assertFalse(start_completed)
        self.assertEqual(import_response.status_code, 200, import_response.text)
        self.assertEqual(start_response.status_code, 200, start_response.text)
        self.assertEqual(imported, "imported")

    def test_training_job_start_serializes_archive_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, manager, logs_root = self._create_app(Path(tmp))
            original_create_job = manager.runtime.create_job_from_command
            start_entered = threading.Event()
            release_start = threading.Event()
            archive = _zip_bytes(
                {"shared_experiment/imported/result.json": "imported"}
            )

            def paused_create_job(command):
                start_entered.set()
                if not release_start.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release job start")
                return original_create_job(command)

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    start_task = asyncio.create_task(
                        client.post(
                            "/training/jobs",
                            json=self._training_request(),
                        )
                    )
                    await _wait_for_thread_event(start_entered, "Training Job start")
                    import_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            files={
                                "archive": ("logs.zip", archive, "application/zip")
                            },
                        )
                    )
                    await asyncio.sleep(0.1)
                    import_completed_before_start = import_task.done()
                    release_start.set()
                    return (
                        await start_task,
                        await import_task,
                        import_completed_before_start,
                    )

            with patch.object(
                manager.runtime,
                "create_job_from_command",
                side_effect=paused_create_job,
            ):
                start_response, import_response, import_completed = asyncio.run(
                    race_requests()
                )
            imported_exists = logs_root.joinpath(
                "shared_experiment/imported/result.json"
            ).exists()

        self.assertFalse(import_completed)
        self.assertEqual(start_response.status_code, 200, start_response.text)
        self.assertEqual(import_response.status_code, 400, import_response.text)
        self.assertIn("training job is still writing", import_response.text.lower())
        self.assertFalse(imported_exists)

    def test_experiment_delete_serializes_archive_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _manager, logs_root = self._create_app(Path(tmp))
            logs_root.joinpath("shared_experiment/imported").write_text(
                "path that deletion removes",
                encoding="utf-8",
            )
            original_rmtree = shutil.rmtree
            delete_entered = threading.Event()
            release_delete = threading.Event()
            archive = _zip_bytes(
                {"shared_experiment/imported/result.json": "imported"}
            )

            def paused_delete_experiment(path: Path):
                delete_entered.set()
                if not release_delete.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release log delete")
                return original_rmtree(path)

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    delete_task = asyncio.create_task(
                        client.delete("/logs/experiments/shared_experiment")
                    )
                    await _wait_for_thread_event(delete_entered, "log delete")
                    import_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            files={
                                "archive": ("logs.zip", archive, "application/zip")
                            },
                        )
                    )
                    await asyncio.sleep(0.1)
                    import_completed_before_delete = import_task.done()
                    release_delete.set()
                    return (
                        await delete_task,
                        await import_task,
                        import_completed_before_delete,
                    )

            with patch(
                "workbench.backend.run_history.deletion.shutil.rmtree",
                side_effect=paused_delete_experiment,
            ):
                delete_response, import_response, import_completed = asyncio.run(
                    race_requests()
                )
            imported = logs_root.joinpath(
                "shared_experiment/imported/result.json"
            ).read_text(encoding="utf-8")

        self.assertFalse(import_completed)
        self.assertEqual(delete_response.status_code, 200, delete_response.text)
        self.assertEqual(import_response.status_code, 200, import_response.text)
        self.assertEqual(imported, "imported")

    def test_archive_import_serializes_experiment_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app, _manager, logs_root = self._create_app(Path(tmp))
            original_import_archive = run_history_service_module.import_log_archive
            import_entered = threading.Event()
            release_import = threading.Event()
            archive = _zip_bytes(
                {"shared_experiment/imported/result.json": "imported"}
            )

            def paused_import_archive(**kwargs):
                import_entered.set()
                if not release_import.wait(timeout=5):
                    raise AssertionError("Timed out waiting to release log import")
                return original_import_archive(**kwargs)

            async def race_requests():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    headers={"X-Workbench-Mutation": "true"},
                ) as client:
                    import_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            files={
                                "archive": ("logs.zip", archive, "application/zip")
                            },
                        )
                    )
                    await _wait_for_thread_event(import_entered, "log import")
                    delete_task = asyncio.create_task(
                        client.delete("/logs/experiments/shared_experiment")
                    )
                    await asyncio.sleep(0.1)
                    delete_completed_before_import = delete_task.done()
                    release_import.set()
                    return (
                        await import_task,
                        await delete_task,
                        delete_completed_before_import,
                    )

            with patch.object(
                run_history_service_module,
                "import_log_archive",
                side_effect=paused_import_archive,
            ):
                import_response, delete_response, delete_completed = asyncio.run(
                    race_requests()
                )
            experiment_exists = logs_root.joinpath("shared_experiment").exists()

        self.assertFalse(delete_completed)
        self.assertEqual(import_response.status_code, 200, import_response.text)
        self.assertEqual(delete_response.status_code, 200, delete_response.text)
        self.assertFalse(experiment_exists)


class LogExperimentMutationCoordinatorTests(unittest.TestCase):
    def test_reversed_multi_experiment_requests_use_one_stable_lock_order(
        self,
    ) -> None:
        coordinator = LogExperimentMutationCoordinator()
        ready = threading.Barrier(2)
        entered: list[str] = []

        def coordinate(label: str, experiments: list[str]) -> None:
            ready.wait(timeout=5)
            with coordinator.coordinate(experiments):
                entered.append(label)

        first = threading.Thread(
            target=coordinate,
            args=("first", ["alpha", "beta"]),
        )
        second = threading.Thread(
            target=coordinate,
            args=("second", ["beta", "alpha"]),
        )
        first.start()
        second.start()
        first.join(timeout=5)
        second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertCountEqual(entered, ["first", "second"])

    def test_different_experiments_do_not_block_each_other(self) -> None:
        coordinator = LogExperimentMutationCoordinator()
        first_held = threading.Event()
        release_first = threading.Event()

        def hold_first_experiment() -> None:
            with coordinator.coordinate(["first"]):
                first_held.set()
                release_first.wait(timeout=5)

        holder = threading.Thread(target=hold_first_experiment)
        holder.start()
        self.assertTrue(first_held.wait(timeout=5))
        try:
            with coordinator.coordinate(["second"]):
                pass
        finally:
            release_first.set()
            holder.join(timeout=5)

        self.assertFalse(holder.is_alive())

    def test_scope_releases_after_exception_and_cancellation(self) -> None:
        coordinator = LogExperimentMutationCoordinator()

        with self.assertRaisesRegex(RuntimeError, "operation failed"):
            with coordinator.coordinate(["experiment"]):
                raise RuntimeError("operation failed")
        with coordinator.coordinate(["experiment"]):
            pass

        with self.assertRaises(asyncio.CancelledError):
            with coordinator.coordinate(["experiment"]):
                raise asyncio.CancelledError
        with coordinator.coordinate(["experiment"]):
            pass

    def test_timeout_releases_partial_scope_and_later_callers_can_continue(
        self,
    ) -> None:
        coordinator = LogExperimentMutationCoordinator(
            acquire_timeout_seconds=0.05
        )
        held = threading.Event()
        release = threading.Event()

        def hold_second_experiment() -> None:
            with coordinator.coordinate(["second"]):
                held.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_second_experiment)
        holder.start()
        self.assertTrue(held.wait(timeout=5))
        try:
            with self.assertRaises(ApiError) as context:
                with coordinator.coordinate(["first", "second"]):
                    pass
            self.assertEqual(context.exception.status_code, 503)
            with coordinator.coordinate(["first"]):
                pass
        finally:
            release.set()
            holder.join(timeout=5)

        self.assertFalse(holder.is_alive())
        with coordinator.coordinate(["second"]):
            pass


if __name__ == "__main__":
    unittest.main()
