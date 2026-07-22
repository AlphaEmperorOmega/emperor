from __future__ import annotations

import asyncio
import io
import shutil
import tempfile
import threading
import unittest
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx

from emperor_workbench.settings import WorkbenchApiSettings
from tests.support.model_packages import list_log_runs
from tests.support.training_jobs import (
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
    write_tensorboard_run,
)

THREAD_COORDINATION_TIMEOUT_SECONDS = 30.0


def _zip_bytes(entries: dict[str, str]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zip_file:
        for name, content in entries.items():
            zip_file.writestr(name, content)
    return archive.getvalue()


async def _wait_for_thread_event(event: threading.Event, label: str) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + THREAD_COORDINATION_TIMEOUT_SECONDS
    while not event.is_set():
        remaining = deadline - loop.time()
        if remaining <= 0:
            if event.is_set():
                return
            raise AssertionError(f"Timed out waiting for {label}")
        await asyncio.sleep(min(0.01, remaining))


@asynccontextmanager
async def _lifespan_client(app):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://localhost",
            headers={
                "X-Workbench-Mutation": "true",
                "Idempotency-Key": uuid.uuid4().hex,
            },
        ) as client:
            yield client


class LogExperimentMutationApiTests(unittest.TestCase):
    @staticmethod
    def _create_app(root: Path):
        logs_root = root / "logs"
        write_tensorboard_run(
            logs_root,
            [
                "shared_experiment",
                "linears",
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
        run = list_log_runs(logs_root=logs_root)[0]
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
            app, manager, logs_root = self._create_app(root)
            snapshot_taken = threading.Event()
            release_snapshot = threading.Event()

            async def race_requests():
                async with _lifespan_client(app) as client:
                    training_service = manager.service
                    original_active_jobs = training_service.active_jobs

                    def paused_active_jobs():
                        snapshot = original_active_jobs()
                        snapshot_taken.set()
                        if not release_snapshot.wait(
                            timeout=THREAD_COORDINATION_TIMEOUT_SECONDS
                        ):
                            raise AssertionError(
                                "Timed out waiting to release blocker snapshot"
                            )
                        return snapshot

                    with patch.object(
                        training_service,
                        "active_jobs",
                        side_effect=paused_active_jobs,
                    ):
                        delete_task = asyncio.create_task(
                            client.delete("/logs/experiments/shared_experiment")
                        )
                        loop = asyncio.get_running_loop()
                        deadline = loop.time() + THREAD_COORDINATION_TIMEOUT_SECONDS
                        while not snapshot_taken.is_set():
                            if delete_task.done():
                                response = await delete_task
                                raise AssertionError(
                                    "Delete completed before reading active "
                                    f"writers: {response.status_code} "
                                    f"{response.text}"
                                )
                            remaining = deadline - loop.time()
                            if remaining <= 0:
                                if snapshot_taken.is_set():
                                    break
                                raise AssertionError(
                                    "Timed out waiting for delete blocker snapshot"
                                )
                            await asyncio.sleep(min(0.01, remaining))
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
            original_materialize = manager.run_plans.materialize
            start_entered = threading.Event()
            release_start = threading.Event()

            def paused_materialize(*args, **kwargs):
                start_entered.set()
                if not release_start.wait(timeout=THREAD_COORDINATION_TIMEOUT_SECONDS):
                    raise AssertionError("Timed out waiting to release job start")
                return original_materialize(*args, **kwargs)

            async def race_requests():
                async with _lifespan_client(app) as client:
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
                manager.run_plans,
                "materialize",
                side_effect=paused_materialize,
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
            original_materialize = manager.run_plans.materialize
            start_entered = threading.Event()
            release_start = threading.Event()

            def paused_materialize(*args, **kwargs):
                start_entered.set()
                if not release_start.wait(timeout=THREAD_COORDINATION_TIMEOUT_SECONDS):
                    raise AssertionError("Timed out waiting to release job start")
                return original_materialize(*args, **kwargs)

            async def race_requests():
                async with _lifespan_client(app) as client:
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
                manager.run_plans,
                "materialize",
                side_effect=paused_materialize,
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
                if not release_delete.wait(timeout=THREAD_COORDINATION_TIMEOUT_SECONDS):
                    raise AssertionError("Timed out waiting to release Run delete")
                return original_rmtree(path)

            async def race_requests():
                async with _lifespan_client(app) as client:
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

            with patch.object(
                shutil,
                "rmtree",
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
            original_open = zipfile.ZipFile.open
            import_entered = threading.Event()
            release_import = threading.Event()
            archive = _zip_bytes({"shared_experiment/imported/result.json": "imported"})

            def paused_open(zip_file, member, *args, **kwargs):
                import_entered.set()
                if not release_import.wait(timeout=THREAD_COORDINATION_TIMEOUT_SECONDS):
                    raise AssertionError("Timed out waiting to release log import")
                return original_open(zip_file, member, *args, **kwargs)

            async def race_requests():
                async with _lifespan_client(app) as client:
                    import_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            files={"archive": ("logs.zip", archive, "application/zip")},
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

            with patch.object(zipfile.ZipFile, "open", new=paused_open):
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
            original_materialize = manager.run_plans.materialize
            start_entered = threading.Event()
            release_start = threading.Event()
            archive = _zip_bytes({"shared_experiment/imported/result.json": "imported"})

            def paused_materialize(*args, **kwargs):
                start_entered.set()
                if not release_start.wait(timeout=THREAD_COORDINATION_TIMEOUT_SECONDS):
                    raise AssertionError("Timed out waiting to release job start")
                return original_materialize(*args, **kwargs)

            async def race_requests():
                async with _lifespan_client(app) as client:
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
                            files={"archive": ("logs.zip", archive, "application/zip")},
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
                manager.run_plans,
                "materialize",
                side_effect=paused_materialize,
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
            archive = _zip_bytes({"shared_experiment/imported/result.json": "imported"})

            def paused_delete_experiment(path: Path, *args, **kwargs):
                if Path(path) == logs_root / "shared_experiment":
                    delete_entered.set()
                    if not release_delete.wait(
                        timeout=THREAD_COORDINATION_TIMEOUT_SECONDS
                    ):
                        raise AssertionError("Timed out waiting to release log delete")
                return original_rmtree(path, *args, **kwargs)

            async def race_requests():
                async with _lifespan_client(app) as client:
                    delete_task = asyncio.create_task(
                        client.delete("/logs/experiments/shared_experiment")
                    )
                    await _wait_for_thread_event(delete_entered, "log delete")
                    import_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            files={"archive": ("logs.zip", archive, "application/zip")},
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

            with patch.object(
                shutil,
                "rmtree",
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
            original_open = zipfile.ZipFile.open
            import_entered = threading.Event()
            release_import = threading.Event()
            archive = _zip_bytes({"shared_experiment/imported/result.json": "imported"})

            def paused_open(zip_file, member, *args, **kwargs):
                import_entered.set()
                if not release_import.wait(timeout=THREAD_COORDINATION_TIMEOUT_SECONDS):
                    raise AssertionError("Timed out waiting to release log import")
                return original_open(zip_file, member, *args, **kwargs)

            async def race_requests():
                async with _lifespan_client(app) as client:
                    import_task = asyncio.create_task(
                        client.post(
                            "/logs/import",
                            files={"archive": ("logs.zip", archive, "application/zip")},
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

            with patch.object(zipfile.ZipFile, "open", new=paused_open):
                import_response, delete_response, delete_completed = asyncio.run(
                    race_requests()
                )
            experiment_exists = logs_root.joinpath("shared_experiment").exists()

        self.assertFalse(delete_completed)
        self.assertEqual(import_response.status_code, 200, import_response.text)
        self.assertEqual(delete_response.status_code, 200, delete_response.text)
        self.assertFalse(experiment_exists)


if __name__ == "__main__":
    unittest.main()
