from __future__ import annotations

import asyncio
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock

from emperor_workbench.api import create_app
from emperor_workbench.settings import WorkbenchApiSettings
from emperor_workbench.training_jobs import ActiveTrainingJob, TrainingJobService
from tests.support import lifespan_client
from tests.support.training_jobs import (
    FakeRunner,
    TrainingJobServiceHarness,
    create_app_with_training_service,
    write_tensorboard_run,
)


class RunHistoryHttpTests(unittest.TestCase):
    def test_log_api_deletes_experiment_and_refreshes_runs(self) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )

            async def call_api() -> tuple[
                httpx.Response, httpx.Response, httpx.Response
            ]:
                app = create_app(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    )
                )
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    before_response = await client.get("/logs/runs")
                    delete_response = await client.delete(
                        "/logs/experiments/test_model",
                    )
                    after_response = await client.get("/logs/runs")
                    return before_response, delete_response, after_response

            before_response, delete_response, after_response = asyncio.run(call_api())

            self.assertEqual(before_response.status_code, 200)
            self.assertEqual(delete_response.status_code, 200)
            self.assertFalse(logs_root.joinpath("test_model").exists())
            self.assertTrue(logs_root.joinpath("test_model_2").exists())

        delete_payload = delete_response.json()
        self.assertEqual(delete_payload["experiment"], "test_model")
        self.assertEqual(delete_payload["deletedRunCount"], 1)
        self.assertEqual(delete_payload["deletedRelativePath"], "test_model")
        self.assertEqual(len(delete_payload["deletedRunIds"]), 1)
        self.assertEqual(
            [run["experiment"] for run in after_response.json()["runs"]],
            ["test_model_2"],
        )

    def test_log_api_deletes_valid_empty_experiment_folder(self) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            logs_root.mkdir()
            logs_root.joinpath("new_empty").mkdir()

            async def call_api() -> httpx.Response:
                app = create_app(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    )
                )
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.delete("/logs/experiments/new_empty")

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {
                    "experiment": "new_empty",
                    "deletedRunIds": [],
                    "deletedRunCount": 0,
                    "deletedRelativePath": "new_empty",
                },
            )
            self.assertFalse(logs_root.joinpath("new_empty").exists())

    def test_log_api_blocks_experiment_delete_with_matching_active_job(self) -> None:
        import httpx

        cases = (
            (
                "matching_running_job",
                "test_model",
                "running",
                True,
            ),
            (
                "matching_queued_job",
                "test_model",
                "queued",
                True,
            ),
            (
                "matching_completed_job",
                "test_model",
                "completed",
                False,
            ),
            (
                "matching_failed_job",
                "test_model",
                "failed",
                False,
            ),
            (
                "matching_cancelled_job",
                "test_model",
                "cancelled",
                False,
            ),
            (
                "non_matching_running_job",
                "other_model",
                "running",
                False,
            ),
        )

        for label, job_log_folder, job_status, should_block in cases:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as tmp:
                    logs_root = Path(tmp) / "logs"
                    run_dir = write_tensorboard_run(
                        logs_root,
                        [
                            "test_model",
                            "linears",
                            "linear",
                            "BASELINE",
                            "Mnist",
                            "aaa_20260601_010203",
                            "version_0",
                        ],
                    )
                    settings = WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    )
                    if job_status == "queued":
                        job_id = "queued-job"
                        queued_jobs = Mock(spec=TrainingJobService)
                        queued_jobs.active_jobs.return_value = [
                            ActiveTrainingJob(
                                id=job_id,
                                status="queued",
                                log_folder=job_log_folder,
                            )
                        ]
                        app = create_app(settings, training_jobs=queued_jobs)
                        manager = None
                    else:
                        manager = TrainingJobServiceHarness(
                            root=Path(tmp) / "jobs",
                            logs_root=logs_root,
                            runner=FakeRunner(),
                        )
                        job = manager.create_job_payload(
                            model="linears/linear",
                            preset="baseline",
                            datasets=["Mnist"],
                            overrides={},
                            log_folder=job_log_folder,
                            monitors=[],
                        )
                        job_id = str(job["id"])
                        if job_status == "cancelled":
                            manager.cancel_job_payload(job_id)
                        elif job_status == "completed":
                            manager.runner.process.exit_code = 0
                        elif job_status == "failed":
                            manager.runner.process.exit_code = 1
                        app = create_app_with_training_service(settings, manager)

                    async def call_api(app=app) -> httpx.Response:
                        async with lifespan_client(
                            app,
                            headers={
                                "X-Workbench-Mutation": "true",
                                "Idempotency-Key": uuid.uuid4().hex,
                            },
                        ) as client:
                            return await client.delete("/logs/experiments/test_model")

                    response = asyncio.run(call_api())

                    if should_block:
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(
                            response.json(),
                            {
                                "detail": (
                                    "A training job is still writing to this "
                                    "log folder."
                                )
                            },
                        )
                        self.assertTrue(logs_root.joinpath("test_model").exists())
                        self.assertTrue(run_dir.exists())
                    else:
                        self.assertEqual(response.status_code, 200)
                        payload = response.json()
                        self.assertFalse(logs_root.joinpath("test_model").exists())
                        self.assertFalse(run_dir.exists())
                        if job_log_folder != "test_model":
                            self.assertTrue(logs_root.joinpath(job_log_folder).exists())
                        self.assertEqual(
                            payload,
                            {
                                "experiment": "test_model",
                                "deletedRunIds": payload["deletedRunIds"],
                                "deletedRunCount": 1,
                                "deletedRelativePath": "test_model",
                            },
                        )
                        self.assertEqual(len(payload["deletedRunIds"]), 1)

                    if manager is None:
                        queued_jobs.active_jobs.assert_called()
                    else:
                        self.assertEqual(
                            manager.get_job_payload(job_id)["status"],
                            job_status,
                        )

    def test_log_api_restart_behavior_fresh_manager_preserves_active_delete_blocker(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            original_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )

            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            self.assertEqual(
                fresh_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "unknown",
                        "logFolder": "test_model",
                    }
                ],
            )

            async def call_api() -> httpx.Response:
                app = create_app_with_training_service(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    ),
                    fresh_manager,
                )
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.delete("/logs/experiments/test_model")

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {"detail": ("A training job is still writing to this log folder.")},
            )
            self.assertTrue(logs_root.joinpath("test_model").exists())
            self.assertTrue(run_dir.exists())
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )

    def test_restart_fresh_manager_preserves_unknown_run_delete_blocker(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            original_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            async def list_runs() -> httpx.Response:
                app = create_app_with_training_service(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    ),
                    fresh_manager,
                )
                async with lifespan_client(app) as client:
                    return await client.get("/logs/runs")

            list_response = asyncio.run(list_runs())
            self.assertEqual(list_response.status_code, 200, list_response.text)
            run = list_response.json()["runs"][0]
            filters = {
                "experiments": [run["experiment"]],
                "datasets": [run["dataset"]],
                "models": [
                    {
                        "modelType": run["modelType"],
                        "model": run["model"],
                    }
                ],
                "presets": [run["preset"]],
                "runIds": [run["id"]],
            }

            async def create_plan() -> httpx.Response:
                app = create_app_with_training_service(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    ),
                    fresh_manager,
                )
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.post(
                        "/logs/runs/delete-plan",
                        json=filters,
                    )

            plan_response = asyncio.run(create_plan())

            self.assertEqual(plan_response.status_code, 200, plan_response.text)
            plan_payload = plan_response.json()
            self.assertFalse(plan_payload["canDelete"])
            self.assertEqual(
                plan_payload["blockedByActiveJobs"],
                [
                    {
                        "id": job_id,
                        "logFolder": "test_model",
                        "status": "unknown",
                    }
                ],
            )

            async def delete_runs() -> httpx.Response:
                app = create_app_with_training_service(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    ),
                    fresh_manager,
                )
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.post(
                        "/logs/runs/delete",
                        json=filters,
                    )

            delete_response = asyncio.run(delete_runs())

            self.assertEqual(delete_response.status_code, 400)
            self.assertIn(
                "A training job is still writing to this log folder.",
                delete_response.text,
            )
            self.assertTrue(run_dir.exists())

    def test_restart_fresh_manager_blocks_experiment_delete_for_unknown_job(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_root = root / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            original_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            job = original_manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )
            job_id = str(job["id"])
            self.assertEqual(
                original_manager.active_job_payloads(),
                [
                    {
                        "id": job_id,
                        "status": "running",
                        "logFolder": "test_model",
                    }
                ],
            )
            fresh_manager = TrainingJobServiceHarness(
                root=root / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )

            async def call_api() -> httpx.Response:
                app = create_app_with_training_service(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    ),
                    fresh_manager,
                )
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.delete("/logs/experiments/test_model")

            response = asyncio.run(call_api())

            self.assertEqual(response.status_code, 400)
            self.assertEqual(
                response.json(),
                {"detail": ("A training job is still writing to this log folder.")},
            )
            self.assertTrue(logs_root.joinpath("test_model").exists())
            self.assertTrue(run_dir.exists())

    def test_log_api_plans_and_deletes_filtered_runs_with_active_job_guard(
        self,
    ) -> None:
        import httpx

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            run_dir = write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=logs_root,
                runner=FakeRunner(),
            )
            manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="test_model",
                monitors=[],
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                app = create_app_with_training_service(
                    WorkbenchApiSettings(
                        logs_root=str(logs_root),
                        allow_unsafe_local_mutations=True,
                    ),
                    manager,
                )
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run = runs_response.json()["runs"][0]
                    filters = {
                        "experiments": [run["experiment"]],
                        "datasets": [run["dataset"]],
                        "models": [
                            {
                                "modelType": run["modelType"],
                                "model": run["model"],
                            }
                        ],
                        "presets": [run["preset"]],
                        "runIds": [run["id"]],
                    }
                    plan_response = await client.post(
                        "/logs/runs/delete-plan",
                        json=filters,
                    )
                    delete_response = await client.post(
                        "/logs/runs/delete",
                        json=filters,
                    )
                    return plan_response, delete_response

            plan_response, delete_response = asyncio.run(call_api())

            self.assertEqual(plan_response.status_code, 200)
            plan_payload = plan_response.json()
            self.assertEqual(plan_payload["candidateCount"], 1)
            self.assertFalse(plan_payload["canDelete"])
            self.assertEqual(
                plan_payload["blockedByActiveJobs"][0]["logFolder"],
                "test_model",
            )
            self.assertEqual(delete_response.status_code, 400)
            self.assertIn(
                "A training job is still writing to this log folder.",
                delete_response.text,
            )
            self.assertTrue(run_dir.exists())

    def test_log_api_lists_experiments_including_empty_folders(self) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            logs_root.mkdir()
            logs_root.joinpath("new_empty").mkdir()
            logs_root.joinpath("bad-name").mkdir()
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )

            async def call_api() -> httpx.Response:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    return await client.get("/logs/experiments")

            response = asyncio.run(call_api())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["experiments"],
            [
                {
                    "experiment": "new_empty",
                    "runCount": 0,
                    "relativePath": "new_empty",
                },
                {
                    "experiment": "test_model",
                    "runCount": 1,
                    "relativePath": "test_model",
                },
            ],
        )

    def test_log_api_paginates_unbounded_list_endpoints(self) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            for index, experiment in enumerate(("exp_a", "exp_b", "exp_c"), start=1):
                write_tensorboard_run(
                    logs_root,
                    [
                        experiment,
                        "linears",
                        "linear",
                        "BASELINE",
                        "Mnist",
                        f"run_{index}_2026060{index}_010203",
                        "version_0",
                    ],
                )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    runs_response = await client.get(
                        "/logs/runs",
                        params={"limit": 2, "offset": 1},
                    )
                    experiments_response = await client.get(
                        "/logs/experiments",
                        params={"limit": 1, "offset": 1},
                    )
                    return runs_response, experiments_response

            runs_response, experiments_response = asyncio.run(call_api())

        self.assertEqual(runs_response.status_code, 200)
        runs_payload = runs_response.json()
        self.assertEqual(runs_payload["total"], 3)
        self.assertEqual(runs_payload["limit"], 2)
        self.assertEqual(runs_payload["offset"], 1)
        self.assertFalse(runs_payload["hasMore"])
        self.assertEqual(len(runs_payload["runs"]), 2)
        self.assertEqual(
            [run["experiment"] for run in runs_payload["runs"]],
            ["exp_b", "exp_a"],
        )

        self.assertEqual(experiments_response.status_code, 200)
        experiments_payload = experiments_response.json()
        self.assertEqual(experiments_payload["total"], 3)
        self.assertEqual(experiments_payload["limit"], 1)
        self.assertEqual(experiments_payload["offset"], 1)
        self.assertTrue(experiments_payload["hasMore"])
        self.assertEqual(len(experiments_payload["experiments"]), 1)
        self.assertEqual(
            [
                experiment["experiment"]
                for experiment in experiments_payload["experiments"]
            ],
            ["exp_b"],
        )

    def test_log_api_reads_tags_scalars_and_rejects_unknown_run_ids(self) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
                scalars={
                    "train/loss": [(1, 0.5), (2, 0.25)],
                    "validation/accuracy": [(2, 0.75)],
                },
                metrics={"test/accuracy": 0.9},
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model_2",
                    "linears",
                    "linear_adaptive",
                    "DUAL_MODEL_WEIGHT",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
                metrics=None,
                hparams=False,
                checkpoint=False,
            )
            no_event_run = logs_root.joinpath(
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "no_events_20260601_040506",
                "version_0",
            )
            no_event_run.mkdir(parents=True)
            malformed_result_run = write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "malformed_20260601_050607",
                    "version_0",
                ],
                metrics=None,
            )
            (malformed_result_run / "result.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )

            async def call_api() -> tuple[
                httpx.Response,
                httpx.Response,
                httpx.Response,
                httpx.Response,
                httpx.Response,
            ]:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run_id = next(
                        run["id"]
                        for run in runs_response.json()["runs"]
                        if run["relativePath"]
                        == "linears/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
                    )
                    tags_response = await client.post(
                        "/logs/tags",
                        json={"runIds": [run_id]},
                    )
                    scalars_response = await client.post(
                        "/logs/scalars",
                        json={"runIds": [run_id], "tags": ["train/loss"]},
                    )
                    unknown_response = await client.post(
                        "/logs/tags",
                        json={"runIds": ["not-a-run"]},
                    )
                    raw_path_response = await client.post(
                        "/logs/scalars",
                        json={
                            "runIds": [
                                "linears/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
                            ],
                            "tags": ["train/loss"],
                        },
                    )
                    return (
                        runs_response,
                        tags_response,
                        scalars_response,
                        unknown_response,
                        raw_path_response,
                    )

            (
                runs_response,
                tags_response,
                scalars_response,
                unknown_response,
                raw_path_response,
            ) = asyncio.run(call_api())

        self.assertEqual(runs_response.status_code, 200)
        runs_payload = runs_response.json()["runs"]
        by_path = {run["relativePath"]: run for run in runs_payload}
        run_payload = by_path[
            "linears/linear/BASELINE/Mnist/aaa_20260601_010203/version_0"
        ]
        incomplete_payload = by_path[
            "test_model_2/linears/linear_adaptive/DUAL_MODEL_WEIGHT/Cifar10/"
            "bbb_20260601_020304/version_0"
        ]
        no_event_payload = by_path[
            "linears/linear/BASELINE/Mnist/no_events_20260601_040506/version_0"
        ]
        malformed_result_payload = by_path[
            "linears/linear/BASELINE/Mnist/malformed_20260601_050607/version_0"
        ]
        self.assertEqual(run_payload["experiment"], "linears")
        self.assertEqual(run_payload["dataset"], "Mnist")
        self.assertTrue(run_payload["hasResult"])
        self.assertGreater(run_payload["eventFileCount"], 0)
        self.assertEqual(run_payload["metrics"]["test/accuracy"], 0.9)
        self.assertEqual(incomplete_payload["experiment"], "test_model_2")
        self.assertFalse(incomplete_payload["hasResult"])
        self.assertFalse(no_event_payload["hasResult"])
        self.assertEqual(no_event_payload["eventFileCount"], 0)
        self.assertEqual(no_event_payload["metrics"], {})
        self.assertTrue(malformed_result_payload["hasResult"])
        self.assertEqual(malformed_result_payload["metrics"], {})

        self.assertEqual(tags_response.status_code, 200)
        self.assertEqual(
            set(tags_response.json()["runs"][0]["scalarTags"]),
            {"train/loss", "validation/accuracy"},
        )

        self.assertEqual(scalars_response.status_code, 200)
        series = scalars_response.json()["series"][0]
        self.assertEqual(series["tag"], "train/loss")
        self.assertEqual([point["step"] for point in series["points"]], [1, 2])
        self.assertEqual(series["points"][1]["value"], 0.25)

        self.assertEqual(unknown_response.status_code, 400)
        self.assertIn("Unknown log run id", unknown_response.json()["detail"])
        self.assertEqual(raw_path_response.status_code, 400)
        self.assertIn("Unknown log run id", raw_path_response.json()["detail"])

    def test_log_api_filters_runs_before_pagination(self) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "BASELINE",
                    "Cifar10",
                    "bbb_20260601_020304",
                    "version_0",
                ],
            )
            write_tensorboard_run(
                logs_root,
                [
                    "test_model",
                    "linears",
                    "linear",
                    "GATING",
                    "Mnist",
                    "ccc_20260601_030405",
                    "version_0",
                ],
            )
            no_event_run = logs_root.joinpath(
                "test_model",
                "linears",
                "linear",
                "BASELINE",
                "Mnist",
                "no_events_20260601_040506",
                "version_0",
            )
            no_event_run.mkdir(parents=True)

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    scoped_response = await client.get(
                        "/logs/runs",
                        params=[
                            ("model", "linears/linear"),
                            ("preset", "BASELINE"),
                            ("dataset", "Mnist"),
                            ("hasEventFiles", "true"),
                            ("limit", "5"),
                        ],
                    )
                    no_event_response = await client.get(
                        "/logs/runs",
                        params={
                            "model": "linears/linear",
                            "preset": "BASELINE",
                            "dataset": "Mnist",
                            "hasEventFiles": "false",
                        },
                    )
                    return scoped_response, no_event_response

            scoped_response, no_event_response = asyncio.run(call_api())

        self.assertEqual(scoped_response.status_code, 200)
        scoped_payload = scoped_response.json()
        self.assertEqual(scoped_payload["total"], 1)
        self.assertEqual(len(scoped_payload["runs"]), 1)
        self.assertEqual(scoped_payload["runs"][0]["dataset"], "Mnist")
        self.assertGreater(scoped_payload["runs"][0]["eventFileCount"], 0)

        self.assertEqual(no_event_response.status_code, 200)
        no_event_payload = no_event_response.json()
        self.assertEqual(no_event_payload["total"], 1)
        self.assertEqual(no_event_payload["runs"][0]["eventFileCount"], 0)

    def test_log_api_reports_layer_monitor_eligibility_from_tag_cache(
        self,
    ) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "layer_20260601_010203",
                    "version_0",
                ],
                scalars={"main_model.layers.0.model/weights/mean": [(1, 0.5)]},
            )
            write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "perf_20260601_020304",
                    "version_0",
                ],
                scalars={"train/loss": [(1, 0.5)]},
            )

            async def call_api() -> tuple[
                httpx.Response, httpx.Response, httpx.Response
            ]:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    before_response = await client.get("/logs/runs")
                    run_ids_by_name = {
                        run["runName"]: run["id"]
                        for run in before_response.json()["runs"]
                    }
                    tags_response = await client.post(
                        "/logs/tags",
                        json={
                            "runIds": [
                                run_ids_by_name["layer_20260601_010203"],
                                run_ids_by_name["perf_20260601_020304"],
                            ]
                        },
                    )
                    after_response = await client.get("/logs/runs")
                    return before_response, tags_response, after_response

            before_response, tags_response, after_response = asyncio.run(call_api())

        self.assertEqual(before_response.status_code, 200)
        self.assertEqual(tags_response.status_code, 200)
        self.assertEqual(after_response.status_code, 200)

        before_by_name = {run["runName"]: run for run in before_response.json()["runs"]}
        self.assertIsNone(
            before_by_name["layer_20260601_010203"]["hasLayerMonitorData"]
        )
        self.assertIsNone(before_by_name["perf_20260601_020304"]["hasLayerMonitorData"])

        tags_by_run_id = {run["runId"]: run for run in tags_response.json()["runs"]}
        layer_run_id = before_by_name["layer_20260601_010203"]["id"]
        perf_run_id = before_by_name["perf_20260601_020304"]["id"]
        self.assertTrue(tags_by_run_id[layer_run_id]["hasLayerMonitorData"])
        self.assertFalse(tags_by_run_id[perf_run_id]["hasLayerMonitorData"])

        after_by_name = {run["runName"]: run for run in after_response.json()["runs"]}
        self.assertTrue(after_by_name["layer_20260601_010203"]["hasLayerMonitorData"])
        self.assertFalse(after_by_name["perf_20260601_020304"]["hasLayerMonitorData"])

    def test_log_api_scalar_request_limits_and_metadata(self) -> None:
        import httpx

        from emperor_workbench.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            logs_root = Path(tmp) / "logs"
            write_tensorboard_run(
                logs_root,
                [
                    "linears",
                    "linear",
                    "BASELINE",
                    "Mnist",
                    "aaa_20260601_010203",
                    "version_0",
                ],
                scalars={"train/loss": [(1, 0.5), (2, 0.25), (3, 0.125)]},
            )

            async def call_api() -> tuple[httpx.Response, httpx.Response]:
                app = create_app(WorkbenchApiSettings(logs_root=str(logs_root)))
                async with lifespan_client(
                    app,
                    headers={
                        "X-Workbench-Mutation": "true",
                        "Idempotency-Key": uuid.uuid4().hex,
                    },
                ) as client:
                    runs_response = await client.get("/logs/runs")
                    run_id = runs_response.json()["runs"][0]["id"]
                    scalars_response = await client.post(
                        "/logs/scalars",
                        json={
                            "runIds": [run_id],
                            "tags": ["train/loss"],
                            "maxPoints": 2,
                            "sampling": "tail",
                        },
                    )
                    invalid_response = await client.post(
                        "/logs/scalars",
                        json={
                            "runIds": [run_id],
                            "tags": ["train/loss"],
                            "maxPoints": 2001,
                        },
                    )
                    return scalars_response, invalid_response

            scalars_response, invalid_response = asyncio.run(call_api())

        self.assertEqual(scalars_response.status_code, 200)
        series = scalars_response.json()["series"][0]
        self.assertEqual([point["step"] for point in series["points"]], [2, 3])
        self.assertEqual(series["sourcePointCount"], 3)
        self.assertTrue(series["truncated"])
        self.assertEqual(invalid_response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
