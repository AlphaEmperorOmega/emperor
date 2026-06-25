from __future__ import annotations

import os
import unittest
from collections.abc import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from viewer.backend.repositories.log_runs import LogRunRepository
from viewer.backend.repositories.training_jobs import TrainingJobRepository
from viewer.backend.training_contracts import (
    ActiveTrainingJob,
    CreateTrainingJobCommand,
    CreateTrainingRunPlanCommand,
    TrainingJobView,
    TrainingRunPlanView,
    TrainingSearch,
)


class RecordingDelegate:
    def __init__(self, return_values: dict[str, object]) -> None:
        self.return_values = return_values
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> object:
        self.calls.append((name, args, kwargs))
        return self.return_values[name]


class RecordingLogRunIndex(RecordingDelegate):
    def list_runs(self) -> object:
        return self._record("list_runs")

    def list_experiments(self) -> object:
        return self._record("list_experiments")

    def delete_experiment(self, experiment: object) -> object:
        return self._record(
            "delete_experiment",
            experiment,
        )

    def create_delete_plan(self, filters: object, *, active_jobs: object) -> object:
        return self._record(
            "create_delete_plan",
            filters,
            active_jobs=active_jobs,
        )

    def delete_runs(self, filters: object, *, active_jobs: object) -> object:
        return self._record("delete_runs", filters, active_jobs=active_jobs)

    def tags_for_runs(self, run_ids: object) -> object:
        return self._record("tags_for_runs", run_ids)

    def scalars_for_runs(
        self,
        *,
        run_ids: object,
        tags: object,
        max_points: object,
        sampling: object,
    ) -> object:
        return self._record(
            "scalars_for_runs",
            run_ids=run_ids,
            tags=tags,
            max_points=max_points,
            sampling=sampling,
        )

    def monitor_data_for_run(self, run_id: object, *, node_path: object) -> object:
        return self._record("monitor_data_for_run", run_id, node_path=node_path)

    def parameter_status_for_runs(self, run_ids: object) -> object:
        return self._record("parameter_status_for_runs", run_ids)


class RecordingTrainingJobManager(RecordingDelegate):
    def create_job(
        self,
        *,
        model: object,
        preset: object,
        presets: object,
        datasets: object,
        overrides: object,
        log_folder: object,
        monitors: object,
        search: object,
        run_plan: object,
    ) -> object:
        return self._record(
            "create_job",
            model=model,
            preset=preset,
            presets=presets,
            datasets=datasets,
            overrides=overrides,
            log_folder=log_folder,
            monitors=monitors,
            search=search,
            run_plan=run_plan,
        )

    def create_run_plan(
        self,
        *,
        model: object,
        preset: object,
        presets: object,
        datasets: object,
        overrides: object,
        log_folder: object,
        monitors: object,
        search: object,
    ) -> object:
        return self._record(
            "create_run_plan",
            model=model,
            preset=preset,
            presets=presets,
            datasets=datasets,
            overrides=overrides,
            log_folder=log_folder,
            monitors=monitors,
            search=search,
        )

    def get_job(self, job_id: object) -> object:
        return self._record("get_job", job_id)

    def get_job_events(
        self,
        job_id: object,
        *,
        offset: object,
        limit: object,
    ) -> object:
        return self._record(
            "get_job_events",
            job_id,
            offset=offset,
            limit=limit,
        )

    def get_monitor_data(
        self,
        job_id: object,
        *,
        node_path: object,
        dataset: object,
        preset: object,
    ) -> object:
        return self._record(
            "get_monitor_data",
            job_id,
            node_path=node_path,
            dataset=dataset,
            preset=preset,
        )

    def get_parameter_status(
        self,
        job_id: object,
        *,
        dataset: object,
        preset: object,
    ) -> object:
        return self._record(
            "get_parameter_status",
            job_id,
            dataset=dataset,
            preset=preset,
        )

    def cancel_job(self, job_id: object) -> object:
        return self._record("cancel_job", job_id)

    def active_jobs(self) -> object:
        return self._record("active_jobs")


class LogRunRepositoryTests(unittest.TestCase):
    def assert_delegates(
        self,
        method_name: str,
        invoke: Callable[[LogRunRepository], object],
        expected_args: tuple[object, ...] = (),
        expected_kwargs: dict[str, object] | None = None,
    ) -> None:
        return_value = object()
        index = RecordingLogRunIndex({method_name: return_value})
        repository = LogRunRepository(index)  # type: ignore[arg-type]

        result = invoke(repository)

        self.assertIs(result, return_value)
        self.assertEqual(
            index.calls,
            [(method_name, expected_args, expected_kwargs or {})],
        )

    def test_public_methods_delegate_to_injected_log_run_index(self) -> None:
        experiment = object()
        filters = object()
        active_jobs = object()
        run_ids = object()
        tags = object()
        run_id = object()
        node_path = object()

        cases: tuple[
            tuple[
                str,
                Callable[[LogRunRepository], object],
                tuple[object, ...],
                dict[str, object],
            ],
            ...,
        ] = (
            ("list_runs", lambda repository: repository.list_runs(), (), {}),
            (
                "list_experiments",
                lambda repository: repository.list_experiments(),
                (),
                {},
            ),
            (
                "delete_experiment",
                lambda repository: repository.delete_experiment(  # type: ignore[arg-type]
                    experiment,
                ),
                (experiment,),
                {},
            ),
            (
                "create_delete_plan",
                lambda repository: repository.create_delete_plan(  # type: ignore[arg-type]
                    filters,
                    active_jobs=active_jobs,
                ),
                (filters,),
                {"active_jobs": active_jobs},
            ),
            (
                "delete_runs",
                lambda repository: repository.delete_runs(  # type: ignore[arg-type]
                    filters,
                    active_jobs=active_jobs,
                ),
                (filters,),
                {"active_jobs": active_jobs},
            ),
            (
                "tags_for_runs",
                lambda repository: repository.tags_for_runs(run_ids),  # type: ignore[arg-type]
                (run_ids,),
                {},
            ),
            (
                "scalars_for_runs",
                lambda repository: repository.scalars_for_runs(  # type: ignore[arg-type]
                    run_ids=run_ids,
                    tags=tags,
                    max_points=500,
                    sampling="tail",
                ),
                (),
                {
                    "run_ids": run_ids,
                    "tags": tags,
                    "max_points": 500,
                    "sampling": "tail",
                },
            ),
            (
                "monitor_data_for_run",
                lambda repository: repository.monitor_data_for_run(  # type: ignore[arg-type]
                    run_id,
                    node_path,
                ),
                (run_id,),
                {"node_path": node_path},
            ),
            (
                "parameter_status_for_runs",
                lambda repository: repository.parameter_status_for_runs(  # type: ignore[arg-type]
                    run_ids,
                ),
                (run_ids,),
                {},
            ),
        )

        for method_name, invoke, expected_args, expected_kwargs in cases:
            with self.subTest(method=method_name):
                self.assert_delegates(
                    method_name,
                    invoke,
                    expected_args,
                    expected_kwargs,
                )


class TrainingJobRepositoryTests(unittest.TestCase):
    def assert_training_delegates(
        self,
        method_name: str,
        invoke: Callable[[TrainingJobRepository], object],
        return_value: object,
        expected_args: tuple[object, ...] = (),
        expected_kwargs: dict[str, object] | None = None,
        expected_result_type: type | None = None,
    ) -> None:
        manager = RecordingTrainingJobManager({method_name: return_value})
        repository = TrainingJobRepository(manager)  # type: ignore[arg-type]

        result = invoke(repository)

        self.assertEqual(
            manager.calls,
            [(method_name, expected_args, expected_kwargs or {})],
        )
        if expected_result_type is None:
            self.assertIs(result, return_value)
        else:
            self.assertIsInstance(result, expected_result_type)

    def test_public_methods_delegate_to_injected_training_job_manager(self) -> None:
        search = TrainingSearch(mode="grid", values={"stack_hidden_dim": [128]})
        run_plan_payload = {
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"stack_hidden_dim": "128"},
            "search": search.to_api_payload(),
            "logFolder": "repository_test",
            "isRandomSearch": False,
            "runs": [
                {
                    "id": "run-1",
                    "index": 1,
                    "status": "Pending",
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "changes": [],
                    "overrides": {},
                    "command": "train",
                    "totalEpochs": 1,
                    "currentEpoch": 0,
                    "metrics": {},
                    "logDir": None,
                    "error": None,
                    "errorTraceback": None,
                }
            ],
            "summary": {
                "totalRuns": 1,
                "pendingRuns": 1,
                "totalEpochs": 1,
                "remainingEpochs": 1,
            },
        }
        run_plan = TrainingRunPlanView.from_payload(run_plan_payload)
        job_payload = {
            "id": "job-1",
            "status": "running",
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"stack_hidden_dim": "128"},
            "search": search.to_api_payload(),
            "plannedRunCount": 1,
            "runPlan": run_plan_payload,
            "monitors": ["linear"],
            "logFolder": "repository_test",
            "createdAt": "2026-06-09T00:00:00+00:00",
            "updatedAt": "2026-06-09T00:00:00+00:00",
            "exitCode": None,
            "pid": 123,
            "currentPreset": None,
            "currentDataset": None,
            "epoch": None,
            "step": None,
            "metrics": {},
            "logDir": None,
            "events": [],
            "eventCount": 0,
            "eventCounts": {},
            "eventsTruncated": False,
            "clusterGrowth": [],
            "logTail": [],
            "resultLinks": [],
        }
        active_jobs_payload = [
            {"id": "job-1", "status": "running", "logFolder": "repository_test"}
        ]
        create_job_command = CreateTrainingJobCommand(
            model="linears/linear",
            preset="baseline",
            presets=["baseline"],
            datasets=["Mnist"],
            overrides={"stack_hidden_dim": "128"},
            log_folder="repository_test",
            search=search,
            monitors=["linear"],
            run_plan=run_plan,
        )
        create_run_plan_command = CreateTrainingRunPlanCommand(
            model="linears/linear",
            preset="baseline",
            presets=["baseline"],
            datasets=["Mnist"],
            overrides={"stack_hidden_dim": "128"},
            log_folder="repository_test",
            monitors=["linear"],
            search=search,
        )
        create_job_kwargs = {
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"stack_hidden_dim": "128"},
            "log_folder": "repository_test",
            "monitors": ["linear"],
            "search": search.to_api_payload(),
            "run_plan": run_plan.to_api_payload(),
        }
        create_run_plan_kwargs = {
            "model": "linears/linear",
            "preset": "baseline",
            "presets": ["baseline"],
            "datasets": ["Mnist"],
            "overrides": {"stack_hidden_dim": "128"},
            "log_folder": "repository_test",
            "monitors": ["linear"],
            "search": search.to_api_payload(),
        }
        job_id = object()
        monitor_job_id = object()
        event_job_id = object()
        monitor_kwargs = {
            "node_path": object(),
            "dataset": object(),
            "preset": object(),
        }
        event_kwargs = {"offset": 10, "limit": 25}
        cancel_job_id = object()

        cases: tuple[
            tuple[
                str,
                Callable[[TrainingJobRepository], object],
                tuple[object, ...],
                dict[str, object],
                object,
                type | None,
            ],
            ...,
        ] = (
            (
                "create_job",
                lambda repository: repository.create_job(create_job_command),
                (),
                create_job_kwargs,
                job_payload,
                TrainingJobView,
            ),
            (
                "create_run_plan",
                lambda repository: repository.create_run_plan(
                    create_run_plan_command
                ),
                (),
                create_run_plan_kwargs,
                run_plan_payload,
                TrainingRunPlanView,
            ),
            (
                "get_job",
                lambda repository: repository.get_job(job_id),  # type: ignore[arg-type]
                (job_id,),
                {},
                job_payload,
                TrainingJobView,
            ),
            (
                "get_monitor_data",
                lambda repository: repository.get_monitor_data(  # type: ignore[arg-type]
                    monitor_job_id,
                    **monitor_kwargs,
                ),
                (monitor_job_id,),
                monitor_kwargs,
                object(),
                None,
            ),
            (
                "get_job_events",
                lambda repository: repository.get_job_events(  # type: ignore[arg-type]
                    event_job_id,
                    offset=event_kwargs["offset"],
                    limit=event_kwargs["limit"],
                ),
                (event_job_id,),
                event_kwargs,
                {"events": []},
                None,
            ),
            (
                "get_parameter_status",
                lambda repository: repository.get_parameter_status(  # type: ignore[arg-type]
                    monitor_job_id,
                    dataset=monitor_kwargs["dataset"],
                    preset=monitor_kwargs["preset"],
                ),
                (monitor_job_id,),
                {
                    "dataset": monitor_kwargs["dataset"],
                    "preset": monitor_kwargs["preset"],
                },
                object(),
                None,
            ),
            (
                "cancel_job",
                lambda repository: repository.cancel_job(cancel_job_id),  # type: ignore[arg-type]
                (cancel_job_id,),
                {},
                job_payload,
                TrainingJobView,
            ),
            (
                "active_jobs",
                lambda repository: repository.active_jobs(),
                (),
                {},
                active_jobs_payload,
                list,
            ),
        )

        for (
            method_name,
            invoke,
            expected_args,
            expected_kwargs,
            return_value,
            expected_result_type,
        ) in cases:
            with self.subTest(method=method_name):
                self.assert_training_delegates(
                    method_name,
                    invoke,
                    return_value,
                    expected_args,
                    expected_kwargs,
                    expected_result_type,
                )

    def test_active_jobs_are_typed_views(self) -> None:
        manager = RecordingTrainingJobManager(
            {
                "active_jobs": [
                    {
                        "id": "job-1",
                        "status": "running",
                        "logFolder": "repository_test",
                    }
                ]
            }
        )
        repository = TrainingJobRepository(manager)  # type: ignore[arg-type]

        result = repository.active_jobs()

        self.assertEqual(
            result,
            [
                ActiveTrainingJob(
                    id="job-1",
                    status="running",
                    log_folder="repository_test",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
