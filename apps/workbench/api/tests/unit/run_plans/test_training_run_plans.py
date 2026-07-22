from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from model_runtime.packages import GridSearch
from model_runtime.runs import RunRequest, plan_runs
from models.catalog import model_package
from models.package_cli import _search_spec

from emperor_workbench.api.v1.model_packages import ConfigSchemaResponse
from emperor_workbench.config_snapshots import (
    ConfigSnapshotService,
)
from emperor_workbench.failures import FailureKind
from emperor_workbench.model_packages import ModelPackageCatalog
from emperor_workbench.run_plans import (
    MAX_TRAINING_PLANNED_RUNS,
    CreateTrainingRunPlanCommand,
    MaterializeTrainingRunPlanCommand,
    RunPlanFailure,
    RunPlanPersistenceCodec,
    RunPlanService,
    RunPlanWorkerAcceptance,
    SubmittedTrainingRun,
    SubmittedTrainingRunPlan,
    TrainingRunPlanView,
)
from emperor_workbench.training_jobs import TrainingJobFailure
from tests.support.model_packages import project_adapter_client
from tests.support.training_jobs import (
    FakeRunner,
    TrainingJobServiceHarness,
    run_plan_payload,
)


def _run_plan_service(*args, **kwargs) -> RunPlanService:
    project_adapter = kwargs.pop("project_adapter", project_adapter_client())
    kwargs.setdefault("model_packages", ModelPackageCatalog(project_adapter))
    return RunPlanService(*args, **kwargs)


def _submitted_plan(plan: TrainingRunPlanView) -> SubmittedTrainingRunPlan:
    return SubmittedTrainingRunPlan(
        runs=[
            SubmittedTrainingRun(
                id=run.id,
                preset=run.preset,
                dataset=run.dataset,
                overrides=dict(run.overrides),
                snapshot_id=run.snapshot_id,
                snapshot_name=run.snapshot_name,
            )
            for run in plan.runs
        ],
        snapshot_revisions=plan.snapshot_revisions,
    )


def _snapshot_service() -> ConfigSnapshotService:
    return ConfigSnapshotService.in_memory(
        model_packages=ModelPackageCatalog(project_adapter_client()),
    )


class TrainingRunPlanTests(unittest.TestCase):
    def test_shared_adapter_owns_preview_submission_persistence_and_worker_decode(
        self,
    ) -> None:
        service = _run_plan_service(random_source=random.Random(13))
        preview_command = CreateTrainingRunPlanCommand(
            model="linears/linear",
            preset="baseline",
            presets=["baseline"],
            datasets=["Mnist"],
            overrides={"num_epochs": 7},
            log_folder="adapter_round_trip",
            monitors=["linear"],
        )
        preview = service.preview(preview_command)
        materialized = service.materialize(
            MaterializeTrainingRunPlanCommand(
                model=preview_command.model,
                preset=preview_command.preset,
                presets=preview_command.presets,
                experiment_task=preview_command.experiment_task,
                datasets=preview_command.datasets,
                overrides=preview_command.overrides,
                log_folder=preview_command.log_folder,
                monitors=preview_command.monitors,
                search=preview_command.search,
                submitted_plan=_submitted_plan(preview),
            ),
            validated_log_folder="adapter_round_trip",
        )
        package = (
            ModelPackageCatalog(project_adapter_client())
            .select("linears/linear")
            .reference
        )

        with patch("model_runtime.runs.planning.plan_runs") as plan_runs:
            semantic_plan = RunPlanWorkerAcceptance.accept(
                package,
                {
                    "id": "job-1",
                    "plannedRunCount": 1,
                    "monitors": ["linear"],
                    "runPlan": RunPlanPersistenceCodec.encode(materialized.plan),
                },
            )

        plan_runs.assert_not_called()
        self.assertEqual(semantic_plan.runs[0].id, preview.runs[0].id)
        self.assertEqual(materialized.plan.runs[0].total_epochs, 7)
        self.assertEqual(materialized.plan.summary.total_epochs, 7)

    def test_snapshot_plan_is_backend_authoritative_and_revision_checked(self) -> None:
        snapshots = _snapshot_service()
        snapshot = snapshots.create_snapshot(
            model="linears/linear",
            preset="baseline",
            name="wide model",
            overrides={"HIDDEN_DIM": "128", "NUM_EPOCHS": "7"},
        )
        service = _run_plan_service(config_snapshots=snapshots)
        preview_command = CreateTrainingRunPlanCommand(
            model="linears/linear",
            preset="baseline",
            presets=[],
            datasets=["Mnist"],
            overrides={"STACK_BIAS_FLAG": "false"},
            log_folder="snapshot_authority",
            snapshot_ids=[snapshot.id],
        )

        preview = service.preview(preview_command)
        payload = run_plan_payload(preview)

        self.assertEqual(len(payload["runs"]), 1)
        self.assertEqual(payload["runs"][0]["snapshotId"], snapshot.id)
        self.assertEqual(payload["runs"][0]["snapshotName"], "wide model")
        self.assertEqual(
            payload["runs"][0]["overrides"],
            {
                "HIDDEN_DIM": 128,
                "NUM_EPOCHS": 7,
                "STACK_BIAS_FLAG": False,
            },
        )
        self.assertEqual(payload["runs"][0]["totalEpochs"], 7)
        self.assertEqual(
            payload["snapshotRevisions"][0]["id"],
            snapshot.id,
        )
        revision = payload["snapshotRevisions"][0]["semanticRevision"]
        self.assertRegex(revision, r"^[0-9a-f]{64}$")

        snapshots.rename_snapshot(snapshot.id, "renamed model")
        materialized = service.materialize(
            MaterializeTrainingRunPlanCommand(
                model=preview_command.model,
                preset=preview_command.preset,
                presets=preview_command.presets,
                datasets=preview_command.datasets,
                overrides=preview_command.overrides,
                log_folder=preview_command.log_folder,
                snapshot_ids=preview_command.snapshot_ids,
                snapshot_revisions=preview.snapshot_revisions,
            ),
            validated_log_folder="snapshot_authority",
        )
        self.assertEqual(
            materialized.plan.runs[0].snapshot_name,
            "renamed model",
        )
        self.assertEqual(
            materialized.plan.snapshot_revisions,
            preview.snapshot_revisions,
        )

        snapshots.update_snapshot(
            snapshot.id,
            overrides={"HIDDEN_DIM": "64", "NUM_EPOCHS": "7"},
        )
        with self.assertRaises(RunPlanFailure) as stale:
            service.materialize(
                MaterializeTrainingRunPlanCommand(
                    model=preview_command.model,
                    preset=preview_command.preset,
                    presets=preview_command.presets,
                    datasets=preview_command.datasets,
                    overrides=preview_command.overrides,
                    log_folder=preview_command.log_folder,
                    snapshot_ids=preview_command.snapshot_ids,
                    snapshot_revisions=preview.snapshot_revisions,
                ),
                validated_log_folder="snapshot_authority",
            )

        self.assertEqual(stale.exception.kind, FailureKind.CONFLICT)

        snapshots.delete_snapshot(snapshot.id)
        with self.assertRaises(RunPlanFailure) as deleted:
            service.materialize(
                MaterializeTrainingRunPlanCommand(
                    model=preview_command.model,
                    preset=preview_command.preset,
                    presets=preview_command.presets,
                    datasets=preview_command.datasets,
                    overrides=preview_command.overrides,
                    log_folder=preview_command.log_folder,
                    snapshot_ids=preview_command.snapshot_ids,
                    snapshot_revisions=preview.snapshot_revisions,
                ),
                validated_log_folder="snapshot_authority",
            )

        self.assertEqual(deleted.exception.kind, FailureKind.CONFLICT)

    def test_snapshot_plan_rejects_missing_and_foreign_snapshot_ids(self) -> None:
        snapshots = _snapshot_service()
        snapshots.create_snapshot(
            model="experts/linear",
            preset="baseline",
            name="foreign",
            overrides={"HIDDEN_DIM": "128"},
            snapshot_id="foreign",
        )
        service = _run_plan_service(config_snapshots=snapshots)

        for snapshot_id, detail in (
            ("missing", "no longer exists"),
            ("foreign", "belongs to Model Package"),
        ):
            with self.subTest(snapshot_id=snapshot_id):
                with self.assertRaisesRegex(RunPlanFailure, detail):
                    service.preview(
                        CreateTrainingRunPlanCommand(
                            model="linears/linear",
                            preset="baseline",
                            presets=[],
                            datasets=["Mnist"],
                            overrides={},
                            log_folder="snapshot_validation",
                            snapshot_ids=[snapshot_id],
                        )
                    )

    def test_snapshot_provenance_without_backend_revision_is_rejected(self) -> None:
        service = _run_plan_service()
        command = MaterializeTrainingRunPlanCommand(
            model="linears/linear",
            preset="baseline",
            presets=["baseline"],
            datasets=["Mnist"],
            overrides={},
            log_folder="invented_snapshot",
            submitted_plan=SubmittedTrainingRunPlan(
                runs=[
                    SubmittedTrainingRun(
                        id="invented-row",
                        preset="baseline",
                        dataset="Mnist",
                        overrides={},
                        snapshot_id="invented",
                        snapshot_name="invented",
                    )
                ]
            ),
        )

        with self.assertRaises(RunPlanFailure) as invented:
            service.materialize(
                command,
                validated_log_folder="invented_snapshot",
            )

        self.assertEqual(invented.exception.kind, FailureKind.CONFLICT)

    def test_planning_does_not_depend_on_http_schema_serialization(self) -> None:
        with patch.object(
            ConfigSchemaResponse,
            "__init__",
            side_effect=AssertionError("HTTP serialization reached"),
        ):
            plan = TrainingJobServiceHarness(runner=FakeRunner()).create_run_plan(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={"learning_rate": "0.01"},
                search=None,
                log_folder="semantic_schema",
            )

        self.assertEqual(plan.overrides, {"LEARNING_RATE": 0.01})

    def test_cli_and_workbench_materialize_equivalent_grid_plan(self) -> None:
        package = model_package("linears/linear")
        if package is None:
            self.fail("Expected the linears/linear Model Package.")
        cli_search = _search_spec(
            SimpleNamespace(
                search_mode=GridSearch(),
                search_keys=["hidden_dim", "stack_activation"],
                search_overrides={
                    "hidden_dim": [64, 128],
                    "stack_activation": ["RELU", "GELU"],
                },
            )
        )
        cli_plan = plan_runs(
            package,
            RunRequest(
                presets=("baseline", "gating"),
                datasets=("Mnist", "Cifar10"),
                overrides={"stack_num_layers": "4"},
                search=cli_search,
            ),
        )
        workbench_plan = run_plan_payload(
            TrainingJobServiceHarness(runner=FakeRunner()).create_run_plan(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist", "Cifar10"],
                overrides={"stack_num_layers": "4"},
                search={
                    "mode": "grid",
                    "values": {
                        "hidden_dim": [64, 128],
                        "stack_activation": ["RELU", "GELU"],
                    },
                },
                log_folder="",
            )
        )

        self.assertEqual(workbench_plan["presets"], list(cli_plan.presets))
        self.assertEqual(
            workbench_plan["experimentTask"],
            cli_plan.experiment_task,
        )
        self.assertEqual(workbench_plan["datasets"], list(cli_plan.datasets))
        self.assertEqual(workbench_plan["overrides"], dict(cli_plan.overrides))
        self.assertEqual(
            [
                (
                    row["id"],
                    row["preset"],
                    row["dataset"],
                    row["overrides"],
                )
                for row in workbench_plan["runs"]
            ],
            [
                (run.id, run.preset, run.dataset, dict(run.overrides))
                for run in cli_plan.runs
            ],
        )

    def test_training_job_accepts_grid_search_and_strips_conflicting_override(
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
                datasets=["Mnist"],
                overrides={"hidden_dim": "999", "stack_num_layers": "4"},
                search={
                    "mode": "grid",
                    "values": {"hidden_dim": [64, 128]},
                },
                log_folder="grid_search",
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        self.assertEqual(payload["overrides"], {"STACK_NUM_LAYERS": 4})
        self.assertEqual(payload["search"]["mode"], "grid")
        self.assertEqual(payload["search"]["values"], {"HIDDEN_DIM": [64, 128]})
        self.assertEqual(payload["plannedRunCount"], 4)
        self.assertEqual(
            worker_payload["runPlan"]["overrides"],
            {"STACK_NUM_LAYERS": 4},
        )
        self.assertEqual(worker_payload["runPlan"]["search"], payload["search"])

    def test_workbench_adapter_serializes_search_payload(self) -> None:
        plan = run_plan_payload(
            TrainingJobServiceHarness(runner=FakeRunner()).create_run_plan(
                model="linears/linear_adaptive",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                search={
                    "mode": "grid",
                    "values": {
                        "hidden_dim": [64],
                        "stack_activation": ["RELU"],
                        "adaptive_generator_stack_num_layers": [1, 2],
                    },
                },
                log_folder="",
            )
        )

        self.assertEqual(
            plan["search"],
            {
                "mode": "grid",
                "values": {
                    "HIDDEN_DIM": [64],
                    "STACK_ACTIVATION": ["RELU"],
                    "ADAPTIVE_GENERATOR_STACK_NUM_LAYERS": [1, 2],
                },
            },
        )

    def test_training_run_plan_materializes_grid_rows_and_commands(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        plan = run_plan_payload(
            manager.create_run_plan(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"hidden_dim": "999", "stack_num_layers": "4"},
                search={
                    "mode": "grid",
                    "values": {"hidden_dim": [64, 128]},
                },
                log_folder="",
            )
        )

        self.assertEqual(plan["summary"]["totalRuns"], 4)
        self.assertEqual(plan["summary"]["remainingEpochs"], 120)
        self.assertFalse(plan["isRandomSearch"])
        self.assertEqual(plan["runs"][0]["preset"], "baseline")
        self.assertEqual(plan["runs"][0]["dataset"], "Mnist")
        self.assertEqual(plan["runs"][0]["status"], "Pending")
        self.assertEqual(plan["runs"][0]["overrides"]["HIDDEN_DIM"], 64)
        self.assertEqual(plan["runs"][0]["overrides"]["STACK_NUM_LAYERS"], "4")
        self.assertEqual(
            [change["source"] for change in plan["runs"][0]["changes"]],
            ["override", "search"],
        )
        self.assertIn("--datasets Mnist", plan["runs"][0]["commands"]["posix"])
        self.assertIn("--hidden-dim 64", plan["runs"][0]["commands"]["posix"])
        self.assertIn("--stack-num-layers 4", plan["runs"][0]["commands"]["posix"])
        self.assertNotIn("--logdir", plan["runs"][0]["commands"]["posix"])

    def test_training_run_plan_commands_include_selected_monitors(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        plan = run_plan_payload(
            manager.create_run_plan(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="monitor_plan",
                monitors=["linear"],
            )
        )

        self.assertEqual(
            plan["runs"][0]["commands"]["posix"],
            "mise run experiment -- --model-type linears --model linear "
            "--preset baseline --experiment-task image-classification "
            "--datasets Mnist --logdir monitor_plan "
            "--monitors linear --config --hidden-dim 128",
        )

    def test_training_run_plan_rejects_path_like_dataset_input(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        with self.assertRaises(RunPlanFailure) as context:
            manager.create_run_plan(
                model="linears/linear",
                preset="baseline",
                datasets=["./Mnist"],
                overrides={},
                log_folder="path_like_dataset",
            )

        message = str(context.exception)
        self.assertIn("./Mnist", message)
        self.assertIn("filesystem path", message)
        self.assertIn("server-known dataset name", message)

    def test_workbench_strictly_rejects_equal_locked_values(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())
        with self.assertRaisesRegex(RunPlanFailure, "locked fields"):
            manager.create_run_plan(
                model="linears/linear",
                preset="gating",
                presets=["gating"],
                datasets=["Mnist"],
                overrides={"stack_gate_flag": "true"},
                log_folder="",
            )
        with self.assertRaisesRegex(RunPlanFailure, "locked by preset"):
            manager.create_run_plan(
                model="linears/linear",
                preset="post-norm",
                presets=["post-norm"],
                datasets=["Mnist"],
                overrides={},
                search={
                    "mode": "grid",
                    "values": {"layer_norm_position": ["AFTER"]},
                },
                log_folder="",
            )

    def test_training_run_plan_grid_search_characterizes_order_and_payload(
        self,
    ) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        plan = run_plan_payload(
            manager.create_run_plan(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist", "Cifar10"],
                overrides={
                    "hidden_dim": "999",
                    "stack_activation": "TANH",
                    "stack_num_layers": "4",
                },
                search={
                    "mode": "grid",
                    "values": {
                        "hidden_dim": [64, 128],
                        "stack_activation": ["RELU", "GELU"],
                    },
                },
                log_folder="grid_plan",
            )
        )

        self.assertEqual(plan["preset"], "baseline")
        self.assertEqual(plan["presets"], ["baseline", "gating"])
        self.assertEqual(plan["datasets"], ["Mnist", "Cifar10"])
        self.assertEqual(plan["overrides"], {"STACK_NUM_LAYERS": 4})
        self.assertEqual(
            plan["search"],
            {
                "mode": "grid",
                "values": {
                    "HIDDEN_DIM": [64, 128],
                    "STACK_ACTIVATION": ["RELU", "GELU"],
                },
            },
        )
        self.assertEqual(plan["logFolder"], "grid_plan")
        self.assertFalse(plan["isRandomSearch"])
        self.assertEqual(
            plan["summary"],
            {
                "totalRuns": 16,
                "completedRuns": 0,
                "runningRuns": 0,
                "pendingRuns": 16,
                "failedRuns": 0,
                "cancelledRuns": 0,
                "skippedRuns": 0,
                "totalEpochs": 480,
                "completedEpochs": 0,
                "remainingEpochs": 480,
            },
        )
        self.assertEqual(
            [
                (
                    run["preset"],
                    run["dataset"],
                    run["overrides"]["HIDDEN_DIM"],
                    run["overrides"]["STACK_ACTIVATION"],
                )
                for run in plan["runs"]
            ],
            [
                ("baseline", "Mnist", 64, "RELU"),
                ("baseline", "Mnist", 64, "GELU"),
                ("baseline", "Mnist", 128, "RELU"),
                ("baseline", "Mnist", 128, "GELU"),
                ("baseline", "Cifar10", 64, "RELU"),
                ("baseline", "Cifar10", 64, "GELU"),
                ("baseline", "Cifar10", 128, "RELU"),
                ("baseline", "Cifar10", 128, "GELU"),
                ("gating", "Mnist", 64, "RELU"),
                ("gating", "Mnist", 64, "GELU"),
                ("gating", "Mnist", 128, "RELU"),
                ("gating", "Mnist", 128, "GELU"),
                ("gating", "Cifar10", 64, "RELU"),
                ("gating", "Cifar10", 64, "GELU"),
                ("gating", "Cifar10", 128, "RELU"),
                ("gating", "Cifar10", 128, "GELU"),
            ],
        )
        self.assertEqual(
            [run["id"] for run in plan["runs"]],
            [f"run-{index:04d}" for index in range(1, 17)],
        )
        self.assertEqual(
            [run["index"] for run in plan["runs"]],
            list(range(1, 17)),
        )
        self.assertEqual(
            set(plan["runs"][0]),
            {
                "id",
                "index",
                "status",
                "preset",
                "experimentTask",
                "dataset",
                "changes",
                "overrides",
                "commandArgv",
                "commands",
                "totalEpochs",
                "currentEpoch",
                "metrics",
                "logDir",
                "error",
                "errorTraceback",
            },
        )
        self.assertEqual(plan["runs"][0]["status"], "Pending")
        self.assertEqual(plan["runs"][0]["totalEpochs"], 30)
        self.assertEqual(plan["runs"][0]["currentEpoch"], 0)
        self.assertEqual(plan["runs"][0]["metrics"], {})
        self.assertIsNone(plan["runs"][0]["logDir"])
        self.assertIsNone(plan["runs"][0]["error"])
        self.assertIsNone(plan["runs"][0]["errorTraceback"])
        self.assertEqual(
            plan["runs"][0]["changes"],
            [
                {
                    "key": "STACK_NUM_LAYERS",
                    "label": "stack num layers",
                    "value": "4",
                    "source": "override",
                },
                {
                    "key": "HIDDEN_DIM",
                    "label": "hidden dim",
                    "value": 64,
                    "source": "search",
                },
                {
                    "key": "STACK_ACTIVATION",
                    "label": "stack activation",
                    "value": "RELU",
                    "source": "search",
                },
            ],
        )
        self.assertEqual(
            plan["runs"][0]["commands"]["posix"],
            "mise run experiment -- --model-type linears --model linear "
            "--preset baseline --experiment-task image-classification --datasets Mnist "
            "--logdir grid_plan --config --hidden-dim 64 "
            "--stack-num-layers 4 --stack-activation RELU",
        )
        self.assertEqual(
            plan["runs"][0]["commandArgv"][:4],
            ["mise", "run", "experiment", "--"],
        )
        self.assertEqual(
            plan["runs"][0]["commands"]["posix"],
            " ".join(plan["runs"][0]["commandArgv"]),
        )
        self.assertEqual(
            plan["runs"][0]["commands"]["powershell"],
            " ".join(plan["runs"][0]["commandArgv"]),
        )
        self.assertEqual(
            plan["runs"][-1]["commands"]["posix"],
            "mise run experiment -- --model-type linears --model linear "
            "--preset gating --experiment-task image-classification --datasets Cifar10 "
            "--logdir grid_plan --config --hidden-dim 128 "
            "--stack-num-layers 4 --stack-activation GELU",
        )

    def test_training_run_plan_serializes_search_values_in_rows_and_commands(
        self,
    ) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        plan = run_plan_payload(
            manager.create_run_plan(
                model="linears/linear_adaptive",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                search={
                    "mode": "grid",
                    "values": {
                        "hidden_dim": [64],
                        "stack_activation": ["RELU"],
                        "adaptive_generator_stack_num_layers": [1, 2],
                    },
                },
                log_folder="serialization_plan",
            )
        )

        self.assertEqual(
            plan["search"],
            {
                "mode": "grid",
                "values": {
                    "HIDDEN_DIM": [64],
                    "STACK_ACTIVATION": ["RELU"],
                    "ADAPTIVE_GENERATOR_STACK_NUM_LAYERS": [1, 2],
                },
            },
        )
        self.assertEqual(plan["summary"]["totalRuns"], 2)

        runs_by_generator_layers = {
            run["overrides"]["ADAPTIVE_GENERATOR_STACK_NUM_LAYERS"]: run
            for run in plan["runs"]
        }
        shallow_run = runs_by_generator_layers[1]
        deep_run = runs_by_generator_layers[2]

        for run, num_layers in (
            (shallow_run, 1),
            (deep_run, 2),
        ):
            with self.subTest(num_layers=num_layers):
                changes_by_key = {change["key"]: change for change in run["changes"]}

                self.assertEqual(run["overrides"]["HIDDEN_DIM"], 64)
                self.assertEqual(run["overrides"]["STACK_ACTIVATION"], "RELU")
                self.assertEqual(
                    run["overrides"]["ADAPTIVE_GENERATOR_STACK_NUM_LAYERS"],
                    num_layers,
                )
                self.assertEqual(changes_by_key["HIDDEN_DIM"]["value"], 64)
                self.assertEqual(changes_by_key["STACK_ACTIVATION"]["value"], "RELU")
                self.assertEqual(
                    changes_by_key["ADAPTIVE_GENERATOR_STACK_NUM_LAYERS"]["value"],
                    num_layers,
                )
                self.assertTrue(
                    all(change["source"] == "search" for change in run["changes"])
                )
                self.assertIn("--hidden-dim 64", run["commands"]["posix"])
                self.assertIn("--stack-activation RELU", run["commands"]["posix"])
                self.assertIn(
                    f"--adaptive-generator-stack-num-layers {num_layers}",
                    run["commands"]["posix"],
                )

    def test_training_job_accepts_random_search_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist", "Cifar10"],
                overrides={},
                search={
                    "mode": "random",
                    "values": {
                        "hidden_dim": [64, 128],
                        "stack_activation": ["RELU", "GELU"],
                    },
                    "randomSamples": 3,
                },
                log_folder="random_search",
            )

        self.assertEqual(payload["search"]["mode"], "random")
        self.assertEqual(payload["search"]["randomSamples"], 3)
        self.assertEqual(payload["plannedRunCount"], 6)

    def test_training_run_plan_materializes_random_search_before_start(self) -> None:
        manager = TrainingJobServiceHarness(runner=FakeRunner())

        plan = run_plan_payload(
            manager.create_run_plan(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist", "Cifar10"],
                overrides={},
                search={
                    "mode": "random",
                    "values": {
                        "hidden_dim": [64, 128],
                        "stack_activation": ["RELU", "GELU"],
                    },
                    "randomSamples": 3,
                },
                log_folder="random_search",
            )
        )

        self.assertTrue(plan["isRandomSearch"])
        self.assertEqual(plan["summary"]["totalRuns"], 6)
        self.assertEqual(plan["summary"]["remainingEpochs"], 180)
        self.assertTrue(
            all(
                any(change["source"] == "search" for change in run["changes"])
                for run in plan["runs"]
            )
        )
        self.assertTrue(
            all(
                "--logdir random_search" in run["commands"]["posix"]
                for run in plan["runs"]
            )
        )

    def test_training_job_random_search_uses_seeded_materialized_run_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
                run_plans=_run_plan_service(random_source=random.Random(13)),
            )

            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist", "Cifar10"],
                overrides={
                    "hidden_dim": "999",
                    "stack_activation": "TANH",
                    "stack_num_layers": "4",
                },
                search={
                    "mode": "random",
                    "values": {
                        "hidden_dim": [64, 128],
                        "stack_activation": ["RELU", "GELU"],
                    },
                    "randomSamples": 3,
                },
                log_folder="random_plan",
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        expected_search = {
            "mode": "random",
            "values": {
                "HIDDEN_DIM": [64, 128],
                "STACK_ACTIVATION": ["RELU", "GELU"],
            },
            "randomSamples": 3,
        }
        plan = payload["runPlan"]

        self.assertEqual(payload["overrides"], {"STACK_NUM_LAYERS": 4})
        self.assertEqual(
            worker_payload["runPlan"]["overrides"],
            {"STACK_NUM_LAYERS": 4},
        )
        self.assertEqual(payload["search"], expected_search)
        self.assertEqual(worker_payload["runPlan"]["search"], expected_search)
        self.assertEqual(payload["plannedRunCount"], 12)
        self.assertEqual(worker_payload["plannedRunCount"], 12)
        self.assertEqual(worker_payload["runPlan"], plan)
        self.assertTrue(plan["isRandomSearch"])
        self.assertEqual(plan["search"], expected_search)
        self.assertEqual(plan["overrides"], {"STACK_NUM_LAYERS": 4})
        self.assertEqual(plan["logFolder"], "random_plan")
        self.assertEqual(
            plan["summary"],
            {
                "totalRuns": 12,
                "completedRuns": 0,
                "runningRuns": 0,
                "pendingRuns": 12,
                "failedRuns": 0,
                "cancelledRuns": 0,
                "skippedRuns": 0,
                "totalEpochs": 360,
                "completedEpochs": 0,
                "remainingEpochs": 360,
            },
        )
        self.assertEqual(
            Counter((run["preset"], run["dataset"]) for run in plan["runs"]),
            Counter(
                {
                    ("baseline", "Mnist"): 3,
                    ("baseline", "Cifar10"): 3,
                    ("gating", "Mnist"): 3,
                    ("gating", "Cifar10"): 3,
                }
            ),
        )
        self.assertEqual(
            [
                (
                    run["preset"],
                    run["dataset"],
                    run["overrides"]["HIDDEN_DIM"],
                    run["overrides"]["STACK_ACTIVATION"],
                )
                for run in plan["runs"]
            ],
            [
                ("baseline", "Mnist", 128, "RELU"),
                ("baseline", "Mnist", 64, "GELU"),
                ("baseline", "Mnist", 64, "RELU"),
                ("baseline", "Cifar10", 64, "GELU"),
                ("baseline", "Cifar10", 128, "RELU"),
                ("baseline", "Cifar10", 64, "RELU"),
                ("gating", "Mnist", 64, "GELU"),
                ("gating", "Mnist", 128, "RELU"),
                ("gating", "Mnist", 64, "RELU"),
                ("gating", "Cifar10", 64, "GELU"),
                ("gating", "Cifar10", 64, "RELU"),
                ("gating", "Cifar10", 128, "RELU"),
            ],
        )
        self.assertEqual(
            plan["runs"][0]["changes"],
            [
                {
                    "key": "STACK_NUM_LAYERS",
                    "label": "stack num layers",
                    "value": "4",
                    "source": "override",
                },
                {
                    "key": "HIDDEN_DIM",
                    "label": "hidden dim",
                    "value": 128,
                    "source": "search",
                },
                {
                    "key": "STACK_ACTIVATION",
                    "label": "stack activation",
                    "value": "RELU",
                    "source": "search",
                },
            ],
        )
        self.assertEqual(
            plan["runs"][0]["commands"]["posix"],
            "mise run experiment -- --model-type linears --model linear "
            "--preset baseline --experiment-task image-classification --datasets Mnist "
            "--logdir random_plan --config --hidden-dim 128 "
            "--stack-num-layers 4 --stack-activation RELU",
        )
        self.assertEqual(
            plan["runs"][-1]["commands"]["posix"],
            "mise run experiment -- --model-type linears --model linear "
            "--preset gating --experiment-task image-classification --datasets Cifar10 "
            "--logdir random_plan --config --hidden-dim 128 "
            "--stack-num-layers 4 --stack-activation RELU",
        )

    def test_training_job_accepts_mixed_submitted_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            plan = run_plan_payload(
                manager.create_run_plan(
                    model="linears/linear",
                    preset="baseline",
                    presets=["baseline", "gating"],
                    datasets=["Mnist"],
                    overrides={"hidden_dim": "128"},
                    log_folder="draft_plan",
                )
            )
            self.assertEqual(len(plan["runs"]), 2)
            plan["runs"][0].update(
                {
                    "id": "frontend-row-b",
                    "index": 99,
                }
            )
            plan["runs"][1].update(
                {
                    "id": "frontend-row-a",
                    "index": 42,
                }
            )

            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="submitted_plan",
                monitors=["linear"],
                run_plan=plan,
            )
            payload_path = Path(tmp) / "jobs" / payload["id"] / "payload.json"
            worker_payload = json.loads(payload_path.read_text())

        normalized_plan = payload["runPlan"]
        self.assertEqual(normalized_plan["summary"]["totalRuns"], 2)
        self.assertEqual(payload["plannedRunCount"], 2)
        self.assertEqual(
            [run["id"] for run in normalized_plan["runs"]],
            ["frontend-row-b", "frontend-row-a"],
        )
        self.assertEqual(
            [run["index"] for run in normalized_plan["runs"]],
            [1, 2],
        )
        self.assertEqual(
            [run["snapshotId"] for run in normalized_plan["runs"]],
            [None, None],
        )
        self.assertEqual(
            [run["snapshotName"] for run in normalized_plan["runs"]],
            [None, None],
        )
        self.assertEqual(
            [run["commands"]["posix"] for run in normalized_plan["runs"]],
            [
                "mise run experiment -- --model-type linears --model linear "
                "--preset baseline "
                "--experiment-task image-classification --datasets Mnist "
                "--logdir submitted_plan --monitors linear "
                "--config --hidden-dim 128",
                "mise run experiment -- --model-type linears --model linear "
                "--preset gating --experiment-task image-classification "
                "--datasets Mnist "
                "--logdir submitted_plan --monitors linear "
                "--config --hidden-dim 128",
            ],
        )
        for run in normalized_plan["runs"]:
            with self.subTest(run=run["id"]):
                self.assertNotIn("draft_plan", run["commands"]["posix"])
                self.assertNotIn(str(run["snapshotId"]), run["commands"]["posix"])
                self.assertNotIn(str(run["snapshotName"]), run["commands"]["posix"])
        self.assertEqual(worker_payload["plannedRunCount"], 2)
        self.assertEqual(worker_payload["runPlan"], normalized_plan)

    def test_submitted_plan_derives_row_projection_and_epoch_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            plan = run_plan_payload(
                manager.create_run_plan(
                    model="linears/linear",
                    preset="baseline",
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="draft_plan",
                )
            )
            submitted_row = plan["runs"][0]
            submitted_row.update(
                {
                    "index": 99,
                    "status": "Completed",
                    "changes": [],
                    "overrides": {"NUM_EPOCHS": 7},
                    "totalEpochs": 999,
                    "currentEpoch": 999,
                    "metrics": {"client": 1},
                    "logDir": "/client/path",
                    "error": "client error",
                    "errorTraceback": "client traceback",
                }
            )
            plan["summary"].update(
                {
                    "totalRuns": 999,
                    "completedRuns": 999,
                    "totalEpochs": 999,
                    "completedEpochs": 999,
                    "remainingEpochs": 0,
                }
            )

            payload = manager.create_job_payload(
                model="linears/linear",
                preset="baseline",
                datasets=["Mnist"],
                overrides={},
                log_folder="accepted_plan",
                run_plan=plan,
            )

        accepted_plan = payload["runPlan"]
        accepted_row = accepted_plan["runs"][0]
        self.assertNotIn("command", accepted_row)
        self.assertEqual(accepted_row["index"], 1)
        self.assertEqual(accepted_row["status"], "Pending")
        self.assertEqual(accepted_row["overrides"], {"NUM_EPOCHS": 7})
        self.assertEqual(
            accepted_row["changes"],
            [
                {
                    "key": "NUM_EPOCHS",
                    "label": "num epochs",
                    "value": 7,
                    "source": "override",
                }
            ],
        )
        self.assertIn("--num-epochs 7", accepted_row["commands"]["posix"])
        self.assertEqual(accepted_row["totalEpochs"], 7)
        self.assertEqual(accepted_row["currentEpoch"], 0)
        self.assertEqual(accepted_row["metrics"], {})
        self.assertIsNone(accepted_row["logDir"])
        self.assertIsNone(accepted_row["error"])
        self.assertIsNone(accepted_row["errorTraceback"])
        self.assertEqual(
            accepted_plan["summary"],
            {
                "totalRuns": 1,
                "completedRuns": 0,
                "runningRuns": 0,
                "pendingRuns": 1,
                "failedRuns": 0,
                "cancelledRuns": 0,
                "skippedRuns": 0,
                "totalEpochs": 7,
                "completedEpochs": 0,
                "remainingEpochs": 7,
            },
        )

    def test_training_job_rejects_invalid_submitted_run_plan_rows(self) -> None:
        invalid_cases = [
            (
                "unknown preset",
                lambda submitted_plan: submitted_plan["runs"][0].update(
                    {"preset": "missing"}
                ),
                "unknown preset",
            ),
            (
                "unknown dataset",
                lambda submitted_plan: submitted_plan["runs"][0].update(
                    {"dataset": "MissingDataset"}
                ),
                "unknown dataset",
            ),
            (
                "empty runs",
                lambda submitted_plan: submitted_plan.update({"runs": []}),
                "at least one training run",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            for name, mutate, expected_message in invalid_cases:
                with self.subTest(name=name):
                    plan = run_plan_payload(
                        manager.create_run_plan(
                            model="linears/linear",
                            preset="baseline",
                            datasets=["Mnist"],
                            overrides={},
                            log_folder="",
                        )
                    )
                    mutate(plan)

                    with self.assertRaises(TrainingJobFailure) as context:
                        manager.create_job_payload(
                            model="linears/linear",
                            preset="baseline",
                            datasets=["Mnist"],
                            overrides={},
                            log_folder="submitted_invalid",
                            run_plan=plan,
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_training_run_plan_rejects_overlarge_submitted_run_plan(self) -> None:
        service = _run_plan_service()
        preview_command = CreateTrainingRunPlanCommand(
            model="linears/linear",
            preset="baseline",
            presets=None,
            datasets=["Mnist"],
            overrides={},
            log_folder="submitted_limit",
        )
        preview = service.preview(preview_command)
        source = preview.runs[0]
        submitted = SubmittedTrainingRunPlan(
            runs=[
                SubmittedTrainingRun(
                    id=f"frontend-row-{index}",
                    preset=source.preset,
                    dataset=source.dataset,
                    overrides=dict(source.overrides),
                )
                for index in range(MAX_TRAINING_PLANNED_RUNS + 1)
            ]
        )

        with self.assertRaises(RunPlanFailure) as context:
            service.materialize(
                MaterializeTrainingRunPlanCommand(
                    model=preview_command.model,
                    preset=preview_command.preset,
                    presets=preview_command.presets,
                    datasets=preview_command.datasets,
                    overrides=preview_command.overrides,
                    log_folder=preview_command.log_folder,
                    submitted_plan=submitted,
                ),
                validated_log_folder="submitted_limit",
            )

        self.assertIn("submitted runs exceeds 2000", str(context.exception))

    def test_training_job_rejects_locked_submitted_run_plan_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobServiceHarness(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            plan = run_plan_payload(
                manager.create_run_plan(
                    model="linears/linear",
                    preset="gating",
                    presets=["gating"],
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="",
                )
            )
            plan["runs"][0]["overrides"]["stack_gate_flag"] = "false"

            with self.assertRaises(TrainingJobFailure) as context:
                manager.create_job_payload(
                    model="linears/linear",
                    preset="gating",
                    presets=["gating"],
                    datasets=["Mnist"],
                    overrides={},
                    log_folder="submitted_locked",
                    run_plan=plan,
                )

        self.assertIn("locked", str(context.exception))


if __name__ == "__main__":
    unittest.main()
