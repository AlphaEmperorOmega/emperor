from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.linears.core.config import AdaptiveLinearLayerConfig
from emperor.base.options import ActivationOptions
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.search import parse_training_search
from viewer.backend.training_jobs import TrainingJobManager
from viewer.backend.training_run_plans import TrainingRunPlanBuilder
from viewer.backend.tests.helpers import FakeRunner


class TrainingRunPlanTests(unittest.TestCase):
    def test_training_job_accepts_grid_search_and_strips_conflicting_override(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job(
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

        self.assertEqual(payload["overrides"], {"stack_num_layers": "4"})
        self.assertEqual(payload["search"]["mode"], "grid")
        self.assertEqual(payload["search"]["values"], {"hidden_dim": [64, 128]})
        self.assertEqual(payload["plannedRunCount"], 4)
        self.assertEqual(worker_payload["overrides"], {"stack_num_layers": "4"})
        self.assertEqual(worker_payload["search"], payload["search"])

    def test_parse_training_search_serializes_values_payload(self) -> None:
        parsed = parse_training_search(
            "linears/linear_adaptive",
            "baseline",
            {
                "mode": "grid",
                "values": {
                    "hidden_dim": [64],
                    "stack_activation": ["RELU"],
                    "input_layer_model_option": [None, "AdaptiveLinearLayerConfig"],
                },
            },
            dataset_count=1,
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed.values,
            {
                "hidden_dim": [64],
                "stack_activation": ["RELU"],
                "input_layer_model_option": [None, "AdaptiveLinearLayerConfig"],
            },
        )
        self.assertEqual(
            parsed.to_payload(),
            {
                "mode": "grid",
                "values": {
                    "hidden_dim": [64],
                    "stack_activation": ["RELU"],
                    "input_layer_model_option": [None, "AdaptiveLinearLayerConfig"],
                },
            },
        )
        self.assertEqual(parsed.search_overrides["hidden_dim"], [64])
        self.assertIs(
            parsed.search_overrides["stack_activation"][0],
            ActivationOptions.RELU,
        )
        self.assertIsNone(parsed.search_overrides["input_layer_model_option"][0])
        self.assertIs(
            parsed.search_overrides["input_layer_model_option"][1],
            AdaptiveLinearLayerConfig,
        )

    def test_training_run_plan_materializes_grid_rows_and_commands(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        plan = manager.create_run_plan(
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

        self.assertEqual(plan["summary"]["totalRuns"], 4)
        self.assertEqual(plan["summary"]["remainingEpochs"], 120)
        self.assertFalse(plan["isRandomSearch"])
        self.assertEqual(plan["runs"][0]["preset"], "baseline")
        self.assertEqual(plan["runs"][0]["dataset"], "Mnist")
        self.assertEqual(plan["runs"][0]["status"], "Pending")
        self.assertEqual(plan["runs"][0]["overrides"]["hidden_dim"], 64)
        self.assertEqual(plan["runs"][0]["overrides"]["stack_num_layers"], "4")
        self.assertEqual(
            [change["source"] for change in plan["runs"][0]["changes"]],
            ["override", "search"],
        )
        self.assertIn("--datasets Mnist", plan["runs"][0]["command"])
        self.assertIn("--hidden-dim 64", plan["runs"][0]["command"])
        self.assertIn("--stack-num-layers 4", plan["runs"][0]["command"])
        self.assertNotIn("--logdir", plan["runs"][0]["command"])

    def test_training_run_plan_rejects_path_like_dataset_input(self) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        with self.assertRaises(InspectorError) as context:
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

    def test_training_run_plan_grid_search_characterizes_order_and_payload(
        self,
    ) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        plan = manager.create_run_plan(
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

        self.assertEqual(plan["preset"], "baseline")
        self.assertEqual(plan["presets"], ["baseline", "gating"])
        self.assertEqual(plan["datasets"], ["Mnist", "Cifar10"])
        self.assertEqual(plan["overrides"], {"stack_num_layers": "4"})
        self.assertEqual(
            plan["search"],
            {
                "mode": "grid",
                "values": {
                    "hidden_dim": [64, 128],
                    "stack_activation": ["RELU", "GELU"],
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
                    run["overrides"]["hidden_dim"],
                    run["overrides"]["stack_activation"],
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
                "dataset",
                "changes",
                "overrides",
                "command",
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
                    "key": "stack_num_layers",
                    "label": "stack num layers",
                    "value": "4",
                    "source": "override",
                },
                {
                    "key": "hidden_dim",
                    "label": "hidden dim",
                    "value": 64,
                    "source": "search",
                },
                {
                    "key": "stack_activation",
                    "label": "stack activation",
                    "value": "RELU",
                    "source": "search",
                },
            ],
        )
        self.assertEqual(
            plan["runs"][0]["command"],
            "source experiment.sh linears/linear --preset baseline --datasets Mnist "
            "--logdir grid_plan --config --hidden-dim 64 "
            "--stack-num-layers 4 --stack-activation RELU",
        )
        self.assertEqual(
            plan["runs"][-1]["command"],
            "source experiment.sh linears/linear --preset gating --datasets Cifar10 "
            "--logdir grid_plan --config --hidden-dim 128 "
            "--stack-num-layers 4 --stack-activation GELU",
        )

    def test_training_run_plan_serializes_search_values_in_rows_and_commands(
        self,
    ) -> None:
        manager = TrainingJobManager(runner=FakeRunner())

        plan = manager.create_run_plan(
            model="linears/linear_adaptive",
            preset="baseline",
            datasets=["Mnist"],
            overrides={},
            search={
                "mode": "grid",
                "values": {
                    "hidden_dim": [64],
                    "stack_activation": ["RELU"],
                    "input_layer_model_option": [None, "AdaptiveLinearLayerConfig"],
                },
            },
            log_folder="serialization_plan",
        )

        self.assertEqual(
            plan["search"],
            {
                "mode": "grid",
                "values": {
                    "hidden_dim": [64],
                    "stack_activation": ["RELU"],
                    "input_layer_model_option": [None, "AdaptiveLinearLayerConfig"],
                },
            },
        )
        self.assertEqual(plan["summary"]["totalRuns"], 2)

        runs_by_input_layer = {
            run["overrides"]["input_layer_model_option"]: run
            for run in plan["runs"]
        }
        none_run = runs_by_input_layer[None]
        class_run = runs_by_input_layer["AdaptiveLinearLayerConfig"]

        for run, input_layer_value, command_value in (
            (none_run, None, "None"),
            (class_run, "AdaptiveLinearLayerConfig", "AdaptiveLinearLayerConfig"),
        ):
            with self.subTest(input_layer_value=input_layer_value):
                changes_by_key = {change["key"]: change for change in run["changes"]}

                self.assertEqual(run["overrides"]["hidden_dim"], 64)
                self.assertEqual(run["overrides"]["stack_activation"], "RELU")
                self.assertEqual(
                    run["overrides"]["input_layer_model_option"],
                    input_layer_value,
                )
                self.assertEqual(changes_by_key["hidden_dim"]["value"], 64)
                self.assertEqual(changes_by_key["stack_activation"]["value"], "RELU")
                self.assertEqual(
                    changes_by_key["input_layer_model_option"]["value"],
                    input_layer_value,
                )
                self.assertTrue(
                    all(change["source"] == "search" for change in run["changes"])
                )
                self.assertIn("--hidden-dim 64", run["command"])
                self.assertIn("--stack-activation RELU", run["command"])
                self.assertIn(
                    f"--input-layer-model-option {command_value}",
                    run["command"],
                )

    def test_training_job_accepts_random_search_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )

            payload = manager.create_job(
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
        manager = TrainingJobManager(runner=FakeRunner())

        plan = manager.create_run_plan(
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
            all("--logdir random_search" in run["command"] for run in plan["runs"])
        )

    def test_training_job_random_search_uses_seeded_materialized_run_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
                run_plan_builder=TrainingRunPlanBuilder(
                    random_source=random.Random(13)
                ),
            )

            payload = manager.create_job(
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
                "hidden_dim": [64, 128],
                "stack_activation": ["RELU", "GELU"],
            },
            "randomSamples": 3,
        }
        plan = payload["runPlan"]

        self.assertEqual(payload["overrides"], {"stack_num_layers": "4"})
        self.assertEqual(worker_payload["overrides"], {"stack_num_layers": "4"})
        self.assertEqual(payload["search"], expected_search)
        self.assertEqual(worker_payload["search"], expected_search)
        self.assertEqual(payload["plannedRunCount"], 12)
        self.assertEqual(worker_payload["plannedRunCount"], 12)
        self.assertEqual(worker_payload["runPlan"], plan)
        self.assertTrue(plan["isRandomSearch"])
        self.assertEqual(plan["search"], expected_search)
        self.assertEqual(plan["overrides"], {"stack_num_layers": "4"})
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
                    run["overrides"]["hidden_dim"],
                    run["overrides"]["stack_activation"],
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
                    "key": "stack_num_layers",
                    "label": "stack num layers",
                    "value": "4",
                    "source": "override",
                },
                {
                    "key": "hidden_dim",
                    "label": "hidden dim",
                    "value": 128,
                    "source": "search",
                },
                {
                    "key": "stack_activation",
                    "label": "stack activation",
                    "value": "RELU",
                    "source": "search",
                },
            ],
        )
        self.assertEqual(
            plan["runs"][0]["command"],
            "source experiment.sh linears/linear --preset baseline --datasets Mnist "
            "--logdir random_plan --config --hidden-dim 128 "
            "--stack-num-layers 4 --stack-activation RELU",
        )
        self.assertEqual(
            plan["runs"][-1]["command"],
            "source experiment.sh linears/linear --preset gating --datasets Cifar10 "
            "--logdir random_plan --config --hidden-dim 128 "
            "--stack-num-layers 4 --stack-activation RELU",
        )

    def test_training_job_accepts_submitted_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            plan = manager.create_run_plan(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="draft_plan",
            )
            self.assertEqual(len(plan["runs"]), 2)
            plan["runs"][0].update(
                {
                    "id": "frontend-row-b",
                    "index": 99,
                    "snapshotId": "snapshot-1",
                    "snapshotName": "wide hidden",
                    "command": "stale --logdir draft_plan snapshot-1 wide hidden",
                }
            )
            plan["runs"][1].update(
                {
                    "id": "frontend-row-a",
                    "index": 42,
                    "snapshotId": "snapshot-2",
                    "snapshotName": "gating hidden",
                    "command": "stale --logdir draft_plan snapshot-2 gating hidden",
                }
            )

            payload = manager.create_job(
                model="linears/linear",
                preset="baseline",
                presets=["baseline", "gating"],
                datasets=["Mnist"],
                overrides={"hidden_dim": "128"},
                log_folder="submitted_plan",
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
            ["snapshot-1", "snapshot-2"],
        )
        self.assertEqual(
            [run["snapshotName"] for run in normalized_plan["runs"]],
            ["wide hidden", "gating hidden"],
        )
        self.assertEqual(
            [run["command"] for run in normalized_plan["runs"]],
            [
                "source experiment.sh linears/linear --preset baseline --datasets Mnist "
                "--logdir submitted_plan --config --hidden-dim 128",
                "source experiment.sh linears/linear --preset gating --datasets Mnist "
                "--logdir submitted_plan --config --hidden-dim 128",
            ],
        )
        for run in normalized_plan["runs"]:
            with self.subTest(run=run["id"]):
                self.assertNotIn("draft_plan", run["command"])
                self.assertNotIn(str(run["snapshotId"]), run["command"])
                self.assertNotIn(str(run["snapshotName"]), run["command"])
        self.assertEqual(worker_payload["plannedRunCount"], 2)
        self.assertEqual(worker_payload["runPlan"], normalized_plan)

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
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            for name, mutate, expected_message in invalid_cases:
                with self.subTest(name=name):
                    plan = manager.create_run_plan(
                        model="linears/linear",
                        preset="baseline",
                        datasets=["Mnist"],
                        overrides={},
                        log_folder="",
                    )
                    mutate(plan)

                    with self.assertRaises(InspectorError) as context:
                        manager.create_job(
                            model="linears/linear",
                            preset="baseline",
                            datasets=["Mnist"],
                            overrides={},
                            log_folder="submitted_invalid",
                            run_plan=plan,
                        )

                    self.assertIn(expected_message, str(context.exception))

    def test_training_job_rejects_locked_submitted_run_plan_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = TrainingJobManager(
                root=Path(tmp) / "jobs",
                logs_root=Path(tmp) / "logs",
                runner=FakeRunner(),
            )
            plan = manager.create_run_plan(
                model="linears/linear",
                preset="gating",
                presets=["gating"],
                datasets=["Mnist"],
                overrides={},
                log_folder="",
            )
            plan["runs"][0]["overrides"]["gate_flag"] = "false"

            with self.assertRaises(InspectorError) as context:
                manager.create_job(
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
