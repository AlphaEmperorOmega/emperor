from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from emperor.experiments.base import GridSearch, RandomSearch
from emperor.experiments.monitors import MonitorOption
from emperor.experiments.progress import JsonlTrainingProgressCallback
from lightning.pytorch.callbacks import Callback

from viewer.backend import training_worker
from viewer.backend.inspector.errors import InspectorError
from viewer.backend.inspector.search import parse_training_search
from viewer.backend.training_events import NeuronClusterGrowthCallback
from viewer.backend.training_worker import search_mode_from_parsed_search


class FakeExperimentPreset(Enum):
    BASELINE = "Baseline"
    WIDE = "Wide"

    @classmethod
    def get_member(cls, name: str) -> FakeExperimentPreset:
        options = {
            "baseline": cls.BASELINE,
            "BASELINE": cls.BASELINE,
            "wide": cls.WIDE,
            "WIDE": cls.WIDE,
        }
        return options[name]

    @classmethod
    def cli_name(cls, name: str) -> str:
        return name.lower().replace("_", "-")


class Mnist:
    pass


class Cifar10:
    pass


FAKE_PAYLOAD_IDENTITY = {"modelType": "linears", "model": "linear"}


COMMON_PROGRESS_EVENT_KEYS = {
    "timestamp",
    "dataset",
    "preset",
    "presetKey",
    "logDir",
    "runId",
    "runIndex",
    "runTotal",
    "totalEpochs",
}


class FakeMonitorCallback(Callback):
    pass


class FakeExperiment:
    instances: list[FakeExperiment] = []
    training_error: Exception | None = None

    def __init__(self, preset: FakeExperimentPreset) -> None:
        self.preset = preset
        self.train_calls: list[dict[str, object]] = []
        self.instances.append(self)

    def train_model(self, **kwargs) -> None:
        self.train_calls.append(kwargs)
        if self.training_error is not None:
            raise self.training_error


def fake_model_parts(*, locked_fields=None):
    config_module = ModuleType("fake_model_config")
    monitor_options_module = ModuleType("fake_monitor_options")
    config_module.HIDDEN_DIM = 256
    config_module.GATE_FLAG = False
    monitor_options_module.MONITOR_OPTIONS = [
        MonitorOption(
            name="fake-monitor",
            label="Fake Monitor",
            description="Fake monitor callback",
            kinds=["scalar"],
            callback_factory=FakeMonitorCallback,
        )
    ]
    return SimpleNamespace(
        name="fake_model",
        config_module=config_module,
        monitor_options_module=monitor_options_module,
        presets_module=SimpleNamespace(Experiment=FakeExperiment),
        model_module=SimpleNamespace(),
        experiment_preset_enum=FakeExperimentPreset,
        presets=SimpleNamespace(locked_fields=locked_fields or (lambda preset: {})),
        model_type=object,
        dataset_options=[Mnist, Cifar10],
        dataset=Mnist,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class TrainingWorkerSearchModeTests(unittest.TestCase):
    def test_search_mode_conversion_uses_experiment_search_types(self) -> None:
        self.assertIsNone(training_worker.search_mode_from_parsed_search(None))

        grid_search = SimpleNamespace(mode="grid", random_samples=None)
        self.assertIsInstance(
            training_worker.search_mode_from_parsed_search(grid_search),
            GridSearch,
        )

        random_search = SimpleNamespace(mode="random", random_samples=3)
        converted_random_search = training_worker.search_mode_from_parsed_search(
            random_search
        )
        self.assertIsInstance(converted_random_search, RandomSearch)
        self.assertEqual(converted_random_search.num_samples, 3)

    def test_search_mode_conversion_uses_default_random_sample_count(self) -> None:
        random_search = SimpleNamespace(mode="random", random_samples=None)

        converted_random_search = training_worker.search_mode_from_parsed_search(
            random_search
        )

        self.assertIsInstance(converted_random_search, RandomSearch)
        self.assertEqual(converted_random_search.num_samples, 10)

    def test_worker_search_mode_conversion_uses_experiment_search_types(self) -> None:
        grid_search = parse_training_search(
            "linears/linear",
            "baseline",
            {"mode": "grid", "values": {"hidden_dim": [64]}},
            dataset_count=1,
        )
        random_search = parse_training_search(
            "linears/linear",
            "baseline",
            {
                "mode": "random",
                "values": {"hidden_dim": [64, 128]},
                "randomSamples": 2,
            },
            dataset_count=1,
        )

        self.assertIsInstance(search_mode_from_parsed_search(grid_search), GridSearch)
        converted_random_search = search_mode_from_parsed_search(random_search)
        self.assertIsInstance(converted_random_search, RandomSearch)
        self.assertEqual(converted_random_search.num_samples, 2)


class TrainingWorkerMaterializedRunConversionTests(unittest.TestCase):
    def materialized_runs(
        self,
        payload: dict[str, object],
        *,
        parts=None,
    ) -> list[dict[str, object]] | None:
        parts = parts or fake_model_parts()
        with (
            patch(
                "viewer.backend.training_worker.load_model_parts", return_value=parts
            ),
            patch(
                "viewer.backend.inspector.schema.load_model_parts", return_value=parts
            ),
        ):
            return training_worker._materialized_runs_from_plan(parts, payload)

    def test_materialized_runs_returns_none_without_nonempty_plan_rows(self) -> None:
        cases = [
            {},
            {"runPlan": None},
            {"runPlan": "not-a-plan"},
            {"runPlan": {}},
            {"runPlan": {"runs": "not-a-list"}},
            {"runPlan": {"runs": []}},
        ]

        for payload in cases:
            with self.subTest(payload=payload):
                self.assertIsNone(
                    self.materialized_runs({**FAKE_PAYLOAD_IDENTITY, **payload})
                )

    def test_materialized_runs_preserves_rows_and_parses_overrides(self) -> None:
        runs = self.materialized_runs(
            {
                **FAKE_PAYLOAD_IDENTITY,
                "runPlan": {
                    "runs": [
                        {
                            "id": "frontend-row-1",
                            "index": 42,
                            "preset": "baseline",
                            "dataset": "Mnist",
                            "overrides": {
                                "hidden-dim": "64",
                                "gate_flag": "true",
                            },
                        },
                        {
                            "preset": "wide",
                            "dataset": "cifar10",
                            "overrides": {},
                        },
                    ]
                },
            }
        )

        self.assertIsNotNone(runs)
        assert runs is not None
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0]["id"], "frontend-row-1")
        self.assertEqual(runs[0]["index"], 42)
        self.assertEqual(runs[0]["run_total"], 2)
        self.assertIs(runs[0]["preset"], FakeExperimentPreset.BASELINE)
        self.assertIs(runs[0]["dataset_type"], Mnist)
        self.assertEqual(
            runs[0]["config_overrides"],
            {
                "hidden_dim": 64,
                "stack_gate_flag": True,
            },
        )
        self.assertEqual(runs[1]["id"], "run-0002")
        self.assertEqual(runs[1]["index"], 2)
        self.assertEqual(runs[1]["run_total"], 2)
        self.assertIs(runs[1]["preset"], FakeExperimentPreset.WIDE)
        self.assertIs(runs[1]["dataset_type"], Cifar10)
        self.assertEqual(runs[1]["config_overrides"], {})

    def test_materialized_runs_skip_non_dict_rows(self) -> None:
        runs = self.materialized_runs(
            {
                **FAKE_PAYLOAD_IDENTITY,
                "runPlan": {
                    "runs": [
                        "not-a-row",
                        {
                            "preset": "baseline",
                            "dataset": "Mnist",
                            "overrides": {},
                        },
                    ]
                },
            }
        )

        self.assertIsNotNone(runs)
        assert runs is not None
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["id"], "run-0002")
        self.assertEqual(runs[0]["index"], 2)
        self.assertEqual(runs[0]["run_total"], 2)

    def test_materialized_runs_reject_invalid_rows(self) -> None:
        invalid_cases = [
            (
                "invalid preset",
                {"preset": "missing", "dataset": "Mnist", "overrides": {}},
                KeyError,
                "missing",
            ),
            (
                "invalid dataset",
                {"preset": "baseline", "dataset": "MissingDataset", "overrides": {}},
                InspectorError,
                "Unknown dataset",
            ),
            (
                "unknown override",
                {
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "overrides": {"missing": "1"},
                },
                InspectorError,
                "Unknown override",
            ),
            (
                "invalid override value",
                {
                    "preset": "baseline",
                    "dataset": "Mnist",
                    "overrides": {"hidden_dim": "wide"},
                },
                InspectorError,
                "Invalid value for override",
            ),
        ]

        for name, row, expected_error, expected_message in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(expected_error) as context:
                    self.materialized_runs(
                        {
                            **FAKE_PAYLOAD_IDENTITY,
                            "runPlan": {"runs": [row]},
                        }
                    )

                self.assertIn(expected_message, str(context.exception))

    def test_materialized_runs_reject_locked_overrides(self) -> None:
        def locked_fields(preset):
            if preset is FakeExperimentPreset.WIDE:
                return {
                    "stack_gate_flag": SimpleNamespace(
                        value=True,
                        reason="locked by fake wide preset",
                    )
                }
            return {}

        with self.assertRaises(InspectorError) as context:
            self.materialized_runs(
                {
                    **FAKE_PAYLOAD_IDENTITY,
                    "runPlan": {
                        "runs": [
                            {
                                "preset": "wide",
                                "dataset": "Mnist",
                                "overrides": {"gate_flag": "false"},
                            }
                        ]
                    },
                },
                parts=fake_model_parts(locked_fields=locked_fields),
            )

        self.assertIn("locked fields", str(context.exception))
        self.assertIn("stack_gate_flag", str(context.exception))


class TrainingWorkerPayloadProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeExperiment.instances = []
        FakeExperiment.training_error = None

    def run_worker(self, payload: dict[str, object], progress_path: Path) -> None:
        payload_path = progress_path.parent / "payload.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        parts = fake_model_parts()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "training_worker",
                    "--payload",
                    str(payload_path),
                    "--progress",
                    str(progress_path),
                ],
            ),
            patch(
                "viewer.backend.training_worker.load_model_parts", return_value=parts
            ),
            patch(
                "viewer.backend.inspector.schema.load_model_parts", return_value=parts
            ),
        ):
            training_worker.main()

    def test_worker_loads_payload_and_writes_started_and_completed_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            payload = {
                "id": "job-123",
                **FAKE_PAYLOAD_IDENTITY,
                "preset": "baseline",
                "presets": ["baseline", "wide", "baseline"],
                "datasets": ["Mnist", "Cifar10", "Mnist"],
                "overrides": {},
                "search": None,
                "monitors": ["fake-monitor"],
                "logFolder": "unit_logs",
            }

            self.run_worker(payload, progress_path)

            self.assertEqual(len(FakeExperiment.instances), 1)
            experiment = FakeExperiment.instances[0]
            self.assertIs(experiment.preset, FakeExperimentPreset.BASELINE)
            self.assertEqual(len(experiment.train_calls), 1)
            train_call = experiment.train_calls[0]
            self.assertIsNone(train_call["search_mode"])
            self.assertEqual(train_call["log_folder"], "unit_logs")
            self.assertEqual(train_call["config_overrides"], {})
            self.assertIsNone(train_call["search_overrides"])
            self.assertEqual(
                train_call["selected_presets"],
                [FakeExperimentPreset.BASELINE, FakeExperimentPreset.WIDE],
            )
            self.assertEqual(train_call["selected_datasets"], [Mnist, Cifar10])
            self.assertIsNone(train_call["materialized_runs"])

            callbacks = train_call["callbacks"]
            self.assertIsInstance(callbacks[0], JsonlTrainingProgressCallback)
            self.assertEqual(callbacks[0].path, progress_path)
            self.assertEqual(
                callbacks[0].step_interval,
                training_worker.VIEWER_PROGRESS_STEP_INTERVAL,
            )
            self.assertIsInstance(callbacks[1], NeuronClusterGrowthCallback)
            self.assertIsInstance(callbacks[2], FakeMonitorCallback)

            events = read_jsonl(progress_path)
            self.assertEqual(
                [event["type"] for event in events], ["started", "completed"]
            )
            self.assertEqual(
                set(events[0]),
                COMMON_PROGRESS_EVENT_KEYS
                | {
                    "type",
                    "status",
                    "jobId",
                    "modelType",
                    "model",
                    "presets",
                    "datasets",
                    "monitors",
                },
            )
            self.assertEqual(
                set(events[1]),
                COMMON_PROGRESS_EVENT_KEYS | {"type", "status", "jobId", "presets"},
            )
            self.assertEqual(events[0]["status"], "running")
            self.assertEqual(events[0]["jobId"], "job-123")
            self.assertEqual(events[0]["modelType"], "linears")
            self.assertEqual(events[0]["model"], "linear")
            self.assertEqual(events[0]["preset"], "baseline")
            self.assertEqual(events[0]["presets"], ["baseline", "wide", "baseline"])
            self.assertEqual(events[0]["datasets"], ["Mnist", "Cifar10", "Mnist"])
            self.assertEqual(events[0]["monitors"], ["fake-monitor"])
            self.assertIn("timestamp", events[0])
            self.assertIsNone(events[0]["runId"])
            self.assertEqual(events[1]["status"], "completed")
            self.assertEqual(events[1]["jobId"], "job-123")
            self.assertEqual(events[1]["preset"], "wide")
            self.assertEqual(events[1]["presets"], ["baseline", "wide"])

    def test_neuron_cluster_growth_callback_caps_coordinate_payloads(self) -> None:
        events: list[dict[str, object]] = []
        callback = NeuronClusterGrowthCallback(events.append)
        initial_names = {"neuron_1_1_1"}
        all_names = {
            *initial_names,
            *(f"neuron_{index}_1_1" for index in range(2, 128)),
        }
        cluster = SimpleNamespace(
            cluster={name: object() for name in all_names},
            x_axis_total_neurons=200,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
        )
        callback._clusters = [("cluster", cluster)]
        callback._known_names = {"cluster": set(initial_names)}
        trainer = SimpleNamespace(current_epoch=3, global_step=42)

        callback.on_train_batch_end(trainer, object(), None, None, 0)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["type"], "neurons_added")
        self.assertEqual(event["node"], "cluster")
        self.assertEqual(event["coordinateCount"], len(all_names) - len(initial_names))
        self.assertEqual(len(event["coordinates"]), 100)
        self.assertTrue(event["coordinatesTruncated"])
        self.assertEqual(event["step"], 42)
        self.assertEqual(event["epoch"], 3)

    def test_worker_uses_materialized_plan_instead_of_search_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            payload = {
                "id": "job-plan",
                **FAKE_PAYLOAD_IDENTITY,
                "preset": "baseline",
                "presets": ["baseline"],
                "datasets": ["Mnist"],
                "overrides": {},
                "search": {"mode": "grid", "values": {"hidden_dim": [64]}},
                "monitors": [],
                "logFolder": "unit_logs",
                "runPlan": {
                    "runs": [
                        {
                            "id": "run-from-plan",
                            "index": 7,
                            "preset": "baseline",
                            "dataset": "Mnist",
                            "overrides": {"hidden_dim": "128"},
                        }
                    ]
                },
            }
            parsed_search = SimpleNamespace(
                mode="grid",
                random_samples=None,
                model_params={"hidden_dim"},
                search_overrides={"hidden_dim": [64]},
            )

            with patch(
                "viewer.backend.training_worker.parse_training_search",
                return_value=parsed_search,
            ) as parse_search:
                self.run_worker(payload, progress_path)

            parse_search.assert_called_once_with(
                "linears/linear",
                "baseline",
                {"mode": "grid", "values": {"hidden_dim": [64]}},
                dataset_count=1,
            )
            self.assertEqual(len(FakeExperiment.instances), 1)
            train_call = FakeExperiment.instances[0].train_calls[0]
            self.assertIsNone(train_call["search_mode"])
            self.assertIsNone(train_call["search_overrides"])
            self.assertEqual(train_call["config_overrides"], {})
            materialized_runs = train_call["materialized_runs"]
            self.assertIsNotNone(materialized_runs)
            assert materialized_runs is not None
            self.assertEqual(len(materialized_runs), 1)
            self.assertEqual(materialized_runs[0]["id"], "run-from-plan")
            self.assertEqual(materialized_runs[0]["index"], 7)
            self.assertEqual(materialized_runs[0]["run_total"], 1)
            self.assertIs(
                materialized_runs[0]["preset"],
                FakeExperimentPreset.BASELINE,
            )
            self.assertIs(materialized_runs[0]["dataset_type"], Mnist)
            self.assertEqual(
                materialized_runs[0]["config_overrides"],
                {"hidden_dim": 128},
            )

            events = read_jsonl(progress_path)
            self.assertEqual(
                [event["type"] for event in events], ["started", "completed"]
            )
            self.assertEqual(events[-1]["status"], "completed")

    def test_worker_writes_error_event_when_training_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            payload = {
                "id": "job-error",
                **FAKE_PAYLOAD_IDENTITY,
                "preset": "baseline",
                "presets": ["baseline"],
                "datasets": ["Mnist"],
                "overrides": {},
                "search": None,
                "monitors": [],
                "logFolder": "unit_logs",
            }
            FakeExperiment.training_error = RuntimeError("training failed")

            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                self.run_worker(payload, progress_path)

            self.assertEqual(raised.exception.code, 1)
            events = read_jsonl(progress_path)
            self.assertEqual([event["type"] for event in events], ["started", "error"])
            error_event = events[-1]
            self.assertEqual(
                set(error_event),
                COMMON_PROGRESS_EVENT_KEYS
                | {"type", "status", "jobId", "error", "traceback"},
            )
            self.assertEqual(error_event["status"], "failed")
            self.assertEqual(error_event["jobId"], "job-error")
            self.assertEqual(error_event["error"], "training failed")
            self.assertIsInstance(error_event["traceback"], str)
            self.assertIn(
                "Traceback (most recent call last):", error_event["traceback"]
            )
            self.assertIn("RuntimeError: training failed", error_event["traceback"])
            self.assertIn("timestamp", error_event)

    def test_worker_writes_started_then_error_for_invalid_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "progress.jsonl"
            payload = {
                "id": "job-invalid-plan",
                **FAKE_PAYLOAD_IDENTITY,
                "preset": "baseline",
                "presets": ["baseline"],
                "datasets": ["Mnist"],
                "overrides": {},
                "search": None,
                "monitors": [],
                "logFolder": "unit_logs",
                "runPlan": {
                    "runs": [
                        {
                            "preset": "baseline",
                            "dataset": "MissingDataset",
                            "overrides": {},
                        }
                    ]
                },
            }

            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                self.run_worker(payload, progress_path)

            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(FakeExperiment.instances, [])
            events = read_jsonl(progress_path)
            self.assertEqual([event["type"] for event in events], ["started", "error"])
            self.assertEqual(events[0]["status"], "running")
            self.assertEqual(events[0]["jobId"], "job-invalid-plan")
            error_event = events[-1]
            self.assertEqual(error_event["status"], "failed")
            self.assertEqual(error_event["jobId"], "job-invalid-plan")
            self.assertIn("Unknown dataset 'MissingDataset'", error_event["error"])
            self.assertIn(
                "Traceback (most recent call last):", error_event["traceback"]
            )


if __name__ == "__main__":
    unittest.main()
