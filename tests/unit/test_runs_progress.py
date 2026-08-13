from __future__ import annotations

import gc
import json
import math
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from model_runtime.runs import JsonlRunProgress
from model_runtime.runs._lightning_progress import lightning_progress_adapter
from model_runtime.runs._metrics import portable_metric_values, sanitize_metric_payload
from model_runtime.runs._progress_events import (
    ClusterInitializedEvent,
    DatasetCompletedEvent,
    DatasetStartedEvent,
    EpochStartedEvent,
    FitCompletedEvent,
    NeuronAddedEvent,
    NeuronsAddedEvent,
    StepEvent,
    TrainingErrorEvent,
    ValidationEvent,
    project_run_progress_event,
)
from model_runtime.runs._progress_events import (
    TestCompletedEvent as CompletedTestEvent,
)
from model_runtime.runs.artifacts import FilesystemRunArtifacts
from model_runtime.runs.progress import ContextualRunProgress, RunProgressContext


def _context() -> RunProgressContext:
    return RunProgressContext(
        experiment_task="image-classification",
        dataset="Mnist",
        preset="baseline",
        preset_key="BASELINE",
        log_dir="logs/run/version_0",
        run_id="run-0001",
        run_index=1,
        run_total=1,
        total_epochs=30,
    )


class RunsProgressTests(unittest.TestCase):
    def test_growth_callback_releases_cluster_state_on_exception(self) -> None:
        events: list[dict[str, object]] = []

        class Cluster:
            def __init__(self) -> None:
                self.cluster = {"neuron_1_1_1": object()}
                self.x_axis_total_neurons = 1
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        cluster = Cluster()
        cluster_reference = weakref.ref(cluster)
        modules: list[tuple[str, object]] = [("cluster", cluster)]
        model = SimpleNamespace(named_modules=lambda: iter(modules))
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )

        with patch("emperor.neuron.NeuronCluster", Cluster):
            callback.on_fit_start(SimpleNamespace(), model)
        event_count = len(events)
        modules.clear()
        del cluster

        callback.on_exception(
            SimpleNamespace(),
            model,
            RuntimeError("fit failed"),
        )
        gc.collect()

        self.assertIsNone(cluster_reference())
        self.assertEqual(len(events), event_count)

    def test_growth_callback_releases_state_when_fit_completion_fails(self) -> None:
        failure = RuntimeError("completion projection failed")

        class Cluster:
            def __init__(self) -> None:
                self.cluster = {"neuron_1_1_1": object()}
                self.x_axis_total_neurons = 1
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        def write_event(event: object) -> None:
            if dict(event)["type"] == "fit_completed":
                raise failure

        cluster = Cluster()
        cluster_reference = weakref.ref(cluster)
        modules: list[tuple[str, object]] = [("cluster", cluster)]
        model = SimpleNamespace(named_modules=lambda: iter(modules))
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: write_event(event)},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )
        trainer = SimpleNamespace(
            current_epoch=1,
            global_step=2,
            callback_metrics={},
        )

        with patch("emperor.neuron.NeuronCluster", Cluster):
            callback.on_fit_start(trainer, model)
        modules.clear()
        del cluster

        with self.assertRaises(RuntimeError) as raised:
            callback.on_fit_end(trainer, model)
        gc.collect()

        self.assertIs(raised.exception, failure)
        self.assertIsNone(cluster_reference())

    def test_growth_callback_cleans_partial_startup_after_base_exception(
        self,
    ) -> None:
        cancellation = KeyboardInterrupt("startup cancelled")
        writes = 0

        class Cluster:
            def __init__(self, coordinate: int) -> None:
                self.cluster = {f"neuron_{coordinate}_1_1": object()}
                self.x_axis_total_neurons = 2
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        def write_event(_event: object) -> None:
            nonlocal writes
            writes += 1
            if writes == 2:
                raise cancellation

        first_cluster = Cluster(1)
        second_cluster = Cluster(2)
        references = (weakref.ref(first_cluster), weakref.ref(second_cluster))
        modules: list[tuple[str, object]] = [
            ("first", first_cluster),
            ("second", second_cluster),
        ]
        model = SimpleNamespace(named_modules=lambda: iter(modules))
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: write_event(event)},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )

        with (
            patch("emperor.neuron.NeuronCluster", Cluster),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            callback.on_fit_start(SimpleNamespace(), model)
        modules.clear()
        del first_cluster
        del second_cluster
        gc.collect()

        self.assertIs(raised.exception, cancellation)
        self.assertTrue(all(reference() is None for reference in references))

    def test_growth_callback_streams_a_bounded_lexical_burst_sample(self) -> None:
        events: list[dict[str, object]] = []

        class Cluster:
            def __init__(self) -> None:
                self.cluster: dict[str, object] = {}
                self.x_axis_total_neurons = 200
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        cluster = Cluster()
        model = SimpleNamespace(
            named_modules=lambda: iter((("cluster", cluster),)),
        )
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )
        trainer = SimpleNamespace(current_epoch=3, global_step=1)
        with patch("emperor.neuron.NeuronCluster", Cluster):
            callback.on_fit_start(trainer, model)
        events.clear()

        added_names = [f"neuron_{index}_1_1" for index in range(1, 102)]
        expected_names = sorted(added_names)[:100]
        cluster.cluster.update({name: object() for name in added_names})
        cluster.cluster["not_a_coordinate"] = object()

        with patch(
            "builtins.sorted",
            side_effect=AssertionError("growth sampling must not fully sort"),
        ):
            callback.on_train_batch_end(trainer, model, None, None, 0)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "neurons_added")
        self.assertEqual(
            events[0]["coordinates"],
            [[int(name.split("_")[1]), 1, 1] for name in expected_names],
        )
        self.assertEqual(events[0]["coordinateCount"], 101)
        self.assertIs(events[0]["coordinatesTruncated"], True)
        self.assertEqual(events[0]["count"], 102)

    def test_growth_callback_streams_initial_numeric_coordinate_sample(self) -> None:
        events: list[dict[str, object]] = []

        class Cluster:
            def __init__(self) -> None:
                self.cluster = {
                    **{f"neuron_{index}_1_1": object() for index in range(1, 102)},
                    "not_a_coordinate": object(),
                }
                self.x_axis_total_neurons = 200
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        cluster = Cluster()
        model = SimpleNamespace(
            named_modules=lambda: iter((("cluster", cluster),)),
        )
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )

        with (
            patch("emperor.neuron.NeuronCluster", Cluster),
            patch(
                "builtins.sorted",
                side_effect=AssertionError("initial sampling must not fully sort"),
            ),
        ):
            callback.on_fit_start(SimpleNamespace(), model)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "cluster_initialized")
        self.assertEqual(
            events[0]["coordinates"],
            [[index, 1, 1] for index in range(1, 101)],
        )
        self.assertEqual(events[0]["coordinateCount"], 101)
        self.assertIs(events[0]["coordinatesTruncated"], True)
        self.assertEqual(events[0]["count"], 102)

    def test_growth_callback_keeps_individual_events_at_burst_limit(self) -> None:
        events: list[dict[str, object]] = []

        class Cluster:
            def __init__(self) -> None:
                self.cluster: dict[str, object] = {}
                self.x_axis_total_neurons = 100
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        cluster = Cluster()
        model = SimpleNamespace(
            named_modules=lambda: iter((("cluster", cluster),)),
        )
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )
        trainer = SimpleNamespace(current_epoch=3, global_step=1)
        with patch("emperor.neuron.NeuronCluster", Cluster):
            callback.on_fit_start(trainer, model)
        events.clear()

        added_names = [f"neuron_{index}_1_1" for index in range(1, 101)]
        cluster.cluster.update({name: object() for name in added_names})
        callback.on_train_batch_end(trainer, model, None, None, 0)

        self.assertEqual(len(events), 100)
        self.assertTrue(all(event["type"] == "neuron_added" for event in events))
        self.assertEqual(
            [event["coord"] for event in events],
            [[int(name.split("_")[1]), 1, 1] for name in sorted(added_names)],
        )

    def test_growth_callback_detects_net_zero_change_and_later_regrowth(self) -> None:
        events: list[dict[str, object]] = []

        class Cluster:
            def __init__(self) -> None:
                self.cluster = {
                    "neuron_1_1_1": object(),
                    "neuron_2_1_1": object(),
                }
                self.x_axis_total_neurons = 10
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        cluster = Cluster()
        model = SimpleNamespace(
            named_modules=lambda: iter((("cluster", cluster),)),
        )
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )
        trainer = SimpleNamespace(current_epoch=0, global_step=1)
        with patch("emperor.neuron.NeuronCluster", Cluster):
            callback.on_fit_start(trainer, model)
        events.clear()

        del cluster.cluster["neuron_2_1_1"]
        cluster.cluster["neuron_10_1_1"] = object()
        callback.on_train_batch_end(trainer, model, None, None, 0)
        del cluster.cluster["neuron_10_1_1"]
        callback.on_train_batch_end(trainer, model, None, None, 1)
        cluster.cluster["neuron_10_1_1"] = object()
        callback.on_train_batch_end(trainer, model, None, None, 2)

        self.assertEqual(
            [(event["type"], event["coord"]) for event in events],
            [
                ("neuron_added", [10, 1, 1]),
                ("neuron_added", [10, 1, 1]),
            ],
        )

    def test_growth_projection_failure_keeps_state_advanced_until_cleanup(
        self,
    ) -> None:
        events: list[dict[str, object]] = []
        failure = RuntimeError("growth projection failed")
        reject_growth = False

        class Cluster:
            def __init__(self) -> None:
                self.cluster: dict[str, object] = {}
                self.x_axis_total_neurons = 2
                self.y_axis_total_neurons = 1
                self.z_axis_total_neurons = 1

        def write_event(event: object) -> None:
            payload = dict(event)
            if reject_growth and payload["type"] == "neuron_added":
                raise failure
            events.append(payload)

        cluster = Cluster()
        model = SimpleNamespace(
            named_modules=lambda: iter((("cluster", cluster),)),
        )
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: write_event(event)},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )
        trainer = SimpleNamespace(current_epoch=0, global_step=1)
        with patch("emperor.neuron.NeuronCluster", Cluster):
            callback.on_fit_start(trainer, model)
        events.clear()
        cluster.cluster["neuron_2_1_1"] = object()
        reject_growth = True

        with self.assertRaises(RuntimeError) as raised:
            callback.on_train_batch_end(trainer, model, None, None, 0)
        reject_growth = False
        callback.on_train_batch_end(trainer, model, None, None, 1)
        callback.on_exception(trainer, model, failure)

        self.assertIs(raised.exception, failure)
        self.assertEqual(events, [])

    def test_non_scalar_tensor_metric_is_omitted_before_host_transfer(self) -> None:
        metric = torch.ones(4, device="meta")

        payload = FilesystemRunArtifacts().result_metrics_payload(
            {"validation/accuracy": metric}
        )

        self.assertEqual(
            payload,
            {"metrics": {"validation/accuracy": "<non-scalar metric omitted>"}},
        )

    def test_lightning_progress_omits_non_scalar_tensor_before_host_transfer(
        self,
    ) -> None:
        events: list[dict[str, object]] = []
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),  # type: ignore[arg-type]
            step_interval=1,
        )
        trainer = SimpleNamespace(
            current_epoch=0,
            global_step=1,
            callback_metrics={"vector": torch.ones(4, device="meta")},
        )

        callback.on_train_batch_end(trainer, None, None, None, 0)

        self.assertEqual(
            events[0]["metrics"],
            {"vector": "<non-scalar metric omitted>"},
        )

    def test_tensor_like_metric_admits_scalar_before_transfer(self) -> None:
        calls: list[str] = []

        class MetricProbe:
            def __init__(self, count: Any) -> None:
                self._count = count

            def numel(self) -> Any:
                calls.append("numel")
                if isinstance(self._count, BaseException):
                    raise self._count
                return self._count

            def detach(self) -> MetricProbe:
                calls.append("detach")
                return self

            def cpu(self) -> MetricProbe:
                calls.append("cpu")
                return self

            def item(self) -> float:
                calls.append("item")
                return 0.75

            def __str__(self) -> str:
                calls.append("str")
                return "probe"

        self.assertEqual(
            portable_metric_values({"metric": MetricProbe(1)}),
            {"metric": 0.75},
        )
        self.assertEqual(calls, ["numel", "detach", "cpu", "item"])

        for count in (4, True, 1.0, RuntimeError("numel failure")):
            with self.subTest(count=count):
                calls.clear()
                self.assertEqual(
                    portable_metric_values({"metric": MetricProbe(count)}),
                    {"metric": "<non-scalar metric omitted>"},
                )
                self.assertEqual(calls, ["numel"])

        cancellation = KeyboardInterrupt("metric cancellation")
        calls.clear()
        with self.assertRaises(KeyboardInterrupt) as raised:
            portable_metric_values({"metric": MetricProbe(cancellation)})
        self.assertIs(raised.exception, cancellation)
        self.assertEqual(calls, ["numel"])

    def test_item_only_metric_wrapper_keeps_legacy_conversion(self) -> None:
        calls: list[str] = []

        class ItemOnlyMetric:
            @staticmethod
            def item() -> float:
                calls.append("item")
                return 0.625

        self.assertEqual(
            portable_metric_values({"metric": ItemOnlyMetric()}),
            {"metric": 0.625},
        )
        self.assertEqual(calls, ["item"])

    def test_metric_selection_precedes_scalar_admission(self) -> None:
        class NumelTrap:
            @staticmethod
            def numel() -> int:
                raise AssertionError("filtered metric must not be inspected")

        sanitized, original_count, dropped_count = sanitize_metric_payload(
            {
                "validation/confusion_matrix": NumelTrap(),
                "retained": 0.5,
                "truncated": NumelTrap(),
            },
            metric_key_limit=1,
            string_value_limit=100,
        )

        self.assertEqual(sanitized, {"retained": 0.5})
        self.assertEqual(original_count, 3)
        self.assertEqual(dropped_count, 2)

    def test_scalar_tensor_metrics_keep_values_and_finite_validation(self) -> None:
        self.assertEqual(
            portable_metric_values(
                {
                    "zero_dimensional": torch.tensor(0.25),
                    "one_element": torch.tensor([0.75]),
                }
            ),
            {"zero_dimensional": 0.25, "one_element": 0.75},
        )
        with self.assertRaises(ValueError):
            sanitize_metric_payload(
                {"loss": torch.tensor(float("inf"))},
                metric_key_limit=1,
                string_value_limit=100,
            )

    def test_typed_progress_events_have_one_exact_wire_projection(self) -> None:
        cases = (
            (
                DatasetStartedEvent(params={"learning_rate": 0.1}),
                [
                    ("type", "dataset_started"),
                    ("status", "running"),
                    ("params", {"learning_rate": 0.1}),
                ],
            ),
            (
                DatasetStartedEvent(
                    params={"learning_rate": 0.1},
                    resumed_from={"epoch": 2},
                ),
                [
                    ("type", "dataset_started"),
                    ("status", "running"),
                    ("params", {"learning_rate": 0.1}),
                    ("resumedFrom", {"epoch": 2}),
                ],
            ),
            (
                DatasetCompletedEvent(metrics={"accuracy": 0.75}),
                [
                    ("type", "dataset_completed"),
                    ("status", "running"),
                    ("metrics", {"accuracy": 0.75}),
                ],
            ),
            (
                TrainingErrorEvent(error="failed", traceback="trace"),
                [
                    ("type", "error"),
                    ("status", "failed"),
                    ("error", "failed"),
                    ("traceback", "trace"),
                ],
            ),
            (
                EpochStartedEvent(epoch=1, step=2),
                [
                    ("type", "epoch_started"),
                    ("status", "running"),
                    ("epoch", 1),
                    ("step", 2),
                ],
            ),
            (
                StepEvent(epoch=1, step=2, batch=3, metrics={"loss": 0.5}),
                [
                    ("type", "step"),
                    ("status", "running"),
                    ("epoch", 1),
                    ("step", 2),
                    ("batch", 3),
                    ("metrics", {"loss": 0.5}),
                ],
            ),
            (
                ValidationEvent(epoch=1, step=2, metrics={"accuracy": 0.75}),
                [
                    ("type", "validation"),
                    ("status", "running"),
                    ("epoch", 1),
                    ("step", 2),
                    ("metrics", {"accuracy": 0.75}),
                ],
            ),
            (
                FitCompletedEvent(epoch=1, step=2, metrics={"loss": 0.25}),
                [
                    ("type", "fit_completed"),
                    ("status", "running"),
                    ("epoch", 1),
                    ("step", 2),
                    ("metrics", {"loss": 0.25}),
                ],
            ),
            (
                CompletedTestEvent(epoch=1, step=2, metrics={"loss": 0.2}),
                [
                    ("type", "test_completed"),
                    ("status", "running"),
                    ("epoch", 1),
                    ("step", 2),
                    ("metrics", {"loss": 0.2}),
                ],
            ),
            (
                ClusterInitializedEvent(
                    node="main.cluster",
                    count=2,
                    capacity=[2, 2, 2],
                    coordinates=[[0, 0, 0], [0, 0, 1]],
                    coordinate_count=2,
                    coordinates_truncated=False,
                ),
                [
                    ("type", "cluster_initialized"),
                    ("node", "main.cluster"),
                    ("count", 2),
                    ("capacity", [2, 2, 2]),
                    ("coordinates", [[0, 0, 0], [0, 0, 1]]),
                    ("coordinateCount", 2),
                    ("coordinatesTruncated", False),
                ],
            ),
            (
                NeuronAddedEvent(
                    coord=[0, 0, 1],
                    node="main.cluster",
                    count=2,
                    capacity=[2, 2, 2],
                    epoch=1,
                    step=2,
                ),
                [
                    ("type", "neuron_added"),
                    ("coord", [0, 0, 1]),
                    ("node", "main.cluster"),
                    ("count", 2),
                    ("capacity", [2, 2, 2]),
                    ("epoch", 1),
                    ("step", 2),
                ],
            ),
            (
                NeuronsAddedEvent(
                    coordinates=[[0, 0, 1], [0, 0, 2]],
                    coordinate_count=2,
                    coordinates_truncated=False,
                    node="main.cluster",
                    count=3,
                    capacity=[2, 2, 2],
                    epoch=1,
                    step=2,
                ),
                [
                    ("type", "neurons_added"),
                    ("coordinates", [[0, 0, 1], [0, 0, 2]]),
                    ("coordinateCount", 2),
                    ("coordinatesTruncated", False),
                    ("node", "main.cluster"),
                    ("count", 3),
                    ("capacity", [2, 2, 2]),
                    ("epoch", 1),
                    ("step", 2),
                ],
            ),
        )

        for event, expected_items in cases:
            with self.subTest(event=type(event).__name__):
                self.assertEqual(
                    list(project_run_progress_event(event).items()),
                    expected_items,
                )

    def test_context_projection_overwrites_values_without_moving_existing_keys(
        self,
    ) -> None:
        payload = project_run_progress_event(
            {
                "type": "legacy",
                "dataset": "untrusted",
                "status": "running",
            },
            context=_context(),
        )

        self.assertEqual(payload["dataset"], "Mnist")
        self.assertEqual(
            list(payload),
            [
                "type",
                "dataset",
                "status",
                "experimentTask",
                "preset",
                "presetKey",
                "logDir",
                "runId",
                "runIndex",
                "runTotal",
                "totalEpochs",
            ],
        )

    def test_typed_event_jsonl_bytes_remain_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            progress = ContextualRunProgress(JsonlRunProgress(path), _context())
            with patch("model_runtime.runs.progress.datetime") as clock:
                clock.now.return_value.isoformat.return_value = (
                    "2026-08-12T10:11:12+00:00"
                )
                progress.write_event(DatasetStartedEvent(params={"seed": 17}))

            payload = path.read_text(encoding="utf-8")

        self.assertEqual(
            payload,
            '{"timestamp": "2026-08-12T10:11:12+00:00", '
            '"type": "dataset_started", "status": "running", '
            '"params": {"seed": 17}, '
            '"experimentTask": "image-classification", '
            '"dataset": "Mnist", "preset": "baseline", '
            '"presetKey": "BASELINE", '
            '"logDir": "logs/run/version_0", "runId": "run-0001", '
            '"runIndex": 1, "runTotal": 1, "totalEpochs": 30}\n',
        )

    def test_jsonl_writer_rejects_non_finite_metrics_before_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            progress = JsonlRunProgress(path)

            with self.assertRaises(ValueError):
                progress.write_event({"type": "step", "metrics": {"loss": math.inf}})

            self.assertFalse(path.exists())

    def test_growth_callback_reports_a_neuron_regrown_after_pruning(self) -> None:
        events: list[dict[str, object]] = []
        cluster = SimpleNamespace(
            cluster={"neuron_1_1_1": object(), "neuron_2_1_1": object()},
            x_axis_total_neurons=2,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
        )
        progress = type(
            "Progress",
            (),
            {"write_event": lambda _self, event: events.append(dict(event))},
        )()
        callback = lightning_progress_adapter(
            ContextualRunProgress(progress, _context()),
            step_interval=10,
        )
        callback._clusters = [("cluster", cluster)]
        callback._known_names = {"cluster": set(cluster.cluster)}
        trainer = SimpleNamespace(current_epoch=0, global_step=1)

        del cluster.cluster["neuron_2_1_1"]
        callback.on_train_batch_end(trainer, None, None, None, 0)
        cluster.cluster["neuron_2_1_1"] = object()
        callback.on_train_batch_end(trainer, None, None, None, 1)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "neuron_added")
        self.assertEqual(events[0]["coord"], [2, 1, 1])
        self.assertEqual(events[0]["runId"], "run-0001")
        self.assertEqual(events[0]["experimentTask"], "image-classification")

    def test_metric_sanitization_and_jsonl_shape_remain_portable(self) -> None:
        sanitized, original_count, dropped_count = sanitize_metric_payload(
            {
                "validation/accuracy": 0.75,
                "validation/confusion_matrix": [[1, 0], [0, 1]],
            },
            metric_key_limit=512,
            string_value_limit=20_000,
        )
        self.assertEqual(sanitized, {"validation/accuracy": 0.75})
        self.assertEqual(original_count, 2)
        self.assertEqual(dropped_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.jsonl"
            progress = ContextualRunProgress(
                JsonlRunProgress(path),
                _context(),
            )
            progress.write_event(
                {
                    "type": "dataset_started",
                    "status": "running",
                }
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["dataset"], "Mnist")
        self.assertEqual(payload["preset"], "baseline")
        self.assertEqual(payload["presetKey"], "BASELINE")
        self.assertEqual(payload["runId"], "run-0001")
        self.assertEqual(payload["runIndex"], 1)
        self.assertEqual(payload["runTotal"], 1)
        self.assertEqual(payload["totalEpochs"], 30)
        self.assertEqual(payload["type"], "dataset_started")
        self.assertEqual(payload["status"], "running")


if __name__ == "__main__":
    unittest.main()
