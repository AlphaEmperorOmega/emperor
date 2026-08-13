from __future__ import annotations

import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from lightning.pytorch.callbacks import Callback

from model_runtime.runs import (
    RunParameter,
    RunRequest,
    RunResult,
)
from model_runtime.runs._handoff import (
    TrainingExecutionRequest,
    TrainingRun,
    TrainingRunRequest,
)
from model_runtime.runs._progress_events import (
    ClusterInitializedEvent,
    DatasetCompletedEvent,
    DatasetStartedEvent,
    FitCompletedEvent,
    NeuronAddedEvent,
    NeuronsAddedEvent,
    StepEvent,
    TestCompletedEvent,
    ValidationEvent,
    project_run_progress_event,
)


def _nested_mapping(seed: int) -> dict[str, Any]:
    return {
        "nested": {
            "values": [seed],
            "tags": {seed},
        }
    }


def _mutate_nested_mapping(value: dict[str, Any]) -> None:
    nested = cast(dict[str, Any], value["nested"])
    cast(list[int], nested["values"]).append(99)
    cast(set[int], nested["tags"]).add(99)
    value["added"] = []


def _assert_fresh_containers(
    case: unittest.TestCase,
    first: object,
    second: object,
) -> None:
    if isinstance(first, dict):
        case.assertIsInstance(second, dict)
        case.assertIsNot(first, second)
        second_mapping = cast(dict[object, object], second)
        case.assertEqual(tuple(first), tuple(second_mapping))
        for key, value in first.items():
            _assert_fresh_containers(case, value, second_mapping[key])
        return
    if isinstance(first, list):
        case.assertIsInstance(second, list)
        case.assertIsNot(first, second)
        second_list = cast(list[object], second)
        case.assertEqual(len(first), len(second_list))
        for left, right in zip(first, second_list, strict=True):
            _assert_fresh_containers(case, left, right)
        return
    if isinstance(first, set):
        case.assertIsInstance(second, set)
        case.assertIsNot(first, second)
        case.assertEqual(first, second)
        return
    case.assertEqual(first, second)


class RunsImmutableRecordTests(unittest.TestCase):
    def test_every_mapping_event_defensively_freezes_and_deeply_projects(self) -> None:
        cases: tuple[
            tuple[
                str,
                Callable[[Mapping[str, dict[str, Any]]], object],
                tuple[tuple[str, str], ...],
            ],
            ...,
        ] = (
            (
                "dataset_started",
                lambda values: DatasetStartedEvent(
                    params=values["params"],
                    resumed_from=values["resumed_from"],
                ),
                (("params", "params"), ("resumed_from", "resumedFrom")),
            ),
            (
                "dataset_completed",
                lambda values: DatasetCompletedEvent(
                    metrics=values["metrics"],
                    resumed_from=values["resumed_from"],
                ),
                (("metrics", "metrics"), ("resumed_from", "resumedFrom")),
            ),
            (
                "step",
                lambda values: StepEvent(
                    epoch=1,
                    step=2,
                    batch=3,
                    metrics=values["metrics"],
                ),
                (("metrics", "metrics"),),
            ),
            (
                "validation",
                lambda values: ValidationEvent(
                    epoch=1,
                    step=2,
                    metrics=values["metrics"],
                ),
                (("metrics", "metrics"),),
            ),
            (
                "fit_completed",
                lambda values: FitCompletedEvent(
                    epoch=1,
                    step=2,
                    metrics=values["metrics"],
                ),
                (("metrics", "metrics"),),
            ),
            (
                "test_completed",
                lambda values: TestCompletedEvent(
                    epoch=1,
                    step=2,
                    metrics=values["metrics"],
                ),
                (("metrics", "metrics"),),
            ),
        )

        for index, (name, build_event, fields) in enumerate(cases, start=1):
            with self.subTest(event=name):
                sources = {
                    attribute: _nested_mapping(index + field_index)
                    for field_index, (attribute, _wire_name) in enumerate(fields)
                }
                event = build_event(sources)
                expected = project_run_progress_event(cast(Any, event))

                for source in sources.values():
                    _mutate_nested_mapping(source)

                first = project_run_progress_event(cast(Any, event))
                second = project_run_progress_event(cast(Any, event))
                self.assertEqual(first, expected)
                self.assertEqual(second, expected)
                for attribute, wire_name in fields:
                    stored = cast(Mapping[str, Any], getattr(event, attribute))
                    with self.assertRaises(TypeError):
                        stored["forbidden"] = True  # type: ignore[index]
                    _assert_fresh_containers(
                        self,
                        first[wire_name],
                        second[wire_name],
                    )

    def test_every_growth_event_defensively_freezes_and_deeply_projects(self) -> None:
        def cluster_event(values: dict[str, Any]) -> ClusterInitializedEvent:
            return ClusterInitializedEvent(
                node="main.cluster",
                count=2,
                capacity=values["capacity"],
                coordinates=values["coordinates"],
                coordinate_count=2,
                coordinates_truncated=False,
            )

        def neuron_event(values: dict[str, Any]) -> NeuronAddedEvent:
            return NeuronAddedEvent(
                coord=values["coord"],
                node="main.cluster",
                count=2,
                capacity=values["capacity"],
                epoch=1,
                step=2,
            )

        def neurons_event(values: dict[str, Any]) -> NeuronsAddedEvent:
            return NeuronsAddedEvent(
                coordinates=values["coordinates"],
                coordinate_count=2,
                coordinates_truncated=False,
                node="main.cluster",
                count=3,
                capacity=values["capacity"],
                epoch=1,
                step=2,
            )

        cases = (
            (
                "cluster_initialized",
                cluster_event,
                ("capacity", "coordinates"),
            ),
            ("neuron_added", neuron_event, ("coord", "capacity")),
            ("neurons_added", neurons_event, ("coordinates", "capacity")),
        )

        for name, build_event, wire_fields in cases:
            with self.subTest(event=name):
                values = {
                    "capacity": [2, 2, 2],
                    "coord": [0, 0, 1],
                    "coordinates": [[0, 0, 1], [0, 0, 2]],
                }
                event = build_event(values)
                expected = project_run_progress_event(event)

                cast(list[int], values["capacity"]).append(9)
                cast(list[int], values["coord"]).append(9)
                coordinates = cast(list[list[int]], values["coordinates"])
                coordinates[0].append(9)
                coordinates.append([9, 9, 9])

                first = project_run_progress_event(event)
                second = project_run_progress_event(event)
                self.assertEqual(first, expected)
                self.assertEqual(second, expected)
                for field in wire_fields:
                    self.assertIsInstance(getattr(event, field), tuple)
                    _assert_fresh_containers(self, first[field], second[field])

    def test_handoff_requests_freeze_membership_but_keep_runtime_objects(self) -> None:
        parameters = _nested_mapping(1)
        config_overrides = _nested_mapping(2)
        request = TrainingRunRequest(
            run_id="run-1",
            run_index=1,
            run_total=1,
            preset=object(),
            dataset_type=object,
            parameters=parameters,
            config_overrides=config_overrides,
        )
        _mutate_nested_mapping(parameters)
        _mutate_nested_mapping(config_overrides)

        self.assertEqual(request.parameters["nested"]["values"], (1,))
        self.assertEqual(request.config_overrides["nested"]["values"], (2,))

        first_callback = Callback()
        second_callback = Callback()
        callbacks = [first_callback, second_callback]
        resumed_from = _nested_mapping(3)
        execution = TrainingExecutionRequest(
            training_run=cast(Any, object()),
            callbacks=callbacks,
            progress=None,
            progress_step_interval=1,
            ckpt_path=Path("resume.ckpt"),
            model_validator=None,
            resumed_from=resumed_from,
        )
        callbacks.reverse()
        callbacks.append(Callback())
        _mutate_nested_mapping(resumed_from)

        self.assertIsInstance(execution.callbacks, tuple)
        self.assertEqual(len(execution.callbacks), 2)
        self.assertIs(execution.callbacks[0], first_callback)
        self.assertIs(execution.callbacks[1], second_callback)
        assert execution.resumed_from is not None
        self.assertEqual(execution.resumed_from["nested"]["values"], (3,))

        training_run = TrainingRun(
            experiment_task=None,
            preset=cast(Any, object()),
            dataset_type=object,
            config=cast(Any, object()),
            config_overrides={},
            num_epochs=1,
        )
        training_run.num_epochs = 2
        self.assertEqual(training_run.num_epochs, 2)

    def test_public_run_records_share_recursive_freezing_policy(self) -> None:
        colliding_values: dict[object, Any] = {
            1: ["first"],
            "1": ["second"],
            "tail": {3},
        }
        parameter = RunParameter("VALUE", colliding_values, "override")
        cast(list[str], colliding_values["1"]).append("mutated")
        cast(set[int], colliding_values["tail"]).add(4)

        frozen_value = cast(Mapping[str, Any], parameter.value)
        self.assertEqual(tuple(frozen_value), ("1", "tail"))
        self.assertEqual(frozen_value["1"], ("second",))
        self.assertEqual(frozen_value["tail"], frozenset({3}))

        overrides = _nested_mapping(4)
        payload = _nested_mapping(5)
        request = RunRequest(
            presets=("baseline",),
            datasets=("Mnist",),
            overrides=overrides,
        )
        result = RunResult(
            run_id="run-1",
            experiment_task="image-classification",
            preset="baseline",
            dataset="Mnist",
            log_dir="logs/run-1",
            payload=payload,
        )
        _mutate_nested_mapping(overrides)
        _mutate_nested_mapping(payload)

        self.assertEqual(request.overrides["nested"]["values"], (4,))
        self.assertEqual(result.payload["nested"]["values"], (5,))

if __name__ == "__main__":
    unittest.main()
