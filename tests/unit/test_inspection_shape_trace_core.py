from __future__ import annotations

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from emperor.experiments import ExperimentTask
from model_runtime.inspection import (
    InspectionCaptureLimits,
    InspectionError,
    InspectionRequest,
    shape_trace,
)
from model_runtime.inspection._shape_runtime import ShapeTraceRuntime
from model_runtime.inspection.capture_limits import InspectionCapture
from model_runtime.inspection.materialization import (
    MaterializedConfiguration,
    MaterializedInspection,
)
from model_runtime.inspection.records import ParsedOverrides
from model_runtime.packages import ModelIdentity, ModelPackage
from model_runtime.task_behavior import SyntheticInputError


class _ImageDataset:
    num_channels = 3
    default_width = 4
    default_height = 5


class _TextDataset:
    pass


class _SamplePackage:
    def __init__(self, task: ExperimentTask, dataset: type) -> None:
        self.task = task
        self.dataset = dataset

    def resolve_experiment_task(self, _requested_task: str | None) -> ExperimentTask:
        return self.task

    def resolve_dataset(
        self,
        _requested_dataset: str | None,
        _task: ExperimentTask,
    ) -> type:
        return self.dataset

    @staticmethod
    def task_name(task: ExperimentTask) -> str:
        return task.name.lower()


class _FixturePackage(ModelPackage):
    def preset_name(self, preset: object) -> str:
        return str(preset)


class _UnusedPackageAdapter:
    pass


def _trace_fixture_model():
    fixture_module = ModuleType("acme_networks.shape_trace_fixture")
    source = """
from torch import nn


class TraceBlock(nn.Module):
    def forward(self, value):
        return value + 1


class TraceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = TraceBlock()

    def forward(self, value):
        value = value.reshape(1, 4)
        value = self.block(value)
        value = self.block(value)
        logits = value.sum(dim=-1)
        return logits
"""
    exec(
        compile(source, "/virtual/acme_networks/shape_trace_fixture.py", "exec"),
        fixture_module.__dict__,
    )
    return fixture_module.TraceModel()


def _shared_trace_fixture_model():
    fixture_module = ModuleType("acme_networks.shared_shape_trace_fixture")
    source = """
from torch import nn


class SharedBlock(nn.Module):
    def forward(self, value):
        return value + 1


class Branch(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(self, value):
        return self.shared(value)


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        shared = SharedBlock()
        self.left = Branch(shared)
        self.right = Branch(shared)

    def forward(self, value):
        return self.left(value) + self.right(value)
"""
    exec(
        compile(
            source,
            "/virtual/acme_networks/shared_shape_trace_fixture.py",
            "exec",
        ),
        fixture_module.__dict__,
    )
    return fixture_module.SharedModel()


class InspectionShapeTraceCoreTests(unittest.TestCase):
    def test_unknown_detail_precedes_package_validation(self) -> None:
        with self.assertRaises(ValueError) as raised:
            shape_trace.inspect_model_shapes(
                object(),  # type: ignore[arg-type]
                InspectionRequest(preset="baseline"),
                detail="unknown",  # type: ignore[arg-type]
            )

        self.assertEqual(
            str(raised.exception),
            "Unknown shape-trace detail: 'unknown'",
        )

    def test_tensor_shapes_are_materialized_before_later_capture_failure(
        self,
    ) -> None:
        converted_names: list[str] = []
        original_tensor_shape = shape_trace._tensor_shape

        def record_tensor_shape(name: str, tensor: torch.Tensor):
            converted_names.append(name)
            return original_tensor_shape(name, tensor)

        with (
            patch.object(
                shape_trace,
                "_tensor_shape",
                side_effect=record_tensor_shape,
            ),
            self.assertRaisesRegex(
                InspectionError,
                "tensor observation limit of 2 exceeded",
            ),
        ):
            shape_trace._tensor_observations(
                [torch.zeros(1), torch.ones(1)],
                "value",
                capture=InspectionCapture(
                    InspectionCaptureLimits(maximum_tensor_observations=2)
                ),
            )

        self.assertEqual(converted_names, ["value[0]"])

    def test_capture_limits_abort_trace_and_restore_all_runtime_state(self) -> None:
        cases = (
            (
                "module call",
                InspectionCaptureLimits(maximum_module_calls=1),
                "outputs",
                "module call limit of 1 exceeded",
            ),
            (
                "method",
                InspectionCaptureLimits(maximum_methods=1),
                "variables",
                "method limit of 1 exceeded",
            ),
            (
                "variable event",
                InspectionCaptureLimits(maximum_variable_events=1),
                "variables",
                "variable event limit of 1 exceeded",
            ),
            (
                "tensor observation",
                InspectionCaptureLimits(maximum_tensor_observations=1),
                "outputs",
                "tensor observation limit of 1 exceeded",
            ),
            (
                "trace event",
                InspectionCaptureLimits(maximum_trace_events=1),
                "variables",
                "trace event limit of 1 exceeded",
            ),
            (
                "output byte",
                InspectionCaptureLimits(maximum_output_bytes=1_000),
                "variables",
                "output byte limit of 1000 exceeded",
            ),
        )
        for label, limits, detail, message in cases:
            with self.subTest(limit=label):
                package = _FixturePackage(
                    ModelIdentity("fixtures", "shape_trace"),
                    _UnusedPackageAdapter(),  # type: ignore[arg-type]
                )
                model = _trace_fixture_model()
                model.train()
                sample_input = torch.zeros((1, 4))
                request = InspectionRequest(
                    preset="baseline",
                    capture_limits=limits,
                )
                prepared = MaterializedConfiguration(
                    package=package,
                    request=request,
                    preset="baseline",
                    experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
                    dataset=_ImageDataset,
                    overrides=ParsedOverrides(),
                    configuration=SimpleNamespace(),
                )
                previous_trace = sys.gettrace()
                random_state = torch.random.get_rng_state().clone()

                with (
                    patch.object(
                        shape_trace,
                        "materialize_inspection",
                        return_value=MaterializedInspection(
                            prepared=prepared,
                            model=model,
                        ),
                    ),
                    patch.object(
                        shape_trace,
                        "_sample_inputs",
                        return_value=(
                            "SyntheticDataset",
                            "image-classification",
                            (sample_input,),
                        ),
                    ),
                    self.assertRaisesRegex(
                        InspectionError,
                        message,
                    ),
                ):
                    shape_trace.inspect_model_shapes(
                        package,
                        request,
                        detail=detail,
                    )

                self.assertIs(sys.gettrace(), previous_trace)
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(torch.random.get_rng_state(), random_state))
                for module in model.modules():
                    self.assertEqual(module._forward_pre_hooks, {})
                    self.assertEqual(module._forward_hooks, {})

    def test_output_trace_skips_variable_distribution_discovery(self) -> None:
        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = _trace_fixture_model()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (sample_input,),
                ),
            ),
            patch.object(
                shape_trace,
                "_trace_module_names",
                side_effect=RuntimeError("variable discovery must not run"),
            ),
        ):
            _, trace = shape_trace.inspect_model_shapes(
                package,
                request,
                detail="outputs",
            )

        self.assertGreater(len(trace.modules), 0)

    def test_trace_setup_failure_removes_registered_hooks(self) -> None:
        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = _trace_fixture_model()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (sample_input,),
                ),
            ),
            patch.object(
                shape_trace,
                "_trace_module_names",
                side_effect=RuntimeError("trace-name failure"),
            ),
            self.assertRaisesRegex(InspectionError, "trace-name failure"),
        ):
            shape_trace.inspect_model_shapes(
                package,
                request,
                detail="variables",
            )

        for module in model.modules():
            self.assertEqual(module._forward_pre_hooks, {})
            self.assertEqual(module._forward_hooks, {})

    def test_model_failure_is_wrapped_after_all_runtime_state_is_restored(self) -> None:
        model_failure = RuntimeError("fixture forward failure")

        class FailingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Identity()

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                self.block(value)
                raise model_failure

        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = FailingModel()
        model.train()
        model.block.eval()
        training_flags = {id(module): module.training for module in model.modules()}
        previous_trace = sys.gettrace()
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (torch.zeros((1, 4)),),
                ),
            ),
            self.assertRaises(InspectionError) as raised,
        ):
            shape_trace.inspect_model_shapes(package, request, detail="outputs")

        self.assertRegex(
            str(raised.exception),
            "Failed to execute shape trace.*fixture forward failure",
        )
        self.assertIs(raised.exception.__cause__, model_failure)
        self.assertIs(sys.gettrace(), previous_trace)
        self.assertEqual(
            {id(module): module.training for module in model.modules()},
            training_flags,
        )
        for module in model.modules():
            self.assertEqual(module._forward_pre_hooks, {})
            self.assertEqual(module._forward_hooks, {})

    def test_keyboard_interrupt_restores_runtime_and_cpu_rng_state(self) -> None:
        interruption = KeyboardInterrupt("fixture cancellation")

        class InterruptingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Identity()

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                self.block(value)
                torch.rand(1)
                raise interruption

        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = InterruptingModel()
        model.train()
        model.block.eval()
        training_flags = {id(module): module.training for module in model.modules()}
        previous_trace = sys.gettrace()
        random_state = torch.random.get_rng_state().clone()
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (torch.zeros((1, 4)),),
                ),
            ),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            shape_trace.inspect_model_shapes(package, request, detail="variables")

        self.assertIs(raised.exception, interruption)
        self.assertIs(sys.gettrace(), previous_trace)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), random_state))
        self.assertEqual(
            {id(module): module.training for module in model.modules()},
            training_flags,
        )
        for module in model.modules():
            self.assertEqual(module._forward_pre_hooks, {})
            self.assertEqual(module._forward_hooks, {})

    def test_runtime_restoration_failure_is_not_translated(self) -> None:
        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = _trace_fixture_model()
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )
        original_restore = ShapeTraceRuntime.restore

        def restore_then_fail(runtime: ShapeTraceRuntime) -> None:
            original_restore(runtime)
            raise RuntimeError("fixture runtime restoration failure")

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (torch.zeros((1, 4)),),
                ),
            ),
            patch.object(ShapeTraceRuntime, "restore", restore_then_fail),
            self.assertRaisesRegex(
                RuntimeError,
                "fixture runtime restoration failure",
            ),
        ):
            shape_trace.inspect_model_shapes(package, request, detail="outputs")

        for module in model.modules():
            self.assertEqual(module._forward_pre_hooks, {})
            self.assertEqual(module._forward_hooks, {})

    def test_successful_trace_restores_each_module_training_flag(self) -> None:
        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = _trace_fixture_model()
        model.train()
        model.block.eval()
        training_flags = {id(module): module.training for module in model.modules()}
        random_state = torch.random.get_rng_state().clone()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (sample_input,),
                ),
            ),
        ):
            shape_trace.inspect_model_shapes(
                package,
                request,
                detail="outputs",
            )

        self.assertEqual(
            {id(module): module.training for module in model.modules()},
            training_flags,
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), random_state))

    def test_successful_trace_restores_preexisting_python_trace(self) -> None:
        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = _trace_fixture_model()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        def existing_trace(frame, event, argument):
            del frame, event, argument
            return existing_trace

        previous_trace = sys.gettrace()
        try:
            sys.settrace(existing_trace)
            with (
                patch.object(
                    shape_trace,
                    "materialize_inspection",
                    return_value=MaterializedInspection(
                        prepared=prepared,
                        model=model,
                    ),
                ),
                patch.object(
                    shape_trace,
                    "_sample_inputs",
                    return_value=(
                        "SyntheticDataset",
                        "image-classification",
                        (sample_input,),
                    ),
                ),
            ):
                shape_trace.inspect_model_shapes(
                    package,
                    request,
                    detail="outputs",
                )

            self.assertIs(sys.gettrace(), existing_trace)
        finally:
            sys.settrace(previous_trace)

    def test_shared_module_calls_belong_to_one_canonical_graph_node(self) -> None:
        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = _shared_trace_fixture_model()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (sample_input,),
                ),
            ),
        ):
            _, trace = shape_trace.inspect_model_shapes(
                package,
                request,
                detail="outputs",
            )

        shared_traces = [
            module_trace
            for module_trace in trace.modules
            if module_trace.node_id.endswith(".shared")
        ]
        self.assertEqual(len(shared_traces), 1)
        self.assertEqual(len(shared_traces[0].calls), 2)

    @unittest.skipIf(
        os.environ.get("MODEL_RUNTIME_COVERAGE") == "1",
        "requires an interpreter without an active coverage trace",
    )
    def test_outputs_only_trace_does_not_change_sys_trace_observable_behavior(
        self,
    ) -> None:
        class TraceSensitive(nn.Module):
            def forward(self, value):
                if sys.gettrace() is None:
                    return value
                return value.unsqueeze(0)

        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = TraceSensitive()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (sample_input,),
                ),
            ),
        ):
            _, trace = shape_trace.inspect_model_shapes(
                package,
                request,
                detail="outputs",
            )

        root_trace = next(
            module_trace
            for module_trace in trace.modules
            if module_trace.node_id == "__root__"
        )
        self.assertEqual(root_trace.calls[0].outputs[0].shape, (1, 4))

    def test_zero_tensor_call_structure_obeys_whole_trace_output_limit(self) -> None:
        class Null(nn.Module):
            def forward(self):
                return None

        class RepeatedCalls(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.null = Null()

            def forward(self):
                for _ in range(500):
                    self.null()
                return None

        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = RepeatedCalls()
        request = InspectionRequest(
            preset="baseline",
            capture_limits=InspectionCaptureLimits(
                maximum_module_calls=5_000,
                maximum_output_bytes=10_000,
            ),
        )
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(
                    prepared=prepared,
                    model=model,
                ),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (),
                ),
            ),
            self.assertRaisesRegex(
                InspectionError,
                "output byte limit of 10000 exceeded",
            ),
        ):
            shape_trace.inspect_model_shapes(
                package,
                request,
                detail="outputs",
            )

    def test_synthetic_inputs_cover_every_supported_task_family(self) -> None:
        token_config = SimpleNamespace(sequence_length=7, input_dim=11)
        translation_config = SimpleNamespace(
            experiment_config=SimpleNamespace(
                source_sequence_length=8,
                target_sequence_length=6,
                vocab_size=13,
                bos_token_id=2,
            )
        )
        cases = (
            (
                ExperimentTask.IMAGE_CLASSIFICATION,
                _ImageDataset,
                SimpleNamespace(),
                ((1, 3, 5, 4),),
            ),
            (
                ExperimentTask.BERT_PRETRAINING,
                _TextDataset,
                token_config,
                ((1, 7),),
            ),
            (
                ExperimentTask.CAUSAL_LANGUAGE_MODELING,
                _TextDataset,
                token_config,
                ((1, 7),),
            ),
            (
                ExperimentTask.TEXT_TRANSLATION,
                _TextDataset,
                translation_config,
                ((1, 8), (1, 5)),
            ),
        )

        for task, dataset, configuration, expected_shapes in cases:
            with self.subTest(task=task):
                _dataset_name, task_name, inputs = shape_trace._sample_inputs(
                    SimpleNamespace(
                        package=_SamplePackage(task, dataset),
                        experiment_task=task,
                        dataset=dataset,
                        configuration=configuration,
                    )
                )

                self.assertEqual(task_name, task.name.lower())
                self.assertEqual(
                    tuple(tuple(tensor.shape) for tensor in inputs),
                    expected_shapes,
                )

    def test_synthetic_input_failure_keeps_the_domain_cause(self) -> None:
        synthetic_failure = SyntheticInputError("fixture input failure")

        def fail_synthetic_inputs(_dataset: type, _configuration: object) -> None:
            raise synthetic_failure

        materialized = SimpleNamespace(
            package=_SamplePackage(
                ExperimentTask.IMAGE_CLASSIFICATION,
                _ImageDataset,
            ),
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            configuration=SimpleNamespace(),
        )
        behavior = SimpleNamespace(synthetic_inputs=fail_synthetic_inputs)

        with (
            patch.object(
                shape_trace,
                "experiment_task_behavior",
                return_value=behavior,
            ),
            self.assertRaises(InspectionError) as raised,
        ):
            shape_trace._sample_inputs(materialized)  # type: ignore[arg-type]

        self.assertEqual(str(raised.exception), "fixture input failure")
        self.assertIs(raised.exception.__cause__, synthetic_failure)

    def test_shape_trace_records_repeated_calls_and_same_shape_variables(self) -> None:
        package = _FixturePackage(
            ModelIdentity("fixtures", "shape_trace"),
            _UnusedPackageAdapter(),  # type: ignore[arg-type]
        )
        model = _trace_fixture_model()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")
        prepared = MaterializedConfiguration(
            package=package,
            request=request,
            preset="baseline",
            experiment_task=ExperimentTask.IMAGE_CLASSIFICATION,
            dataset=_ImageDataset,
            overrides=ParsedOverrides(),
            configuration=SimpleNamespace(),
        )

        with (
            patch.object(
                shape_trace,
                "materialize_inspection",
                return_value=MaterializedInspection(prepared=prepared, model=model),
            ),
            patch.object(
                shape_trace,
                "_sample_inputs",
                return_value=(
                    "SyntheticDataset",
                    "image-classification",
                    (sample_input,),
                ),
            ),
        ):
            result, trace = shape_trace.inspect_model_shapes(
                package,
                request,
                detail="variables",
            )

        self.assertEqual(result.identity, package.identity)
        self.assertEqual(trace.sample_inputs[0].shape, (1, 4))
        module_calls = {module.node_id: module.calls for module in trace.modules}
        self.assertEqual(len(module_calls["__root__"]), 1)
        self.assertEqual(len(module_calls["block"]), 2)

        model_forward = next(
            method
            for method in trace.methods
            if method.qualified_name == "TraceModel.forward"
        )
        self.assertIsNone(model_forward.parent_id)
        self.assertEqual(model_forward.module_path, "model")
        self.assertEqual(
            model_forward.source_path,
            "acme_networks/shape_trace_fixture.py",
        )
        same_shape_assignments = [
            tensor
            for variable in model_forward.variables
            for tensor in variable.tensors
            if tensor.name == "value" and tensor.shape == (1, 4)
        ]
        self.assertGreaterEqual(len(same_shape_assignments), 3)
        self.assertTrue(
            all(variable.line is not None for variable in model_forward.variables)
        )
        self.assertEqual(
            sum(
                method.qualified_name == "TraceBlock.forward"
                for method in trace.methods
            ),
            2,
        )
        self.assertTrue(model.training)
        for module in model.modules():
            self.assertEqual(module._forward_pre_hooks, {})
            self.assertEqual(module._forward_hooks, {})


if __name__ == "__main__":
    unittest.main()
