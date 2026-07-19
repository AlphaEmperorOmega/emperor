from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from emperor.experiments import ExperimentTask
from model_runtime.inspection import InspectionRequest, shape_trace
from model_runtime.packages import ModelPackage


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


def _trace_fixture_model():
    fixture_module = ModuleType("models.shape_trace_fixture")
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
        compile(source, "/virtual/models/shape_trace_fixture.py", "exec"),
        fixture_module.__dict__,
    )
    return fixture_module.TraceModel()


class InspectionShapeTraceCoreTests(unittest.TestCase):
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
                    _SamplePackage(task, dataset),
                    InspectionRequest(preset="baseline"),
                    configuration,
                )

                self.assertEqual(task_name, task.name.lower())
                self.assertEqual(
                    tuple(tuple(tensor.shape) for tensor in inputs),
                    expected_shapes,
                )

    def test_shape_trace_records_repeated_calls_and_same_shape_variables(self) -> None:
        package = _FixturePackage("fixtures", "shape_trace", "unused")
        model = _trace_fixture_model()
        sample_input = torch.zeros((1, 4))
        request = InspectionRequest(preset="baseline")

        with (
            patch.object(
                shape_trace,
                "_instantiate_inspection_model",
                return_value=("baseline", SimpleNamespace(), model),
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
            "models/shape_trace_fixture.py",
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


if __name__ == "__main__":
    unittest.main()
