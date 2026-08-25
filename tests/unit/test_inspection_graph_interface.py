from __future__ import annotations

import os
import subprocess
import sys
import unittest
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError, dataclass, field, replace
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch import nn

from emperor.config import ModelConfig
from emperor.halting import HaltingHiddenStateModeOptions, SoftHaltingConfig
from emperor.layers import (
    ActivationOptions,
    HierarchicalReasoningModelRecurrentConfig,
    LayerConfig,
    LayerNormPositionOptions,
    LayerState,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
    WeightedBlendResidualConfig,
)
from emperor.linears import LinearLayerConfig
from model_runtime.inspection import (
    InspectionCaptureLimits,
    InspectionError,
    InspectionRequest,
    InspectionResult,
    inspect_model,
)
from model_runtime.inspection.capture_limits import (
    InspectionCapture,
    _estimated_output_bytes,
)
from model_runtime.inspection.model_graph import (
    graph_role,
    inspect_model_graph,
    module_details,
    parameter_count,
    parameter_size_bytes,
)
from model_runtime.inspection.preflight import preflight_inspection_configuration
from model_runtime.inspection.shape_trace import _tensor_observations
from model_runtime.packages import (
    InspectionConstructionLimits,
    InspectionFieldProductLimit,
    ModelIdentity,
    ModelPackage,
)
from models.catalog import MODEL_CATALOG, model_package


class _BrokenPackageAdapter:
    @staticmethod
    def _missing(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'models.__inspection_missing__'")

    load_metadata = _missing
    load_runtime_options_type = _missing
    bind_runtime_defaults = _missing
    load_preset_type = _missing
    load_presets = _missing
    build_configuration = _missing
    build_model = _missing
    build_experiment = _missing


def _broken_package() -> ModelPackage:
    return ModelPackage(
        ModelIdentity("broken", "missing"),
        _BrokenPackageAdapter(),
    )


class _ReadOnceDescriptor:
    def __init__(self, value: object) -> None:
        self.value = value
        self.read_count = 0

    def __get__(self, instance: object, owner: type[object]) -> object:
        if instance is None:
            return self
        self.read_count += 1
        if self.read_count > 1:
            raise AssertionError("structural capability was observed twice")
        return self.value


class InspectionGraphInterfaceTests(unittest.TestCase):
    def test_semantic_catalog_does_not_import_unselected_registered_types(self) -> None:
        script = """
import sys
from torch import nn
from model_runtime.inspection.model_graph import graph_role

semantic_module = "emperor.neuron._cluster.model"
assert semantic_module not in sys.modules
assert graph_role(nn.Identity()) == "architecture"
assert semantic_module not in sys.modules
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_same_named_user_types_do_not_inherit_emperor_graph_semantics(
        self,
    ) -> None:
        class Dropout(nn.Module):
            """User-owned module that happens to share a torch class name."""

        class ClassifierMetricsLogger(nn.Module):
            """User-owned module that happens to share an Emperor class name."""

        class Layer(nn.Module):
            """User-owned module that happens to share a layer class name."""

        class RecurrentLayer(nn.Module):
            """User-owned module that happens to share a recurrent class name."""

        for module, expected_description in (
            (
                Dropout(),
                "User-owned module that happens to share a torch class name.",
            ),
            (
                ClassifierMetricsLogger(),
                "User-owned module that happens to share an Emperor class name.",
            ),
            (
                Layer(),
                "User-owned module that happens to share a layer class name.",
            ),
            (
                RecurrentLayer(),
                "User-owned module that happens to share a recurrent class name.",
            ),
        ):
            with self.subTest(module=type(module).__name__):
                root = inspect_model_graph(module).nodes[0]
                self.assertEqual(root.graph_role, "architecture")
                self.assertEqual(root.description, expected_description)

        @dataclass
        class RecurrentLayerConfig:
            """User-owned configuration with a colliding class name."""

            residual_config: object = None

        configured_module = nn.Module()
        configured_module.cfg = RecurrentLayerConfig()
        root = inspect_model_graph(configured_module).nodes[0]

        self.assertEqual(
            root.description,
            "User-owned configuration with a colliding class name.",
        )
        self.assertIsNotNone(root.configuration)
        assert root.configuration is not None
        residual_field = root.configuration.fields[0]
        self.assertEqual(
            residual_field.description,
            "Residual connection behavior. Enabled options require "
            "input_dim == output_dim.",
        )

        class SpoofedDropout(nn.Module):
            pass

        SpoofedDropout.__name__ = "Dropout"
        SpoofedDropout.__qualname__ = "Dropout"
        SpoofedDropout.__module__ = nn.Dropout.__module__
        spoofed = inspect_model_graph(SpoofedDropout()).nodes[0]
        self.assertEqual(spoofed.graph_role, "architecture")
        self.assertIsNone(spoofed.description)

        class DropoutSubclass(nn.Dropout):
            pass

        subclass = inspect_model_graph(DropoutSubclass()).nodes[0]
        self.assertEqual(subclass.graph_role, "architecture")
        self.assertIsNone(subclass.description)

        from models.linears.linear.model import Model as RegisteredProjectModel

        class SpoofedProjectModel(nn.Module):
            pass

        SpoofedProjectModel.__name__ = "Model"
        SpoofedProjectModel.__qualname__ = "Model"
        SpoofedProjectModel.__module__ = RegisteredProjectModel.__module__
        spoofed_project_model = inspect_model_graph(SpoofedProjectModel()).nodes[0]
        self.assertIsNone(spoofed_project_model.description)

        import models.linears.linear.model as project_model_module

        class Outer:
            class Model(nn.Module):
                pass

        Outer.Model.__module__ = project_model_module.__name__
        Outer.Model.__qualname__ = "Outer.Model"
        with patch.object(project_model_module, "Outer", Outer, create=True):
            nested_project_model = inspect_model_graph(Outer.Model()).nodes[0]
        self.assertIsNone(nested_project_model.description)

    def test_exact_known_types_keep_their_graph_semantics(self) -> None:
        dropout = inspect_model_graph(nn.Dropout()).nodes[0]
        self.assertEqual(graph_role(nn.Dropout()), "internal")
        self.assertEqual(dropout.graph_role, "internal")
        self.assertEqual(
            dropout.description,
            "Regularization module that randomly zeroes activations during "
            "training and is inactive during evaluation.",
        )
        self.assertEqual(graph_role(nn.CrossEntropyLoss()), "runtime")

        configured_module = nn.Module()
        configured_module.cfg = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            block_config=LinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=False,
            ),
        )
        root = inspect_model_graph(configured_module).nodes[0]
        self.assertEqual(
            root.description,
            "Builds a recurrent block that can run for multiple steps with "
            "optional gating, normalization, halting, or memory.",
        )
        self.assertIsNotNone(root.configuration)
        assert root.configuration is not None
        residual_field = next(
            field
            for field in root.configuration.fields
            if field.key == "residual_connection_option"
        )
        self.assertEqual(
            residual_field.description,
            "Residual connection behavior between recurrent steps. Set to null "
            "to disable recurrent residuals.",
        )

    def test_capture_limit_values_must_be_positive_plain_integers(self) -> None:
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    InspectionCaptureLimits(
                        maximum_graph_nodes=invalid,  # type: ignore[arg-type]
                    )

    def test_graph_edge_and_configuration_detail_limits_abort_during_capture(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            InspectionError,
            "graph edge limit of 1 exceeded",
        ):
            inspect_model_graph(
                nn.Sequential(nn.Identity(), nn.Identity()),
                limits=InspectionCaptureLimits(maximum_graph_edges=1),
            )

        @dataclass
        class VerboseConfig:
            first: int = 1
            second: int = 2

        configured = nn.Module()
        configured.cfg = VerboseConfig()
        with self.assertRaisesRegex(
            InspectionError,
            "configuration field limit of 1 exceeded",
        ):
            inspect_model_graph(
                configured,
                limits=InspectionCaptureLimits(maximum_configuration_fields=1),
            )

    def test_residual_configuration_expansion_obeys_serialized_field_limit(
        self,
    ) -> None:
        @dataclass
        class Residual:
            model_config: object = None

        @dataclass
        class Config:
            residual_config: Residual = field(default_factory=Residual)

        configured = nn.Module()
        configured.cfg = Config()

        with self.assertRaisesRegex(
            InspectionError,
            "configuration field limit of 1 exceeded",
        ):
            inspect_model_graph(
                configured,
                limits=InspectionCaptureLimits(maximum_configuration_fields=1),
            )

    def test_graph_edge_limit_stops_relationship_iteration(self) -> None:
        relationships_seen = [0]
        shared_child = nn.Identity()

        class RelationshipFlood(nn.Module):
            def named_children(self):
                for index in range(25):
                    relationships_seen[0] += 1
                    yield str(index), shared_child

        with self.assertRaisesRegex(
            InspectionError,
            "graph edge limit of 1 exceeded",
        ):
            inspect_model_graph(
                RelationshipFlood(),
                limits=InspectionCaptureLimits(
                    maximum_graph_edges=1,
                    maximum_graph_nodes=100,
                ),
            )

        self.assertLessEqual(relationships_seen[0], 2)

    def test_parameter_registration_limit_stops_parameter_iteration(self) -> None:
        inspect_model_graph(
            nn.Linear(1, 1, bias=False),
            limits=InspectionCaptureLimits(
                maximum_parameter_registrations=1,
            ),
        )

        registrations_seen = [0]
        parameter = nn.Parameter(torch.ones(1))

        class ParameterFlood(nn.Module):
            def named_parameters(self, *args, **kwargs):
                del args, kwargs
                for index in range(25):
                    registrations_seen[0] += 1
                    yield str(index), parameter

        with self.assertRaisesRegex(
            InspectionError,
            "parameter registration limit of 1 exceeded",
        ):
            inspect_model_graph(
                ParameterFlood(),
                limits=InspectionCaptureLimits(
                    maximum_parameter_registrations=1,
                ),
            )

        self.assertLessEqual(registrations_seen[0], 2)

    def test_tensor_observation_limit_stops_tensor_free_sequence_iteration(
        self,
    ) -> None:
        class Values(Sequence[int]):
            def __init__(self) -> None:
                self.accesses = 0

            def __len__(self) -> int:
                return 10_000

            def __getitem__(self, index: int) -> int:
                self.accesses += 1
                if index >= 10_000:
                    raise IndexError(index)
                return 0

        values = Values()
        capture = InspectionCapture(
            InspectionCaptureLimits(maximum_tensor_observations=1)
        )

        with self.assertRaisesRegex(
            InspectionError,
            "tensor observation limit of 1 exceeded",
        ):
            _tensor_observations(values, "value", capture=capture)

        self.assertLessEqual(values.accesses, 1)

    def test_output_size_estimation_order_boundaries_and_cleanup(self) -> None:
        @dataclass
        class Record:
            value: int

        class MappingSequence(Mapping[str, int], Sequence[int]):
            def __getitem__(self, key: str | int) -> int:
                if key in {"value", 0}:
                    return 1
                raise KeyError(key)

            def __iter__(self):
                return iter(("value",))

            def __len__(self) -> int:
                return 1

        estimates = (
            (None, 4),
            (True, 5),
            (100, 3),
            (1.0, 32),
            ("plain ASCII", 13),
            ('quote"slash\\control\n', 25),
            ("😀", 14),
            (b"x", 14),
            ((1,), 3),
            (Record(1), 11),
            (MappingSequence(), 11),
        )
        for value, expected in estimates:
            with self.subTest(value=type(value).__name__):
                self.assertEqual(_estimated_output_bytes(value, 1_000), expected)

        default_limits = InspectionCaptureLimits()
        self.assertEqual(default_limits.maximum_graph_nodes, 8_192)
        self.assertEqual(default_limits.maximum_graph_edges, 8_192)
        self.assertEqual(default_limits.maximum_output_bytes, 16 * 1024**2)

        exact_capture = InspectionCapture(
            InspectionCaptureLimits(maximum_output_bytes=6)
        )
        exact_capture.ensure_total_output([None])
        with self.assertRaisesRegex(InspectionError, "output byte limit of 5"):
            InspectionCapture(
                InspectionCaptureLimits(maximum_output_bytes=5)
            ).ensure_total_output([None])

        cyclic: list[object] = []
        cyclic.append(cyclic)
        active: set[int] = set()
        self.assertEqual(_estimated_output_bytes(cyclic, 10, active), 11)
        self.assertEqual(active, set())

        class BrokenSequence(Sequence[int]):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> int:
                if index == 0:
                    return 1
                raise RuntimeError("fixture iteration failed")

        with self.assertRaisesRegex(RuntimeError, "fixture iteration failed"):
            _estimated_output_bytes(BrokenSequence(), 100, active)
        self.assertEqual(active, set())

    def test_tensor_observation_nesting_depth_is_bounded(self) -> None:
        nested: object = torch.ones(1)
        for _ in range(10):
            nested = [nested]

        with self.assertRaisesRegex(
            InspectionError,
            "tensor nesting depth limit of 2 exceeded",
        ):
            _tensor_observations(
                nested,
                "value",
                capture=InspectionCapture(
                    InspectionCaptureLimits(maximum_tensor_nesting_depth=2)
                ),
            )

    def test_inspection_request_applies_capture_limits_to_service_graph(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        with self.assertRaisesRegex(
            InspectionError,
            "graph node limit of 1 exceeded",
        ):
            inspect_model(
                package,
                InspectionRequest(
                    preset="baseline",
                    capture_limits=InspectionCaptureLimits(maximum_graph_nodes=1),
                ),
            )

    def test_graph_node_limit_aborts_before_building_more_node_details(self) -> None:
        model = nn.Sequential(*(nn.Linear(2, 2) for _ in range(8)))

        with self.assertRaisesRegex(
            InspectionError,
            "graph node limit of 3 exceeded",
        ):
            inspect_model_graph(
                model,
                limits=InspectionCaptureLimits(maximum_graph_nodes=3),
            )

    def test_falsey_invalid_graph_limits_are_rejected(self) -> None:
        for inspect in (inspect_model_graph, module_details):
            with (
                self.subTest(entrypoint=inspect.__name__),
                self.assertRaisesRegex(
                    TypeError,
                    "limits must be InspectionCaptureLimits",
                ),
            ):
                inspect(nn.Identity(), limits=False)

    def test_registered_module_cycles_are_rejected(self) -> None:
        self_cycle = nn.Module()
        self_cycle._modules["loop"] = self_cycle

        left = nn.Module()
        right = nn.Module()
        left._modules["right"] = right
        right._modules["left"] = left

        for model in (self_cycle, left):
            with (
                self.subTest(kind="self" if model is self_cycle else "two-node"),
                self.assertRaisesRegex(
                    InspectionError,
                    "graph contains a registered module cycle",
                ),
            ):
                inspect_model_graph(model)

    def test_dotted_registered_module_names_are_rejected(self) -> None:
        model = nn.Module()
        model._modules["invalid.child"] = nn.Identity()

        with self.assertRaisesRegex(
            InspectionError,
            "registered module names without",
        ):
            inspect_model_graph(model)

    def test_shared_modules_have_one_canonical_node_and_multiple_edges(self) -> None:
        shared = nn.Linear(2, 2)

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = shared

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.left = Parent()
                self.right = Parent()

        graph = inspect_model_graph(Root())
        shared_nodes = [node for node in graph.nodes if node.path.endswith(".shared")]

        self.assertEqual(len(shared_nodes), 1)
        shared_node = shared_nodes[0]
        self.assertEqual(
            {edge.source for edge in graph.edges if edge.target == shared_node.id},
            {"left", "right"},
        )

    def test_layer_stack_container_is_transparent_with_stable_preorder(self) -> None:
        class LayerStack(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.Identity(),
                        nn.Sequential(nn.Identity()),
                    ]
                )
                self.tail = nn.Identity()

        graph = inspect_model_graph(LayerStack())

        self.assertEqual(
            [(node.id, node.path) for node in graph.nodes],
            [
                ("__root__", "model"),
                ("layers.0", "layers.0"),
                ("layers.1", "layers.1"),
                ("layers.1.0", "layers.1.0"),
                ("tail", "tail"),
            ],
        )
        self.assertEqual(
            [(edge.source, edge.target) for edge in graph.edges],
            [
                ("__root__", "layers.0"),
                ("__root__", "layers.1"),
                ("layers.1", "layers.1.0"),
                ("__root__", "tail"),
            ],
        )

    def test_tied_parameter_is_counted_once_per_containing_subtree(self) -> None:
        shared_weight = nn.Parameter(torch.ones(2, 3))

        class Child(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = shared_weight

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.left = Child()
                self.right = Child()

        graph = inspect_model_graph(Root())
        statistics_by_path = {
            node.path: (node.parameter_count, node.parameter_size_bytes)
            for node in graph.nodes
        }

        self.assertEqual(
            statistics_by_path,
            {
                "model": (6, 24),
                "left": (6, 24),
                "right": (6, 24),
            },
        )

    def test_exported_parameter_statistics_deduplicate_and_skip_lazy_storage(
        self,
    ) -> None:
        shared_weight = nn.Parameter(torch.ones(2, 3, dtype=torch.float64))

        class Child(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = shared_weight

        model = nn.Module()
        model.left = Child()
        model.right = Child()
        model.lazy = nn.LazyLinear(4, bias=False)

        self.assertEqual(parameter_count(model), 6)
        self.assertEqual(parameter_size_bytes(model), 48)

    def test_colliding_path_edge_names_remain_unique_on_wire(self) -> None:
        from model_runtime.cli import (
            inspection_result_from_wire,
            inspection_result_to_wire,
        )

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.add_module("r", nn.Identity())
                parent = nn.Module()
                parent.add_module("q-r", nn.Identity())
                self.add_module("p", parent)
                other_parent = nn.Module()
                nested_parent = nn.Module()
                nested_parent.add_module("r", self.r)
                other_parent.add_module("q", nested_parent)
                self.add_module("p-p", other_parent)

        graph = inspect_model_graph(Root())
        result = InspectionResult(
            identity=ModelIdentity("fixtures", "edge_ids"),
            preset="baseline",
            parameter_count=graph.nodes[0].parameter_count,
            parameter_size_bytes=graph.nodes[0].parameter_size_bytes,
            nodes=graph.nodes,
            edges=graph.edges,
        )

        edge_ids = [edge.id for edge in graph.edges]
        self.assertEqual(len(edge_ids), len(set(edge_ids)))
        self.assertEqual(
            inspection_result_from_wire(inspection_result_to_wire(result)),
            result,
        )

    def test_reserved_root_child_id_remains_unique_on_wire(self) -> None:
        from model_runtime.cli import (
            inspection_result_from_wire,
            inspection_result_to_wire,
        )

        model = nn.Module()
        model.add_module("__root__", nn.Sequential(nn.Linear(1, 1)))

        graph = inspect_model_graph(model)
        result = InspectionResult(
            identity=ModelIdentity("fixtures", "reserved_node_id"),
            preset="baseline",
            parameter_count=graph.nodes[0].parameter_count,
            parameter_size_bytes=graph.nodes[0].parameter_size_bytes,
            nodes=graph.nodes,
            edges=graph.edges,
        )

        node_ids = [node.id for node in graph.nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        self.assertEqual(graph.nodes[1].path, "__root__")
        nested_edge = next(edge for edge in graph.edges if edge.target == "__root__.0")
        self.assertEqual(nested_edge.source, "__root__#2")
        self.assertNotIn(
            ("__root__", "__root__.0"),
            {(edge.source, edge.target) for edge in graph.edges},
        )
        self.assertEqual(
            inspection_result_from_wire(inspection_result_to_wire(result)),
            result,
        )

    def test_repeated_shared_edges_have_unique_ids(self) -> None:
        shared = nn.Identity()

        class RepeatedEdges(nn.Module):
            def named_children(self):
                for _ in range(512):
                    yield "same", shared

        graph = inspect_model_graph(RepeatedEdges())
        edge_ids = [edge.id for edge in graph.edges]

        self.assertEqual(len(edge_ids), 512)
        self.assertEqual(len(edge_ids), len(set(edge_ids)))
        self.assertEqual(edge_ids[0], "__root__-same")
        self.assertEqual(edge_ids[-1], "__root__-same#512")

    def test_lazy_parameters_are_marked_as_uninitialized(self) -> None:
        root = inspect_model_graph(nn.LazyLinear(4, bias=False)).nodes[0]

        self.assertEqual(root.parameter_count, 0)
        self.assertEqual(root.details["uninitialized_parameter_count"], 1)

    def test_shared_parameter_membership_work_is_bounded(self) -> None:
        inspect_model_graph(
            nn.Linear(1, 1, bias=False),
            limits=InspectionCaptureLimits(
                maximum_parameter_memberships=1,
            ),
        )

        class SharedBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.parameters_by_index = nn.ParameterList(
                    nn.Parameter(torch.ones(1)) for _ in range(8)
                )

        class Parent(nn.Module):
            def __init__(self, shared: nn.Module) -> None:
                super().__init__()
                self.shared = shared

        shared = SharedBlock()
        root = nn.ModuleList(Parent(shared) for _ in range(8))

        with self.assertRaisesRegex(
            InspectionError,
            "parameter membership limit of 20 exceeded",
        ):
            inspect_model_graph(
                root,
                limits=InspectionCaptureLimits(
                    maximum_parameter_memberships=20,
                ),
            )

    def test_neuron_coordinates_are_bounded_with_exact_truncation_metadata(
        self,
    ) -> None:
        cluster = nn.Module()
        cluster.x_axis_total_neurons = 5
        cluster.y_axis_total_neurons = 1
        cluster.z_axis_total_neurons = 1
        cluster.cluster = nn.ModuleDict(
            {f"neuron_{index}_1_1": nn.Identity() for index in range(1, 6)}
        )

        root = inspect_model_graph(
            cluster,
            limits=InspectionCaptureLimits(maximum_neuron_coordinates=2),
        ).nodes[0]

        details = root.details["cluster"]
        self.assertEqual(details["instantiated"], 5)
        self.assertEqual(details["coordinates"], ((1, 1, 1), (2, 1, 1)))
        self.assertEqual(details["coordinates_total"], 5)
        self.assertIs(details["coordinates_truncated"], True)

    def test_terminal_connections_are_sliced_before_host_materialization(self) -> None:
        class ConnectionProbe(torch.Tensor):
            events: ClassVar[list[tuple[str, object]]] = []

            @staticmethod
            def __new__(_cls):
                return torch.tensor(
                    [
                        [1, 1, 1],
                        [2, 1, 1],
                        [3, 1, 1],
                        [4, 1, 1],
                        [5, 1, 1],
                    ]
                ).as_subclass(ConnectionProbe)

            def __getitem__(self, selected_slice):
                self.events.append(("slice", selected_slice))
                return super().__getitem__(selected_slice)

            def detach(self):
                self.events.append(("detach", tuple(self.shape)))
                return super().detach()

            def cpu(self, *args, **kwargs):
                self.events.append(("cpu", tuple(self.shape)))
                if tuple(self.shape) != (2, 3):
                    raise AssertionError("connections reached CPU before slicing")
                return super().cpu(*args, **kwargs)

        terminal = nn.Module()
        terminal.x_axis_position = 1
        terminal.y_axis_position = 1
        terminal.z_axis_position = 1
        terminal.total_neuron_connections = 5
        terminal.neuron_connections = ConnectionProbe()

        root = inspect_model_graph(
            terminal,
            limits=InspectionCaptureLimits(maximum_terminal_connections=2),
        ).nodes[0]

        details = root.details["terminal_reach"]
        self.assertEqual(details["connections"], ((1, 1, 1), (2, 1, 1)))
        self.assertEqual(details["total"], 5)
        self.assertIs(details["truncated"], True)
        self.assertEqual(
            ConnectionProbe.events,
            [
                ("slice", slice(None, 2, None)),
                ("detach", (2, 3)),
                ("cpu", (2, 3)),
            ],
        )

    def test_terminal_connections_reject_wide_tensor_before_host_transfer(
        self,
    ) -> None:
        class HostTransferTrap(torch.Tensor):
            @staticmethod
            def __new__(_cls):
                return torch.zeros((1, 4_096)).as_subclass(HostTransferTrap)

            def cpu(self, *_args, **_kwargs):
                raise AssertionError("invalid connections reached host transfer")

        terminal = nn.Module()
        terminal.x_axis_position = 1
        terminal.y_axis_position = 1
        terminal.z_axis_position = 1
        terminal.neuron_connections = HostTransferTrap()

        with self.assertRaisesRegex(
            InspectionError,
            r"neuron_connections must be a Tensor shaped \[connections, 3\]",
        ):
            module_details(terminal)

    def test_terminal_connections_reject_invalid_structural_values(self) -> None:
        invalid_connections = (
            torch.zeros(3),
            torch.zeros((1, 2)),
            torch.zeros((1, 4)),
            torch.zeros((1, 2, 3)),
            [[1, 2, 3]],
        )

        for connections in invalid_connections:
            with self.subTest(connections=connections):
                terminal = nn.Module()
                terminal.x_axis_position = 1
                terminal.y_axis_position = 1
                terminal.z_axis_position = 1
                terminal.neuron_connections = connections

                with self.assertRaisesRegex(
                    InspectionError,
                    r"neuron_connections must be a Tensor shaped \[connections, 3\]",
                ):
                    module_details(terminal)

    def test_terminal_connection_row_boundaries_preserve_projection(self) -> None:
        cases = (
            (torch.empty((0, 3), dtype=torch.long), 2, 0, False),
            (torch.tensor([[1, 2, 3], [4, 5, 6]]), 2, 2, False),
            (torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 2, 3, True),
        )

        for connections, limit, expected_total, expected_truncated in cases:
            with self.subTest(total=expected_total, limit=limit):
                terminal = nn.Module()
                terminal.x_axis_position = 1
                terminal.y_axis_position = 1
                terminal.z_axis_position = 1
                terminal.neuron_connections = connections

                details = module_details(
                    terminal,
                    limits=InspectionCaptureLimits(
                        maximum_terminal_connections=limit,
                    ),
                )["terminalReach"]

                self.assertEqual(details["connections"], connections[:limit].tolist())
                self.assertEqual(details["total"], expected_total)
                self.assertIs(details["truncated"], expected_truncated)

    def test_graph_output_budget_aborts_before_returning_oversized_result(self) -> None:
        class VerboseModule(nn.Module):
            pass

        VerboseModule.__doc__ = "x" * 10_000

        with self.assertRaisesRegex(
            InspectionError,
            "output byte limit of 256 exceeded",
        ):
            inspect_model_graph(
                VerboseModule(),
                limits=InspectionCaptureLimits(maximum_output_bytes=256),
            )

    def test_inspection_request_rejects_invalid_memory_limits(self) -> None:
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    InspectionRequest(
                        preset="baseline",
                        memory_limit_bytes=invalid,  # type: ignore[arg-type]
                    )

    def test_parameter_accounting_visits_chain_registrations_linearly(self) -> None:
        visited_parameter_entries = [0]

        class CountingChain(nn.Module):
            def __init__(self, depth: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1))
                if depth > 1:
                    self.child = CountingChain(depth - 1)

            def named_parameters(self, *args, **kwargs):
                for item in super().named_parameters(*args, **kwargs):
                    visited_parameter_entries[0] += 1
                    yield item

        graph = inspect_model_graph(CountingChain(100))

        self.assertEqual(graph.nodes[0].parameter_count, 100)
        self.assertLessEqual(visited_parameter_entries[0], 100)

    def test_broken_package_construction_failure_is_transport_neutral(self) -> None:
        package = _broken_package()

        with self.assertRaisesRegex(
            InspectionError,
            "Failed to import model package 'broken/missing'",
        ):
            inspect_model(package, InspectionRequest(preset="baseline"))

    def test_selected_package_produces_frozen_semantic_graph_records(self) -> None:
        package = model_package("linears/linear")
        self.assertIsNotNone(package)
        assert package is not None

        result = inspect_model(
            package,
            InspectionRequest(
                preset="baseline",
                dataset="Mnist",
                experiment_task="image-classification",
            ),
        )

        self.assertIsInstance(result, InspectionResult)
        self.assertEqual(result.identity, package.identity)
        self.assertEqual(result.preset, "baseline")
        self.assertIsInstance(result.nodes, tuple)
        self.assertIsInstance(result.edges, tuple)
        root = result.nodes[0]
        self.assertEqual(root.id, "__root__")
        self.assertEqual(root.path, "model")
        self.assertEqual(root.type_name, "Model")
        self.assertEqual(
            root.description,
            "Top-level inspected model wrapper that owns the architecture, loss, "
            "metrics, and runtime modules for the selected preset.",
        )
        self.assertEqual(root.graph_role, "architecture")
        self.assertGreater(root.parameter_count, 0)
        self.assertNotIn("parameterCount", root.details)
        with self.assertRaises(FrozenInstanceError):
            result.preset = "gating"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            root.details["changed"] = True  # type: ignore[index]

    def test_empty_module_graph_has_only_the_stable_root(self) -> None:
        graph = inspect_model_graph(nn.Sequential())

        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[0].id, "__root__")
        self.assertEqual(graph.nodes[0].parameter_count, 0)
        self.assertEqual(graph.edges, ())

    def test_layer_residual_config_preserves_flat_inspection_fields(self) -> None:
        configured_module = nn.Module()
        configured_module.cfg = LayerConfig(
            residual_config=WeightedBlendResidualConfig(
                model_config=LinearLayerConfig(bias_flag=True),
            )
        )

        configuration = inspect_model_graph(configured_module).nodes[0].configuration

        self.assertIsNotNone(configuration)
        assert configuration is not None
        serialized_fields = {field.key: field.value for field in configuration.fields}
        self.assertNotIn("residual_config", serialized_fields)
        self.assertEqual(
            serialized_fields["residual_connection_option"],
            "WeightedBlendResidualConfig",
        )
        self.assertEqual(
            serialized_fields["residual_model_config"],
            "LinearLayerConfig",
        )

    def test_graph_configuration_value_rendering_precedence_is_stable(self) -> None:
        @dataclass
        class NestedRecord:
            value: int = 1

        class RenderedValue:
            def __str__(self) -> str:
                return "rendered-value"

        @dataclass
        class DisplayConfig:
            none_value: object = None
            enum_value: object = ActivationOptions.RELU
            config_value: object = field(default_factory=LinearLayerConfig)
            type_value: object = LinearLayerConfig
            string_value: object = "text"
            integer_value: object = 3
            float_value: object = 1.5
            boolean_value: object = True
            record_value: object = field(default_factory=NestedRecord)
            fallback_value: object = field(default_factory=RenderedValue)

        configured_module = nn.Module()
        configured_module.cfg = DisplayConfig()

        configuration = inspect_model_graph(configured_module).nodes[0].configuration

        self.assertIsNotNone(configuration)
        assert configuration is not None
        self.assertEqual(
            tuple((item.key, item.value) for item in configuration.fields),
            (
                ("none_value", None),
                ("enum_value", "RELU"),
                ("config_value", "LinearLayerConfig"),
                ("type_value", "LinearLayerConfig"),
                ("string_value", "text"),
                ("integer_value", 3),
                ("float_value", 1.5),
                ("boolean_value", True),
                ("record_value", "NestedRecord"),
                ("fallback_value", "rendered-value"),
            ),
        )

    def test_semantic_detail_adapters_preserve_complete_top_level_order(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(3, 2))
        module.bias = nn.Parameter(torch.zeros(3))
        module.input_dim = 2
        module.hidden_dim = 3
        module.output_dim = 4
        module.embedding_dim = 8
        module.num_heads = 2
        module.num_layers = 5
        module.source_sequence_length = 6
        module.target_sequence_length = 7
        module.top_k = 2
        module.num_experts = 4
        module.routing_initialization_mode = "layer"
        module.postprocessing = SimpleNamespace(
            dropout_probability=0.25,
            gate=SimpleNamespace(option=ActivationOptions.GELU),
            gate_config=None,
            activation_function=ActivationOptions.RELU,
        )
        module.halting = SimpleNamespace(model=SimpleNamespace(min_steps=2))
        module.normalization = SimpleNamespace(position=LayerNormPositionOptions.AFTER)
        module.max_steps = 9
        module.recurrent_gate = SimpleNamespace(
            option=ActivationOptions.GELU,
            model=object(),
        )
        module.supports_recurrent_diagnostics = True
        module.causal_attention_mask_flag = True

        details = module_details(module)

        self.assertEqual(
            tuple(details),
            (
                "weightShape",
                "biasShape",
                "inputDim",
                "hiddenDim",
                "outputDim",
                "dims",
                "embeddingDim",
                "numHeads",
                "numLayers",
                "sourceSequenceLength",
                "targetSequenceLength",
                "topK",
                "numExperts",
                "routingMode",
                "dropout",
                "gateOption",
                "gate",
                "halting",
                "activation",
                "layerNorm",
                "recurrent",
                "causalAttention",
            ),
        )

    def test_neuron_cluster_capability_explicitly_suppresses_recurrence(self) -> None:
        cluster = nn.Module()
        cluster.x_axis_total_neurons = 1
        cluster.y_axis_total_neurons = 1
        cluster.z_axis_total_neurons = 1
        cluster.cluster = nn.ModuleDict({"neuron_1_1_1": nn.Identity()})
        cluster.max_steps = 9
        cluster.recurrent_diagnostic_step_limit = 7

        details = module_details(cluster)

        self.assertIn("cluster", details)
        self.assertNotIn("recurrent", details)

    def test_partial_neuron_cluster_capability_keeps_explicit_failure(self) -> None:
        for missing_attribute in (
            "y_axis_total_neurons",
            "z_axis_total_neurons",
        ):
            with self.subTest(missing_attribute=missing_attribute):
                cluster = nn.Module()
                cluster.x_axis_total_neurons = 1
                cluster.cluster = nn.ModuleDict()
                if missing_attribute == "z_axis_total_neurons":
                    cluster.y_axis_total_neurons = 1

                with self.assertRaisesRegex(AttributeError, missing_attribute):
                    module_details(cluster)

    def test_partial_terminal_capability_keeps_explicit_failure(self) -> None:
        for missing_attribute in ("y_axis_position", "z_axis_position"):
            with self.subTest(missing_attribute=missing_attribute):
                terminal = nn.Module()
                terminal.neuron_connections = torch.tensor([[1, 2, 3]])
                terminal.x_axis_position = 4
                if missing_attribute == "z_axis_position":
                    terminal.y_axis_position = 5

                with self.assertRaisesRegex(AttributeError, missing_attribute):
                    module_details(terminal)

    def test_partial_recurrent_schedule_capability_keeps_explicit_failure(self) -> None:
        recurrent = nn.Module()
        recurrent.recurrent_iteration_schedule = object()

        with self.assertRaisesRegex(AttributeError, "snapshot"):
            module_details(recurrent)

    def test_direct_terminal_reach_does_not_read_irrelevant_terminal(self) -> None:
        class DirectTerminalReach(nn.Module):
            @property
            def terminal(self) -> object:
                raise AssertionError("direct reach must not inspect terminal")

        reach = DirectTerminalReach()
        reach.neuron_connections = torch.tensor([[1, 2, 3]])
        reach.x_axis_position = 4
        reach.y_axis_position = 5
        reach.z_axis_position = 6

        self.assertEqual(
            module_details(reach)["terminalReach"]["position"],
            [4, 5, 6],
        )

    def test_recurrent_schedule_descriptor_is_read_once(self) -> None:
        @dataclass
        class ScheduleSnapshot:
            maximum_transition_count: int = 1
            active_transition_count: int = 1
            gradient_transition_count: int | None = None
            iteration_unit: str = "transition"
            initial_iterations: int = 1
            maximum_iterations: int = 1
            active_iterations: int = 1
            iteration_increment: int = 1
            forward_calls_before_iteration_increment: int = 1
            forward_call_progress: int = 0
            complete: bool = False
            no_gradient_transition_count: int = 0
            smooth_iteration_growth: bool = False
            settled_iterations: int = 1
            transitioning: bool = False
            transition_source_iterations: int | None = None
            transition_target_iterations: int | None = None
            transition_forward_index: int | None = None
            transition_forward_count: int = 0
            transition_weight: float = 0.0

        class Schedule:
            @staticmethod
            def snapshot() -> ScheduleSnapshot:
                return ScheduleSnapshot()

        class Recurrent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.schedule_reads = 0

            @property
            def recurrent_iteration_schedule(self) -> Schedule:
                self.schedule_reads += 1
                return Schedule()

        recurrent = Recurrent()

        module_details(recurrent)

        self.assertEqual(recurrent.schedule_reads, 1)

    def test_expert_capability_descriptor_is_read_once(self) -> None:
        top_k = _ReadOnceDescriptor(2)

        class Expert(nn.Module):
            pass

        Expert.top_k = top_k  # type: ignore[attr-defined]
        details = module_details(Expert())

        self.assertEqual(details["topK"], 2)
        self.assertEqual(top_k.read_count, 1)

    def test_delegated_layer_model_descriptors_are_read_once(self) -> None:
        gate = _ReadOnceDescriptor(SimpleNamespace(option=ActivationOptions.GELU))
        halting_model = _ReadOnceDescriptor(SimpleNamespace(min_steps=2))

        class Postprocessing:
            gate_config = None

        class Halting:
            pass

        Postprocessing.gate = gate  # type: ignore[attr-defined]
        Halting.model = halting_model  # type: ignore[attr-defined]

        class Layer(nn.Module):
            postprocessing = Postprocessing()
            halting = Halting()
            normalization = SimpleNamespace(position=None)

        details = module_details(Layer())

        self.assertEqual(details["gateOption"], "GELU")
        self.assertIs(details["gate"], True)
        self.assertIs(details["halting"], True)
        self.assertEqual(gate.read_count, 1)
        self.assertEqual(halting_model.read_count, 1)

    def test_recurrent_halting_descriptor_is_shared_across_detail_adapters(
        self,
    ) -> None:
        halting_model = _ReadOnceDescriptor(SimpleNamespace(min_steps=2))

        class Recurrent(nn.Module):
            max_steps = 3

        Recurrent.halting_model = halting_model  # type: ignore[attr-defined]

        details = module_details(Recurrent())

        self.assertIs(details["halting"], True)
        self.assertIs(details["recurrent"]["halting"], True)
        self.assertEqual(details["recurrent"]["minSteps"], 2)
        self.assertEqual(halting_model.read_count, 1)

    def test_cluster_capability_descriptors_are_read_once(self) -> None:
        x_axis_total_neurons = _ReadOnceDescriptor(2)
        cluster_members = _ReadOnceDescriptor(
            nn.ModuleDict({"neuron_1_1_1": nn.Identity()})
        )

        class Cluster(nn.Module):
            y_axis_total_neurons = 1
            z_axis_total_neurons = 1

        Cluster.x_axis_total_neurons = x_axis_total_neurons  # type: ignore[attr-defined]
        Cluster.cluster = cluster_members  # type: ignore[attr-defined]
        details = module_details(Cluster())

        self.assertEqual(details["cluster"]["capacity"], [2, 1, 1])
        self.assertEqual(x_axis_total_neurons.read_count, 1)
        self.assertEqual(cluster_members.read_count, 1)

    def test_terminal_capability_descriptors_are_read_once(self) -> None:
        connections = _ReadOnceDescriptor(torch.tensor([[1, 2, 3]]))
        x_axis_position = _ReadOnceDescriptor(4)

        class Terminal(nn.Module):
            y_axis_position = 5
            z_axis_position = 6

        Terminal.neuron_connections = connections  # type: ignore[attr-defined]
        Terminal.x_axis_position = x_axis_position  # type: ignore[attr-defined]
        details = module_details(Terminal())

        self.assertEqual(details["terminalReach"]["position"], [4, 5, 6])
        self.assertEqual(connections.read_count, 1)
        self.assertEqual(x_axis_position.read_count, 1)

    def test_residual_capability_descriptor_is_read_once(self) -> None:
        model_config = _ReadOnceDescriptor(LinearLayerConfig)

        class Residual:
            pass

        Residual.model_config = model_config  # type: ignore[attr-defined]

        @dataclass
        class Config:
            residual_config: object = field(default_factory=Residual)

        configured = nn.Module()
        configured.cfg = Config()

        configuration = inspect_model_graph(configured).nodes[0].configuration

        assert configuration is not None
        fields_by_key = {item.key: item.value for item in configuration.fields}
        self.assertEqual(fields_by_key["residual_model_config"], "LinearLayerConfig")
        self.assertEqual(model_config.read_count, 1)

    def test_recurrent_detail_order_and_fallback_are_stable(self) -> None:
        @dataclass
        class ScheduleSnapshot:
            maximum_transition_count: int = 9
            active_transition_count: int = 4
            gradient_transition_count: int | None = 2
            iteration_unit: str = "transition"
            initial_iterations: int = 2
            maximum_iterations: int = 9
            active_iterations: int = 4
            iteration_increment: int = 1
            forward_calls_before_iteration_increment: int = 3
            forward_call_progress: int = 2
            complete: bool = False
            no_gradient_transition_count: int = 3
            smooth_iteration_growth: bool = False
            settled_iterations: int = 4
            transitioning: bool = False
            transition_source_iterations: int | None = None
            transition_target_iterations: int | None = None
            transition_forward_index: int | None = None
            transition_forward_count: int = 0
            transition_weight: float = 0.0

        class Schedule:
            @staticmethod
            def snapshot() -> ScheduleSnapshot:
                return ScheduleSnapshot()

        recurrent = nn.Module()
        recurrent.recurrent_iteration_schedule = Schedule()
        recurrent.recurrent_gate = SimpleNamespace(
            option=ActivationOptions.GELU,
            model=object(),
        )
        recurrent.supports_recurrent_diagnostics = True
        recurrent.halting_model = SimpleNamespace(min_steps=2)
        recurrent.recurrent_layer_norm_position = LayerNormPositionOptions.BEFORE
        recurrent.answer_update_count = 5
        recurrent.latent_updates_per_answer_update = 6
        recurrent.high_cycles = 7
        recurrent.low_cycles = 8

        details = inspect_model_graph(recurrent).nodes[0].details["recurrent"]

        self.assertEqual(
            tuple(details),
            (
                "max_steps",
                "diagnostics",
                "gate",
                "gate_option",
                "halting",
                "active_steps",
                "gradient_transition_count",
                "iteration_schedule",
                "min_steps",
                "no_gradient_transition_count",
                "layer_norm",
                "answer_update_count",
                "latent_updates_per_answer_update",
                "high_cycles",
                "low_cycles",
            ),
        )
        self.assertEqual(
            dict(details),
            {
                "max_steps": 9,
                "diagnostics": True,
                "gate": True,
                "gate_option": "GELU",
                "halting": True,
                "active_steps": 4,
                "gradient_transition_count": 2,
                "iteration_schedule": {
                    "unit": "transition",
                    "initial_iterations": 2,
                    "maximum_iterations": 9,
                    "active_iterations": 4,
                    "iteration_increment": 1,
                    "forward_calls_before_iteration_increment": 3,
                    "forward_call_progress": 2,
                    "complete": False,
                    "smooth_iteration_growth": False,
                    "settled_iterations": 4,
                    "transitioning": False,
                    "transition_source_iterations": None,
                    "transition_target_iterations": None,
                    "transition_forward_index": None,
                    "transition_forward_count": 0,
                    "transition_weight": 0.0,
                },
                "min_steps": 2,
                "no_gradient_transition_count": 3,
                "layer_norm": "BEFORE",
                "answer_update_count": 5,
                "latent_updates_per_answer_update": 6,
                "high_cycles": 7,
                "low_cycles": 8,
            },
        )

        fallback = nn.Module()
        fallback.recurrent_diagnostic_step_limit = 7
        fallback_details = inspect_model_graph(fallback).nodes[0].details["recurrent"]
        self.assertEqual(
            dict(fallback_details),
            {
                "max_steps": 7,
                "diagnostics": False,
                "gate": False,
                "gate_option": None,
                "halting": False,
            },
        )

    def test_standard_recurrent_diagnostics_are_exposed_as_graph_capability(
        self,
    ) -> None:
        recurrent = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            block_config=LinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=False,
            ),
            max_steps=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            halting_config=SoftHaltingConfig(
                input_dim=4,
                threshold=0.99,
                ponder_cost_weight=1.0,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=None,
                min_steps=2,
            ),
        ).build()

        graph = inspect_model_graph(recurrent)
        root = graph.nodes[0]

        self.assertEqual(root.type_name, "RecurrentLayer")
        self.assertIs(root.details["recurrent"]["diagnostics"], True)
        self.assertEqual(root.details["recurrent"]["min_steps"], 2)
        self.assertIsNotNone(root.configuration)
        assert root.configuration is not None
        configuration = {field.key: field.value for field in root.configuration.fields}
        self.assertNotIn("min_steps", configuration)
        halting_node = next(
            node for node in graph.nodes if node.path == "halting_model"
        )
        self.assertIsNotNone(halting_node.configuration)
        assert halting_node.configuration is not None
        halting_configuration = {
            field.key: field.value for field in halting_node.configuration.fields
        }
        self.assertEqual(halting_configuration["min_steps"], 2)

    def test_standard_recurrent_iteration_schedule_is_exposed_as_graph_capability(
        self,
    ) -> None:
        recurrent = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            block_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=False,
                ),
            ),
            max_steps=10,
            initial_iterations=2,
            gradient_transition_count=2,
            iteration_increment=2,
            forward_calls_before_iteration_increment=3,
        ).build()
        for _ in range(3):
            recurrent(LayerState(hidden=torch.ones(1, 4)))

        graph = inspect_model_graph(recurrent)
        details = graph.nodes[0].details["recurrent"]

        self.assertEqual(details["max_steps"], 10)
        self.assertEqual(details["active_steps"], 4)
        self.assertEqual(details["gradient_transition_count"], 2)
        self.assertEqual(
            details["iteration_schedule"],
            {
                "unit": "transition",
                "initial_iterations": 2,
                "maximum_iterations": 10,
                "active_iterations": 4,
                "iteration_increment": 2,
                "forward_calls_before_iteration_increment": 3,
                "forward_call_progress": 3,
                "complete": False,
                "smooth_iteration_growth": False,
                "settled_iterations": 4,
                "transitioning": False,
                "transition_source_iterations": None,
                "transition_target_iterations": None,
                "transition_forward_index": None,
                "transition_forward_count": 0,
                "transition_weight": 0.0,
            },
        )

    def test_smooth_recurrent_handoff_phase_is_exposed_in_graph_inspection(
        self,
    ) -> None:
        recurrent = RecurrentLayerConfig(
            input_dim=4,
            output_dim=4,
            block_config=LayerConfig(
                activation=ActivationOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                layer_model_config=LinearLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    bias_flag=False,
                ),
            ),
            max_steps=3,
            initial_iterations=2,
            gradient_transition_count=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=4,
            smooth_iteration_growth_flag=True,
        ).build()
        for _ in range(4):
            recurrent(LayerState(hidden=torch.ones(1, 4)))

        details = inspect_model_graph(recurrent).nodes[0].details["recurrent"]

        self.assertEqual(details["active_steps"], 3)
        self.assertEqual(
            details["iteration_schedule"],
            {
                "unit": "transition",
                "initial_iterations": 2,
                "maximum_iterations": 3,
                "active_iterations": 3,
                "iteration_increment": 1,
                "forward_calls_before_iteration_increment": 4,
                "forward_call_progress": 4,
                "complete": False,
                "smooth_iteration_growth": True,
                "settled_iterations": 2,
                "transitioning": True,
                "transition_source_iterations": 2,
                "transition_target_iterations": 3,
                "transition_forward_index": 1,
                "transition_forward_count": 2,
                "transition_weight": 0.5,
            },
        )

    def test_tiny_recursive_model_is_discovered_through_generic_graph_inspection(
        self,
    ) -> None:
        recurrent = TinyRecursiveModelRecurrentConfig(
            input_dim=4,
            output_dim=4,
            block_config=LinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=False,
            ),
            latent_updates_per_answer_update=2,
            answer_update_count=3,
            initial_iterations=3,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            recurrent_layer_norm_position=LayerNormPositionOptions.BEFORE,
        ).build()

        graph = inspect_model_graph(recurrent)
        root = graph.nodes[0]

        self.assertEqual(root.type_name, "TinyRecursiveModelRecurrent")
        self.assertIs(root.details["recurrent"]["diagnostics"], True)
        self.assertEqual(root.details["recurrent"]["max_steps"], 9)
        self.assertEqual(
            root.details["recurrent"]["no_gradient_transition_count"],
            6,
        )
        self.assertEqual(root.details["recurrent"]["answer_update_count"], 3)
        self.assertEqual(root.details["recurrent"]["layer_norm"], "BEFORE")
        self.assertEqual(
            root.details["recurrent"]["latent_updates_per_answer_update"],
            2,
        )
        self.assertIsNotNone(root.configuration)
        assert root.configuration is not None
        self.assertEqual(
            root.configuration.type_name, "TinyRecursiveModelRecurrentConfig"
        )
        configuration = {field.key: field.value for field in root.configuration.fields}
        self.assertEqual(configuration["block_config"], "LinearLayerConfig")
        self.assertEqual(configuration["latent_updates_per_answer_update"], 2)
        self.assertEqual(configuration["answer_update_count"], 3)
        self.assertEqual(
            [node.path for node in graph.nodes[1:] if node.path == "block_model"],
            ["block_model"],
        )

    def test_hierarchical_reasoning_model_is_discovered_through_generic_graph_inspection(
        self,
    ) -> None:
        transition_config = LinearLayerConfig(
            input_dim=4,
            output_dim=4,
            bias_flag=False,
        )
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=4,
            output_dim=4,
            high_block_config=transition_config,
            low_block_config=transition_config,
            high_cycles=2,
            low_cycles=3,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            initialization_standard_deviation=0.0,
            recurrent_layer_norm_position=LayerNormPositionOptions.AFTER,
        ).build()

        graph = inspect_model_graph(recurrent)
        root = graph.nodes[0]

        self.assertEqual(root.type_name, "HierarchicalReasoningModelRecurrent")
        self.assertIs(root.details["recurrent"]["diagnostics"], True)
        self.assertEqual(root.details["recurrent"]["max_steps"], 8)
        self.assertEqual(
            root.details["recurrent"]["no_gradient_transition_count"],
            6,
        )
        self.assertEqual(root.details["recurrent"]["high_cycles"], 2)
        self.assertEqual(root.details["recurrent"]["low_cycles"], 3)
        self.assertEqual(root.details["recurrent"]["layer_norm"], "AFTER")
        self.assertIsNotNone(root.configuration)
        assert root.configuration is not None
        self.assertEqual(
            root.configuration.type_name, "HierarchicalReasoningModelRecurrentConfig"
        )
        configuration = {field.key: field.value for field in root.configuration.fields}
        self.assertEqual(configuration["high_block_config"], "LinearLayerConfig")
        self.assertEqual(configuration["low_block_config"], "LinearLayerConfig")
        child_paths = {node.path for node in graph.nodes[1:]}
        self.assertIn("high_model", child_paths)
        self.assertIn("low_model", child_paths)

    def test_semantic_graph_preserves_stable_paths_and_referential_integrity(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None

        result = inspect_model(package, InspectionRequest(preset="baseline"))

        node_ids = [node.id for node in result.nodes]
        edge_ids = [edge.id for edge in result.edges]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        self.assertEqual(len(edge_ids), len(set(edge_ids)))
        self.assertEqual(len(result.edges), len(result.nodes) - 1)
        self.assertTrue(
            all(
                edge.source in node_ids and edge.target in node_ids
                for edge in result.edges
            )
        )

    def test_override_and_lock_validation_flow_through_the_same_interface(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        result = inspect_model(
            package,
            InspectionRequest(
                preset="baseline",
                overrides={"hidden-dim": "128"},
            ),
        )
        nodes = {node.id: node for node in result.nodes}
        self.assertEqual(nodes["main_model.layers.0"].details["dims"], "128 -> 128")

        with self.assertRaisesRegex(InspectionError, "locked fields"):
            inspect_model(
                package,
                InspectionRequest(
                    preset="gating",
                    overrides={"stack_gate_flag": "false"},
                ),
            )

    def test_oversized_structure_is_rejected_before_model_constructor_lookup(
        self,
    ) -> None:
        package = model_package("linears/linear")
        assert package is not None
        limits = package.inspection_construction_limits

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ):
            with self.assertRaisesRegex(
                InspectionError,
                "HIDDEN_DIM.*exceeds.*maximum",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "hidden_dim": limits.maximum_hidden_dimension + 1,
                        },
                    ),
                )

    def test_dataset_dimensions_are_preflighted_before_model_construction(
        self,
    ) -> None:
        cases = (
            ("linears/linear", "Cifar100", 1_120_000, 144_512),
            ("gpt/linear", "WikiText103", 160_000_000, 69_429_376),
        )
        for package_key, dataset, memory_limit, expected_estimate in cases:
            with self.subTest(package=package_key, dataset=dataset):
                package = model_package(package_key)
                assert package is not None
                with patch.object(
                    ModelPackage,
                    "build_model",
                    side_effect=AssertionError("model constructor was observed"),
                ) as build_model:
                    with self.assertRaisesRegex(
                        InspectionError,
                        rf"estimated parameter count {expected_estimate} exceeds "
                        r"the memory-derived maximum",
                    ):
                        inspect_model(
                            package,
                            InspectionRequest(
                                preset="baseline",
                                dataset=dataset,
                                memory_limit_bytes=memory_limit,
                            ),
                        )

                build_model.assert_not_called()

    def test_effective_preflight_cannot_lower_the_existing_estimate(self) -> None:
        for package_key in (
            "vit/expert_linear",
            "vit/expert_linear_adaptive",
        ):
            with self.subTest(package=package_key):
                package = model_package(package_key)
                assert package is not None
                raw_estimate = preflight_inspection_configuration(
                    package,
                    {},
                    package.default_preset,
                )
                configuration = package.build_configuration(package.default_preset)

                self.assertEqual(
                    preflight_inspection_configuration(
                        package,
                        {},
                        package.default_preset,
                        effective_configuration=configuration,
                    ),
                    raw_estimate,
                )

    def test_effective_preflight_rejects_a_non_model_configuration(self) -> None:
        package = model_package("linears/linear")
        assert package is not None

        with self.assertRaisesRegex(TypeError, "must be a ModelConfig"):
            preflight_inspection_configuration(
                package,
                {},
                package.default_preset,
                effective_configuration=object(),  # type: ignore[arg-type]
            )

    def test_effective_root_fields_are_independent_of_runtime_default_names(
        self,
    ) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        with self.assertRaisesRegex(
            InspectionError,
            "field 'INPUT_DIM' value 2000000 exceeds",
        ):
            preflight_inspection_configuration(
                package,
                {},
                package.default_preset,
                effective_configuration=ModelConfig(
                    input_dim=2_000_000,
                    hidden_dim=20_000,
                    output_dim=2_000_000,
                    sequence_length=64,
                ),
            )

    def test_raw_preflight_still_precedes_dataset_resolution(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        maximum = package.inspection_construction_limits.maximum_hidden_dimension

        with patch.object(
            ModelPackage,
            "resolve_dataset",
            side_effect=AssertionError("dataset resolution was observed"),
        ) as resolve_dataset:
            with self.assertRaisesRegex(
                InspectionError,
                "HIDDEN_DIM.*exceeds.*maximum",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        dataset="missing-dataset",
                        overrides={"hidden_dim": maximum + 1},
                    ),
                )

        resolve_dataset.assert_not_called()

    def test_obvious_parameter_growth_is_rejected_before_construction(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        limits = package.inspection_construction_limits

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ):
            with self.assertRaisesRegex(
                InspectionError,
                "estimated parameter count.*exceeds",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "hidden_dim": limits.maximum_hidden_dimension,
                            "stack_num_layers": limits.maximum_layer_count,
                        },
                    ),
                )

    def test_oversized_vocabulary_embedding_is_rejected_before_construction(
        self,
    ) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ) as build_model:
            with self.assertRaisesRegex(
                InspectionError,
                "estimated parameter count.*exceeds",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "vocab_size": 1_000_000,
                            "model_dim": 16_384,
                        },
                    ),
                )

        build_model.assert_not_called()

    def test_oversized_learned_positions_are_rejected_before_construction(
        self,
    ) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ) as build_model:
            with self.assertRaisesRegex(
                InspectionError,
                "estimated parameter count.*exceeds",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="learned-positional",
                        overrides={
                            "model_dim": 1_024,
                            "source_sequence_length": 1_000_000,
                            "target_sequence_length": 1_000_000,
                        },
                    ),
                )

        build_model.assert_not_called()

    def test_parameter_storage_budget_rejects_before_construction(self) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ) as build_model:
            with self.assertRaisesRegex(
                InspectionError,
                "estimated parameter count.*exceeds",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "vocab_size": 131_072,
                            "model_dim": 2_048,
                        },
                    ),
                )

        build_model.assert_not_called()

    def test_oversized_neuron_cluster_is_rejected_before_construction(self) -> None:
        for package_key in (
            "neuron/linear",
            "neuron/linear_adaptive",
            "neuron/expert_linear",
            "neuron/expert_linear_adaptive",
        ):
            with self.subTest(package=package_key):
                package = model_package(package_key)
                assert package is not None
                with patch.object(
                    ModelPackage,
                    "build_model",
                    side_effect=AssertionError("model constructor was observed"),
                ) as build_model:
                    with self.assertRaisesRegex(
                        InspectionError,
                        "initial neuron count.*exceeds",
                    ):
                        inspect_model(
                            package,
                            InspectionRequest(
                                preset="baseline",
                                overrides={
                                    "cluster_x_axis_total_neurons": 100_000,
                                    "cluster_y_axis_total_neurons": 100_000,
                                    "cluster_z_axis_total_neurons": 1,
                                    "cluster_initial_x_axis_total_neurons": 100_000,
                                    "cluster_initial_y_axis_total_neurons": 100_000,
                                    "cluster_initial_z_axis_total_neurons": 1,
                                },
                            ),
                        )

                build_model.assert_not_called()

    def test_oversized_neuron_capacity_is_rejected_with_small_initial_cluster(
        self,
    ) -> None:
        package = model_package("neuron/linear")
        assert package is not None

        with patch.object(ModelPackage, "build_model") as build_model:
            with self.assertRaisesRegex(
                InspectionError,
                "neuron capacity.*exceeds",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "cluster_x_axis_total_neurons": 100_000,
                            "cluster_y_axis_total_neurons": 100_000,
                            "cluster_z_axis_total_neurons": 1,
                            "cluster_initial_x_axis_total_neurons": 1,
                            "cluster_initial_y_axis_total_neurons": 1,
                            "cluster_initial_z_axis_total_neurons": 1,
                        },
                    ),
                )

        build_model.assert_not_called()

    def test_product_limit_declarations_fail_closed_on_unknown_fields(self) -> None:
        package = model_package("linears/linear")
        assert package is not None
        package = replace(
            package,
            inspection_construction_limits=InspectionConstructionLimits(
                field_product_limits=(
                    InspectionFieldProductLimit(
                        label="fixture product",
                        factors=(("TYPO_DIMENSION",),),
                        maximum=10,
                    ),
                )
            ),
        )

        with self.assertRaisesRegex(
            InspectionError,
            "fixture product.*unknown Runtime Defaults field 'TYPO_DIMENSION'",
        ):
            preflight_inspection_configuration(
                package,
                {},
                package.resolve_preset("baseline"),
            )

        package = replace(
            package,
            inspection_construction_limits=InspectionConstructionLimits(
                field_maximums={"TYPO_DIMENSION": 10},
            ),
        )
        with self.assertRaisesRegex(
            InspectionError,
            "field maximums declare unknown Runtime Defaults.*TYPO_DIMENSION",
        ):
            preflight_inspection_configuration(
                package,
                {},
                package.resolve_preset("baseline"),
            )

    def test_preflight_validation_precedence_is_stable(self) -> None:
        base_package = model_package("linears/linear")
        assert base_package is not None
        unknown_product = InspectionFieldProductLimit(
            label="fixture product",
            factors=(("TYPO_PRODUCT_DIMENSION",),),
            maximum=1,
        )

        package = replace(
            base_package,
            inspection_construction_limits=InspectionConstructionLimits(
                maximum_parameter_estimate=1,
                field_maximums={
                    "HIDDEN_DIM": 1,
                    "TYPO_MAXIMUM_DIMENSION": 1,
                },
                field_product_limits=(unknown_product,),
            ),
        )
        with self.assertRaisesRegex(
            InspectionError,
            "field maximums declare unknown Runtime Defaults.*TYPO_MAXIMUM_DIMENSION",
        ):
            preflight_inspection_configuration(
                package,
                {"hidden_dim": 2},
                package.resolve_preset("baseline"),
            )

        package = replace(
            base_package,
            inspection_construction_limits=InspectionConstructionLimits(
                maximum_parameter_estimate=1,
                field_maximums={"HIDDEN_DIM": 1},
                field_product_limits=(unknown_product,),
            ),
        )
        with self.assertRaisesRegex(
            InspectionError,
            "field 'HIDDEN_DIM' value 2 exceeds",
        ):
            preflight_inspection_configuration(
                package,
                {"hidden_dim": 2},
                package.resolve_preset("baseline"),
            )

        package = replace(
            base_package,
            inspection_construction_limits=InspectionConstructionLimits(
                maximum_parameter_estimate=1,
                field_product_limits=(unknown_product,),
            ),
        )
        with self.assertRaisesRegex(
            InspectionError,
            "fixture product.*unknown Runtime Defaults field",
        ):
            preflight_inspection_configuration(
                package,
                {},
                package.resolve_preset("baseline"),
            )

    def test_product_limit_declarations_reject_invalid_keys_and_flags(self) -> None:
        with self.assertRaisesRegex(TypeError, "keys must be strings"):
            InspectionFieldProductLimit(
                label="fixture",
                factors=((123,),),  # type: ignore[arg-type]
                maximum=10,
            )
        with self.assertRaisesRegex(ValueError, "keys must be non-empty"):
            InspectionFieldProductLimit(
                label="fixture",
                factors=(("   ",),),
                maximum=10,
            )
        with self.assertRaisesRegex(TypeError, "flags must be boolean"):
            InspectionFieldProductLimit(
                label="fixture",
                factors=(("HIDDEN_DIM",),),
                maximum=10,
                repeats_dense_parameter_estimate=1,  # type: ignore[arg-type]
            )

    def test_neuron_product_multiplies_only_per_neuron_parameter_estimate(
        self,
    ) -> None:
        package = model_package("neuron/linear")
        assert package is not None

        estimate = preflight_inspection_configuration(
            package,
            {"input_dim": 1_000_000},
            package.resolve_preset("baseline"),
        )

        self.assertLess(
            estimate,
            package.inspection_construction_limits.maximum_parameter_estimate,
        )

    def test_catalog_defaults_fit_production_inspection_memory_policy(self) -> None:
        checked = 0
        for package in MODEL_CATALOG.values():
            for preset in package.preset_type:
                with self.subTest(
                    package=package.catalog_key,
                    preset=package.preset_name(preset),
                ):
                    preflight_inspection_configuration(
                        package,
                        {},
                        preset,
                        memory_limit_bytes=4 * 1024**3,
                    )
                checked += 1

        self.assertGreater(checked, 0)

    def test_worker_memory_limit_tightens_preconstruction_budget(self) -> None:
        package = model_package("transformer/linear")
        assert package is not None

        with patch.object(
            ModelPackage,
            "build_model",
            side_effect=AssertionError("model constructor was observed"),
        ) as build_model:
            with self.assertRaisesRegex(
                InspectionError,
                "memory-derived maximum",
            ):
                inspect_model(
                    package,
                    InspectionRequest(
                        preset="baseline",
                        overrides={
                            "vocab_size": 32_768,
                            "model_dim": 1_024,
                        },
                        memory_limit_bytes=512 * 1024**2,
                    ),
                )

        build_model.assert_not_called()

    def test_construction_limits_reject_non_integer_and_non_finite_bounds(
        self,
    ) -> None:
        for invalid in (True, 1.5, float("nan"), float("inf"), 0, -1):
            with (
                self.subTest(fixed_limit=invalid),
                self.assertRaisesRegex(ValueError, "must be positive"),
            ):
                InspectionConstructionLimits(
                    maximum_parameter_estimate=invalid,
                )

        for invalid in (True, float("nan"), float("inf"), 0, -1):
            with (
                self.subTest(field_maximum=invalid),
                self.assertRaisesRegex(ValueError, "finite positive numbers"),
            ):
                InspectionConstructionLimits(
                    field_maximums={"HIDDEN_DIM": invalid},
                )

        huge_limit = 10**1_000
        limits = InspectionConstructionLimits(
            field_maximums={"HIDDEN_DIM": huge_limit},
        )
        self.assertEqual(limits.field_maximums["HIDDEN_DIM"], huge_limit)


if __name__ == "__main__":
    unittest.main()
