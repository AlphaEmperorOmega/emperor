import ast
import copy
import importlib
import inspect
import io
import textwrap
import unittest
from unittest.mock import patch

import torch

from emperor.neuron import NeuronCluster, NeuronClusterConfig
from emperor.neuron._cluster.checkpointing import ClusterCheckpointDelegate
from emperor.neuron._cluster.plasticity import ClusterPlasticityDelegate
from emperor.neuron._cluster.routing.beam import BeamRoutingDelegate
from emperor.neuron._cluster.routing.delegate import ClusterRoutingDelegate
from emperor.neuron._cluster.routing.state import (
    RouteStateDelegate,
    _NeuronClusterForwardContext,
)
from emperor.neuron._cluster.topology import ClusterTopologyDelegate
from emperor.nn import Module
from unit.test_neuron import NeuronTestCase, ScriptedNeuron, ScriptedSampler


class TestNeuronClusterCollaborationContracts(NeuronTestCase):
    def test_cluster_inherits_only_the_framework_module(self) -> None:
        self.assertEqual(NeuronCluster.__bases__, (Module,))

    def test_routing_package_exposes_only_the_traversal_delegate(self) -> None:
        routing_package = importlib.import_module("emperor.neuron._cluster.routing")
        self.assertEqual(routing_package.__all__, ("ClusterRoutingDelegate",))
        self.assertIs(routing_package.ClusterRoutingDelegate, ClusterRoutingDelegate)

    def test_pickle_restores_current_routing(self) -> None:
        original = self.cluster_config().build().eval()
        payload = io.BytesIO()
        torch.save(original, payload)
        payload.seek(0)
        rng_state = torch.get_rng_state().clone()

        restored = torch.load(payload, weights_only=False)

        torch.testing.assert_close(torch.get_rng_state(), rng_state)
        torch.testing.assert_close(restored.state_dict(), original.state_dict())
        for delegate in self.delegates(restored):
            self.assertIs(
                getattr(delegate, f"_{type(delegate).__name__}__owner"), restored
            )
        input_batch = torch.ones(self.batch_size, self.input_dim)
        torch.testing.assert_close(restored(input_batch), original(input_batch))
        restored.load_state_dict(original.state_dict(), strict=True)

    def test_delegates_consume_only_explicitly_initialized_dependencies(self) -> None:
        delegates = (
            ClusterTopologyDelegate,
            RouteStateDelegate,
            ClusterRoutingDelegate,
            BeamRoutingDelegate,
            ClusterPlasticityDelegate,
            ClusterCheckpointDelegate,
        )

        for delegate in delegates:
            with self.subTest(delegate=delegate.__name__):
                self.assertEqual(delegate.__bases__, (object,))
                class_definition = ast.parse(
                    textwrap.dedent(inspect.getsource(delegate))
                )
                class_node = class_definition.body[0]
                owned_methods = {
                    node.name
                    for node in class_node.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                consumed_attributes = {
                    node.attr
                    for node in ast.walk(class_node)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and isinstance(node.ctx, ast.Load)
                }
                initializer = next(
                    node
                    for node in class_node.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                )
                initialized_attributes = {
                    node.attr
                    for node in ast.walk(initializer)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and isinstance(node.ctx, ast.Store)
                }

                self.assertEqual(
                    consumed_attributes - initialized_attributes - owned_methods,
                    set(),
                )

    def test_delegates_are_not_registered_modules_or_tensor_owners(self) -> None:
        model = self.cluster_config().build()

        self.assertEqual(tuple(model._modules), ("cluster", "entry_sampler"))
        for delegate in self.delegates(model):
            self.assertNotIsInstance(delegate, torch.nn.Module)
            self.assertFalse(
                any(
                    isinstance(value, torch.Tensor) for value in vars(delegate).values()
                )
            )
        self.assertFalse(any("delegate" in key for key in model.state_dict()))

    def cluster_config(self) -> NeuronClusterConfig:
        return NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        )

    @staticmethod
    def delegates(model):
        routing = model._NeuronCluster__routing
        return (
            model._NeuronCluster__topology,
            model._NeuronCluster__plasticity,
            routing,
            routing._ClusterRoutingDelegate__state,
            routing._ClusterRoutingDelegate__beam_routes,
            model._NeuronCluster__checkpointing,
        )

    def test_routing_uses_replaced_owner_modules(self) -> None:
        model = self.cluster_config().build().eval()
        model.entry_sampler = ScriptedSampler(indices=[0], probabilities=[1.0])
        replacement_neuron = ScriptedNeuron(
            routes=[[0, 0, 0]],
            probabilities=[1.0],
            delta=[2.0] * self.input_dim,
        )
        model.cluster = torch.nn.ModuleDict({"neuron_1_1_1": replacement_neuron})
        input_batch = torch.ones(self.batch_size, self.input_dim)

        output, _ = model(input_batch)

        torch.testing.assert_close(output, input_batch + 2.0)

    def test_copy_and_pickle_rebind_delegates_to_the_restored_owner(self) -> None:
        for copy_method in ("deepcopy", "pickle"):
            with self.subTest(copy_method=copy_method):
                original = self.cluster_config().build()
                if copy_method == "deepcopy":
                    restored = copy.deepcopy(original)
                else:
                    payload = io.BytesIO()
                    torch.save(original, payload)
                    payload.seek(0)
                    restored = torch.load(payload, weights_only=False)

                for original_delegate, restored_delegate in zip(
                    self.delegates(original), self.delegates(restored), strict=True
                ):
                    self.assertIsNot(original_delegate, restored_delegate)
                    owner_field = f"_{type(restored_delegate).__name__}__owner"
                    self.assertIs(getattr(restored_delegate, owner_field), restored)
                self.assertEqual(
                    tuple(original.state_dict()), tuple(restored.state_dict())
                )
                original_counters = {
                    name: neuron.batch_counter.clone()
                    for name, neuron in original.cluster.items()
                }
                restored(torch.ones(self.batch_size, self.input_dim))
                for name, neuron in original.cluster.items():
                    torch.testing.assert_close(
                        neuron.batch_counter, original_counters[name]
                    )
                self.assertGreater(
                    restored.cluster["neuron_1_1_1"].batch_counter.item(), 0
                )

    def test_legacy_module_pickle_restores_delegates_without_resetting_state(
        self,
    ) -> None:
        original = self.cluster_config().build().eval()
        original._growth_counters_are_global = True
        original._checkpoint_removed_parameter_ids = {123}
        original._load_state_dict_pre_hooks.clear()
        original._load_state_dict_post_hooks.clear()
        original.register_load_state_dict_pre_hook(
            original._reconcile_cluster_with_state_dict
        )
        original.register_load_state_dict_post_hook(
            original._mark_growth_counters_global_after_load
        )
        input_batch = torch.ones(self.batch_size, self.input_dim)
        expected_output = original(input_batch)
        original_state_dict = copy.deepcopy(original.state_dict())
        for delegate_name in ("topology", "plasticity", "routing", "checkpointing"):
            delattr(original, f"_NeuronCluster__{delegate_name}")
        payload = io.BytesIO()
        torch.save(original, payload)
        payload.seek(0)
        rng_state = torch.get_rng_state().clone()

        restored = torch.load(payload, weights_only=False)

        torch.testing.assert_close(torch.get_rng_state(), rng_state)
        self.assertTrue(restored._growth_counters_are_global)
        self.assertEqual(restored._checkpoint_removed_parameter_ids, {123})
        self.assertEqual(len(restored._load_state_dict_pre_hooks), 1)
        self.assertEqual(len(restored._load_state_dict_post_hooks), 1)
        for delegate in self.delegates(restored):
            owner_field = f"_{type(delegate).__name__}__owner"
            self.assertIs(getattr(delegate, owner_field), restored)
        torch.testing.assert_close(restored.state_dict(), original_state_dict)
        torch.testing.assert_close(restored(input_batch), expected_output)
        restored._growth_counters_are_global = False
        restored.load_state_dict(original_state_dict, strict=True)
        self.assertTrue(restored._growth_counters_are_global)

    def test_forward_orders_delegates_and_shares_one_local_context(self) -> None:
        model = self.cluster_config().build()
        plasticity = model._NeuronCluster__plasticity
        routing = model._NeuronCluster__routing
        input_batch = torch.ones(self.batch_size, self.input_dim)
        baseline = object()
        events = []

        def capture_baseline():
            events.append("capture")
            return baseline

        def propagate_routes(*args):
            events.append("route")
            return input_batch, input_batch.new_zeros(()), None

        with (
            patch.object(
                plasticity,
                "capture_growth_counter_baseline",
                side_effect=capture_baseline,
            ),
            patch.object(
                routing,
                "propagate",
                side_effect=propagate_routes,
            ) as propagate,
            patch.object(
                plasticity,
                "advance_grown_neuron_warmup",
                side_effect=lambda: events.append("warmup"),
            ),
            patch.object(
                plasticity,
                "check_neuron_growth",
                side_effect=lambda *args: events.append("grow"),
            ) as check_growth,
            patch.object(
                plasticity,
                "check_neuron_atrophy",
                side_effect=lambda *args: events.append("prune"),
            ) as check_atrophy,
        ):
            model(input_batch)

        self.assertEqual(events, ["capture", "route", "warmup", "grow", "prune"])
        context = propagate.call_args.args[-1]
        self.assertIsInstance(context, _NeuronClusterForwardContext)
        self.assertIs(check_growth.call_args.args[0], baseline)
        self.assertIs(check_growth.call_args.args[1], context)
        self.assertIs(check_atrophy.call_args.args[0], context)

    def test_forward_context_is_fresh_and_does_not_become_module_state(self) -> None:
        first_context = _NeuronClusterForwardContext()
        second_context = _NeuronClusterForwardContext()
        first_context.called_neuron_names.add("neuron_1_1_1")

        self.assertEqual(first_context.called_neuron_names, {"neuron_1_1_1"})
        self.assertEqual(second_context.called_neuron_names, set())

        model = NeuronClusterConfig(
            x_axis_total_neurons=1,
            y_axis_total_neurons=1,
            z_axis_total_neurons=1,
            max_steps=1,
            growth_threshold=None,
            neuron_config=self.full_sampler_neuron_config(),
        ).build()
        model(torch.randn(self.batch_size, self.input_dim))

        self.assertFalse(hasattr(model, "_neurons_called_this_forward"))
        self.assertNotIn("forward_context", model.state_dict())


if __name__ == "__main__":
    unittest.main()
