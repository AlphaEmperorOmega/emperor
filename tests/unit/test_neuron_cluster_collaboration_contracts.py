import ast
import inspect
import textwrap
import unittest

import torch

from emperor.neuron import NeuronClusterConfig
from emperor.neuron._cluster.beam_routes import _NeuronClusterBeamRoutesMixin
from emperor.neuron._cluster.checkpointing import _NeuronClusterCheckpointingMixin
from emperor.neuron._cluster.plasticity import _NeuronClusterPlasticityMixin
from emperor.neuron._cluster.recurrent_routes import (
    _NeuronClusterRecurrentRoutesMixin,
)
from emperor.neuron._cluster.state import (
    _NeuronClusterForwardContext,
    _NeuronClusterStateMixin,
)
from emperor.neuron._cluster.topology import _NeuronClusterTopologyMixin
from unit.test_neuron import NeuronTestCase


class TestNeuronClusterCollaborationContracts(NeuronTestCase):
    def test_mixins_declare_every_sibling_capability_they_consume(self) -> None:
        mixins = (
            _NeuronClusterTopologyMixin,
            _NeuronClusterStateMixin,
            _NeuronClusterRecurrentRoutesMixin,
            _NeuronClusterBeamRoutesMixin,
            _NeuronClusterPlasticityMixin,
            _NeuronClusterCheckpointingMixin,
        )

        for mixin in mixins:
            with self.subTest(mixin=mixin.__name__):
                class_definition = ast.parse(textwrap.dedent(inspect.getsource(mixin)))
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

                self.assertEqual(
                    consumed_attributes - set(mixin.__annotations__) - owned_methods,
                    set(),
                )

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
