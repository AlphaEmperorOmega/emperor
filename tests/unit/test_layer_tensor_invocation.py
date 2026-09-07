import unittest
from dataclasses import fields, replace

import torch
import torch.nn as nn

from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    RecurrentLayerConfig,
    ResidualConfig,
    WeightedResidualConfig,
)
from emperor.layers._composition.gate import LayerGate
from emperor.linears import LinearLayerConfig
from emperor.memory import MemoryPositionOptions
from support.adaptive_grouping import grouping_value


def linear_stack_config(
    dim: int = 2,
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> LayerStackConfig:
    resolved_input_dim = dim if input_dim is None else input_dim
    resolved_output_dim = dim if output_dim is None else output_dim
    return LayerStackConfig(
        input_dim=resolved_input_dim,
        hidden_dim=max(resolved_input_dim, resolved_output_dim),
        output_dim=resolved_output_dim,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=LayerConfig(
            input_dim=resolved_input_dim,
            output_dim=resolved_output_dim,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=resolved_input_dim,
                output_dim=resolved_output_dim,
                bias_flag=True,
            ),
        ),
    )


def grouped_residual_config(dim: int = 2) -> ResidualConfig:
    return WeightedResidualConfig(
        model_config=AdaptiveLinearLayerConfig(
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                grouping_config=grouping_value(
                    AdaptiveParameterGroupingScopeOptions.ROWS, 2
                ),
                bias_config=AdditiveDynamicBiasConfig(
                    decay_schedule=WeightDecayScheduleOptions.DISABLED,
                    decay_rate=0.0,
                    decay_warmup_batches=0,
                    model_config=linear_stack_config(
                        input_dim=dim * 2,
                        output_dim=dim,
                    ),
                ),
            ),
        ),
    )


def plain_layer(
    dim: int = 2,
    *,
    residual_config: ResidualConfig | None = None,
) -> Layer:
    return Layer(
        LayerConfig(
            input_dim=dim,
            output_dim=dim,
            activation=ActivationOptions.DISABLED,
            residual_config=residual_config,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(
                input_dim=dim,
                output_dim=dim,
                bias_flag=True,
            ),
        )
    )


class OrdinaryTensorSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input):
        self.calls += 1
        return input + 1.0


class GateStateSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_states = []

    def forward(self, state):
        self.received_states.append(state)
        return replace(state, hidden=torch.zeros_like(state.hidden))


class PassthroughHalting(nn.Module):
    def update_halting_state(self, previous_state, hidden):
        return previous_state, hidden


class PassthroughMemory(nn.Module):
    memory_position_option = MemoryPositionOptions.BEFORE_AFFINE

    def forward(self, hidden):
        return hidden


class LayerTensorInvocationTests(unittest.TestCase):
    def test_layer_invokes_ordinary_tensor_model_and_retains_state(self):
        layer = plain_layer()
        spy = OrdinaryTensorSpy()
        layer.model = spy
        inputs = torch.zeros(4, 2)
        state = LayerState(hidden=inputs)
        result = layer(state)
        self.assertIs(result, state)
        self.assertEqual(spy.calls, 1)
        torch.testing.assert_close(result.hidden, inputs + 1)
        torch.testing.assert_close(
            Layer.run_model_from_hidden(layer, inputs).hidden, inputs + 1
        )

    def test_grouped_residual_coefficients_receive_each_current_group_context(self):
        layer = plain_layer(residual_config=grouped_residual_config())
        with torch.no_grad():
            layer.model.weight_params.copy_(torch.eye(2))
            layer.model.bias_params.zero_()
        inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2).requires_grad_()
        contexts = []
        generator = layer.residual.connection.model.adaptive_behaviour.bias_model.model[
            0
        ].model
        hook = generator.register_forward_pre_hook(
            lambda _module, args: contexts.append(args[0].detach().clone())
        )
        try:
            output = layer(LayerState(hidden=inputs)).hidden
        finally:
            hook.remove()
        expected = torch.cat((inputs, inputs), -1).reshape(2, 2, 4).sum(1)
        torch.testing.assert_close(contexts[0], expected)
        output.sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_gate_creates_fresh_state_and_preserves_arithmetic(self):
        gate = LayerGate(
            GateConfig(
                gate_dim=2,
                option=LayerGateOptions.ADDITION,
                activation=ActivationOptions.DISABLED,
                model_config=linear_stack_config(),
            )
        )
        spy = GateStateSpy()
        gate.model = spy
        inputs = torch.randn(4, 2)
        torch.testing.assert_close(gate(inputs), inputs)
        state = spy.received_states[0]
        self.assertIsNone(state.loss)
        self.assertIsNone(state.halting_state)
        self.assertIsNone(state.residual_state)

    def test_recurrence_recomputes_grouped_residual_context_each_transition(self):
        recurrent = RecurrentLayer(
            RecurrentLayerConfig(
                input_dim=2,
                output_dim=2,
                max_steps=2,
                initial_iterations=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
                recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                block_config=linear_stack_config(),
                residual_config=grouped_residual_config(),
                gate_config=None,
                halting_config=None,
                memory_config=None,
            )
        )
        inputs = torch.randn(4, 2, requires_grad=True)
        contexts = []
        generator = (
            recurrent.residual_connection.model.adaptive_behaviour.bias_model.model[
                0
            ].model
        )
        hook = generator.register_forward_pre_hook(
            lambda _module, args: contexts.append(args[0].detach().clone())
        )
        try:
            result = recurrent(LayerState(hidden=inputs))
        finally:
            hook.remove()
        self.assertEqual(len(contexts), 2)
        self.assertFalse(torch.equal(contexts[0], contexts[1]))
        result.hidden.sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_attention_residual_retains_forward_local_history(self):
        layer = plain_layer(residual_config=AttentionResidualConfig())
        inputs = torch.zeros(4, 2)
        residual_state = layer.residual.connection.new_state(inputs)
        layer(LayerState(hidden=inputs, residual_state=residual_state))
        self.assertEqual(len(residual_state.sources), 2)

    def test_controller_delegates_retain_state_and_hidden(self):
        layer = plain_layer()
        layer.halting.model = PassthroughHalting()
        layer.memory.model = PassthroughMemory()
        inputs = torch.zeros(4, 2)
        for process in (layer.halting.apply_halting, layer.memory.before_model):
            state = LayerState(hidden=inputs)
            self.assertIs(process(state), state)
            self.assertIs(state.hidden, inputs)

    def test_layer_state_fields_match_execution_contract(self):
        self.assertEqual(
            {field.name for field in fields(LayerState)},
            {"hidden", "loss", "halting_state", "residual_state"},
        )
