import unittest
from dataclasses import replace

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
    ResidualConnectionOptions,
    RowLayout,
    RowLayoutAwareModule,
)
from emperor.layers._composition.gate import LayerGate
from emperor.linears import LinearLayerConfig
from emperor.memory import MemoryPositionOptions


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
        apply_output_pipeline_flag=False,
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
    return ResidualConfig(
        option=ResidualConnectionOptions.WEIGHTED_RESIDUAL,
        model_config=AdaptiveLinearLayerConfig(
            bias_flag=True,
            adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                grouping_scope=AdaptiveParameterGroupingScopeOptions.ROWS,
                group_count=2,
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


class LayoutAwareTensorSpy(RowLayoutAwareModule, nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.layouts = []

    def forward(self, input, *, row_layout=None):
        self.layouts.append(row_layout)
        if self.fail:
            raise RuntimeError("stop after layout capture")
        return input + 1.0


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


class RecurrentBlockSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_layouts = []

    def forward(self, state):
        self.received_layouts.append(state.row_layout)
        return replace(state, hidden=state.hidden + 1.0)


class LayoutDroppingRecurrentBlockSpy(nn.Module):
    def forward(self, state):
        return replace(state, hidden=state.hidden + 1.0, row_layout=None)


class PassthroughHalting(nn.Module):
    def update_halting_state(self, previous_state, hidden):
        return previous_state, hidden


class PassthroughMemory(nn.Module):
    memory_position_option = MemoryPositionOptions.BEFORE_AFFINE

    def forward(self, hidden):
        return hidden


class RowLayoutTransportTests(unittest.TestCase):
    def setUp(self):
        self.layout = RowLayout.rows(
            4,
            context_sharing_restricted=False,
        )
        self.inputs = torch.zeros(4, 2)

    def test_layer_passes_layout_only_to_nominally_aware_tensor_models(self):
        aware_layer = plain_layer()
        aware_spy = LayoutAwareTensorSpy()
        aware_layer.model = aware_spy

        aware_output = aware_layer(
            LayerState(hidden=self.inputs, row_layout=self.layout)
        )

        self.assertIs(aware_spy.layouts[0], self.layout)
        torch.testing.assert_close(aware_output.hidden, self.inputs + 1.0)

        ordinary_layer = plain_layer()
        ordinary_spy = OrdinaryTensorSpy()
        ordinary_layer.model = ordinary_spy
        ordinary_output = ordinary_layer(
            LayerState(hidden=self.inputs, row_layout=self.layout)
        )

        self.assertEqual(ordinary_spy.calls, 1)
        torch.testing.assert_close(ordinary_output.hidden, self.inputs + 1.0)

    def test_layer_passes_exact_layout_to_grouped_residual_coefficient_model(self):
        layer = plain_layer(residual_config=grouped_residual_config())
        coefficient_model = layer.residual_connection.model
        received_layouts = []
        hook = coefficient_model.register_forward_pre_hook(
            lambda _module, _args, kwargs: received_layouts.append(
                kwargs.get("row_layout")
            ),
            with_kwargs=True,
        )

        try:
            output_state = layer(LayerState(hidden=self.inputs, row_layout=self.layout))
        finally:
            hook.remove()

        self.assertIs(output_state.row_layout, self.layout)
        self.assertEqual(received_layouts, [self.layout])

    def test_layer_does_not_pass_layout_to_ordinary_residual_coefficient_model(self):
        layer = plain_layer(
            residual_config=ResidualConfig(
                option=ResidualConnectionOptions.WEIGHTED_RESIDUAL,
                model_config=LinearLayerConfig(bias_flag=True),
            )
        )
        coefficient_model = layer.residual_connection.model
        received_keywords = []
        hook = coefficient_model.register_forward_pre_hook(
            lambda _module, _args, kwargs: received_keywords.append(kwargs),
            with_kwargs=True,
        )

        try:
            output_state = layer(LayerState(hidden=self.inputs, row_layout=self.layout))
        finally:
            hook.remove()

        self.assertIs(output_state.row_layout, self.layout)
        self.assertEqual(received_keywords, [{}])

    def test_layer_layout_does_not_change_attention_residual_contract(self):
        layer = plain_layer(
            residual_config=ResidualConfig(
                option=ResidualConnectionOptions.ATTENTION_RESIDUAL,
            )
        )
        residual_state = layer.residual_connection.new_state(self.inputs)

        output_state = layer(
            LayerState(
                hidden=self.inputs,
                residual_state=residual_state,
                row_layout=self.layout,
            )
        )

        self.assertIs(output_state.row_layout, self.layout)
        self.assertEqual(len(residual_state.sources), 2)

    def test_layer_tensor_helpers_create_state_with_layout(self):
        layer = plain_layer()
        spy = LayoutAwareTensorSpy()
        layer.model = spy

        output_state = Layer.run_model_returning_state(
            layer,
            self.inputs,
            row_layout=self.layout,
        )

        self.assertIs(output_state.row_layout, self.layout)
        self.assertIs(spy.layouts[0], self.layout)

    def test_gate_fresh_state_carries_only_hidden_and_layout(self):
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

        output = gate(self.inputs, row_layout=self.layout)

        torch.testing.assert_close(output, self.inputs)
        received_state = spy.received_states[0]
        self.assertIs(received_state.row_layout, self.layout)
        self.assertIsNone(received_state.loss)
        self.assertIsNone(received_state.halting_state)
        self.assertIsNone(received_state.residual_state)

    def test_recurrent_state_preserves_layout_at_every_two_dimensional_step(self):
        recurrent = RecurrentLayer(
            RecurrentLayerConfig(
                input_dim=2,
                output_dim=2,
                max_steps=3,
                recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                block_config=linear_stack_config(),
                gate_config=None,
                residual_config=None,
                halting_config=None,
                memory_config=None,
            )
        )
        block_spy = RecurrentBlockSpy()
        recurrent.block_model = block_spy

        output_state = recurrent(LayerState(hidden=self.inputs, row_layout=self.layout))

        torch.testing.assert_close(output_state.hidden, self.inputs + 3.0)
        self.assertEqual(block_spy.received_layouts, [self.layout] * 3)
        self.assertIs(output_state.row_layout, self.layout)

    def test_recurrent_passes_exact_layout_to_grouped_residual_coefficient_model(
        self,
    ):
        recurrent = RecurrentLayer(
            RecurrentLayerConfig(
                input_dim=2,
                output_dim=2,
                max_steps=2,
                recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                block_config=linear_stack_config(),
                gate_config=None,
                residual_config=grouped_residual_config(),
                halting_config=None,
                memory_config=None,
            )
        )
        coefficient_model = recurrent.residual_connection.model
        received_layouts = []
        hook = coefficient_model.register_forward_pre_hook(
            lambda _module, _args, kwargs: received_layouts.append(
                kwargs.get("row_layout")
            ),
            with_kwargs=True,
        )

        try:
            output_state = recurrent(
                LayerState(hidden=self.inputs, row_layout=self.layout)
            )
        finally:
            hook.remove()

        self.assertIs(output_state.row_layout, self.layout)
        self.assertEqual(received_layouts, [self.layout, self.layout])

    def test_recurrent_block_cannot_drop_or_replace_layout_metadata(self):
        recurrent = RecurrentLayer(
            RecurrentLayerConfig(
                input_dim=2,
                output_dim=2,
                max_steps=1,
                recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                block_config=linear_stack_config(),
                gate_config=None,
                residual_config=None,
                halting_config=None,
                memory_config=None,
            )
        )
        recurrent.block_model = LayoutDroppingRecurrentBlockSpy()

        with self.assertRaisesRegex(ValueError, "preserve the exact row_layout"):
            recurrent(LayerState(hidden=self.inputs, row_layout=self.layout))

    def test_halting_owner_marks_context_sharing_restricted_before_model_call(self):
        layer = plain_layer()
        spy = LayoutAwareTensorSpy(fail=True)
        layer.model = spy
        layer.halting_model = nn.Identity()

        with self.assertRaisesRegex(RuntimeError, "stop after layout capture"):
            layer(LayerState(hidden=self.inputs, row_layout=self.layout))

        self.assertEqual(len(spy.layouts), 1)
        self.assertTrue(spy.layouts[0].context_sharing_restricted)
        self.assertIsNot(spy.layouts[0], self.layout)

    def test_layer_controller_restriction_is_local_to_the_owner_execution(self):
        for controller_name, controller in (
            ("halting_model", PassthroughHalting()),
            ("memory_model", PassthroughMemory()),
        ):
            with self.subTest(controller=controller_name):
                layer = plain_layer()
                spy = LayoutAwareTensorSpy()
                layer.model = spy
                setattr(layer, controller_name, controller)

                output_state = layer(
                    LayerState(hidden=self.inputs, row_layout=self.layout)
                )

                execution_layout = spy.layouts[0]
                self.assertTrue(execution_layout.context_sharing_restricted)
                self.assertIsNot(execution_layout, self.layout)
                self.assertIs(output_state.row_layout, self.layout)
                self.assertFalse(output_state.row_layout.context_sharing_restricted)

    def test_recurrent_controller_restriction_is_local_to_all_steps_and_gate(self):
        for controller_name, controller in (
            ("halting_model", PassthroughHalting()),
            ("memory_model", PassthroughMemory()),
        ):
            with self.subTest(controller=controller_name):
                recurrent = RecurrentLayer(
                    RecurrentLayerConfig(
                        input_dim=2,
                        output_dim=2,
                        max_steps=2,
                        recurrent_layer_norm_position=(
                            LayerNormPositionOptions.DISABLED
                        ),
                        block_config=linear_stack_config(),
                        gate_config=None,
                        residual_config=None,
                        halting_config=None,
                        memory_config=None,
                    )
                )
                block_spy = RecurrentBlockSpy()
                gate_spy = LayoutAwareTensorSpy()
                recurrent.block_model = block_spy
                recurrent.recurrent_gate = gate_spy
                setattr(recurrent, controller_name, controller)

                output_state = recurrent(
                    LayerState(hidden=self.inputs, row_layout=self.layout)
                )

                execution_layouts = block_spy.received_layouts + gate_spy.layouts
                self.assertEqual(len(execution_layouts), 4)
                self.assertTrue(
                    all(
                        layout.context_sharing_restricted
                        for layout in execution_layouts
                    )
                )
                self.assertTrue(
                    all(layout is execution_layouts[0] for layout in execution_layouts)
                )
                self.assertIs(output_state.row_layout, self.layout)
                self.assertFalse(output_state.row_layout.context_sharing_restricted)

    def test_already_restricted_layout_keeps_identity_through_controller_owner(self):
        restricted_layout = self.layout.with_context_sharing_restricted()
        layer = plain_layer()
        spy = LayoutAwareTensorSpy()
        layer.model = spy
        layer.memory_model = PassthroughMemory()

        output_state = layer(
            LayerState(hidden=self.inputs, row_layout=restricted_layout)
        )

        self.assertIs(spy.layouts[0], restricted_layout)
        self.assertIs(output_state.row_layout, restricted_layout)


if __name__ == "__main__":
    unittest.main()
