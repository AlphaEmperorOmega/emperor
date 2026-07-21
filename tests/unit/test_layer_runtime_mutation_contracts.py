import unittest

import torch
import torch.nn.functional as F

from emperor.config import ModelConfig
from emperor.halting import (
    HaltingHiddenStateModeOptions,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    RecurrentLayer,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.layers._composition.gate import LayerGate
from emperor.layers._recurrent import _RecurrentState
from emperor.layers._support import LayerModuleBase
from emperor.linears import LinearLayerConfig
from emperor.memory import (
    MemoryPositionOptions,
    WeightedDynamicMemoryConfig,
)
from support.layers import (
    base_layer_config,
    configure_weighted_memory,
    linear_stack_config,
    recurrent_config,
    set_layer_identity,
)


def _stale_memory_config(
    position: MemoryPositionOptions,
) -> WeightedDynamicMemoryConfig:
    return WeightedDynamicMemoryConfig(
        input_dim=7,
        output_dim=8,
        memory_position_option=position,
        test_time_training_learning_rate=None,
        test_time_training_num_inner_steps=None,
        model_config=linear_stack_config(7, bias_flag=True),
    )


def _stick_breaking_config(
    input_dim: int,
    *,
    threshold: float = 0.99,
) -> StickBreakingConfig:
    gate_config = linear_stack_config(
        input_dim,
        output_dim=2,
        bias_flag=True,
    )
    gate_config.last_layer_bias_option = LastLayerBiasOptions.DISABLED
    return StickBreakingConfig(
        input_dim=input_dim,
        threshold=threshold,
        dropout_probability=0.0,
        hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
        halting_gate_config=gate_config,
    )


def _run_identity_halting_step(halting_model, hidden: torch.Tensor):
    return halting_model.run_step(
        None,
        hidden,
        lambda computation: computation.raw_hidden,
    )


class LayerRuntimeMutationContractTests(unittest.TestCase):
    def test_gate_shape_error_preserves_the_active_option_name(self) -> None:
        gate = LayerGate(
            GateConfig(
                gate_dim=2,
                option=LayerGateOptions.MULTIPLIER,
                activation=ActivationOptions.DISABLED,
                model_config=linear_stack_config(2),
            )
        )
        gate.model = LayerStack(linear_stack_config(2, output_dim=3))
        current = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        with self.assertRaises(ValueError) as caught:
            gate(current)

        self.assertEqual(
            str(caught.exception),
            "MULTIPLIER requires gate output and current shapes to match, "
            "got gate output shape (2, 3) and current shape (2, 2).",
        )

    def test_memory_dispatch_is_disabled_without_a_memory_model(self) -> None:
        hidden = torch.tensor([[1.0, -2.0]])
        harness = LayerModuleBase()

        self.assertIsNone(harness.memory_model)
        self.assertIs(
            harness._maybe_apply_memory_by_position(
                hidden,
                MemoryPositionOptions.AFTER_AFFINE,
            ),
            hidden,
        )

    def test_layer_memory_dimensions_and_position_override_stale_config(self) -> None:
        weight = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, -1.0],
            ]
        )
        bias = torch.tensor([0.5, -1.0, 2.0])
        hidden_values = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])

        for position in (
            MemoryPositionOptions.BEFORE_AFFINE,
            MemoryPositionOptions.AFTER_AFFINE,
        ):
            with self.subTest(position=position):
                memory_config = _stale_memory_config(position)
                layer = Layer(
                    LayerConfig(
                        input_dim=2,
                        output_dim=3,
                        activation=ActivationOptions.DISABLED,
                        residual_config=None,
                        dropout_probability=0.0,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        gate_config=None,
                        halting_config=None,
                        memory_config=memory_config,
                        layer_model_config=LinearLayerConfig(
                            input_dim=2,
                            output_dim=3,
                            bias_flag=True,
                        ),
                    )
                )
                with torch.no_grad():
                    layer.model.weight_params.copy_(weight)
                    layer.model.bias_params.copy_(bias)
                configure_weighted_memory(layer.memory_model)
                hidden = hidden_values.clone().requires_grad_(True)

                output = layer(LayerState(hidden=hidden)).hidden

                self.assertEqual(layer.memory_model.input_dim, 2)
                self.assertEqual(layer.memory_model.output_dim, 3)
                self.assertEqual(memory_config.input_dim, 7)
                self.assertEqual(memory_config.output_dim, 8)
                if position == MemoryPositionOptions.BEFORE_AFFINE:
                    expected = (hidden * 1.75) @ weight + bias
                else:
                    expected = (hidden @ weight + bias) * 1.75
                torch.testing.assert_close(output, expected)
                output.sum().backward()
                self.assertIsNotNone(hidden.grad)
                self.assertTrue(torch.isfinite(hidden.grad).all())
                self.assertTrue(hidden.grad.ne(0).any())

    def test_plain_layer_ignores_completed_halting_state_without_a_controller(
        self,
    ) -> None:
        layer = Layer(base_layer_config(2))
        with torch.no_grad():
            layer.model.weight_params.copy_(2.0 * torch.eye(2))
        halting_model = _stick_breaking_config(2, threshold=0.4).build().eval()
        hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        completed_state = _run_identity_halting_step(halting_model, hidden)
        halting_model.finalize(completed_state, completed_state.raw_hidden)
        self.assertTrue(completed_state.halt_mask.all())
        state = LayerState(
            hidden=hidden.clone(),
            loss=torch.tensor(0.25),
            halting_state=completed_state,
        )

        result = layer(state)

        self.assertIs(result, state)
        self.assertIs(result.halting_state, completed_state)
        torch.testing.assert_close(result.hidden, hidden * 2.0)
        torch.testing.assert_close(result.loss, torch.tensor(0.25))

    def test_recurrent_memory_dimensions_override_stale_config(self) -> None:
        hidden_values = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        for position in (
            MemoryPositionOptions.BEFORE_AFFINE,
            MemoryPositionOptions.AFTER_AFFINE,
        ):
            with self.subTest(position=position):
                memory_config = _stale_memory_config(position)
                recurrent = RecurrentLayer(
                    recurrent_config(memory_config=memory_config)
                )
                configure_weighted_memory(recurrent.memory_model)
                hidden = hidden_values.clone().requires_grad_(True)

                output = recurrent(LayerState(hidden=hidden)).hidden

                self.assertEqual(recurrent.memory_model.input_dim, 2)
                self.assertEqual(recurrent.memory_model.output_dim, 2)
                self.assertEqual(memory_config.input_dim, 7)
                self.assertEqual(memory_config.output_dim, 8)
                torch.testing.assert_close(output, hidden * 1.75)
                output.square().sum().backward()
                self.assertIsNotNone(hidden.grad)
                self.assertTrue(torch.isfinite(hidden.grad).all())
                self.assertTrue(hidden.grad.ne(0).all())

    def test_recurrent_halting_override_finalization_and_guard_are_exact(
        self,
    ) -> None:
        halting_config = _stick_breaking_config(7)
        recurrent = RecurrentLayer(
            recurrent_config(halting_config=halting_config)
        ).eval()
        hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        outer_halting_state = _run_identity_halting_step(
            recurrent.halting_model,
            hidden,
        )
        public_state = LayerState(
            hidden=hidden.clone(),
            loss=torch.tensor(1.25),
            halting_state=outer_halting_state,
        )

        public_result = recurrent(public_state)

        self.assertEqual(recurrent.halting_model.input_dim, 2)
        self.assertEqual(halting_config.input_dim, 7)
        self.assertIs(public_result, public_state)
        self.assertIs(public_result.halting_state, outer_halting_state)
        torch.testing.assert_close(public_result.hidden, hidden)
        torch.testing.assert_close(public_result.loss, torch.tensor(1.75))

        internal_halting_state = _run_identity_halting_step(
            recurrent.halting_model,
            hidden,
        )
        context_state = LayerState(hidden=hidden + 10.0, loss=torch.tensor(8.0))
        run_state = _RecurrentState(
            hidden=hidden,
            loss=torch.tensor(1.25),
            context_state=context_state,
            halting_state=internal_halting_state,
        )

        finalized = recurrent._RecurrentLayer__maybe_finalize_recurrent_halting(
            run_state
        )

        torch.testing.assert_close(finalized.hidden, hidden)
        torch.testing.assert_close(finalized.loss, torch.tensor(1.75))
        self.assertIs(finalized.context_state, context_state)
        self.assertIs(finalized.halting_state, internal_halting_state)

        recurrent.halting_model = None
        guarded = recurrent._RecurrentLayer__maybe_finalize_recurrent_halting(run_state)
        self.assertIs(guarded, run_state)

    def test_stack_accepts_real_model_config_wrapper(self) -> None:
        layer_stack_config = linear_stack_config(2)
        wrapper = ModelConfig()
        wrapper.layer_stack_config = layer_stack_config

        stack = LayerStack(wrapper)
        set_layer_identity(stack[0])
        hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        output = stack(LayerState(hidden=hidden)).hidden

        self.assertIs(stack.cfg, layer_stack_config)
        torch.testing.assert_close(output, hidden)

    def test_stack_output_override_merges_pipeline_and_bias_behavior(self) -> None:
        stack = LayerStack(
            LayerStackConfig(
                input_dim=2,
                hidden_dim=2,
                output_dim=2,
                num_layers=2,
                last_layer_bias_option=LastLayerBiasOptions.ENABLED,
                apply_output_pipeline_flag=False,
                shared_gate_config=None,
                shared_halting_config=None,
                shared_memory_config=None,
                layer_config=LayerConfig(
                    input_dim=2,
                    output_dim=2,
                    activation=ActivationOptions.TANH,
                    residual_config=ResidualConfig(
                        option=(ResidualConnectionOptions.RESIDUAL)
                    ),
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.AFTER,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=2,
                        output_dim=2,
                        bias_flag=False,
                    ),
                ),
            )
        ).eval()
        first, output_layer = stack

        self.assertIs(first.last_layer_flag, False)
        self.assertIs(output_layer.last_layer_flag, True)
        self.assertEqual(first.activation_function, ActivationOptions.TANH)
        self.assertEqual(
            first.residual_config.option,
            ResidualConnectionOptions.RESIDUAL,
        )
        self.assertEqual(
            first.layer_norm_position,
            LayerNormPositionOptions.AFTER,
        )
        self.assertEqual(output_layer.activation_function, ActivationOptions.DISABLED)
        self.assertIsNone(output_layer.residual_config)
        self.assertEqual(
            output_layer.layer_norm_position,
            LayerNormPositionOptions.DISABLED,
        )
        self.assertIsNone(output_layer.residual_connection)
        self.assertIsNone(output_layer.layer_norm_module)
        self.assertTrue(output_layer.model.bias_flag)
        self.assertIsNotNone(output_layer.model.bias_params)

        first_weight = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        output_weight = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
        output_bias = torch.tensor([1.0, -2.0])
        with torch.no_grad():
            first.model.weight_params.copy_(first_weight)
            output_layer.model.weight_params.copy_(output_weight)
            output_layer.model.bias_params.copy_(output_bias)
        hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        output = stack(LayerState(hidden=hidden)).hidden

        first_affine = hidden @ first_weight
        first_residual = torch.tanh(first_affine) + hidden
        first_normalized = F.layer_norm(
            first_residual,
            (2,),
            eps=first.layer_norm_module.eps,
        )
        expected = first_normalized @ output_weight + output_bias
        torch.testing.assert_close(output, expected)

    def test_shared_controller_dimensions_override_stale_configs(self) -> None:
        memory_config = _stale_memory_config(MemoryPositionOptions.AFTER_AFFINE)
        memory_stack = LayerStack(
            LayerStackConfig(
                input_dim=2,
                hidden_dim=2,
                output_dim=2,
                num_layers=2,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                shared_gate_config=None,
                shared_halting_config=None,
                shared_memory_config=memory_config,
                layer_config=base_layer_config(2),
            )
        )
        for layer in memory_stack:
            set_layer_identity(layer)
        shared_memory = memory_stack[0].memory_model
        configure_weighted_memory(shared_memory)
        hidden = torch.tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)

        memory_output = memory_stack(LayerState(hidden=hidden)).hidden

        self.assertEqual(shared_memory.input_dim, 2)
        self.assertEqual(shared_memory.output_dim, 2)
        self.assertEqual(memory_config.input_dim, 7)
        self.assertEqual(memory_config.output_dim, 8)
        self.assertTrue(
            all(layer.memory_model is shared_memory for layer in memory_stack)
        )
        torch.testing.assert_close(memory_output, hidden * (1.75**2))
        memory_output.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertTrue(hidden.grad.ne(0).all())

        halting_config = _stick_breaking_config(7)
        halting_stack = LayerStack(
            LayerStackConfig(
                input_dim=2,
                hidden_dim=2,
                output_dim=2,
                num_layers=2,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                shared_gate_config=None,
                shared_halting_config=halting_config,
                shared_memory_config=None,
                layer_config=base_layer_config(2),
            )
        ).eval()
        for layer in halting_stack:
            set_layer_identity(layer)
        shared_halting = halting_stack[0].halting_model
        halting_input = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        halting_result = halting_stack(
            LayerState(hidden=halting_input, loss=torch.tensor(0.25))
        )

        self.assertEqual(shared_halting.input_dim, 2)
        self.assertEqual(halting_config.input_dim, 7)
        self.assertTrue(
            all(layer.halting_model is shared_halting for layer in halting_stack)
        )
        torch.testing.assert_close(halting_result.hidden, halting_input)
        torch.testing.assert_close(halting_result.loss, torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
