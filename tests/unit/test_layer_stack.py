import unittest
from unittest.mock import patch

import pytest
import torch

from emperor.halting import (
    HaltingConfig,
    HaltingHiddenStateModeOptions,
    SoftHalting,
    SoftHaltingConfig,
    StickBreakingConfig,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    AttentionResidualConfig,
    GateConfig,
    LastLayerBiasOptions,
    Layer,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    LayerState,
    ResidualConfig,
)
from emperor.layers._composition.residual.variants.attention import AttentionResidual
from emperor.linears import LinearLayerConfig


class TestLayerStack(unittest.TestCase):
    def attention_residual_stack(
        self,
        scales: tuple[float, ...],
        *,
        block_size: int = 1,
    ) -> LayerStack:
        dim = 2
        stack = LayerStack(
            LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=dim,
                num_layers=len(scales),
                apply_output_postprocessing_flag=True,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                shared_gate_config=None,
                shared_halting_config=None,
                shared_memory_config=None,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    residual_config=AttentionResidualConfig(
                        block_size=block_size,
                        rms_norm_epsilon=1e-6,
                    ),
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                ),
            )
        )
        with torch.no_grad():
            for layer, scale in zip(stack, scales, strict=True):
                layer.model.weight_params.copy_(torch.eye(dim) * scale)
                layer.model.bias_params.zero_()
        return stack

    def test_attention_residual_stack_mixes_raw_outputs_across_depth(self):
        scales = (2.0, 3.0, 4.0)
        stack = self.attention_residual_stack(scales)

        initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        result = stack(LayerState(hidden=initial.clone()))

        first_raw_output = scales[0] * initial
        first_hidden = (initial + first_raw_output) / 2.0
        second_raw_output = scales[1] * first_hidden
        second_hidden = (initial + first_raw_output + second_raw_output) / 3.0
        third_raw_output = scales[2] * second_hidden
        expected = (
            initial + first_raw_output + second_raw_output + third_raw_output
        ) / 4.0
        torch.testing.assert_close(result.hidden, expected)
        self.assertIsNone(result.residual_state)

    def test_attention_residual_stack_uses_each_outgoing_query_in_depth_order(self):
        stack = self.attention_residual_stack((1.0, 1.0, 1.0))
        matrices = (
            torch.tensor([[1.2, -0.4], [0.3, 0.8]]),
            torch.tensor([[0.7, 0.5], [-0.6, 1.1]]),
            torch.tensor([[1.4, 0.2], [0.1, -0.9]]),
        )
        biases = (
            torch.tensor([0.2, -0.3]),
            torch.tensor([-0.1, 0.4]),
            torch.tensor([0.3, 0.1]),
        )
        queries = (
            torch.tensor([0.8, -0.3]),
            torch.tensor([-0.5, 0.9]),
            torch.tensor([0.25, 0.6]),
        )
        norm_weights = (
            torch.tensor([1.1, 0.7]),
            torch.tensor([0.6, 1.4]),
            torch.tensor([1.3, 0.8]),
        )
        with torch.no_grad():
            for layer, matrix, bias, query, norm_weight in zip(
                stack,
                matrices,
                biases,
                queries,
                norm_weights,
                strict=True,
            ):
                layer.model.weight_params.copy_(matrix)
                layer.model.bias_params.copy_(bias)
                attention_residual = layer.residual.connection
                attention_residual.query.copy_(query)
                attention_residual.key_norm.weight.copy_(norm_weight)

        def mix_sources(sources, layer):
            attention_residual = layer.residual.connection
            values = torch.stack(sources, dim=0)
            keys = torch.nn.functional.rms_norm(
                values,
                normalized_shape=(values.shape[-1],),
                weight=attention_residual.key_norm.weight,
                eps=attention_residual.rms_norm_epsilon,
            )
            logits = torch.sum(keys * attention_residual.query, dim=-1)
            weights = torch.softmax(logits, dim=0)
            return torch.sum(weights.unsqueeze(-1) * values, dim=0)

        initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
        first_raw_output = stack[0].model(initial)
        first_hidden = mix_sources((initial, first_raw_output), stack[0])
        second_raw_output = stack[1].model(first_hidden)
        second_hidden = mix_sources(
            (initial, first_raw_output, second_raw_output),
            stack[1],
        )
        third_raw_output = stack[2].model(second_hidden)
        expected = mix_sources(
            (initial, first_raw_output, second_raw_output, third_raw_output),
            stack[2],
        )

        actual = stack(LayerState(hidden=initial.clone())).hidden

        torch.testing.assert_close(actual, expected)

    def test_attention_residual_stack_supports_cpu_autocast_source_promotion(self):
        stack = self.attention_residual_stack((2.0, 3.0))
        initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            first_raw_output = stack[0].model(initial)
            first_hidden = torch.stack((initial, first_raw_output), dim=0).mean(dim=0)
            second_raw_output = stack[1].model(first_hidden)
            expected = torch.stack(
                (initial, first_raw_output, second_raw_output),
                dim=0,
            ).mean(dim=0)
            actual = stack(LayerState(hidden=initial.clone())).hidden

        self.assertEqual(first_raw_output.dtype, torch.bfloat16)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_block_attention_residual_stack_mixes_completed_and_partial_blocks(self):
        scales = (2.0, 3.0, 4.0)
        stack = self.attention_residual_stack(scales, block_size=2)
        initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        result = stack(LayerState(hidden=initial.clone()))

        first_raw_output = scales[0] * initial
        first_hidden = (initial + first_raw_output) / 2.0
        second_raw_output = scales[1] * first_hidden
        first_completed_block = first_raw_output + second_raw_output
        second_hidden = (initial + first_completed_block) / 2.0
        third_raw_output = scales[2] * second_hidden
        expected = (initial + first_completed_block + third_raw_output) / 3.0
        torch.testing.assert_close(result.hidden, expected)

    def test_attention_residual_stack_rejects_adaptive_halting(self):
        dim = 3
        config = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=2,
            stack_residual_connection_option=(AttentionResidualConfig),
            stack_dropout_probability=0.0,
            gate_enabled=False,
            halting_config=self.halting_config(dim, threshold=0.9),
        )

        with self.assertRaisesRegex(
            ValueError,
            "halting.*AttentionResidualConfig|AttentionResidualConfig.*halting",
        ):
            LayerStack(config)

    def test_attention_residual_stack_starts_fresh_history_per_forward(self):
        scales = (2.0, 3.0, 4.0)
        stack = self.attention_residual_stack(scales)
        fresh_stack = self.attention_residual_stack(scales)
        observed_histories = []
        handle = stack[0].residual.connection.register_forward_pre_hook(
            lambda _module, _args, kwargs: observed_histories.append(
                kwargs["residual_state"]
            ),
            with_kwargs=True,
        )
        try:
            stack(LayerState(hidden=torch.tensor([[1.0, -2.0]])))
            next_input = torch.tensor([[0.5, 3.0], [2.0, -1.0], [-4.0, 0.25]])
            reused_result = stack(LayerState(hidden=next_input.clone()))
        finally:
            handle.remove()
        fresh_result = fresh_stack(LayerState(hidden=next_input.clone()))

        self.assertEqual(len(observed_histories), 2)
        self.assertIsNot(observed_histories[0], observed_histories[1])
        torch.testing.assert_close(reused_result.hidden, fresh_result.hidden)

    def test_five_attention_layers_lazily_create_and_share_one_history(self):
        stack = self.attention_residual_stack((1.0, 1.0, 1.0, 1.0, 1.0))
        observations = []
        handles = []
        for layer in stack:
            handles.append(
                layer.residual.connection.register_forward_pre_hook(
                    lambda _module, args, kwargs: observations.append(
                        (args[0], kwargs["residual_state"])
                    ),
                    with_kwargs=True,
                )
            )

        state_creation_calls = []
        original_new_state = AttentionResidual.new_state

        def tracked_new_state(residual, initial_source):
            state_creation_calls.append((residual, initial_source))
            return original_new_state(residual, initial_source)

        stack_input = torch.tensor([[1.0, -2.0]])
        layer_state = LayerState(hidden=stack_input)
        try:
            with patch.object(AttentionResidual, "new_state", tracked_new_state):
                result = stack(layer_state)
        finally:
            for handle in handles:
                handle.remove()

        self.assertIs(result, layer_state)
        self.assertIsNone(result.residual_state)
        self.assertEqual(len(state_creation_calls), 1)
        self.assertIs(state_creation_calls[0][1], stack_input)
        self.assertEqual(len(observations), 5)
        shared_history = observations[0][1]
        self.assertTrue(
            all(history is shared_history for _current, history in observations)
        )
        self.assertIs(shared_history.initial_source, stack_input)
        self.assertEqual(len(shared_history.sources), 6)
        for source, (raw_output, _history) in zip(
            shared_history.sources[1:],
            observations,
            strict=True,
        ):
            self.assertIs(source, raw_output)

    def test_stateless_stack_masks_and_restores_an_enclosing_residual_state(self):
        class ResidualStateProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.observed_residual_state = object()

            def forward(self, state):
                self.observed_residual_state = state.residual_state
                return state

        stack = LayerStack(
            self.preset(
                input_dim=2,
                hidden_dim=2,
                output_dim=2,
                stack_num_layers=1,
                stack_residual_connection_option=None,
                gate_enabled=False,
                halting_config=None,
            )
        )
        probe = ResidualStateProbe()
        stack.layers = torch.nn.Sequential(probe)
        enclosing_residual_state = object()
        state = LayerState(
            hidden=torch.ones(1, 2),
            residual_state=enclosing_residual_state,
        )

        result = stack(state)

        self.assertIsNone(probe.observed_residual_state)
        self.assertIs(result.residual_state, enclosing_residual_state)

    def test_nested_stateless_stack_masks_outer_attention_history(self):
        class ResidualStateProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.observed_residual_state = object()

            def forward(self, state):
                self.observed_residual_state = state.residual_state
                return state

        outer_stack = self.attention_residual_stack((1.0, 1.0))
        outer_first, outer_second = tuple(outer_stack)
        inner_stack = LayerStack(
            self.preset(
                input_dim=2,
                hidden_dim=2,
                output_dim=2,
                stack_num_layers=1,
                stack_residual_connection_option=None,
                gate_enabled=False,
                halting_config=None,
            )
        )
        probe = ResidualStateProbe()
        inner_stack.layers = torch.nn.Sequential(probe)
        outer_stack.layers = torch.nn.Sequential(
            outer_first,
            inner_stack,
            outer_second,
        )
        outer_histories = []
        handles = [
            layer.residual.connection.register_forward_pre_hook(
                lambda _module, _args, kwargs: outer_histories.append(
                    kwargs["residual_state"]
                ),
                with_kwargs=True,
            )
            for layer in (outer_first, outer_second)
        ]

        try:
            outer_stack(LayerState(hidden=torch.ones(1, 2)))
        finally:
            for handle in handles:
                handle.remove()

        self.assertIsNone(probe.observed_residual_state)
        self.assertEqual(len(outer_histories), 2)
        self.assertIs(outer_histories[0], outer_histories[1])
        self.assertEqual(len(outer_histories[0].sources), 3)

    def test_nested_attention_stack_owns_independent_history(self):
        outer_stack = self.attention_residual_stack((1.0, 1.0))
        inner_stack = self.attention_residual_stack((1.0,))
        outer_first, outer_second = tuple(outer_stack)
        outer_stack.layers = torch.nn.Sequential(
            outer_first,
            inner_stack,
            outer_second,
        )
        outer_histories = []
        inner_histories = []
        handles = [
            outer_first.residual.connection.register_forward_pre_hook(
                lambda _module, _args, kwargs: outer_histories.append(
                    kwargs["residual_state"]
                ),
                with_kwargs=True,
            ),
            inner_stack[0].residual.connection.register_forward_pre_hook(
                lambda _module, _args, kwargs: inner_histories.append(
                    kwargs["residual_state"]
                ),
                with_kwargs=True,
            ),
            outer_second.residual.connection.register_forward_pre_hook(
                lambda _module, _args, kwargs: outer_histories.append(
                    kwargs["residual_state"]
                ),
                with_kwargs=True,
            ),
        ]
        enclosing_residual_state = object()
        layer_state = LayerState(
            hidden=torch.ones(1, 2),
            residual_state=enclosing_residual_state,
        )

        try:
            result = outer_stack(layer_state)
        finally:
            for handle in handles:
                handle.remove()

        self.assertIs(result.residual_state, enclosing_residual_state)
        self.assertEqual(len(outer_histories), 2)
        self.assertEqual(len(inner_histories), 1)
        self.assertIs(outer_histories[0], outer_histories[1])
        self.assertIsNot(outer_histories[0], inner_histories[0])
        self.assertEqual(len(outer_histories[0].sources), 3)
        self.assertEqual(len(inner_histories[0].sources), 2)

    def test_attention_residual_stack_restores_enclosing_residual_state(self):
        stack = self.attention_residual_stack((2.0, 3.0))
        initial = torch.tensor([[1.0, -2.0]])
        enclosing_residual_state = stack[0].residual.connection.new_state(initial)
        state = LayerState(
            hidden=initial.clone(),
            residual_state=enclosing_residual_state,
        )

        result = stack(state)

        self.assertIs(result, state)
        self.assertIs(result.residual_state, enclosing_residual_state)
        self.assertEqual(len(enclosing_residual_state.sources), 1)

    def test_attention_residual_stack_restores_state_when_a_layer_fails(self):
        class FailingModel(torch.nn.Module):
            def forward(self, _input):
                raise RuntimeError("deliberate layer failure")

        stack = self.attention_residual_stack((2.0, 3.0))
        stack[1].model = FailingModel()
        initial = torch.tensor([[1.0, -2.0]])
        enclosing_residual_state = stack[0].residual.connection.new_state(initial)
        state = LayerState(
            hidden=initial.clone(),
            residual_state=enclosing_residual_state,
        )

        with self.assertRaisesRegex(RuntimeError, "deliberate layer failure"):
            stack(state)

        self.assertIs(state.residual_state, enclosing_residual_state)
        self.assertEqual(len(enclosing_residual_state.sources), 1)

    def test_attention_residual_stack_owns_one_router_per_layer(self):
        stack = self.attention_residual_stack((2.0, 3.0, 4.0))

        queries = tuple(layer.residual.connection.query for layer in stack)

        self.assertEqual(len({id(query) for query in queries}), len(queries))
        for query in queries:
            torch.testing.assert_close(query, torch.zeros_like(query))

    def test_attention_residual_stack_backpropagates_through_all_sources_and_routers(
        self,
    ):
        stack = self.attention_residual_stack((1.0, 1.0, 1.0))
        matrices = (
            torch.tensor([[1.2, -0.4], [0.3, 0.8]]),
            torch.tensor([[0.7, 0.5], [-0.6, 1.1]]),
            torch.tensor([[1.4, 0.2], [0.1, -0.9]]),
        )
        biases = (
            torch.tensor([0.2, -0.3]),
            torch.tensor([-0.1, 0.4]),
            torch.tensor([0.3, 0.1]),
        )
        with torch.no_grad():
            for index, (layer, matrix, bias) in enumerate(
                zip(stack, matrices, biases, strict=True)
            ):
                layer.model.weight_params.copy_(matrix)
                layer.model.bias_params.copy_(bias)
                layer.residual.connection.query.copy_(
                    torch.tensor([0.35 + 0.1 * index, -0.2])
                )
        initial = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0]],
            requires_grad=True,
        )

        result = stack(LayerState(hidden=initial))
        result.hidden.square().sum().backward()

        self.assertIsNotNone(initial.grad)
        self.assertTrue(torch.isfinite(initial.grad).all())
        self.assertGreater(torch.count_nonzero(initial.grad).item(), 0)
        for layer in stack:
            attention_residual = layer.residual.connection
            parameters = (
                layer.model.weight_params,
                layer.model.bias_params,
                attention_residual.query,
                attention_residual.key_norm.weight,
            )
            for parameter in parameters:
                with self.subTest(parameter_shape=tuple(parameter.shape)):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_attention_residual_stack_checkpoint_contains_no_forward_history(self):
        scales = (2.0, 3.0, 4.0)
        stack = self.attention_residual_stack(scales, block_size=2)
        with torch.no_grad():
            for index, layer in enumerate(stack):
                layer.residual.connection.query.copy_(
                    torch.tensor([0.2 + 0.1 * index, -0.15])
                )
        initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]])

        expected = stack(LayerState(hidden=initial.clone())).hidden
        checkpoint = {
            name: value.detach().clone() for name, value in stack.state_dict().items()
        }
        residual_parameter_names = tuple(
            name for name in checkpoint if ".residual.connection." in name
        )
        expected_residual_parameter_names = tuple(
            parameter_name
            for layer_index in range(len(stack))
            for parameter_name in (
                f"layers.{layer_index}.residual.connection.query",
                f"layers.{layer_index}.residual.connection.key_norm.weight",
            )
        )
        restored = self.attention_residual_stack(scales, block_size=2)

        incompatible_keys = restored.load_state_dict(checkpoint, strict=True)
        actual = restored(LayerState(hidden=initial.clone())).hidden

        self.assertEqual(
            residual_parameter_names,
            expected_residual_parameter_names,
        )
        self.assertEqual(incompatible_keys.missing_keys, [])
        self.assertEqual(incompatible_keys.unexpected_keys, [])
        torch.testing.assert_close(actual, expected)

    def preset(
        self,
        input_dim: int = 12,
        hidden_dim: int = 24,
        output_dim: int = 6,
        bias_flag: bool = True,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DISABLED,
        stack_num_layers: int = 2,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_connection_option: type[ResidualConfig] | None = None,
        stack_dropout_probability: float = 0.2,
        shared_gate_config: "LayerStackConfig | GateConfig | None" = None,
        shared_halting_config: "StickBreakingConfig | None" = None,
        last_layer_bias_option: LastLayerBiasOptions = LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag: bool = True,
        gate_enabled: bool = True,
        gate_config: "LayerStackConfig | GateConfig | None" = None,
        gate_option: LayerGateOptions | None = None,
        halting_config: "StickBreakingConfig | None" = None,
    ) -> "LayerStackConfig":
        if gate_enabled and gate_config is None and shared_gate_config is None:
            gate_config = LayerStackConfig(
                hidden_dim=hidden_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=last_layer_bias_option,
                apply_output_postprocessing_flag=apply_output_postprocessing_flag,
                layer_config=LayerConfig(
                    activation=stack_activation,
                    layer_norm_position=layer_norm_position,
                    residual_config=None
                    if stack_residual_connection_option is None
                    else stack_residual_connection_option(),
                    dropout_probability=stack_dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=bias_flag,
                    ),
                ),
            )

        if (
            halting_config is None
            and shared_halting_config is None
            and stack_num_layers > 1
            and input_dim == hidden_dim == output_dim
        ):
            halting_config = StickBreakingConfig(
                threshold=0.99,
                ponder_cost_weight=1.0,
                min_steps=1,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
                halting_gate_config=LayerStackConfig(
                    hidden_dim=output_dim,
                    output_dim=2,
                    num_layers=stack_num_layers,
                    last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                    apply_output_postprocessing_flag=False,
                    layer_config=LayerConfig(
                        activation=ActivationOptions.DISABLED,
                        layer_norm_position=LayerNormPositionOptions.DISABLED,
                        residual_config=None
                        if stack_residual_connection_option is None
                        else stack_residual_connection_option(),
                        dropout_probability=stack_dropout_probability,
                        halting_config=None,
                        gate_config=None,
                        layer_model_config=LinearLayerConfig(
                            bias_flag=True,
                        ),
                    ),
                ),
            )

        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=last_layer_bias_option,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            shared_gate_config=self.layer_gate_config(shared_gate_config, gate_option),
            shared_halting_config=shared_halting_config,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=layer_norm_position,
                residual_config=None
                if stack_residual_connection_option is None
                else stack_residual_connection_option(),
                dropout_probability=stack_dropout_probability,
                gate_config=self.layer_gate_config(gate_config, gate_option),
                halting_config=halting_config,
                layer_model_config=LinearLayerConfig(
                    bias_flag=bias_flag,
                ),
            ),
        )

    def layer_gate_config(
        self,
        model_config: "LayerStackConfig | GateConfig | None",
        option: LayerGateOptions | None,
    ) -> GateConfig | None:
        if isinstance(model_config, GateConfig):
            return model_config
        if model_config is None:
            return None
        if option is None:
            option = LayerGateOptions.MULTIPLIER
        return GateConfig(
            model_config=model_config,
            option=option,
            activation=ActivationOptions.SIGMOID,
        )

    def gate_stack_config(
        self,
        dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_layers: int = 1,
        apply_output_postprocessing_flag: bool = False,
    ) -> LayerStackConfig:
        return LayerStackConfig(
            input_dim=dim,
            hidden_dim=hidden_dim if hidden_dim is not None else dim,
            output_dim=output_dim if output_dim is not None else dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=apply_output_postprocessing_flag,
            layer_config=LayerConfig(
                input_dim=dim,
                output_dim=output_dim if output_dim is not None else dim,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=dim,
                    output_dim=output_dim if output_dim is not None else dim,
                    bias_flag=True,
                ),
            ),
        )

    def halting_config(
        self,
        dim: int,
        threshold: float,
    ) -> StickBreakingConfig:
        return StickBreakingConfig(
            input_dim=dim,
            threshold=threshold,
            ponder_cost_weight=1.0,
            min_steps=1,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=2,
                num_layers=1,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_postprocessing_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=True,
                    ),
                ),
            ),
        )

    def test_init_stores_stack_owned_config_attributes(self):
        cfg = self.preset()
        stack = LayerStack(cfg)

        self.assertIsInstance(stack, LayerStack)
        self.assertEqual(stack.input_dim, cfg.input_dim)
        self.assertEqual(stack.hidden_dim, cfg.hidden_dim)
        self.assertEqual(stack.output_dim, cfg.output_dim)
        self.assertEqual(stack.num_layers, cfg.num_layers)
        self.assertEqual(
            stack.apply_output_postprocessing_flag, cfg.apply_output_postprocessing_flag
        )
        self.assertEqual(
            stack.last_layer_bias_option,
            cfg.last_layer_bias_option,
        )

        model = stack
        layers = [model] if isinstance(model, Layer) else list(model)

        for i, layer in enumerate(layers):
            is_last_layer = i == len(layers) - 1
            with self.subTest(layer_index=i, is_last_layer=is_last_layer):
                self.assertIsInstance(layer, Layer)
                self.assertIsNotNone(layer.model)
                self.assertEqual(
                    layer.output_dim,
                    cfg.output_dim if is_last_layer else cfg.hidden_dim,
                )

                if is_last_layer and not cfg.apply_output_postprocessing_flag:
                    self.assertEqual(
                        layer.postprocessing.activation_function,
                        ActivationOptions.DISABLED,
                    )
                    self.assertEqual(layer.postprocessing.dropout_probability, 0.0)
                    self.assertIsNone(layer.residual.config)
                else:
                    self.assertEqual(
                        layer.postprocessing.activation_function,
                        cfg.layer_config.activation,
                    )
                    self.assertEqual(
                        layer.postprocessing.dropout_probability,
                        cfg.layer_config.dropout_probability,
                    )

                if layer.postprocessing.gate is not None:
                    gate = layer.postprocessing.gate
                    gate_layers = [gate] if isinstance(gate, Layer) else list(gate)
                    for j, gate_layer in enumerate(gate_layers):
                        with self.subTest(gate_layer_index=j):
                            self.assertIsInstance(gate_layer, Layer)
                            self.assertIsNotNone(gate_layer.model)
                            self.assertIsNone(gate_layer.postprocessing.gate_config)
                            self.assertIsNone(gate_layer.halting.config)

    def test_build_returns_correct_type_for_num_layers(self):
        num_layers_options = [1, 2, 3, 4]
        for num_layers in num_layers_options:
            with self.subTest(num_layers=num_layers):
                cfg = self.preset(
                    input_dim=8,
                    hidden_dim=8,
                    output_dim=8,
                    stack_num_layers=num_layers,
                )
                model = LayerStack(cfg)

                self.assertIsInstance(model, LayerStack)
                for layer in model:
                    self.assertIsInstance(layer.postprocessing.gate.model, LayerStack)

    def test_layer_overrides_apply_correctly(self):
        cfg = self.preset(
            input_dim=8,
            output_dim=16,
            stack_dropout_probability=0.5,
            gate_enabled=False,
        ).layer_config
        overrides = LayerConfig(
            input_dim=12,
            output_dim=24,
        )
        layer = Layer(cfg=cfg, overrides=overrides)

        self.assertEqual(layer.input_dim, 12)
        self.assertEqual(layer.output_dim, 24)
        self.assertEqual(
            layer.postprocessing.activation_function,
            ActivationOptions.RELU,
        )
        self.assertEqual(layer.postprocessing.dropout_probability, 0.5)

    def test_stack_overrides_apply_correctly(self):
        cfg = self.preset(input_dim=8, hidden_dim=16, output_dim=4, stack_num_layers=3)
        overrides = LayerStackConfig(input_dim=12, hidden_dim=24, output_dim=6)
        stack = LayerStack(cfg, overrides)

        self.assertEqual(stack.input_dim, 12)
        self.assertEqual(stack.hidden_dim, 24)
        self.assertEqual(stack.output_dim, 6)
        self.assertEqual(stack.num_layers, 3)

    def test_validation_errors_for_base_stack_config_contract(self):
        required_fields = [
            "input_dim",
            "hidden_dim",
            "output_dim",
            "num_layers",
            "apply_output_postprocessing_flag",
            "last_layer_bias_option",
            "layer_config",
        ]

        for field_name in required_fields:
            with self.subTest(field_name=field_name):
                cfg = self.preset(gate_enabled=False)
                setattr(cfg, field_name, None)

                with self.assertRaisesRegex(ValueError, field_name):
                    LayerStack(cfg)

        wrong_type_cases = [
            ("input_dim", "8", TypeError),
            ("hidden_dim", "8", TypeError),
            ("output_dim", "8", TypeError),
            ("num_layers", "2", TypeError),
            ("apply_output_postprocessing_flag", "yes", TypeError),
            ("layer_config", object(), TypeError),
        ]
        for field_name, value, error_type in wrong_type_cases:
            with self.subTest(field_name=field_name):
                cfg = self.preset(gate_enabled=False)
                setattr(cfg, field_name, value)

                with self.assertRaisesRegex(error_type, field_name):
                    LayerStack(cfg)

        with self.assertRaisesRegex(ValueError, "num_layers"):
            LayerStack(self.preset(stack_num_layers=0, gate_enabled=False))

    def test_last_layer_bias_option_applies_correctly(self):
        num_layers_options = [1, 2, 3]
        bias_options = [
            LastLayerBiasOptions.DEFAULT,
            LastLayerBiasOptions.DISABLED,
            LastLayerBiasOptions.ENABLED,
        ]
        bias_flags = [True, False]

        for num_layers in num_layers_options:
            for bias_option in bias_options:
                for bias_flag in bias_flags:
                    message = (
                        f"num_layers={num_layers}, "
                        f"bias_option={bias_option}, "
                        f"bias_flag={bias_flag}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            stack_num_layers=num_layers,
                            bias_flag=bias_flag,
                            last_layer_bias_option=bias_option,
                        )
                        model = LayerStack(cfg)
                        layers = [model] if isinstance(model, Layer) else list(model)
                        last_layer = layers[-1]

                        match bias_option:
                            case LastLayerBiasOptions.DEFAULT:
                                if bias_flag:
                                    self.assertIsNotNone(last_layer.model.bias_params)
                                else:
                                    self.assertIsNone(last_layer.model.bias_params)
                            case LastLayerBiasOptions.DISABLED:
                                self.assertFalse(last_layer.model.bias_flag)
                            case LastLayerBiasOptions.ENABLED:
                                self.assertTrue(last_layer.model.bias_flag)

    def test_add_output_layer(self):
        num_layers_options = [1, 2, 3]
        output_dims = [6, 16]
        apply_postprocessing_flags = [True, False]

        for num_layers in num_layers_options:
            for output_dim in output_dims:
                for apply_postprocessing in apply_postprocessing_flags:
                    message = (
                        f"num_layers={num_layers}, "
                        f"output_dim={output_dim}, "
                        f"apply_postprocessing={apply_postprocessing}"
                    )
                    with self.subTest(msg=message):
                        cfg = self.preset(
                            stack_num_layers=num_layers,
                            output_dim=output_dim,
                            apply_output_postprocessing_flag=apply_postprocessing,
                        )
                        stack = LayerStack(cfg)
                        layer = stack.layers[-1]
                        expected_input_dim = (
                            cfg.hidden_dim if num_layers > 1 else cfg.input_dim
                        )
                        self.assertIsInstance(layer, Layer)
                        self.assertEqual(layer.input_dim, expected_input_dim)
                        self.assertEqual(layer.output_dim, output_dim)
                        self.assertTrue(layer.halting.is_terminal)

                        if apply_postprocessing:
                            self.assertEqual(
                                layer.postprocessing.activation_function,
                                cfg.layer_config.activation,
                            )
                            self.assertEqual(
                                layer.postprocessing.dropout_probability,
                                cfg.layer_config.dropout_probability,
                            )
                        else:
                            self.assertEqual(
                                layer.postprocessing.activation_function,
                                ActivationOptions.DISABLED,
                            )
                            self.assertEqual(
                                layer.postprocessing.dropout_probability, 0.0
                            )
                            self.assertIsNone(layer.residual.config)

    def test_gate_config_rejects_nested_gates(self):
        gate_inner = LayerStackConfig(
            input_dim=6,
            hidden_dim=6,
            output_dim=6,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=False,
            layer_config=LayerConfig(
                input_dim=6,
                output_dim=6,
                activation=ActivationOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=True),
            ),
        )
        invalid_cases = [
            (
                "gate_config",
                {
                    "gate_config": GateConfig(
                        model_config=gate_inner,
                        option=LayerGateOptions.MULTIPLIER,
                    )
                },
                {},
            ),
            (
                "shared_gate_config",
                {},
                {
                    "shared_gate_config": GateConfig(
                        model_config=gate_inner,
                        option=LayerGateOptions.MULTIPLIER,
                    )
                },
            ),
            ("halting_config", {"halting_config": HaltingConfig()}, {}),
            (
                "shared_halting_config",
                {},
                {"shared_halting_config": self.halting_config(6, threshold=0.99)},
            ),
        ]
        for invalid_field, layer_invalid, stack_invalid in invalid_cases:
            message = f"invalid_field={invalid_field}"
            with self.subTest(msg=message):
                gate_layer_config = LayerConfig(
                    input_dim=6,
                    output_dim=6,
                    activation=ActivationOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                    **layer_invalid,
                )
                gate_config = LayerStackConfig(
                    input_dim=6,
                    hidden_dim=6,
                    output_dim=6,
                    num_layers=1,
                    last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                    apply_output_postprocessing_flag=False,
                    layer_config=gate_layer_config,
                    **stack_invalid,
                )
                cfg = self.preset(gate_config=gate_config)
                with self.assertRaises(ValueError):
                    LayerStack(cfg)

    def test_shared_gate_reuses_one_module_across_stack(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_gate_config=self.gate_stack_config(dim),
        )
        model = LayerStack(cfg)
        layers = list(model)
        gated_layers = [
            layer for layer in layers if layer.postprocessing.gate is not None
        ]
        gate_models = [layer.postprocessing.gate.model for layer in gated_layers]

        self.assertEqual(len(gated_layers), len(layers))
        self.assertTrue(all(gate_model is not None for gate_model in gate_models))
        shared_gate = gated_layers[0].postprocessing.gate
        self.assertEqual(shared_gate.gate_dim, dim)
        self.assertTrue(
            all(layer.postprocessing.gate is shared_gate for layer in layers)
        )
        shared_gate_model = gate_models[0]
        self.assertTrue(
            all(gate_model is shared_gate_model for gate_model in gate_models)
        )
        self.assertEqual(
            tuple(name for name, _module in model.named_children()),
            ("shared_controllers", "layers"),
        )
        self.assertFalse(
            any(
                state_name.startswith(("shared_controllers.", "topology."))
                for state_name in model.state_dict()
            )
        )
        for layer in gated_layers:
            self.assertIsNone(layer.cfg.gate_config)
            self.assertIs(layer.postprocessing.gate_config, cfg.shared_gate_config)

    def test_unshared_gate_builds_separate_modules_per_layer(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
        )
        model = LayerStack(cfg)
        layers = list(model)
        gated_layers = [
            layer for layer in layers if layer.postprocessing.gate is not None
        ]
        gate_models = [layer.postprocessing.gate.model for layer in gated_layers]

        self.assertLess(len(gated_layers), len(layers))
        self.assertTrue(all(gate_model is not None for gate_model in gate_models))
        self.assertEqual(
            len({id(gate_model) for gate_model in gate_models}),
            len(gated_layers),
        )

    def test_stack_created_layers_inherit_gate_option(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            gate_option=LayerGateOptions.MULTIPLIER,
        )

        model = LayerStack(cfg)

        for index, layer in enumerate(model):
            with self.subTest(layer_index=index):
                if layer.input_dim != layer.output_dim:
                    self.assertIsNone(layer.postprocessing.gate)
                else:
                    self.assertIsNotNone(layer.postprocessing.gate)
                    self.assertEqual(
                        layer.postprocessing.gate.option, LayerGateOptions.MULTIPLIER
                    )

    def test_shared_gate_model_works_with_non_default_gate_option(self):
        dim = 4
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=2,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_gate_config=self.gate_stack_config(dim),
            gate_option=LayerGateOptions.MULTIPLIER,
        )
        model = LayerStack(cfg)
        for layer in model:
            with torch.no_grad():
                layer.model.weight_params.zero_()
                layer.model.weight_params[:dim, :dim].copy_(torch.eye(dim))
                layer.model.bias_params.zero_()
        gated_layers = [
            layer for layer in model if layer.postprocessing.gate is not None
        ]
        shared_gate_model = gated_layers[0].postprocessing.gate.model
        with torch.no_grad():
            shared_gate_model[0].model.weight_params.zero_()
            shared_gate_model[0].model.bias_params.zero_()
        x = torch.ones(2, dim + 1)

        result = model(LayerState(hidden=x.clone()))

        self.assertTrue(
            all(
                layer.postprocessing.gate.model is shared_gate_model
                for layer in gated_layers
            )
        )
        self.assertEqual(result.hidden.shape, (2, dim))

    def test_shared_gate_rejects_invalid_config_type(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_gate_config=object(),
        )

        with self.assertRaises(TypeError):
            LayerStack(cfg)

    def test_shared_gate_rejects_per_layer_gate_config(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_gate_config=self.gate_stack_config(dim),
            gate_config=self.gate_stack_config(dim),
        )

        with self.assertRaises(ValueError):
            LayerStack(cfg)

    def test_shared_gate_rejects_hidden_output_mismatch(self):
        cfg = self.preset(
            input_dim=5,
            hidden_dim=8,
            output_dim=6,
            stack_num_layers=3,
            shared_gate_config=self.gate_stack_config(6),
        )

        with self.assertRaisesRegex(ValueError, "hidden_dim and output_dim"):
            LayerStack(cfg)

    def test_shared_gate_rejects_single_layer_hidden_output_mismatch(self):
        cfg = self.preset(
            input_dim=5,
            hidden_dim=8,
            output_dim=6,
            stack_num_layers=1,
            shared_gate_config=self.gate_stack_config(6),
        )

        with self.assertRaisesRegex(ValueError, "hidden_dim and output_dim"):
            LayerStack(cfg)

    def test_shared_gate_stack_forward_pass(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_gate_config=self.gate_stack_config(dim),
        )
        model = LayerStack(cfg)
        model.eval()

        state = model(LayerState(hidden=torch.randn(batch_size, dim + 1)))
        gated_layers = [
            layer for layer in model if layer.postprocessing.gate is not None
        ]
        shared_gate_model = gated_layers[0].postprocessing.gate.model

        self.assertEqual(state.hidden.shape, (batch_size, dim))
        self.assertTrue(
            all(
                layer.postprocessing.gate.model is shared_gate_model
                for layer in gated_layers
            )
        )

    @pytest.mark.training
    def test_shared_gate_receives_shared_gradients(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_gate_config=self.gate_stack_config(dim),
        )
        model = LayerStack(cfg)
        layers = list(model)
        gated_layers = [
            layer for layer in layers if layer.postprocessing.gate is not None
        ]
        shared_gate_model = gated_layers[0].postprocessing.gate.model
        before = [
            parameter.detach().clone()
            for parameter in shared_gate_model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        state = model(LayerState(hidden=torch.randn(batch_size, dim + 1)))
        optimizer.zero_grad()
        state.hidden.sum().backward()
        optimizer.step()

        self.assertTrue(
            all(
                layer.postprocessing.gate.model is shared_gate_model
                for layer in gated_layers
            )
        )
        after = [
            parameter.detach()
            for parameter in shared_gate_model.parameters()
            if parameter.requires_grad
        ]
        changed = any(
            not torch.equal(before_parameter, after_parameter)
            for before_parameter, after_parameter in zip(before, after, strict=True)
        )
        self.assertTrue(changed)

    def test_unshared_stack_layers_receive_gradients(self):
        cfg = self.preset(
            input_dim=4,
            hidden_dim=5,
            output_dim=6,
            stack_num_layers=3,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            gate_enabled=False,
            halting_config=None,
            shared_halting_config=None,
        )
        model = LayerStack(cfg)
        hidden = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [0.0, 1.0, 2.0, -0.5],
                [2.0, -1.0, 0.5, 1.0],
                [1.5, 0.0, -0.5, 0.5],
            ],
            requires_grad=True,
        )

        state = model(LayerState(hidden=hidden))
        state.hidden.sum().backward()

        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.any(hidden.grad.abs() > 0))
        for index, layer in enumerate(model):
            with self.subTest(layer_index=index):
                gradients = [
                    parameter.grad
                    for parameter in layer.model.parameters()
                    if parameter.requires_grad
                ]
                nonzero_gradients = [
                    gradient
                    for gradient in gradients
                    if gradient is not None and torch.any(gradient.abs() > 0)
                ]
                self.assertTrue(len(nonzero_gradients) > 0)

    def test_output_postprocessing_flag_false_does_not_disable_gates(self):
        dim = 8
        per_layer_cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=3,
            apply_output_postprocessing_flag=False,
        )
        shared_cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=3,
            apply_output_postprocessing_flag=False,
            shared_gate_config=self.gate_stack_config(dim),
        )
        no_gate_cfg = self.preset(
            input_dim=dim + 1,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=3,
            apply_output_postprocessing_flag=False,
            gate_enabled=False,
        )

        per_layer_model = LayerStack(per_layer_cfg)
        shared_model = LayerStack(shared_cfg)
        no_gate_model = LayerStack(no_gate_cfg)

        self.assertIsNotNone(per_layer_model[-1].postprocessing.gate)
        self.assertIsNotNone(shared_model[-1].postprocessing.gate)
        self.assertIsNotNone(shared_model[0].postprocessing.gate)
        self.assertIs(
            shared_model[0].postprocessing.gate, shared_model[1].postprocessing.gate
        )
        self.assertIs(
            shared_model[-1].postprocessing.gate, shared_model[1].postprocessing.gate
        )
        self.assertIsNone(no_gate_model[-1].postprocessing.gate)

    def test_halting_rejects_single_layer_and_mismatched_dims(self):
        dim = 8
        halting_config = StickBreakingConfig(
            input_dim=dim,
            threshold=0.99,
            ponder_cost_weight=1.0,
            min_steps=1,
            dropout_probability=0.0,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=LayerStackConfig(
                input_dim=dim,
                hidden_dim=dim,
                output_dim=2,
                num_layers=2,
                last_layer_bias_option=LastLayerBiasOptions.DISABLED,
                apply_output_postprocessing_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        input_dim=dim,
                        output_dim=dim,
                        bias_flag=True,
                    ),
                ),
            ),
        )

        invalid_cases = [
            {
                "stack_num_layers": 1,
                "input_dim": dim,
                "hidden_dim": dim,
                "output_dim": dim,
            },
            {
                "stack_num_layers": 2,
                "input_dim": dim,
                "hidden_dim": dim * 2,
                "output_dim": dim,
            },
            {
                "stack_num_layers": 2,
                "input_dim": dim,
                "hidden_dim": dim,
                "output_dim": dim * 2,
            },
        ]

        for case in invalid_cases:
            message = ", ".join(f"{k}={v}" for k, v in case.items())
            with self.subTest(msg=message):
                with self.assertRaises(ValueError):
                    LayerStack(self.preset(halting_config=halting_config, **case))

    def test_shared_halting_rejects_per_layer_halting_config(self):
        dim = 8
        shared_halting_config = self.halting_config(dim, threshold=0.99)
        per_layer_halting_config = self.halting_config(dim, threshold=0.99)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_halting_config=shared_halting_config,
            halting_config=per_layer_halting_config,
        )

        with self.assertRaises(ValueError):
            LayerStack(cfg)

    def test_shared_halting_rejects_invalid_config_type(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_halting_config=object(),
        )

        with self.assertRaises(TypeError):
            LayerStack(cfg)

    def test_shared_halting_rejects_single_layer_and_mismatched_dims(self):
        dim = 8
        shared_halting_config = self.halting_config(dim, threshold=0.99)
        invalid_cases = [
            {
                "stack_num_layers": 1,
                "input_dim": dim,
                "hidden_dim": dim,
                "output_dim": dim,
            },
            {
                "stack_num_layers": 2,
                "input_dim": dim,
                "hidden_dim": dim * 2,
                "output_dim": dim,
            },
            {
                "stack_num_layers": 2,
                "input_dim": dim,
                "hidden_dim": dim,
                "output_dim": dim * 2,
            },
        ]

        for case in invalid_cases:
            message = ", ".join(f"{k}={v}" for k, v in case.items())
            with self.subTest(msg=message):
                cfg = self.preset(
                    shared_halting_config=shared_halting_config,
                    **case,
                )
                with self.assertRaises(ValueError):
                    LayerStack(cfg)

    def test_shared_halting_reuses_one_module_across_stack(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            shared_halting_config=self.halting_config(dim, threshold=0.99),
        )
        model = LayerStack(cfg)
        layers = [model] if isinstance(model, Layer) else list(model)
        halting_models = [layer.halting.model for layer in layers]

        self.assertTrue(
            all(halting_model is not None for halting_model in halting_models)
        )
        first_halting_model = halting_models[0]
        self.assertTrue(
            all(
                halting_model is first_halting_model for halting_model in halting_models
            )
        )
        for layer in layers:
            self.assertIsNone(layer.cfg.halting_config)
            self.assertIsNone(layer.halting.config)

    def test_unshared_halting_builds_separate_modules_per_layer(self):
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            halting_config=self.halting_config(dim, threshold=0.99),
        )
        model = LayerStack(cfg)
        layers = [model] if isinstance(model, Layer) else list(model)
        halting_models = [layer.halting.model for layer in layers]

        self.assertTrue(
            all(halting_model is not None for halting_model in halting_models)
        )
        self.assertEqual(
            len({id(halting_model) for halting_model in halting_models}), len(layers)
        )

    def test_shared_layer_stack_accepts_soft_halting_interface(self):
        dim = 4
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=3,
            gate_enabled=False,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_halting_config=SoftHaltingConfig(
                input_dim=dim,
                threshold=0.999,
                ponder_cost_weight=1.0,
                min_steps=1,
                dropout_probability=0.0,
                hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            ),
        )
        model = LayerStack(cfg)
        layers = [model] if isinstance(model, Layer) else list(model)
        halting_models = [layer.halting.model for layer in layers]

        self.assertTrue(
            all(
                isinstance(halting_model, SoftHalting)
                for halting_model in halting_models
            )
        )
        self.assertTrue(
            all(halting_model is halting_models[0] for halting_model in halting_models)
        )

    def test_shared_halting_stack_forward_pass(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_halting_config=self.halting_config(dim, threshold=1.0),
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertEqual(state.hidden.shape, (batch_size, dim))
        self.assertIsNotNone(state.halting_state)
        self.assertIsNotNone(state.loss)

    @pytest.mark.training
    def test_shared_halting_remains_shared_after_training_step(self):
        batch_size = 4
        dim = 8
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=4,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            shared_halting_config=self.halting_config(dim, threshold=1.0),
        )
        model = LayerStack(cfg)
        model.eval()
        layers = [model] if isinstance(model, Layer) else list(model)
        shared_halting_model = layers[0].halting.model
        before = [
            parameter.detach().clone()
            for parameter in shared_halting_model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)
        loss = state.hidden.sum() + state.loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertTrue(
            all(layer.halting.model is shared_halting_model for layer in layers)
        )
        after = [
            parameter.detach()
            for parameter in shared_halting_model.parameters()
            if parameter.requires_grad
        ]
        changed = any(
            not torch.equal(before_parameter, after_parameter)
            for before_parameter, after_parameter in zip(before, after, strict=True)
        )
        self.assertTrue(changed)

    def test_halting_stack_finalizes_early_and_skips_remaining_layers(self):
        batch_size = 4
        dim = 8
        num_layers = 3
        halting_config = self.halting_config(dim, threshold=1e-9)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=num_layers,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            halting_config=halting_config,
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertTrue(state.halting_state.halt_mask.all().item())
        self.assertEqual(state.halting_state.step_count, 0)
        self.assertIsNotNone(state.loss)

    def test_halting_stack_finalizes_at_last_layer_when_not_all_halted(self):
        batch_size = 4
        dim = 8
        num_layers = 10
        halting_config = self.halting_config(dim, threshold=1.0)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=num_layers,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            halting_config=halting_config,
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertFalse(state.halting_state.halt_mask.all().item())
        self.assertEqual(state.halting_state.step_count, num_layers - 1)
        self.assertIsNotNone(state.loss)

    def test_halting_stack_with_ten_layers_finalizes_between_first_and_last_layer(self):
        batch_size = 4
        dim = 8
        num_layers = 10
        halting_config = self.halting_config(dim, threshold=0.7)
        cfg = self.preset(
            input_dim=dim,
            hidden_dim=dim,
            output_dim=dim,
            stack_num_layers=num_layers,
            stack_activation=ActivationOptions.DISABLED,
            stack_dropout_probability=0.0,
            halting_config=halting_config,
        )
        model = LayerStack(cfg)
        model.eval()

        input = LayerState(hidden=torch.randn(batch_size, dim))
        state = model(input)

        self.assertTrue(state.halting_state.halt_mask.all().item())
        self.assertGreater(state.halting_state.step_count, 0)
        self.assertLess(state.halting_state.step_count, num_layers - 1)
        self.assertIsNotNone(state.loss)

    def test_build_forward_pass_output_shape(self):
        batch_size = 4
        num_layers_options = [1, 2, 3]
        input_dims = [6, 12]
        hidden_dims = [6, 24]
        output_dims = [6, 8]
        activations = [ActivationOptions.RELU, ActivationOptions.DISABLED]
        residual_options = [
            AdditiveResidualConfig,
            None,
        ]
        dropout_probabilities = [0.0, 0.2]
        layer_norm_positions = [
            LayerNormPositionOptions.DISABLED,
            LayerNormPositionOptions.DEFAULT,
            LayerNormPositionOptions.BEFORE,
            LayerNormPositionOptions.AFTER,
        ]

        for num_layers in num_layers_options:
            for input_dim in input_dims:
                for hidden_dim in hidden_dims:
                    for output_dim in output_dims:
                        for activation in activations:
                            for residual_option in residual_options:
                                for dropout in dropout_probabilities:
                                    for layer_norm in layer_norm_positions:
                                        message = (
                                            f"num_layers={num_layers}, "
                                            f"input_dim={input_dim}, "
                                            f"output_dim={output_dim}, "
                                            f"activation={activation}, "
                                            f"residual_connection_option={residual_option}, "
                                            f"dropout={dropout}, "
                                            f"layer_norm={layer_norm}"
                                        )
                                        with self.subTest(msg=message):
                                            cfg = self.preset(
                                                stack_num_layers=num_layers,
                                                input_dim=input_dim,
                                                hidden_dim=hidden_dim,
                                                output_dim=output_dim,
                                                stack_activation=activation,
                                                stack_residual_connection_option=residual_option,
                                                stack_dropout_probability=dropout,
                                                layer_norm_position=layer_norm,
                                            )
                                            model = LayerStack(cfg)
                                            x = torch.randn(batch_size, input_dim)
                                            state = LayerState(hidden=x)
                                            output_state = model(state)
                                            expected_shape = (batch_size, output_dim)

                                            self.assertEqual(
                                                output_state.hidden.shape,
                                                expected_shape,
                                            )

                                            layers = (
                                                [model]
                                                if isinstance(model, Layer)
                                                else list(model)
                                            )
                                            for layer in layers:
                                                if layer.input_dim != layer.output_dim:
                                                    self.assertIsNone(
                                                        layer.residual.config
                                                    )
