import copy
import io
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
)
from emperor.linears import LinearLayerConfig


def query_stack_config() -> LayerStackConfig:
    return LayerStackConfig(
        hidden_dim=3,
        num_layers=2,
        apply_output_postprocessing_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=ActivationOptions.GELU,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def mix_reference(initial, raw_outputs, block_size, query, norm_weight, epsilon):
    sources = [initial]
    for start in range(0, len(raw_outputs), block_size):
        sources.append(torch.stack(raw_outputs[start : start + block_size]).sum(0))
    # Put depth next to features, independently of the production layout.
    values = torch.stack(sources, dim=-2)
    keys = values * (values.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    scores = torch.einsum("...sd,...d->...s", keys * norm_weight, query)
    unnormalized_weights = (scores - scores.amax(-1, keepdim=True)).exp()
    weights = unnormalized_weights / unnormalized_weights.sum(-1, keepdim=True)
    return torch.einsum("...s,...sd->...d", weights, values)


class TestAttentionResidualQueryModel(unittest.TestCase):
    def test_query_model_preserves_initialization_and_overrides_feature_dimensions(
        self,
    ):
        for model_config in (
            LinearLayerConfig(input_dim=17, output_dim=19, bias_flag=True),
            LinearLayerConfig(bias_flag=False),
            query_stack_config(),
        ):
            with (
                self.subTest(model=type(model_config).__name__),
                torch.random.fork_rng(),
            ):
                original_config = copy.deepcopy(model_config)
                torch.manual_seed(17)
                expected_model = model_config.build(
                    overrides=type(model_config)(input_dim=2, output_dim=2)
                )
                expected_rng_state = torch.random.get_rng_state().clone()
                torch.manual_seed(17)
                residual = AttentionResidualConfig(
                    block_size=1,
                    rms_norm_epsilon=1e-6,
                    residual_dim=2,
                    model_config=model_config,
                ).build()

                self.assertIsNone(residual.query)
                self.assertNotIn("query", residual.state_dict())
                self.assertEqual(residual.query_model.input_dim, 2)
                self.assertEqual(residual.query_model.output_dim, 2)
                self.assertEqual(model_config, original_config)
                self.assertEqual(
                    tuple(residual.query_model.state_dict()),
                    tuple(expected_model.state_dict()),
                )
                for name, expected in expected_model.state_dict().items():
                    torch.testing.assert_close(
                        residual.query_model.state_dict()[name],
                        expected,
                        rtol=0,
                        atol=0,
                    )
                torch.testing.assert_close(
                    torch.random.get_rng_state(), expected_rng_state
                )

    def test_config_overrides_can_enable_the_query_model(self):
        config = AttentionResidualConfig(
            block_size=1, rms_norm_epsilon=1e-6, residual_dim=2
        )
        residual = config.build(
            overrides=AttentionResidualConfig(
                model_config=LinearLayerConfig(bias_flag=False)
            )
        )
        self.assertIsNone(config.model_config)
        self.assertIsNone(residual.query)
        self.assertEqual(
            set(residual.state_dict()), {"query_model.weight_params", "key_norm.weight"}
        )

    def test_generated_queries_match_independent_outputs_and_gradients(self):
        for shape in ((2,), (2, 2), (2, 3, 2)):
            for block_size in (1, 2, 3):
                with self.subTest(shape=shape, block_size=block_size):
                    residual = (
                        AttentionResidualConfig(
                            rms_norm_epsilon=1e-6,
                            residual_dim=2,
                            block_size=block_size,
                            model_config=LinearLayerConfig(bias_flag=True),
                        )
                        .build()
                        .double()
                    )
                    with torch.no_grad():
                        residual.query_model.weight_params.copy_(
                            torch.tensor([[0.3, -0.2], [0.1, 0.4]])
                        )
                        residual.query_model.bias_params.copy_(
                            torch.tensor([0.2, -0.15])
                        )
                        residual.key_norm.weight.copy_(torch.tensor([1.2, 0.8]))
                    generator = torch.Generator().manual_seed(42)
                    sources = [
                        torch.randn(
                            shape, generator=generator, dtype=torch.float64
                        ).requires_grad_()
                        for _ in range(4)
                    ]
                    actual_parameters = [
                        residual.query_model.weight_params,
                        residual.query_model.bias_params,
                        residual.key_norm.weight,
                    ]
                    reference_inputs = [
                        value.detach().clone().requires_grad_()
                        for value in [*sources, *actual_parameters]
                    ]
                    reference_sources = reference_inputs[:4]
                    query_weight, query_bias, norm_weight = reference_inputs[4:]
                    state = residual.new_state(sources[0])
                    actual_outputs, expected_outputs = [], []
                    for step, current in enumerate(sources[1:], start=1):
                        actual_outputs.append(
                            residual(current, -sources[0], residual_state=state)
                        )
                        query = reference_sources[step] @ query_weight + query_bias
                        expected_outputs.append(
                            mix_reference(
                                reference_sources[0],
                                reference_sources[1 : step + 1],
                                block_size,
                                query,
                                norm_weight,
                                residual.rms_norm_epsilon,
                            )
                        )
                    actual = torch.stack(actual_outputs)
                    expected = torch.stack(expected_outputs)
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
                    actual_gradients = torch.autograd.grad(
                        actual.square().sum(), [*sources, *actual_parameters]
                    )
                    reference_gradients = torch.autograd.grad(
                        expected.square().sum(), reference_inputs
                    )
                    for actual_gradient, expected_gradient in zip(
                        actual_gradients, reference_gradients, strict=True
                    ):
                        torch.testing.assert_close(
                            actual_gradient, expected_gradient, rtol=1e-9, atol=1e-10
                        )
                        self.assertGreater(
                            torch.count_nonzero(actual_gradient).item(), 0
                        )

    def test_stack_query_model_uses_fresh_layer_state_and_preserves_outer_context(self):
        residual = (
            AttentionResidualConfig(
                block_size=1,
                rms_norm_epsilon=1e-6,
                residual_dim=2,
                model_config=query_stack_config(),
            )
            .build()
            .double()
        )
        with torch.no_grad():
            for layer in residual.query_model:
                layer.model.weight_params.fill_(0.15)
                layer.model.bias_params.fill_(0.1)
        initial = torch.tensor([[0.5, -0.2], [-0.3, 0.4]], dtype=torch.float64)
        current = torch.tensor(
            [[0.2, 0.3], [-0.4, 0.1]], dtype=torch.float64, requires_grad=True
        )
        history = residual.new_state(initial)
        loss = torch.tensor(3.0)
        outer_state = LayerState(hidden=current, residual_state=history, loss=loss)
        observed_inputs = []
        handle = residual.query_model.register_forward_pre_hook(
            lambda _module, args: observed_inputs.append(
                (args[0], args[0].residual_state)
            )
        )
        try:
            actual = residual.apply_to_layer_state(outer_state, initial)
        finally:
            handle.remove()
        first_model = residual.query_model[0].model
        last_model = residual.query_model[1].model
        generated_query = F.gelu(
            current @ first_model.weight_params + first_model.bias_params
        )
        generated_query = (
            generated_query @ last_model.weight_params + last_model.bias_params
        )
        expected = mix_reference(
            initial,
            [current],
            1,
            generated_query,
            residual.key_norm.weight,
            residual.rms_norm_epsilon,
        )
        torch.testing.assert_close(actual.hidden, expected)
        self.assertIs(actual, outer_state)
        self.assertIs(actual.residual_state, history)
        self.assertIs(actual.loss, loss)
        self.assertEqual(len(observed_inputs), 1)
        self.assertIsNot(observed_inputs[0][0], outer_state)
        self.assertIsNone(observed_inputs[0][1])
        actual.hidden.square().sum().backward()
        for parameter in residual.query_model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_generated_query_supports_low_precision_and_autocast(self):
        for source_dtype, autocast_enabled in (
            (torch.float16, False),
            (torch.bfloat16, False),
            (torch.float32, True),
        ):
            with self.subTest(dtype=source_dtype, autocast=autocast_enabled):
                residual = (
                    AttentionResidualConfig(
                        block_size=1,
                        rms_norm_epsilon=1e-6,
                        residual_dim=2,
                        model_config=LinearLayerConfig(bias_flag=False),
                    )
                    .build()
                    .to(dtype=source_dtype)
                )
                with torch.no_grad():
                    residual.query_model.weight_params.copy_(
                        torch.tensor([[0.3, -0.2], [0.1, 0.4]])
                    )
                initial = torch.tensor([[1.0, -0.5], [-0.25, 0.75]], dtype=source_dtype)
                current = torch.tensor([[-0.5, 1.0], [0.25, -0.75]], dtype=source_dtype)
                with torch.autocast(
                    "cpu", dtype=torch.bfloat16, enabled=autocast_enabled
                ):
                    generated_query = current @ residual.query_model.weight_params
                    actual = residual(
                        current, initial, residual_state=residual.new_state(initial)
                    )
                expected = mix_reference(
                    initial.float(),
                    [current.float()],
                    1,
                    generated_query.float(),
                    residual.key_norm.weight.float(),
                    residual.rms_norm_epsilon,
                ).to(source_dtype)
                self.assertEqual(actual.dtype, source_dtype)
                torch.testing.assert_close(actual, expected)

    def test_query_stack_accepts_a_single_feature_vector(self):
        residual = (
            AttentionResidualConfig(
                block_size=1,
                rms_norm_epsilon=1e-6,
                residual_dim=2,
                model_config=query_stack_config(),
            )
            .build()
            .double()
        )
        initial = torch.tensor([0.2, -0.3], dtype=torch.float64)
        current = torch.tensor([-0.5, 0.7], dtype=torch.float64, requires_grad=True)
        query = residual.query_model(
            LayerState(hidden=current.unsqueeze(0))
        ).hidden.squeeze(0)
        expected = mix_reference(
            initial,
            [current],
            1,
            query,
            residual.key_norm.weight,
            residual.rms_norm_epsilon,
        )
        actual = residual(current, initial, residual_state=residual.new_state(initial))
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        self.assertEqual(actual.shape, current.shape)

    def test_partial_block_forks_generate_queries_for_their_own_current_output(self):
        residual = (
            AttentionResidualConfig(
                rms_norm_epsilon=1e-6,
                residual_dim=2,
                block_size=3,
                model_config=LinearLayerConfig(bias_flag=False),
            )
            .build()
            .double()
        )
        with torch.no_grad():
            residual.query_model.weight_params.copy_(torch.eye(2))
        initial = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
        first = torch.tensor([[0.1, 0.4]], dtype=torch.float64, requires_grad=True)
        left = torch.tensor([[0.5, -0.2]], dtype=torch.float64, requires_grad=True)
        right = torch.tensor([[-0.6, 0.3]], dtype=torch.float64, requires_grad=True)
        state = residual.new_state(initial)
        state.append(first)
        branch = state.fork()
        actual_left = residual(left, initial, residual_state=state)
        actual_right = residual(right, initial, residual_state=branch)
        for current, actual in ((left, actual_left), (right, actual_right)):
            expected = mix_reference(
                initial,
                [first, current],
                3,
                current @ residual.query_model.weight_params,
                residual.key_norm.weight,
                residual.rms_norm_epsilon,
            )
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
        self.assertFalse(torch.allclose(actual_left, actual_right))
        left_gradient, right_gradient = torch.autograd.grad(
            actual_left.sum(), (left, right), allow_unused=True
        )
        self.assertIsNone(right_gradient)
        self.assertGreater(torch.count_nonzero(left_gradient).item(), 0)

    def test_query_model_checkpoint_round_trip_and_repeated_forwards(self):
        for model_config in (LinearLayerConfig(bias_flag=True), query_stack_config()):
            with self.subTest(model=type(model_config).__name__):
                config = AttentionResidualConfig(
                    rms_norm_epsilon=1e-6,
                    residual_dim=2,
                    block_size=2,
                    model_config=model_config,
                )
                original, restored = config.build(), config.build()
                initial = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
                raw_outputs = [initial + shift for shift in (0.2, -0.4, 0.5)]
                checkpoint = io.BytesIO()
                torch.save(original.state_dict(), checkpoint)
                checkpoint.seek(0)
                restored.load_state_dict(
                    torch.load(checkpoint, weights_only=True), strict=True
                )
                for _ in range(2):
                    original_state, restored_state = (
                        original.new_state(initial),
                        restored.new_state(initial),
                    )
                    for current in raw_outputs:
                        actual = restored(
                            current, initial, residual_state=restored_state
                        )
                        expected = original(
                            current, initial, residual_state=original_state
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_query_model_config_is_rejected_before_parameter_initialization(
        self,
    ):
        invalid_models = (
            object(),
            replace(query_stack_config(), layer_config=None),
            replace(
                query_stack_config(),
                layer_config=replace(
                    query_stack_config().layer_config, layer_model_config=object()
                ),
            ),
        )
        for model_config in invalid_models:
            with self.subTest(model=model_config):
                random_state = torch.random.get_rng_state().clone()
                with self.assertRaisesRegex(
                    TypeError, "AttentionResidualConfig.model_config"
                ):
                    AttentionResidualConfig(
                        block_size=1,
                        rms_norm_epsilon=1e-6,
                        residual_dim=2,
                        model_config=model_config,
                    ).build()
                torch.testing.assert_close(torch.random.get_rng_state(), random_state)

    def test_query_model_controllers_are_rejected_without_mutating_config(self):
        for path in (
            "layer_config.gate_config",
            "layer_config.halting_config",
            "layer_config.memory_config",
            "shared_gate_config",
            "shared_halting_config",
            "shared_memory_config",
        ):
            with self.subTest(path=path):
                model_config = query_stack_config()
                target, field = (
                    (model_config.layer_config, path.split(".")[1])
                    if "." in path
                    else (model_config, path)
                )
                sentinel = object()
                setattr(target, field, sentinel)
                random_state = torch.random.get_rng_state().clone()
                with self.assertRaisesRegex(
                    ValueError, f"{path} must be None.*query model"
                ):
                    AttentionResidualConfig(
                        block_size=1,
                        rms_norm_epsilon=1e-6,
                        residual_dim=2,
                        model_config=model_config,
                    ).build()
                self.assertIs(getattr(target, field), sentinel)
                torch.testing.assert_close(torch.random.get_rng_state(), random_state)

    def test_malformed_generated_queries_are_rejected_before_history_mutation(self):
        current = torch.ones(2, 3, 2)
        for generated_query, expected_error in (
            (None, TypeError),
            (torch.ones_like(current, dtype=torch.long), TypeError),
            (torch.ones(2), ValueError),
            (torch.ones(2, 3, 1), ValueError),
            (torch.ones_like(current, device="meta"), ValueError),
        ):
            with self.subTest(query=generated_query):
                residual = AttentionResidualConfig(
                    rms_norm_epsilon=1e-6,
                    residual_dim=2,
                    block_size=2,
                    model_config=LinearLayerConfig(bias_flag=False),
                ).build()
                state = residual.new_state(torch.zeros_like(current))
                state.append(current * 0.5)
                previous_sources = state.sources
                with patch.object(
                    residual.query_model, "forward", return_value=generated_query
                ):
                    with self.assertRaisesRegex(expected_error, "query model"):
                        residual(current, current, residual_state=state)
                self.assertEqual(state._partial_count, 1)
                self.assertEqual(len(state.sources), len(previous_sources))
                for before, after in zip(previous_sources, state.sources, strict=True):
                    self.assertIs(before, after)

    def test_query_model_exception_does_not_append_a_source(self):
        residual = AttentionResidualConfig(
            block_size=1,
            rms_norm_epsilon=1e-6,
            residual_dim=2,
            model_config=LinearLayerConfig(bias_flag=False),
        ).build()
        initial = torch.zeros(1, 2)
        state = residual.new_state(initial)
        with patch.object(
            residual.query_model, "forward", side_effect=RuntimeError("query failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                residual(torch.ones_like(initial), initial, residual_state=state)
        self.assertEqual(len(state.sources), 1)
        self.assertIs(state.sources[0], initial)

    def test_invalid_current_source_is_rejected_before_query_model_execution(self):
        residual = AttentionResidualConfig(
            block_size=1,
            rms_norm_epsilon=1e-6,
            residual_dim=2,
            model_config=LinearLayerConfig(bias_flag=False),
        ).build()
        initial = torch.zeros(1, 2)
        state = residual.new_state(initial)
        with patch.object(residual.query_model, "forward") as query_forward:
            with self.assertRaisesRegex(ValueError, "last dimension"):
                residual(torch.ones(1, 3), initial, residual_state=state)
        query_forward.assert_not_called()
        self.assertEqual(len(state.sources), 1)


if __name__ == "__main__":
    unittest.main()
