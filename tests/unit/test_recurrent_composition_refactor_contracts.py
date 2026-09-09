"""Behavioral contracts for the shared execution and Inner Thinking refactor."""

import unittest
from dataclasses import dataclass, replace

import torch

from emperor.attention import AttentionLayerState
from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    HierarchicalReasoningModelRecurrentConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerState,
    RecurrentLayerConfig,
    TinyRecursiveModelRecurrentConfig,
    WeightedResidualConfig,
)
from emperor.nn import Module
from emperor.transformer import (
    TransformerDecoderBlockLayerConfig,
    TransformerDecoderLayerState,
)
from support.layers import _set_affine_parameters, linear_stack_config
from unit.test_recurrent_composition_inner_thinking import (
    RecordingBlockConfig,
    config,
    set_router,
)
from unit.test_recurrent_layer import FailingStochasticStateBlockConfig


@dataclass
class StochasticScoreConfig(ConfigBase):
    input_dim: int | None = optional_field("Input width.")
    output_dim: int | None = optional_field("Score width.")

    def _registry_owner(self):
        return StochasticScore


class StochasticScore(Module):
    def __init__(self, cfg, overrides=None):
        super().__init__()
        cfg = self._override_config(cfg, overrides)
        self.weight = torch.nn.Parameter(torch.ones(cfg.input_dim, cfg.output_dim))
        self.register_buffer("calls", torch.zeros(()))

    def forward(self, state):
        self.calls.add_(1)
        scores = state.hidden @ self.weight
        return replace(state, hidden=scores + torch.rand_like(scores))


def _thinking_oracle(hidden, parameters, depth, gradient_count=None, controllers=False):
    """Direct equations with independent adjacent-depth gradient windows."""
    prefix = 0 if gradient_count is None else max(0, depth - gradient_count)
    for step in range(depth):
        if prefix and step == prefix:
            hidden = hidden.detach()
        with torch.set_grad_enabled(step >= prefix):
            previous = hidden
            if step == 0:
                candidate = hidden * parameters["block_model.scale"]
                candidate = candidate * parameters["thinking_step_weights"][step]
            else:
                router = f"samplers.{step - 1}.router.model.layers.0.model"
                logits = (
                    hidden @ parameters[f"{router}.weight_params"]
                    + parameters[f"{router}.bias_params"]
                ).squeeze(-1)
                indices = logits.topk(hidden.shape[-2] // 2, dim=-1).indices
                indices = indices.sort(dim=-1).values
                feature_indices = indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
                selected = hidden.gather(-2, feature_indices)
                update = selected * parameters["block_model.scale"]
                update = update * logits.sigmoid().gather(-1, indices).unsqueeze(-1)
                update = update * parameters["thinking_step_weights"][step] * 0.3
                candidate = hidden.scatter_add(-2, feature_indices, update)
            if controllers:
                gate = "recurrent_gate.model.layers.0.model"
                candidate = candidate + (
                    candidate @ parameters[f"{gate}.weight_params"]
                    + parameters[f"{gate}.bias_params"]
                )
                candidate = (
                    previous
                    + parameters["residual_connection.raw_weight"].tanh() * candidate
                )
            hidden = candidate
    return hidden


class TestRecurrentRefactorContracts(unittest.TestCase):
    def assert_same_gradients(self, actual, expected, actual_inputs, expected_inputs):
        actual_gradients = torch.autograd.grad(
            actual.sum(), actual_inputs, allow_unused=True
        )
        expected_gradients = torch.autograd.grad(
            expected.sum(), expected_inputs, allow_unused=True
        )
        for index, (actual_gradient, expected_gradient) in enumerate(
            zip(actual_gradients, expected_gradients, strict=True)
        ):
            with self.subTest(gradient=index):
                if expected_gradient is None:
                    self.assertIsNone(actual_gradient)
                else:
                    self.assertIsNotNone(actual_gradient)
                    torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_inner_thinking_build_preserves_component_initialization_order(self):
        cfg = config(block_config=linear_stack_config(2))
        torch.manual_seed(187)
        model = cfg.build()
        actual_rng = torch.get_rng_state()
        torch.manual_seed(187)
        block = replace(cfg.block_config, input_dim=2, output_dim=2).build()
        samplers = [cfg.sampler_config.build_with_router_input_dim(2) for _ in range(2)]
        self.assertTrue(torch.equal(actual_rng, torch.get_rng_state()))
        for actual, expected in zip(
            [model.block_model, *model.samplers], [block, *samplers], strict=True
        ):
            torch.testing.assert_close(actual.state_dict(), expected.state_dict())
        self.assertIsNone(cfg.sampler_config.input_dim)
        self.assertIsNone(cfg.sampler_config.router_config.input_dim)
        torch.testing.assert_close(model.thinking_step_weights, torch.ones(3, 2))

    def test_inner_thinking_smooth_outputs_losses_and_all_gradients_match_oracles(self):
        for gradient_count in (2, None):
            for progress, weight in ((3, 0.0), (4, 0.5), (5, 1.0)):
                with self.subTest(gradient_count=gradient_count, weight=weight):
                    controls = (
                        {"no_gradient_transition_count": 0}
                        if gradient_count is None
                        else {"gradient_transition_count": gradient_count}
                    )
                    model = (
                        config(
                            initial_iterations=2,
                            thinking_step_scale=0.3,
                            forward_calls_before_iteration_increment=4,
                            smooth_iteration_growth_flag=True,
                            **controls,
                        )
                        .build()
                        .double()
                    )
                    set_router(model, 0, 0.4)
                    set_router(model, 1, -0.2)
                    model.recurrent_iteration_schedule.forward_call_progress.fill_(
                        progress
                    )
                    parameters = dict(model.named_parameters())
                    oracle_parameters = {
                        name: value.detach().clone().requires_grad_()
                        for name, value in parameters.items()
                    }
                    hidden = torch.tensor(
                        [[[4.0, 1.0], [1.0, 2.0], [3.0, 3.0], [2.0, 4.0]]],
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    oracle_hidden = hidden.detach().clone().requires_grad_()
                    caller_loss = hidden.new_tensor(2.0, requires_grad=True)
                    result = model(LayerState(hidden, loss=caller_loss))
                    source = _thinking_oracle(
                        oracle_hidden, oracle_parameters, 2, gradient_count
                    )
                    target = _thinking_oracle(
                        oracle_hidden, oracle_parameters, 3, gradient_count
                    )
                    expected = (
                        source
                        if weight == 0
                        else target
                        if weight == 1
                        else (1 - weight) * source + weight * target
                    )
                    torch.testing.assert_close(result.hidden, expected)
                    torch.testing.assert_close(
                        result.loss, caller_loss + 0.25 * (2 + weight)
                    )
                    torch.testing.assert_close(
                        torch.autograd.grad(result.loss, caller_loss)[0],
                        torch.ones_like(caller_loss),
                    )
                    self.assert_same_gradients(
                        result.hidden,
                        expected,
                        (hidden, *parameters.values()),
                        (oracle_hidden, *oracle_parameters.values()),
                    )

    def test_outer_gate_and_residual_transform_selected_and_skipped_tokens(self):
        model = (
            config(
                max_steps=2,
                initial_iterations=2,
                thinking_step_scale=0.3,
                gate_config=GateConfig(
                    option=LayerGateOptions.ADDITION,
                    model_config=linear_stack_config(2, bias_flag=True),
                ),
                residual_config=WeightedResidualConfig(),
            )
            .build()
            .double()
        )
        set_router(model, 0, 0.4)
        _set_affine_parameters(
            model.recurrent_gate.model, torch.eye(2) * 0.1, torch.ones(2) * 0.2
        )
        with torch.no_grad():
            model.residual_connection.raw_weight.fill_(0.5)
            model.block_model.scale.fill_(0.75)
        parameters = dict(model.named_parameters())
        oracle_parameters = {
            name: value.detach().clone().requires_grad_()
            for name, value in parameters.items()
        }
        hidden = torch.tensor(
            [[[4.0, 1.0], [1.0, 2.0], [3.0, 3.0], [2.0, 4.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        oracle_hidden = hidden.detach().clone().requires_grad_()
        actual = model(LayerState(hidden)).hidden
        expected = _thinking_oracle(
            oracle_hidden, oracle_parameters, 2, controllers=True
        )
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.equal(actual[:, [1, 3]], hidden[:, [1, 3]]))
        self.assert_same_gradients(
            actual,
            expected,
            (hidden, *parameters.values()),
            (oracle_hidden, *oracle_parameters.values()),
        )

    def test_inner_thinking_checkpoint_preserves_names_aliases_and_next_handoff(self):
        for controls in (
            {"gradient_transition_count": 2},
            {"no_gradient_transition_count": 0},
        ):
            with self.subTest(controls=controls):
                cfg = config(
                    initial_iterations=2,
                    smooth_iteration_growth_flag=True,
                    forward_calls_before_iteration_increment=4,
                    **controls,
                )
                original = cfg.build().double()
                set_router(original, 0, 0.7)
                set_router(original, 1, -0.4)
                original.recurrent_iteration_schedule.forward_call_progress.fill_(4)
                expected_keys = [
                    "thinking_step_weights",
                    "recurrent_iteration_schedule.forward_call_progress",
                    "block_model.scale",
                ]
                expected_keys.extend(
                    f"samplers.{step}.router.model.layers.0.model.{parameter}"
                    for step in range(2)
                    for parameter in ("weight_params", "bias_params")
                )
                self.assertEqual(list(original.state_dict()), expected_keys)
                restored = cfg.build().double()
                identities = tuple(restored.parameters())
                restored.load_state_dict(original.state_dict(), strict=True)
                self.assertTrue(
                    all(
                        first is second
                        for first, second in zip(
                            identities, restored.parameters(), strict=True
                        )
                    )
                )
                self.assertEqual(
                    len(
                        {
                            parameter.data_ptr()
                            for parameter in restored.samplers.parameters()
                        }
                    ),
                    len(tuple(restored.samplers.parameters())),
                )
                hidden = torch.randn(2, 4, 2, dtype=torch.float64)
                actual = restored(LayerState(hidden, loss=hidden.new_tensor(2.0)))
                expected = original(LayerState(hidden, loss=hidden.new_tensor(2.0)))
                torch.testing.assert_close(actual.hidden, expected.hidden)
                torch.testing.assert_close(actual.loss, expected.loss)
                torch.testing.assert_close(restored.state_dict(), original.state_dict())

    def test_failed_inner_thinking_handoff_restores_buffers_rng_and_caller_state(self):
        for controls in (
            {"gradient_transition_count": 2},
            {"no_gradient_transition_count": 0},
        ):
            with self.subTest(controls=controls):
                sampler_config = config().sampler_config
                sampler_config = replace(
                    sampler_config,
                    router_config=replace(
                        sampler_config.router_config,
                        model_config=StochasticScoreConfig(),
                    ),
                )
                model = config(
                    initial_iterations=2,
                    smooth_iteration_growth_flag=True,
                    forward_calls_before_iteration_increment=4,
                    block_config=FailingStochasticStateBlockConfig(
                        fail_on_transition_step=3
                    ),
                    sampler_config=sampler_config,
                    **controls,
                ).build()
                model.recurrent_iteration_schedule.forward_call_progress.fill_(4)
                buffer = model.block_model.transition_step
                buffer_identities = dict(model.named_buffers())
                before = {name: value.clone() for name, value in model.named_buffers()}
                hidden, loss = torch.ones(1, 4, 2), torch.tensor(2.0)
                state = LayerState(hidden, loss=loss)
                rng = torch.get_rng_state()
                with self.assertRaisesRegex(RuntimeError, "target transition failed"):
                    model(state)
                self.assertIs(state.hidden, hidden)
                self.assertIs(state.loss, loss)
                self.assertIs(model.block_model.transition_step, buffer)
                torch.testing.assert_close(dict(model.named_buffers()), before)
                for name, value in model.named_buffers():
                    self.assertIs(value, buffer_identities[name])
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_real_decoder_receives_selected_masks_and_preserves_encoder_context(self):
        from unit.test_transformer_layer import TestTransformerDecoderLayer

        decoder = TestTransformerDecoderLayer().preset(
            embedding_dim=4,
            batch_size=2,
            num_heads=2,
            target_sequence_length=4,
            source_sequence_length=3,
            query_key_projection_dim=4,
            value_projection_dim=4,
            feed_forward_hidden_dim=4,
        )
        decoder.self_attention_config.batch_first_flag = True
        decoder.cross_attention_config.batch_first_flag = True
        block = TransformerDecoderBlockLayerConfig(
            input_dim=4,
            output_dim=4,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            dropout_probability=0.0,
            layer_model_config=decoder,
        )
        model = config(
            input_dim=4,
            output_dim=4,
            max_steps=2,
            initial_iterations=2,
            block_config=block,
        ).build()
        hidden = torch.randn(2, 4, 4, requires_grad=True)
        encoder = torch.randn(2, 3, 4, requires_grad=True)
        target_mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
        cross_mask = torch.tensor(
            [
                [False, False, True],
                [False, True, False],
                [False, False, False],
                [False, True, True],
            ]
        )
        padding = torch.tensor(
            [[False, False, False, True], [False, False, True, False]]
        )
        state = TransformerDecoderLayerState(
            hidden=hidden,
            encoder_output=encoder,
            target_attention_mask=target_mask,
            target_key_padding_mask=padding,
            cross_attention_mask=cross_mask,
            encoder_padding_mask=torch.zeros(2, 3, dtype=torch.bool),
            controller_state=object(),
        )
        block_inputs, selections = [], []
        block_hook = model.block_model.register_forward_pre_hook(
            lambda _, args: block_inputs.append(replace(args[0]))
        )
        sampler_hook = model.samplers[0].register_forward_hook(
            lambda _, args, result: selections.append(result.indices)
        )
        try:
            result = model(state)
        finally:
            block_hook.remove()
            sampler_hook.remove()
        indices = selections[0]
        selected = block_inputs[1]
        expected_self = torch.stack(
            [target_mask[item][:, item] for item in indices]
        ).repeat_interleave(2, dim=0)
        expected_cross = torch.stack(
            [cross_mask[item] for item in indices]
        ).repeat_interleave(2, dim=0)
        torch.testing.assert_close(selected.target_attention_mask, expected_self)
        torch.testing.assert_close(selected.cross_attention_mask, expected_cross)
        torch.testing.assert_close(
            selected.target_key_padding_mask, padding.gather(1, indices)
        )
        self.assertIs(selected.encoder_output, encoder)
        self.assertIs(selected.encoder_padding_mask, state.encoder_padding_mask)
        self.assertIs(selected.controller_state, state.controller_state)
        self.assertIs(result.target_attention_mask, target_mask)
        self.assertIs(result.cross_attention_mask, cross_mask)
        result.hidden.square().sum().backward()
        for gradient in (hidden.grad, encoder.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_noncontiguous_tokens_match_contiguous_outputs_and_gradients(self):
        model = config().build().double()
        hidden = (
            torch.randn(2, 2, 4, dtype=torch.float64).transpose(1, 2).requires_grad_()
        )
        contiguous = hidden.detach().contiguous().requires_grad_()
        self.assertFalse(hidden.is_contiguous())
        actual = model(LayerState(hidden)).hidden
        expected = model(LayerState(contiguous)).hidden
        torch.testing.assert_close(actual, expected)
        self.assert_same_gradients(actual, expected, (hidden,), (contiguous,))

    def test_higher_rank_attention_masks_keep_the_existing_rejection(self):
        model = config().build()
        hidden = torch.ones(2, 3, 4, 2)
        state = AttentionLayerState(hidden=hidden, attention_mask=torch.zeros(4, 4))
        with self.assertRaisesRegex(ValueError, "Attention masks require"):
            model(state)
        self.assertIs(state.hidden, hidden)
        self.assertEqual(
            model.recurrent_iteration_schedule.forward_call_progress.item(), 0
        )

    def test_recurrent_validation_preserves_first_failure_across_variants(self):
        shared = dict(
            input_dim=2,
            output_dim=2,
            initial_iterations=1,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
        )
        configs = (
            RecurrentLayerConfig(
                **shared, max_steps=2, block_config=RecordingBlockConfig()
            ),
            TinyRecursiveModelRecurrentConfig(
                **shared,
                answer_update_count=2,
                latent_updates_per_answer_update=1,
                block_config=RecordingBlockConfig(),
                initialization_standard_deviation=0.0,
            ),
            HierarchicalReasoningModelRecurrentConfig(
                **shared,
                high_cycles=2,
                low_cycles=1,
                high_block_config=RecordingBlockConfig(),
                low_block_config=RecordingBlockConfig(),
                initialization_standard_deviation=0.0,
            ),
        )
        for cfg in configs:
            cases = (
                (
                    {"initial_iterations": 0, "gradient_transition_count": -1},
                    ValueError,
                    "initial_iterations",
                ),
                (
                    {
                        "no_gradient_transition_count": -1,
                        "gradient_transition_count": -1,
                    },
                    ValueError,
                    "no_gradient_transition_count",
                ),
                (
                    {"gradient_transition_count": 1, "no_gradient_transition_count": 0},
                    ValueError,
                    "mutually exclusive",
                ),
                (
                    {
                        "recurrent_layer_norm_position": object(),
                        "gate_config": object(),
                    },
                    TypeError,
                    "recurrent_layer_norm_position",
                ),
                (
                    {"gate_config": object(), "memory_config": object()},
                    TypeError,
                    "gate_config",
                ),
            )
            for changes, error, message in cases:
                with self.subTest(variant=type(cfg).__name__, changes=changes):
                    with self.assertRaisesRegex(error, message):
                        replace(cfg, **changes).build()
