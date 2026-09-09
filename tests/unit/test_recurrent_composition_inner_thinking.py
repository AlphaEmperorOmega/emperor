import unittest
from dataclasses import dataclass, replace

import torch

from emperor.attention import AttentionLayerState
from emperor.config import ConfigBase, optional_field
from emperor.layers import (
    InnerThinkingRecurrentConfig,
    LayerState,
)
from emperor.layers._composition.recurrent.variants.inner_thinking import (
    InnerThinkingRecurrent,
)
from emperor.nn import Module
from emperor.sampler import RouterConfig, TokenSamplerConfig, TokenSamplingResult
from emperor.transformer._state import TransformerDecoderLayerState
from support.layers import _set_affine_parameters, linear_stack_config


@dataclass
class RecordingBlockConfig(ConfigBase):
    input_dim: int | None = optional_field("Input dimension.")
    output_dim: int | None = optional_field("Output dimension.")

    def _registry_owner(self) -> type:
        return RecordingBlock


class RecordingBlock(Module):
    def __init__(self, cfg, overrides=None):
        super().__init__()
        self.cfg = self._override_config(cfg, overrides)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []
        self.grad_modes = []

    def forward(self, state):
        self.calls.append(replace(state, hidden=state.hidden.detach().clone()))
        self.grad_modes.append(torch.is_grad_enabled())
        state.hidden = state.hidden * self.scale
        state.loss = state.hidden.new_tensor(0.25) + (
            0 if state.loss is None else state.loss
        )
        return state


def config(**changes):
    ratio = changes.pop("selection_ratio", 0.5)
    return replace(
        InnerThinkingRecurrentConfig(
            input_dim=2,
            output_dim=2,
            max_steps=3,
            initial_iterations=3,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            block_config=RecordingBlockConfig(),
            sampler_config=TokenSamplerConfig(
                selection_ratio=ratio,
                router_config=RouterConfig(
                    num_experts=1,
                    noisy_topk_flag=False,
                    model_config=linear_stack_config(2, output_dim=1, bias_flag=True),
                ),
            ),
        ),
        **changes,
    )


def set_router(model, step, direction):
    _set_affine_parameters(
        model.samplers[step].router.model, torch.tensor([[direction], [0.0]]), None
    )


class TestInnerThinkingRecurrent(unittest.TestCase):
    def test_initial_pass_then_gather_and_masked_residual_with_new_selection(self):
        model = config().build()
        assert isinstance(model, InnerThinkingRecurrent)
        set_router(model, 0, 1)
        set_router(model, 1, -1)
        hidden = torch.tensor([[[4.0, 1], [1.0, 2], [3.0, 3], [2.0, 4]]])
        original = hidden.clone()
        result = model(LayerState(hidden, loss=torch.tensor(2.0)))
        first_extra = original.clone()
        first_extra[:, [0, 2]] += (
            original[:, [0, 2]] * original[:, [0, 2], :1].sigmoid()
        )
        expected = first_extra.clone()
        expected[:, [1, 3]] += (
            first_extra[:, [1, 3]] * (-first_extra[:, [1, 3], :1]).sigmoid()
        )
        torch.testing.assert_close(result.hidden, expected)
        calls = model.block_model.calls
        assert [call.hidden.shape[-2] for call in calls] == [4, 2, 2]
        torch.testing.assert_close(calls[1].hidden, original[:, [0, 2]])
        torch.testing.assert_close(calls[2].hidden, first_extra[:, [1, 3]])
        torch.testing.assert_close(hidden, original)
        torch.testing.assert_close(result.loss, torch.tensor(2.75))

    def test_batch_items_route_independently_and_skipped_gradients_survive(self):
        model = config(max_steps=2, initial_iterations=2).build()
        set_router(model, 0, 1)
        hidden = torch.tensor(
            [[[1.0, 1], [3.0, 2]], [[4.0, 3], [2.0, 4]]], requires_grad=True
        )
        result = model(LayerState(hidden)).hidden
        torch.testing.assert_close(result[0, 0], hidden[0, 0])
        torch.testing.assert_close(result[1, 1], hidden[1, 1])
        result.sum().backward()
        torch.testing.assert_close(hidden.grad[0, 0], torch.ones(2))
        torch.testing.assert_close(hidden.grad[1, 1], torch.ones(2))
        assert (
            model.samplers[0].router.model[0].model.weight_params.grad.abs().sum() > 0
        )
        assert model.block_model.scale.grad.abs().sum() > 0
        assert model.thinking_step_weights.grad.abs().sum() > 0

    def test_learned_initial_and_step_weights_and_configured_scale(self):
        model = config(
            max_steps=2, initial_iterations=2, thinking_step_scale=2.0
        ).build()
        set_router(model, 0, 0)
        with torch.no_grad():
            model.thinking_step_weights.copy_(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
        hidden = torch.ones(1, 1, 2)
        result = model(LayerState(hidden)).hidden
        torch.testing.assert_close(result, torch.tensor([[[10.0, 18.0]]]))

    def test_feature_last_layouts_and_capacity_floor(self):
        for shape in [(5, 2), (2, 5, 2), (2, 3, 5, 2)]:
            with self.subTest(shape=shape):
                model = config(selection_ratio=0.3).build()
                result = model(LayerState(torch.randn(shape)))
                assert result.hidden.shape == shape
                assert [call.hidden.shape[-2] for call in model.block_model.calls] == [
                    5,
                    1,
                    1,
                ]

    def test_full_ratio_preserves_token_order(self):
        model = config(selection_ratio=1.0, max_steps=2, initial_iterations=2).build()
        set_router(model, 0, -1)
        hidden = torch.tensor([[[2.0, 0], [1.0, 0], [3.0, 0]]])
        model(LayerState(hidden))
        torch.testing.assert_close(model.block_model.calls[1].hidden, hidden)

    def test_padding_excluded_and_unused_capacity_has_zero_updates(self):
        model = config(selection_ratio=0.75, max_steps=2, initial_iterations=2).build()
        set_router(model, 0, 1)
        hidden = torch.tensor([[[100.0, 1], [2.0, 2], [200.0, 3], [300.0, 4]]])
        padding = torch.tensor([[True, False, True, True]])
        state = AttentionLayerState(hidden=hidden, key_padding_mask=padding)
        result = model(state)
        expected = hidden.clone()
        expected[:, 1] += hidden[:, 1] * hidden[:, 1, :1].sigmoid()
        torch.testing.assert_close(result.hidden, expected)
        assert result.key_padding_mask is padding
        torch.testing.assert_close(
            model.block_model.calls[1].key_padding_mask,
            torch.tensor([[False, True, True]]),
        )

    def test_batched_multihead_attention_masks_select_original_positions(self):
        model = config(max_steps=2, initial_iterations=2).build()
        model.block_model.num_heads = 2
        set_router(model, 0, 1)
        hidden = torch.tensor(
            [
                [[1.0, 0], [4.0, 0], [2.0, 0], [3.0, 0]],
                [[5.0, 0], [2.0, 0], [4.0, 0], [1.0, 0]],
            ]
        )
        mask = torch.arange(16.0).reshape(4, 4)
        result = model(AttentionLayerState(hidden=hidden, attention_mask=mask))
        selected_mask = model.block_model.calls[1].attention_mask
        expected = torch.stack(
            [mask[[1, 3]][:, [1, 3]], mask[[0, 2]][:, [0, 2]]]
        ).repeat_interleave(2, dim=0)
        torch.testing.assert_close(selected_mask, expected)
        assert result.attention_mask is mask

    def test_decoder_cross_context_preserved_and_only_query_mask_rows_selected(self):
        model = config(max_steps=2, initial_iterations=2).build()
        set_router(model, 0, 1)
        hidden = torch.tensor([[[1.0, 0], [4.0, 0], [2.0, 0], [3.0, 0]]])
        encoder = torch.randn(1, 3, 2)
        mask = torch.arange(12.0).reshape(4, 3)
        controller = object()
        state = TransformerDecoderLayerState(
            hidden=hidden,
            encoder_output=encoder,
            cross_attention_mask=mask,
            controller_state=controller,
        )
        result = model(state)
        selected = model.block_model.calls[1]
        torch.testing.assert_close(
            selected.cross_attention_mask, mask[[1, 3]].unsqueeze(0)
        )
        assert selected.encoder_output is encoder
        assert result.encoder_output is encoder
        assert result.controller_state is controller
        assert result.cross_attention_mask is mask

    def test_growth_and_gradient_suffix_use_existing_runtime_and_checkpoint(self):
        cfg = config(initial_iterations=1, gradient_transition_count=1)
        model = cfg.build()
        for expected in (1, 2, 3):
            model.block_model.calls.clear()
            model.block_model.grad_modes.clear()
            model(LayerState(torch.ones(1, 4, 2))).hidden.sum().backward()
            assert len(model.block_model.calls) == expected
            assert model.block_model.grad_modes == [False] * (expected - 1) + [True]
        restored = cfg.build()
        restored.load_state_dict(model.state_dict())
        hidden = torch.randn(1, 4, 2)
        torch.testing.assert_close(
            model(LayerState(hidden)).hidden, restored(LayerState(hidden)).hidden
        )

    def test_single_initial_pass_and_outer_no_grad(self):
        model = config(max_steps=1, initial_iterations=1).build()
        assert len(model.samplers) == 0
        hidden = torch.randn(1, 4, 2, requires_grad=True)
        with torch.no_grad():
            result = model(LayerState(hidden)).hidden
        assert not result.requires_grad
        torch.testing.assert_close(result, hidden)

    def test_real_configurable_blocks_can_be_stacked_and_backpropagate(self):
        cfg = config(block_config=linear_stack_config(2), thinking_step_scale=0.2)
        model = cfg.build(
            overrides=InnerThinkingRecurrentConfig(input_dim=2, output_dim=2)
        ).double()
        assert model.thinking_step_scale == 0.2
        next_model = cfg.build().double()
        hidden = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)
        result = next_model(model(LayerState(hidden)))
        result.hidden.square().sum().backward()
        assert hidden.grad.abs().sum() > 0
        assert all(parameter.grad is not None for parameter in model.parameters())
        assert result.hidden.dtype == torch.float64

    def test_invalid_configuration_rejected(self):
        for name, value in [
            ("selection_ratio", 0),
            ("selection_ratio", -0.1),
            ("selection_ratio", 1.1),
            ("selection_ratio", float("nan")),
            ("selection_ratio", True),
            ("thinking_step_scale", -1),
            ("thinking_step_scale", float("inf")),
            ("thinking_step_scale", False),
            ("max_steps", True),
        ]:
            with self.subTest(name=name, value=value):
                with self.assertRaises((ValueError, TypeError)):
                    config(**{name: value}).build()

    def test_failure_does_not_commit_progress_or_mutate_input(self):
        model = config().build()
        hidden = torch.ones(1, 4, 2)
        state = AttentionLayerState(
            hidden=hidden, key_padding_mask=torch.ones(1, 4, dtype=torch.bool)
        )
        with self.assertRaisesRegex(ValueError, "at least one unpadded"):
            model(state)
        assert model.recurrent_iteration_schedule.forward_call_progress.item() == 0
        assert state.hidden is hidden
        torch.testing.assert_close(hidden, torch.ones_like(hidden))

    def test_real_encoder_preserves_masks_and_backpropagates_through_selected_tokens(
        self,
    ):
        from unit.test_recurrent_composition_blocks import (
            _encoder_block,
            _self_attention,
        )

        model = config(
            input_dim=4,
            output_dim=4,
            block_config=_encoder_block(4, _self_attention(4, 4)),
        ).build()
        hidden = torch.randn(2, 4, 4, requires_grad=True)
        padding = torch.tensor(
            [[False, False, False, True], [False, False, True, True]]
        )
        causal_mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
        lengths = []
        handle = model.block_model.register_forward_pre_hook(
            lambda _, args: lengths.append(args[0].hidden.shape[-2])
        )
        try:
            result = model(
                AttentionLayerState(
                    hidden=hidden,
                    key_padding_mask=padding,
                    attention_mask=causal_mask,
                )
            )
        finally:
            handle.remove()
        assert lengths == [4, 2, 2]
        assert result.key_padding_mask is padding
        assert result.attention_mask is causal_mask
        result.hidden.square().sum().backward()
        assert torch.isfinite(hidden.grad).all()
        assert hidden.grad.abs().sum() > 0

    def test_recurrent_layer_uses_sampler_decisions_and_weights_directly(self):
        model = config(max_steps=2, initial_iterations=2).build()
        set_router(model, 0, 1)
        hidden = torch.tensor([[[10.0, 1], [1.0, 2], [9.0, 3], [2.0, 4]]])
        # Deliberately select the lowest-scoring token. The layer must obey the sampler.
        model.samplers[0].forward = lambda hidden, padding: TokenSamplingResult(
            indices=torch.tensor([[1]]),
            weights=torch.tensor([[0.25]]),
            valid=torch.tensor([[True]]),
            sequence_length=4,
        )
        result = model(LayerState(hidden)).hidden
        expected = hidden.clone()
        expected[:, 1] *= 1.25
        torch.testing.assert_close(result, expected)
        assert model.block_model.calls[1].hidden.shape == (1, 1, 2)

    def test_config_composes_as_a_generic_block_of_another_recurrent_variant(self):
        from emperor.layers import RecurrentLayerConfig

        outer = RecurrentLayerConfig(
            input_dim=2,
            output_dim=2,
            max_steps=2,
            initial_iterations=2,
            iteration_increment=1,
            forward_calls_before_iteration_increment=1,
            block_config=config(),
        ).build()
        hidden = torch.randn(2, 4, 2, requires_grad=True)
        result = outer(LayerState(hidden))
        result.hidden.sum().backward()
        assert hidden.grad.abs().sum() > 0
        assert isinstance(outer.block_model, InnerThinkingRecurrent)
        assert len(outer.block_model.block_model.calls) == 6

    def test_smooth_growth_matches_adjacent_depth_outputs(self):
        for gradient_controls in [
            {"gradient_transition_count": 1},
            {"no_gradient_transition_count": 0},
        ]:
            with self.subTest(gradient_controls=gradient_controls):
                smooth = config(
                    max_steps=2,
                    initial_iterations=1,
                    forward_calls_before_iteration_increment=4,
                    smooth_iteration_growth_flag=True,
                    **gradient_controls,
                ).build()
                set_router(smooth, 0, 1)
                hidden = torch.tensor([[[1.0, 2], [2.0, 1]]])
                for _ in range(4):
                    smooth(LayerState(hidden))
                source = config(max_steps=2, initial_iterations=1).build()
                target = config(max_steps=2, initial_iterations=2).build()
                set_router(target, 0, 1)
                actual = smooth(LayerState(hidden))
                source_result = source(LayerState(hidden))
                target_result = target(LayerState(hidden))
                torch.testing.assert_close(
                    actual.hidden, 0.5 * (source_result.hidden + target_result.hidden)
                )
                torch.testing.assert_close(
                    actual.loss, 0.5 * (source_result.loss + target_result.loss)
                )
                actual.hidden.sum().backward()
                assert (
                    smooth.samplers[0].router.model[0].model.weight_params.grad
                    is not None
                )

    def test_transition_sampler_parameters_are_independent_and_registered(self):
        model = config().build()
        first = next(model.samplers[0].parameters())
        second = next(model.samplers[1].parameters())
        assert first is not second
        assert first.data_ptr() != second.data_ptr()
        assert any(name.startswith("samplers.0.") for name in model.state_dict())

    def test_missing_or_invalid_sampler_config_rejected(self):
        for sampler_config in [None, object()]:
            with self.subTest(sampler_config=sampler_config):
                with self.assertRaisesRegex((TypeError, ValueError), "sampler_config"):
                    config(sampler_config=sampler_config).build()

    def test_batch_first_layout_is_validated_before_the_block_runs(self):
        model = config().build()
        model.block_model.batch_first_flag = False
        with self.assertRaisesRegex(ValueError, "batch-first"):
            model(LayerState(torch.ones(2, 3, 2)))
        assert model.block_model.calls == []

    def test_incompatible_attention_head_counts_are_rejected_by_validator(self):
        model = config().build()
        model.block_model.num_heads = 2
        child = torch.nn.Identity()
        child.num_heads = 4
        model.block_model.add_module("child", child)
        with self.assertRaisesRegex(ValueError, "common head count"):
            model(
                AttentionLayerState(
                    hidden=torch.ones(2, 3, 2),
                    attention_mask=torch.zeros(3, 3),
                )
            )

    def test_mismatched_attention_mask_shape_is_rejected_by_validator(self):
        model = config().build()
        with self.assertRaisesRegex(ValueError, "match the token count"):
            model(
                AttentionLayerState(
                    hidden=torch.ones(2, 3, 2),
                    attention_mask=torch.zeros(2, 2),
                )
            )
