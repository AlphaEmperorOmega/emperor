import unittest
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from emperor.attention import (
    AttentionLayerState,
    IndependentAttentionConfig,
    SelfAttentionConfig,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveLinearLayerConfig,
    AdaptiveParameterAugmentationConfig,
    AdaptiveParameterGroupingScopeOptions,
    AdditiveDynamicBiasConfig,
    WeightDecayScheduleOptions,
)
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterInputOrderOptions as Order,
)
from emperor.augmentations.adaptive_parameters._linear_adapter import (
    AdaptiveLinearLayer,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentLayer,
    RecurrentLayerConfig,
)
from emperor.transformer import (
    FeedForward,
    FeedForwardConfig,
    TransformerDecoderLayer,
    TransformerDecoderLayerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayer,
    TransformerEncoderLayerConfig,
)
from support.adaptive_grouping import grouping_value
from support.attention import build_attention_config, make_projection_model_config


def grouped_feed_forward_stack(group_count: int = 2) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_postprocessing_flag=False,
        layer_config=LayerConfig(
            input_dim=4,
            output_dim=4,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=AdaptiveLinearLayerConfig(
                input_dim=4,
                output_dim=4,
                bias_flag=True,
                adaptive_augmentation_config=AdaptiveParameterAugmentationConfig(
                    grouping_config=grouping_value(
                        AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                        group_count,
                        input_order="BATCH_FIRST",
                    ),
                    bias_config=AdditiveDynamicBiasConfig(
                        decay_schedule=WeightDecayScheduleOptions.DISABLED,
                        decay_rate=0.0,
                        decay_warmup_batches=0,
                        model_config=make_projection_model_config(),
                    ),
                ),
            ),
        ),
    )


def encoder_config(*, grouped_feed_forward: bool = True):
    attention_config = build_attention_config(
        config_class=SelfAttentionConfig,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=4,
        source_sequence_length=4,
    )
    attention_config.batch_first_flag = True
    stack_config = (
        grouped_feed_forward_stack()
        if grouped_feed_forward
        else make_projection_model_config(hidden_dim=4)
    )
    return TransformerEncoderLayerConfig(
        embedding_dim=4,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        dropout_probability=0.0,
        residual_config=None,
        attention_config=attention_config,
        feed_forward_config=FeedForwardConfig(
            input_dim=4,
            output_dim=4,
            stack_config=stack_config,
        ),
    )


def decoder_config(
    *,
    grouped_feed_forward: bool = False,
    cross_attention_config=None,
):
    self_attention_config = build_attention_config(
        config_class=SelfAttentionConfig,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=4,
        source_sequence_length=4,
    )
    self_attention_config.batch_first_flag = True
    stack_config = (
        grouped_feed_forward_stack()
        if grouped_feed_forward
        else make_projection_model_config(hidden_dim=4)
    )
    return TransformerDecoderLayerConfig(
        embedding_dim=4,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        dropout_probability=0.0,
        residual_config=None,
        self_attention_config=self_attention_config,
        cross_attention_config=cross_attention_config,
        feed_forward_config=FeedForwardConfig(
            input_dim=4,
            output_dim=4,
            stack_config=stack_config,
        ),
    )


class AttentionLayoutSpy(nn.Module):
    batch_first_flag = True
    causal_attention_mask_flag = False

    def forward(
        self,
        *,
        q,
        k,
        v,
        k_padding_mask=None,
        attention_mask=None,
    ):
        return q, None, q.new_zeros(())


def adaptive_leaves(module):
    return [
        child for child in module.modules() if isinstance(child, AdaptiveLinearLayer)
    ]


@pytest.mark.parametrize("owner", ["encoder", "decoder", "decoder_cross_attention"])
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("mask_dtype", [None, torch.bool, torch.float64])
def test_unbatched_grouping_matches_single_sequence_batch(
    owner, batch_first, mask_dtype
):
    sequence_length = 6
    source_length = 5
    feature_dim = 4
    cross_config = None
    if owner == "decoder_cross_attention":
        cross_config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=2,
            embedding_dim=feature_dim,
            target_sequence_length=sequence_length,
            source_sequence_length=source_length,
        )
        cross_config.batch_first_flag = batch_first
    if owner == "encoder":
        config = encoder_config(grouped_feed_forward=True)
        attention_config = config.attention_config
        padding_argument = "source_key_padding_mask"
    else:
        config = decoder_config(
            grouped_feed_forward=True, cross_attention_config=cross_config
        )
        attention_config = config.self_attention_config
        padding_argument = "key_padding_mask"
    attention_config.batch_first_flag = batch_first
    attention_config.target_sequence_length = sequence_length
    attention_config.source_sequence_length = sequence_length
    layer_config = config.feed_forward_config.stack_config.layer_config
    augmentation = layer_config.layer_model_config.adaptive_augmentation_config
    input_order = Order.SEQUENCE_FIRST
    batch_axis = 1
    if batch_first:
        input_order = Order.BATCH_FIRST
        batch_axis = 0
    augmentation.grouping_config = replace(
        augmentation.grouping_config,
        sequence_length=sequence_length,
        input_order=input_order,
    )
    model = config.build().double().eval()
    inputs = torch.randn(
        sequence_length, feature_dim, dtype=torch.float64, requires_grad=True
    )
    detached_inputs = inputs.detach()
    batched_inputs = detached_inputs.unsqueeze(batch_axis)
    batched_inputs.requires_grad_()
    unbatched_arguments = {}
    batched_arguments = {}
    if mask_dtype is not None:
        padding_mask = torch.zeros(sequence_length, dtype=mask_dtype)
        unbatched_arguments[padding_argument] = padding_mask
        batched_arguments[padding_argument] = padding_mask.unsqueeze(0)
    if cross_config is not None:
        source = torch.randn(source_length, feature_dim, dtype=torch.float64)
        unbatched_arguments["encoder_output"] = source
        batched_arguments["encoder_output"] = source.unsqueeze(batch_axis)
        if mask_dtype is not None:
            source_padding = torch.zeros(source_length, dtype=mask_dtype)
            unbatched_arguments["encoder_padding_mask"] = source_padding
            batched_arguments["encoder_padding_mask"] = source_padding.unsqueeze(0)

    output, loss = model(inputs, **unbatched_arguments)
    batched_output, batched_loss = model(batched_inputs, **batched_arguments)
    assert output.shape == (sequence_length, feature_dim)
    torch.testing.assert_close(output, batched_output.squeeze(batch_axis))
    torch.testing.assert_close(loss, batched_loss)
    (output.square().sum() + loss).backward()
    (batched_output.square().sum() + batched_loss).backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    torch.testing.assert_close(inputs.grad, batched_inputs.grad.squeeze(batch_axis))


class TransformerAdaptiveGroupingTests(unittest.TestCase):
    def test_encoder_rejects_rows_feed_forward_grouping_before_rng_consumption(self):
        config = encoder_config()
        feed_forward_stack = config.feed_forward_config.stack_config
        adaptive_layer_config = feed_forward_stack.layer_config.layer_model_config
        adaptive_config = adaptive_layer_config.adaptive_augmentation_config
        adaptive_config.grouping_config = grouping_value(
            AdaptiveParameterGroupingScopeOptions.ROWS, 2
        )
        with torch.random.fork_rng():
            torch.manual_seed(47)
            rng_before_construction = torch.get_rng_state().clone()

            with self.assertRaisesRegex(
                ValueError,
                "TransformerEncoderLayerConfig feed-forward does not support ROWS "
                "adaptive parameter grouping",
            ):
                TransformerEncoderLayer(config)

            torch.testing.assert_close(torch.get_rng_state(), rng_before_construction)

    def test_decoder_rejects_rows_feed_forward_grouping_before_rng_consumption(self):
        config = decoder_config(grouped_feed_forward=True)
        feed_forward_stack = config.feed_forward_config.stack_config
        adaptive_layer_config = feed_forward_stack.layer_config.layer_model_config
        adaptive_config = adaptive_layer_config.adaptive_augmentation_config
        adaptive_config.grouping_config = grouping_value(
            AdaptiveParameterGroupingScopeOptions.ROWS, 2
        )
        with torch.random.fork_rng():
            torch.manual_seed(53)
            rng_before_construction = torch.get_rng_state().clone()

            with self.assertRaisesRegex(
                ValueError,
                "TransformerDecoderLayerConfig feed-forward does not support ROWS "
                "adaptive parameter grouping",
            ):
                TransformerDecoderLayer(config)

            torch.testing.assert_close(torch.get_rng_state(), rng_before_construction)

    def test_encoder_padding_shape_uses_attention_mask_diagnostic(self):
        model = TransformerEncoderLayer(encoder_config(grouped_feed_forward=False))

        with self.assertRaisesRegex(
            RuntimeError,
            r"key_padding_mask must have shape \(2, 4\), got \(2, 3\)\.",
        ):
            model(
                torch.randn(2, 4, 4),
                source_key_padding_mask=torch.zeros(2, 3, dtype=torch.bool),
            )

    def test_decoder_target_padding_shape_uses_attention_mask_diagnostic(self):
        model = TransformerDecoderLayer(decoder_config())

        with self.assertRaisesRegex(
            RuntimeError,
            r"key_padding_mask must have shape \(2, 4\), got \(2, 3\)\.",
        ):
            model(
                torch.randn(2, 4, 4),
                key_padding_mask=torch.zeros(2, 3, dtype=torch.bool),
            )

    def test_direct_feed_forward_uses_configured_order_for_grouped_leaves(self):
        model = FeedForward(
            FeedForwardConfig(
                input_dim=4,
                output_dim=4,
                stack_config=grouped_feed_forward_stack(),
            )
        )
        generator_shapes = []
        hooks = []
        for leaf in adaptive_leaves(model):
            generator = leaf.adaptive_behaviour.bias_model.model[0].model
            hooks.append(
                generator.register_forward_hook(
                    lambda _module, args, _output: generator_shapes.append(
                        tuple(args[0].shape)
                    )
                )
            )
        inputs = torch.randn(2, 4, 4)

        try:
            output, loss = model(inputs)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(generator_shapes)
        self.assertEqual(set(generator_shapes), {(4, 4)})

    def test_encoder_grouped_feed_forward_runs_forward_and_backward(self):
        model = TransformerEncoderLayer(encoder_config())
        generator_shapes = []
        hooks = []
        for leaf in adaptive_leaves(model.feed_forward_model):
            generator = leaf.adaptive_behaviour.bias_model.model[0].model
            hooks.append(
                generator.register_forward_hook(
                    lambda _module, args, _output: generator_shapes.append(
                        tuple(args[0].shape)
                    )
                )
            )
        inputs = torch.randn(2, 4, 4, requires_grad=True)
        padding_mask = torch.tensor(
            [[False, False, False, False], [False, False, False, False]]
        )

        try:
            output, loss = model(
                inputs,
                source_key_padding_mask=padding_mask,
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertTrue(generator_shapes)
        self.assertEqual(set(generator_shapes), {(4, 4)})
        (output.square().mean() + loss).backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_outer_recurrent_encoder_without_controllers_recomputes_each_step(self):
        model = RecurrentLayer(
            RecurrentLayerConfig(
                input_dim=4,
                output_dim=4,
                max_steps=2,
                initial_iterations=2,
                iteration_increment=1,
                forward_calls_before_iteration_increment=1,
                recurrent_layer_norm_position=LayerNormPositionOptions.DISABLED,
                block_config=TransformerEncoderBlockLayerConfig(
                    input_dim=4,
                    output_dim=4,
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_config=None,
                    dropout_probability=0.0,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=encoder_config(),
                ),
                gate_config=None,
                residual_config=None,
                halting_config=None,
                memory_config=None,
            )
        )
        generator_contexts = []
        hooks = []
        for leaf in adaptive_leaves(model.block_model):
            generator = leaf.adaptive_behaviour.bias_model.model[0].model
            hooks.append(
                generator.register_forward_hook(
                    lambda _module, args, _output: generator_contexts.append(
                        args[0].detach().clone()
                    )
                )
            )
        inputs = torch.randn(2, 4, 4, requires_grad=True)
        padding_mask = torch.tensor(
            [[False, False, False, False], [False, False, False, False]]
        )

        try:
            output_state = model(
                AttentionLayerState(
                    hidden=inputs,
                    key_padding_mask=padding_mask,
                )
            )
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output_state.hidden.shape), (2, 4, 4))
        self.assertEqual(len(generator_contexts), 4)
        self.assertTrue(
            all(tuple(context.shape) == (4, 4) for context in generator_contexts)
        )
        objective = output_state.hidden.square().mean()
        if output_state.loss is not None:
            objective = objective + output_state.loss
        objective.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_attention_restriction_persists_into_feed_forward_before_generator(self):
        model = TransformerEncoderLayer(encoder_config())
        calls = []
        leaf = adaptive_leaves(model.feed_forward_model)[0]
        generator = leaf.adaptive_behaviour.bias_model.model[0].model
        hook = generator.register_forward_hook(lambda *_args: calls.append(True))
        inputs = torch.randn(2, 4, 4)

        try:
            with self.assertRaisesRegex(
                ValueError,
                "context sharing is restricted",
            ):
                model(inputs, attention_mask=torch.zeros(4, 4))
        finally:
            hook.remove()

        self.assertEqual(calls, [])

    def test_decoder_causal_restriction_reaches_grouped_feed_forward(self):
        self_attention_config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=4,
            causal_attention_mask_flag=True,
        )
        self_attention_config.batch_first_flag = True
        model = TransformerDecoderLayer(
            TransformerDecoderLayerConfig(
                embedding_dim=4,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                dropout_probability=0.0,
                residual_config=None,
                self_attention_config=self_attention_config,
                cross_attention_config=None,
                feed_forward_config=FeedForwardConfig(
                    input_dim=4,
                    output_dim=4,
                    stack_config=grouped_feed_forward_stack(),
                ),
            )
        )
        calls = []
        leaf = adaptive_leaves(model.feed_forward_model)[0]
        generator = leaf.adaptive_behaviour.bias_model.model[0].model
        hook = generator.register_forward_hook(lambda *_args: calls.append(True))

        try:
            with self.assertRaisesRegex(
                ValueError,
                "context sharing is restricted",
            ):
                model(torch.randn(2, 4, 4))
        finally:
            hook.remove()

        self.assertEqual(calls, [])

    def test_decoder_cross_attention_mask_restriction_reaches_grouped_feed_forward(
        self,
    ):
        cross_attention_config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=3,
        )
        cross_attention_config.batch_first_flag = True
        model = TransformerDecoderLayer(
            decoder_config(
                grouped_feed_forward=True,
                cross_attention_config=cross_attention_config,
            )
        )
        calls = []
        leaf = adaptive_leaves(model.feed_forward_model)[0]
        generator = leaf.adaptive_behaviour.bias_model.model[0].model
        hook = generator.register_forward_hook(lambda *_args: calls.append(True))

        try:
            with self.assertRaisesRegex(
                ValueError,
                "context sharing is restricted",
            ):
                model(
                    torch.randn(2, 4, 4),
                    encoder_output=torch.randn(2, 3, 4),
                    encoder_attention_mask=torch.zeros(4, 3),
                )
        finally:
            hook.remove()

        self.assertEqual(calls, [])

    def test_decoder_cross_attention_causality_reaches_grouped_feed_forward(self):
        cross_attention_config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=4,
            source_sequence_length=3,
            causal_attention_mask_flag=True,
        )
        cross_attention_config.batch_first_flag = True
        model = TransformerDecoderLayer(
            decoder_config(
                grouped_feed_forward=True,
                cross_attention_config=cross_attention_config,
            )
        )
        calls = []
        leaf = adaptive_leaves(model.feed_forward_model)[0]
        generator = leaf.adaptive_behaviour.bias_model.model[0].model
        hook = generator.register_forward_hook(lambda *_args: calls.append(True))

        try:
            with self.assertRaisesRegex(
                ValueError,
                "context sharing is restricted",
            ):
                model(
                    torch.randn(2, 4, 4),
                    encoder_output=torch.randn(2, 3, 4),
                )
        finally:
            hook.remove()

        self.assertEqual(calls, [])

    def test_decoder_ignores_cross_mask_when_cross_attention_is_disabled(self):
        model = TransformerDecoderLayer(decoder_config(grouped_feed_forward=True))

        output, loss = model(
            torch.randn(2, 4, 4),
            encoder_attention_mask=torch.zeros(4, 3),
        )

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertEqual(loss.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize(
    "mask,eligible",
    [
        (torch.zeros(2, 4, dtype=torch.bool), True),
        (torch.full((2, 4), -0.25), True),
        (torch.tensor([[False, False, False, True], [False] * 4]), False),
        (torch.tensor([[0.0, 0.0, 0.0, -torch.inf], [0.0] * 4]), False),
    ],
)
def test_feed_forward_only_grouping_obeys_target_padding_before_generation(
    mask, eligible
):
    model = encoder_config().build()
    calls = []
    hooks = [
        leaf.adaptive_behaviour.bias_model.register_forward_pre_hook(
            lambda *_: calls.append(True)
        )
        for leaf in adaptive_leaves(model)
    ]
    inputs = torch.randn(2, 4, 4)
    try:
        if eligible:
            output, _ = model(inputs, source_key_padding_mask=mask)
            assert output.shape == inputs.shape and calls
        else:
            with pytest.raises(ValueError, match="all-valid"):
                model(inputs, source_key_padding_mask=mask)
            assert not calls
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("order", list(Order))
@pytest.mark.parametrize("batch", [2, 4])
def test_standalone_feed_forward_uses_declared_order(order, batch):
    stack = grouped_feed_forward_stack()
    augmentation = stack.layer_config.layer_model_config.adaptive_augmentation_config
    augmentation.grouping_config = replace(
        augmentation.grouping_config, input_order=order
    )
    model = FeedForwardConfig(input_dim=4, output_dim=4, stack_config=stack).build()
    logical = torch.randn(batch, 4, 4, requires_grad=True)
    ordered = logical if order is Order.BATCH_FIRST else logical.transpose(0, 1)
    actual, _ = model(ordered)
    flat, _ = model(ordered.reshape(-1, 4))
    torch.testing.assert_close(actual, flat.reshape(ordered.shape))
    actual.sum().backward()
    assert logical.grad is not None and torch.isfinite(logical.grad).all()


@pytest.mark.parametrize("shape", [(2, 2, 4), (4, 2, 4), (2, 2, 2, 4)])
def test_feed_forward_rejects_wrong_actual_length_or_rank_before_generator(shape):
    model = FeedForwardConfig(
        input_dim=4, output_dim=4, stack_config=grouped_feed_forward_stack()
    ).build()
    calls = []
    hooks = [
        leaf.adaptive_behaviour.bias_model.register_forward_pre_hook(
            lambda *_: calls.append(True)
        )
        for leaf in adaptive_leaves(model)
    ]
    try:
        with pytest.raises(ValueError, match="sequence_length|rank"):
            model(torch.ones(shape))
        assert not calls
    finally:
        for hook in hooks:
            hook.remove()


def test_decoder_source_padding_does_not_invalidate_target_grouping():
    cross = build_attention_config(
        config_class=IndependentAttentionConfig,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=4,
        source_sequence_length=3,
    )
    cross.batch_first_flag = True
    model = decoder_config(
        grouped_feed_forward=True, cross_attention_config=cross
    ).build()
    inputs = torch.randn(2, 4, 4, requires_grad=True)
    output, loss = model(
        inputs,
        encoder_output=torch.randn(2, 3, 4),
        encoder_padding_mask=torch.tensor(
            [[False, False, True], [False, False, False]]
        ),
    )
    assert output.shape == inputs.shape
    (output.square().mean() + loss).backward()
    assert torch.isfinite(inputs.grad).all()


def test_feed_forward_checks_all_leaf_orders_when_batch_equals_sequence():
    from emperor.layers import GateConfig, LayerGateOptions

    stack = grouped_feed_forward_stack()
    gate_stack = grouped_feed_forward_stack()
    augmentation = (
        gate_stack.layer_config.layer_model_config.adaptive_augmentation_config
    )
    augmentation.grouping_config = replace(
        augmentation.grouping_config, input_order=Order.SEQUENCE_FIRST
    )
    stack.layer_config.gate_config = GateConfig(
        gate_dim=4,
        option=LayerGateOptions.ADDITION,
        activation=ActivationOptions.DISABLED,
        model_config=gate_stack,
    )
    model = FeedForwardConfig(input_dim=4, output_dim=4, stack_config=stack).build()
    calls = []
    hooks = [
        leaf.adaptive_behaviour.bias_model.register_forward_pre_hook(
            lambda *_: calls.append(True)
        )
        for leaf in adaptive_leaves(model)
    ]
    try:
        with pytest.raises(ValueError, match="input_order"):
            model(torch.randn(4, 4, 4))
        assert not calls
    finally:
        for hook in hooks:
            hook.remove()


def test_active_cross_attention_mask_shape_is_validated_before_grouping_restriction():
    cross = build_attention_config(
        config_class=IndependentAttentionConfig,
        batch_size=2,
        num_heads=2,
        embedding_dim=4,
        target_sequence_length=4,
        source_sequence_length=3,
    )
    cross.batch_first_flag = True
    model = decoder_config(
        grouped_feed_forward=True, cross_attention_config=cross
    ).build()
    with pytest.raises(RuntimeError, match="target/source dimensions"):
        model(
            torch.randn(2, 4, 4),
            encoder_output=torch.randn(2, 3, 4),
            encoder_attention_mask=torch.zeros(4, 2),
        )
