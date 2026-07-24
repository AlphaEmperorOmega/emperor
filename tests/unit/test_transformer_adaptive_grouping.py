import unittest

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
    RowLayout,
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
from support.attention import build_attention_config, make_projection_model_config


def grouped_feed_forward_stack(group_count: int = 2) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=4,
        hidden_dim=4,
        output_dim=4,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
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
                    grouping_scope=AdaptiveParameterGroupingScopeOptions.SEQUENCE,
                    group_count=group_count,
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


class FeedForwardLayoutSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layouts = []

    def forward(self, input_batch, *, row_layout=None):
        self.layouts.append(row_layout)
        return input_batch, input_batch.new_zeros(())


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


class TransformerAdaptiveGroupingTests(unittest.TestCase):
    def test_encoder_rejects_rows_feed_forward_grouping_before_rng_consumption(self):
        config = encoder_config()
        feed_forward_stack = config.feed_forward_config.stack_config
        adaptive_layer_config = feed_forward_stack.layer_config.layer_model_config
        adaptive_config = adaptive_layer_config.adaptive_augmentation_config
        adaptive_config.grouping_scope = AdaptiveParameterGroupingScopeOptions.ROWS
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
        adaptive_config.grouping_scope = AdaptiveParameterGroupingScopeOptions.ROWS
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

    def test_direct_feed_forward_threads_explicit_layout_to_grouped_leaves(self):
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
        layout = RowLayout.sequence(
            leading_shape=(2, 4),
            batch_axis=0,
            sequence_axis=1,
            context_sharing_restricted=False,
        )

        try:
            output, loss = model(inputs, row_layout=layout)
        finally:
            for hook in hooks:
                hook.remove()

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(generator_shapes)
        self.assertEqual(set(generator_shapes), {(4, 4)})

    def test_encoder_owner_supplies_batch_major_padding_layout_to_feed_forward(self):
        model = TransformerEncoderLayer(encoder_config(grouped_feed_forward=False))
        spy = FeedForwardLayoutSpy()
        model.feed_forward_layer.model = spy
        inputs = torch.randn(2, 4, 4)
        padding_mask = torch.tensor(
            [[False, False, True, True], [False, True, False, True]]
        )

        output, _loss = model(
            inputs,
            source_key_padding_mask=padding_mask,
        )

        self.assertEqual(tuple(output.shape), (2, 4, 4))
        layout = spy.layouts[0]
        self.assertEqual(layout.leading_shape, (2, 4))
        self.assertEqual(layout.batch_axis, 0)
        self.assertEqual(layout.sequence_axis, 1)
        self.assertFalse(layout.context_sharing_restricted)
        torch.testing.assert_close(
            layout.valid_rows,
            torch.tensor([True, True, False, False, True, False, True, False]),
        )

    def test_finite_additive_padding_values_remain_valid_rows(self):
        model = TransformerEncoderLayer(encoder_config(grouped_feed_forward=False))
        spy = FeedForwardLayoutSpy()
        model.feed_forward_layer.model = spy
        additive_padding_mask = torch.tensor(
            [
                [0.0, -0.25, float("-inf"), 0.75],
                [-1.0, 0.0, 2.0, float("-inf")],
            ]
        )

        model(
            torch.randn(2, 4, 4),
            source_key_padding_mask=additive_padding_mask,
        )

        torch.testing.assert_close(
            spy.layouts[0].valid_rows,
            torch.tensor([True, True, False, True, True, True, True, False]),
        )

    def test_padding_valid_rows_follow_hidden_device(self):
        model = TransformerEncoderLayer(encoder_config(grouped_feed_forward=False))
        model.self_attention_layer.model = AttentionLayoutSpy()
        feed_forward_spy = FeedForwardLayoutSpy()
        model.feed_forward_layer.model = feed_forward_spy

        model(
            torch.empty(2, 4, 4, device="meta"),
            source_key_padding_mask=torch.zeros(2, 4, dtype=torch.bool),
        )

        self.assertEqual(feed_forward_spy.layouts[0].valid_rows.device.type, "meta")

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
            [[False, False, True, True], [False, True, False, True]]
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
            [[False, False, True, True], [False, True, False, True]]
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
