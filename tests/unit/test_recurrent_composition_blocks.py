import unittest

import torch

from emperor.attention import (
    AttentionLayerState,
    MixerAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.layers import (
    ActivationOptions,
    AdditiveResidualConfig,
    HierarchicalReasoningModelRecurrentConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    RecurrentCompositionConfig,
    RowLayout,
    TinyRecursiveModelRecurrentConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerConfig,
    TransformerEncoderBlockLayerConfig,
    TransformerEncoderLayerConfig,
)


def _linear_stack(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=output_dim,
        num_layers=1,
        apply_output_pipeline_flag=False,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        shared_gate_config=None,
        shared_halting_config=None,
        shared_memory_config=None,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def _self_attention(
    model_dim: int,
    sequence_length: int,
    *,
    projection_model_config: LayerStackConfig
    | RecurrentCompositionConfig
    | None = None,
) -> SelfAttentionConfig:
    return SelfAttentionConfig(
        batch_size=2,
        num_heads=2,
        embedding_dim=model_dim,
        query_key_projection_dim=model_dim,
        value_projection_dim=model_dim,
        target_sequence_length=sequence_length,
        source_sequence_length=sequence_length,
        target_dtype=torch.float32,
        dropout_probability=0.0,
        zero_attention_flag=False,
        causal_attention_mask_flag=False,
        add_key_value_bias_flag=False,
        average_attention_weights_flag=True,
        return_attention_weights_flag=False,
        batch_first_flag=True,
        projection_model_config=(
            _linear_stack(model_dim, model_dim)
            if projection_model_config is None
            else projection_model_config
        ),
        relative_positional_embedding_config=None,
        projection_strategy=SelfAttentionProjectionStrategy.FUSED,
    )


def _encoder_block(
    model_dim: int,
    attention_config: SelfAttentionConfig | MixerAttentionConfig,
) -> TransformerEncoderBlockLayerConfig:
    encoder = TransformerEncoderLayerConfig(
        embedding_dim=model_dim,
        layer_norm_position=LayerNormPositionOptions.BEFORE,
        dropout_probability=0.0,
        residual_config=AdditiveResidualConfig(),
        attention_config=attention_config,
        feed_forward_config=FeedForwardConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            stack_config=_linear_stack(model_dim, model_dim),
        ),
    )
    return TransformerEncoderBlockLayerConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=encoder,
    )


def _runtime(block_config: TransformerEncoderBlockLayerConfig):
    model_dim = block_config.input_dim
    return TinyRecursiveModelRecurrentConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        block_config=block_config,
        latent_updates_per_answer_update=1,
        answer_update_count=1,
        initialization_standard_deviation=0.0,
    ).build()


def _hierarchical_reasoning_model_runtime(
    block_config: TransformerEncoderBlockLayerConfig,
):
    model_dim = block_config.input_dim
    return HierarchicalReasoningModelRecurrentConfig(
        input_dim=model_dim,
        output_dim=model_dim,
        high_block_config=block_config,
        low_block_config=block_config,
        high_cycles=1,
        low_cycles=1,
        initialization_standard_deviation=0.0,
    ).build()


def _state(
    *,
    model_dim: int,
    sequence_length: int,
    key_padding_mask: torch.Tensor | None = None,
) -> AttentionLayerState:
    hidden = torch.randn(2, sequence_length, model_dim, requires_grad=True)
    row_layout = RowLayout.sequence(
        leading_shape=(2, sequence_length),
        batch_axis=0,
        sequence_axis=1,
        context_sharing_restricted=False,
    )
    return AttentionLayerState(
        hidden=hidden,
        row_layout=row_layout,
        key_padding_mask=key_padding_mask,
    )


class TestTinyRecursiveModelRecurrentSharedBlocks(unittest.TestCase):
    def test_transformer_transition_preserves_masks_layout_and_gradients(self) -> None:
        model_dim = 4
        sequence_length = 3
        runtime = _runtime(
            _encoder_block(
                model_dim,
                _self_attention(model_dim, sequence_length),
            )
        )
        key_padding_mask = torch.tensor([[False, False, True], [False, True, True]])
        state = _state(
            model_dim=model_dim,
            sequence_length=sequence_length,
            key_padding_mask=key_padding_mask,
        )
        row_layout = state.row_layout
        inputs = state.hidden

        result = runtime(state)
        result.hidden.sum().backward()

        self.assertEqual(result.hidden.shape, (2, sequence_length, model_dim))
        self.assertIs(result.row_layout, row_layout)
        self.assertIs(result.key_padding_mask, key_padding_mask)
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in runtime.parameters())
        )

    def test_mixer_transition_supports_fixed_shapes_without_masks(self) -> None:
        model_dim = 4
        sequence_length = 3
        mixer = MixerAttentionConfig(
            embedding_dim=model_dim,
            sequence_length=sequence_length,
            batch_first_flag=True,
            mixing_model_config=_linear_stack(sequence_length, sequence_length),
        )
        runtime = _runtime(_encoder_block(model_dim, mixer))
        state = _state(model_dim=model_dim, sequence_length=sequence_length)
        inputs = state.hidden

        result = runtime(state)
        result.hidden.sum().backward()

        self.assertEqual(result.hidden.shape, (2, sequence_length, model_dim))
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(result.hidden).all())

    def test_mixer_transition_rejects_padding_masks_explicitly(self) -> None:
        model_dim = 4
        sequence_length = 3
        mixer = MixerAttentionConfig(
            embedding_dim=model_dim,
            sequence_length=sequence_length,
            batch_first_flag=True,
            mixing_model_config=_linear_stack(sequence_length, sequence_length),
        )
        runtime = _runtime(_encoder_block(model_dim, mixer))
        state = _state(
            model_dim=model_dim,
            sequence_length=sequence_length,
            key_padding_mask=torch.zeros(2, sequence_length, dtype=torch.bool),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "MixerAttention does not support key padding masks",
        ):
            runtime(state)


class TestHierarchicalReasoningModelRecurrentSharedBlocks(unittest.TestCase):
    def test_transformer_transitions_preserve_masks_layout_and_gradients(self) -> None:
        model_dim = 4
        sequence_length = 3
        runtime = _hierarchical_reasoning_model_runtime(
            _encoder_block(
                model_dim,
                _self_attention(model_dim, sequence_length),
            )
        )
        key_padding_mask = torch.tensor([[False, False, True], [False, True, True]])
        state = _state(
            model_dim=model_dim,
            sequence_length=sequence_length,
            key_padding_mask=key_padding_mask,
        )
        row_layout = state.row_layout
        inputs = state.hidden

        result = runtime(state)
        result.hidden.sum().backward()

        self.assertEqual(result.hidden.shape, (2, sequence_length, model_dim))
        self.assertIs(result.row_layout, row_layout)
        self.assertIs(result.key_padding_mask, key_padding_mask)
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in runtime.parameters())
        )

    def test_mixer_transitions_support_fixed_shapes_without_masks(self) -> None:
        model_dim = 4
        sequence_length = 3
        mixer = MixerAttentionConfig(
            embedding_dim=model_dim,
            sequence_length=sequence_length,
            batch_first_flag=True,
            mixing_model_config=_linear_stack(sequence_length, sequence_length),
        )
        runtime = _hierarchical_reasoning_model_runtime(
            _encoder_block(model_dim, mixer)
        )
        state = _state(model_dim=model_dim, sequence_length=sequence_length)
        inputs = state.hidden

        result = runtime(state)
        result.hidden.sum().backward()

        self.assertEqual(result.hidden.shape, (2, sequence_length, model_dim))
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(result.hidden).all())

    def test_mixer_transitions_reject_padding_masks_explicitly(self) -> None:
        model_dim = 4
        sequence_length = 3
        mixer = MixerAttentionConfig(
            embedding_dim=model_dim,
            sequence_length=sequence_length,
            batch_first_flag=True,
            mixing_model_config=_linear_stack(sequence_length, sequence_length),
        )
        runtime = _hierarchical_reasoning_model_runtime(
            _encoder_block(model_dim, mixer)
        )
        state = _state(
            model_dim=model_dim,
            sequence_length=sequence_length,
            key_padding_mask=torch.zeros(2, sequence_length, dtype=torch.bool),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "MixerAttention does not support key padding masks",
        ):
            runtime(state)


class TestRecurrentCompositionConsumers(unittest.TestCase):
    def test_feed_forward_accepts_tiny_recursive_model_as_its_recurrent_stack(
        self,
    ) -> None:
        model_dim = 4
        recurrent = TinyRecursiveModelRecurrentConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            block_config=_linear_stack(model_dim, model_dim),
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            initialization_standard_deviation=0.0,
        )
        feed_forward = FeedForwardConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            stack_config=recurrent,
        ).build()
        inputs = torch.randn(2, model_dim, requires_grad=True)

        output, loss = feed_forward(inputs)
        output.sum().backward()

        self.assertEqual(output.shape, inputs.shape)
        torch.testing.assert_close(loss, torch.tensor(0.0))
        self.assertIsNotNone(inputs.grad)

    def test_feed_forward_accepts_hierarchical_reasoning_model_as_its_recurrent_stack(
        self,
    ) -> None:
        model_dim = 4
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            high_block_config=_linear_stack(model_dim, model_dim),
            low_block_config=_linear_stack(model_dim, model_dim),
            high_cycles=1,
            low_cycles=1,
            initialization_standard_deviation=0.0,
        )
        feed_forward = FeedForwardConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            stack_config=recurrent,
        ).build()
        inputs = torch.randn(2, model_dim, requires_grad=True)

        output, loss = feed_forward(inputs)
        output.sum().backward()

        self.assertEqual(output.shape, inputs.shape)
        torch.testing.assert_close(loss, torch.tensor(0.0))
        self.assertIsNotNone(inputs.grad)

    def test_mixer_accepts_tiny_recursive_model_as_its_token_mixing_recurrence(
        self,
    ) -> None:
        model_dim = 4
        sequence_length = 3
        recurrent = TinyRecursiveModelRecurrentConfig(
            input_dim=sequence_length,
            output_dim=sequence_length,
            block_config=_linear_stack(sequence_length, sequence_length),
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            initialization_standard_deviation=0.0,
        )
        mixer = MixerAttentionConfig(
            embedding_dim=model_dim,
            sequence_length=sequence_length,
            batch_first_flag=True,
            mixing_model_config=recurrent,
        ).build()
        inputs = torch.randn(2, sequence_length, model_dim, requires_grad=True)

        output, weights, loss = mixer(inputs, inputs, inputs)
        output.sum().backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNone(weights)
        self.assertIsNone(loss)
        self.assertIsNotNone(inputs.grad)

    def test_mixer_accepts_hierarchical_reasoning_model_as_its_token_mixing_recurrence(
        self,
    ) -> None:
        model_dim = 4
        sequence_length = 3
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=sequence_length,
            output_dim=sequence_length,
            high_block_config=_linear_stack(sequence_length, sequence_length),
            low_block_config=_linear_stack(sequence_length, sequence_length),
            high_cycles=1,
            low_cycles=1,
            initialization_standard_deviation=0.0,
        )
        mixer = MixerAttentionConfig(
            embedding_dim=model_dim,
            sequence_length=sequence_length,
            batch_first_flag=True,
            mixing_model_config=recurrent,
        ).build()
        inputs = torch.randn(2, sequence_length, model_dim, requires_grad=True)

        output, weights, loss = mixer(inputs, inputs, inputs)
        output.sum().backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNone(weights)
        self.assertIsNone(loss)
        self.assertIsNotNone(inputs.grad)

    def test_transformer_accepts_tiny_recursive_model_as_its_encoder_recurrence(
        self,
    ) -> None:
        model_dim = 4
        sequence_length = 3
        recurrent = TinyRecursiveModelRecurrentConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            block_config=_encoder_block(
                model_dim,
                _self_attention(model_dim, sequence_length),
            ),
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            initialization_standard_deviation=0.0,
        )
        transformer = TransformerConfig(
            encoder_stack_config=recurrent,
            decoder_stack_config=None,
        ).build()
        inputs = torch.randn(
            2,
            sequence_length,
            model_dim,
            requires_grad=True,
        )

        output, loss = transformer(source_token_embeddings=inputs)
        (output.sum() + loss).backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNotNone(inputs.grad)

    def test_transformer_accepts_hierarchical_reasoning_model_as_its_encoder_recurrence(
        self,
    ) -> None:
        model_dim = 4
        sequence_length = 3
        block_config = _encoder_block(
            model_dim,
            _self_attention(model_dim, sequence_length),
        )
        recurrent = HierarchicalReasoningModelRecurrentConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            high_block_config=block_config,
            low_block_config=block_config,
            high_cycles=1,
            low_cycles=1,
            initialization_standard_deviation=0.0,
        )
        transformer = TransformerConfig(
            encoder_stack_config=recurrent,
            decoder_stack_config=None,
        ).build()
        inputs = torch.randn(
            2,
            sequence_length,
            model_dim,
            requires_grad=True,
        )

        output, loss = transformer(source_token_embeddings=inputs)
        (output.sum() + loss).backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNotNone(inputs.grad)

    def test_self_attention_accepts_tiny_recursive_model_for_separate_projections(
        self,
    ) -> None:
        model_dim = 4
        sequence_length = 3
        projection_recurrence = TinyRecursiveModelRecurrentConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            block_config=_linear_stack(model_dim, model_dim),
            latent_updates_per_answer_update=1,
            answer_update_count=1,
            initialization_standard_deviation=0.0,
        )
        attention_config = _self_attention(
            model_dim,
            sequence_length,
            projection_model_config=projection_recurrence,
        )
        attention_config.projection_strategy = SelfAttentionProjectionStrategy.SEPARATE
        attention = attention_config.build()
        inputs = torch.randn(
            2,
            sequence_length,
            model_dim,
            requires_grad=True,
        )

        output, _weights, loss = attention(inputs, inputs, inputs)
        total = output.sum()
        if loss is not None:
            total = total + loss
        total.backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNotNone(inputs.grad)

    def test_self_attention_accepts_hierarchical_reasoning_model_for_separate_projections(
        self,
    ) -> None:
        model_dim = 4
        sequence_length = 3
        projection_recurrence = HierarchicalReasoningModelRecurrentConfig(
            input_dim=model_dim,
            output_dim=model_dim,
            high_block_config=_linear_stack(model_dim, model_dim),
            low_block_config=_linear_stack(model_dim, model_dim),
            high_cycles=1,
            low_cycles=1,
            initialization_standard_deviation=0.0,
        )
        attention_config = _self_attention(
            model_dim,
            sequence_length,
            projection_model_config=projection_recurrence,
        )
        attention_config.projection_strategy = SelfAttentionProjectionStrategy.SEPARATE
        attention = attention_config.build()
        inputs = torch.randn(
            2,
            sequence_length,
            model_dim,
            requires_grad=True,
        )

        output, _weights, loss = attention(inputs, inputs, inputs)
        total = output.sum()
        if loss is not None:
            total = total + loss
        total.backward()

        self.assertEqual(output.shape, inputs.shape)
        self.assertIsNotNone(inputs.grad)


if __name__ == "__main__":
    unittest.main()
