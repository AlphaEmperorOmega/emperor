from emperor.base.layer.residual import ResidualConnectionOptions
import torch
import itertools
import unittest

from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.attention.core.variants.self_attention.config import (
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.attention.core.variants.independent_attention.config import IndependentAttentionConfig
from emperor.transformer.feed_forward.core.config import FeedForwardConfig
from emperor.transformer.core.config import (
    TransformerEncoderLayerConfig,
    TransformerDecoderLayerConfig,
)
from emperor.transformer.core.layers import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


def create_key_padding_mask(
    batch_size: int, source_sequence_length: int
) -> torch.Tensor:
    key_padding_mask_shape = (
        batch_size,
        source_sequence_length,
    )
    key_padding_mask = torch.randint(0, 2, key_padding_mask_shape)
    key_padding_mask = torch.where(
        key_padding_mask > 0,
        torch.tensor(float("-inf")),
        torch.tensor(0.0),
    )
    return key_padding_mask


def create_attention_mask(
    target_sequence_length: int,
    source_sequence_length: int,
    attention_mask_repeat: int = 1,
) -> torch.Tensor:
    attention_mask = torch.triu(
        torch.full((target_sequence_length, source_sequence_length), float("-inf")),
        diagonal=1,
    )
    return attention_mask.unsqueeze(0).repeat(attention_mask_repeat, 1, 1)


class TestTransformerEncoderLayer(unittest.TestCase):
    def preset(
        self,
        embedding_dim: int = 10,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT,
        dropout_probability: float = 0.0,
        residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.RESIDUAL
        ),
        causal_attention_mask_flag: bool = False,
        batch_size: int = 4,
        num_heads: int = 2,
        target_sequence_length: int = 6,
        source_sequence_length: int = 6,
        query_key_projection_dim: int = 0,
        value_projection_dim: int = 0,
        add_key_value_bias_flag: bool = False,
        zero_attention_flag: bool = False,
        average_attention_weights_flag: bool = False,
        feed_forward_hidden_dim: int = 20,
        feed_forward_num_layers: int = 2,
        projection_num_layers: int = 1,
        projection_bias_flag: bool = True,
        projection_model_config: "LayerStackConfig | None" = None,
        feed_forward_stack_config: "LayerStackConfig | None" = None,
        attention_config: "SelfAttentionConfig | None" = None,
        feed_forward_config: "FeedForwardConfig | None" = None,
    ) -> TransformerEncoderLayerConfig:

        if projection_model_config is None:
            projection_model_config = LayerStackConfig(
                input_dim=embedding_dim,
                hidden_dim=embedding_dim,
                output_dim=embedding_dim,
                num_layers=projection_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=projection_bias_flag,
                    ),
                ),
            )

        if attention_config is None:
            attention_config = SelfAttentionConfig(
                batch_size=batch_size,
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
                target_sequence_length=target_sequence_length,
                source_sequence_length=source_sequence_length,
                target_dtype=torch.float32,
                dropout_probability=dropout_probability,
                zero_attention_flag=zero_attention_flag,
                causal_attention_mask_flag=causal_attention_mask_flag,
                add_key_value_bias_flag=add_key_value_bias_flag,
                average_attention_weights_flag=average_attention_weights_flag,
                return_attention_weights_flag=True,
                projection_model_config=projection_model_config,
                projection_strategy=SelfAttentionProjectionStrategy.FUSED,
            )

        if feed_forward_stack_config is None:
            feed_forward_stack_config = LayerStackConfig(
                input_dim=embedding_dim,
                hidden_dim=feed_forward_hidden_dim,
                output_dim=embedding_dim,
                num_layers=feed_forward_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.RELU,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=True,
                    ),
                ),
            )

        if feed_forward_config is None:
            feed_forward_config = FeedForwardConfig(
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                stack_config=feed_forward_stack_config,
            )

        return TransformerEncoderLayerConfig(
            embedding_dim=embedding_dim,
            layer_norm_position=layer_norm_position,
            dropout_probability=dropout_probability,
            residual_connection_option=residual_connection_option,
            causal_attention_mask_flag=causal_attention_mask_flag,
            attention_config=attention_config,
            feed_forward_config=feed_forward_config,
        )

    def test_init(self):
        cfg = self.preset()
        m = TransformerEncoderLayer(cfg)

        self.assertIsInstance(
            m.self_attention_model, cfg.attention_config._registry_owner()
        )
        self.assertIsInstance(
            m.feed_forward_model, cfg.feed_forward_config._registry_owner()
        )
        self.assertEqual(
            m.residual_connection_option,
            cfg.residual_connection_option,
        )

    def test_weighted_residual_uses_separate_parameters_per_encoder_join(self):
        cfg = self.preset(
            residual_connection_option=ResidualConnectionOptions.WEIGHTED_RESIDUAL
        )
        model = TransformerEncoderLayer(cfg)

        self.assertEqual(
            model.self_attention_residual_connection.option,
            ResidualConnectionOptions.WEIGHTED_RESIDUAL,
        )
        self.assertEqual(
            model.feed_forward_residual_connection.option,
            ResidualConnectionOptions.WEIGHTED_RESIDUAL,
        )
        self.assertIsNot(
            model.self_attention_residual_connection.raw_weight,
            model.feed_forward_residual_connection.raw_weight,
        )

    def test_forward_with_different_inputs(self):
        batch_size = 4
        num_heads = 2
        embedding_dim = 10
        sequence_lengths = [6, 10]
        bool_options = [True, False]

        for sequence_length in sequence_lengths:
            for add_key_value_bias_flag in bool_options:
                for zero_attention_flag in bool_options:
                    for average_attention_weights_flag in bool_options:
                        cfg = self.preset(
                            batch_size=batch_size,
                            embedding_dim=embedding_dim,
                            num_heads=num_heads,
                            target_sequence_length=sequence_length,
                            source_sequence_length=sequence_length,
                            add_key_value_bias_flag=add_key_value_bias_flag,
                            zero_attention_flag=zero_attention_flag,
                            average_attention_weights_flag=average_attention_weights_flag,
                        )
                        m = TransformerEncoderLayer(cfg)
                        source_token_embeddings = torch.randn(
                            sequence_length,
                            batch_size,
                            embedding_dim,
                        )

                        key_padding_mask_options = (
                            None,
                            create_key_padding_mask(batch_size, sequence_length),
                        )
                        attention_mask_options = (
                            None,
                            create_attention_mask(
                                sequence_length,
                                sequence_length,
                                batch_size * num_heads,
                            ),
                        )

                        for (
                            key_padding_mask,
                            attention_mask,
                        ) in itertools.product(
                            key_padding_mask_options,
                            attention_mask_options,
                        ):
                            message = f"Test failed for the inputs: sequence_length={sequence_length}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}, average_attention_weights_flag={average_attention_weights_flag}, key_padding_mask={key_padding_mask.shape if key_padding_mask is not None else None}, attention_mask={attention_mask.shape if attention_mask is not None else None}"
                            with self.subTest(i=message):
                                output = m(
                                    source_token_embeddings=source_token_embeddings,
                                    source_key_padding_mask=key_padding_mask,
                                    attention_mask=attention_mask,
                                )

                                expected_output_shape = (
                                    sequence_length,
                                    batch_size,
                                    embedding_dim,
                                )

                                if isinstance(output, tuple):
                                    output, _ = output

                                self.assertEqual(output.shape, expected_output_shape)


class TestTransformerDecoderLayer(unittest.TestCase):
    def preset(
        self,
        embedding_dim: int = 10,
        layer_norm_position: LayerNormPositionOptions = LayerNormPositionOptions.DEFAULT,
        dropout_probability: float = 0.0,
        residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.RESIDUAL
        ),
        causal_attention_mask_flag: bool = False,
        batch_size: int = 4,
        num_heads: int = 2,
        target_sequence_length: int = 6,
        source_sequence_length: int = 6,
        query_key_projection_dim: int = 0,
        value_projection_dim: int = 0,
        add_key_value_bias_flag: bool = False,
        zero_attention_flag: bool = False,
        average_attention_weights_flag: bool = False,
        feed_forward_hidden_dim: int = 20,
        feed_forward_num_layers: int = 2,
        projection_num_layers: int = 1,
        projection_bias_flag: bool = True,
        projection_model_config: "LayerStackConfig | None" = None,
        feed_forward_stack_config: "LayerStackConfig | None" = None,
        self_attention_config: "SelfAttentionConfig | None" = None,
        cross_attention_config: "IndependentAttentionConfig | None" = None,
        feed_forward_config: "FeedForwardConfig | None" = None,
    ) -> TransformerDecoderLayerConfig:

        if projection_model_config is None:
            projection_model_config = LayerStackConfig(
                input_dim=embedding_dim,
                hidden_dim=embedding_dim,
                output_dim=embedding_dim,
                num_layers=projection_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.DISABLED,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=0.0,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=projection_bias_flag,
                    ),
                ),
            )

        if self_attention_config is None:
            self_attention_config = SelfAttentionConfig(
                batch_size=batch_size,
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
                target_sequence_length=target_sequence_length,
                source_sequence_length=target_sequence_length,
                target_dtype=torch.float32,
                dropout_probability=dropout_probability,
                zero_attention_flag=zero_attention_flag,
                causal_attention_mask_flag=causal_attention_mask_flag,
                add_key_value_bias_flag=add_key_value_bias_flag,
                average_attention_weights_flag=average_attention_weights_flag,
                return_attention_weights_flag=True,
                projection_model_config=projection_model_config,
                projection_strategy=SelfAttentionProjectionStrategy.FUSED,
            )

        if cross_attention_config is None:
            cross_attention_config = IndependentAttentionConfig(
                batch_size=batch_size,
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                query_key_projection_dim=query_key_projection_dim,
                value_projection_dim=value_projection_dim,
                target_sequence_length=target_sequence_length,
                source_sequence_length=source_sequence_length,
                target_dtype=torch.float32,
                dropout_probability=dropout_probability,
                zero_attention_flag=zero_attention_flag,
                causal_attention_mask_flag=causal_attention_mask_flag,
                add_key_value_bias_flag=add_key_value_bias_flag,
                average_attention_weights_flag=average_attention_weights_flag,
                return_attention_weights_flag=False,
                projection_model_config=projection_model_config,
            )

        if feed_forward_stack_config is None:
            feed_forward_stack_config = LayerStackConfig(
                input_dim=embedding_dim,
                hidden_dim=feed_forward_hidden_dim,
                output_dim=embedding_dim,
                num_layers=feed_forward_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=ActivationOptions.RELU,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=ResidualConnectionOptions.DISABLED,
                    dropout_probability=dropout_probability,
                    halting_config=None,
                    gate_config=None,
                    layer_model_config=LinearLayerConfig(
                        bias_flag=True,
                    ),
                ),
            )

        if feed_forward_config is None:
            feed_forward_config = FeedForwardConfig(
                input_dim=embedding_dim,
                output_dim=embedding_dim,
                stack_config=feed_forward_stack_config,
            )

        return TransformerDecoderLayerConfig(
            embedding_dim=embedding_dim,
            layer_norm_position=layer_norm_position,
            dropout_probability=dropout_probability,
            residual_connection_option=residual_connection_option,
            causal_attention_mask_flag=causal_attention_mask_flag,
            self_attention_config=self_attention_config,
            cross_attention_config=cross_attention_config,
            feed_forward_config=feed_forward_config,
        )

    def test_init(self):
        cfg = self.preset()
        model = TransformerDecoderLayer(cfg)

        self.assertIsInstance(
            model.self_attention_model,
            cfg.self_attention_config._registry_owner(),
        )
        self.assertIsInstance(
            model.cross_attention_model,
            cfg.cross_attention_config._registry_owner(),
        )
        self.assertIsInstance(
            model.feed_forward_model,
            cfg.feed_forward_config._registry_owner(),
        )
        self.assertEqual(
            model.residual_connection_option,
            cfg.residual_connection_option,
        )

    def test_weighted_residual_uses_separate_parameters_per_decoder_join(self):
        cfg = self.preset(
            residual_connection_option=ResidualConnectionOptions.WEIGHTED_RESIDUAL
        )
        model = TransformerDecoderLayer(cfg)

        residual_connections = [
            model.self_attention_residual_connection,
            model.cross_attention_residual_connection,
            model.feed_forward_residual_connection,
        ]

        self.assertTrue(
            all(
                connection.option == ResidualConnectionOptions.WEIGHTED_RESIDUAL
                for connection in residual_connections
            )
        )
        raw_weights = [connection.raw_weight for connection in residual_connections]
        self.assertEqual(len({id(raw_weight) for raw_weight in raw_weights}), 3)

    def test_forward_with_different_inputs(self):
        batch_size = 4
        num_heads = 2
        embedding_dim = 10
        sequence_lengths = [6, 10]
        bool_options = [True, False]

        for target_sequence_length in sequence_lengths:
            for source_sequence_length in sequence_lengths:
                for add_key_value_bias_flag in bool_options:
                    for zero_attention_flag in bool_options:
                        for average_attention_weights_flag in bool_options:
                            cfg = self.preset(
                                batch_size=batch_size,
                                embedding_dim=embedding_dim,
                                num_heads=num_heads,
                                target_sequence_length=target_sequence_length,
                                source_sequence_length=source_sequence_length,
                                add_key_value_bias_flag=add_key_value_bias_flag,
                                zero_attention_flag=zero_attention_flag,
                                average_attention_weights_flag=average_attention_weights_flag,
                            )
                            m = TransformerDecoderLayer(cfg)
                            target_token_embeddings = torch.randn(
                                target_sequence_length,
                                batch_size,
                                embedding_dim,
                            )
                            encoder_output = torch.randn(
                                source_sequence_length,
                                batch_size,
                                embedding_dim,
                            )
                            key_padding_mask_options = (
                                None,
                                create_key_padding_mask(
                                    batch_size, target_sequence_length
                                ),
                            )
                            encoder_padding_mask_options = (
                                None,
                                create_key_padding_mask(
                                    batch_size, source_sequence_length
                                ),
                            )
                            attention_mask_options = (
                                None,
                                create_attention_mask(
                                    target_sequence_length,
                                    target_sequence_length,
                                    batch_size * num_heads,
                                ),
                            )
                            encoder_attention_mask_options = (
                                None,
                                create_attention_mask(
                                    target_sequence_length,
                                    source_sequence_length,
                                    batch_size * num_heads,
                                ),
                            )

                            for (
                                key_padding_mask,
                                encoder_padding_mask,
                                attention_mask,
                                encoder_attention_mask,
                            ) in itertools.product(
                                key_padding_mask_options,
                                encoder_padding_mask_options,
                                attention_mask_options,
                                encoder_attention_mask_options,
                            ):
                                message = f"Test failed for the inputs: target_sequence_length={target_sequence_length}, source_sequence_length={source_sequence_length}, add_key_value_bias_flag={add_key_value_bias_flag}, zero_attention_flag={zero_attention_flag}, average_attention_weights_flag={average_attention_weights_flag}, key_padding_mask={key_padding_mask.shape if key_padding_mask is not None else None}, encoder_padding_mask={encoder_padding_mask.shape if encoder_padding_mask is not None else None}, attention_mask={attention_mask.shape if attention_mask is not None else None}, encoder_attention_mask={encoder_attention_mask.shape if encoder_attention_mask is not None else None}"
                                with self.subTest(i=message):
                                    output = m(
                                        target_token_embeddings=target_token_embeddings,
                                        encoder_output=encoder_output,
                                        key_padding_mask=key_padding_mask,
                                        encoder_padding_mask=encoder_padding_mask,
                                        attention_mask=attention_mask,
                                        encoder_attention_mask=encoder_attention_mask,
                                    )
                                    expected_output = (
                                        target_sequence_length,
                                        batch_size,
                                        embedding_dim,
                                    )

                                    if isinstance(output, tuple):
                                        output, _ = output

                                    self.assertEqual(output.shape, expected_output)
