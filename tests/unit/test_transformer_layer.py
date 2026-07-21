import itertools
import unittest

import torch

from emperor.attention import (
    IndependentAttentionConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    ResidualConfig,
    ResidualConnectionOptions,
)
from emperor.linears import LinearLayerConfig
from emperor.transformer import (
    FeedForwardConfig,
    TransformerDecoderLayer,
    TransformerDecoderLayerConfig,
    TransformerEncoderLayer,
    TransformerEncoderLayerConfig,
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
        residual_model_config: LinearLayerConfig | None = None,
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
                    residual_config=None,
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
                    residual_config=None,
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
            residual_config=None
            if residual_connection_option is None
            else ResidualConfig(
                option=residual_connection_option, model_config=residual_model_config
            ),
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
            m.residual_config,
            cfg.residual_config,
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

    def test_data_dependent_residual_uses_separate_models_per_encoder_join(self):
        residual_model_config = LinearLayerConfig(
            input_dim=99,
            output_dim=99,
            bias_flag=True,
        )
        cfg = self.preset(
            residual_connection_option=ResidualConnectionOptions.WEIGHTED_BLEND,
            residual_model_config=residual_model_config,
        )
        model = TransformerEncoderLayer(cfg)
        residual_connections = [
            model.self_attention_residual_connection,
            model.feed_forward_residual_connection,
        ]
        coefficient_models = [connection.model for connection in residual_connections]

        self.assertEqual(len({id(model) for model in coefficient_models}), 2)
        for connection in residual_connections:
            self.assertEqual(connection.model_config, residual_model_config)
            self.assertIsNot(connection.model_config, residual_model_config)
        for coefficient_model in coefficient_models:
            self.assertEqual(coefficient_model.input_dim, cfg.embedding_dim * 2)
            self.assertEqual(coefficient_model.output_dim, cfg.embedding_dim)

    def test_implicit_causal_mask_with_padding_matches_explicit_sequence_mask(self):
        torch.manual_seed(1701)
        model = TransformerEncoderLayer(self.preset(causal_attention_mask_flag=True))
        model.eval()
        source = torch.arange(6 * 4 * 10, dtype=torch.float32).reshape(6, 4, 10)
        source = source / source.numel()
        padding_mask = torch.tensor(
            [
                [False, False, False, False, False, True],
                [False, False, True, False, False, False],
                [False, True, False, False, True, False],
                [False, False, False, True, False, False],
            ]
        )
        explicit_causal_mask = torch.triu(
            torch.ones(6, 6, dtype=torch.bool),
            diagonal=1,
        )

        implicit_output, implicit_loss = model(
            source_token_embeddings=source,
            source_key_padding_mask=padding_mask,
        )
        explicit_output, explicit_loss = model(
            source_token_embeddings=source,
            source_key_padding_mask=padding_mask,
            attention_mask=explicit_causal_mask,
        )

        torch.testing.assert_close(implicit_output, explicit_output)
        torch.testing.assert_close(implicit_loss, explicit_loss)
        self.assertTrue(torch.isfinite(implicit_output).all())
        self.assertTrue(torch.isfinite(implicit_loss))

    def test_batch_first_implicit_causal_mask_matches_explicit_and_gradients(self):
        torch.manual_seed(1702)
        config = self.preset(
            embedding_dim=4,
            batch_size=2,
            num_heads=2,
            target_sequence_length=5,
            source_sequence_length=5,
            feed_forward_hidden_dim=8,
            causal_attention_mask_flag=True,
        )
        config.attention_config.batch_first_flag = True
        implicit_model = TransformerEncoderLayer(config).eval()
        explicit_model = TransformerEncoderLayer(config).eval()
        explicit_model.load_state_dict(implicit_model.state_dict(), strict=True)
        source_values = torch.tensor(
            (
                (
                    (0.10, -0.20, 0.30, -0.40),
                    (0.50, -0.60, 0.70, -0.80),
                    (0.90, 0.15, -0.25, 0.35),
                    (-0.45, 0.55, -0.65, 0.75),
                    (0.85, -0.95, 0.05, -0.15),
                ),
                (
                    (-0.12, 0.22, -0.32, 0.42),
                    (-0.52, 0.62, -0.72, 0.82),
                    (-0.92, -0.18, 0.28, -0.38),
                    (0.48, -0.58, 0.68, -0.78),
                    (-0.88, 0.98, -0.08, 0.18),
                ),
            ),
            dtype=torch.float32,
        )
        implicit_source = source_values.clone().requires_grad_()
        explicit_source = source_values.clone().requires_grad_()
        padding_mask = torch.tensor(
            (
                (False, False, False, True, False),
                (False, False, True, False, False),
            )
        )
        explicit_causal_mask = torch.triu(
            torch.ones(5, 5, dtype=torch.bool),
            diagonal=1,
        )

        implicit_output, implicit_loss = implicit_model(
            implicit_source,
            source_key_padding_mask=padding_mask,
        )
        explicit_output, explicit_loss = explicit_model(
            explicit_source,
            source_key_padding_mask=padding_mask,
            attention_mask=explicit_causal_mask,
        )

        torch.testing.assert_close(implicit_output, explicit_output)
        torch.testing.assert_close(implicit_loss, explicit_loss)
        self.assertEqual(implicit_output.shape, (2, 5, 4))
        self.assertTrue(torch.isfinite(implicit_output).all())
        self.assertTrue(torch.isfinite(implicit_loss))
        (implicit_output.square().sum() + implicit_loss).backward()
        (explicit_output.square().sum() + explicit_loss).backward()
        torch.testing.assert_close(implicit_source.grad, explicit_source.grad)
        self.assertGreater(torch.count_nonzero(implicit_source.grad).item(), 0)
        for (implicit_name, implicit_parameter), (
            explicit_name,
            explicit_parameter,
        ) in zip(
            implicit_model.named_parameters(),
            explicit_model.named_parameters(),
            strict=True,
        ):
            with self.subTest(parameter=implicit_name):
                self.assertEqual(implicit_name, explicit_name)
                torch.testing.assert_close(
                    implicit_parameter.grad,
                    explicit_parameter.grad,
                )
                self.assertTrue(torch.isfinite(implicit_parameter.grad).all())

        changed_future = source_values.clone()
        changed_future[:, -1] += torch.tensor((3.0, -4.0, 5.0, -6.0))
        baseline_output, _ = implicit_model(
            source_values,
            source_key_padding_mask=padding_mask,
        )
        changed_output, _ = implicit_model(
            changed_future,
            source_key_padding_mask=padding_mask,
        )
        torch.testing.assert_close(
            changed_output[:, :-1],
            baseline_output[:, :-1],
        )
        self.assertFalse(torch.allclose(changed_output[:, -1], baseline_output[:, -1]))

    def test_unbatched_implicit_causal_mask_uses_sequence_dimension(self):
        torch.manual_seed(1703)
        config = self.preset(
            embedding_dim=4,
            batch_size=1,
            num_heads=2,
            target_sequence_length=5,
            source_sequence_length=5,
            feed_forward_hidden_dim=8,
            causal_attention_mask_flag=True,
        )
        model = TransformerEncoderLayer(config).eval()
        source = torch.tensor(
            (
                (0.10, -0.20, 0.30, -0.40),
                (0.50, -0.60, 0.70, -0.80),
                (0.90, 0.15, -0.25, 0.35),
                (-0.45, 0.55, -0.65, 0.75),
                (0.85, -0.95, 0.05, -0.15),
            ),
            requires_grad=True,
        )
        padding_mask = torch.tensor((False, False, True, False, False))
        explicit_causal_mask = torch.triu(
            torch.ones(5, 5, dtype=torch.bool),
            diagonal=1,
        )

        implicit_output, implicit_loss = model(
            source,
            source_key_padding_mask=padding_mask,
        )
        explicit_output, explicit_loss = model(
            source,
            source_key_padding_mask=padding_mask,
            attention_mask=explicit_causal_mask,
        )

        torch.testing.assert_close(implicit_output, explicit_output)
        torch.testing.assert_close(implicit_loss, explicit_loss)
        self.assertEqual(implicit_output.shape, (5, 4))
        self.assertTrue(torch.isfinite(implicit_output).all())
        (implicit_output.square().sum() + implicit_loss).backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(torch.count_nonzero(source.grad).item(), 0)

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
        residual_model_config: LinearLayerConfig | None = None,
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
                    residual_config=None,
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
                    residual_config=None,
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
            residual_config=None
            if residual_connection_option is None
            else ResidualConfig(
                option=residual_connection_option, model_config=residual_model_config
            ),
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
            model.residual_config,
            cfg.residual_config,
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

    def test_data_dependent_residual_uses_separate_models_per_decoder_join(self):
        residual_model_config = LinearLayerConfig(
            input_dim=99,
            output_dim=99,
            bias_flag=True,
        )
        cfg = self.preset(
            residual_connection_option=ResidualConnectionOptions.WEIGHTED_BLEND,
            residual_model_config=residual_model_config,
        )
        model = TransformerDecoderLayer(cfg)
        residual_connections = [
            model.self_attention_residual_connection,
            model.cross_attention_residual_connection,
            model.feed_forward_residual_connection,
        ]
        coefficient_models = [connection.model for connection in residual_connections]

        self.assertEqual(len({id(model) for model in coefficient_models}), 3)
        for connection in residual_connections:
            self.assertEqual(connection.model_config, residual_model_config)
            self.assertIsNot(connection.model_config, residual_model_config)
        for coefficient_model in coefficient_models:
            self.assertEqual(coefficient_model.input_dim, cfg.embedding_dim * 2)
            self.assertEqual(coefficient_model.output_dim, cfg.embedding_dim)

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
