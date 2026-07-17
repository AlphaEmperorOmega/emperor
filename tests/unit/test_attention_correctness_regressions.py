import unittest
from dataclasses import replace

import torch

from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
    SelfAttentionProjectionStrategy,
)
from emperor.embedding.relative import DynamicPositionalBiasConfig
from support.attention import build_attention_config, make_router_config


class TestIndependentRelativePositioning(unittest.TestCase):
    def model(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        model = config.build()
        with torch.no_grad():
            for projection in (
                model.projector.query_model,
                model.projector.key_model,
                model.projector.value_model,
                model.projector.output_model,
            ):
                layer = projection.layers[0].model
                layer.weight_params.copy_(torch.eye(2))
                layer.bias_params.zero_()

            relative = model.processor.relative_positional_embedding
            relative.relative_positional_embeddings.zero_()
            positive_one_index = relative.max_positions + 1
            relative.relative_positional_embeddings[0, 0, positive_one_index] = 2.0
        return model

    def test_relative_logits_change_attention_by_the_expected_amount(self):
        model = self.model()

        query = torch.eye(2).unsqueeze(1)
        output, weights, auxiliary_loss = model(
            query,
            query,
            query,
            attention_mask=torch.zeros(2, 2),
        )

        scale = 2**-0.5
        scores = torch.tensor([[scale, 2 * scale], [0.0, scale]], dtype=query.dtype)
        expected = torch.softmax(scores, dim=-1).unsqueeze(1)
        torch.testing.assert_close(output, expected)
        self.assertIsNone(weights)
        self.assertIsNone(auxiliary_loss)

    def test_relative_parameters_and_query_receive_nonzero_gradients(self):
        model = self.model()
        query = torch.eye(2).unsqueeze(1).requires_grad_()

        output, _, _ = model(query, query, query)
        output[0, 0, 0].backward()

        relative = model.processor.relative_positional_embedding
        relative_grad = relative.relative_positional_embeddings.grad
        self.assertIsNotNone(relative_grad)
        self.assertTrue(torch.any(relative_grad.abs() > 0))
        self.assertIsNotNone(query.grad)
        self.assertTrue(torch.any(query.grad.abs() > 0))


class TestFullyMaskedAttentionRows(unittest.TestCase):
    def test_independent_attention_returns_zero_for_a_fully_masked_query_row(
        self,
    ) -> None:
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
        )
        model = config.build().eval()
        with torch.no_grad():
            for projection in (
                model.projector.query_model,
                model.projector.key_model,
                model.projector.value_model,
                model.projector.output_model,
            ):
                layer = projection.layers[0].model
                layer.weight_params.copy_(torch.eye(2))
                layer.bias_params.zero_()
        inputs = torch.eye(2).unsqueeze(1).requires_grad_()
        attention_mask = torch.tensor(
            [[True, True], [False, False]],
        )
        unmasked_probabilities = torch.softmax(
            torch.tensor([0.0, 2**-0.5]),
            dim=0,
        )
        expected_output = torch.tensor(
            [
                [[0.0, 0.0]],
                [
                    [
                        unmasked_probabilities[0],
                        unmasked_probabilities[1],
                    ]
                ],
            ]
        )

        output, weights, auxiliary_loss = model(
            inputs,
            inputs,
            inputs,
            attention_mask=attention_mask,
        )

        torch.testing.assert_close(output, expected_output)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNone(weights)
        self.assertIsNone(auxiliary_loss)

        output[1, 0, 1].backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.any(inputs.grad.abs() > 0))

    def test_self_attention_returns_zero_for_a_fully_masked_query_row(self) -> None:
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            return_attention_weights_flag=True,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        model = config.build().eval()
        with torch.no_grad():
            for projection in (
                model.projector.query_model,
                model.projector.key_model,
                model.projector.value_model,
                model.projector.output_model,
            ):
                layer = projection.layers[0].model
                layer.weight_params.copy_(torch.eye(2))
                layer.bias_params.zero_()
        inputs = torch.eye(2).unsqueeze(1).requires_grad_()
        attention_mask = torch.tensor(
            [[True, True], [False, False]],
        )
        probability = torch.softmax(
            torch.tensor([0.0, 2**-0.5]),
            dim=0,
        )
        expected_weights = torch.tensor(
            [[[[0.0, 0.0], probability.tolist()]]],
        )
        expected_output = torch.tensor(
            [
                [[0.0, 0.0]],
                [[probability[0], probability[1]]],
            ]
        )

        output, weights, auxiliary_loss = model(
            inputs,
            inputs,
            inputs,
            attention_mask=attention_mask,
        )

        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(output, expected_output)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNone(auxiliary_loss)

        output[1, 0, 1].backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.any(inputs.grad.abs() > 0))

    def test_mixture_attention_returns_zero_for_a_fully_masked_query_row(
        self,
    ) -> None:
        config = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            use_kv_expert_models_flag=False,
            experts_top_k=1,
            experts_num_experts=2,
            experts_stack_num_layers=1,
        )
        model = config.build().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for expert in model.projector.query_model.expert_modules:
                expert.layers[0].model.weight_params.copy_(torch.eye(2))
            for expert in model.projector.output_model.expert_modules:
                expert.layers[0].model.weight_params.copy_(torch.eye(2))
            model.projector.key_model.layers[0].model.weight_params.copy_(torch.eye(2))
            model.projector.value_model.layers[0].model.weight_params.copy_(
                torch.eye(2)
            )
        inputs = torch.eye(2).unsqueeze(1).requires_grad_()
        attention_mask = torch.tensor(
            [[True, True], [False, False]],
        )
        unmasked_probabilities = 0.5 * torch.softmax(
            torch.tensor([0.0, 2**-0.5]),
            dim=0,
        )
        expected_output = torch.tensor(
            [
                [[0.0, 0.0]],
                [
                    [
                        unmasked_probabilities[0],
                        unmasked_probabilities[1],
                    ]
                ],
            ]
        )

        output, weights, auxiliary_loss = model(
            inputs,
            inputs,
            inputs,
            attention_mask=attention_mask,
        )

        torch.testing.assert_close(output, expected_output)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNone(weights)
        torch.testing.assert_close(auxiliary_loss, torch.tensor(0.0))

        output[1, 0, 1].backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.any(inputs.grad.abs() > 0))


class TestStaticProjectionContracts(unittest.TestCase):
    def test_rank_two_static_key_is_rejected_with_a_contract_error(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=2,
        )
        model = config.build()
        query = torch.randn(1, 1, 2)

        with self.assertRaisesRegex(RuntimeError, "static_keys must be rank 3"):
            model(
                query,
                query,
                query,
                static_k=torch.randn(1, 2),
                static_v=torch.randn(1, 2, 2),
            )

    def test_static_projection_dtype_must_match_runtime_inputs(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=1,
        )
        model = config.build()
        query = torch.randn(1, 1, 2)
        static = torch.randn(1, 1, 2, dtype=torch.float64)

        with self.assertRaisesRegex(
            RuntimeError,
            "static_keys dtype must match query dtype",
        ):
            model(
                query,
                query,
                query,
                static_k=static,
                static_v=static,
            )

    def test_selected_static_key_and_value_lengths_must_match(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=3,
        )
        model = config.build()
        query = torch.randn(1, 1, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "Selected key and value sources must have equal sequence lengths",
        ):
            model(
                query,
                query,
                query,
                static_k=torch.randn(1, 2, 2),
                static_v=torch.randn(1, 3, 2),
            )

    def test_static_values_use_the_value_projection_head_width(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            query_key_projection_dim=4,
            value_projection_dim=6,
            target_sequence_length=2,
            source_sequence_length=4,
        )
        model = config.build()
        query = torch.randn(2, 2, 4)
        key = torch.randn(3, 2, 4)
        value = torch.randn(3, 2, 4)
        static_key = torch.randn(4, 4, 2)
        static_value = torch.randn(4, 4, 3)

        output, weights, auxiliary_loss = model(
            query,
            key,
            value,
            static_k=static_key,
            static_v=static_value,
        )

        self.assertEqual(output.shape, (2, 2, 4))
        self.assertIsNone(weights)
        self.assertIsNone(auxiliary_loss)

    def test_masks_describe_the_selected_static_source(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=4,
        )
        model = config.build()
        query = torch.randn(2, 1, 2)
        dynamic_source = torch.randn(2, 1, 2)
        static_key = torch.randn(1, 3, 2)
        static_value = torch.randn(1, 3, 2)
        static_source_mask = torch.zeros(2, 3, dtype=torch.bool)

        output, _, _ = model(
            query,
            dynamic_source,
            dynamic_source,
            attention_mask=static_source_mask,
            static_k=static_key,
            static_v=static_value,
        )

        self.assertEqual(output.shape, (2, 1, 2))

    def test_learnable_bias_is_appended_to_static_key_and_value(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
            add_key_value_bias_flag=True,
        )
        model = config.build()
        with torch.no_grad():
            for projection in (
                model.projector.query_model,
                model.projector.key_model,
                model.projector.value_model,
                model.projector.output_model,
            ):
                layer = projection.layers[0].model
                layer.weight_params.fill_(1.0)
                layer.bias_params.zero_()
            model.bias.key_bias_vector.zero_()
            model.bias.value_bias_vector.fill_(2.0)

        query = torch.ones(1, 1, 1)
        static_key = torch.zeros(1, 1, 1)
        static_value = torch.zeros(1, 1, 1)

        output, _, _ = model(
            query,
            query,
            query,
            static_k=static_key,
            static_v=static_value,
        )

        torch.testing.assert_close(output, torch.ones_like(output))

    def test_relative_logits_are_zero_for_the_synthetic_bias_position(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
            add_key_value_bias_flag=True,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        model = config.build()
        with torch.no_grad():
            for projection in (
                model.projector.query_model,
                model.projector.key_model,
                model.projector.value_model,
                model.projector.output_model,
            ):
                layer = projection.layers[0].model
                layer.weight_params.fill_(1.0)
                layer.bias_params.zero_()
            model.bias.key_bias_vector.zero_()
            model.bias.value_bias_vector.fill_(2.0)
            relative = model.processor.relative_positional_embedding
            relative.relative_positional_embeddings.fill_(3.0)

        query = torch.ones(1, 1, 1)
        source = torch.zeros(1, 1, 1)

        output, _, _ = model(query, source, source)

        expected = torch.tensor([[[2.0 / (torch.exp(torch.tensor(3.0)) + 1.0)]]])
        torch.testing.assert_close(output, expected)

    def test_relative_static_bias_and_zero_attention_compose_exactly(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=2,
            add_key_value_bias_flag=True,
            zero_attention_flag=True,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        model = config.build()
        with torch.no_grad():
            for projection in (
                model.projector.query_model,
                model.projector.key_model,
                model.projector.value_model,
                model.projector.output_model,
            ):
                layer = projection.layers[0].model
                layer.weight_params.fill_(1.0)
                layer.bias_params.zero_()
            model.bias.key_bias_vector.zero_()
            model.bias.value_bias_vector.fill_(3.0)
            relative = model.processor.relative_positional_embedding
            relative.relative_positional_embeddings.zero_()
            relative.relative_positional_embeddings[0, 0, relative.max_positions] = 3.0

        query = torch.ones(1, 1, 1)
        static_key = torch.tensor([[[0.0], [1.0]]])
        static_value = torch.tensor([[[0.0], [2.0]]])
        output, _, _ = model(
            query,
            query,
            query,
            static_k=static_key,
            static_v=static_value,
        )

        exp_one = torch.exp(torch.tensor(1.0))
        exp_three = torch.exp(torch.tensor(3.0))
        expected = torch.full_like(
            output,
            (2.0 * exp_one + 3.0) / (exp_three + exp_one + 2.0),
        )
        torch.testing.assert_close(output, expected)


class TestSelfAttentionSourceExtensions(unittest.TestCase):
    def test_unbatched_inputs_return_unbatched_per_head_weights(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=2,
            source_sequence_length=2,
            return_attention_weights_flag=True,
        )
        model = replace(config, batch_first_flag=False).build()
        value = torch.randn(2, 4)

        output, weights, _ = model(value, value, value)

        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(weights.shape, (2, 2, 2))

    def test_relative_positioning_and_key_value_bias_compose_exactly(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
            add_key_value_bias_flag=True,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        model = config.build()
        with torch.no_grad():
            qkv_layer = model.projector.qkv_model.layers[0].model
            qkv_layer.weight_params.fill_(1.0)
            qkv_layer.bias_params.zero_()
            output_layer = model.projector.output_model.layers[0].model
            output_layer.weight_params.fill_(1.0)
            output_layer.bias_params.zero_()
            model.bias.key_bias_vector.zero_()
            model.bias.value_bias_vector.fill_(2.0)
            relative = model.processor.relative_positional_embedding
            relative.relative_positional_embeddings.fill_(3.0)

        value = torch.ones(1, 1, 1)
        output, _, _ = model(value, value, value)

        exponential = torch.exp(torch.tensor(4.0))
        expected = torch.full_like(output, (exponential + 2.0) / (exponential + 1.0))
        torch.testing.assert_close(output, expected)

    def test_fused_and_separate_relative_paths_preserve_named_gradients(self):
        outputs = {}
        weights_by_strategy = {}
        input_gradients = {}
        relative_gradients = {}
        for strategy in (
            SelfAttentionProjectionStrategy.FUSED,
            SelfAttentionProjectionStrategy.SEPARATE,
        ):
            with self.subTest(strategy=strategy):
                config = build_attention_config(
                    config_class=SelfAttentionConfig,
                    batch_size=2,
                    num_heads=1,
                    embedding_dim=2,
                    target_sequence_length=3,
                    source_sequence_length=3,
                    return_attention_weights_flag=True,
                    self_attention_projection_strategy=strategy,
                    relative_positional_embedding_config_cls=(
                        DynamicPositionalBiasConfig
                    ),
                )
                model = replace(
                    config,
                    target_dtype=torch.float64,
                    batch_first_flag=False,
                ).build()
                model.eval()
                with torch.no_grad():
                    identity = torch.eye(2, dtype=torch.float64)
                    if strategy is SelfAttentionProjectionStrategy.FUSED:
                        qkv_layer = model.projector.qkv_model.layers[0].model
                        qkv_layer.weight_params.copy_(
                            torch.cat((identity, identity, identity), dim=1)
                        )
                        qkv_layer.bias_params.zero_()
                    else:
                        for projection in (
                            model.projector.query_model,
                            model.projector.key_model,
                            model.projector.value_model,
                        ):
                            layer = projection.layers[0].model
                            layer.weight_params.copy_(identity)
                            layer.bias_params.zero_()
                    output_layer = model.projector.output_model.layers[0].model
                    output_layer.weight_params.copy_(identity)
                    output_layer.bias_params.zero_()
                    relative = model.processor.relative_positional_embedding
                    relative_values = torch.arange(
                        relative.relative_positional_embeddings.numel(),
                        dtype=torch.float64,
                    ).reshape_as(relative.relative_positional_embeddings)
                    relative.relative_positional_embeddings.copy_(
                        relative_values / 37.0
                    )

                inputs = (
                    torch.arange(12, dtype=torch.float64)
                    .reshape(3, 2, 2)
                    .div_(7.0)
                    .sub_(0.5)
                    .requires_grad_()
                )
                output, weights, auxiliary_loss = model(inputs, inputs, inputs)
                marker = torch.arange(
                    1,
                    output.numel() + 1,
                    dtype=output.dtype,
                ).reshape_as(output)
                (output * marker).sum().backward()

                relative_gradient = relative.relative_positional_embeddings.grad
                self.assertIsNotNone(relative_gradient)
                self.assertTrue(torch.isfinite(relative_gradient).all())
                self.assertTrue(torch.any(relative_gradient.abs() > 0))
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertTrue(torch.any(inputs.grad.abs() > 0))
                self.assertIsNone(auxiliary_loss)

                outputs[strategy] = output.detach()
                weights_by_strategy[strategy] = weights.detach()
                input_gradients[strategy] = inputs.grad.detach()
                relative_gradients[strategy] = relative_gradient.detach()

        torch.testing.assert_close(
            outputs[SelfAttentionProjectionStrategy.FUSED],
            outputs[SelfAttentionProjectionStrategy.SEPARATE],
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            weights_by_strategy[SelfAttentionProjectionStrategy.FUSED],
            weights_by_strategy[SelfAttentionProjectionStrategy.SEPARATE],
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            input_gradients[SelfAttentionProjectionStrategy.FUSED],
            input_gradients[SelfAttentionProjectionStrategy.SEPARATE],
            rtol=1e-11,
            atol=1e-11,
        )
        torch.testing.assert_close(
            relative_gradients[SelfAttentionProjectionStrategy.FUSED],
            relative_gradients[SelfAttentionProjectionStrategy.SEPARATE],
            rtol=1e-11,
            atol=1e-11,
        )

    def test_relative_positioning_supports_a_longer_static_source_exactly(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=2,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        model = config.build()
        with torch.no_grad():
            qkv_layer = model.projector.qkv_model.layers[0].model
            qkv_layer.weight_params.fill_(1.0)
            qkv_layer.bias_params.zero_()
            output_layer = model.projector.output_model.layers[0].model
            output_layer.weight_params.fill_(1.0)
            output_layer.bias_params.zero_()
            relative = model.processor.relative_positional_embedding
            relative.relative_positional_embeddings.zero_()
            relative.relative_positional_embeddings[0, 0, relative.max_positions] = 3.0

        value = torch.ones(1, 1, 1)
        static_key = torch.zeros(1, 2, 1)
        static_value = torch.tensor([[[0.0], [2.0]]])
        output, _, _ = model(
            value,
            value,
            value,
            static_k=static_key,
            static_v=static_value,
        )

        expected = torch.full_like(
            output,
            2.0 / (torch.exp(torch.tensor(3.0)) + 1.0),
        )
        torch.testing.assert_close(output, expected)

    def test_relative_positioning_bias_and_zero_attention_compose_exactly(self):
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
            add_key_value_bias_flag=True,
            zero_attention_flag=True,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        model = config.build()
        with torch.no_grad():
            qkv_layer = model.projector.qkv_model.layers[0].model
            qkv_layer.weight_params.fill_(1.0)
            qkv_layer.bias_params.zero_()
            output_layer = model.projector.output_model.layers[0].model
            output_layer.weight_params.fill_(1.0)
            output_layer.bias_params.zero_()
            model.bias.key_bias_vector.zero_()
            model.bias.value_bias_vector.fill_(3.0)
            model.processor.relative_positional_embedding.relative_positional_embeddings.fill_(
                3.0
            )

        value = torch.ones(1, 1, 1)
        output, _, _ = model(value, value, value)

        exponential = torch.exp(torch.tensor(4.0))
        expected = torch.full_like(
            output,
            (exponential + 3.0) / (exponential + 2.0),
        )
        torch.testing.assert_close(output, expected)


class TestMixtureOfAttentionHeadsSourceExtensions(unittest.TestCase):
    def test_synthetic_positions_have_no_relative_embedding(self):
        cases = (
            ("bias", True, False, (1,)),
            ("zero", False, True, (1,)),
            ("bias_and_zero", True, True, (1, 2)),
        )
        for name, add_bias, add_zero, synthetic_offsets in cases:
            with self.subTest(name=name):
                config = build_attention_config(
                    config_class=MixtureOfAttentionHeadsConfig,
                    batch_size=1,
                    num_heads=1,
                    embedding_dim=1,
                    target_sequence_length=1,
                    source_sequence_length=1,
                    add_key_value_bias_flag=add_bias,
                    zero_attention_flag=add_zero,
                    relative_positional_embedding_config_cls=(
                        DynamicPositionalBiasConfig
                    ),
                    experts_top_k=1,
                    experts_num_experts=2,
                )
                model = config.build()
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(0.5)
                    if add_bias:
                        model.bias.key_bias_vector.zero_()
                        model.bias.value_bias_vector.fill_(2.0)
                    relative = model.processor.relative_positional_embedding
                    relative.relative_positional_embeddings.zero_()

                value = torch.ones(1, 1, 1)
                baseline, _, _ = model(value, value, value)

                with torch.no_grad():
                    for offset in synthetic_offsets:
                        relative.relative_positional_embeddings[
                            0, 0, relative.max_positions + offset
                        ] = 10.0
                changed_table, _, _ = model(value, value, value)

                torch.testing.assert_close(changed_table, baseline)

    def test_relative_static_bias_and_zero_attention_compose_with_gradients(self):
        config = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=2,
            add_key_value_bias_flag=True,
            zero_attention_flag=True,
            use_kv_expert_models_flag=False,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
            experts_top_k=1,
            experts_num_experts=2,
        )
        model = config.build()
        value = torch.ones(1, 1, 1, requires_grad=True)
        static_key = torch.tensor([[[0.0], [1.0]]])
        static_value = torch.tensor([[[0.0], [2.0]]])

        output, _, auxiliary_loss = model(
            value,
            value,
            value,
            static_k=static_key,
            static_v=static_value,
        )
        loss = output.square().sum()
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
        loss.backward()

        relative = model.processor.relative_positional_embedding
        relative_grad = relative.relative_positional_embeddings.grad
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(relative_grad)
        self.assertTrue(torch.any(relative_grad.abs() > 0))


class TestMixtureRoutingStateLifecycle(unittest.TestCase):
    def test_invalid_padding_mask_is_rejected_before_noisy_routing_uses_rng(
        self,
    ) -> None:
        config = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=3,
            source_sequence_length=3,
            experts_top_k=1,
            experts_num_experts=2,
            experts_stack_num_layers=1,
        )
        experts_config = config.experts_config
        sampler_config = experts_config.sampler_config
        config = replace(
            config,
            experts_config=replace(
                experts_config,
                sampler_config=replace(
                    sampler_config,
                    noisy_topk_flag=True,
                    router_config=make_router_config(
                        input_dim=4,
                        num_experts=2,
                        noisy_topk_flag=True,
                    ),
                ),
            ),
        )
        model = config.build().train()
        inputs = torch.arange(24.0).reshape(3, 2, 4).div_(7.0)
        invalid_padding_mask = torch.zeros(2, 4, dtype=torch.bool)

        torch.manual_seed(919)
        expected_rng_state = torch.random.get_rng_state().clone()
        with self.assertRaises(RuntimeError) as caught:
            model(
                inputs,
                inputs,
                inputs,
                k_padding_mask=invalid_padding_mask,
            )

        self.assertEqual(
            str(caught.exception),
            "key_padding_mask must have shape (2, 3), got (2, 4).",
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(),
            expected_rng_state,
            rtol=0,
            atol=0,
        )

    def test_public_forward_clears_routing_graph_after_successful_calls(
        self,
    ) -> None:
        torch.manual_seed(811)
        config = build_attention_config(
            config_class=MixtureOfAttentionHeadsConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            experts_top_k=1,
            experts_num_experts=2,
            experts_stack_num_layers=1,
        )
        model = config.build().eval()
        inputs = torch.tensor(
            [[[1.0, -0.5]], [[2.0, 3.0]]],
            requires_grad=True,
        )

        output, weights, auxiliary_loss = model(inputs, inputs, inputs)

        self.assertIsNone(weights)
        self.assertIsNotNone(auxiliary_loss)
        self.assertIsNone(model.projector.probabilities)
        self.assertIsNone(model.projector.indices)
        self.assertIsNone(model.projector.skip_mask)
        objective = output.square().sum() + auxiliary_loss
        objective.backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.any(inputs.grad.abs() > 0))

        repeated_output, repeated_weights, repeated_auxiliary_loss = model(
            inputs.detach(),
            inputs.detach(),
            inputs.detach(),
        )
        torch.testing.assert_close(repeated_output, output.detach())
        self.assertIsNone(repeated_weights)
        torch.testing.assert_close(repeated_auxiliary_loss, auxiliary_loss.detach())
        self.assertIsNone(model.projector.probabilities)
        self.assertIsNone(model.projector.indices)
        self.assertIsNone(model.projector.skip_mask)


class TestAttentionConfigurationValidation(unittest.TestCase):
    def test_invalid_head_dimensions_are_rejected_before_arithmetic_and_rng(
        self,
    ) -> None:
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=1,
        )
        cases = (
            (
                "missing_num_heads",
                {"num_heads": None},
                ValueError,
                "num_heads is required for IndependentAttentionConfig, received None",
            ),
            (
                "zero_num_heads",
                {"num_heads": 0},
                ValueError,
                "num_heads must be greater than 0, received 0",
            ),
            (
                "wrong_num_heads_type",
                {"num_heads": "one"},
                TypeError,
                "num_heads must be int for IndependentAttentionConfig, got str",
            ),
            (
                "missing_embedding_dim",
                {"embedding_dim": None},
                ValueError,
                (
                    "embedding_dim is required for IndependentAttentionConfig, "
                    "received None"
                ),
            ),
            (
                "zero_batch_size",
                {"batch_size": 0},
                ValueError,
                "batch_size must be greater than 0, received 0",
            ),
        )

        for name, changes, exception_type, message in cases:
            with self.subTest(name=name), torch.random.fork_rng():
                torch.manual_seed(723)
                expected_next_values = torch.randn(4)

                torch.manual_seed(723)
                with self.assertRaisesRegex(exception_type, message):
                    replace(config, **changes).build()
                actual_next_values = torch.randn(4)

                torch.testing.assert_close(actual_next_values, expected_next_values)

    def test_batch_first_flag_rejects_non_boolean_values(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
        )

        for invalid in (0, 1, "batch-first"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    TypeError,
                    "batch_first_flag must be True, False, or None",
                ):
                    replace(config, batch_first_flag=invalid).build()

    def test_runtime_sequence_maxima_must_be_positive(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
        )
        cases = (
            ("target_sequence_length", 0),
            ("source_sequence_length", 0),
            ("target_sequence_length", -1),
            ("source_sequence_length", -1),
        )

        for field_name, invalid in cases:
            with self.subTest(field_name=field_name, invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{field_name} must be greater than 0",
                ):
                    replace(config, **{field_name: invalid}).build()

    def test_projection_dimensions_must_be_zero_or_positive(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=1,
        )

        for field_name in ("query_key_projection_dim", "value_projection_dim"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{field_name} must be 0 or greater",
                ):
                    replace(config, **{field_name: -1}).build()

    def test_dropout_probability_must_be_in_closed_unit_interval(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
        )

        for invalid in (-0.1, 1.1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "dropout_probability must be between 0 and 1",
                ):
                    replace(config, dropout_probability=invalid).build()

    def test_target_dtype_must_be_a_floating_torch_dtype(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
        )

        with self.assertRaisesRegex(TypeError, "target_dtype must be a torch dtype"):
            replace(config, target_dtype="float32").build()
        with self.assertRaisesRegex(
            ValueError,
            "target_dtype must be a floating torch dtype",
        ):
            replace(config, target_dtype=torch.int64).build()

    def test_nested_configuration_objects_are_type_checked(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=1,
            target_sequence_length=1,
            source_sequence_length=1,
        )
        cases = (
            ("projection_model_config", "projection model configuration"),
            (
                "relative_positional_embedding_config",
                "relative positional embedding configuration",
            ),
        )

        for field_name, description in cases:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    TypeError,
                    f"{description} must be",
                ):
                    replace(config, **{field_name: object()}).build()

    def test_relative_configuration_matches_attention_heads_and_qk_width(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=1,
            source_sequence_length=1,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        relative = config.relative_positional_embedding_config
        cases = (
            (replace(relative, num_heads=1), "num_heads must match attention"),
            (
                replace(relative, embedding_dim=6),
                "embedding_dim must match effective query/key projection",
            ),
        )

        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    replace(
                        config,
                        relative_positional_embedding_config=invalid,
                    ).build()

    def test_booleans_are_not_accepted_as_integer_dimensions(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=1,
            source_sequence_length=1,
        )

        for field_name in ("batch_size", "query_key_projection_dim"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    TypeError,
                    f"{field_name} must be int",
                ):
                    replace(config, **{field_name: True}).build()


class TestAttentionRuntimeValidation(unittest.TestCase):
    def model(self):
        return build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=2,
            num_heads=2,
            embedding_dim=4,
            target_sequence_length=3,
            source_sequence_length=3,
        ).build()

    def test_zero_length_query_is_rejected_before_projection(self):
        model = self.model()
        query = torch.empty(0, 2, 4)
        source = torch.randn(1, 2, 4)

        with self.assertRaisesRegex(
            RuntimeError,
            "query sequence length must be greater than 0",
        ):
            model(query, source, source)

    def test_zero_length_key_value_source_is_rejected_before_projection(self):
        model = self.model()
        query = torch.randn(1, 2, 4)
        source = torch.empty(0, 2, 4)

        with self.assertRaisesRegex(
            RuntimeError,
            "key and value sequence lengths must be greater than 0",
        ):
            model(query, source, source)

    def test_mismatched_batches_are_rejected_before_projection(self):
        model = self.model()
        query = torch.randn(1, 2, 4)
        source = torch.randn(1, 1, 4)

        with self.assertRaisesRegex(
            RuntimeError,
            "query, key, and value batch sizes must match",
        ):
            model(query, source, source)

    def test_zero_runtime_batch_is_rejected_uniformly_before_projection(self):
        for config_class in (
            SelfAttentionConfig,
            IndependentAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        ):
            with self.subTest(config_class=config_class):
                options = {}
                if config_class is MixtureOfAttentionHeadsConfig:
                    options = {
                        "experts_top_k": 1,
                        "experts_num_experts": 2,
                    }
                config = build_attention_config(
                    config_class=config_class,
                    batch_size=1,
                    num_heads=2,
                    embedding_dim=4,
                    target_sequence_length=3,
                    source_sequence_length=3,
                    **options,
                )
                model = replace(config, batch_first_flag=False).build()
                query = torch.empty(2, 0, 4)
                if config_class is SelfAttentionConfig:
                    key = value = query
                else:
                    key = value = torch.empty(3, 0, 4)

                with self.assertRaises(RuntimeError) as raised:
                    model(query, key, value)
                self.assertEqual(
                    str(raised.exception),
                    "runtime batch size must be greater than 0.",
                )

                self.assertIsNone(model.projector.auxiliary_loss)
                if config_class is MixtureOfAttentionHeadsConfig:
                    self.assertIsNone(model.projector.probabilities)
                    self.assertIsNone(model.projector.indices)
                    self.assertIsNone(model.projector.skip_mask)

    def test_public_forward_rejects_malformed_masks_before_layout_conversion(
        self,
    ) -> None:
        query = torch.randn(1, 2, 4)
        source = torch.randn(3, 2, 4)
        cases = (
            (
                "key_padding_mask",
                {
                    "k_padding_mask": torch.zeros(
                        2,
                        3,
                        1,
                        dtype=torch.bool,
                    )
                },
                "For a batched (3-D) query, expected key_padding_mask to be None "
                "or 2-D but found a 3-D tensor instead.",
            ),
            (
                "attention_mask",
                {
                    "attention_mask": torch.zeros(
                        1,
                        2,
                        1,
                        3,
                        dtype=torch.bool,
                    )
                },
                "Expected attention_mask to be None, 2-D, or 3-D but found a 4-D "
                "tensor instead.",
            ),
        )

        for name, mask_arguments, message in cases:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError) as raised:
                    self.model()(query, source, source, **mask_arguments)
                self.assertEqual(str(raised.exception), message)

    def test_incorrect_embedding_widths_are_rejected_before_projection(self):
        valid = torch.randn(1, 2, 4)
        invalid = torch.randn(1, 2, 5)
        cases = (
            ("query", invalid, valid, valid),
            ("key", valid, invalid, valid),
            ("value", valid, valid, invalid),
        )

        for name, query, key, value in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"{name} embedding width must be 4",
                ):
                    self.model()(query, key, value)

    def test_mismatched_input_dtypes_are_rejected_before_projection(self):
        query = torch.randn(1, 2, 4, dtype=torch.float32)
        source = torch.randn(1, 2, 4, dtype=torch.float64)

        with self.assertRaisesRegex(
            RuntimeError,
            "query, key, and value dtypes must match",
        ):
            self.model()(query, source, source)

    def test_non_floating_inputs_are_rejected_before_projection(self):
        values = torch.ones(1, 2, 4, dtype=torch.int64)

        with self.assertRaisesRegex(
            RuntimeError,
            "query, key, and value must be floating point tensors",
        ):
            self.model()(values, values, values)


if __name__ == "__main__":
    unittest.main()
