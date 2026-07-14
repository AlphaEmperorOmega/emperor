import unittest
from dataclasses import replace

import torch
from emperor.attention import (
    IndependentAttentionConfig,
    MixtureOfAttentionHeadsConfig,
    SelfAttentionConfig,
)
from emperor.embedding.relative import DynamicPositionalBiasConfig

from support.attention import build_attention_config


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


class TestAttentionConfigurationValidation(unittest.TestCase):
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
