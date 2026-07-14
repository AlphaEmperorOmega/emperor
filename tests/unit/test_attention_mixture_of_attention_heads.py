import unittest

import torch
from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.attention.core.variants.mixture_of_attention_heads.bias import (
    MixtureOfAttentionHeadsKeyValueBias,
)
from emperor.attention.core.variants.mixture_of_attention_heads.zero_attention import (
    MixtureOfAttentionHeadsZeroAttention,
)
from emperor.embedding.relative.core.config import DynamicPositionalBiasConfig

from support.attention import build_attention_config


class TestMixtureOfAttentionHeadsExpertKeyValue(unittest.TestCase):
    def preset(self, **overrides):
        values = {
            "config_class": MixtureOfAttentionHeadsConfig,
            "batch_size": 2,
            "num_heads": 2,
            "embedding_dim": 8,
            "query_key_projection_dim": 8,
            "value_projection_dim": 8,
            "target_sequence_length": 4,
            "source_sequence_length": 4,
            "use_kv_expert_models_flag": True,
            "experts_top_k": 2,
            "experts_num_experts": 4,
            "experts_compute_expert_mixture_flag": False,
            "experts_stack_num_layers": 1,
        }
        values.update(overrides)
        return build_attention_config(**values)

    def assert_has_nonzero_parameter_gradient(self, model, label):
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None and torch.any(gradient.abs() > 0)
                for gradient in gradients
            ),
            f"Expected a non-zero gradient through {label}.",
        )

    def input_tensor(self, cfg, *, requires_grad=False):
        return torch.randn(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
            requires_grad=requires_grad,
        )

    def test_self_attention_forward(self):
        torch.manual_seed(0)
        cfg = self.preset()
        model = cfg.build()
        inputs = torch.randn(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
        )

        output, attention_weights, auxiliary_loss = model(inputs, inputs, inputs)

        self.assertEqual(
            output.shape,
            (
                cfg.target_sequence_length,
                cfg.batch_size,
                cfg.embedding_dim,
            ),
        )
        self.assertIsNone(attention_weights)
        self.assertIsInstance(auxiliary_loss, torch.Tensor)

    def test_build_selects_mixture_specific_handlers(self):
        model = self.preset().build()

        self.assertIsInstance(model.bias, MixtureOfAttentionHeadsKeyValueBias)
        self.assertIsInstance(
            model.zero_attention,
            MixtureOfAttentionHeadsZeroAttention,
        )

    def test_key_value_bias_supports_shared_and_expert_projection_ranks(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                cfg = self.preset(
                    value_projection_dim=12,
                    add_key_value_bias_flag=True,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = MixtureOfAttentionHeadsKeyValueBias(cfg)
                if use_kv_expert_models_flag:
                    key = torch.randn(4, 2, 2, 8)
                    value = torch.randn(4, 2, 2, 12)
                    expected_key_bias = model.key_bias_vector.unsqueeze(2).expand(
                        1, 2, 2, 8
                    )
                    expected_value_bias = model.value_bias_vector.unsqueeze(2).expand(
                        1, 2, 2, 12
                    )
                else:
                    key = torch.randn(4, 2, 8)
                    value = torch.randn(4, 2, 12)
                    expected_key_bias = model.key_bias_vector.expand(1, 2, 8)
                    expected_value_bias = model.value_bias_vector.expand(1, 2, 12)
                key_padding_mask = torch.zeros(2, 4, dtype=torch.bool)
                attention_mask = torch.zeros(8, 4, 4, dtype=torch.bool)

                output_key, output_value, output_padding_mask, output_attention_mask = (
                    model.add_kv_learnable_bias_vectors(
                        key,
                        value,
                        key_padding_mask,
                        attention_mask,
                    )
                )

                self.assertEqual(output_key.shape, (5, *key.shape[1:]))
                self.assertEqual(output_value.shape, (5, *value.shape[1:]))
                self.assertEqual(output_padding_mask.shape, (2, 5))
                self.assertEqual(output_attention_mask.shape, (8, 4, 5))
                torch.testing.assert_close(output_key[-1], expected_key_bias[0])
                torch.testing.assert_close(output_value[-1], expected_value_bias[0])

    def test_zero_attention_supports_shared_and_expert_projection_branches(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                cfg = self.preset(
                    value_projection_dim=12,
                    zero_attention_flag=True,
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = MixtureOfAttentionHeadsZeroAttention(cfg)
                branch_count = cfg.batch_size * cfg.num_heads
                if use_kv_expert_models_flag:
                    branch_count *= cfg.experts_config.top_k
                attention_branch_count = (
                    cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
                )
                key = torch.randn(branch_count, 4, 4)
                value = torch.randn(branch_count, 4, 6)
                key_padding_mask = torch.zeros(2, 4, dtype=torch.bool)
                attention_mask = torch.zeros(
                    attention_branch_count,
                    4,
                    4,
                    dtype=torch.bool,
                )

                output_key, output_value, output_padding_mask, output_attention_mask = (
                    model.add_zero_attention(
                        key,
                        value,
                        key_padding_mask,
                        attention_mask,
                    )
                )

                self.assertEqual(output_key.shape, (branch_count, 5, 4))
                self.assertEqual(output_value.shape, (branch_count, 5, 6))
                self.assertEqual(output_padding_mask.shape, (2, 5))
                self.assertEqual(
                    output_attention_mask.shape,
                    (attention_branch_count, 4, 5),
                )
                torch.testing.assert_close(
                    output_key[:, -1],
                    torch.zeros_like(key[:, 0]),
                )
                torch.testing.assert_close(
                    output_value[:, -1],
                    torch.zeros_like(value[:, 0]),
                )

    def test_shared_and_expert_key_value_top_k_matrix_forward_backward(self):
        for use_kv_expert_models_flag in (False, True):
            for top_k in (1, 2):
                with self.subTest(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                    top_k=top_k,
                ):
                    torch.manual_seed(100 + top_k + use_kv_expert_models_flag)
                    cfg = self.preset(
                        use_kv_expert_models_flag=use_kv_expert_models_flag,
                        experts_top_k=top_k,
                        experts_num_experts=4,
                    )
                    model = cfg.build()
                    inputs = self.input_tensor(cfg, requires_grad=True)

                    output, attention_weights, auxiliary_loss = model(
                        inputs,
                        inputs,
                        inputs,
                    )

                    self.assertEqual(output.shape, inputs.shape)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertIsNone(attention_weights)
                    self.assertIsInstance(auxiliary_loss, torch.Tensor)
                    self.assertTrue(torch.isfinite(auxiliary_loss).all())
                    self.assertIsNone(model.projector.auxiliary_loss)

                    (output.square().mean() + auxiliary_loss).backward()

                    self.assertIsNotNone(inputs.grad)
                    self.assertTrue(torch.any(inputs.grad.abs() > 0))
                    self.assert_has_nonzero_parameter_gradient(
                        model.projector.sampler.router,
                        "router",
                    )
                    self.assert_has_nonzero_parameter_gradient(
                        model.projector.query_model,
                        "query_model",
                    )
                    self.assert_has_nonzero_parameter_gradient(
                        model.projector.output_model,
                        "output_model",
                    )
                    for label in ("key_model", "value_model"):
                        self.assert_has_nonzero_parameter_gradient(
                            getattr(model.projector, label),
                            label,
                        )

    def test_relative_positional_embedding_with_shared_and_expert_key_value(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                torch.manual_seed(200 + use_kv_expert_models_flag)
                cfg = self.preset(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                    relative_positional_embedding_config_cls=(
                        DynamicPositionalBiasConfig
                    ),
                )
                model = cfg.build()
                inputs = self.input_tensor(cfg, requires_grad=True)

                output, _, auxiliary_loss = model(inputs, inputs, inputs)

                self.assertEqual(output.shape, inputs.shape)
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNotNone(model.processor.relative_positional_embedding)
                (output.square().mean() + auxiliary_loss).backward()
                self.assert_has_nonzero_parameter_gradient(
                    model.processor.relative_positional_embedding,
                    "relative_positional_embedding",
                )

    def test_mask_formats_normalize_in_batch_expert_head_order(self):
        cfg = self.preset()
        model = cfg.build()
        branch_count = cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
        key = torch.randn(
            branch_count,
            cfg.source_sequence_length,
            cfg.query_key_projection_dim // cfg.num_heads,
        )
        sequence_shape = (
            cfg.target_sequence_length,
            cfg.source_sequence_length,
        )

        mask_2d = torch.arange(
            cfg.target_sequence_length * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(sequence_shape)
        normalized_2d = model.masks.merge_padding_and_attention_mask(
            key,
            attention_mask=mask_2d,
        )
        expected_2d = mask_2d.view(1, *sequence_shape).expand(
            branch_count,
            -1,
            -1,
        )
        torch.testing.assert_close(normalized_2d, expected_2d)

        standard_mask = torch.arange(
            cfg.batch_size
            * cfg.num_heads
            * cfg.target_sequence_length
            * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(cfg.batch_size, cfg.num_heads, *sequence_shape)
        normalized_standard = model.masks.merge_padding_and_attention_mask(
            key,
            attention_mask=standard_mask.reshape(-1, *sequence_shape),
        )
        expected_standard = (
            standard_mask.unsqueeze(1)
            .expand(
                -1,
                cfg.experts_config.top_k,
                -1,
                -1,
                -1,
            )
            .reshape(branch_count, *sequence_shape)
        )
        torch.testing.assert_close(normalized_standard, expected_standard)

        expanded_mask = torch.arange(
            branch_count * cfg.target_sequence_length * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(
            cfg.batch_size,
            cfg.experts_config.top_k,
            cfg.num_heads,
            *sequence_shape,
        )
        normalized_expanded = model.masks.merge_padding_and_attention_mask(
            key,
            attention_mask=expanded_mask.reshape(-1, *sequence_shape),
        )
        torch.testing.assert_close(
            normalized_expanded,
            expanded_mask.reshape(branch_count, *sequence_shape),
        )

    def test_standard_and_equivalent_expanded_masks_produce_same_output(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                torch.manual_seed(300 + use_kv_expert_models_flag)
                cfg = self.preset(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = cfg.build().eval()
                inputs = self.input_tensor(cfg)
                sequence_shape = (
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                )
                standard_mask = torch.zeros(
                    cfg.batch_size,
                    cfg.num_heads,
                    *sequence_shape,
                    dtype=cfg.target_dtype,
                )
                for batch_index in range(cfg.batch_size):
                    for head_index in range(cfg.num_heads):
                        source_index = (
                            batch_index * cfg.num_heads + head_index
                        ) % cfg.source_sequence_length
                        standard_mask[batch_index, head_index, :, source_index] = float(
                            "-inf"
                        )
                expanded_mask = standard_mask.unsqueeze(1).expand(
                    -1,
                    cfg.experts_config.top_k,
                    -1,
                    -1,
                    -1,
                )

                standard_output, _, _ = model(
                    inputs,
                    inputs,
                    inputs,
                    attention_mask=standard_mask.reshape(-1, *sequence_shape),
                )
                expanded_output, _, _ = model(
                    inputs,
                    inputs,
                    inputs,
                    attention_mask=expanded_mask.reshape(-1, *sequence_shape),
                )

                torch.testing.assert_close(standard_output, expanded_output)

    def test_padding_mask_expands_across_experts_and_heads(self):
        cfg = self.preset()
        model = cfg.build()
        branch_count = cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
        key = torch.randn(
            branch_count,
            cfg.source_sequence_length,
            cfg.query_key_projection_dim // cfg.num_heads,
        )
        padding_mask = torch.arange(
            cfg.batch_size * cfg.source_sequence_length,
            dtype=cfg.target_dtype,
        ).view(cfg.batch_size, cfg.source_sequence_length)

        normalized = model.masks.merge_padding_and_attention_mask(
            key,
            key_padding_mask=padding_mask,
        )

        expected = (
            padding_mask.view(
                cfg.batch_size,
                1,
                1,
                1,
                cfg.source_sequence_length,
            )
            .expand(
                -1,
                cfg.experts_config.top_k,
                cfg.num_heads,
                -1,
                -1,
            )
            .reshape(branch_count, 1, cfg.source_sequence_length)
        )
        torch.testing.assert_close(normalized, expected)

    def test_invalid_mask_dimensions_raise_clear_errors_and_allow_retry(self):
        cfg = self.preset()
        model = cfg.build()
        inputs = self.input_tensor(cfg)
        invalid_cases = [
            (
                "leading dimension",
                None,
                torch.zeros(
                    3,
                    cfg.target_sequence_length,
                    cfg.source_sequence_length,
                ),
            ),
            (
                "target/source dimensions",
                None,
                torch.zeros(
                    cfg.target_sequence_length + 1,
                    cfg.source_sequence_length,
                ),
            ),
            (
                "target/source dimensions",
                None,
                torch.zeros(
                    cfg.target_sequence_length,
                    cfg.source_sequence_length + 1,
                ),
            ),
            (
                "key_padding_mask must have shape",
                torch.zeros(
                    cfg.batch_size,
                    cfg.source_sequence_length + 1,
                    dtype=torch.bool,
                ),
                None,
            ),
        ]

        for error_pattern, padding_mask, attention_mask in invalid_cases:
            with self.subTest(error_pattern=error_pattern):
                with self.assertRaisesRegex(RuntimeError, error_pattern):
                    model(
                        inputs,
                        inputs,
                        inputs,
                        padding_mask,
                        attention_mask,
                    )
                output, _, auxiliary_loss = model(inputs, inputs, inputs)
                self.assertEqual(output.shape, inputs.shape)
                self.assertIsInstance(auxiliary_loss, torch.Tensor)
                self.assertIsNone(model.projector.auxiliary_loss)

    def test_shared_static_key_value_remains_supported(self):
        cfg = self.preset(
            use_kv_expert_models_flag=False,
            target_sequence_length=3,
            source_sequence_length=4,
        )
        model = cfg.build()
        query = self.input_tensor(cfg)
        key = value = torch.randn(
            cfg.source_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
        )
        static_source_sequence_length = 5
        head_dim = cfg.embedding_dim // cfg.num_heads
        static_key = torch.randn(
            cfg.batch_size * cfg.num_heads,
            static_source_sequence_length,
            head_dim,
        )
        static_value = torch.randn_like(static_key)

        output, attention_weights, auxiliary_loss = model(
            query,
            key,
            value,
            static_k=static_key,
            static_v=static_value,
        )

        self.assertEqual(output.shape, query.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNone(attention_weights)
        self.assertIsInstance(auxiliary_loss, torch.Tensor)

    def test_expert_static_key_value_is_rejected_before_routing(self):
        cfg = self.preset()
        model = cfg.build()
        inputs = self.input_tensor(cfg)
        head_dim = cfg.embedding_dim // cfg.num_heads
        static_key = torch.randn(
            cfg.batch_size * cfg.num_heads,
            cfg.source_sequence_length,
            head_dim,
        )

        with self.assertRaisesRegex(
            ValueError,
            "static key/value projections are not supported",
        ):
            model(
                inputs,
                inputs,
                inputs,
                static_k=static_key,
                static_v=static_key,
            )

        self.assertIsNone(model.projector.auxiliary_loss)
        output, _, auxiliary_loss = model(inputs, inputs, inputs)
        self.assertEqual(output.shape, inputs.shape)
        self.assertIsInstance(auxiliary_loss, torch.Tensor)
        self.assertIsNone(model.projector.auxiliary_loss)

    def test_routing_is_sampled_once_reused_and_loss_is_cleared(self):
        for use_kv_expert_models_flag in (False, True):
            with self.subTest(
                use_kv_expert_models_flag=use_kv_expert_models_flag,
            ):
                torch.manual_seed(400 + use_kv_expert_models_flag)
                cfg = self.preset(
                    use_kv_expert_models_flag=use_kv_expert_models_flag,
                )
                model = cfg.build()
                inputs = self.input_tensor(cfg)
                sampler_calls = []
                routing_inputs = {}

                original_sample = (
                    model.projector.sampler.sample_probabilities_and_indices
                )

                def sample_with_loss(
                    input_matrix,
                    skip_mask=None,
                    *,
                    _original_sample=original_sample,
                    _sampler_calls=sampler_calls,
                ):
                    result = _original_sample(input_matrix, skip_mask)
                    _sampler_calls.append(input_matrix)
                    probabilities, indices, updated_skip_mask, loss = result
                    return (
                        probabilities,
                        indices,
                        updated_skip_mask,
                        loss + input_matrix.new_tensor(1.0),
                    )

                model.projector.sampler.sample_probabilities_and_indices = (
                    sample_with_loss
                )

                expert_models = {
                    "query": model.projector.query_model,
                    "output": model.projector.output_model,
                }
                if use_kv_expert_models_flag:
                    expert_models.update(
                        key=model.projector.key_model,
                        value=model.projector.value_model,
                    )

                for label, expert_model in expert_models.items():
                    original_forward = expert_model.forward
                    routing_inputs[label] = []

                    def forward_with_loss(
                        input_batch,
                        probabilities=None,
                        indices=None,
                        *,
                        _label=label,
                        _original_forward=original_forward,
                        _routing_inputs=routing_inputs,
                    ):
                        _routing_inputs[_label].append((probabilities, indices))
                        output, loss = _original_forward(
                            input_batch,
                            probabilities,
                            indices,
                        )
                        return output, loss + input_batch.new_tensor(1.0)

                    expert_model.forward = forward_with_loss

                expected_loss = 5.0 if use_kv_expert_models_flag else 3.0
                for forward_index in range(2):
                    output, _, auxiliary_loss = model(inputs, inputs, inputs)

                    self.assertEqual(output.shape, inputs.shape)
                    self.assertEqual(auxiliary_loss.item(), expected_loss)
                    self.assertIsNone(model.projector.auxiliary_loss)
                    routed_probabilities = [
                        records[forward_index][0] for records in routing_inputs.values()
                    ]
                    routed_indices = [
                        records[forward_index][1] for records in routing_inputs.values()
                    ]
                    self.assertTrue(
                        all(
                            probabilities is routed_probabilities[0]
                            for probabilities in routed_probabilities[1:]
                        )
                    )
                    self.assertTrue(
                        all(
                            indices is routed_indices[0]
                            for indices in routed_indices[1:]
                        )
                    )

                self.assertEqual(len(sampler_calls), 2)
                for records in routing_inputs.values():
                    self.assertEqual(len(records), 2)

    def test_returning_attention_weights_is_unsupported(self):
        cfg = self.preset(return_attention_weights_flag=True)
        model = cfg.build()
        inputs = self.input_tensor(cfg)

        with self.assertRaisesRegex(RuntimeError, "attention_weights"):
            model(inputs, inputs, inputs)

    def test_expert_projections_are_reshaped_for_processor(self):
        torch.manual_seed(0)
        cfg = self.preset()
        model = cfg.build()
        inputs = torch.randn(
            cfg.target_sequence_length,
            cfg.batch_size,
            cfg.embedding_dim,
        )
        query, key, value = model.projector.compute_qkv_projections(
            inputs,
            inputs,
            inputs,
        )

        query, key, value = model.reshaper.reshape_qkv_for_attention(
            query,
            key,
            value,
        )
        query, key, value = model.reshaper.reshape_before_attention(
            query,
            key,
            value,
        )

        head_dim = cfg.query_key_projection_dim // cfg.num_heads
        value_head_dim = cfg.value_projection_dim // cfg.num_heads
        self.assertEqual(
            query.shape,
            (
                cfg.batch_size,
                cfg.experts_config.top_k,
                cfg.num_heads,
                cfg.target_sequence_length,
                head_dim,
            ),
        )
        self.assertEqual(
            key.shape,
            (
                cfg.batch_size,
                cfg.experts_config.top_k,
                cfg.num_heads,
                cfg.source_sequence_length,
                head_dim,
            ),
        )
        self.assertEqual(
            value.shape,
            (
                cfg.batch_size,
                cfg.experts_config.top_k,
                cfg.num_heads,
                cfg.source_sequence_length,
                value_head_dim,
            ),
        )

    def test_expert_key_value_requires_equal_configured_sequence_lengths(self):
        cfg = self.preset(
            target_sequence_length=3,
            source_sequence_length=4,
        )

        with self.assertRaisesRegex(
            ValueError,
            "target_sequence_length and source_sequence_length must be equal",
        ):
            cfg.build()

    def test_expert_sampler_and_router_dimensions_must_match(self):
        cfg = self.preset()
        cfg.experts_config.sampler_config.top_k = 1
        with self.assertRaisesRegex(ValueError, "top_k must match"):
            cfg.build()

        cfg = self.preset()
        cfg.experts_config.sampler_config.num_experts = 3
        with self.assertRaisesRegex(ValueError, "num_experts must match"):
            cfg.build()

        cfg = self.preset()
        cfg.experts_config.sampler_config.router_config.num_experts = 3
        with self.assertRaisesRegex(ValueError, "router_config.num_experts"):
            cfg.build()

    def test_dense_expert_routing_is_rejected_clearly(self):
        cfg = self.preset(
            experts_top_k=4,
            experts_num_experts=4,
        )

        with self.assertRaisesRegex(ValueError, "dense routing is not supported"):
            cfg.build()

    def test_expert_key_value_requires_shared_query_key_value_tensor(self):
        cfg = self.preset()
        model = cfg.build()
        query = torch.randn(4, 2, 8)
        key = query.clone()
        value = query.clone()

        with self.assertRaisesRegex(
            ValueError,
            "query, key, and value must be the same tensor",
        ):
            model(query, key, value)

        self.assertIsNone(model.projector.auxiliary_loss)

    def test_flag_matrix_propagates_auxiliary_loss_and_gradients(self):
        cases = [
            ("plain", False, False, False),
            ("masks", False, False, True),
            ("bias", True, False, False),
            ("zero_attention", False, True, False),
            ("combined", True, True, True),
        ]
        for case_index, (
            case_name,
            add_key_value_bias_flag,
            zero_attention_flag,
            use_masks,
        ) in enumerate(cases):
            with self.subTest(case_name=case_name):
                torch.manual_seed(case_index)
                cfg = self.preset(
                    add_key_value_bias_flag=add_key_value_bias_flag,
                    zero_attention_flag=zero_attention_flag,
                )
                model = cfg.build()
                inputs = torch.randn(
                    cfg.target_sequence_length,
                    cfg.batch_size,
                    cfg.embedding_dim,
                    requires_grad=True,
                )
                key_padding_mask = None
                attention_mask = None
                if use_masks:
                    key_padding_mask = torch.tensor(
                        [
                            [False, True, False, False],
                            [False, False, True, False],
                        ]
                    )
                    total_batch_size = (
                        cfg.batch_size * cfg.experts_config.top_k * cfg.num_heads
                    )
                    attention_mask = torch.zeros(
                        total_batch_size,
                        cfg.target_sequence_length,
                        cfg.source_sequence_length,
                        dtype=torch.bool,
                    )
                    attention_mask[:, 0, -1] = True

                output, attention_weights, auxiliary_loss = model(
                    inputs,
                    inputs,
                    inputs,
                    key_padding_mask,
                    attention_mask,
                )

                self.assertEqual(
                    output.shape,
                    (
                        cfg.target_sequence_length,
                        cfg.batch_size,
                        cfg.embedding_dim,
                    ),
                )
                self.assertIsNone(attention_weights)
                self.assertIsInstance(auxiliary_loss, torch.Tensor)

                (output.square().sum() + auxiliary_loss).backward()

                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.any(inputs.grad.abs() > 0))
                for label in (
                    "query_model",
                    "key_model",
                    "value_model",
                    "output_model",
                ):
                    self.assert_has_nonzero_parameter_gradient(
                        getattr(model.projector, label),
                        label,
                    )
                if add_key_value_bias_flag:
                    self.assertIsNotNone(model.bias.key_bias_vector.grad)
                    self.assertIsNotNone(model.bias.value_bias_vector.grad)
                    self.assertTrue(
                        torch.any(model.bias.key_bias_vector.grad.abs() > 0)
                    )
                    self.assertTrue(
                        torch.any(model.bias.value_bias_vector.grad.abs() > 0)
                    )


if __name__ == "__main__":
    unittest.main()
