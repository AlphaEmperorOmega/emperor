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
from support.attention import build_attention_config

CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_BFLOAT16_AVAILABLE = CUDA_AVAILABLE and torch.cuda.is_bf16_supported()


class TestAttentionNumericalContracts(unittest.TestCase):
    def config(self, config_class, dtype):
        options = {}
        if config_class is MixtureOfAttentionHeadsConfig:
            options = {"experts_top_k": 1, "experts_num_experts": 2}
        config = build_attention_config(
            config_class=config_class,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.0,
            **options,
        )
        return replace(config, target_dtype=dtype)

    def assert_forward_and_gradients(self, config_class, dtype, device):
        torch.manual_seed(61)
        model = (
            self.config(config_class, dtype)
            .build()
            .to(
                device=device,
                dtype=dtype,
            )
        )
        model.eval()
        query = torch.randn(
            2,
            1,
            2,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        if config_class is SelfAttentionConfig:
            key = value = query
            differentiable_inputs = (query,)
        else:
            key = torch.randn(
                2,
                1,
                2,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            value = torch.randn(
                2,
                1,
                2,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            differentiable_inputs = (query, key, value)

        output, _, auxiliary_loss = model(query, key, value)
        loss = output.square().sum()
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss
        loss.backward()

        self.assertEqual(output.device.type, device.type)
        self.assertEqual(output.dtype, dtype)
        self.assertTrue(torch.isfinite(output).all())
        for tensor in differentiable_inputs:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertTrue(torch.any(tensor.grad != 0))

    def test_cpu_float32_and_float64_forward_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for config_class in (
                SelfAttentionConfig,
                IndependentAttentionConfig,
                MixtureOfAttentionHeadsConfig,
            ):
                with self.subTest(dtype=dtype, config_class=config_class):
                    self.assert_forward_and_gradients(
                        config_class,
                        dtype,
                        torch.device("cpu"),
                    )

    def test_runtime_dtype_conversion_keeps_generated_and_boolean_masks_usable(
        self,
    ) -> None:
        config = build_attention_config(
            config_class=SelfAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            causal_attention_mask_flag=True,
            return_attention_weights_flag=True,
            self_attention_projection_strategy=(
                SelfAttentionProjectionStrategy.SEPARATE
            ),
        )
        for dtype in (torch.float64, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = config.build().to(dtype=dtype).eval()
                with torch.no_grad():
                    for projection in (
                        model.projector.query_model,
                        model.projector.key_model,
                        model.projector.value_model,
                        model.projector.output_model,
                    ):
                        layer = projection.layers[0].model
                        layer.weight_params.copy_(torch.eye(2, dtype=dtype))
                        layer.bias_params.zero_()
                inputs = torch.eye(2, dtype=dtype).unsqueeze(1).requires_grad_()
                key_padding_mask = torch.tensor([[False, True]])
                expected_output = torch.tensor(
                    [[[1.0, 0.0]], [[1.0, 0.0]]],
                    dtype=dtype,
                )
                expected_weights = torch.tensor(
                    [[[[1.0, 0.0], [1.0, 0.0]]]],
                    dtype=dtype,
                )

                output, weights, auxiliary_loss = model(
                    inputs,
                    inputs,
                    inputs,
                    k_padding_mask=key_padding_mask,
                )

                torch.testing.assert_close(output, expected_output)
                torch.testing.assert_close(weights, expected_weights)
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(weights.dtype, dtype)
                self.assertIsNone(auxiliary_loss)

                output.square().sum().backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertTrue(torch.any(inputs.grad.abs() > 0))

    def test_target_dtype_constructs_each_variant_in_the_requested_dtype(self) -> None:
        for config_class in (
            SelfAttentionConfig,
            IndependentAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        ):
            with self.subTest(config_class=config_class), torch.random.fork_rng():
                torch.manual_seed(619)
                model = self.config(config_class, torch.float64).build().eval()
                floating_state = tuple(
                    value
                    for value in model.state_dict().values()
                    if torch.is_floating_point(value)
                )
                self.assertTrue(floating_state)
                self.assertEqual(
                    {value.dtype for value in floating_state},
                    {torch.float64},
                )

                query = torch.tensor(
                    [[[1.0, -2.0]], [[0.5, 3.0]]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                if config_class is SelfAttentionConfig:
                    key = value = query
                    differentiable_inputs = (query,)
                else:
                    key = torch.tensor(
                        [[[2.0, 1.0]], [[-1.0, 0.5]]],
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    value = torch.tensor(
                        [[[0.25, -3.0]], [[4.0, 2.0]]],
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    differentiable_inputs = (query, key, value)

                output, _, auxiliary_loss = model(query, key, value)
                objective = output.square().sum()
                if auxiliary_loss is not None:
                    objective = objective + auxiliary_loss
                objective.backward()

                self.assertEqual(output.dtype, torch.float64)
                self.assertTrue(torch.isfinite(output).all())
                self.assertGreater(torch.count_nonzero(output).item(), 0)
                for tensor in differentiable_inputs:
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertTrue(torch.any(tensor.grad.abs() > 0))

    def test_bfloat16_additive_mask_matches_boolean_mask_for_every_variant(
        self,
    ) -> None:
        boolean_mask = torch.tensor(
            [[False, True], [False, False]],
        )
        additive_mask = torch.zeros(2, 2, dtype=torch.float32).masked_fill_(
            boolean_mask,
            -torch.inf,
        )
        for config_class in (
            SelfAttentionConfig,
            IndependentAttentionConfig,
            MixtureOfAttentionHeadsConfig,
        ):
            with self.subTest(config_class=config_class), torch.random.fork_rng():
                torch.manual_seed(701)
                model = self.config(config_class, torch.float32).build()
                model = model.to(dtype=torch.bfloat16).eval()
                query = torch.tensor(
                    [[[1.0, -2.0]], [[0.5, 3.0]]],
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                if config_class is SelfAttentionConfig:
                    key = value = query
                    differentiable_inputs = (query,)
                else:
                    key = torch.tensor(
                        [[[2.0, 1.0]], [[-1.0, 0.5]]],
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    value = torch.tensor(
                        [[[0.25, -3.0]], [[4.0, 2.0]]],
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    )
                    differentiable_inputs = (query, key, value)

                boolean_output, _, boolean_loss = model(
                    query,
                    key,
                    value,
                    attention_mask=boolean_mask,
                )
                additive_output, _, additive_loss = model(
                    query,
                    key,
                    value,
                    attention_mask=additive_mask,
                )

                torch.testing.assert_close(additive_output, boolean_output)
                self.assertEqual(additive_output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(additive_output).all())
                self.assertGreater(torch.count_nonzero(additive_output).item(), 0)
                if boolean_loss is None:
                    self.assertIsNone(additive_loss)
                else:
                    torch.testing.assert_close(additive_loss, boolean_loss)

                objective = additive_output.square().sum()
                if additive_loss is not None:
                    objective = objective + additive_loss
                objective.backward()
                for tensor in differentiable_inputs:
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertTrue(torch.any(tensor.grad.abs() > 0))

    def test_zero_attention_preserves_bfloat16_for_standard_and_mixture_kv(
        self,
    ) -> None:
        cases = (
            (SelfAttentionConfig, None),
            (IndependentAttentionConfig, None),
            (MixtureOfAttentionHeadsConfig, False),
            (MixtureOfAttentionHeadsConfig, True),
        )
        for config_class, use_expert_kv in cases:
            with self.subTest(
                config_class=config_class,
                use_expert_kv=use_expert_kv,
            ):
                options = {}
                if config_class is MixtureOfAttentionHeadsConfig:
                    options = {
                        "experts_top_k": 1,
                        "experts_num_experts": 2,
                        "experts_stack_num_layers": 1,
                        "use_kv_expert_models_flag": use_expert_kv,
                    }
                config = build_attention_config(
                    config_class=config_class,
                    batch_size=1,
                    num_heads=1,
                    embedding_dim=2,
                    target_sequence_length=1,
                    source_sequence_length=1,
                    zero_attention_flag=True,
                    return_attention_weights_flag=(config_class is SelfAttentionConfig),
                    self_attention_projection_strategy=(
                        SelfAttentionProjectionStrategy.SEPARATE
                    ),
                    **options,
                )
                model = replace(
                    config,
                    target_dtype=torch.bfloat16,
                    batch_first_flag=False,
                ).build()
                model.eval()
                with torch.no_grad():
                    if config_class is MixtureOfAttentionHeadsConfig:
                        for parameter in model.parameters():
                            parameter.zero_()
                        for expert in model.projector.query_model.expert_modules:
                            layer = expert.layers[0].model
                            layer.weight_params.copy_(
                                torch.eye(2, dtype=torch.bfloat16)
                            )
                        for expert in model.projector.output_model.expert_modules:
                            layer = expert.layers[0].model
                            layer.weight_params.copy_(
                                torch.eye(2, dtype=torch.bfloat16)
                            )
                        key_value_models = (
                            model.projector.key_model,
                            model.projector.value_model,
                        )
                        for projection in key_value_models:
                            if hasattr(projection, "expert_modules"):
                                for expert in projection.expert_modules:
                                    expert.layers[0].model.weight_params.copy_(
                                        torch.eye(2, dtype=torch.bfloat16)
                                    )
                            else:
                                projection.layers[0].model.weight_params.copy_(
                                    torch.eye(2, dtype=torch.bfloat16)
                                )
                    else:
                        for projection in (
                            model.projector.query_model,
                            model.projector.key_model,
                            model.projector.value_model,
                            model.projector.output_model,
                        ):
                            layer = projection.layers[0].model
                            layer.weight_params.copy_(
                                torch.eye(2, dtype=torch.bfloat16)
                            )
                            layer.bias_params.zero_()

                inputs = torch.tensor(
                    [[[1.0, 0.0]]],
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                output, weights, auxiliary_loss = model(inputs, inputs, inputs)
                probabilities = torch.softmax(
                    torch.tensor([2**-0.5, 0.0], dtype=torch.bfloat16),
                    dim=0,
                )
                output_scale = (
                    0.5 if config_class is MixtureOfAttentionHeadsConfig else 1.0
                )
                expected_output = torch.tensor(
                    [[[probabilities[0] * output_scale, 0.0]]],
                    dtype=torch.bfloat16,
                )

                torch.testing.assert_close(output, expected_output)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertEqual(output.device, inputs.device)
                if config_class is SelfAttentionConfig:
                    expected_weights = probabilities.reshape(1, 1, 1, 2)
                    torch.testing.assert_close(weights, expected_weights)
                    self.assertEqual(weights.dtype, torch.bfloat16)
                else:
                    self.assertIsNone(weights)
                if config_class is MixtureOfAttentionHeadsConfig:
                    torch.testing.assert_close(
                        auxiliary_loss,
                        auxiliary_loss.new_zeros(()),
                    )
                else:
                    self.assertIsNone(auxiliary_loss)

                output[..., 0].sum().backward()
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertTrue(torch.any(inputs.grad.abs() > 0))

    def test_float64_relative_attention_passes_gradcheck(self):
        config = build_attention_config(
            config_class=IndependentAttentionConfig,
            batch_size=1,
            num_heads=1,
            embedding_dim=2,
            target_sequence_length=2,
            source_sequence_length=2,
            dropout_probability=0.0,
            relative_positional_embedding_config_cls=DynamicPositionalBiasConfig,
        )
        model = replace(config, target_dtype=torch.float64).build().double().eval()
        query = torch.randn(2, 1, 2, dtype=torch.float64, requires_grad=True)
        key = torch.randn(2, 1, 2, dtype=torch.float64, requires_grad=True)
        value = torch.randn(2, 1, 2, dtype=torch.float64, requires_grad=True)

        self.assertTrue(
            torch.autograd.gradcheck(
                lambda q, k, v: model(q, k, v)[0],
                (query, key, value),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
            )
        )

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
    def test_cuda_float32_forward_and_gradients(self):
        self.assert_forward_and_gradients(
            IndependentAttentionConfig,
            torch.float32,
            torch.device("cuda"),
        )

    @unittest.skipUnless(
        CUDA_BFLOAT16_AVAILABLE,
        "CUDA bfloat16 is unavailable",
    )
    def test_cuda_bfloat16_forward_and_gradients(self):
        self.assert_forward_and_gradients(
            SelfAttentionConfig,
            torch.bfloat16,
            torch.device("cuda"),
        )


class TestAttentionFloat64Regression(unittest.TestCase):
    def test_moah_sampler_loss_preserves_float64_for_sparse_and_topk(self):
        for top_k in (1, 2):
            with self.subTest(top_k=top_k):
                config = build_attention_config(
                    config_class=MixtureOfAttentionHeadsConfig,
                    batch_size=1,
                    num_heads=1,
                    embedding_dim=2,
                    target_sequence_length=2,
                    source_sequence_length=2,
                    dropout_probability=0.0,
                    experts_top_k=top_k,
                    experts_num_experts=3,
                )
                model = replace(config, target_dtype=torch.float64).build().double()
                model.eval()
                query = torch.randn(
                    2,
                    1,
                    2,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                key = torch.randn(
                    2,
                    1,
                    2,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                value = torch.randn(
                    2,
                    1,
                    2,
                    dtype=torch.float64,
                    requires_grad=True,
                )

                output, weights, auxiliary_loss = model(query, key, value)
                loss = output.square().sum()
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss
                loss.backward()

                self.assertEqual(output.dtype, torch.float64)
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNone(weights)
                for tensor in (query, key, value):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
