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
