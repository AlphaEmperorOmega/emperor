import unittest

import torch

from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
from emperor.sampler import RouterConfig, SamplerConfig, SamplerModel
from emperor.sampler._selection.full import SamplerFull
from emperor.sampler._selection.sparse import SamplerSparse
from emperor.sampler._selection.top_k import SamplerTopk
from unit.test_routers import ConstantRouterConfig


class TestSamplerModelValidator(unittest.TestCase):
    def sampler_config(self, **overrides) -> SamplerConfig:
        values = {
            "top_k": 2,
            "threshold": 0.0,
            "filter_above_threshold": False,
            "num_topk_samples": 0,
            "normalize_probabilities_flag": False,
            "noisy_topk_flag": False,
            "num_experts": 4,
            "coefficient_of_variation_loss_weight": 0.0,
            "switch_loss_weight": 0.0,
            "zero_centred_loss_weight": 0.0,
            "mutual_information_loss_weight": 0.0,
            "router_config": None,
        }
        values.update(overrides)
        return SamplerConfig(**values)

    def router_config(
        self,
        input_dim: int = 8,
        hidden_dim: int = 12,
        num_experts: int = 4,
        noisy_topk_flag: bool = False,
    ) -> RouterConfig:
        return RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=noisy_topk_flag,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=num_experts,
                num_layers=2,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=True,
                layer_config=LayerConfig(
                    activation=ActivationOptions.RELU,
                    residual_config=None,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(bias_flag=True),
                ),
            ),
        )

    def test_dispatches_to_expected_sampler_model(self):
        cases = [
            (1, SamplerSparse),
            (2, SamplerTopk),
            (4, SamplerFull),
        ]
        for top_k, expected_cls in cases:
            with self.subTest(top_k=top_k):
                model = SamplerModel(self.sampler_config(top_k=top_k, num_experts=4))

                self.assertIsInstance(model.sampler_model, expected_cls)

    def test_accepts_optional_router_config(self):
        cfg = self.sampler_config(router_config=self.router_config())
        model = SamplerModel(cfg)

        self.assertIsNotNone(model.router)

    def test_accepts_router_config_with_config_base_model_config(self):
        router_config = self.router_config(input_dim=6)
        router_config.model_config = ConstantRouterConfig(input_dim=6, output_dim=4)
        cfg = self.sampler_config(router_config=router_config)
        model = SamplerModel(cfg)

        probabilities, indices, skip_mask, loss = (
            model.sample_probabilities_and_indices(torch.randn(3, 6))
        )

        self.assertEqual(probabilities.shape, (3, 2))
        self.assertEqual(indices.shape, (3, 2))
        self.assertIsNone(skip_mask)
        self.assertIsInstance(loss, torch.Tensor)

    def test_sample_probabilities_and_indices_runs_with_router(self):
        cases = [
            (False, 4),
            (True, 8),
        ]
        for noisy_topk_flag, expected_router_output_dim in cases:
            with self.subTest(noisy_topk_flag=noisy_topk_flag):
                cfg = self.sampler_config(
                    top_k=2,
                    num_experts=4,
                    noisy_topk_flag=noisy_topk_flag,
                    router_config=self.router_config(
                        input_dim=6,
                        num_experts=4,
                        noisy_topk_flag=noisy_topk_flag,
                    ),
                )
                model = SamplerModel(cfg)
                input_matrix = torch.randn(3, 6)

                router_logits = model.router.compute_logit_scores(input_matrix)
                probabilities, indices, skip_mask, loss = (
                    model.sample_probabilities_and_indices(input_matrix)
                )

                self.assertEqual(router_logits.shape, (3, expected_router_output_dim))
                self.assertEqual(probabilities.shape, (3, 2))
                self.assertEqual(indices.shape, (3, 2))
                self.assertIsNone(skip_mask)
                self.assertIsInstance(loss, torch.Tensor)

    def test_build_with_router_input_dim_overrides_router_input_dim(self):
        router_config = self.router_config(
            input_dim=4,
            num_experts=4,
            noisy_topk_flag=True,
        )
        cfg = self.sampler_config(
            top_k=2,
            num_experts=4,
            noisy_topk_flag=True,
            router_config=router_config,
        )

        model = cfg.build_with_router_input_dim(6)
        router_logits = model.router.compute_logit_scores(torch.randn(2, 6))

        self.assertEqual(router_config.input_dim, 4)
        self.assertEqual(model.router.input_dim, 6)
        self.assertEqual(router_logits.shape, (2, 8))

    def test_build_with_router_input_dim_requires_router_config(self):
        cfg = self.sampler_config(router_config=None)

        with self.assertRaises(ValueError):
            cfg.build_with_router_input_dim(6)

    def test_config_preflight_uses_effective_router_input_without_mutation_or_rng(self):
        router_config = self.router_config(input_dim=4)
        router_config.input_dim = None
        cfg = self.sampler_config(router_config=router_config)
        torch.manual_seed(20260719)
        rng_before = torch.random.get_rng_state().clone()

        cfg.validate_for_router_input_dim(6)

        self.assertIsNone(router_config.input_dim)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    def test_config_preflight_covers_logits_only_and_selection_contracts(self):
        logits_only_config = self.sampler_config(router_config=None)
        invalid_selection_config = self.sampler_config(
            router_config=None,
            threshold=-0.5,
        )

        logits_only_config.validate_for_router_input_dim()
        with self.assertRaisesRegex(
            ValueError,
            "threshold must be between 0.0 and 1.0",
        ):
            invalid_selection_config.validate_for_router_input_dim()

    def test_rejects_invalid_model_config_values(self):
        cases = [
            ("top_k", 0, ValueError),
            ("top_k", True, TypeError),
            ("top_k", 5, ValueError),
            ("top_k", 1.5, TypeError),
            ("num_experts", 0, ValueError),
            ("num_experts", True, TypeError),
            ("num_experts", 1.5, TypeError),
            ("router_config", object(), TypeError),
        ]
        for field_name, value, error_type in cases:
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(error_type):
                    SamplerModel(self.sampler_config(**{field_name: value}))

    def test_rejects_mismatched_router_config_values(self):
        cases = [
            {
                "num_experts": 3,
                "noisy_topk_flag": False,
            },
            {
                "num_experts": 4,
                "noisy_topk_flag": True,
            },
        ]
        for router_overrides in cases:
            with self.subTest(router_overrides=router_overrides):
                with self.assertRaises(ValueError):
                    SamplerModel(
                        self.sampler_config(
                            top_k=2,
                            num_experts=4,
                            noisy_topk_flag=False,
                            router_config=self.router_config(**router_overrides),
                        )
                    )

    def test_rejects_invalid_router_model_config(self):
        router_config = RouterConfig(
            input_dim=8,
            num_experts=4,
            noisy_topk_flag=False,
            model_config=object(),
        )

        with self.assertRaises(TypeError):
            SamplerModel(self.sampler_config(router_config=router_config))

    def test_sample_probabilities_and_indices_validates_input_matrix(self):
        model = SamplerModel(self.sampler_config(top_k=2, num_experts=4))
        invalid_type_inputs = [
            [[1.0, 2.0, 3.0, 4.0]],
        ]
        for input_matrix in invalid_type_inputs:
            with self.subTest(input_type=type(input_matrix).__name__):
                with self.assertRaises(TypeError):
                    model.sample_probabilities_and_indices(input_matrix)

        invalid_shape_inputs = [
            torch.randn(4),
            torch.randn(2, 3, 4),
        ]

        for input_matrix in invalid_shape_inputs:
            with self.subTest(input_shape=tuple(input_matrix.shape)):
                with self.assertRaises(ValueError):
                    model.sample_probabilities_and_indices(input_matrix)

    def test_sample_probabilities_and_indices_runs_without_router(self):
        model = SamplerModel(self.sampler_config(top_k=2, num_experts=4))
        probabilities, indices, skip_mask, loss = (
            model.sample_probabilities_and_indices(torch.randn(3, 4))
        )

        self.assertEqual(probabilities.shape, (3, 2))
        self.assertEqual(indices.shape, (3, 2))
        self.assertIsNone(skip_mask)
        self.assertIsInstance(loss, torch.Tensor)
