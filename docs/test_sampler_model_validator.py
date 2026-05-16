import torch
import unittest

from emperor.base.enums import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.base.layer import LayerConfig, LayerStackConfig
from emperor.linears.core.config import LinearLayerConfig
from emperor.sampler.model import SamplerModel
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from emperor.sampler.core.samplers import SamplerFull, SamplerSparse, SamplerTopk


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
                    residual_flag=False,
                    dropout_probability=0.0,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    shared_halting_flag=False,
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

    def test_rejects_invalid_model_config_values(self):
        cases = [
            ("top_k", 0, ValueError),
            ("top_k", True, ValueError),
            ("top_k", 5, ValueError),
            ("top_k", 1.5, TypeError),
            ("num_experts", 0, ValueError),
            ("num_experts", True, ValueError),
            ("num_experts", 1.5, TypeError),
            ("router_config", object(), TypeError),
        ]
        for field_name, value, error_type in cases:
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(error_type):
                    SamplerModel(self.sampler_config(**{field_name: value}))

    def test_sample_probabilities_and_indices_validates_input_matrix(self):
        model = SamplerModel(self.sampler_config(top_k=2, num_experts=4))
        invalid_inputs = [
            [[1.0, 2.0, 3.0, 4.0]],
            torch.randn(4),
            torch.randn(2, 3, 4),
        ]

        for input_matrix in invalid_inputs:
            with self.subTest(input_type=type(input_matrix).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    model.sample_probabilities_and_indices(input_matrix)

    def test_sample_probabilities_and_indices_runs_without_router(self):
        model = SamplerModel(self.sampler_config(top_k=2, num_experts=4))
        probabilities, indices, skip_mask, loss = model.sample_probabilities_and_indices(
            torch.randn(3, 4)
        )

        self.assertEqual(probabilities.shape, (3, 2))
        self.assertEqual(indices.shape, (3, 2))
        self.assertIsNone(skip_mask)
        self.assertIsInstance(loss, torch.Tensor)
