import torch
import unittest

from emperor.sampler.core.config import SamplerConfig
from emperor.sampler.core.samplers import (
    SamplerBase,
    SamplerFull,
    SamplerSparse,
    SamplerTopk,
)


class TestSamplerValidators(unittest.TestCase):
    def preset(self, **overrides) -> SamplerConfig:
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

    def test_base_validator_accepts_valid_config(self):
        model = SamplerBase(self.preset())

        self.assertEqual(model.top_k, 2)
        self.assertEqual(model.num_experts, 4)

    def test_base_validator_rejects_invalid_config_values(self):
        cases = [
            ("top_k", 0, ValueError),
            ("top_k", True, ValueError),
            ("top_k", 1.5, TypeError),
            ("threshold", -0.1, ValueError),
            ("threshold", 1.1, ValueError),
            ("num_topk_samples", -1, ValueError),
            ("num_topk_samples", 3, ValueError),
            ("num_experts", 0, ValueError),
            ("num_experts", True, ValueError),
            ("coefficient_of_variation_loss_weight", -0.1, ValueError),
            ("switch_loss_weight", -0.1, ValueError),
            ("zero_centred_loss_weight", -0.1, ValueError),
            ("mutual_information_loss_weight", -0.1, ValueError),
        ]
        for field_name, value, error_type in cases:
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaises(error_type):
                    SamplerBase(self.preset(**{field_name: value}))

    def test_sparse_validator_rejects_unsupported_options(self):
        cases = [
            {"normalize_probabilities_flag": True},
            {"num_topk_samples": 1},
            {"mutual_information_loss_weight": 0.1},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    SamplerSparse(self.preset(top_k=1, **overrides))

    def test_topk_validator_requires_partial_topk(self):
        for top_k in [4, 5]:
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    SamplerTopk(self.preset(top_k=top_k, num_experts=4))

    def test_full_validator_rejects_unsupported_options(self):
        cases = [
            {"top_k": 3},
            {"num_topk_samples": 1},
            {"coefficient_of_variation_loss_weight": 0.1},
            {"switch_loss_weight": 0.1},
            {"zero_centred_loss_weight": 0.1},
            {"mutual_information_loss_weight": 0.1},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    values = {"top_k": 4, "num_experts": 4}
                    values.update(overrides)
                    SamplerFull(self.preset(**values))

    def test_get_probabilities_and_indices_validates_runtime_inputs(self):
        model = SamplerTopk(self.preset(top_k=2, num_experts=4))

        invalid_inputs = [
            [[1.0, 2.0, 3.0, 4.0]],
            torch.randn(4),
            torch.randn(2, 3, 4),
            torch.randn(2, 3),
        ]
        for router_logit_scores in invalid_inputs:
            with self.subTest(input_type=type(router_logit_scores).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    model.get_probabilities_and_indices(router_logit_scores)

    def test_get_probabilities_and_indices_validates_skip_mask(self):
        model = SamplerTopk(self.preset(top_k=2, num_experts=4))
        router_logit_scores = torch.randn(3, 4)

        with self.assertRaises(TypeError):
            model.get_probabilities_and_indices(router_logit_scores, skip_mask=[1, 1, 1])

        with self.assertRaises(ValueError):
            model.get_probabilities_and_indices(
                router_logit_scores, skip_mask=torch.ones(2, 1)
            )
