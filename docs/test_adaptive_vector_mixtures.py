import torch
import unittest
import torch.nn.functional as F

from torch.nn import Parameter
from Emperor.adaptive.utils.presets import AdaptiveParameterLayerPresets
from Emperor.adaptive.utils.mixtures.base import (
    AdaptiveMixtureBase,
    AdaptiveMixtureConfig,
)
from Emperor.adaptive.utils.mixtures.types.vector import (
    VectorMixtureBase,
    VectorWeightsMixture,
)


class TestVectorMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = AdaptiveParameterLayerPresets.adaptive_generator_mixture_preset(
            input_dim=8, output_dim=8
        )

    def test_init(self):
        model_types = [VectorWeightsMixture]
        c = self.cfg

        for model_type in model_types:
            message = f"Testing model type: {model_type.__name__}"
            with self.subTest(msg=message):
                m = model_type(self.cfg)

                bank_shape = (c.output_dim, c.num_experts)
                if model_type is VectorWeightsMixture:
                    bank_shape = (c.input_dim, c.num_experts, c.output_dim)

                self.assertIsInstance(m, AdaptiveMixtureBase)
                self.assertIsInstance(m, VectorMixtureBase)
                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertEqual(m.top_k, c.top_k)
                self.assertEqual(m.num_experts, c.num_experts)
                self.assertEqual(m.weighted_parameters_flag, c.weighted_parameters_flag)
                self.assertIsInstance(m.parameter_bank, Parameter)
                self.assertEqual(m.parameter_bank.shape, bank_shape)

    def test_init_parameter_select_range(self):
        model_types = [VectorWeightsMixture]
        top_k_values = [1, 3, 6]
        for model_type in model_types:
            for top_k in top_k_values:
                message = (
                    f"Testing model type: {model_type.__name__} with top_k={top_k}"
                )
                with self.subTest(msg=message):
                    overrides = AdaptiveMixtureConfig(
                        top_k=top_k,
                        num_experts=6,
                        weighted_parameters_flag=True,
                    )
                    m = model_type(self.cfg, overrides)
                    range_indices = m._init_parameter_select_range()

                    expected_range_shape = (1, m.range_dim)
                    if 1 < top_k < m.num_experts:
                        expected_range_shape = (1, m.range_dim, 1)

                    self.assertEqual(range_indices.shape, expected_range_shape)

    def test_select_parameters_weights(self):
        top_k_values = [1, 3, 6]
        for top_k in top_k_values:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                overrides = AdaptiveMixtureConfig(
                    top_k=top_k,
                    num_experts=6,
                )
                m = VectorWeightsMixture(self.cfg, overrides)

                batch_size = 5

                if top_k == 1:
                    indexes_shape = (m.input_dim, batch_size)
                    indexes = torch.randint(0, m.num_experts, indexes_shape)
                elif 1 < top_k < m.num_experts:
                    indexes_shape = (m.input_dim, batch_size, top_k)
                    indexes = torch.randint(0, m.num_experts, indexes_shape)
                else:
                    indexes = None

                selected_params = m._select_parameters(indexes)

                if top_k == 1:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)
                elif 1 < top_k < m.num_experts:
                    expected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
                else:
                    expected_shape = (m.input_dim, m.num_experts, m.output_dim)

                self.assertEqual(selected_params.shape, expected_shape)

    def test_compute_weighted_parameters_weights(self):
        top_k_values = [1, 3, 6]
        for top_k in top_k_values:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                overrides = AdaptiveMixtureConfig(
                    top_k=top_k,
                    num_experts=6,
                    weighted_parameters_flag=True,
                )
                m = VectorWeightsMixture(self.cfg, overrides)

                batch_size = 5

                if top_k == 1:
                    selected_shape = (batch_size, m.input_dim, m.output_dim)
                    probs_shape = (m.input_dim, batch_size)
                elif 1 < top_k < m.num_experts:
                    selected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
                    probs_shape = (m.input_dim, batch_size, top_k)
                else:
                    selected_shape = (m.input_dim, m.num_experts, m.output_dim)
                    probs_shape = (m.input_dim, batch_size, m.num_experts)

                selected_parameters = torch.randn(selected_shape)
                probs = F.sigmoid(torch.randn(probs_shape))
                selected_params = m._compute_weighted_parameters(
                    selected_parameters, probs
                )

                if top_k == 1:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)
                elif 1 < top_k < m.num_experts:
                    expected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
                else:
                    expected_shape = (
                        batch_size,
                        m.input_dim,
                        m.num_experts,
                        m.output_dim,
                    )

                self.assertEqual(selected_params.shape, expected_shape)

    def test__compute_parameter_mixture_weights(self):
        top_k_values = [1, 3, 6]
        for top_k in top_k_values:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                overrides = AdaptiveMixtureConfig(
                    top_k=top_k,
                    num_experts=6,
                    weighted_parameters_flag=True,
                )
                m = VectorWeightsMixture(self.cfg, overrides)

                batch_size = 5

                if top_k == 1:
                    selected_shape = (batch_size, m.input_dim, m.output_dim)
                    probs_shape = (m.input_dim, batch_size)
                elif 1 < top_k < m.num_experts:
                    selected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
                    probs_shape = (m.input_dim, batch_size, top_k)
                else:
                    selected_shape = (m.input_dim, m.num_experts, m.output_dim)
                    probs_shape = (m.input_dim, batch_size, m.num_experts)

                selected_parameters = torch.randn(selected_shape)
                probs = F.sigmoid(torch.randn(probs_shape))
                selected_params = m._VectorMixtureBase__compute_parameter_mixture(
                    selected_parameters, probs
                )

                if top_k == 1:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)
                elif 1 < top_k < m.num_experts:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)
                else:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)

                self.assertEqual(selected_params.shape, expected_shape)

    def test_compute_mixture_weights(self):
        top_k_values = [1, 3, 6]
        for top_k in top_k_values:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                overrides = AdaptiveMixtureConfig(
                    top_k=top_k,
                    num_experts=6,
                    weighted_parameters_flag=True,
                )
                m = VectorWeightsMixture(self.cfg, overrides)

                batch_size = 5

                if top_k == 1:
                    shape = (m.input_dim, batch_size)
                    indices = torch.randint(0, m.num_experts, shape)
                elif 1 < top_k < m.num_experts:
                    shape = (m.input_dim, batch_size, top_k)
                    indices = torch.randint(0, m.num_experts, shape)
                else:
                    shape = (m.input_dim, batch_size, m.num_experts)
                    indices = None

                probs = F.softmax(torch.randn(shape), dim=-1)
                selected_params = m.compute_mixture(probs, indices)

                if top_k == 1:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)
                elif 1 < top_k < m.num_experts:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)
                else:
                    expected_shape = (batch_size, m.input_dim, m.output_dim)

                self.assertEqual(selected_params.shape, expected_shape)

    def test_should_compute_weighted_parameters_subtests(self):
        test_cases = [
            {
                "flag": False,
                "input": None,
                "expected": False,
                "raises": False,
            },
            {
                "flag": False,
                "input": torch.rand(2, 2),
                "expected": False,
                "raises": False,
            },
            {
                "flag": True,
                "input": torch.rand(2, 2),
                "expected": True,
                "raises": False,
            },
            {
                "flag": True,
                "input": None,
                "expected": None,
                "raises": True,
            },
        ]
        for case in test_cases:
            message = (
                f"weighted_parameters_flag flag {case['flag']}, input {case['input']}"
            )
            with self.subTest(msg=message):
                overrides = AdaptiveMixtureConfig(
                    weighted_parameters_flag=case["flag"],
                )
                m = VectorMixtureBase(self.cfg, overrides)
                if case["raises"]:
                    with self.assertRaises(ValueError):
                        m._VectorMixtureBase__should_compute_weighted_parameters(
                            case["input"]
                        )
                else:
                    result = m._VectorMixtureBase__should_compute_weighted_parameters(
                        case["input"]
                    )
                    self.assertEqual(result, case["expected"])
