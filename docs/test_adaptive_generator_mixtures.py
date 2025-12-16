import torch
import unittest

from torch.nn import functional as F
from Emperor.adaptive.utils.mixture import AdaptiveMixtureConfig
from Emperor.adaptive.utils.presets import ParameterGeneratorConfigs
from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureBase
from Emperor.adaptive.utils.mixtures.generator import (
    GeneratorBiasMixture,
    GeneratorMixtureBase,
    GeneratorWeightsMixture,
)
from Emperor.experts.utils.layers import MixtureOfExperts


class TestGeneratorMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = ParameterGeneratorConfigs.adaptive_generator_mixture_preset(
            return_model_config_flag=True,
        )

    def tearDown(self):
        self.cfg = None

    def test_init(self):
        model_types = [GeneratorBiasMixture, GeneratorWeightsMixture]
        c = self.cfg

        for model_type in model_types:
            message = f"Testing model type: {model_type.__name__}"
            with self.subTest(msg=message):
                m = model_type(self.cfg)

                self.assertIsInstance(m, AdaptiveMixtureBase)
                self.assertIsInstance(m, GeneratorMixtureBase)
                self.assertEqual(m.input_dim, c.input_dim)
                self.assertEqual(m.output_dim, c.output_dim)
                self.assertEqual(m.top_k, c.top_k)
                self.assertEqual(m.num_experts, c.num_experts)
                self.assertEqual(m.weighted_parameters_flag, c.weighted_parameters_flag)
                if model_type == GeneratorWeightsMixture:
                    self.assertIsInstance(m.input_vector_generator, MixtureOfExperts)
                    self.assertIsInstance(m.output_vector_generator, MixtureOfExperts)
                else:
                    self.assertIsInstance(m.bias_generator, MixtureOfExperts)

    def test__compute_outer_product(self):
        top_k_values = [1, 3, 6]
        c = self.cfg

        for top_k in top_k_values:
            message = f"Testing top_k value: {top_k}"
            with self.subTest(msg=message):
                c = ParameterGeneratorConfigs.adaptive_generator_mixture_preset(
                    top_k=top_k,
                )
                m = GeneratorWeightsMixture(c)

                batch_size = 5
                input_batch = torch.randn(batch_size, m.input_dim)
                input_vectors, _ = m.input_vector_generator(input_batch)
                output_vectors, _ = m.output_vector_generator(input_batch)

                outer_product = m._GeneratorWeightsMixture__compute_outer_product(
                    input_vectors=input_vectors,
                    output_vectors=output_vectors,
                )

                # print(outer_product.shape)

                # selected_parameters = torch.randn(selected_shape)
                # probs = F.sigmoid(torch.randn(probs_shape))

    # def test_compute_weighted_parameters_weights(self):
    #     top_k_values = [1, 3, 6]
    #     for top_k in top_k_values:
    #         message = f"Testing with top_k={top_k}"
    #         with self.subTest(msg=message):
    #             overrides = MixtureConfig(
    #                 top_k=top_k,
    #                 depth_dim=6,
    #                 num_experts=6,
    #                 weighted_parameters_flag=True,
    #             )
    #             m = VectorWeightsMixture(self.cfg, overrides)
    #
    #             batch_size = 5
    #
    #             if top_k == 1:
    #                 selected_shape = (batch_size, m.input_dim, m.output_dim)
    #                 probs_shape = (m.input_dim, batch_size)
    #             elif 1 < top_k < m.depth_dim:
    #                 selected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
    #                 probs_shape = (m.input_dim, batch_size, top_k)
    #             else:
    #                 selected_shape = (m.input_dim, m.depth_dim, m.output_dim)
    #                 probs_shape = (m.input_dim, batch_size, m.depth_dim)
    #
    #             selected_parameters = torch.randn(selected_shape)
    #             probs = F.sigmoid(torch.randn(probs_shape))
    #             selected_params = m._compute_weighted_parameters(
    #                 selected_parameters, probs
    #             )
    #
    #             if top_k == 1:
    #                 expected_shape = (batch_size, m.input_dim, m.output_dim)
    #             elif 1 < top_k < m.depth_dim:
    #                 expected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
    #             else:
    #                 expected_shape = (
    #                     batch_size,
    #                     m.input_dim,
    #                     m.depth_dim,
    #                     m.output_dim,
    #                 )
    #
    #             self.assertEqual(selected_params.shape, expected_shape)
    #
    # def test_compute_weighted_parameters_biases(self):
    #     model_types = [VectorWeightsMixture, VectorBiasMixture]
    #     top_k_values = [1, 3, 6]
    #     for model_type in model_types:
    #         for top_k in top_k_values:
    #             message = (
    #                 f"Testing model type: {model_type.__name__} with top_k={top_k}"
    #             )
    #             with self.subTest(msg=message):
    #                 overrides = MixtureConfig(
    #                     top_k=top_k,
    #                     depth_dim=6,
    #                     num_experts=6,
    #                     weighted_parameters_flag=True,
    #                 )
    #                 m = VectorBiasMixture(self.cfg, overrides)
    #
    #                 batch_size = 5
    #
    #                 if top_k == 1:
    #                     selected_shape = (batch_size, m.output_dim)
    #                     probs_shape = (m.output_dim, batch_size)
    #                 elif 1 < top_k < m.depth_dim:
    #                     selected_shape = (batch_size, m.output_dim, top_k)
    #                     probs_shape = (m.output_dim, batch_size, m.top_k)
    #                 else:
    #                     selected_shape = (m.output_dim, m.depth_dim)
    #                     probs_shape = (m.output_dim, batch_size, m.depth_dim)
    #
    #                 selected_parameters = torch.randn(selected_shape)
    #                 probs = F.sigmoid(torch.randn(probs_shape))
    #                 selected_params = m._compute_weighted_parameters(
    #                     selected_parameters, probs
    #                 )
    #
    #                 if top_k == 1:
    #                     expected_shape = (batch_size, m.output_dim)
    #                 elif 1 < top_k < m.depth_dim:
    #                     expected_shape = (batch_size, m.output_dim, top_k)
    #                 else:
    #                     expected_shape = (batch_size, m.output_dim, m.depth_dim)
    #
    #                 self.assertEqual(selected_params.shape, expected_shape)
    #
    # def test__compute_parameter_mixture_weights(self):
    #     model_types = [VectorWeightsMixture, VectorBiasMixture]
    #     top_k_values = [1, 3, 6]
    #     for model_type in model_types:
    #         for top_k in top_k_values:
    #             message = (
    #                 f"Testing model type: {model_type.__name__} with top_k={top_k}"
    #             )
    #             with self.subTest(msg=message):
    #                 overrides = MixtureConfig(
    #                     top_k=top_k,
    #                     depth_dim=6,
    #                     num_experts=6,
    #                     weighted_parameters_flag=True,
    #                 )
    #                 m = VectorWeightsMixture(self.cfg, overrides)
    #
    #                 batch_size = 5
    #
    #                 if top_k == 1:
    #                     selected_shape = (batch_size, m.input_dim, m.output_dim)
    #                     probs_shape = (m.input_dim, batch_size)
    #                 elif 1 < top_k < m.depth_dim:
    #                     selected_shape = (batch_size, m.input_dim, top_k, m.output_dim)
    #                     probs_shape = (m.input_dim, batch_size, top_k)
    #                 else:
    #                     selected_shape = (m.input_dim, m.depth_dim, m.output_dim)
    #                     probs_shape = (m.input_dim, batch_size, m.depth_dim)
    #
    #                 selected_parameters = torch.randn(selected_shape)
    #                 probs = F.sigmoid(torch.randn(probs_shape))
    #                 selected_params = m._VectorMixtureBase__compute_parameter_mixture(
    #                     selected_parameters, probs
    #                 )
    #
    #                 if top_k == 1:
    #                     expected_shape = (batch_size, m.input_dim, m.output_dim)
    #                 elif 1 < top_k < m.depth_dim:
    #                     expected_shape = (batch_size, m.input_dim, m.output_dim)
    #                 else:
    #                     expected_shape = (batch_size, m.input_dim, m.output_dim)
    #
    #                 self.assertEqual(selected_params.shape, expected_shape)
    #
    # def test__compute_parameter_mixture_biases(self):
    #     top_k_values = [1, 3, 6]
    #     for top_k in top_k_values:
    #         message = f"Testing with top_k={top_k}"
    #         with self.subTest(msg=message):
    #             overrides = MixtureConfig(
    #                 top_k=top_k,
    #                 depth_dim=6,
    #                 num_experts=6,
    #                 weighted_parameters_flag=True,
    #             )
    #             m = VectorBiasMixture(self.cfg, overrides)
    #
    #             batch_size = 5
    #
    #             if top_k == 1:
    #                 selected_shape = (batch_size, m.output_dim)
    #                 probs_shape = (m.output_dim, batch_size)
    #             elif 1 < top_k < m.depth_dim:
    #                 selected_shape = (batch_size, m.output_dim, top_k)
    #                 probs_shape = (m.output_dim, batch_size, m.top_k)
    #             else:
    #                 selected_shape = (m.output_dim, m.depth_dim)
    #                 probs_shape = (m.output_dim, batch_size, m.depth_dim)
    #
    #             selected_parameters = torch.randn(selected_shape)
    #             probs = F.sigmoid(torch.randn(probs_shape))
    #             selected_params = m._VectorMixtureBase__compute_parameter_mixture(
    #                 selected_parameters, probs
    #             )
    #
    #             if top_k == 1:
    #                 expected_shape = (batch_size, m.output_dim)
    #             elif 1 < top_k < m.depth_dim:
    #                 expected_shape = (batch_size, m.output_dim)
    #             else:
    #                 expected_shape = (batch_size, m.output_dim)
    #
    #             self.assertEqual(selected_params.shape, expected_shape)
    #
    # def test_compute_mixture_weights(self):
    #     top_k_values = [1, 3, 6]
    #     for top_k in top_k_values:
    #         message = f"Testing with top_k={top_k}"
    #         with self.subTest(msg=message):
    #             overrides = MixtureConfig(
    #                 top_k=top_k,
    #                 depth_dim=6,
    #                 num_experts=6,
    #                 weighted_parameters_flag=True,
    #             )
    #             m = VectorWeightsMixture(self.cfg, overrides)
    #
    #             batch_size = 5
    #
    #             if top_k == 1:
    #                 shape = (m.input_dim, batch_size)
    #                 indices = torch.randint(0, m.depth_dim, shape)
    #             elif 1 < top_k < m.depth_dim:
    #                 shape = (m.input_dim, batch_size, top_k)
    #                 indices = torch.randint(0, m.depth_dim, shape)
    #             else:
    #                 shape = (m.input_dim, batch_size, m.depth_dim)
    #                 indices = None
    #
    #             probs = F.softmax(torch.randn(shape), dim=-1)
    #             selected_params = m.compute_mixture(probs, indices)
    #
    #             if top_k == 1:
    #                 expected_shape = (batch_size, m.input_dim, m.output_dim)
    #             elif 1 < top_k < m.depth_dim:
    #                 expected_shape = (batch_size, m.input_dim, m.output_dim)
    #             else:
    #                 expected_shape = (batch_size, m.input_dim, m.output_dim)
    #
    #             self.assertEqual(selected_params.shape, expected_shape)
    #
    # def test_compute_mixture_biases(self):
    #     top_k_values = [1, 3, 6]
    #     for top_k in top_k_values:
    #         message = f"Testing with top_k={top_k}"
    #         with self.subTest(msg=message):
    #             overrides = MixtureConfig(
    #                 top_k=top_k,
    #                 depth_dim=6,
    #                 num_experts=6,
    #                 weighted_parameters_flag=True,
    #             )
    #             m = VectorBiasMixture(self.cfg, overrides)
    #
    #             batch_size = 5
    #
    #             if top_k == 1:
    #                 shape = (m.output_dim, batch_size)
    #                 indices = torch.randint(0, m.depth_dim, shape)
    #             elif 1 < top_k < m.depth_dim:
    #                 shape = (m.output_dim, batch_size, top_k)
    #                 indices = torch.randint(0, m.depth_dim, shape)
    #             else:
    #                 shape = (m.output_dim, batch_size, m.depth_dim)
    #                 indices = None
    #
    #             probs = F.softmax(torch.randn(shape), dim=-1)
    #             selected_params = m.compute_mixture(probs, indices)
    #
    #             if top_k == 1:
    #                 expected_shape = (batch_size, m.output_dim)
    #             elif 1 < top_k < m.depth_dim:
    #                 expected_shape = (batch_size, m.output_dim)
    #             else:
    #                 expected_shape = (batch_size, m.output_dim)
    #
    #             self.assertEqual(selected_params.shape, expected_shape)
    #
    # def test_should_compute_weighted_parameters_subtests(self):
    #     test_cases = [
    #         {
    #             "flag": False,
    #             "input": None,
    #             "expected": False,
    #             "raises": False,
    #         },
    #         {
    #             "flag": False,
    #             "input": torch.rand(2, 2),
    #             "expected": False,
    #             "raises": False,
    #         },
    #         {
    #             "flag": True,
    #             "input": torch.rand(2, 2),
    #             "expected": True,
    #             "raises": False,
    #         },
    #         {
    #             "flag": True,
    #             "input": None,
    #             "expected": None,
    #             "raises": True,
    #         },
    #     ]
    #     for case in test_cases:
    #         message = (
    #             f"weighted_parameters_flag flag {case['flag']}, input {case['input']}"
    #         )
    #         with self.subTest(msg=message):
    #             overrides = MixtureConfig(
    #                 weighted_parameters_flag=case["flag"],
    #             )
    #             m = VectorMixtureBase(self.cfg, overrides)
    #             if case["raises"]:
    #                 with self.assertRaises(ValueError):
    #                     m._VectorMixtureBase__should_compute_weighted_parameters(
    #                         case["input"]
    #                     )
    #             else:
    #                 result = m._VectorMixtureBase__should_compute_weighted_parameters(
    #                     case["input"]
    #                 )
    #                 self.assertEqual(result, case["expected"])
