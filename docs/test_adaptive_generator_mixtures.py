import torch
import unittest

from torch.types import Tensor
from Emperor.adaptive.utils.enums import ClipParameterOptions
from Emperor.adaptive.utils.presets import ParameterGeneratorConfigs
from Emperor.adaptive.utils.mixtures.base import AdaptiveMixtureBase
from Emperor.adaptive.utils.mixtures.generator import (
    GeneratorBiasMixture,
    GeneratorMixtureBase,
    GeneratorWeightsMixture,
)
from Emperor.experts.utils.layers import MixtureOfExperts
from Emperor.sampler.model import SamplerModel
from Emperor.sampler.utils.presets import SamplerPresets
from Emperor.sampler.utils.routers import RouterModel


class TestGeneratorMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = ParameterGeneratorConfigs.adaptive_generator_mixture_preset()

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
        boolean_flags = [True, False]
        c = self.cfg

        for top_k in top_k_values:
            for init_sampler_model_flag in boolean_flags:
                message = f"Testing top_k value: {top_k}, init_sampler_model_flag: {init_sampler_model_flag}"
                with self.subTest(msg=message):
                    c = ParameterGeneratorConfigs.adaptive_generator_mixture_preset(
                        top_k=top_k,
                        init_sampler_model_flag=init_sampler_model_flag,
                        experts_weighted_parameters_flag=True,
                    )
                    m = GeneratorWeightsMixture(c)

                    batch_size = 5
                    input_batch = torch.randn(batch_size, m.input_dim)
                    probabilities, indices = None, None
                    if not init_sampler_model_flag:
                        router_cfg = SamplerPresets.router_preset(input_dim=8)
                        sampler_cfg = SamplerPresets.sampler_preset(top_k=top_k)
                        router = RouterModel(router_cfg)
                        sampler = SamplerModel(sampler_cfg)

                        logits = router.compute_logit_scores(input_batch)
                        probabilities, indices, _, _ = (
                            sampler.sample_probabilities_and_indices(logits)
                        )

                    input_vectors, _ = m.input_vector_generator(
                        input_batch, probabilities, indices
                    )
                    output_vectors, _ = m.output_vector_generator(
                        input_batch, probabilities, indices
                    )

                    outer_product = m._GeneratorWeightsMixture__compute_outer_product(
                        input_vectors,
                        output_vectors,
                    )

                    expected_shape = (batch_size, m.top_k, m.input_dim, m.output_dim)
                    self.assertEqual(outer_product.shape, expected_shape)
                    self.assertIsInstance(outer_product, torch.Tensor)

    def test__compute_parameter_mixture(self):
        top_k_values = [1, 3, 6]
        boolean_flags = [True, False]
        num_experts = 6
        c = self.cfg

        for top_k in top_k_values:
            for weighted_parameters_flag in boolean_flags:
                message = f"Testing with top_k={top_k}, weighted_parameters_flag={weighted_parameters_flag}"
                with self.subTest(msg=message):
                    if top_k == num_experts:
                        weighted_parameters_flag = True
                    c = ParameterGeneratorConfigs.adaptive_generator_mixture_preset(
                        top_k=top_k,
                        weighted_parameters_flag=weighted_parameters_flag,
                    )
                    m = GeneratorWeightsMixture(c)

                    batch_size = 5
                    generated_parameters = torch.randn(
                        batch_size, top_k, m.input_dim, m.output_dim
                    )
                    probabilities = torch.randn(batch_size, top_k)
                    if top_k > 1:
                        probabilities = torch.softmax(
                            torch.randn(batch_size, top_k), dim=-1
                        )

                    parameter_mixture = (
                        m._GeneratorWeightsMixture__compute_parameter_mixture(
                            generated_parameters, probabilities
                        )
                    )

                    expected_shape = (batch_size, m.input_dim, m.output_dim)
                    self.assertEqual(parameter_mixture.shape, expected_shape)
                    self.assertIsInstance(parameter_mixture, torch.Tensor)
                    if weighted_parameters_flag:
                        if top_k == 1:
                            self.assertFalse(
                                torch.allclose(
                                    parameter_mixture,
                                    generated_parameters.squeeze(1),
                                )
                            )
                        else:
                            self.assertFalse(
                                torch.allclose(
                                    parameter_mixture,
                                    generated_parameters.sum(dim=1),
                                )
                            )
                    else:
                        if top_k == 1:
                            self.assertTrue(
                                torch.allclose(
                                    parameter_mixture,
                                    generated_parameters.squeeze(1),
                                )
                            )
                        else:
                            self.assertTrue(
                                torch.allclose(
                                    parameter_mixture,
                                    generated_parameters.sum(dim=1),
                                )
                            )

    def test__apply_parameter_weighting(self):
        top_k_values = [1, 3, 6]
        for top_k in top_k_values:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                c = ParameterGeneratorConfigs.adaptive_generator_mixture_preset(
                    top_k=top_k,
                    weighted_parameters_flag=True,
                )
                m = GeneratorWeightsMixture(c)

                batch_size = 5
                generated_parameters = torch.randn(
                    batch_size, top_k, m.input_dim, m.output_dim
                )

                probabilities = torch.softmax(torch.randn(batch_size, top_k), dim=-1)
                if top_k == 1:
                    probabilities = torch.randn(batch_size, top_k)

                weighted_parameters = (
                    m._GeneratorWeightsMixture__apply_parameter_weighting(
                        generated_parameters, probabilities
                    )
                )

                expected_shape = (batch_size, top_k, m.input_dim, m.output_dim)
                self.assertEqual(weighted_parameters.shape, expected_shape)
                self.assertFalse(
                    torch.allclose(weighted_parameters, generated_parameters)
                )

    def test_compute_mixture(self):
        num_experts = 6
        top_k_values = [1, 3, 6]
        boolean_flags = [True, False]
        c = self.cfg

        for top_k in top_k_values:
            for init_sampler_model_flag in boolean_flags:
                for weighted_parameters_flag in boolean_flags:
                    for clip_parameter_option in ClipParameterOptions:
                        message = f"Testing top_k={top_k}, init_sampler_model_flag={init_sampler_model_flag}, weighted_parameters_flag={weighted_parameters_flag}, clip_parameter_option={clip_parameter_option}"
                        with self.subTest(msg=message):
                            c = ParameterGeneratorConfigs.adaptive_generator_mixture_preset(
                                top_k=top_k,
                                num_experts=num_experts,
                                init_sampler_model_flag=init_sampler_model_flag,
                                weighted_parameters_flag=weighted_parameters_flag,
                                experts_weighted_parameters_flag=not weighted_parameters_flag,
                                clip_parameter_option=clip_parameter_option,
                            )
                            m = GeneratorWeightsMixture(c)

                            batch_size = 5
                            input_batch = torch.randn(batch_size, m.input_dim)
                            probabilities, indices = None, None
                            if (
                                not init_sampler_model_flag
                                or m.weighted_parameters_flag
                                or m.num_experts == m.top_k
                                and not init_sampler_model_flag
                            ):
                                router_cfg = SamplerPresets.router_preset(input_dim=8)
                                sampler_cfg = SamplerPresets.sampler_preset(top_k=top_k)
                                router = RouterModel(router_cfg)
                                sampler = SamplerModel(sampler_cfg)

                                logits = router.compute_logit_scores(input_batch)
                                probabilities, indices, _, _ = (
                                    sampler.sample_probabilities_and_indices(logits)
                                )

                            generated_parameters, loss = m.compute_mixture(
                                input_batch,
                                probabilities,
                                indices,
                            )

                            expected_shape = (batch_size, m.input_dim, m.output_dim)
                            self.assertEqual(generated_parameters.shape, expected_shape)
                            self.assertIsInstance(generated_parameters, Tensor)
                            self.assertIsInstance(loss, Tensor)

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
