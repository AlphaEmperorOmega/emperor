import torch
import unittest

from torch.types import Tensor
from emperor.experts.utils.enums import InitSamplerOptions
from emperor.sampler.model import SamplerModel
from emperor.sampler.utils.routers import RouterModel
from emperor.sampler.utils.presets import SamplerPresets
from emperor.experts.utils.layers import MixtureOfExperts
from emperor.parametric.utils.presets import ParametricLayerPresets
from emperor.parametric.utils.mixtures.base import AdaptiveMixtureBase
from emperor.parametric.utils.mixtures.types.utils.enums import ClipParameterOptions
from emperor.parametric.utils.mixtures.types.generator import (
    GeneratorBiasMixture,
    GeneratorMixtureBase,
    GeneratorWeightsMixture,
)


class TestGeneratorMixture(unittest.TestCase):
    def setUp(self):
        self.cfg = (
            ParametricLayerPresets.adaptive_generator_mixture_generator_preset()
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
        init_sampler_model_options = [
            InitSamplerOptions.DISABLED,
            InitSamplerOptions.LAYER,
        ]

        c = self.cfg
        for top_k in top_k_values:
            for init_sampler_model_option in init_sampler_model_options:
                message = f"Testing top_k value: {top_k}, init_sampler_model_option: {init_sampler_model_option}"
                with self.subTest(msg=message):
                    c = ParametricLayerPresets.adaptive_generator_mixture_generator_preset(
                        top_k=top_k,
                        experts_init_sampler_option=init_sampler_model_option,
                        experts_weighted_parameters_flag=True,
                    )
                    m = GeneratorWeightsMixture(c)

                    batch_size = 5
                    input_batch = torch.randn(batch_size, m.input_dim)
                    probabilities, indices = None, None
                    if init_sampler_model_option == InitSamplerOptions.DISABLED:
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
                    c = ParametricLayerPresets.adaptive_generator_mixture_generator_preset(
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
                                    parameter_mixture.round(decimals=4),
                                    generated_parameters.squeeze(1).round(decimals=4),
                                    atol=1e-6,
                                    rtol=1e-5,
                                )
                            )
                        else:
                            self.assertFalse(
                                torch.allclose(
                                    parameter_mixture.round(decimals=4),
                                    generated_parameters.sum(dim=1).round(decimals=4),
                                    atol=1e-6,
                                    rtol=1e-5,
                                )
                            )
                    else:
                        if top_k == 1:
                            self.assertTrue(
                                torch.allclose(
                                    parameter_mixture.round(decimals=4),
                                    generated_parameters.squeeze(1).round(decimals=4),
                                    atol=1e-6,
                                    rtol=1e-5,
                                )
                            )
                        else:
                            self.assertTrue(
                                torch.allclose(
                                    parameter_mixture.round(decimals=4),
                                    generated_parameters.sum(dim=1).round(decimals=4),
                                    atol=1e-6,
                                    rtol=1e-5,
                                )
                            )

    def test__apply_parameter_weighting(self):
        top_k_values = [1, 3, 6]
        for top_k in top_k_values:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                c = ParametricLayerPresets.adaptive_generator_mixture_generator_preset(
                    top_k=top_k,
                    weighted_parameters_flag=True,
                )
                m = GeneratorWeightsMixture(c)

                batch_size = 8
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
                    torch.allclose(
                        weighted_parameters.round(decimals=4),
                        generated_parameters.round(decimals=4),
                        atol=1e-6,
                        rtol=1e-5,
                    )
                )

    def test_compute_mixture(self):
        num_experts = 6
        top_k_values = [1, 3, 6]
        boolean_flags = [True, False]
        init_sampler_model_options = [
            InitSamplerOptions.DISABLED,
            InitSamplerOptions.LAYER,
        ]
        c = self.cfg

        for top_k in top_k_values:
            for init_sampler_model_option in init_sampler_model_options:
                for weighted_parameters_flag in boolean_flags:
                    for clip_parameter_option in ClipParameterOptions:
                        message = f"Testing top_k={top_k}, init_sampler_model_option={init_sampler_model_option}, weighted_parameters_flag={weighted_parameters_flag}, clip_parameter_option={clip_parameter_option}"
                        with self.subTest(msg=message):
                            c = ParametricLayerPresets.adaptive_generator_mixture_generator_preset(
                                top_k=top_k,
                                num_experts=num_experts,
                                experts_init_sampler_option=init_sampler_model_option,
                                weighted_parameters_flag=weighted_parameters_flag,
                                experts_weighted_parameters_flag=not weighted_parameters_flag,
                                clip_parameter_option=clip_parameter_option,
                            )
                            m = GeneratorWeightsMixture(c)

                            batch_size = 5
                            input_batch = torch.randn(batch_size, m.input_dim)
                            probabilities, indices = None, None
                            if (
                                init_sampler_model_option == InitSamplerOptions.DISABLED
                                or m.weighted_parameters_flag
                                or m.num_experts == m.top_k
                                and init_sampler_model_option
                                == InitSamplerOptions.DISABLED
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
                                probabilities,
                                indices,
                                input_batch,
                            )

                            expected_shape = (batch_size, m.input_dim, m.output_dim)
                            self.assertEqual(generated_parameters.shape, expected_shape)
                            self.assertIsInstance(generated_parameters, Tensor)
                            self.assertIsInstance(loss, Tensor)
