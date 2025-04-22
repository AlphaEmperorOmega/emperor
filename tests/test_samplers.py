import unittest
import torch
import torch.nn as nn
from Emperor.components.parameter_generators.utils.samplers import (
    SamplerBase,
    SamplerConfig,
    SamplerSparse,
    SamplerTopk,
    SamplerFull,
    SamplerModel,
)
from Emperor.components.parameter_generators.utils.routers import (
    RouterModel,
    VectorChoiceRouterModel,
)
from Emperor.config import ModelConfig


class TestProbabilitySampler(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()
        self.router_cfg = self.cfg.router_model_config
        self.sampler_cfg = self.cfg.sampler_model_config

    def test__init_with_cfg(self):
        sampler = SamplerBase(cfg=self.cfg)

        self.assertEqual(sampler.top_k, self.sampler_cfg.top_k)
        self.assertEqual(sampler.threshold, self.sampler_cfg.threshold)
        self.assertEqual(sampler.num_topk_samples, self.sampler_cfg.num_topk_samples)
        self.assertEqual(sampler.noisy_topk_flag, self.sampler_cfg.noisy_topk_flag)

    def test__init_with_custom_config(self):
        config = SamplerConfig(
            top_k=4,
            threshold=0.1,
            dynamic_topk_threshold=0.2,
            num_topk_samples=2,
            normalize_probabilities_flag=True,
            noisy_topk_flag=True,
        )

        sampler = SamplerBase(config)
        sampler.set_router_model(VectorChoiceRouterModel, self.cfg)

        self.assertEqual(sampler.top_k, config.top_k)
        self.assertEqual(sampler.threshold, config.threshold)
        self.assertEqual(sampler.dynamic_topk_threshold, config.dynamic_topk_threshold)
        self.assertEqual(sampler.num_topk_samples, config.num_topk_samples)
        self.assertEqual(
            sampler.normalize_probabilities_flag, config.normalize_probabilities_flag
        )
        self.assertEqual(sampler.noisy_topk_flag, config.noisy_topk_flag)
        self.assertTrue(isinstance(sampler.router_model, VectorChoiceRouterModel))

    def test__init_with_custom_router_model(self):
        overrides = SamplerConfig(
            router_model=lambda cfg: VectorChoiceRouterModel(cfg),
        )

        sampler = SamplerBase(self.cfg, overrides)

        self.assertTrue(isinstance(sampler.router_model, VectorChoiceRouterModel))

    def test__router_setter_and_main_config(self):
        sampler = SamplerBase(self.cfg)
        sampler.set_router_model(VectorChoiceRouterModel, self.cfg)

        self.assertTrue(isinstance(sampler.router_model, VectorChoiceRouterModel))

    def test__if_missing_config_values_raise_errors(self):
        default_config = {
            "top_k": 10,
            "threshold": 0.01,
            "dynamic_topk_threshold": 0.05,
            "num_topk_samples": 3,
            "normalize_probabilities_flag": True,
            "noisy_topk_flag": False,
            "router_model": None,
        }
        elements = 1
        for _, _ in default_config.items():
            dynamic_config = dict(list(default_config.items())[:elements])
            elements += 1
            custom_config = SamplerConfig()

            for cfg_name, cfg_val in dynamic_config.items():
                setattr(custom_config, cfg_name, cfg_val)

            if "noisy_topk_flag" in dynamic_config:
                # This is here because the router_model is optional
                # and if the value is in the dictionary the assertion
                # will look at the router_model to return an error
                # but it will not
                continue

            with self.assertRaises(ValueError):
                SamplerBase(custom_config)

    def test__add_noise_to_logits_flag_false(self):
        sampler = SamplerBase(cfg=self.cfg)

        sampler.noisy_topk_flag = False
        logits = torch.ones(2, 3, 8)
        result = sampler._SamplerBase__add_noise_to_logits(logits)
        self.assertEqual(logits.mean(), result.mean())

    def test__add_noise_to_logits_flag_true(self):
        sampler = SamplerBase(cfg=self.cfg)

        sampler.noisy_topk_flag = True
        sampler.set_is_training_flag(True)
        logits = torch.ones(2, 3, 16)
        result = sampler._SamplerBase__add_noise_to_logits(logits)

        self.assertNotEqual(logits.mean(), result.mean())
        self.assertEqual(result.shape, torch.Size([2, 3, 8]))

    def test__apply_skip_mask(self):
        sampler = SamplerBase(cfg=self.cfg)

        probs = torch.ones(6, 4)
        logits = torch.ones(6, 4)

        mask = torch.zeros(2, 3).reshape(-1, 1)
        mask[0, :] = 1

        masked_probs, _ = sampler._SamplerBase__apply_skip_mask(probs, logits, mask)

        self.assertTrue(torch.sum(masked_probs[0, :]).item() > 0)
        self.assertTrue(torch.sum(masked_probs[1:, :]).item() == 0)

    def test__normalize_probabilities(self):
        sampler = SamplerBase(cfg=self.cfg)

        probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
        result = sampler._normalize_probabilities(probs)
        sums = torch.sum(result, dim=1)

        torch.testing.assert_close(sums, torch.ones_like(sums))

    def test__compute_masked_probabilities_input_batch_only(self):
        sampler = SamplerBase(cfg=self.cfg)
        sequence_length = 2
        batch_size = 3
        input_shape = [
            sequence_length,
            batch_size,
            self.cfg.router_model_config.input_dim,
        ]
        test_input = torch.randn(*input_shape)

        full_probabilities, _ = sampler._SamplerBase__compute_masked_probabilities(
            test_input
        )

        expectedShape = [
            sequence_length,
            batch_size,
            self.cfg.router_model_config.output_dim,
        ]
        result_shape = list(full_probabilities.shape)

        self.assertEqual(result_shape, expectedShape)
        self.assertAlmostEqual(
            full_probabilities.sum().item(), sequence_length * batch_size, places=5
        )

    def test__compute_masked_probabilities_input_batch_and_skip_mask(self):
        sampler = SamplerBase(cfg=self.cfg)
        sequence_length = 2
        batch_size = 3
        input_shape = [
            sequence_length,
            batch_size,
            self.cfg.router_model_config.input_dim,
        ]
        test_input = torch.randn(*input_shape)
        skip_mask = torch.zeros(sequence_length, batch_size)
        skip_mask[0, :] = 1
        skip_mask[0, 1] = 0
        skip_mask_reshaped = skip_mask.unsqueeze(-1)

        full_probabilities, _ = sampler._SamplerBase__compute_masked_probabilities(
            test_input, skip_mask_reshaped
        )

        expectedShape = [
            sequence_length,
            batch_size,
            self.cfg.router_model_config.output_dim,
        ]

        result_shape = list(full_probabilities.shape)
        self.assertEqual(result_shape, expectedShape)
        torch.testing.assert_close(skip_mask, full_probabilities.sum(dim=-1))

    def test__compute_masked_probabilities_input_batch_matrix_and_skip_mask(self):
        sampler = SamplerBase(cfg=self.cfg)
        batch_size = 3
        input_shape = [
            batch_size,
            self.cfg.router_model_config.input_dim,
        ]
        test_input = torch.randn(*input_shape)
        skip_mask = torch.zeros(batch_size)
        skip_mask[0] = 1
        skip_mask[1:] = 0
        skip_mask_reshaped = skip_mask.unsqueeze(-1)

        full_probabilities, _ = sampler._SamplerBase__compute_masked_probabilities(
            test_input, skip_mask_reshaped
        )

        expectedShape = [
            batch_size,
            self.cfg.router_model_config.output_dim,
        ]

        result_shape = list(full_probabilities.shape)
        self.assertEqual(result_shape, expectedShape)
        torch.testing.assert_close(skip_mask, full_probabilities.sum(dim=-1))

    def test__update_mask_given_threshold(self):
        sampler = SamplerBase(cfg=self.cfg)
        sampler.threshold = 0.4

        probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
        skip_mask = torch.ones(2).reshape(-1, 1)
        updated_mask = sampler._SamplerBase__update_mask_given_threshold(
            probabilities=probs, skip_mask=skip_mask
        )

        expected_mask = skip_mask
        expected_mask[1] = 0
        torch.testing.assert_close(skip_mask, updated_mask)


class TestProbabilitySamplerSparse(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()

    def test_probability_sampling_strategy(self):
        sampler = SamplerSparse(self.cfg)

        probs = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2], [0.4, 0.1, 0.5]],
                [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]],
            ]
        )

        result_probs, result_indices = sampler._probability_sampling_strategy(probs)

        expected_probs = torch.tensor([[0.7, 0.5, 0.5], [0.5, 0.7, 0.4]])
        expected_indices = torch.tensor([[2, 1, 2], [2, 1, 1]])

        torch.testing.assert_close(result_probs, expected_probs)
        torch.testing.assert_close(result_indices, expected_indices)

    def test_compute_loss_hook(self):
        sampler = SamplerSparse(self.cfg)

        output_dim = self.cfg.router_model_config.output_dim
        logits = torch.randn(2, 3, output_dim)
        full_probs = torch.softmax(logits, dim=-1)
        probs = torch.tensor([[0.7, 0.5, 0.5], [0.5, 0.7, 0.4]])
        skip_mask = torch.tensor([[2, 1, 2], [2, 1, 1]])

        sampler._compute_loss(logits, full_probs, probs, skip_mask)
        aux = sampler.auxiliary_losses

        self.assertFalse(torch.all(aux.probability_accumulation == 0))
        self.assertFalse(torch.all(aux.gate_accumulation == 0))
        self.assertFalse(torch.all(aux.frequency_accumulation == 0))
        self.assertFalse(aux.squared_log_sum_exp_accumulation == 0)
        self.assertFalse(aux.count_accumulation == 0)


class TestProbabilitySamplerTopk(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()

    def test_probability_sampling_strategy_only_topk(self):
        overrides = SamplerConfig(top_k=2, num_topk_samples=0, threshold=0.0)
        sampler = SamplerTopk(self.cfg, overrides)

        probs = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [0.3, 0.5, 0.2],
                [0.4, 0.1, 0.5],
                [0.2, 0.3, 0.5],
                [0.1, 0.7, 0.2],
                [0.3, 0.4, 0.3],
            ]
        )

        expected_probs = torch.tensor(
            [
                [0.7, 0.2],
                [0.5, 0.3],
                [0.5, 0.4],
                [0.5, 0.3],
                [0.7, 0.2],
                [0.4, 0.3],
            ]
        )
        expected_indices = torch.tensor(
            [
                [2, 1],
                [1, 0],
                [2, 0],
                [2, 1],
                [1, 2],
                [1, 0],
            ]
        )

        result_probs, result_indices = sampler._probability_sampling_strategy(probs)
        torch.testing.assert_close(result_probs, expected_probs)
        torch.testing.assert_close(result_indices, expected_indices)

    def test__probability_sampling_strategy_with_random_samples(self):
        top_k = 5
        num_topk_samples = 4
        overrides = SamplerConfig(
            top_k=top_k,
            num_topk_samples=num_topk_samples,
        )

        sampler = SamplerTopk(self.cfg, overrides)
        sampler.set_is_training_flag(True)
        probs = torch.tensor(
            [
                [0.05, 0.11, 0.08, 0.29, 0.34, 0.09],
                [0.45, 0.08, 0.15, 0.07, 0.19, 0.04],
                [0.08, 0.39, 0.16, 0.01, 0.21, 0.12],
                [0.16, 0.09, 0.41, 0.17, 0.09, 0.04],
                [0.15, 0.15, 0.06, 0.04, 0.07, 0.50],
                [0.09, 0.08, 0.03, 0.26, 0.09, 0.43],
            ]
        )

        expected_shape = [6, top_k]
        expected_top_deterministic, _ = torch.topk(probs, num_topk_samples)
        result_probs, _ = sampler._probability_sampling_strategy(probs)
        result_top_deterministic, _ = torch.topk(probs, num_topk_samples)
        result_shape = torch.tensor(result_probs.shape).numpy().tolist()

        torch.testing.assert_close(expected_top_deterministic, result_top_deterministic)
        self.assertEqual(expected_shape, result_shape)

    def test_sample_topk_probabilities_training_false(self):
        top_k = 4
        overrides = SamplerConfig(
            top_k=top_k,
            num_topk_samples=2,
            threshold=0.0,
        )

        sampler = SamplerTopk(self.cfg, overrides)
        sampler.set_is_training_flag(False)

        probs = torch.tensor(
            [
                [0.05, 0.11, 0.08, 0.29, 0.34, 0.09],
                [0.45, 0.08, 0.15, 0.07, 0.19, 0.04],
                [0.08, 0.39, 0.16, 0.01, 0.21, 0.12],
                [0.16, 0.09, 0.41, 0.17, 0.09, 0.04],
                [0.15, 0.15, 0.06, 0.04, 0.07, 0.50],
                [0.09, 0.08, 0.03, 0.26, 0.09, 0.43],
            ]
        )

        expected_shape = torch.tensor([6, 4])
        expected_top_k, _ = torch.topk(probs, top_k)
        result_probs, _ = sampler._probability_sampling_strategy(probs)
        result_shape = torch.tensor(result_probs.shape)

        torch.testing.assert_close(expected_top_k, result_probs)
        self.assertEqual(expected_shape.numpy().tolist(), result_shape.numpy().tolist())


class TestProbabilitySamplerFull(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig()

    def test_probability_sampling_strategy(self):
        overrides = SamplerConfig(threshold=0.0)
        sampler = SamplerFull(self.cfg, overrides)

        probs = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2], [0.4, 0.1, 0.5]],
                [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.3, 0.4, 0.3]],
            ]
        )

        result_probs, result_indices = sampler._probability_sampling_strategy(probs)

        torch.testing.assert_close(result_probs, probs)
        self.assertIsNone(result_indices)

    def test_sample_full_probabilities_with_threshold(self):
        overrides = SamplerConfig(
            dynamic_topk_threshold=0.3,
            normalize_probabilities_flag=False,
        )
        sampler = SamplerFull(self.cfg, overrides)

        probs = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.5, 0.2],
                    [0.4, 0.1, 0.5],
                ],
                [
                    [0.2, 0.3, 0.5],
                    [0.1, 0.7, 0.2],
                    [0.3, 0.4, 0.3],
                ],
            ]
        )

        expected_masked_probs = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.7],
                    [0.3, 0.5, 0.0],
                    [0.4, 0.0, 0.5],
                ],
                [
                    [0.0, 0.3, 0.5],
                    [0.0, 0.7, 0.0],
                    [0.3, 0.4, 0.3],
                ],
            ]
        )

        result_probs = sampler._SamplerFull__apply_dynamic_topk_threshold_mask(probs)
        torch.testing.assert_close(result_probs, expected_masked_probs)

    def test_sample_full_probabilities_no_threshold(self):
        overrides = SamplerConfig(
            dynamic_topk_threshold=0.0,
        )

        sampler = SamplerFull(self.cfg, overrides)

        probs = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.5, 0.2],
                    [0.4, 0.1, 0.5],
                ],
                [
                    [0.2, 0.3, 0.5],
                    [0.1, 0.7, 0.2],
                    [0.3, 0.4, 0.3],
                ],
            ]
        )

        result_probs = sampler._SamplerFull__apply_dynamic_topk_threshold_mask(probs)

        torch.testing.assert_close(result_probs, probs)


#
#     def test_mask_probs_by_threshold(self):
#         sampler = ProbabilitySamplerFull(self.cfg, threshold=0.3)
#
#         probs = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
#
#         result_probs = sampler._ProbabilitySamplerFull__mask_probs_by_threshold(probs)
#
#         expected = torch.tensor([[0.0, 0.0, 0.7], [0.0, 0.3, 0.5]])
#
#         torch.testing.assert_close(result_probs, expected)
#
#
# class TestSamplerModel(unittest.TestCase):
#     def setUp(self):
#         self.cfg = ModelConfig()
#         self.cfg.router_model_config.output_dim = 8
#
#         self.input_batch = torch.randn(
#             self.cfg.batch_size,
#             self.cfg.router_model_config.input_dim,
#         )
#
#         self.skip_mask = torch.ones(
#             self.cfg.batch_size,
#             self.cfg.router_model_config.output_dim,
#         )
#
#     def test_init_no_config(self):
#         router_model = RouterModel(self.cfg)
#         model = SamplerModel(
#             top_k=3,
#             threshold=0.0,
#             num_topk_samples=1,
#             noisy_topk_flag=True,
#             custom_softmax_flag=True,
#             router_model=router_model,
#         )
#
#         self.assertTrue(isinstance(model.sampler_model, ProbabilitySamplerTopk))
#
#     def test_init_sparse(self):
#         # Test with top_k=1 -> should create ProbabilitySamplerSparse
#         model = SamplerModel(self.cfg, top_k=1, num_topk_samples=0)
#         self.assertTrue(isinstance(model.sampler_model, ProbabilitySamplerSparse))
#
#     def test_init_full(self):
#         # Test with top_k=num_expserts -> should create ProbabilitySamplerFull
#         model = SamplerModel(self.cfg, top_k=8, num_topk_samples=0)
#         self.assertTrue(isinstance(model.sampler_model, ProbabilitySamplerFull))
#
#     def test_init_topk(self):
#         # Test with 1 < top_k < num_expserts -> should create ProbabilitySamplerTopk
#         model = SamplerModel(self.cfg, top_k=4, num_topk_samples=0)
#         self.assertTrue(isinstance(model.sampler_model, ProbabilitySamplerTopk))
#
#     def test_sample_probs_and_indexes_sparse(self):
#         sampler_model = SamplerModel(self.cfg, top_k=1, num_topk_samples=0)
#
#         sampler_model(self.input_batch, self.skip_mask)
#
#     def test_sample_probs_and_indexes_top_k(self):
#         sampler_model = SamplerModel(self.cfg, top_k=3, num_topk_samples=1)
#
#         sampler_model(self.input_batch, self.skip_mask)
#
#     def test_sample_probs_and_indexes_full(self):
#         sampler_model = SamplerModel(
#             self.cfg, top_k=self.cfg.router_model_config.output_dim
#         )
#
#         sampler_model(self.input_batch, self.skip_mask)
