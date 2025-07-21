import copy
import unittest
import torch
from math import prod
from Emperor.layers.utils.losses import (
    CoefficientOfVariationLoss,
    MutualInformationLoss,
    SwitchLoss,
    ZeroCentredLoss,
)
from Emperor.layers.utils.samplers import (
    SamplerAuxiliaryLosses,
    SamplerBase,
    SamplerConfig,
    SamplerSparse,
    SamplerTopk,
    SamplerFull,
    SamplerModel,
)


class TestSamplerAuxiliaryLosses(unittest.TestCase):
    def setUp(self):
        self.cfg = SamplerConfig(
            top_k=3,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=5,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )

    def test__init_with_cfg(self):
        c = copy.deepcopy(self.cfg)
        m = SamplerAuxiliaryLosses(c)

        self.assertIsInstance(
            m.coefficient_of_variation_loss, CoefficientOfVariationLoss
        )
        self.assertIsInstance(m.switch_loss, SwitchLoss)
        self.assertIsInstance(m.zero_centred_loss, ZeroCentredLoss)
        self.assertIsInstance(m.mutual_information_loss, MutualInformationLoss)

    def test__update_accumulated_statistics(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerAuxiliaryLosses(c, overrides)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()
        probabilities = torch.arange(prod(shape)).reshape(shape)
        gates = torch.arange(prod(shape)).reshape(shape)
        skip_masks = (torch.randn(*shape) > 0).float()

        m.update_accumulated_statistics(logits, probabilities, gates, skip_masks)

        self.assertIsNotNone(m.coefficient_of_variation_loss.gates_accumulation)
        self.assertIsNotNone(m.switch_loss.probability_accumulation)
        self.assertIsNotNone(m.switch_loss.frequency_accumulation)
        self.assertIsNotNone(m.zero_centred_loss.squared_log_sum_exp_accumulation)
        self.assertIsNotNone(m.zero_centred_loss.count_accumulation)
        self.assertTrue(len(m.mutual_information_loss.log_probabilities) > 0)
        self.assertTrue(len(m.mutual_information_loss.probabilities) > 0)
        self.assertTrue(len(m.mutual_information_loss.skip_masks) > 0)

    def test__reset_all_accumulations(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerAuxiliaryLosses(c, overrides)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()
        probabilities = torch.arange(prod(shape)).reshape(shape)
        gates = torch.arange(prod(shape)).reshape(shape)
        skip_masks = (torch.randn(*shape) > 0).float()

        m.update_accumulated_statistics(logits, probabilities, gates, skip_masks)
        m._SamplerAuxiliaryLosses__reset_all_accumulations()

        self.assertIsNone(m.coefficient_of_variation_loss.gates_accumulation)
        self.assertIsNone(m.switch_loss.probability_accumulation)
        self.assertIsNone(m.switch_loss.frequency_accumulation)
        self.assertIsNone(m.zero_centred_loss.squared_log_sum_exp_accumulation)
        self.assertIsNone(m.zero_centred_loss.count_accumulation)
        self.assertTrue(len(m.mutual_information_loss.log_probabilities) == 0)
        self.assertTrue(len(m.mutual_information_loss.probabilities) == 0)
        self.assertTrue(len(m.mutual_information_loss.skip_masks) == 0)

    def test__compute_total_loss(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            coefficient_of_variation_loss_weight=1.0,
            switch_loss_weight=1.0,
            zero_centred_loss_weight=1.0,
            mutual_information_loss_weight=1.0,
        )
        m = SamplerAuxiliaryLosses(c, overrides)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        gates = torch.arange(prod(shape)).reshape(shape).float()
        skip_masks = torch.abs(torch.randn(*shape))

        m.update_accumulated_statistics(logits, probabilities, gates, skip_masks)
        output = m._SamplerAuxiliaryLosses__compute_total_loss()

        self.assertIsInstance(output, torch.Tensor)
        self.assertIsNot(output, 0.0)

    def test__get_auxiliary_loss_and_clear(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerAuxiliaryLosses(c, overrides)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        gates = torch.arange(prod(shape)).reshape(shape).float()
        skip_masks = torch.abs(torch.randn(*shape))

        m.update_accumulated_statistics(logits, probabilities, gates, skip_masks)
        output = m.get_auxiliary_loss_and_clear()

        self.assertIsInstance(output, torch.Tensor)
        self.assertIsNot(output, 0.0)
        self.assertIsNone(m.coefficient_of_variation_loss.gates_accumulation)
        self.assertIsNone(m.switch_loss.probability_accumulation)
        self.assertIsNone(m.switch_loss.frequency_accumulation)
        self.assertIsNone(m.zero_centred_loss.squared_log_sum_exp_accumulation)
        self.assertIsNone(m.zero_centred_loss.count_accumulation)
        self.assertTrue(len(m.mutual_information_loss.log_probabilities) == 0)
        self.assertTrue(len(m.mutual_information_loss.probabilities) == 0)
        self.assertTrue(len(m.mutual_information_loss.skip_masks) == 0)


class TestProbabilitySampler(unittest.TestCase):
    def setUp(self):
        self.cfg = SamplerConfig(
            top_k=3,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=5,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )

    def test__init_with_cfg(self):
        sampler = SamplerBase(cfg=self.cfg)

        self.assertEqual(sampler.top_k, self.cfg.top_k)
        self.assertEqual(sampler.threshold, self.cfg.threshold)
        self.assertEqual(sampler.num_topk_samples, self.cfg.num_topk_samples)
        self.assertEqual(sampler.noisy_topk_flag, self.cfg.noisy_topk_flag)
        self.assertEqual(sampler.num_experts, self.cfg.num_experts)

    def test__init_with_custom_config(self):
        config = SamplerConfig(
            top_k=4,
            threshold=0.1,
            filter_above_threshold=True,
            noisy_topk_flag=True,
            num_topk_samples=2,
            normalize_probabilities_flag=True,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )

        sampler = SamplerBase(config)

        self.assertEqual(sampler.top_k, config.top_k)
        self.assertEqual(sampler.threshold, config.threshold)
        self.assertEqual(sampler.num_topk_samples, config.num_topk_samples)
        self.assertEqual(
            sampler.normalize_probabilities_flag,
            config.normalize_probabilities_flag,
        )
        self.assertEqual(sampler.noisy_topk_flag, config.noisy_topk_flag)
        self.assertEqual(sampler.num_experts, config.num_experts)

    def test__normalize_probabilities__normalize_probabilities_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            normalize_probabilities_flag=False,
        )
        m = SamplerBase(c, overrides)
        probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
        result = m._normalize_probabilities(probs)

        self.assertTrue(torch.allclose(probs, result))

    def test__normalize_probabilities__normalize_probabilities_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            normalize_probabilities_flag=True,
        )
        m = SamplerBase(c, overrides)
        probs = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
        result = m._normalize_probabilities(probs)
        distribution_check = result.sum(dim=-1)
        expected_distribution_check = torch.tensor([1.0, 1.0])

        self.assertTrue(torch.allclose(distribution_check, expected_distribution_check))

    def test__probability_sampling_strategy(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            normalize_probabilities_flag=True,
        )
        m = SamplerBase(c, overrides)
        probabilities = torch.ones(2, 4)

        with self.assertRaises(NotImplementedError) as context:
            m._sample_probabilities_and_indices(probabilities)

    def test__update_mask_given_threshold__threshold__zero(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.0,
        )
        m = SamplerBase(c, overrides)
        skip_mask = torch.ones(2).reshape(-1, 1)

        probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
        output = m._SamplerBase__update_mask_given_threshold(
            probabilities,
            skip_mask,
        )
        self.assertTrue(torch.allclose(skip_mask, output))

    def test__update_mask_given_threshold__threshold__positive(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.4,
        )
        m = SamplerBase(c, overrides)
        batch_size = 2
        skip_mask = torch.ones(batch_size).reshape(-1, 1)

        probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
        output = m._SamplerBase__update_mask_given_threshold(
            probabilities,
            skip_mask,
        )
        expected_mask = torch.ones(batch_size).reshape(-1, 1)
        expected_mask[1] = 0.0

        self.assertTrue(torch.allclose(output, expected_mask))

    def test__update_mask_given_threshold__filter_above_threshold__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.4,
            filter_above_threshold=True,
        )
        m = SamplerBase(c, overrides)
        batch_size = 2
        skip_mask = torch.ones(batch_size).reshape(-1, 1)

        probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.2, 0.3]])
        output = m._SamplerBase__update_mask_given_threshold(
            probabilities,
            skip_mask,
        )
        expected_mask = torch.ones(batch_size).reshape(-1, 1)
        expected_mask[0] = 0.0

        self.assertTrue(torch.allclose(output, expected_mask))

    def test__apply_skip_mask__threshold__zero(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.0,
            filter_above_threshold=True,
        )
        m = SamplerBase(c, overrides)
        batch_size = 2
        sequence_length = 3
        feature_dim = 4
        probs = torch.ones(batch_size * sequence_length, feature_dim)
        logits = torch.ones(batch_size * sequence_length, feature_dim)
        mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1
        masked_probs, router_logit_scores = m._SamplerBase__apply_skip_mask(
            probs, logits, mask
        )

        self.assertTrue(torch.allclose(masked_probs, probs))
        self.assertTrue(torch.allclose(router_logit_scores, logits))

    def test__apply_skip_mask__threshold__positive(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.4,
            filter_above_threshold=True,
        )
        m = SamplerBase(c, overrides)
        batch_size = 2
        sequence_length = 3
        feature_dim = 4

        shape = (batch_size * sequence_length, feature_dim)
        probs = torch.arange(prod(shape)).reshape(shape).float()
        logits = torch.arange(prod(shape)).reshape(shape).float()
        mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)

        unmasked_token = 0
        mask[unmasked_token, :] = 1
        masked_probs, router_logit_scores = m._SamplerBase__apply_skip_mask(
            probs, logits, mask
        )

        self.assertTrue(torch.sum(masked_probs[unmasked_token, :]).item() > 0)
        self.assertTrue(torch.sum(masked_probs[1:, :]).item() == 0)
        self.assertTrue(torch.sum(router_logit_scores[unmasked_token, :]).item() > 0)
        self.assertTrue(torch.sum(router_logit_scores[1:, :]).item() == 0)

    def test__add_noise_to_logits_flag__False(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=False,
        )
        m = SamplerBase(c, overrides)

        batch_size = 2

        shape = (batch_size, m.cfg.num_experts)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        result = m._SamplerBase__add_noise_to_logits(logits)

        self.assertTrue(torch.allclose(logits, result))

    def test__add_noise_to_logits_flag_True(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=True,
        )
        m = SamplerBase(c, overrides)
        m.training = True

        batch_size = 2
        shape = (batch_size, m.num_experts * 2)
        logits = torch.arange(prod(shape)).reshape(shape).float()
        result = m._SamplerBase__add_noise_to_logits(logits)

        self.assertNotEqual(logits.mean(), result.mean())
        self.assertListEqual(list(result.shape), [batch_size, m.num_experts])

    def test__compute_masked_probabilities__router_logit_scores__only(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=False,
            threshold=0.0,
        )
        m = SamplerBase(c, overrides)

        batch_size = 3
        shape = (batch_size, m.cfg.num_experts)
        router_logit_scores = torch.arange(prod(shape)).reshape(shape).float()

        masked_probabilities, router_logit_scores = (
            m._SamplerBase__compute_masked_probabilities(router_logit_scores)
        )

        self.assertListEqual(
            list(masked_probabilities.shape), list(router_logit_scores.shape)
        )
        self.assertTrue(
            torch.allclose(
                torch.sum(masked_probabilities, dim=-1),
                torch.ones(batch_size).float(),
            )
        )

    def test__compute_masked_probabilities__noisy_topk_flag__True(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=True,
            threshold=0.0,
        )
        m = SamplerBase(c, overrides)

        batch_size = 3
        shape = (batch_size, m.cfg.num_experts * 2)
        router_logit_scores = torch.arange(prod(shape)).reshape(shape).float()

        masked_probabilities, router_logit_scores = (
            m._SamplerBase__compute_masked_probabilities(router_logit_scores)
        )

        self.assertListEqual(
            list(masked_probabilities.shape), [batch_size, m.cfg.num_experts]
        )
        self.assertTrue(
            torch.allclose(
                torch.sum(masked_probabilities, dim=-1),
                torch.ones(batch_size).float(),
            )
        )

    def test__compute_masked_probabilities__noisy_topk_flag__True__and__threshold__positive(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=True,
            threshold=0.2,
        )
        m = SamplerBase(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.cfg.num_experts * 2)
        router_logit_scores = torch.arange(prod(shape)).reshape(shape).float()
        mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        masked_probabilities, router_logit_scores = (
            m._SamplerBase__compute_masked_probabilities(router_logit_scores, mask)
        )
        self.assertListEqual(
            list(masked_probabilities.shape),
            [batch_size * sequence_length, m.cfg.num_experts],
        )
        self.assertTrue(torch.sum(masked_probabilities[0, :]).item() > 0)
        self.assertTrue(torch.sum(masked_probabilities[1:, :]).item() == 0)
        self.assertTrue(torch.sum(router_logit_scores[0, :]).item() > 0)
        self.assertTrue(torch.sum(router_logit_scores[1:, :]).item() == 0)
        self.assertTrue(
            torch.allclose(
                torch.sum(masked_probabilities[0, :], dim=-1),
                torch.tensor(1.0).float(),
            )
        )


class TestSamplerSparse(unittest.TestCase):
    def setUp(self):
        self.cfg = SamplerConfig(
            top_k=3,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=5,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )

    def test__sample_probabilities_and_indices(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=True,
            threshold=0.2,
        )
        m = SamplerSparse(c, overrides)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)

        probability, indices = m._sample_probabilities_and_indices(probabilities)

        self.assertListEqual(list(probability.shape), [batch_size])
        self.assertListEqual(list(indices.shape), [batch_size])

    def test__get_probabilities_and_indices(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerSparse(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss > 0)
        self.assertListEqual(list(probabilities.shape), [batch_size * sequence_length])
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length]
        )

    def test__get_probabilities_and_indices__noisy_topk_flag__True(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=True,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerSparse(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss > 0)
        self.assertListEqual(list(probabilities.shape), [batch_size * sequence_length])
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length]
        )

    def test__get_probabilities_and_indices__noisy_topk_flag__True__threshold__positive(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=True,
            threshold=0.2,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerSparse(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss > 0)
        self.assertListEqual(list(probabilities.shape), [batch_size * sequence_length])
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length]
        )

    def test__get_probabilities_and_indices__all_flags__True(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            noisy_topk_flag=True,
            threshold=0.8,
            filter_above_threshold=True,
            normalize_probabilities_flag=False,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerSparse(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(loss > 0)
        self.assertListEqual(list(probabilities.shape), [batch_size * sequence_length])
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length]
        )

    def test__prepare_loss_gates__skip_maks__None(self):
        c = copy.deepcopy(self.cfg)
        m = SamplerSparse(c)

        skip_maks = m._SamplerSparse__prepare_loss_skip_mask()

        self.assertIsNone(skip_maks)

    def test__prepare_loss_gates__skip_maks__sparse(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=1,
        )
        m = SamplerSparse(c, overrides)

        batch_size = 3

        shape = (batch_size, m.top_k)
        sampled_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        indices = torch.randint(0, m.num_experts, (batch_size, m.top_k))

        gates = m._SamplerSparse__prepare_loss_gates(sampled_probabilities, indices)

        self.assertListEqual(list(gates.shape), [batch_size, m.num_experts])
        self.assertTrue(
            torch.allclose(
                torch.sum(gates, dim=-1),
                torch.ones(batch_size).float(),
            )
        )

    def test__compute_loss(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=1,
            noisy_topk_flag=True,
            threshold=0.8,
            filter_above_threshold=True,
            normalize_probabilities_flag=False,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerSparse(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts)

        logits = torch.randn(*shape)
        full_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        sampled_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        indices = torch.randint(0, m.num_experts, (batch_size, m.top_k))
        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        total_loss = m._compute_loss(
            logits,
            full_probabilities,
            sampled_probabilities,
            indices,
            mask,
        )

        self.assertTrue(total_loss > 0)
        self.assertIsInstance(total_loss, torch.Tensor)


class TestSamplerTopk(unittest.TestCase):
    def setUp(self):
        self.cfg = SamplerConfig(
            top_k=3,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=5,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )

    def test__sample_probabilities_and_indices(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=3,
            noisy_topk_flag=True,
            threshold=0.2,
        )
        m = SamplerTopk(c, overrides)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)

        probability, indices = m._sample_probabilities_and_indices(probabilities)

        self.assertListEqual(list(probability.shape), [batch_size, m.top_k])
        self.assertListEqual(list(indices.shape), [batch_size, m.top_k])

    def test__get_probabilities_and_indices(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerTopk(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss > 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.top_k]
        )
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length, m.top_k]
        )

    def test__get_probabilities_and_indices__noisy_topk_flag__True(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=3,
            noisy_topk_flag=True,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerTopk(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss > 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.top_k]
        )
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length, m.top_k]
        )

    def test__get_probabilities_and_indices__noisy_topk_flag__True__threshold__positive(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=3,
            noisy_topk_flag=True,
            threshold=0.2,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerTopk(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss > 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.top_k]
        )
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length, m.top_k]
        )

    def test__get_probabilities_and_indices__all_flags__True(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=3,
            noisy_topk_flag=True,
            threshold=0.8,
            filter_above_threshold=True,
            normalize_probabilities_flag=True,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerTopk(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(loss > 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.top_k]
        )
        self.assertListEqual(
            list(selected_indices.shape), [batch_size * sequence_length, m.top_k]
        )

    def test__prepare_loss_gates__skip_maks__None(self):
        c = copy.deepcopy(self.cfg)
        m = SamplerTopk(c)

        skip_maks = m._SamplerTopk__prepare_loss_skip_mask()

        self.assertIsNone(skip_maks)

    def test__prepare_loss_gates__skip_maks__sparse(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=1,
        )
        m = SamplerTopk(c, overrides)

        batch_size = 3

        shape = (batch_size, m.top_k)
        sampled_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        indices = torch.randint(0, m.num_experts, (batch_size, m.top_k))

        gates = m._SamplerTopk__prepare_loss_gates(sampled_probabilities, indices)

        self.assertListEqual(list(gates.shape), [batch_size, m.num_experts])
        self.assertTrue(
            torch.allclose(
                torch.sum(gates, dim=-1),
                torch.ones(batch_size).float(),
            )
        )

    def test__compute_loss(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=3,
            noisy_topk_flag=True,
            threshold=0.8,
            filter_above_threshold=True,
            normalize_probabilities_flag=True,
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerTopk(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts)

        logits = torch.randn(*shape)
        full_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        sampled_probabilities = torch.softmax(torch.randn(*shape), dim=-1)
        indices = torch.randint(0, m.num_experts, (batch_size, m.top_k))
        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        total_loss = m._compute_loss(
            logits,
            full_probabilities,
            sampled_probabilities,
            indices,
            mask,
        )

        self.assertTrue(total_loss > 0)
        self.assertIsInstance(total_loss, torch.Tensor)


class TestSamplerFull(unittest.TestCase):
    def setUp(self):
        self.cfg = SamplerConfig(
            top_k=5,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=5,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )

    def test__sample_probabilities_and_indices(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=5,
            noisy_topk_flag=True,
            threshold=0.2,
        )
        m = SamplerFull(c, overrides)

        batch_size = 3
        shape = (batch_size, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)

        probability, _ = m._sample_probabilities_and_indices(probabilities)

        self.assertListEqual(list(probability.shape), [batch_size, m.top_k])

    def test__get_probabilities_and_indices(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerFull(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss == 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.num_experts]
        )
        self.assertIsNone(selected_indices)

    def test__get_probabilities_and_indices__noisy_topk_flag__True(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=5,
            noisy_topk_flag=True,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerFull(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss == 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.num_experts]
        )
        self.assertIsNone(selected_indices)

    def test__get_probabilities_and_indices__noisy_topk_flag__True__threshold__positive(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=5,
            noisy_topk_flag=True,
            threshold=0.2,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerFull(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.zeros(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(torch.allclose(skip_mask, mask))
        self.assertTrue(loss == 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.num_experts]
        )
        self.assertIsNone(selected_indices)

    def test__get_probabilities_and_indices__all_flags__True(
        self,
    ):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=5,
            noisy_topk_flag=True,
            threshold=0.8,
            filter_above_threshold=True,
            normalize_probabilities_flag=True,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )
        m = SamplerFull(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        logits = torch.softmax(torch.randn(*shape), dim=-1)

        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, selected_indices, skip_mask, loss = (
            m.get_probabilities_and_indices(logits, mask)
        )

        self.assertTrue(loss == 0)
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, m.num_experts]
        )
        self.assertIsNone(selected_indices)

    def test__apply_dynamic_topk_threshold_mask__threshold__zero(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.0,
        )
        m = SamplerFull(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts * 2)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)

        masked_probabilities = m._SamplerFull__apply_dynamic_topk_threshold_mask(
            probabilities
        )

        self.assertTrue(torch.allclose(probabilities, masked_probabilities))

    def test__apply_dynamic_topk_threshold_mask__threshold__positive(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.1,
            normalize_probabilities_flag=True,
        )
        m = SamplerFull(c, overrides)

        batch_size = 3
        sequence_length = 4
        shape = (batch_size * sequence_length, m.num_experts)
        probabilities = torch.softmax(torch.randn(*shape), dim=-1)

        masked_probabilities = m._SamplerFull__apply_dynamic_topk_threshold_mask(
            probabilities
        )

        self.assertListEqual(
            list(masked_probabilities.shape),
            [batch_size * sequence_length, c.num_experts],
        )


class TestSamplerModel(unittest.TestCase):
    def setUp(self):
        self.cfg = SamplerConfig(
            top_k=5,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=5,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
        )

    def test__init_no_config(self):
        config = SamplerConfig(
            top_k=5,
            threshold=0.5,
            filter_above_threshold=True,
            num_topk_samples=2,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.5,
            switch_loss_weight=0.5,
            zero_centred_loss_weight=0.5,
            mutual_information_loss_weight=0.5,
        )
        model = SamplerModel(config)
        self.assertTrue(isinstance(model.sampler_model, SamplerTopk))

    def test__init_sparse(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=1,
            num_experts=5,
        )
        m = SamplerModel(c, overrides)
        self.assertTrue(isinstance(m.sampler_model, SamplerSparse))

    def test__init_full(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=5,
            num_experts=5,
        )
        m = SamplerModel(c, overrides)
        self.assertTrue(isinstance(m.sampler_model, SamplerFull))

    def test__init_topk(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            top_k=3,
            num_experts=5,
        )
        m = SamplerModel(c, overrides)
        self.assertTrue(isinstance(m.sampler_model, SamplerTopk))

    def test__sample_probs_and_indexes_sparse__logits_only(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.0,
            mutual_information_loss_weight=0.0,
            #############################
            top_k=1,
            num_topk_samples=0,
            filter_above_threshold=True,
            normalize_probabilities_flag=False,
            noisy_topk_flag=True,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.5,
            switch_loss_weight=0.5,
            zero_centred_loss_weight=0.5,
        )
        m = SamplerModel(c, overrides)

        batch_size = 3
        sequence_length = 4
        logits = torch.randn(batch_size * sequence_length, c.num_experts * 2)

        probabilities, indices, skip_mask, loss = m.sample_probabilities_and_indices(
            logits
        )

        self.assertListEqual(list(probabilities.shape), [batch_size * sequence_length])
        self.assertListEqual(list(indices.shape), [batch_size * sequence_length])
        self.assertIsNone(skip_mask)
        self.assertTrue(loss > 0.0)

    def test__sample_probs_and_indexes_sparse__logits__and__skip_mask(self):
        c = copy.deepcopy(self.cfg)

        overrides = SamplerConfig(
            threshold=0.1,
            mutual_information_loss_weight=0.1,
            #############################
            top_k=1,
            num_topk_samples=0,
            filter_above_threshold=True,
            normalize_probabilities_flag=False,
            noisy_topk_flag=True,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.5,
            switch_loss_weight=0.5,
            zero_centred_loss_weight=0.5,
        )
        m = SamplerModel(c, overrides)

        batch_size = 3
        sequence_length = 4
        logits = torch.randn(batch_size * sequence_length, c.num_experts * 2)
        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, indices, skip_mask, loss = m.sample_probabilities_and_indices(
            logits, mask
        )

        self.assertListEqual(list(probabilities.shape), [batch_size * sequence_length])
        self.assertListEqual(list(indices.shape), [batch_size * sequence_length])
        self.assertListEqual(list(skip_mask.shape), [batch_size * sequence_length, 1])
        self.assertTrue(loss > 0.0)

    def test__sample_probs_and_indexes_topk__logits_only(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.0,
            mutual_information_loss_weight=0.0,
            #############################
            top_k=3,
            num_topk_samples=1,
            filter_above_threshold=True,
            normalize_probabilities_flag=True,
            noisy_topk_flag=True,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.5,
            switch_loss_weight=0.5,
            zero_centred_loss_weight=0.5,
        )
        m = SamplerModel(c, overrides)

        batch_size = 3
        sequence_length = 4
        logits = torch.randn(batch_size * sequence_length, c.num_experts * 2)

        probabilities, indices, skip_mask, loss = m.sample_probabilities_and_indices(
            logits
        )

        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, overrides.top_k]
        )
        self.assertListEqual(
            list(indices.shape), [batch_size * sequence_length, overrides.top_k]
        )
        self.assertIsNone(skip_mask)
        self.assertTrue(loss > 0.0)

    def test__sample_probs_and_indexes_topk__logits__and__skip_mask(self):
        c = copy.deepcopy(self.cfg)

        overrides = SamplerConfig(
            threshold=0.1,
            mutual_information_loss_weight=0.1,
            #############################
            top_k=3,
            num_topk_samples=1,
            filter_above_threshold=True,
            normalize_probabilities_flag=True,
            noisy_topk_flag=True,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.5,
            switch_loss_weight=0.5,
            zero_centred_loss_weight=0.5,
        )
        m = SamplerModel(c, overrides)

        batch_size = 3
        sequence_length = 4
        logits = torch.randn(batch_size * sequence_length, c.num_experts * 2)
        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, indices, skip_mask, loss = m.sample_probabilities_and_indices(
            logits, mask
        )

        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, overrides.top_k]
        )
        self.assertListEqual(
            list(indices.shape), [batch_size * sequence_length, overrides.top_k]
        )
        self.assertListEqual(list(skip_mask.shape), [batch_size * sequence_length, 1])
        self.assertTrue(loss > 0.0)

    def test__sample_probs_and_indexes_full_mixture__logits_only(self):
        c = copy.deepcopy(self.cfg)
        overrides = SamplerConfig(
            threshold=0.0,
            mutual_information_loss_weight=0.0,
            #############################
            top_k=10,
            num_topk_samples=0,
            filter_above_threshold=True,
            normalize_probabilities_flag=True,
            noisy_topk_flag=True,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
        )
        m = SamplerModel(c, overrides)

        batch_size = 3
        sequence_length = 4
        logits = torch.randn(batch_size * sequence_length, c.num_experts * 2)

        probabilities, indices, skip_mask, loss = m.sample_probabilities_and_indices(
            logits
        )

        self.assertFalse((probabilities == 0).any())
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, overrides.top_k]
        )
        self.assertIsNone(indices)
        self.assertIsNone(skip_mask)
        self.assertTrue(loss == 0.0)

    def test__sample_probs_and_indexes_full_mixture__logits__and__skip_mask(self):
        c = copy.deepcopy(self.cfg)

        overrides = SamplerConfig(
            threshold=0.1,
            mutual_information_loss_weight=0.0,
            #############################
            top_k=10,
            num_topk_samples=0,
            filter_above_threshold=True,
            normalize_probabilities_flag=True,
            noisy_topk_flag=True,
            num_experts=10,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
        )
        m = SamplerModel(c, overrides)

        batch_size = 3
        sequence_length = 4
        logits = torch.randn(batch_size * sequence_length, c.num_experts * 2)
        mask = torch.ones(batch_size, sequence_length).reshape(-1, 1)
        unmasked_token = 0
        mask[unmasked_token, :] = 1

        probabilities, indices, skip_mask, loss = m.sample_probabilities_and_indices(
            logits, mask
        )

        self.assertTrue((probabilities == 0).any())
        self.assertListEqual(
            list(probabilities.shape), [batch_size * sequence_length, overrides.top_k]
        )
        self.assertIsNone(indices)
        self.assertListEqual(list(skip_mask.shape), [batch_size * sequence_length, 1])
        self.assertTrue(loss == 0.0)
