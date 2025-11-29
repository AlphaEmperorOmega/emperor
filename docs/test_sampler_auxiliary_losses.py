import torch
import unittest
from math import prod

from Emperor.sampler.utils.config import SamplerConfigs
from Emperor.sampler.utils.losses import (
    CoefficientOfVariationLoss,
    MutualInformationLoss,
    SamplerAuxiliaryLosses,
    SwitchLoss,
    ZeroCentredLoss,
)
from Emperor.sampler.utils.samplers import SamplerConfig


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

    def test__init(self):
        cfg = SamplerConfigs.sampler_preset()
        m = SamplerAuxiliaryLosses(cfg)

        self.assertIsInstance(
            m.coefficient_of_variation_loss, CoefficientOfVariationLoss
        )
        self.assertIsInstance(m.switch_loss, SwitchLoss)
        self.assertIsInstance(m.zero_centred_loss, ZeroCentredLoss)
        self.assertIsInstance(m.mutual_information_loss, MutualInformationLoss)

    def test_update_accumulated_statistics(self):
        cfg = SamplerConfigs.sampler_preset(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerAuxiliaryLosses(cfg)

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
        cfg = SamplerConfigs.sampler_preset(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerAuxiliaryLosses(cfg)

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

    def test_compute_total_loss(self):
        cfg = SamplerConfigs.sampler_preset(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerAuxiliaryLosses(cfg)

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
        cfg = SamplerConfigs.sampler_preset(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        m = SamplerAuxiliaryLosses(cfg)

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
