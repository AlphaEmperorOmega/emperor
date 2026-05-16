import torch
import unittest

from emperor.sampler.utils.config import SamplerConfig
from emperor.sampler.utils.losses import (
    CoefficientOfVariationLoss,
    MutualInformationLoss,
    SamplerAuxiliaryLosses,
    SwitchLoss,
    ZeroCentredLoss,
)


class TestSamplerAuxiliaryLosses(unittest.TestCase):
    def preset(
        self,
        num_experts: int = 5,
        coefficient_of_variation_loss_weight: float = 0.0,
        switch_loss_weight: float = 0.0,
        zero_centred_loss_weight: float = 0.0,
        mutual_information_loss_weight: float = 0.0,
    ) -> SamplerConfig:
        return SamplerConfig(
            top_k=3,
            threshold=0.0,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
            switch_loss_weight=switch_loss_weight,
            zero_centred_loss_weight=zero_centred_loss_weight,
            mutual_information_loss_weight=mutual_information_loss_weight,
            router_config=None,
        )

    def sample_statistics(self, batch_size: int = 3, num_experts: int = 5):
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 2.0, 1.0, -1.0, -2.0],
                [1.0, 0.0, 2.0, -1.0, -2.0],
            ]
        )[:batch_size, :num_experts]
        probabilities = torch.softmax(logits, dim=-1)
        gates = torch.zeros(batch_size, num_experts)
        selected_indices = torch.arange(batch_size) % num_experts
        gates[torch.arange(batch_size), selected_indices] = 1.0
        skip_mask = torch.ones(batch_size, 1)
        return logits, probabilities, gates, skip_mask

    def assert_accumulations_clear(self, model: SamplerAuxiliaryLosses):
        self.assertIsNone(model.coefficient_of_variation_loss.gates_accumulation)
        self.assertIsNone(model.switch_loss.probability_accumulation)
        self.assertIsNone(model.switch_loss.frequency_accumulation)
        self.assertIsNone(model.zero_centred_loss.squared_log_sum_exp_accumulation)
        self.assertIsNone(model.zero_centred_loss.count_accumulation)
        self.assertEqual(len(model.mutual_information_loss.log_probabilities), 0)
        self.assertEqual(len(model.mutual_information_loss.probabilities), 0)
        self.assertEqual(len(model.mutual_information_loss.skip_masks), 0)

    def test_init_creates_all_loss_modules(self):
        cfg = self.preset()
        model = SamplerAuxiliaryLosses(cfg)

        self.assertIsInstance(
            model.coefficient_of_variation_loss, CoefficientOfVariationLoss
        )
        self.assertIsInstance(model.switch_loss, SwitchLoss)
        self.assertIsInstance(model.zero_centred_loss, ZeroCentredLoss)
        self.assertIsInstance(model.mutual_information_loss, MutualInformationLoss)
        self.assertEqual(model.num_experts, cfg.num_experts)

    def test_all_disabled_returns_zero_without_inputs(self):
        model = SamplerAuxiliaryLosses(self.preset())

        model.update_accumulated_statistics()
        total_loss = model.get_auxiliary_loss_and_clear()

        torch.testing.assert_close(total_loss, torch.tensor(0.0))
        self.assert_accumulations_clear(model)

    def test_update_accumulated_statistics_updates_enabled_losses(self):
        cfg = self.preset(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        model = SamplerAuxiliaryLosses(cfg)
        logits, probabilities, gates, skip_mask = self.sample_statistics()

        model.update_accumulated_statistics(logits, probabilities, gates, skip_mask)

        self.assertIsNotNone(
            model.coefficient_of_variation_loss.gates_accumulation
        )
        self.assertIsNotNone(model.switch_loss.probability_accumulation)
        self.assertIsNotNone(model.switch_loss.frequency_accumulation)
        self.assertIsNotNone(
            model.zero_centred_loss.squared_log_sum_exp_accumulation
        )
        self.assertIsNotNone(model.zero_centred_loss.count_accumulation)
        self.assertEqual(len(model.mutual_information_loss.log_probabilities), 1)
        self.assertEqual(len(model.mutual_information_loss.probabilities), 1)
        self.assertEqual(len(model.mutual_information_loss.skip_masks), 1)

    def test_single_enabled_loss_requires_only_its_inputs(self):
        logits, probabilities, gates, skip_mask = self.sample_statistics()
        cases = [
            (
                "coefficient_of_variation_loss_weight",
                {"gates": gates},
            ),
            (
                "switch_loss_weight",
                {"probabilities": probabilities, "gates": gates},
            ),
            (
                "zero_centred_loss_weight",
                {"logits": logits},
            ),
            (
                "mutual_information_loss_weight",
                {
                    "logits": logits,
                    "probabilities": probabilities,
                    "skip_mask": skip_mask,
                },
            ),
        ]

        for field_name, kwargs in cases:
            with self.subTest(field_name=field_name):
                cfg = self.preset(**{field_name: 0.1})
                model = SamplerAuxiliaryLosses(cfg)

                model.update_accumulated_statistics(**kwargs)
                total_loss = model.get_auxiliary_loss_and_clear()

                self.assertIsInstance(total_loss, torch.Tensor)
                self.assert_accumulations_clear(model)

    def test_enabled_loss_raises_when_required_input_is_missing(self):
        logits, probabilities, gates, skip_mask = self.sample_statistics()
        cases = [
            (
                "coefficient_of_variation_loss_weight",
                {"gates": None},
            ),
            (
                "switch_loss_weight",
                {"probabilities": None, "gates": gates},
            ),
            (
                "switch_loss_weight",
                {"probabilities": probabilities, "gates": None},
            ),
            (
                "zero_centred_loss_weight",
                {"logits": None},
            ),
            (
                "mutual_information_loss_weight",
                {
                    "logits": None,
                    "probabilities": probabilities,
                    "skip_mask": skip_mask,
                },
            ),
            (
                "mutual_information_loss_weight",
                {
                    "logits": logits,
                    "probabilities": None,
                    "skip_mask": skip_mask,
                },
            ),
            (
                "mutual_information_loss_weight",
                {
                    "logits": logits,
                    "probabilities": probabilities,
                    "skip_mask": None,
                },
            ),
        ]

        for field_name, kwargs in cases:
            with self.subTest(field_name=field_name, kwargs=kwargs):
                cfg = self.preset(**{field_name: 0.1})
                model = SamplerAuxiliaryLosses(cfg)

                with self.assertRaises(ValueError):
                    model.update_accumulated_statistics(**kwargs)

    def test_compute_total_loss_matches_sum_of_weighted_losses(self):
        cfg = self.preset(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        model = SamplerAuxiliaryLosses(cfg)
        logits, probabilities, gates, skip_mask = self.sample_statistics()
        model.update_accumulated_statistics(logits, probabilities, gates, skip_mask)

        total_loss = model._SamplerAuxiliaryLosses__compute_total_loss()
        expected = (
            model.coefficient_of_variation_loss.get_weighted_loss()
            + model.switch_loss.get_weighted_loss()
            + model.zero_centred_loss.get_weighted_loss()
            + model.mutual_information_loss.get_weighted_loss()
        )

        torch.testing.assert_close(total_loss, expected)

    def test_get_auxiliary_loss_and_clear_resets_all_accumulations(self):
        cfg = self.preset(
            coefficient_of_variation_loss_weight=0.1,
            switch_loss_weight=0.1,
            zero_centred_loss_weight=0.1,
            mutual_information_loss_weight=0.1,
        )
        model = SamplerAuxiliaryLosses(cfg)
        logits, probabilities, gates, skip_mask = self.sample_statistics()

        model.update_accumulated_statistics(logits, probabilities, gates, skip_mask)
        total_loss = model.get_auxiliary_loss_and_clear()

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assert_accumulations_clear(model)
