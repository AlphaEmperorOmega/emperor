import unittest
from unittest import mock

import torch

from emperor.sampler.core.config import SamplerConfig
from emperor.sampler.core.losses import (
    AuxiliaryLossBase,
    CoefficientOfVariationLoss,
    MutualInformationLoss,
    SamplerAuxiliaryLosses,
    SwitchLoss,
    ZeroCentredLoss,
)


def accelerator_device_is_usable(device: torch.device) -> bool:
    try:
        torch.ones(1, device=device).sum().item()
    except (RuntimeError, AssertionError):
        return False
    return True


def available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    accelerator_devices = []
    if torch.cuda.is_available():
        accelerator_devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        accelerator_devices.append(torch.device("mps"))
    devices.extend(
        device for device in accelerator_devices if accelerator_device_is_usable(device)
    )
    return devices


class TestAvailableDevices(unittest.TestCase):
    def test_skips_cuda_when_tensor_probe_fails(self):
        original_ones = torch.ones

        def failing_ones(*args, **kwargs):
            if torch.device(kwargs.get("device", "cpu")).type == "cuda":
                raise RuntimeError("cudaErrorNoKernelImageForDevice")
            return original_ones(*args, **kwargs)

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            mock.patch("torch.ones", side_effect=failing_ones),
        ):
            devices = available_devices()

        self.assertEqual(devices, [torch.device("cpu")])

    def test_includes_cuda_when_tensor_probe_succeeds(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            mock.patch("torch.ones", return_value=torch.ones(1)),
        ):
            devices = available_devices()

        self.assertEqual(devices, [torch.device("cpu"), torch.device("cuda")])


class TestCoefficientOfVariationLoss(unittest.TestCase):
    def gates(self) -> torch.Tensor:
        return torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 1.0],
            ]
        )

    def test_init_stores_loss_weight(self):
        loss = CoefficientOfVariationLoss(loss_weight=0.25)

        self.assertEqual(loss.loss_weight, 0.25)
        self.assertEqual(loss.eps, 1e-10)
        self.assertIsNone(loss.gates_accumulation)
        self.assertFalse(hasattr(loss, "default_error"))
        self.assertIsInstance(loss, AuxiliaryLossBase)

    def test_zero_weight_update_is_noop(self):
        loss = CoefficientOfVariationLoss(loss_weight=0.0)

        loss.update_accumulation(self.gates())

        self.assertIsNone(loss.gates_accumulation)

    def test_update_accumulation_sums_gates(self):
        loss = CoefficientOfVariationLoss(loss_weight=1.0)

        loss.update_accumulation(self.gates())

        torch.testing.assert_close(
            loss.gates_accumulation,
            torch.tensor([1.0, 3.0, 3.0]),
        )

    def test_consecutive_updates_accumulate_gates(self):
        loss = CoefficientOfVariationLoss(loss_weight=1.0)

        loss.update_accumulation(self.gates())
        loss.update_accumulation(self.gates())

        torch.testing.assert_close(
            loss.gates_accumulation,
            torch.tensor([2.0, 6.0, 6.0]),
        )

    def test_compute_loss_matches_coefficient_of_variation(self):
        loss = CoefficientOfVariationLoss(loss_weight=1.0)
        loss.update_accumulation(self.gates())

        output = loss._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        torch.testing.assert_close(output, torch.tensor(12.0 / 49.0))

    def test_single_sample_returns_zero_scalar_on_accumulation_device(self):
        for device in available_devices():
            with self.subTest(device=device):
                loss = CoefficientOfVariationLoss(loss_weight=0.1)
                gates = torch.ones(2, 1, device=device)

                loss.update_accumulation(gates)
                total_loss = loss.get_weighted_loss(gates.new_zeros(()))

                self.assertEqual(total_loss.shape, torch.Size([]))
                self.assertEqual(total_loss.device, loss.gates_accumulation.device)
                torch.testing.assert_close(
                    total_loss, loss.gates_accumulation.new_zeros(())
                )

    def test_missing_input_raises_value_error(self):
        loss = CoefficientOfVariationLoss(loss_weight=1.0)

        with self.assertRaises(ValueError):
            loss.update_accumulation(None)

    def test_compute_loss_without_accumulation_raises_value_error(self):
        loss = CoefficientOfVariationLoss(loss_weight=1.0)

        with self.assertRaises(ValueError):
            loss._compute_loss()

    def test_reset_clears_accumulation(self):
        loss = CoefficientOfVariationLoss(loss_weight=1.0)
        loss.update_accumulation(self.gates())

        loss.reset_loss()

        self.assertIsNone(loss.gates_accumulation)

    def test_weighted_loss_returns_caller_default_when_disabled(self):
        loss = CoefficientOfVariationLoss(loss_weight=0.0)
        default_loss = torch.ones(())

        output = loss.get_weighted_loss(default_loss)

        self.assertIs(output, default_loss)

    def test_weighted_loss_scales_by_loss_weight(self):
        loss = CoefficientOfVariationLoss(loss_weight=2.5)
        loss.update_accumulation(self.gates())

        output = loss.get_weighted_loss(torch.zeros(()))

        torch.testing.assert_close(output, torch.tensor((12.0 / 49.0) * 2.5))


class TestSwitchLoss(unittest.TestCase):
    def probabilities(self) -> torch.Tensor:
        return torch.tensor(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.6, 0.3],
                [0.2, 0.2, 0.6],
            ]
        )

    def gates(self) -> torch.Tensor:
        return torch.tensor(
            [
                [0.7, 0.0, 0.0],
                [0.0, 0.6, 0.0],
                [0.2, 0.0, 0.6],
            ]
        )

    def test_init_stores_num_experts_and_loss_weight(self):
        loss = SwitchLoss(num_experts=3, loss_weight=0.25)

        self.assertEqual(loss.loss_weight, 0.25)
        self.assertEqual(loss.num_experts, 3)
        self.assertIsNone(loss.probability_accumulation)
        self.assertIsNone(loss.frequency_accumulation)
        self.assertFalse(hasattr(loss, "default_error"))
        self.assertIsInstance(loss, AuxiliaryLossBase)

    def test_zero_weight_update_is_noop(self):
        loss = SwitchLoss(num_experts=3, loss_weight=0.0)

        loss.update_accumulation(self.probabilities(), self.gates())

        self.assertIsNone(loss.probability_accumulation)
        self.assertIsNone(loss.frequency_accumulation)

    def test_update_accumulation_sums_probabilities_and_gate_frequency(self):
        loss = SwitchLoss(num_experts=3, loss_weight=1.0)

        loss.update_accumulation(self.probabilities(), self.gates())

        torch.testing.assert_close(
            loss.probability_accumulation,
            torch.tensor([1.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(
            loss.frequency_accumulation,
            torch.tensor([2.0, 1.0, 1.0]),
        )

    def test_consecutive_updates_accumulate_probabilities_and_gate_frequency(self):
        loss = SwitchLoss(num_experts=3, loss_weight=1.0)

        loss.update_accumulation(self.probabilities(), self.gates())
        loss.update_accumulation(self.probabilities(), self.gates())

        torch.testing.assert_close(
            loss.probability_accumulation,
            torch.tensor([2.0, 2.0, 2.0]),
        )
        torch.testing.assert_close(
            loss.frequency_accumulation,
            torch.tensor([4.0, 2.0, 2.0]),
        )

    def test_compute_loss_matches_switch_loss(self):
        loss = SwitchLoss(num_experts=3, loss_weight=1.0)
        loss.update_accumulation(self.probabilities(), self.gates())

        output = loss._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        torch.testing.assert_close(output, torch.tensor(1.0))

    def test_missing_inputs_raise_value_error(self):
        loss = SwitchLoss(num_experts=3, loss_weight=1.0)

        with self.assertRaises(ValueError):
            loss.update_accumulation(None, self.gates())

        with self.assertRaises(ValueError):
            loss.update_accumulation(self.probabilities(), None)

    def test_compute_loss_without_accumulation_raises_value_error(self):
        loss = SwitchLoss(num_experts=3, loss_weight=1.0)

        with self.assertRaises(ValueError):
            loss._compute_loss()

    def test_reset_clears_accumulation(self):
        loss = SwitchLoss(num_experts=3, loss_weight=1.0)
        loss.update_accumulation(self.probabilities(), self.gates())

        loss.reset_loss()

        self.assertIsNone(loss.probability_accumulation)
        self.assertIsNone(loss.frequency_accumulation)

    def test_weighted_loss_returns_caller_default_when_disabled(self):
        loss = SwitchLoss(num_experts=3, loss_weight=0.0)
        default_loss = torch.ones(())

        output = loss.get_weighted_loss(default_loss)

        self.assertIs(output, default_loss)

    def test_weighted_loss_scales_by_loss_weight(self):
        loss = SwitchLoss(num_experts=3, loss_weight=2.0)
        loss.update_accumulation(self.probabilities(), self.gates())

        output = loss.get_weighted_loss(torch.zeros(()))

        torch.testing.assert_close(output, torch.tensor(2.0))


class TestZeroCentredLoss(unittest.TestCase):
    def logits(self) -> torch.Tensor:
        return torch.zeros(2, 3)

    def expected_squared_log_sum_exp(self) -> torch.Tensor:
        return 2 * torch.log(torch.tensor(3.0)) ** 2

    def expected_loss(self) -> torch.Tensor:
        return torch.log(torch.tensor(3.0)) ** 2

    def test_init_stores_loss_weight(self):
        loss = ZeroCentredLoss(loss_weight=0.25)

        self.assertEqual(loss.loss_weight, 0.25)
        self.assertIsNone(loss.squared_log_sum_exp_accumulation)
        self.assertIsNone(loss.count_accumulation)
        self.assertFalse(hasattr(loss, "default_error"))
        self.assertIsInstance(loss, AuxiliaryLossBase)

    def test_zero_weight_update_is_noop(self):
        loss = ZeroCentredLoss(loss_weight=0.0)

        loss.update_accumulation(self.logits())

        self.assertIsNone(loss.squared_log_sum_exp_accumulation)
        self.assertIsNone(loss.count_accumulation)

    def test_update_accumulation_sums_squared_log_sum_exp_and_count(self):
        loss = ZeroCentredLoss(loss_weight=1.0)

        loss.update_accumulation(self.logits())

        torch.testing.assert_close(
            loss.squared_log_sum_exp_accumulation,
            self.expected_squared_log_sum_exp(),
        )
        self.assertEqual(loss.count_accumulation.item(), 2)

    def test_consecutive_updates_accumulate_squared_log_sum_exp_and_count(self):
        loss = ZeroCentredLoss(loss_weight=1.0)

        loss.update_accumulation(self.logits())
        loss.update_accumulation(self.logits())

        torch.testing.assert_close(
            loss.squared_log_sum_exp_accumulation,
            self.expected_squared_log_sum_exp() * 2,
        )
        self.assertEqual(loss.count_accumulation.item(), 4)

    def test_compute_loss_matches_zero_centred_loss(self):
        loss = ZeroCentredLoss(loss_weight=1.0)
        loss.update_accumulation(self.logits())

        output = loss._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        torch.testing.assert_close(output, self.expected_loss())

    def test_compute_loss_matches_varied_logits(self):
        loss = ZeroCentredLoss(loss_weight=1.0)
        logits = torch.tensor(
            [
                [1.0, 0.0, -1.0],
                [0.5, -0.5, 0.0],
            ]
        )

        loss.update_accumulation(logits)
        output = loss._compute_loss()

        expected = torch.logsumexp(logits, dim=-1).square().mean()
        torch.testing.assert_close(output, expected)

    def test_missing_input_raises_value_error(self):
        loss = ZeroCentredLoss(loss_weight=1.0)

        with self.assertRaises(ValueError):
            loss.update_accumulation(None)

    def test_compute_loss_without_accumulation_raises_value_error(self):
        loss = ZeroCentredLoss(loss_weight=1.0)

        with self.assertRaises(ValueError):
            loss._compute_loss()

    def test_reset_clears_accumulation(self):
        loss = ZeroCentredLoss(loss_weight=1.0)
        loss.update_accumulation(self.logits())

        loss.reset_loss()

        self.assertIsNone(loss.squared_log_sum_exp_accumulation)
        self.assertIsNone(loss.count_accumulation)

    def test_weighted_loss_returns_caller_default_when_disabled(self):
        loss = ZeroCentredLoss(loss_weight=0.0)
        default_loss = torch.ones(())

        output = loss.get_weighted_loss(default_loss)

        self.assertIs(output, default_loss)

    def test_weighted_loss_scales_by_loss_weight(self):
        loss = ZeroCentredLoss(loss_weight=2.0)
        loss.update_accumulation(self.logits())

        output = loss.get_weighted_loss(torch.zeros(()))

        torch.testing.assert_close(output, self.expected_loss() * 2)


class TestMutualInformationLoss(unittest.TestCase):
    def statistics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = torch.tensor(
            [
                [0.8, 0.2],
                [0.25, 0.75],
            ]
        )
        logits = probabilities.log()
        skip_masks = torch.ones(2, 1)
        return logits, probabilities, skip_masks

    def expected_loss(self) -> torch.Tensor:
        conditional_entropy_term = 0.5 * (
            0.8 * torch.log(torch.tensor(0.8))
            + 0.2 * torch.log(torch.tensor(0.2))
            + 0.25 * torch.log(torch.tensor(0.25))
            + 0.75 * torch.log(torch.tensor(0.75))
        )
        marginal_entropy_term = (
            0.525 * torch.log(torch.tensor(0.525))
            + 0.475 * torch.log(torch.tensor(0.475))
        )
        return -(conditional_entropy_term + marginal_entropy_term)

    def test_init_stores_loss_weight(self):
        loss = MutualInformationLoss(loss_weight=0.25)

        self.assertEqual(loss.loss_weight, 0.25)
        self.assertListEqual(loss.log_probabilities, [])
        self.assertListEqual(loss.probabilities, [])
        self.assertListEqual(loss.skip_masks, [])
        self.assertFalse(hasattr(loss, "default_error"))
        self.assertIsInstance(loss, AuxiliaryLossBase)

    def test_zero_weight_update_is_noop(self):
        loss = MutualInformationLoss(loss_weight=0.0)
        logits, probabilities, skip_masks = self.statistics()

        loss.update_accumulation(logits, probabilities, skip_masks)

        self.assertListEqual(loss.log_probabilities, [])
        self.assertListEqual(loss.probabilities, [])
        self.assertListEqual(loss.skip_masks, [])

    def test_update_accumulation_stores_log_probabilities_probabilities_and_masks(self):
        loss = MutualInformationLoss(loss_weight=1.0)
        logits, probabilities, skip_masks = self.statistics()

        loss.update_accumulation(logits, probabilities, skip_masks)

        self.assertEqual(len(loss.log_probabilities), 1)
        torch.testing.assert_close(loss.log_probabilities[0], probabilities.log())
        self.assertEqual(len(loss.probabilities), 1)
        torch.testing.assert_close(loss.probabilities[0], probabilities)
        self.assertEqual(len(loss.skip_masks), 1)
        torch.testing.assert_close(loss.skip_masks[0], skip_masks)

    def test_consecutive_updates_append_accumulations(self):
        loss = MutualInformationLoss(loss_weight=1.0)
        logits, probabilities, skip_masks = self.statistics()

        loss.update_accumulation(logits, probabilities, skip_masks)
        loss.update_accumulation(logits, probabilities, skip_masks)

        self.assertEqual(len(loss.log_probabilities), 2)
        torch.testing.assert_close(loss.log_probabilities[0], probabilities.log())
        torch.testing.assert_close(loss.log_probabilities[1], probabilities.log())
        self.assertEqual(len(loss.probabilities), 2)
        torch.testing.assert_close(loss.probabilities[0], probabilities)
        torch.testing.assert_close(loss.probabilities[1], probabilities)
        self.assertEqual(len(loss.skip_masks), 2)
        torch.testing.assert_close(loss.skip_masks[0], skip_masks)
        torch.testing.assert_close(loss.skip_masks[1], skip_masks)

    def test_compute_loss_matches_mutual_information_loss(self):
        loss = MutualInformationLoss(loss_weight=1.0)
        logits, probabilities, skip_masks = self.statistics()
        loss.update_accumulation(logits, probabilities, skip_masks)

        output = loss._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        torch.testing.assert_close(output, self.expected_loss())

    def test_compute_loss_weights_rows_by_skip_masks(self):
        loss = MutualInformationLoss(loss_weight=1.0)
        logits, probabilities, _ = self.statistics()
        skip_masks = torch.tensor([[1.0], [0.0]])

        loss.update_accumulation(logits, probabilities, skip_masks)
        output = loss._compute_loss()

        kept_probabilities = probabilities[:1]
        expected = -(
            (
                kept_probabilities * kept_probabilities.log()
            ).sum()
            + (
                kept_probabilities.squeeze(0)
                * kept_probabilities.squeeze(0).log()
            ).sum()
        )
        torch.testing.assert_close(output, expected)

    def test_missing_inputs_raise_value_error(self):
        loss = MutualInformationLoss(loss_weight=1.0)
        logits, probabilities, skip_masks = self.statistics()

        with self.assertRaises(ValueError):
            loss.update_accumulation(None, probabilities, skip_masks)

        with self.assertRaises(ValueError):
            loss.update_accumulation(logits, None, skip_masks)

        with self.assertRaises(ValueError):
            loss.update_accumulation(logits, probabilities, None)

    def test_compute_loss_without_accumulation_raises_value_error(self):
        loss = MutualInformationLoss(loss_weight=1.0)

        with self.assertRaises(ValueError):
            loss._compute_loss()

    def test_reset_clears_accumulation(self):
        loss = MutualInformationLoss(loss_weight=1.0)
        logits, probabilities, skip_masks = self.statistics()
        loss.update_accumulation(logits, probabilities, skip_masks)

        loss.reset_loss()

        self.assertListEqual(loss.log_probabilities, [])
        self.assertListEqual(loss.probabilities, [])
        self.assertListEqual(loss.skip_masks, [])

    def test_weighted_loss_returns_caller_default_when_disabled(self):
        loss = MutualInformationLoss(loss_weight=0.0)
        default_loss = torch.ones(())

        output = loss.get_weighted_loss(default_loss)

        self.assertIs(output, default_loss)

    def test_weighted_loss_scales_by_loss_weight(self):
        loss = MutualInformationLoss(loss_weight=2.0)
        logits, probabilities, skip_masks = self.statistics()
        loss.update_accumulation(logits, probabilities, skip_masks)

        output = loss.get_weighted_loss(torch.zeros(()))

        torch.testing.assert_close(output, self.expected_loss() * 2)


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

        torch.testing.assert_close(total_loss, model.default_loss)
        self.assertEqual(total_loss.shape, torch.Size([]))
        self.assertEqual(total_loss.device, model.default_loss.device)
        self.assert_accumulations_clear(model)

    def test_disabled_total_loss_follows_module_device(self):
        for device in available_devices():
            with self.subTest(device=device):
                model = SamplerAuxiliaryLosses(self.preset()).to(device)

                total_loss = model.get_auxiliary_loss_and_clear()

                self.assertEqual(total_loss.shape, torch.Size([]))
                self.assertEqual(total_loss.device, model.default_loss.device)

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
        default_loss = model.default_loss
        expected = (
            model.coefficient_of_variation_loss.get_weighted_loss(default_loss)
            + model.switch_loss.get_weighted_loss(default_loss)
            + model.zero_centred_loss.get_weighted_loss(default_loss)
            + model.mutual_information_loss.get_weighted_loss(default_loss)
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

    def test_routing_losses_match_float64_reference_for_moderate_logits(self):
        model = SamplerAuxiliaryLosses(
            self.preset(
                zero_centred_loss_weight=1.0,
                mutual_information_loss_weight=1.0,
            )
        )
        logits = torch.tensor(
            [
                [1.25, -0.5, 0.75],
                [-0.25, 0.5, 1.0],
                [0.75, 0.25, -1.0],
            ]
        )
        probabilities = torch.softmax(logits, dim=-1)
        skip_mask = torch.tensor([[1.0], [0.5], [0.0]])

        model.update_accumulated_statistics(
            logits=logits,
            probabilities=probabilities,
            skip_mask=skip_mask,
        )
        output = model.get_auxiliary_loss_and_clear()

        reference_logits = logits.double()
        reference_probabilities = torch.softmax(reference_logits, dim=-1)
        reference_mask = skip_mask.double()
        p_x = reference_mask / reference_mask.sum()
        p_e = (p_x * reference_probabilities).sum(dim=0)
        zero_centred = torch.logsumexp(
            reference_logits, dim=-1
        ).square().mean()
        marginal_entropy_term = torch.special.xlogy(p_e, p_e).sum()
        conditional_entropy_term = (
            p_x
            * reference_probabilities
            * torch.log_softmax(reference_logits, dim=-1)
        ).sum()
        expected = zero_centred - (
            conditional_entropy_term + marginal_entropy_term
        )

        torch.testing.assert_close(output.double(), expected)

    def test_zero_centred_loss_is_finite_for_extreme_logits_and_dtypes(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = SamplerAuxiliaryLosses(
                    self.preset(zero_centred_loss_weight=1.0)
                )
                logits = torch.tensor(
                    [[1000.0, 999.0], [-1000.0, -999.0]],
                    dtype=dtype,
                    requires_grad=True,
                )

                model.update_accumulated_statistics(logits=logits)
                output = model.get_auxiliary_loss_and_clear()
                output.backward()

                self.assertTrue(torch.isfinite(output).item())
                self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_mutual_information_loss_is_finite_with_unused_experts(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = SamplerAuxiliaryLosses(
                    self.preset(
                        num_experts=3,
                        mutual_information_loss_weight=1.0,
                    )
                )
                logits = torch.tensor(
                    [
                        [1000.0, -1000.0, -1000.0],
                        [-1000.0, 1000.0, -1000.0],
                        [1000.0, -1000.0, -1000.0],
                    ],
                    dtype=dtype,
                    requires_grad=True,
                )
                probabilities = torch.softmax(logits, dim=-1)
                skip_mask = torch.tensor(
                    [[1.0], [0.0], [0.5]], dtype=dtype
                )

                model.update_accumulated_statistics(
                    logits=logits,
                    probabilities=probabilities,
                    skip_mask=skip_mask,
                )
                output = model.get_auxiliary_loss_and_clear()
                output.backward()

                self.assertTrue(torch.isfinite(output).item())
                self.assertTrue(torch.isfinite(logits.grad).all().item())
