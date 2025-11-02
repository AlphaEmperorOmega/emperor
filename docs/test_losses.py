import torch
import unittest
from math import prod
import torch.nn.functional as F
from Emperor.generators.utils.losses import (
    AuxiliaryLossBase,
    CoefficientOfVariationLoss,
    MutualInformationLoss,
    SwitchLoss,
    ZeroCentredLoss,
)


class TestCoefficientOfVariationLoss(unittest.TestCase):
    def test__init(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        self.assertEqual(m.loss_weight, loss_weight)
        self.assertIsInstance(m.default_error, torch.Tensor)
        self.assertIsInstance(m, AuxiliaryLossBase)

    def test__update_accumulation__weight_loss__0(self):
        loss_weight = 0.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(probabilities)

        self.assertIsNone(m.gates_accumulation)

    def test__update_accumulation__weight_loss__1(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(probabilities)

        expected_accumulation = torch.sum(probabilities, dim=0)
        self.assertTrue(torch.allclose(m.gates_accumulation, expected_accumulation))

    def test__update_accumulation__consecutive_updates(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(probabilities)
        m.update_accumulation(probabilities)

        expected_accumulation = torch.sum(probabilities, dim=0)
        expected_accumulation += torch.sum(probabilities, dim=0)

        self.assertTrue(torch.allclose(m.gates_accumulation, expected_accumulation))

    def test__compute_coefficient_of_variation(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()

        m.update_accumulation(probabilities)

        output = m._CoefficientOfVariationLoss__compute_coefficient_of_variation()

        probabilities = F.normalize(m.gates_accumulation, p=1, dim=0)
        variation = probabilities.float().var()
        mean = probabilities.float().mean() ** 2
        expected_loss = variation / (mean + m.eps)

        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.allclose(output, expected_loss))
        self.assertNotEqual(output, 0.0)

    def test__is_accumulation_shape_valid__no_update_accumulation(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        with self.assertRaises(ValueError):
            m._CoefficientOfVariationLoss__is_accumulation_shape_valid()

    def test__is_accumulation_shape_valid__with__update_accumulation(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()

        m.update_accumulation(probabilities)

        output = m._CoefficientOfVariationLoss__is_accumulation_shape_valid()

        self.assertFalse(output)

    def test__is_accumulation_shape_valid__with__update_accumulation__force_True(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()

        m.update_accumulation(probabilities)
        m.gates_accumulation = torch.unsqueeze(m.gates_accumulation, dim=0)

        output = m._CoefficientOfVariationLoss__is_accumulation_shape_valid()

        self.assertTrue(output)

    def test__compute_loss(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()

        m.update_accumulation(probabilities)

        output = m._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output, 0.0)

    def test__compute_loss__is_accumulation_valid__force_True(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()

        m.update_accumulation(probabilities)
        m.gates_accumulation = torch.unsqueeze(m.gates_accumulation, dim=0)

        output = m._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output, 0.0)

    def test__reset_loss(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()

        m.update_accumulation(probabilities)
        m.reset_loss()

        self.assertIsNone(m.gates_accumulation)

    def test__get_weighted_loss__loss_weight__0(self):
        loss_weight = 0.0
        m = CoefficientOfVariationLoss(loss_weight)
        output = m.get_weighted_loss()
        self.assertEqual(output, 0.0)

    def test__get_weighted_loss__loss_weight__1(self):
        loss_weight = 1.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        m.update_accumulation(probabilities)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss()

        self.assertEqual(output, expected_loss)

    def test__get_weighted_loss__loss_weight__2(self):
        loss_weight = 2.0
        m = CoefficientOfVariationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        m.update_accumulation(probabilities)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss() * 2

        self.assertEqual(output, expected_loss)


class TestSwitchLoss(unittest.TestCase):
    def test__init(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        self.assertEqual(m.loss_weight, loss_weight)
        self.assertEqual(m.num_experts, top_k)
        self.assertIsInstance(m.default_error, torch.Tensor)
        self.assertIsInstance(m, AuxiliaryLossBase)

    def test__update_accumulation__weight_loss__0(self):
        loss_weight = 0.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        gates = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(probabilities, gates)

        self.assertIsNone(m.probability_accumulation)
        self.assertIsNone(m.frequency_accumulation)

    def test__update_accumulation__weight_loss__1(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)
        shape = (batch_size, output_dim)
        gates = torch.randn(batch_size, output_dim)

        m.update_accumulation(probabilities, gates)

        expected_probabilities = torch.sum(probabilities, dim=0)
        expected_frequency = torch.sum((gates > 0).float(), dim=0)
        self.assertTrue(
            torch.allclose(m.probability_accumulation, expected_probabilities)
        )
        self.assertTrue(torch.allclose(m.frequency_accumulation, expected_frequency))

    def test__update_accumulation__consecutive_updates(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)
        shape = (batch_size, output_dim)
        gates = torch.randn(batch_size, output_dim)

        m.update_accumulation(probabilities, gates)
        m.update_accumulation(probabilities, gates)

        expected_probabilities = torch.sum(probabilities, dim=0) * 2
        expected_frequency = torch.sum((gates > 0).float(), dim=0) * 2
        self.assertTrue(
            torch.allclose(m.probability_accumulation, expected_probabilities)
        )
        self.assertTrue(torch.allclose(m.frequency_accumulation, expected_frequency))

    def test__compute_switch_loss(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        shape = (batch_size, output_dim)
        gates = torch.randn(batch_size, output_dim)

        m.update_accumulation(probabilities, gates)

        output = m._SwitchLoss__compute_switch_loss()

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output, 0.0)

    def test__compute_loss(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        shape = (batch_size, output_dim)
        gates = torch.randn(batch_size, output_dim)

        m.update_accumulation(probabilities, gates)

        output = m._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output, 0.0)

    def test__compute_loss__is_accumulation_valid__force_True(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        with self.assertRaises(ValueError):
            m._compute_loss()

    def test__reset_loss(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        shape = (batch_size, output_dim)
        gates = torch.randn(batch_size, output_dim)

        m.update_accumulation(probabilities, gates)
        m.reset_loss()

        self.assertIsNone(m.probability_accumulation)
        self.assertIsNone(m.frequency_accumulation)

    def test__get_weighted_loss__loss_weight__0(self):
        loss_weight = 0.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)
        output = m.get_weighted_loss()
        self.assertEqual(output, 0.0)

    def test__get_weighted_loss__loss_weight__1(self):
        loss_weight = 1.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        shape = (batch_size, output_dim)
        gates = torch.randn(batch_size, output_dim)

        m.update_accumulation(probabilities, gates)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss()

        self.assertEqual(output, expected_loss)

    def test__get_weighted_loss__loss_weight__2(self):
        loss_weight = 2.0
        top_k = 3
        m = SwitchLoss(top_k, loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape).float()
        shape = (batch_size, output_dim)
        gates = torch.randn(batch_size, output_dim)

        m.update_accumulation(probabilities, gates)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss() * 2

        self.assertEqual(output, expected_loss)


class TestZeroCentredLoss(unittest.TestCase):
    def test__init(self):
        loss_weight = 1.0
        m = ZeroCentredLoss(loss_weight)

        self.assertEqual(m.loss_weight, loss_weight)
        self.assertIsInstance(m.default_error, torch.Tensor)
        self.assertIsInstance(m, AuxiliaryLossBase)

    def test__compute_squared_log_sum_exp(self):
        loss_weight = 0.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        output = m._ZeroCentredLoss__compute_squared_log_sum_exp(logits)

        expected_output = torch.exp(logits).sum(dim=-1)
        expected_output = torch.log(expected_output) ** 2
        expected_output = torch.sum(expected_output)

        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.allclose(output, expected_output))

    def test__update_accumulation__weight_loss__0(self):
        loss_weight = 0.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(logits)

        self.assertIsNone(m.squared_log_sum_exp_accumulation)
        self.assertIsNone(m.count_accumulation)

    def test__update_accumulation__weight_loss__1(self):
        loss_weight = 1.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(logits)

        squared_log_sum_exp = m._ZeroCentredLoss__compute_squared_log_sum_exp(logits)
        expected_squared_log_sum_exp_accumulation = torch.sum(squared_log_sum_exp)
        expected_count_accumulation = logits.size(0)

        self.assertTrue(
            torch.allclose(
                m.squared_log_sum_exp_accumulation,
                expected_squared_log_sum_exp_accumulation,
            )
        )
        self.assertEqual(m.count_accumulation, expected_count_accumulation)

    def test__update_accumulation__consecutive_updates(self):
        loss_weight = 1.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(logits)
        m.update_accumulation(logits)

        squared_log_sum_exp = m._ZeroCentredLoss__compute_squared_log_sum_exp(logits)
        expected_squared_log_sum_exp_accumulation = torch.sum(squared_log_sum_exp) * 2
        expected_count_accumulation = torch.tensor(logits.size(0) * 2)

        self.assertTrue(
            torch.allclose(
                m.squared_log_sum_exp_accumulation,
                expected_squared_log_sum_exp_accumulation,
            )
        )
        self.assertTrue(
            torch.allclose(m.count_accumulation, expected_count_accumulation)
        )

    def test__compute_zero_centred_loss(self):
        loss_weight = 1.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(logits)

        output = m._ZeroCentredLoss__compute_zero_centred_loss()
        expected_loss = m.squared_log_sum_exp_accumulation / m.count_accumulation

        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(torch.allclose(output, expected_loss))
        self.assertNotEqual(output, 0.0)

    def test__compute_loss(self):
        loss_weight = 1.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(logits)

        output = m._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output, 0.0)

    def test__reset_loss(self):
        loss_weight = 1.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        m.update_accumulation(logits)
        m.reset_loss()

        self.assertIsNone(m.squared_log_sum_exp_accumulation)
        self.assertIsNone(m.count_accumulation)

    def test__get_weighted_loss__loss_weight__0(self):
        loss_weight = 0.0
        m = ZeroCentredLoss(loss_weight)
        output = m.get_weighted_loss()
        self.assertEqual(output, 0.0)

    def test__get_weighted_loss__loss_weight__1(self):
        loss_weight = 1.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)
        m.update_accumulation(logits)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss()

        self.assertEqual(output, expected_loss)

    def test__get_weighted_loss__loss_weight__2(self):
        loss_weight = 2.0
        m = ZeroCentredLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)
        m.update_accumulation(logits)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss() * 2

        self.assertEqual(output, expected_loss)


class TestMutualInformationLoss(unittest.TestCase):
    def test__init(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        self.assertEqual(m.loss_weight, loss_weight)
        self.assertIsInstance(m.default_error, torch.Tensor)
        self.assertIsInstance(m, AuxiliaryLossBase)

    def test__update_accumulation__weight_loss__0(self):
        loss_weight = 0.0
        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_maks = (torch.randn(*shape) > 0).float()

        m.update_accumulation(logits, probabilities, skip_maks)

        self.assertListEqual(m.log_probabilities, [])
        self.assertListEqual(m.probabilities, [])
        self.assertListEqual(m.skip_masks, [])

    def test__update_accumulation__weight_loss__1(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_masks = (torch.randn(*shape) > 0).float()

        m.update_accumulation(logits, probabilities, skip_masks)

        expected_log_probabilities = torch.log_softmax(logits, dim=-1)
        self.assertEqual(len(m.log_probabilities), 1)
        self.assertTrue(
            torch.allclose(m.log_probabilities[0], expected_log_probabilities)
        )
        self.assertEqual(len(m.probabilities), 1)
        self.assertTrue(torch.allclose(m.probabilities[0], probabilities))
        self.assertEqual(len(m.skip_masks), 1)
        self.assertTrue(torch.allclose(m.skip_masks[0], skip_masks))

    def test__update_accumulation__consecutive_updates(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_masks = (torch.randn(*shape) > 0).float()

        m.update_accumulation(logits, probabilities, skip_masks)
        m.update_accumulation(logits, probabilities, skip_masks)

        expected_log_probabilities = torch.log_softmax(logits, dim=-1)
        self.assertEqual(len(m.log_probabilities), 2)
        self.assertTrue(
            torch.allclose(m.log_probabilities[0], expected_log_probabilities)
        )
        self.assertTrue(
            torch.allclose(m.log_probabilities[1], expected_log_probabilities)
        )
        self.assertEqual(len(m.probabilities), 2)
        self.assertTrue(torch.allclose(m.probabilities[0], probabilities))
        self.assertTrue(torch.allclose(m.probabilities[1], probabilities))
        self.assertEqual(len(m.skip_masks), 2)
        self.assertTrue(torch.allclose(m.skip_masks[0], skip_masks))
        self.assertTrue(torch.allclose(m.skip_masks[1], skip_masks))

    def test__compute_mutual_information_loss(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_masks = torch.abs(torch.randn(*shape))

        m.update_accumulation(logits, probabilities, skip_masks)
        output = m._MutualInformationLoss__compute_mutual_information_loss()

        probabilities = torch.cat(m.probabilities, dim=0)
        log_probabilities = torch.cat(m.log_probabilities, dim=0)
        masks = torch.cat(m.skip_masks, dim=0)

        p_x = masks / (masks.sum() + 1e-12)
        p_e = (p_x * probabilities).sum(0)
        H_e = (p_e * p_e.log()).sum()

        meg_H_e_given_x = (p_x * probabilities * log_probabilities).sum()
        expected_output = -(meg_H_e_given_x + H_e)

        self.assertTrue(torch.allclose(output, expected_output))
        self.assertNotEqual(output, 0.0)

    def test__compute_loss(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_masks = torch.abs(torch.randn(*shape))

        m.update_accumulation(logits, probabilities, skip_masks)

        output = m._compute_loss()

        self.assertIsInstance(output, torch.Tensor)
        self.assertNotEqual(output, 0.0)

    def test__compute_loss__is_accumulation_valid__force_True(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        with self.assertRaises(ValueError):
            m._compute_loss()

    def test__reset_loss(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_masks = torch.abs(torch.randn(*shape))

        m.update_accumulation(logits, probabilities, skip_masks)
        m.reset_loss()

        self.assertListEqual(m.log_probabilities, [])
        self.assertListEqual(m.probabilities, [])
        self.assertListEqual(m.skip_masks, [])

    def test__get_weighted_loss__loss_weight__0(self):
        loss_weight = 0.0
        m = MutualInformationLoss(loss_weight)
        output = m.get_weighted_loss()
        self.assertEqual(output, 0.0)

    def test__get_weighted_loss__loss_weight__1(self):
        loss_weight = 1.0
        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_masks = torch.abs(torch.randn(*shape))

        m.update_accumulation(logits, probabilities, skip_masks)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss()

        self.assertEqual(output, expected_loss)

    def test__get_weighted_loss__loss_weight__2(self):
        loss_weight = 2.0

        m = MutualInformationLoss(loss_weight)

        batch_size = 2
        output_dim = 5
        shape = (batch_size, output_dim)
        logits = torch.arange(prod(shape)).reshape(shape).float()

        shape = (batch_size, output_dim)
        probabilities = torch.arange(prod(shape)).reshape(shape)

        shape = (batch_size, output_dim)
        skip_masks = torch.abs(torch.randn(*shape))

        m.update_accumulation(logits, probabilities, skip_masks)

        output = m.get_weighted_loss()
        expected_loss = m._compute_loss() * 2

        self.assertEqual(output, expected_loss)
