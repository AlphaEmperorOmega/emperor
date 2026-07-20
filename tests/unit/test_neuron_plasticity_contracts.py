import math
import unittest

import torch
import torch.nn as nn

from emperor.neuron._cluster.plasticity import _NeuronClusterPlasticityMixin

_COPY_AND_PERTURB_PARAMETERS = (
    "_NeuronClusterPlasticityMixin__copy_and_perturb_parent_parameters"
)


class _SingleParameterModule(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(values.clone())


class _TiedAndUntiedParameterModule(nn.Module):
    def __init__(
        self,
        shared_values: torch.Tensor,
        later_values: torch.Tensor,
    ) -> None:
        super().__init__()
        shared_parameter = nn.Parameter(shared_values.clone())
        self.shared_role_a = shared_parameter
        self.shared_role_b = shared_parameter
        self.later_weight = nn.Parameter(later_values.clone())


class TestNeuronPlasticityNumerics(unittest.TestCase):
    def test_mitosis_uses_exact_float64_population_variance_scale(self) -> None:
        parent = _SingleParameterModule(
            torch.tensor(
                [100_000_000.0, 100_000_002.0] * 4,
                dtype=torch.float64,
            )
        )
        grown = _SingleParameterModule(torch.zeros(8, dtype=torch.float64))
        parent_before = parent.weight.detach().clone()
        population_std = parent_before.std(correction=0)
        torch.manual_seed(20260719)
        rng_state = torch.random.get_rng_state().clone()
        expected_noise = torch.randn_like(parent.weight)
        expected_weight = parent_before + expected_noise * population_std * 0.01
        torch.random.set_rng_state(rng_state)

        plasticity = _NeuronClusterPlasticityMixin()
        copy_and_perturb = (
            plasticity._NeuronClusterPlasticityMixin__copy_and_perturb_parent_parameters
        )
        result = copy_and_perturb(grown, parent)

        self.assertIs(result, grown)
        self.assertEqual(float(population_std), 1.0)
        torch.testing.assert_close(
            grown.weight,
            expected_weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            parent.weight,
            parent_before,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(grown.weight.requires_grad)

    def test_mitosis_tied_alias_does_not_skip_later_parameter_role(self) -> None:
        parent = _TiedAndUntiedParameterModule(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([10.0, 14.0], dtype=torch.float64),
        )
        grown = _TiedAndUntiedParameterModule(
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        )
        parent_shared_before = parent.shared_role_a.detach().clone()
        parent_later_before = parent.later_weight.detach().clone()
        torch.manual_seed(20260720)
        rng_state = torch.random.get_rng_state().clone()
        shared_noise = torch.randn_like(parent.shared_role_a)
        later_noise = torch.randn_like(parent.later_weight)
        expected_shared = (
            parent_shared_before
            + shared_noise * parent_shared_before.std(correction=0) * 0.01
        )
        expected_later = (
            parent_later_before
            + later_noise * parent_later_before.std(correction=0) * 0.01
        )
        torch.random.set_rng_state(rng_state)

        plasticity = _NeuronClusterPlasticityMixin()
        copy_and_perturb = (
            plasticity._NeuronClusterPlasticityMixin__copy_and_perturb_parent_parameters
        )
        copy_and_perturb(grown, parent)

        self.assertIs(grown.shared_role_a, grown.shared_role_b)
        torch.testing.assert_close(
            grown.shared_role_a,
            expected_shared,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            grown.later_weight,
            expected_later,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(parent.shared_role_a, parent_shared_before)
        torch.testing.assert_close(parent.later_weight, parent_later_before)

    def test_low_precision_mitosis_uses_population_scale_in_parameter_dtype(
        self,
    ) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                parent = _SingleParameterModule(
                    torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
                )
                grown = _SingleParameterModule(torch.zeros(3, dtype=dtype))
                population_std = torch.tensor(
                    math.sqrt(2.0 / 3.0),
                    dtype=torch.float32,
                ).to(dtype)
                torch.manual_seed(20260721)
                rng_state = torch.random.get_rng_state().clone()
                expected_noise = torch.randn_like(parent.weight)
                expected_rng_state = torch.random.get_rng_state().clone()
                expected_weight = (
                    parent.weight.detach() + expected_noise * population_std * 0.01
                )
                torch.random.set_rng_state(rng_state)

                plasticity = _NeuronClusterPlasticityMixin()
                copy_and_perturb = getattr(
                    plasticity,
                    _COPY_AND_PERTURB_PARAMETERS,
                )
                copy_and_perturb(grown, parent)

                self.assertTrue(torch.equal(grown.weight, expected_weight))
                self.assertEqual(grown.weight.dtype, dtype)
                torch.testing.assert_close(
                    torch.random.get_rng_state(),
                    expected_rng_state,
                )

    def test_mitosis_at_noise_floor_copies_without_consuming_rng(self) -> None:
        parent = _SingleParameterModule(
            torch.tensor([-1e-6, 1e-6], dtype=torch.float64)
        )
        grown = _SingleParameterModule(torch.zeros(2, dtype=torch.float64))
        torch.manual_seed(20260722)
        expected_rng_state = torch.random.get_rng_state().clone()

        plasticity = _NeuronClusterPlasticityMixin()
        copy_and_perturb = (
            plasticity._NeuronClusterPlasticityMixin__copy_and_perturb_parent_parameters
        )
        copy_and_perturb(grown, parent)

        torch.testing.assert_close(grown.weight, parent.weight, rtol=0.0, atol=0.0)
        torch.testing.assert_close(torch.random.get_rng_state(), expected_rng_state)
