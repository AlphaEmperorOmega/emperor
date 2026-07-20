import unittest

import torch
import torch.nn as nn

from emperor.neuron._cluster.plasticity import _NeuronClusterPlasticityMixin


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
