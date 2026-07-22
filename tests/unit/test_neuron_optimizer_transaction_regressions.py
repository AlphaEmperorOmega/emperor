import unittest

import torch
from torch import nn

from emperor.neuron._optimizer_transaction import NeuronOptimizerLoadTransaction


class TestNeuronOptimizerLoadTransaction(unittest.TestCase):
    def test_load_transaction_requires_every_expected_optimizer(self) -> None:
        parameters = [nn.Parameter(torch.tensor(float(index))) for index in range(3)]
        optimizers = [torch.optim.SGD([parameter], lr=0.1) for parameter in parameters]
        transaction = NeuronOptimizerLoadTransaction()

        transaction.prepare_for_load(optimizers[:2])
        transaction.mark_optimizer_loaded(optimizers[2])
        transaction.mark_optimizer_loaded(optimizers[0])

        with self.assertRaisesRegex(RuntimeError, "partial Neuron optimizer"):
            transaction.commit_loaded()

        transaction.mark_optimizer_loaded(optimizers[1])
        transaction.commit_loaded()
        self.assertFalse(transaction.optimizer_requires_completion(optimizers[0]))

    def test_cancelled_load_restores_groups_state_and_container_identity(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
        optimizer.state[parameter] = {
            "momentum_buffer": torch.tensor(2.0),
        }
        original_groups = optimizer.param_groups
        original_group = optimizer.param_groups[0]
        original_parameters = original_group["params"]
        original_state = optimizer.state
        transaction = NeuronOptimizerLoadTransaction()

        transaction.prepare_for_load([optimizer])
        optimizer.param_groups = []
        optimizer.state = {}
        transaction.clear()

        self.assertIs(optimizer.param_groups, original_groups)
        self.assertIs(optimizer.param_groups[0], original_group)
        self.assertIs(optimizer.param_groups[0]["params"], original_parameters)
        self.assertIs(optimizer.state, original_state)
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"],
            torch.tensor(2.0),
        )


if __name__ == "__main__":
    unittest.main()
