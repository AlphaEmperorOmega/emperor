import unittest

import torch
from torch import nn

from emperor.neuron._optimizer_layout import NeuronOptimizerNamedLayout


class TestNeuronOptimizerNamedLayout(unittest.TestCase):
    @staticmethod
    def _module() -> nn.ParameterDict:
        return nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(index)))
                for index, name in enumerate(("a", "b", "c"), start=1)
            }
        )

    def test_saved_state_is_reordered_to_live_parameter_identity(self) -> None:
        module = self._module()
        source_optimizer = torch.optim.SGD(
            module.parameters(),
            lr=0.1,
            momentum=0.9,
        )
        for sentinel, parameter in enumerate(module.parameters(), start=11):
            source_optimizer.state[parameter] = {
                "momentum_buffer": torch.full_like(parameter, float(sentinel))
            }
        saved_state = source_optimizer.state_dict()
        original_saved_ids = tuple(saved_state["param_groups"][0]["params"])
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            [source_optimizer],
            [saved_state],
            {},
        )
        target_optimizer = torch.optim.SGD(
            [module["c"], module["b"], module["a"]],
            lr=0.1,
            momentum=0.9,
        )
        manager = NeuronOptimizerNamedLayout()

        manager.prepare_for_load(
            module,
            [target_optimizer],
            [saved_state],
            layout,
        )
        self.assertEqual(saved_state["param_groups"][0]["params"], [2, 1, 0])
        target_optimizer.load_state_dict(saved_state)
        manager.complete_optimizer_load(target_optimizer)

        self.assertEqual(
            tuple(saved_state["param_groups"][0]["params"]),
            original_saved_ids,
        )
        for name, sentinel in (("a", 11), ("b", 12), ("c", 13)):
            torch.testing.assert_close(
                target_optimizer.state[module[name]]["momentum_buffer"],
                torch.tensor(float(sentinel)),
            )

    def test_cancelled_load_restores_saved_ids_and_live_groups(self) -> None:
        module = self._module()
        source_optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        saved_state = source_optimizer.state_dict()
        original_saved_ids = tuple(saved_state["param_groups"][0]["params"])
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            [source_optimizer],
            [saved_state],
            {},
        )
        target_optimizer = torch.optim.SGD(
            [module["c"], module["b"], module["a"]],
            lr=0.1,
        )
        original_groups = list(target_optimizer.param_groups)
        manager = NeuronOptimizerNamedLayout()

        manager.prepare_for_load(
            module,
            [target_optimizer],
            [saved_state],
            layout,
        )
        manager.clear()

        self.assertEqual(
            tuple(saved_state["param_groups"][0]["params"]),
            original_saved_ids,
        )
        self.assertEqual(target_optimizer.param_groups, original_groups)
        self.assertFalse(manager.optimizer_requires_completion(target_optimizer))

    def test_unregistered_optimizer_parameter_defers_to_native_layout(self) -> None:
        module = self._module()
        external_parameter = nn.Parameter(torch.tensor(4.0))
        optimizer = torch.optim.SGD([external_parameter], lr=0.1)

        layout = NeuronOptimizerNamedLayout.capture_if_supported(
            module,
            [optimizer],
            [optimizer.state_dict()],
            {},
        )

        self.assertIsNone(layout)
