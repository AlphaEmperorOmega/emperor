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

    def test_unregistered_optimizer_parameter_is_rejected(self) -> None:
        module = self._module()
        external_parameter = nn.Parameter(torch.tensor(4.0))
        optimizer = torch.optim.SGD([external_parameter], lr=0.1)

        with self.assertRaisesRegex(RuntimeError, "must be registered"):
            NeuronOptimizerNamedLayout.capture(
                module,
                [optimizer],
                [optimizer.state_dict()],
            )

    def test_retired_append_layout_is_rejected(self) -> None:
        module = self._module()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        saved_state = optimizer.state_dict()
        layout = NeuronOptimizerNamedLayout.capture(
            module,
            [optimizer],
            [saved_state],
        )
        layout["optimizers"][0].update(
            {
                "sync_policy": "legacy_append",
                "legacy_base_group_count": 1,
                "legacy_reference_group_index": 0,
            }
        )

        with self.assertRaisesRegex(RuntimeError, "Invalid named Neuron optimizer"):
            NeuronOptimizerNamedLayout().prepare_for_load(
                module,
                [optimizer],
                [saved_state],
                layout,
            )

    def test_saved_metadata_is_validated_before_live_membership(self) -> None:
        module = self._module()
        source_optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        saved_state = source_optimizer.state_dict()
        layout = NeuronOptimizerNamedLayout.capture(
            module, [source_optimizer], [saved_state]
        )
        target_optimizer = torch.optim.SGD([module["b"], module["a"]], lr=0.1)
        original_group = target_optimizer.param_groups[0]
        original_parameter_list = original_group["params"]
        saved_parameter_ids = saved_state["param_groups"][0]["params"]
        saved_state["param_groups"][0]["param_names"] = ["a"]
        manager = NeuronOptimizerNamedLayout()

        with self.assertRaisesRegex(RuntimeError, "param_names metadata"):
            manager.prepare_for_load(module, [target_optimizer], [saved_state], layout)

        del saved_state["param_groups"][0]["param_names"]
        with self.assertRaisesRegex(RuntimeError, "parameter membership differs"):
            manager.prepare_for_load(module, [target_optimizer], [saved_state], layout)

        self.assertIs(target_optimizer.param_groups[0], original_group)
        self.assertIs(original_group["params"], original_parameter_list)
        self.assertIs(saved_state["param_groups"][0]["params"], saved_parameter_ids)
        self.assertEqual(saved_parameter_ids, [0, 1, 2])
        self.assertFalse(manager.optimizer_requires_completion(target_optimizer))
