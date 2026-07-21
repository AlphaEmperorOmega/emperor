import unittest

from torch import Tensor

from emperor.halting import (
    HaltingBase,
    HaltingInterface,
    HaltingStateBase,
    SoftHalting,
    StickBreaking,
)


class _IncompleteHalting(HaltingBase[HaltingStateBase]):
    def update_halting_state(
        self,
        previous_state: HaltingStateBase | None,
        model_hidden_state: Tensor,
    ) -> tuple[HaltingStateBase, Tensor]:
        if previous_state is None:
            raise ValueError("test adapter requires an existing state")
        return previous_state, model_hidden_state


class _CompleteHalting(_IncompleteHalting):
    def finalize_weighted_accumulation(
        self,
        state: HaltingStateBase,
        current_hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return current_hidden, current_hidden.new_zeros(())


class HaltingStrategyInterfaceTests(unittest.TestCase):
    def test_interface_contains_only_the_stick_breaking_owner_lifecycle(self) -> None:
        interface_methods = {
            name
            for name, value in vars(HaltingInterface).items()
            if not name.startswith("_") and callable(value)
        }

        self.assertEqual(
            interface_methods,
            {
                "update_halting_state",
                "finalize_weighted_accumulation",
            },
        )

    def test_every_halting_strategy_inherits_the_shared_interface(self) -> None:
        self.assertTrue(issubclass(HaltingBase, HaltingInterface))
        self.assertTrue(issubclass(StickBreaking, HaltingInterface))
        self.assertTrue(issubclass(SoftHalting, HaltingInterface))

    def test_only_complete_adapters_are_supported_by_owners(self) -> None:
        self.assertTrue(StickBreaking.implements_halting_interface())
        self.assertTrue(_CompleteHalting.implements_halting_interface())
        self.assertFalse(_IncompleteHalting.implements_halting_interface())
        self.assertFalse(SoftHalting.implements_halting_interface())


if __name__ == "__main__":
    unittest.main()
