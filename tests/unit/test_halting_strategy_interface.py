import unittest
from inspect import signature

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

    def test_soft_and_stick_breaking_use_the_exact_interface_signatures(self) -> None:
        for method_name in (
            "update_halting_state",
            "finalize_weighted_accumulation",
        ):
            expected_parameters = tuple(
                (name, parameter.kind)
                for name, parameter in signature(
                    getattr(HaltingInterface, method_name)
                ).parameters.items()
            )
            with self.subTest(method=method_name, strategy="StickBreaking"):
                self.assertEqual(
                    tuple(
                        (name, parameter.kind)
                        for name, parameter in signature(
                            getattr(StickBreaking, method_name)
                        ).parameters.items()
                    ),
                    expected_parameters,
                )
            with self.subTest(method=method_name, strategy="SoftHalting"):
                self.assertEqual(
                    tuple(
                        (name, parameter.kind)
                        for name, parameter in signature(
                            getattr(SoftHalting, method_name)
                        ).parameters.items()
                    ),
                    expected_parameters,
                )

    def test_only_complete_strategies_are_supported_by_owners(self) -> None:
        self.assertTrue(StickBreaking.implements_halting_interface())
        self.assertTrue(SoftHalting.implements_halting_interface())
        self.assertTrue(_CompleteHalting.implements_halting_interface())
        self.assertFalse(_IncompleteHalting.implements_halting_interface())

    def test_strategies_expose_only_the_official_owner_lifecycle(self) -> None:
        removed_callback_methods = (
            "run_step",
            "finalize",
            "owner_stop_mask",
            "prepare_owner_step",
            "complete_owner_step",
            "gather_owner_step_rows",
            "restrict_owner_step_updates",
        )

        for strategy_type in (HaltingBase, StickBreaking, SoftHalting):
            for method_name in removed_callback_methods:
                with self.subTest(
                    strategy=strategy_type.__name__,
                    method=method_name,
                ):
                    self.assertFalse(hasattr(strategy_type, method_name))


if __name__ == "__main__":
    unittest.main()
