import unittest

from torch import Tensor, nn

from emperor.halting import HaltingBase, HaltingMonitorCallback, HaltingStateBase
from emperor.halting._monitoring.tracking import HaltingUsageTrackerManager


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


class _HaltingOwner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.incomplete = _IncompleteHalting()
        self.complete = _CompleteHalting()


class HaltingMonitorSupportTests(unittest.TestCase):
    def test_manager_rejects_an_incomplete_adapter(self) -> None:
        with self.assertRaisesRegex(TypeError, "does not implement"):
            HaltingUsageTrackerManager().attach(_IncompleteHalting())

    def test_callback_tracks_only_complete_adapters(self) -> None:
        owner = _HaltingOwner()
        callback = HaltingMonitorCallback()

        callback.setup(None, owner, "fit")

        self.assertEqual(
            [(name, type(module)) for name, module in callback._halting_layers],
            [("complete", _CompleteHalting)],
        )
        self.assertFalse(hasattr(owner.incomplete, "_usage_tracker"))
        self.assertTrue(hasattr(owner.complete, "_usage_tracker"))

        callback.on_fit_end(None, owner)
        self.assertFalse(hasattr(owner.complete, "_usage_tracker"))


if __name__ == "__main__":
    unittest.main()
