import unittest

from emperor.halting.config import SoftHaltingConfig, StickBreakingConfig
from emperor.halting.core._validator import StickBreakingValidator
from emperor.halting.core.base import HaltingBase
from emperor.halting.core.variants import SoftHalting, StickBreaking
from emperor.halting.options import HaltingHiddenStateModeOptions


def make_config(**overrides) -> StickBreakingConfig:
    values = {
        "input_dim": 4,
        "threshold": 0.99,
        "halting_dropout": None,
        "hidden_state_mode": HaltingHiddenStateModeOptions.RAW,
        "halting_gate_config": object(),
    }
    values.update(overrides)
    return StickBreakingConfig(**values)


class TestHaltingValidatorAdapter(unittest.TestCase):
    def test_halting_modules_share_the_base_owner_adapter(self):
        for module_type in (HaltingBase, StickBreaking, SoftHalting):
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, StickBreakingValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(StickBreakingValidator):
            @classmethod
            def _validate_halting_gate_config(cls, halting_gate_config):
                raise RuntimeError("substituted construction validator was called")

        class TrackingStickBreaking(StickBreaking):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingStickBreaking(make_config())

    def test_threshold_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            r"threshold must be between 0.0 \(exclusive\) and 1.0 "
            r"\(inclusive\), received 0.0",
        ):
            StickBreaking(make_config(threshold=0.0))

    def test_soft_halting_config_uses_the_shared_validator(self):
        cfg = SoftHaltingConfig(
            input_dim=4,
            threshold=0.0,
            halting_dropout=None,
            hidden_state_mode=HaltingHiddenStateModeOptions.RAW,
            halting_gate_config=object(),
        )

        with self.assertRaisesRegex(ValueError, "threshold"):
            SoftHalting(cfg)


if __name__ == "__main__":
    unittest.main()
