import unittest
from unittest.mock import patch

from emperor.halting import SoftHaltingConfig, StickBreakingConfig


class _CapturedBuild:
    def __init__(self, config, overrides) -> None:
        self.config = config
        self.overrides = overrides


class HaltingThresholdDefaultTests(unittest.TestCase):
    def test_each_strategy_supplies_the_recommended_default_threshold(self) -> None:
        for config_type in (StickBreakingConfig, SoftHaltingConfig):
            with self.subTest(config=config_type.__name__):
                config = config_type()
                with patch.object(
                    config_type,
                    "_registry_owner",
                    return_value=_CapturedBuild,
                ):
                    built = config.build()

                self.assertEqual(built.overrides.threshold, 0.999)
                self.assertIsNone(config.threshold)

    def test_explicit_override_wins_without_being_mutated(self) -> None:
        config = SoftHaltingConfig()
        overrides = SoftHaltingConfig(threshold=0.8)
        with patch.object(
            SoftHaltingConfig,
            "_registry_owner",
            return_value=_CapturedBuild,
        ):
            built = config.build(overrides)

        self.assertIs(built.overrides, overrides)
        self.assertEqual(overrides.threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
