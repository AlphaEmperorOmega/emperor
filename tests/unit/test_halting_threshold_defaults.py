import unittest
from unittest.mock import patch

from emperor.halting import HaltingConfig, SoftHaltingConfig, StickBreakingConfig


class _CapturedBuild:
    def __init__(self, config, overrides) -> None:
        self.config = config
        self.overrides = overrides


class HaltingThresholdConfigurationTests(unittest.TestCase):
    def test_each_strategy_leaves_an_omitted_threshold_unresolved(self) -> None:
        for config_type in (StickBreakingConfig, SoftHaltingConfig):
            with self.subTest(config=config_type.__name__):
                config = config_type()
                with patch.object(
                    config_type,
                    "_registry_owner",
                    return_value=_CapturedBuild,
                ):
                    built = config.build()

                self.assertIsNone(built.overrides)
                self.assertIsNone(config.threshold)
                self.assertFalse(hasattr(config, "DEFAULT_THRESHOLD"))

    def test_threshold_help_recommends_an_explicit_value(self) -> None:
        help_text = HaltingConfig.__dataclass_fields__["threshold"].metadata["help"]

        self.assertIn("Set this to 0.999", help_text)

    def test_explicit_override_wins_without_being_mutated(self) -> None:
        for config_type in (StickBreakingConfig, SoftHaltingConfig):
            with self.subTest(config=config_type.__name__):
                config = config_type(threshold=0.9)
                overrides = config_type(threshold=0.8)
                with patch.object(
                    config_type,
                    "_registry_owner",
                    return_value=_CapturedBuild,
                ):
                    built = config.build(overrides)

                self.assertIs(built.config, config)
                self.assertIs(built.overrides, overrides)
                self.assertEqual(config.threshold, 0.9)
                self.assertEqual(overrides.threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
