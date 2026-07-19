import unittest
from dataclasses import fields

from emperor.halting._config import SoftHaltingConfig, StickBreakingConfig
from emperor.halting._validation import StickBreakingValidator


class HaltingDropoutOptionTests(unittest.TestCase):
    def test_configs_expose_only_the_shared_dropout_probability_name(self) -> None:
        for config_type in (StickBreakingConfig, SoftHaltingConfig):
            with self.subTest(config=config_type.__name__):
                field_names = {field.name for field in fields(config_type)}
                self.assertIn("dropout_probability", field_names)
                self.assertNotIn("halting_dropout", field_names)

    def test_validator_reports_the_shared_dropout_probability_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "^dropout_probability must be"):
            StickBreakingValidator._validate_dropout_probability(1.5)


if __name__ == "__main__":
    unittest.main()
