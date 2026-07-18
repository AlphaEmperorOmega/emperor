from __future__ import annotations

import unittest
from dataclasses import fields

from emperor.config import ModelConfig


class ConfigFoundationTests(unittest.TestCase):
    def test_model_config_excludes_legacy_gather_frequency_setting(self) -> None:
        self.assertNotIn(
            "gather_frequency_flag",
            {config_field.name for config_field in fields(ModelConfig)},
        )


if __name__ == "__main__":
    unittest.main()
