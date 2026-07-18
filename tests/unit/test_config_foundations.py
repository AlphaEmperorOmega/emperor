from __future__ import annotations

import unittest
from dataclasses import dataclass, field, fields

from emperor.config import ConfigBase, ModelConfig


@dataclass
class _BehaviorConfig(ConfigBase):
    width: int = field(default=3, metadata={"help": "Configured width."})


class ConfigFoundationTests(unittest.TestCase):
    def test_model_config_fields_use_optional_defaults(self) -> None:
        default = ModelConfig()
        customized = ModelConfig(
            batch_size=4,
            learning_rate=0.25,
            experiment_config=_BehaviorConfig(width=9),
        )

        self.assertEqual(
            (
                default.batch_size,
                default.learning_rate,
                default.sequence_length,
                default.input_dim,
                default.hidden_dim,
                default.output_dim,
                default.experiment_config,
            ),
            (None, None, None, None, None, None, None),
        )
        self.assertEqual(default.get_custom_parameters(), {})
        self.assertEqual(customized.batch_size, 4)
        self.assertEqual(customized.learning_rate, 0.25)
        self.assertEqual(customized.experiment_config, _BehaviorConfig(width=9))

    def test_model_config_excludes_legacy_gather_frequency_setting(self) -> None:
        self.assertNotIn(
            "gather_frequency_flag",
            {config_field.name for config_field in fields(ModelConfig)},
        )


if __name__ == "__main__":
    unittest.main()
