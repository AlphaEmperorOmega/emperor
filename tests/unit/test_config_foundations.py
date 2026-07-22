from __future__ import annotations

import unittest
from dataclasses import dataclass, field, fields

import torch

from emperor.config import BaseOptions, ConfigBase, ModelConfig, optional_field
from emperor.linears import LinearLayer, LinearLayerConfig


@dataclass
class _BehaviorConfig(ConfigBase):
    width: int = field(default=3, metadata={"help": "Configured width."})
    optional_depth: int | None = optional_field("Optional configured depth.")
    enabled: bool = False
    passed_args: object | None = None


class _BuildRecord:
    def __init__(self, cfg: ConfigBase, overrides: ConfigBase | None) -> None:
        self.cfg = cfg
        self.overrides = overrides


class _ModelTypeOwner:
    @classmethod
    def build_from_config(
        cls,
        cfg: ConfigBase,
        overrides: ConfigBase | None,
    ) -> _BuildRecord:
        return _BuildRecord(cfg, overrides)


@dataclass
class _ModelTypeConfig(ConfigBase):
    model_type: str = "tiny"

    def _registry_owner(self) -> type:
        return _ModelTypeOwner


class _ExampleOptions(BaseOptions):
    FIRST_MODE = "first"
    SECOND_MODE = "second"


class ConfigFoundationTests(unittest.TestCase):
    def test_optional_field_has_none_default_and_help_metadata(self) -> None:
        generated_field = optional_field("Generated help.")
        config_fields = {
            config_field.name: config_field for config_field in fields(_BehaviorConfig)
        }

        optional_depth = config_fields["optional_depth"]

        self.assertIsNone(generated_field.default)
        self.assertEqual(dict(generated_field.metadata), {"help": "Generated help."})
        self.assertIsNone(optional_depth.default)
        self.assertEqual(
            dict(optional_depth.metadata),
            {"help": "Optional configured depth."},
        )

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

    def test_custom_parameter_tracking_ignores_none_default_and_passed_args_field(
        self,
    ) -> None:
        cfg = _BehaviorConfig(
            width=8,
            optional_depth=5,
            enabled=True,
            passed_args={"not": "configuration"},
        )

        self.assertEqual(
            cfg.get_custom_parameters(),
            {"width": 8, "enabled": True},
        )

    def test_get_returns_existing_value_and_missing_key_default(self) -> None:
        cfg = _BehaviorConfig(width=7)
        sentinel = object()

        self.assertEqual(cfg.get("width", 99), 7)
        self.assertIs(cfg.get("missing", sentinel), sentinel)
        self.assertIsNone(cfg.get("missing"))

    def test_update_applies_non_none_values_without_flattening_nested_configs(
        self,
    ) -> None:
        nested = _BehaviorConfig(width=11)
        base = ModelConfig(batch_size=2, experiment_config=None)
        overrides = ModelConfig(
            batch_size=5,
            learning_rate=0.2,
            experiment_config=nested,
        )

        returned = base.update(overrides)

        self.assertIs(returned, base)
        self.assertEqual(base.batch_size, 5)
        self.assertEqual(base.learning_rate, 0.2)
        self.assertIs(base.experiment_config, nested)

        base.update(ModelConfig(experiment_config=None))
        self.assertIs(base.experiment_config, nested)

    def test_base_registry_owner_failure_names_required_contract(self) -> None:
        cfg = ConfigBase()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"ConfigBase must implement `_registry_owner` or override `build`",
        ):
            cfg.registry_owner()

    def test_concrete_config_builds_real_registered_emperor_module(self) -> None:
        cfg = LinearLayerConfig(input_dim=2, output_dim=2, bias_flag=False)

        layer = cfg.build()

        self.assertIsInstance(layer, LinearLayer)
        self.assertIs(layer.cfg, cfg)
        with torch.no_grad():
            layer.weight_params.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        output = layer(torch.tensor([[5.0, 6.0]]))
        torch.testing.assert_close(output, torch.tensor([[23.0, 34.0]]))

    def test_model_type_config_dispatches_to_build_from_config(self) -> None:
        cfg = _ModelTypeConfig(model_type="sequence")
        overrides = _ModelTypeConfig(model_type="image")

        built = cfg.build(overrides)

        self.assertIsInstance(built, _BuildRecord)
        self.assertIs(built.cfg, cfg)
        self.assertIs(built.overrides, overrides)
        self.assertIs(cfg.registry_owner(), _ModelTypeOwner)

    def test_base_options_normalize_names_and_reject_unknown_values(self) -> None:
        self.assertEqual(_ExampleOptions.cli_name("SECOND_MODE"), "second-mode")
        self.assertEqual(
            _ExampleOptions.cli_names(),
            ["first-mode", "second-mode"],
        )
        self.assertEqual(
            _ExampleOptions.names(),
            ["FIRST_MODE", "SECOND_MODE"],
        )
        self.assertIs(
            _ExampleOptions.get_member("FIRST_MODE"), _ExampleOptions.FIRST_MODE
        )
        self.assertIs(
            _ExampleOptions.get_member("second-mode"),
            _ExampleOptions.SECOND_MODE,
        )
        self.assertIsNone(_ExampleOptions.get_member(None))
        with self.assertRaisesRegex(
            ValueError,
            r"Option 'missing' does not exist in _ExampleOptions\.",
        ):
            _ExampleOptions.get_member("missing")


if __name__ == "__main__":
    unittest.main()
