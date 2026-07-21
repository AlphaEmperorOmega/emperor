import math
import unittest

import torch

from emperor.halting import (
    HaltingBase,
    HaltingHiddenStateModeOptions,
    SoftHalting,
    SoftHaltingConfig,
    StickBreaking,
    StickBreakingConfig,
)
from emperor.halting._validation import SoftHaltingValidator, StickBreakingValidator
from emperor.layers import (
    ActivationOptions,
    GateConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerGateOptions,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def gate_config(input_dim: int = 4) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=input_dim,
        output_dim=2,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DISABLED,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def config(
    config_type: type[StickBreakingConfig] | type[SoftHaltingConfig] = (
        StickBreakingConfig
    ),
    **overrides,
) -> StickBreakingConfig | SoftHaltingConfig:
    values = {
        "input_dim": 4,
        "threshold": 0.99,
        "dropout_probability": None,
        "hidden_state_mode": HaltingHiddenStateModeOptions.RAW,
        "halting_gate_config": gate_config(),
    }
    values.update(overrides)
    return config_type(**values)


class HaltingValidatorTests(unittest.TestCase):
    def test_each_strategy_uses_its_public_config_contract(self) -> None:
        self.assertIs(HaltingBase.VALIDATOR, StickBreakingValidator)
        self.assertIs(StickBreaking.VALIDATOR, StickBreakingValidator)
        self.assertIs(SoftHalting.VALIDATOR, SoftHaltingValidator)

    def test_valid_boundary_configuration_builds_both_strategies(self) -> None:
        for config_type, strategy_type in (
            (StickBreakingConfig, StickBreaking),
            (SoftHaltingConfig, SoftHalting),
        ):
            with self.subTest(strategy=strategy_type.__name__):
                model = strategy_type(
                    config(
                        config_type,
                        input_dim=1,
                        threshold=1.0,
                        dropout_probability=1.0,
                        hidden_state_mode=(HaltingHiddenStateModeOptions.ACCUMULATED),
                        halting_gate_config=gate_config(1),
                    )
                )
                self.assertEqual(model.input_dim, 1)
                self.assertEqual(model.threshold, 1.0)

    def test_required_fields_report_the_exact_field_and_config_type(self) -> None:
        for field_name in (
            "input_dim",
            "hidden_state_mode",
            "halting_gate_config",
        ):
            with self.subTest(field_name=field_name):
                cfg = config()
                setattr(cfg, field_name, None)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^{field_name} is required for StickBreakingConfig, "
                    r"received None$",
                ):
                    StickBreaking(cfg)

    def test_config_build_resolves_to_each_strategy_default(self) -> None:
        stick = config(threshold=None).build()
        soft = config(SoftHaltingConfig, threshold=None).build()

        self.assertEqual(stick.threshold, 0.999)
        self.assertEqual(soft.threshold, 0.999)

    def test_direct_stick_constructor_retains_required_threshold_contract(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"^threshold is required for StickBreakingConfig, received None$",
        ):
            StickBreaking(config(threshold=None))

    def test_legacy_pad_mask_validation_contract_is_preserved(self) -> None:
        hidden = torch.zeros(2, 3, 4)

        StickBreakingValidator.validate_pad_mask(None, hidden)
        StickBreakingValidator.validate_pad_mask(
            torch.tensor(
                ((True, False, True), (False, True, False)),
            ),
            hidden,
        )
        StickBreakingValidator.validate_pad_mask(
            torch.tensor(
                ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
            ),
            hidden,
        )

        with self.assertRaisesRegex(TypeError, "required_by_test"):
            StickBreakingValidator.validate_pad_mask(
                None,
                hidden,
                required_by="required_by_test",
            )
        with self.assertRaisesRegex(TypeError, "must be a Tensor or None"):
            StickBreakingValidator.validate_pad_mask([[True]], hidden)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            StickBreakingValidator.validate_pad_mask(torch.ones(2, 2), hidden)
        with self.assertRaisesRegex(TypeError, "must use bool or floating dtype"):
            StickBreakingValidator.validate_pad_mask(
                torch.ones(2, 3, dtype=torch.int64),
                hidden,
            )
        for invalid_value in (float("nan"), -0.1, 1.1):
            with self.subTest(invalid_value=invalid_value):
                invalid_mask = torch.zeros(2, 3)
                invalid_mask[0, 0] = invalid_value
                with self.assertRaisesRegex(
                    ValueError,
                    "must be finite and between 0.0 and 1.0",
                ):
                    StickBreakingValidator.validate_pad_mask(invalid_mask, hidden)

    def test_soft_canonical_gate_does_not_require_a_gate_config(self) -> None:
        model = SoftHalting(config(SoftHaltingConfig, halting_gate_config=None))

        self.assertIsNone(model.halting_gate_config)

    def test_soft_validates_dropout_for_the_canonical_gate(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout_probability must be finite and between 0.0 and 1.0",
        ):
            SoftHalting(config(SoftHaltingConfig, dropout_probability=-0.1))

    def test_soft_custom_gate_rejects_a_nested_layer_gate(self) -> None:
        cfg = config(SoftHaltingConfig)
        cfg.halting_gate_config.layer_config.gate_config = GateConfig(
            gate_dim=4,
            option=LayerGateOptions.MULTIPLIER,
            activation=ActivationOptions.DISABLED,
            model_config=gate_config(),
        )

        with self.assertRaisesRegex(
            ValueError,
            r"^halting_gate_config.layer_config.gate_config must be None, "
            r"nested gates are not allowed in halting$",
        ):
            SoftHalting(cfg)

    def test_input_dim_rejects_non_integer_and_non_positive_values(self) -> None:
        for value in (True, 1.5, "4"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^input_dim must be a positive integer, received "
                    rf"{type(value).__name__}$",
                ):
                    StickBreaking(config(input_dim=value))
        with self.assertRaisesRegex(
            ValueError,
            r"^input_dim must be greater than 0, received 0$",
        ):
            StickBreakingValidator._validate_input_dim(0)
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^input_dim must be greater than 0, received {value}$",
                ):
                    StickBreaking(config(input_dim=value))

    def test_threshold_rejects_non_numeric_non_finite_and_out_of_range(self) -> None:
        for value in (True, "0.5"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^threshold must be a number, received "
                    rf"{type(value).__name__}$",
                ):
                    StickBreaking(config(threshold=value))
        for value in (math.nan, math.inf, -math.inf, 0.0, -0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^threshold must be finite and between 0.0 \(exclusive\) "
                    r"and 1.0 \(inclusive\), received ",
                ):
                    StickBreaking(config(threshold=value))

    def test_optional_dropout_probability_must_be_finite(self) -> None:
        for value in (None, 0, 0.5, 1.0):
            with self.subTest(value=value):
                self.assertIsInstance(
                    StickBreaking(config(dropout_probability=value)),
                    StickBreaking,
                )
        for value in (True, "0.5"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^dropout_probability must be a number or None, received "
                    rf"{type(value).__name__}$",
                ):
                    StickBreaking(config(dropout_probability=value))
        for value in (math.nan, math.inf, -0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^dropout_probability must be finite and between 0.0 and 1.0, "
                    r"received ",
                ):
                    StickBreaking(config(dropout_probability=value))

    def test_hidden_state_mode_requires_the_public_enum(self) -> None:
        for value in (0, "RAW"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^hidden_state_mode must be a "
                    rf"HaltingHiddenStateModeOptions value, received "
                    rf"{type(value).__name__}$",
                ):
                    StickBreaking(config(hidden_state_mode=value))

    def test_gate_config_type_and_output_contract_have_exact_errors(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"^halting_gate_config must be an instance of LayerStackConfig, "
            r"got object$",
        ):
            StickBreaking(config(halting_gate_config=object()))

        cfg = config()
        cfg.halting_gate_config.output_dim = 3
        with self.assertRaisesRegex(
            ValueError,
            r"^halting_gate_config.output_dim must be 2 "
            r"\(continuation and halting logits\), received 3$",
        ):
            StickBreaking(cfg)

    def test_gate_config_rejects_bias_and_nested_controllers(self) -> None:
        invalid_cases = (
            (
                "last_layer_bias_option",
                lambda cfg: setattr(
                    cfg.halting_gate_config,
                    "last_layer_bias_option",
                    LastLayerBiasOptions.DEFAULT,
                ),
                (
                    "halting_gate_config.last_layer_bias_option must be DISABLED, "
                    "received LastLayerBiasOptions.DEFAULT"
                ),
            ),
            (
                "shared_halting_config",
                lambda cfg: setattr(
                    cfg.halting_gate_config,
                    "shared_halting_config",
                    config(),
                ),
                (
                    "halting_gate_config.shared_halting_config must be None, "
                    "nested halting is not allowed"
                ),
            ),
            (
                "shared_gate_config",
                lambda cfg: setattr(
                    cfg.halting_gate_config,
                    "shared_gate_config",
                    GateConfig(
                        gate_dim=4,
                        option=LayerGateOptions.MULTIPLIER,
                        activation=ActivationOptions.DISABLED,
                        model_config=gate_config(),
                    ),
                ),
                (
                    "halting_gate_config.shared_gate_config must be inactive, "
                    "nested gates are not allowed in halting"
                ),
            ),
            (
                "layer_gate_config",
                lambda cfg: setattr(
                    cfg.halting_gate_config.layer_config,
                    "gate_config",
                    GateConfig(
                        gate_dim=4,
                        option=LayerGateOptions.MULTIPLIER,
                        activation=ActivationOptions.DISABLED,
                        model_config=gate_config(),
                    ),
                ),
                (
                    "halting_gate_config.layer_config.gate_config must be None, "
                    "nested gates are not allowed in halting"
                ),
            ),
            (
                "layer_halting_config",
                lambda cfg: setattr(
                    cfg.halting_gate_config.layer_config,
                    "halting_config",
                    config(),
                ),
                (
                    "halting_gate_config.layer_config.halting_config must be None, "
                    "nested halting is not allowed"
                ),
            ),
        )
        for name, mutate, error in invalid_cases:
            with self.subTest(name=name):
                cfg = config()
                mutate(cfg)
                with self.assertRaisesRegex(ValueError, f"^{error}$"):
                    StickBreaking(cfg)

    def test_gate_layer_config_requires_layer_config_or_none(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            r"^halting_gate_config.layer_config must be a LayerConfig or None, "
            r"got object$",
        ):
            StickBreakingValidator._validate_halting_gate_layer_config(object())

        StickBreakingValidator._validate_halting_gate_layer_config(None)

    def test_inactive_gate_detection_is_exact(self) -> None:
        self.assertFalse(StickBreakingValidator._is_gate_config_active(None))
        self.assertTrue(
            StickBreakingValidator._is_gate_config_active(
                GateConfig(
                    gate_dim=4,
                    option=LayerGateOptions.ADDITION,
                    activation=ActivationOptions.DISABLED,
                    model_config=gate_config(),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
