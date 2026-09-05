import copy
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch

from emperor.layers import (
    ActivationOptions,
    AttentionResidualConfig,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStack,
    LayerStackConfig,
    ResidualConfig,
)
from emperor.layers._composition.residual.base import ResidualStackRequirements
from emperor.layers._stack.shared_controllers import LayerStackSharedControllers
from emperor.layers._stack.validation import LayerStackValidator
from emperor.linears import LinearLayerConfig


def make_config(**overrides) -> LayerStackConfig:
    values = {
        "input_dim": 3,
        "hidden_dim": 3,
        "output_dim": 3,
        "num_layers": 1,
        "apply_output_postprocessing_flag": False,
        "last_layer_bias_option": LastLayerBiasOptions.DEFAULT,
        "shared_gate_config": None,
        "shared_halting_config": None,
        "shared_memory_config": None,
        "layer_config": LayerConfig(),
    }
    values.update(overrides)
    return LayerStackConfig(**values)


def attention_residual_layer_config() -> LayerConfig:
    return LayerConfig(
        activation=ActivationOptions.DISABLED,
        residual_config=AttentionResidualConfig(
            block_size=1,
            rms_norm_epsilon=1e-6,
        ),
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=LinearLayerConfig(bias_flag=True),
    )


class TestLayerStackValidatorAdapter(unittest.TestCase):
    def test_pure_config_validation_does_not_construct_or_mutate(self):
        config = make_config(num_layers=0)
        original = copy.deepcopy(config)
        random_state = torch.get_rng_state().clone()
        with patch.object(
            LayerStack, "__init__", side_effect=AssertionError("stack constructed")
        ):
            with self.assertRaisesRegex(ValueError, "num_layers"):
                LayerStackValidator.validate_config(config)
        self.assertEqual(config, original)
        torch.testing.assert_close(torch.get_rng_state(), random_state)

    def test_attention_residual_requires_the_final_output_postprocessing(self):
        with self.assertRaisesRegex(
            ValueError,
            "apply_output_postprocessing_flag must be True",
        ):
            LayerStack(
                make_config(
                    num_layers=2,
                    apply_output_postprocessing_flag=False,
                    layer_config=attention_residual_layer_config(),
                )
            )

    def test_attention_residual_requires_one_stable_stack_dimension(self):
        with self.assertRaisesRegex(
            ValueError,
            "input_dim, hidden_dim, and output_dim must all be equal",
        ):
            LayerStack(
                make_config(
                    input_dim=2,
                    hidden_dim=3,
                    output_dim=3,
                    apply_output_postprocessing_flag=True,
                    layer_config=attention_residual_layer_config(),
                )
            )

    def test_stack_requirements_are_read_from_the_registered_residual_owner(self):
        class SyntheticResidualOwner:
            STACK_REQUIREMENTS = ResidualStackRequirements(
                requires_output_postprocessing=True,
            )

        @dataclass
        class SyntheticResidualConfig(ResidualConfig):
            def _registry_owner(self) -> type:
                return SyntheticResidualOwner

        with self.assertRaisesRegex(
            ValueError,
            "apply_output_postprocessing_flag must be True when "
            "SyntheticResidualConfig is enabled",
        ):
            LayerStack(
                make_config(
                    apply_output_postprocessing_flag=False,
                    layer_config=LayerConfig(
                        residual_config=SyntheticResidualConfig(),
                    ),
                )
            )

    def test_module_exposes_validator_adapter(self):
        self.assertIs(LayerStack.VALIDATOR, LayerStackValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(LayerStackValidator):
            @classmethod
            def _validate_gate_config(cls, cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerStack(LayerStack):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerStack(make_config())

    def test_shared_model_validation_dispatches_through_shared_controllers(self):
        class TrackingValidator(LayerStackValidator):
            @staticmethod
            def validate_shared_gate_model(model):
                raise RuntimeError("substituted shared gate validator was called")

            @staticmethod
            def validate_shared_halting_model(model):
                raise RuntimeError("substituted shared halting validator was called")

            @staticmethod
            def validate_shared_memory_model(model):
                raise RuntimeError("substituted shared memory validator was called")

        class TrackingSharedControllers(LayerStackSharedControllers):
            VALIDATOR = TrackingValidator

            def _build_from_config(self, config, **kwargs):
                return object()

        cases = (
            (
                "shared_gate_config",
                "substituted shared gate validator was called",
            ),
            (
                "shared_halting_config",
                "substituted shared halting validator was called",
            ),
            (
                "shared_memory_config",
                "substituted shared memory validator was called",
            ),
        )

        for config_name, expected_message in cases:
            with self.subTest(config_name=config_name):
                config = make_config(layer_config=attention_residual_layer_config())
                setattr(config, config_name, object())
                with self.assertRaisesRegex(RuntimeError, expected_message):
                    TrackingSharedControllers(config).bind([])

    def test_missing_shared_configs_do_not_build_controllers(self):
        class RejectingSharedControllers(LayerStackSharedControllers):
            def _build_from_config(self, config, **kwargs):
                raise AssertionError("inactive shared controller was built")

        RejectingSharedControllers(make_config()).bind([])

    def test_shared_model_validators_reject_invalid_built_models(self):
        cases = (
            (
                LayerStackValidator.validate_shared_gate_model,
                "shared_gate_config must build a LayerGate.",
            ),
            (
                LayerStackValidator.validate_shared_halting_model,
                "shared_halting_config must build a model implementing "
                "HaltingInterface.",
            ),
            (
                LayerStackValidator.validate_shared_memory_model,
                "shared_memory_config must build a model implementing MemoryInterface.",
            ),
        )

        for validate_model, expected_message in cases:
            with self.subTest(validator=validate_model.__name__):
                with self.assertRaises(TypeError) as raised:
                    validate_model(object())
                self.assertEqual(str(raised.exception), expected_message)

    def test_dimension_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "num_layers must be greater than 0, received 0",
        ):
            LayerStack(make_config(num_layers=0))


if __name__ == "__main__":
    unittest.main()
