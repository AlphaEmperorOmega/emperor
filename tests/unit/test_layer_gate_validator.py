import unittest

import torch

from emperor.layers import (
    GateConfig,
    LayerGateOptions,
    ResidualConfig,
    ResidualConnection,
    ResidualConnectionOptions,
)
from emperor.layers._composition.gate import LayerGate
from emperor.layers._validation import (
    LayerGateValidator,
    ResidualConnectionValidator,
)
from emperor.linears import LinearLayerConfig


class TestLayerGateValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(LayerGate.VALIDATOR, LayerGateValidator)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(LayerGateValidator):
            @classmethod
            def _validate_dimensions(cls, cfg):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerGate(LayerGate):
            VALIDATOR = TrackingValidator

        cfg = GateConfig(
            gate_dim=3,
            option=LayerGateOptions.MULTIPLIER,
            activation=None,
            model_config=object(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerGate(cfg)

    def test_runtime_dispatches_through_substituted_validator(self):
        class RejectingValidator(LayerGateValidator):
            @staticmethod
            def validate_gate_model(model):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingLayerGate(LayerGate):
            VALIDATOR = RejectingValidator

        model = RejectingLayerGate.__new__(RejectingLayerGate)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Identity()
        model.option = LayerGateOptions.MULTIPLIER

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model(torch.ones(1, 3))

    def test_gate_dimension_error_contract_is_preserved(self):
        cfg = GateConfig(
            gate_dim=0,
            option=LayerGateOptions.MULTIPLIER,
            activation=None,
            model_config=object(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "gate_dim must be greater than 0, received 0",
        ):
            LayerGate(cfg)


class TestResidualConnectionValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(ResidualConnection.VALIDATOR, ResidualConnectionValidator)

    def test_successful_runtime_validations_are_check_only(self):
        weighted_connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.WEIGHTED_RESIDUAL)
        )
        attention_connection = ResidualConnection(
            ResidualConfig(
                option=ResidualConnectionOptions.ATTENTION_RESIDUAL,
                residual_dim=2,
            )
        )
        state = attention_connection.new_state(torch.ones(1, 2))
        validator = attention_connection.VALIDATOR

        results = (
            validator.validate_raw_mix_coefficient(
                weighted_connection.raw_weight,
                weighted_connection.option,
            ),
            validator.validate_attention_residual_available(
                attention_connection.attention_residual,
            ),
            validator.validate_attention_residual_state(state),
        )

        self.assertTupleEqual(results, (None, None, None))

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(ResidualConnectionValidator):
            @staticmethod
            def _validate_data_dependent_residual_dim(residual_dim):
                raise RuntimeError("substituted residual validator was called")

        class TrackingResidualConnection(ResidualConnection):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted residual validator was called",
        ):
            TrackingResidualConnection(
                ResidualConfig(
                    option=ResidualConnectionOptions.WEIGHTED_BLEND,
                    residual_dim=3,
                    model_config=LinearLayerConfig(bias_flag=True),
                ),
            )

    def test_new_state_dispatches_through_substituted_validator(self):
        class RejectingValidator(ResidualConnectionValidator):
            @staticmethod
            def validate_attention_residual_available(attention_residual):
                raise RuntimeError("substituted state factory validator was called")

        class RejectingResidualConnection(ResidualConnection):
            VALIDATOR = RejectingValidator

        connection = RejectingResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL)
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted state factory validator was called",
        ):
            connection.new_state(torch.ones(1, 2))

    def test_attention_forward_dispatches_through_substituted_validator(self):
        class RejectingValidator(ResidualConnectionValidator):
            @staticmethod
            def validate_attention_residual_state(residual_state):
                raise RuntimeError("substituted forward validator was called")

        class RejectingResidualConnection(ResidualConnection):
            VALIDATOR = RejectingValidator

        connection = RejectingResidualConnection(
            ResidualConfig(
                option=ResidualConnectionOptions.ATTENTION_RESIDUAL,
                residual_dim=2,
            )
        )
        hidden = torch.ones(1, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted forward validator was called",
        ):
            connection(hidden, hidden)

    def test_defensive_mixing_option_error_dispatches_through_validator(self):
        class RejectingValidator(ResidualConnectionValidator):
            @classmethod
            def validate(cls, model):
                pass

            @staticmethod
            def reject_unsupported_mixing_coefficient_option(option):
                raise RuntimeError("substituted mixing validator was called")

        class RejectingResidualConnection(ResidualConnection):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted mixing validator was called",
        ):
            RejectingResidualConnection(ResidualConfig(option="UNSUPPORTED"))

    def test_defensive_mixing_option_error_contract_is_preserved(self):
        class PermissiveValidator(ResidualConnectionValidator):
            @classmethod
            def validate(cls, model):
                pass

        class PermissiveResidualConnection(ResidualConnection):
            VALIDATOR = PermissiveValidator

        with self.assertRaisesRegex(
            ValueError,
            "Residual option does not use mixing coefficients: UNSUPPORTED",
        ):
            PermissiveResidualConnection(ResidualConfig(option="UNSUPPORTED"))

    def test_defensive_runtime_option_error_dispatches_through_validator(self):
        class RejectingValidator(ResidualConnectionValidator):
            @staticmethod
            def reject_unsupported_runtime_option(option):
                raise RuntimeError("substituted runtime option validator was called")

        class RejectingResidualConnection(ResidualConnection):
            VALIDATOR = RejectingValidator

        connection = RejectingResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL)
        )
        connection.option = "UNSUPPORTED"
        hidden = torch.ones(1, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted runtime option validator was called",
        ):
            connection(hidden, hidden)

    def test_defensive_runtime_option_error_contract_is_preserved(self):
        connection = ResidualConnection(
            ResidualConfig(option=ResidualConnectionOptions.RESIDUAL)
        )
        connection.option = "UNSUPPORTED"
        hidden = torch.ones(1, 2)

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported residual connection option UNSUPPORTED for ResidualConnection",
        ):
            connection(hidden, hidden)


if __name__ == "__main__":
    unittest.main()
