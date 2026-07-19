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


if __name__ == "__main__":
    unittest.main()
