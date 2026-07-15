import unittest

import torch
from emperor.base.layer.gate._validator import LayerGateValidator
from emperor.base.layer.gate.config import GateConfig
from emperor.base.layer.gate.model import LayerGate
from emperor.base.layer.gate.options import LayerGateOptions


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


if __name__ == "__main__":
    unittest.main()
