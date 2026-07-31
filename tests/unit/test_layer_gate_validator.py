import unittest

import torch

from emperor.layers import (
    AttentionResidualConfig,
    GateConfig,
    LayerGateOptions,
    WeightedBlendResidualConfig,
    WeightedResidualConfig,
)
from emperor.layers._composition.gate import LayerGate
from emperor.layers._composition.residual.validation import (
    ResidualConnectionValidator,
)
from emperor.layers._composition.residual.variants.additive import AdditiveResidual
from emperor.layers._composition.residual.variants.attention import AttentionResidual
from emperor.layers._composition.residual.variants.weighted import WeightedResidual
from emperor.layers._composition.residual.variants.weighted_blend import (
    WeightedBlendResidual,
)
from emperor.layers._validation import LayerGateValidator
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
    def test_shared_validator_lives_in_dedicated_validation_module(self):
        self.assertEqual(
            ResidualConnectionValidator.__module__,
            "emperor.layers._composition.residual.validation.common",
        )

    def test_each_runtime_exposes_the_shared_validator_adapter(self):
        for runtime_type in (
            AdditiveResidual,
            WeightedResidual,
            WeightedBlendResidual,
            AttentionResidual,
        ):
            with self.subTest(runtime_type=runtime_type.__name__):
                self.assertIs(runtime_type.VALIDATOR, ResidualConnectionValidator)

    def test_successful_runtime_validations_are_check_only(self):
        weighted_connection = WeightedResidualConfig().build()
        attention_connection = AttentionResidualConfig(residual_dim=2).build()
        state = attention_connection.new_state(torch.ones(1, 2))
        validator = attention_connection.VALIDATOR

        results = (
            validator.validate_raw_mix_coefficient(
                weighted_connection.raw_weight,
            ),
            validator.validate_attention_state(state, block_size=1),
        )

        self.assertTupleEqual(results, (None, None))

    def test_runtime_rejects_non_residual_config(self):
        connection = AdditiveResidual.__new__(AdditiveResidual)
        torch.nn.Module.__init__(connection)
        connection.cfg = object()

        with self.assertRaisesRegex(
            TypeError,
            "residual connection cfg must be a ResidualConfig",
        ):
            ResidualConnectionValidator.validate(connection)

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(ResidualConnectionValidator):
            @classmethod
            def _validate_weighted_config(cls, config):
                raise RuntimeError("substituted residual validator was called")

        class TrackingWeightedBlendResidual(WeightedBlendResidual):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted residual validator was called",
        ):
            TrackingWeightedBlendResidual(
                WeightedBlendResidualConfig(
                    residual_dim=3,
                    model_config=LinearLayerConfig(bias_flag=True),
                )
            )

    def test_missing_coefficient_dispatches_through_substituted_validator(self):
        class RejectingValidator(ResidualConnectionValidator):
            @staticmethod
            def validate_raw_mix_coefficient(raw_mix_coefficient):
                raise RuntimeError("substituted coefficient validator was called")

        class RejectingWeightedResidual(WeightedResidual):
            VALIDATOR = RejectingValidator

        connection = RejectingWeightedResidual(WeightedResidualConfig())
        connection.raw_weight = None
        hidden = torch.ones(1, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted coefficient validator was called",
        ):
            connection(hidden, hidden)

    def test_missing_coefficient_error_contract_is_preserved(self):
        connection = WeightedResidualConfig().build()
        connection.raw_weight = None
        hidden = torch.ones(1, 2)

        with self.assertRaisesRegex(
            RuntimeError,
            "weighted residual requires either raw_weight or a coefficient model",
        ):
            connection(hidden, hidden)


if __name__ == "__main__":
    unittest.main()
