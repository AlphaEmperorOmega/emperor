import unittest

from emperor.config import ConfigBase
from emperor.layers import (
    ActivationOptions,
    Layer,
    LayerConfig,
    LayerNormPositionOptions,
)
from emperor.layers._layer.pipeline.halting import LayerHaltingDelegate
from emperor.layers._layer.pipeline.memory import LayerMemoryDelegate
from emperor.layers._layer.pipeline.normalization import LayerNormalizationDelegate
from emperor.layers._layer.pipeline.postprocessing import LayerPostprocessingDelegate
from emperor.layers._layer.pipeline.residual import LayerResidualDelegate
from emperor.layers._layer.validation import (
    LayerHaltingDelegateValidator,
    LayerMemoryDelegateValidator,
    LayerNormalizationDelegateValidator,
    LayerPostprocessingDelegateValidator,
    LayerResidualDelegateValidator,
    LayerValidator,
)


def make_config(**overrides) -> LayerConfig:
    values = {
        "input_dim": 3,
        "output_dim": 3,
        "activation": ActivationOptions.DISABLED,
        "residual_config": None,
        "dropout_probability": 0.0,
        "layer_norm_position": LayerNormPositionOptions.DISABLED,
        "gate_config": None,
        "halting_config": None,
        "memory_config": None,
        "layer_model_config": ConfigBase(),
    }
    values.update(overrides)
    return LayerConfig(**values)


class TestLayerValidatorAdapter(unittest.TestCase):
    def test_module_exposes_validator_adapter(self):
        self.assertIs(Layer.VALIDATOR, LayerValidator)
        self.assertFalse(hasattr(Layer, "_validate_configuration"))

    def test_construction_dispatches_through_substituted_validator(self):
        class TrackingValidator(LayerValidator):
            @staticmethod
            def _validate_dropout_probability(dropout_probability):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayer(Layer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayer(make_config())

    def test_gate_validation_uses_replaceable_collaborator(self):
        class TrackingGateValidator:
            @classmethod
            def validate_layer_gate_config(cls, gate_config, owner_name=None):
                raise RuntimeError("substituted gate validator was called")

        class TrackingValidator(LayerValidator):
            GATE_VALIDATOR = TrackingGateValidator

        class TrackingLayer(Layer):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted gate validator was called",
        ):
            TrackingLayer(make_config())

    def test_built_model_validation_dispatches_through_substituted_validator(self):
        class TrackingValidator(LayerValidator):
            @staticmethod
            def validate_layer_model(model):
                raise RuntimeError("substituted layer-model validator was called")

        class TrackingLayer(Layer):
            VALIDATOR = TrackingValidator

            def _build_from_config(self, config, **kwargs):
                return object()

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted layer-model validator was called",
        ):
            TrackingLayer(make_config())

    def test_built_model_error_contract_is_preserved(self):
        with self.assertRaises(RuntimeError) as raised:
            LayerValidator.validate_layer_model(None)

        self.assertEqual(
            str(raised.exception),
            "layer_model_config must build a model.",
        )

    def test_dropout_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "dropout_probability must be between 0.0 and 1.0, received 1.1",
        ):
            Layer(make_config(dropout_probability=1.1))

    def test_dropout_probability_rejects_nonfinite_values(self):
        for dropout_probability in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(dropout_probability=dropout_probability):
                with self.assertRaises(ValueError) as error:
                    Layer(make_config(dropout_probability=dropout_probability))
                self.assertEqual(
                    str(error.exception),
                    "dropout_probability must be between 0.0 and 1.0, "
                    f"received {dropout_probability}",
                )


class TestLayerHaltingDelegateValidatorAdapter(unittest.TestCase):
    def test_module_exposes_layer_validator_adapter(self):
        self.assertIs(
            LayerHaltingDelegate.VALIDATOR,
            LayerHaltingDelegateValidator,
        )

    def test_construction_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerHaltingDelegateValidator):
            @classmethod
            def validate(cls, model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerHaltingDelegate(LayerHaltingDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerHaltingDelegate(make_config())

    def test_built_model_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerHaltingDelegateValidator):
            @staticmethod
            def validate_built_halting_model_interface(model):
                raise RuntimeError("substituted halting-model validator was called")

        class TrackingLayerHaltingDelegate(LayerHaltingDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted halting-model validator was called",
        ):
            TrackingLayerHaltingDelegate(make_config())

    def test_halting_output_dimension_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "Layer halting requires a resolved output_dim.",
        ):
            LayerHaltingDelegateValidator.validate_resolved_output_dim(None)

    def test_halting_model_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            "halting_config must build a model implementing HaltingInterface.",
        ):
            LayerHaltingDelegateValidator.validate_built_halting_model_interface(
                object()
            )


class TestLayerMemoryDelegateValidatorAdapter(unittest.TestCase):
    def test_module_exposes_memory_validator_adapter(self):
        self.assertIs(LayerMemoryDelegate.VALIDATOR, LayerMemoryDelegateValidator)

    def test_construction_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerMemoryDelegateValidator):
            @classmethod
            def validate(cls, model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerMemoryDelegate(LayerMemoryDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerMemoryDelegate(make_config())

    def test_built_model_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerMemoryDelegateValidator):
            @staticmethod
            def validate_built_memory_model_interface(model):
                raise RuntimeError("substituted memory-model validator was called")

        class TrackingLayerMemoryDelegate(LayerMemoryDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted memory-model validator was called",
        ):
            TrackingLayerMemoryDelegate(make_config())

    def test_memory_dimensions_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "Layer memory requires resolved input and output dimensions.",
        ):
            LayerMemoryDelegateValidator.validate_resolved_dimensions(None, 3)

    def test_memory_model_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            "memory_config must build a model implementing MemoryInterface.",
        ):
            LayerMemoryDelegateValidator.validate_built_memory_model_interface(object())


class TestLayerNormalizationDelegateValidatorAdapter(unittest.TestCase):
    def test_module_exposes_normalization_validator_adapter(self):
        self.assertIs(
            LayerNormalizationDelegate.VALIDATOR,
            LayerNormalizationDelegateValidator,
        )

    def test_construction_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerNormalizationDelegateValidator):
            @classmethod
            def validate(cls, model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerNormalizationDelegate(LayerNormalizationDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerNormalizationDelegate(make_config())

    def test_normalization_position_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "Layer normalization requires a resolved position.",
        ):
            LayerNormalizationDelegateValidator.validate_resolved_position(None)

    def test_normalization_dimensions_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "Layer normalization requires resolved input and output dimensions.",
        ):
            LayerNormalizationDelegateValidator.validate_resolved_dimensions(3, None)


class TestLayerPostprocessingDelegateValidatorAdapter(unittest.TestCase):
    def test_module_exposes_postprocessing_validator_adapter(self):
        self.assertIs(
            LayerPostprocessingDelegate.VALIDATOR,
            LayerPostprocessingDelegateValidator,
        )

    def test_construction_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerPostprocessingDelegateValidator):
            @classmethod
            def validate(cls, model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerPostprocessingDelegate(LayerPostprocessingDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerPostprocessingDelegate(make_config())

    def test_built_gate_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerPostprocessingDelegateValidator):
            @staticmethod
            def validate_built_gate_type(gate):
                raise RuntimeError("substituted built-gate validator was called")

        class TrackingLayerPostprocessingDelegate(LayerPostprocessingDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted built-gate validator was called",
        ):
            TrackingLayerPostprocessingDelegate(make_config())

    def test_resolved_field_error_contracts_are_preserved(self):
        cases = (
            (
                LayerPostprocessingDelegateValidator.validate_resolved_activation,
                "Layer postprocessing requires a resolved activation.",
            ),
            (
                LayerPostprocessingDelegateValidator.validate_resolved_output_dim,
                "Layer postprocessing requires a resolved output_dim.",
            ),
            (
                LayerPostprocessingDelegateValidator.validate_resolved_dropout_probability,
                "Layer postprocessing requires a resolved dropout_probability.",
            ),
        )

        for validator, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validator(None)

    def test_built_gate_type_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            "gate_config must build a LayerGate.",
        ):
            LayerPostprocessingDelegateValidator.validate_built_gate_type(object())


class TestLayerResidualDelegateValidatorAdapter(unittest.TestCase):
    def test_module_exposes_residual_validator_adapter(self):
        self.assertIs(
            LayerResidualDelegate.VALIDATOR,
            LayerResidualDelegateValidator,
        )

    def test_construction_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerResidualDelegateValidator):
            @classmethod
            def validate(cls, model):
                raise RuntimeError("substituted construction validator was called")

        class TrackingLayerResidualDelegate(LayerResidualDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            TrackingLayerResidualDelegate(make_config())

    def test_built_connection_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerResidualDelegateValidator):
            @staticmethod
            def validate_built_residual_connection_type(connection):
                raise RuntimeError("substituted built-connection validator was called")

        class TrackingLayerResidualDelegate(LayerResidualDelegate):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted built-connection validator was called",
        ):
            TrackingLayerResidualDelegate(make_config())

    def test_lifecycle_validation_dispatches_through_adapter(self):
        class TrackingValidator(LayerResidualDelegateValidator):
            @staticmethod
            def validate_built_residual_connection_type(connection):
                return connection

            @staticmethod
            def validate_forward_local_state_lifecycle_requirement(connection):
                raise RuntimeError("substituted lifecycle validator was called")

        class TrackingLayerResidualDelegate(LayerResidualDelegate):
            VALIDATOR = TrackingValidator

            def _build_from_config(self, config, **kwargs):
                return object()

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted lifecycle validator was called",
        ):
            TrackingLayerResidualDelegate(make_config())

    def test_residual_output_dimension_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            ValueError,
            "Layer residual processing requires a resolved output_dim.",
        ):
            LayerResidualDelegateValidator.validate_resolved_output_dim(None)

    def test_built_connection_type_error_contract_is_preserved(self):
        with self.assertRaisesRegex(
            TypeError,
            "residual_config must build a ResidualConnectionAbstract.",
        ):
            LayerResidualDelegateValidator.validate_built_residual_connection_type(
                object()
            )


if __name__ == "__main__":
    unittest.main()
