import unittest

import torch

from emperor.layers import (
    ActivationOptions,
    Layer,
    LayerNormPositionOptions,
    LayerState,
)
from emperor.parametric import (
    AdaptiveRouterOptions,
    ParametricLayerConfig,
    ParametricLayerHandler,
    ParametricLayerHandlerConfig,
    VectorWeightsMixtureConfig,
)
from emperor.parametric._handlers import (
    GeneratorParameterHandler,
    MatrixParameterHandler,
    ParameterHandlerBase,
    VectorParameterHandler,
)
from emperor.parametric._validation import ParametricHandlerValidator


class TestParametricHandlerValidatorAdapter(unittest.TestCase):
    def test_parameter_handlers_expose_the_parametric_validator(self):
        module_types = (
            ParameterHandlerBase,
            VectorParameterHandler,
            MatrixParameterHandler,
            GeneratorParameterHandler,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, ParametricHandlerValidator)

        self.assertIs(ParametricLayerHandler.VALIDATOR, Layer.VALIDATOR)
        self.assertIs(
            ParametricLayerHandler.PARAMETRIC_VALIDATOR,
            ParametricHandlerValidator,
        )

    def test_parameter_handler_construction_dispatches_through_adapter(self):
        class TrackingValidator(ParametricHandlerValidator):
            @staticmethod
            def _validate_parameter_handler(model):
                raise RuntimeError("substituted parameter validator was called")

        class TrackingHandler(ParameterHandlerBase):
            VALIDATOR = TrackingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted parameter validator was called",
        ):
            TrackingHandler(ParametricLayerConfig())

    def test_layer_handler_construction_dispatches_through_adapter(self):
        class TrackingValidator(ParametricHandlerValidator):
            @staticmethod
            def _validate_layer_handler(model):
                raise RuntimeError("substituted layer validator was called")

        class TrackingLayerHandler(ParametricLayerHandler):
            PARAMETRIC_VALIDATOR = TrackingValidator

        cfg = ParametricLayerHandlerConfig(
            input_dim=3,
            output_dim=3,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=ParametricLayerConfig(),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted layer validator was called",
        ):
            TrackingLayerHandler(cfg)

    def test_runtime_state_dispatches_through_adapter(self):
        class RejectingValidator(ParametricHandlerValidator):
            @staticmethod
            def validate_state(state):
                raise RuntimeError("substituted runtime validator was called")

        class RejectingLayerHandler(ParametricLayerHandler):
            PARAMETRIC_VALIDATOR = RejectingValidator

        model = RejectingLayerHandler.__new__(RejectingLayerHandler)
        torch.nn.Module.__init__(model)

        with self.assertRaisesRegex(
            RuntimeError, "substituted runtime validator was called"
        ):
            model._handle_model_processing(
                torch.ones(1, 3), LayerState(hidden=torch.ones(1, 3))
            )

    def test_vector_shared_router_error_contract_is_preserved(self):
        cfg = ParametricLayerConfig(
            input_dim=3,
            output_dim=3,
            weight_mixture_config=VectorWeightsMixtureConfig(),
            bias_mixture_config=None,
            routing_initialization_mode=AdaptiveRouterOptions.SHARED_ROUTER,
            router_config=object(),
            sampler_config=object(),
            adaptive_augmentation_config=None,
        )

        with self.assertRaisesRegex(
            ValueError,
            "VectorWeightsMixtureConfig does not support SHARED_ROUTER routing",
        ):
            ParameterHandlerBase(cfg)


if __name__ == "__main__":
    unittest.main()
