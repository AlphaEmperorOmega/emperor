import unittest

import torch
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.layer.state import LayerState
from emperor.base.options import ActivationOptions, LayerNormPositionOptions
from emperor.parametric.core._validator import ParametricHandlerValidator
from emperor.parametric.core.config import (
    AdaptiveRouterOptions,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
)
from emperor.parametric.core.handlers import (
    GeneratorParameterHandler,
    MatrixParameterHandler,
    ParameterHandlerBase,
    ParametricLayerHandler,
    VectorParameterHandler,
)
from emperor.parametric.core.mixtures.config import VectorWeightsMixtureConfig


class TestParametricHandlerValidatorAdapter(unittest.TestCase):
    def test_handler_modules_expose_the_shared_validator_adapter(self):
        module_types = (
            ParameterHandlerBase,
            VectorParameterHandler,
            MatrixParameterHandler,
            GeneratorParameterHandler,
            ParametricLayerHandler,
        )

        for module_type in module_types:
            with self.subTest(module_type=module_type.__name__):
                self.assertIs(module_type.VALIDATOR, ParametricHandlerValidator)

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
            VALIDATOR = TrackingValidator

        cfg = ParametricLayerHandlerConfig(
            input_dim=3,
            output_dim=3,
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
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
            VALIDATOR = RejectingValidator

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
