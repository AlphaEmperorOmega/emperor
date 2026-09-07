import unittest

import torch

from emperor.config import ConfigBase
from emperor.experts import MixtureOfExpertsLayerConfig, MixtureOfExpertsLayerState
from emperor.experts._layers.layer import MixtureOfExpertsLayer
from emperor.experts._validation.layer import MixtureOfExpertsLayerValidator
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
)
from emperor.layers._layer.validation import LayerValidator


class _RoutingStub(torch.nn.Module):
    compute_expert_mixture_flag = True
    top_k = 1

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.call_count = 0

    def forward(
        self,
        _input_batch,
        _probabilities,
        _indices,
        _skip_mask,
    ):
        self.call_count += 1
        return self.output, None, self.output.new_zeros(())


def _layer_with_routing_output(output: torch.Tensor) -> MixtureOfExpertsLayer:
    layer = MixtureOfExpertsLayer.__new__(MixtureOfExpertsLayer)
    torch.nn.Module.__init__(layer)
    layer.model = _RoutingStub(output)
    return layer


class TestMixtureOfExpertsLayerValidatorAdapter(unittest.TestCase):
    def test_layer_exposes_specialized_layer_validator(self) -> None:
        self.assertIs(MixtureOfExpertsLayer.VALIDATOR, MixtureOfExpertsLayerValidator)
        self.assertTrue(issubclass(MixtureOfExpertsLayerValidator, LayerValidator))

    def test_construction_dispatches_through_substituted_validator(self) -> None:
        class RejectingValidator(MixtureOfExpertsLayerValidator):
            @classmethod
            def validate(cls, model) -> None:
                raise RuntimeError("substituted construction validator was called")

        class RejectingLayer(MixtureOfExpertsLayer):
            VALIDATOR = RejectingValidator

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted construction validator was called",
        ):
            RejectingLayer(MixtureOfExpertsLayerConfig())

    def test_specialized_validator_preserves_base_layer_validation(self) -> None:
        layer = MixtureOfExpertsLayer.__new__(MixtureOfExpertsLayer)
        torch.nn.Module.__init__(layer)
        layer.cfg = LayerConfig(
            input_dim=3,
            output_dim=3,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=1.1,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=ConfigBase(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "dropout_probability must be between 0.0 and 1.0",
        ):
            MixtureOfExpertsLayerValidator.validate(layer)

    def test_post_routing_dispatches_through_substituted_validator(self) -> None:
        class RejectingValidator(MixtureOfExpertsLayerValidator):
            @staticmethod
            def validate_output_rows(layer, main_model_input, output) -> None:
                raise RuntimeError("substituted post-routing validator was called")

        class RejectingLayer(MixtureOfExpertsLayer):
            VALIDATOR = RejectingValidator

        layer = RejectingLayer.__new__(RejectingLayer)
        torch.nn.Module.__init__(layer)
        layer.model = _RoutingStub(torch.ones(2, 3))

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted post-routing validator was called",
        ):
            layer._handle_model_processing(
                MixtureOfExpertsLayerState(hidden=torch.ones(2, 3)),
            )

    def test_rejects_output_that_does_not_restore_input_row_count(self) -> None:
        layer = _layer_with_routing_output(torch.ones(3, 3))
        state = MixtureOfExpertsLayerState(
            hidden=torch.ones(2, 3),
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected 2, received shape \\(3, 3\\)",
        ):
            layer._handle_model_processing(state)

        self.assertEqual(layer.model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
