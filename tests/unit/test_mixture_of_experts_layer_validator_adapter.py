import unittest

import torch

from emperor._validation import ValidatorBase
from emperor.experts import MixtureOfExpertsLayerConfig, MixtureOfExpertsLayerState
from emperor.experts._layers.layer import MixtureOfExpertsLayer
from emperor.experts._validation.layer import MixtureOfExpertsLayerValidator
from emperor.layers import RowLayout


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
        self.assertTrue(issubclass(MixtureOfExpertsLayerValidator, ValidatorBase))

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

    def test_pre_routing_dispatches_through_substituted_validator(self) -> None:
        class RejectingValidator(MixtureOfExpertsLayerValidator):
            @staticmethod
            def validate_layout_can_cross_routing(
                layer,
                state,
                main_model_input,
            ) -> None:
                raise RuntimeError("substituted pre-routing validator was called")

        class RejectingLayer(MixtureOfExpertsLayer):
            VALIDATOR = RejectingValidator

        layer = RejectingLayer.__new__(RejectingLayer)
        torch.nn.Module.__init__(layer)

        with self.assertRaisesRegex(
            RuntimeError,
            "substituted pre-routing validator was called",
        ):
            layer._handle_model_processing(
                torch.ones(2, 3),
                MixtureOfExpertsLayerState(hidden=torch.ones(2, 3)),
            )

    def test_post_routing_dispatches_through_substituted_validator(self) -> None:
        class RejectingValidator(MixtureOfExpertsLayerValidator):
            @staticmethod
            def validate_layout_restored(state, output) -> None:
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
                torch.ones(2, 3),
                MixtureOfExpertsLayerState(hidden=torch.ones(2, 3)),
            )

    def test_rejects_layout_with_wrong_input_row_count_before_routing(self) -> None:
        layer = _layer_with_routing_output(torch.ones(2, 3))
        state = MixtureOfExpertsLayerState(
            hidden=torch.ones(2, 3),
            row_layout=RowLayout.rows(
                3,
                context_sharing_restricted=False,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "row_layout row_count=3 does not match input row count 2",
        ):
            layer._handle_model_processing(torch.ones(2, 3), state)

        self.assertEqual(layer.model.call_count, 0)

    def test_rejects_output_that_does_not_restore_layout_row_count(self) -> None:
        layer = _layer_with_routing_output(torch.ones(3, 3))
        state = MixtureOfExpertsLayerState(
            hidden=torch.ones(2, 3),
            row_layout=RowLayout.rows(
                2,
                context_sharing_restricted=False,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected 2, received shape \\(3, 3\\)",
        ):
            layer._handle_model_processing(torch.ones(2, 3), state)

        self.assertEqual(layer.model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
