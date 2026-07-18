import unittest

import torch
from emperor.parametric import (
    ClipParameterOptions,
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerHandler,
)
from torch import nn


class ParametricCommitRegressionTests(unittest.TestCase):
    def test_handler_rejects_non_layer_state_before_processing(self) -> None:
        handler = ParametricLayerHandler.__new__(ParametricLayerHandler)
        nn.Module.__init__(handler)

        with self.assertRaisesRegex(
            TypeError,
            "^state must be a LayerState for ParametricLayerHandler, got object\\.$",
        ):
            handler(object())

    def test_dense_unweighted_matrix_parameters_broadcast_across_batch(self) -> None:
        mixture = MatrixWeightsMixtureConfig(
            input_dim=2,
            output_dim=2,
            top_k=2,
            num_experts=2,
            weighted_parameters_flag=False,
            clip_parameter_option=ClipParameterOptions.DISABLED,
            clip_range=1.0,
        ).build()
        parameter_bank = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.5, -1.0], [2.0, 1.5]],
            ]
        )
        with torch.no_grad():
            mixture.parameter_bank.copy_(parameter_bank)
        generated_weights = mixture.compute_mixture(
            torch.full((3, 2), 0.5),
            None,
        )
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [3.0, -2.0]])
        layer = ParametricLayer.__new__(ParametricLayer)
        nn.Module.__init__(layer)

        output = layer._compute_affine_transformation_callback(
            generated_weights,
            None,
            inputs,
        )

        expected_weights = parameter_bank.sum(dim=0)
        torch.testing.assert_close(generated_weights, expected_weights)
        torch.testing.assert_close(output, inputs @ expected_weights)


if __name__ == "__main__":
    unittest.main()
