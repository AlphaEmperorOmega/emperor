import unittest
from types import SimpleNamespace

import torch
from emperor.parametric import (
    ClipParameterOptions,
    GeneratorBiasMixture,
    GeneratorWeightsMixture,
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerHandler,
    VectorWeightsMixtureConfig,
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

    def test_explicit_full_top_k_matrix_routes_stay_sample_local(self) -> None:
        mixture = MatrixWeightsMixtureConfig(
            input_dim=2,
            output_dim=2,
            top_k=2,
            num_experts=2,
            weighted_parameters_flag=True,
            clip_parameter_option=ClipParameterOptions.DISABLED,
            clip_range=1.0,
        ).build()
        parameter_bank = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, -1.0], [3.0, 0.5]],
            ]
        )
        with torch.no_grad():
            mixture.parameter_bank.copy_(parameter_bank)
        probabilities = torch.tensor([[0.75, 0.25], [0.1, 0.9]])
        indices = torch.tensor([[0, 1], [1, 0]])

        output = mixture.compute_mixture(probabilities, indices)

        expected = torch.stack(
            (
                0.75 * parameter_bank[0] + 0.25 * parameter_bank[1],
                0.1 * parameter_bank[1] + 0.9 * parameter_bank[0],
            )
        )
        torch.testing.assert_close(output, expected)

    def test_vector_full_top_k_routes_preserve_axis_selection(self) -> None:
        mixture = VectorWeightsMixtureConfig(
            input_dim=2,
            output_dim=2,
            top_k=2,
            num_experts=3,
            weighted_parameters_flag=True,
            clip_parameter_option=ClipParameterOptions.DISABLED,
            clip_range=1.0,
        ).build()
        parameter_bank = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 1.0], [-1.0, 3.0]],
                [[0.0, 1.0], [4.0, -2.0], [2.0, 2.0]],
            ]
        )
        with torch.no_grad():
            mixture.parameter_bank.copy_(parameter_bank)
        probabilities = torch.tensor(
            [
                [[0.25, 0.75], [0.6, 0.4]],
                [[0.8, 0.2], [0.1, 0.9]],
            ]
        )
        indices = torch.tensor(
            [
                [[0, 2], [1, 0]],
                [[2, 1], [0, 2]],
            ]
        )

        output = mixture.compute_mixture(probabilities, indices)

        expected = torch.empty(2, 2, 2)
        for axis in range(2):
            for sample in range(2):
                expected[sample, axis] = sum(
                    probabilities[axis, sample, route]
                    * parameter_bank[axis, indices[axis, sample, route]]
                    for route in range(2)
                )
        torch.testing.assert_close(output, expected)

    def test_generator_mixtures_consume_expert_metadata(self) -> None:
        weights = GeneratorWeightsMixture.__new__(GeneratorWeightsMixture)
        nn.Module.__init__(weights)
        weights.cfg = SimpleNamespace(weighted_parameters_flag=False)
        weights.top_k = 2
        weights.num_experts = 2
        weights.input_dim = 2
        weights.output_dim = 2
        weights.weighted_parameters_flag = False
        weights.clip_parameter_option = ClipParameterOptions.DISABLED
        weights.clip_range = 1.0
        input_vectors = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        output_vectors = torch.tensor([[[2.0, 1.0], [3.0, -1.0]]])
        weights.input_vector_generator = lambda *_: (
            input_vectors,
            torch.ones(1, 1),
            torch.tensor(0.25),
        )
        weights.output_vector_generator = lambda *_: (
            output_vectors,
            torch.zeros(1, 1),
            torch.tensor(0.75),
        )

        generated_weights, weight_loss = weights.compute_mixture(
            None,
            None,
            torch.ones(1, 2),
        )

        expected_weights = sum(
            torch.outer(input_vectors[0, route], output_vectors[0, route])
            for route in range(2)
        ).unsqueeze(0)
        torch.testing.assert_close(generated_weights, expected_weights)
        torch.testing.assert_close(weight_loss, torch.tensor(1.0))

        bias = GeneratorBiasMixture.__new__(GeneratorBiasMixture)
        nn.Module.__init__(bias)
        bias.top_k = 2
        bias.num_experts = 2
        expected_bias = torch.tensor([[1.5, -2.0]])
        bias.bias_generator = lambda *_: (
            expected_bias,
            torch.ones(1, 1),
            torch.tensor(0.5),
        )

        generated_bias, bias_loss = bias.compute_mixture(
            None,
            None,
            torch.ones(1, 2),
        )

        torch.testing.assert_close(generated_bias, expected_bias)
        torch.testing.assert_close(bias_loss, torch.tensor(0.5))


if __name__ == "__main__":
    unittest.main()
