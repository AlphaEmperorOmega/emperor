import sys
import unittest

import torch

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig


def _linear_expert_stack(*, dimension: int = 1, bias: bool = True) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=dimension,
        hidden_dim=dimension,
        output_dim=dimension,
        num_layers=1,
        last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
        apply_output_pipeline_flag=False,
        layer_config=LayerConfig(
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=LinearLayerConfig(bias_flag=bias),
        ),
    )


def _mixture_config(
    *,
    top_k: int,
    num_experts: int,
    capacity_factor: float = 0.0,
    dropped_token_behavior: DroppedTokenOptions | None = DroppedTokenOptions.ZEROS,
    compute_expert_mixture: bool = True,
    weighted: bool = False,
    weighting_position: ExpertWeightingPositionOptions = (
        ExpertWeightingPositionOptions.AFTER_EXPERTS
    ),
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=1,
        output_dim=1,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        dropped_token_behavior=dropped_token_behavior,
        compute_expert_mixture_flag=compute_expert_mixture,
        weighted_parameters_flag=weighted,
        weighting_position_option=weighting_position,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        expert_model_config=_linear_expert_stack(),
    )


def _set_affine_experts(
    model: MixtureOfExperts,
    *,
    weights: tuple[float, ...],
    biases: tuple[float, ...] | None = None,
) -> None:
    if biases is None:
        biases = (0.0,) * len(weights)
    with torch.no_grad():
        for expert, weight, bias in zip(
            model.expert_modules,
            weights,
            biases,
            strict=True,
        ):
            expert[0].model.weight_params.fill_(weight)
            expert[0].model.bias_params.fill_(bias)


class ExpertOptionContractTests(unittest.TestCase):
    def test_capacity_uses_top_k_and_rounds_fractional_assignments_up(self) -> None:
        model = (
            _mixture_config(
                top_k=2,
                num_experts=4,
                capacity_factor=0.5,
            )
            .build()
            .eval()
        )
        _set_affine_experts(model, weights=(1.0, 0.0, 0.0, 0.0))
        inputs = torch.arange(1.0, 6.0).reshape(-1, 1)
        probabilities = torch.ones(5, 2)
        indices = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
            ]
        )

        output, _skip_mask, _loss = model(
            inputs,
            probabilities=probabilities,
            indices=indices,
        )

        self.assertEqual(torch.count_nonzero(output).item(), 2)

    def test_capacity_is_stable_and_does_not_consume_rng_in_evaluation(self) -> None:
        model = (
            _mixture_config(
                top_k=1,
                num_experts=2,
                capacity_factor=1.0,
            )
            .build()
            .eval()
        )
        _set_affine_experts(model, weights=(1.0, 0.0))
        inputs = torch.arange(1.0, 5.0).reshape(-1, 1)
        probabilities = torch.ones(4)
        indices = torch.tensor([0, 0, 0, 1])
        rng_state_before = torch.random.get_rng_state()

        first_output, _skip_mask, _loss = model(
            inputs,
            probabilities=probabilities,
            indices=indices,
        )
        rng_state_after = torch.random.get_rng_state()
        second_output, _skip_mask, _loss = model(
            inputs,
            probabilities=probabilities,
            indices=indices,
        )

        torch.testing.assert_close(
            first_output,
            torch.tensor([[1.0], [2.0], [0.0], [0.0]]),
        )
        torch.testing.assert_close(second_output, first_output)
        torch.testing.assert_close(rng_state_after, rng_state_before)

    def test_disabled_capacity_does_not_consume_rng_during_training(self) -> None:
        model = (
            _mixture_config(
                top_k=1,
                num_experts=2,
                capacity_factor=0.0,
            )
            .build()
            .train()
        )
        _set_affine_experts(model, weights=(1.0, 0.0))
        rng_state_before = torch.random.get_rng_state()

        output, _skip_mask, _loss = model(
            torch.tensor([[2.0], [5.0]]),
            probabilities=torch.ones(2),
            indices=torch.zeros(2, dtype=torch.long),
        )

        torch.testing.assert_close(output, torch.tensor([[2.0], [5.0]]))
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state_before)

    def test_training_capacity_is_seeded_and_preserves_route_alignment(self) -> None:
        model = (
            _mixture_config(
                top_k=1,
                num_experts=2,
                capacity_factor=1.0,
                weighted=True,
            )
            .build()
            .train()
        )
        _set_affine_experts(model, weights=(1.0, 0.0))
        inputs = torch.tensor([[2.0], [5.0], [7.0], [11.0]])
        probabilities = torch.tensor([0.1, 0.3, 0.9, 0.5])
        indices = torch.tensor([0, 0, 0, 1])
        expected_if_retained = probabilities.reshape(-1, 1) * inputs
        seed = 127

        torch.manual_seed(seed)
        rng_state_before = torch.random.get_rng_state()
        first_output, _skip_mask, _loss = model(
            inputs,
            probabilities=probabilities,
            indices=indices,
        )
        rng_state_after = torch.random.get_rng_state()
        torch.manual_seed(seed)
        second_output, _skip_mask, _loss = model(
            inputs,
            probabilities=probabilities,
            indices=indices,
        )

        torch.testing.assert_close(second_output, first_output)
        self.assertFalse(torch.equal(rng_state_after, rng_state_before))
        self.assertEqual(torch.count_nonzero(first_output[:3]).item(), 2)
        for actual, expected in zip(
            first_output[:3],
            expected_if_retained[:3],
            strict=True,
        ):
            self.assertTrue(
                torch.equal(actual, expected)
                or torch.equal(actual, torch.zeros_like(actual))
            )

    def test_huge_finite_capacity_factor_is_overflow_safe(self) -> None:
        model = (
            _mixture_config(
                top_k=1,
                num_experts=2,
                capacity_factor=sys.float_info.max,
            )
            .build()
            .train()
        )
        _set_affine_experts(model, weights=(1.0, 0.0))
        rng_state_before = torch.random.get_rng_state()

        output, _skip_mask, _loss = model(
            torch.tensor([[2.0], [5.0], [7.0], [11.0]]),
            probabilities=torch.ones(4),
            indices=torch.zeros(4, dtype=torch.long),
        )

        torch.testing.assert_close(output, torch.tensor([[2.0], [5.0], [7.0], [11.0]]))
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state_before)

    def test_missing_dropped_token_option_uses_zero_fallback(self) -> None:
        for behavior in (None, DroppedTokenOptions.ZEROS):
            with self.subTest(behavior=behavior):
                model = (
                    _mixture_config(
                        top_k=1,
                        num_experts=2,
                        capacity_factor=1.0,
                        dropped_token_behavior=behavior,
                    )
                    .build()
                    .eval()
                )
                _set_affine_experts(model, weights=(1.0, 0.0))

                output, _skip_mask, _loss = model(
                    torch.tensor([[2.0], [5.0]]),
                    probabilities=torch.ones(2),
                    indices=torch.zeros(2, dtype=torch.long),
                )

                torch.testing.assert_close(output, torch.tensor([[2.0], [0.0]]))

    def test_identity_fallback_matches_weighting_position_and_flag(self) -> None:
        for weighted in (False, True):
            for weighting_position in ExpertWeightingPositionOptions:
                with self.subTest(
                    weighted=weighted,
                    weighting_position=weighting_position,
                ):
                    model = (
                        _mixture_config(
                            top_k=1,
                            num_experts=2,
                            capacity_factor=1.0,
                            dropped_token_behavior=DroppedTokenOptions.IDENTITY,
                            weighted=weighted,
                            weighting_position=weighting_position,
                        )
                        .build()
                        .eval()
                    )
                    _set_affine_experts(
                        model,
                        weights=(2.0, 0.0),
                        biases=(3.0, 0.0),
                    )
                    if not weighted:
                        expected = torch.tensor([[7.0], [5.0]])
                    elif (
                        weighting_position
                        == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                    ):
                        expected = torch.tensor([[4.0], [3.75]])
                    else:
                        expected = torch.tensor([[1.75], [3.75]])

                    output, _skip_mask, _loss = model(
                        torch.tensor([[2.0], [5.0]]),
                        probabilities=torch.tensor([0.25, 0.75]),
                        indices=torch.zeros(2, dtype=torch.long),
                    )

                    torch.testing.assert_close(output, expected)

    def test_dropped_route_jacobian_matches_zero_and_identity_fallbacks(
        self,
    ) -> None:
        for behavior in DroppedTokenOptions:
            for weighted in (False, True):
                for weighting_position in ExpertWeightingPositionOptions:
                    with self.subTest(
                        behavior=behavior,
                        weighted=weighted,
                        weighting_position=weighting_position,
                    ):
                        model = (
                            _mixture_config(
                                top_k=1,
                                num_experts=2,
                                capacity_factor=1.0,
                                dropped_token_behavior=behavior,
                                weighted=weighted,
                                weighting_position=weighting_position,
                            )
                            .build()
                            .eval()
                        )
                        _set_affine_experts(
                            model,
                            weights=(2.0, 0.0),
                            biases=(3.0, 0.0),
                        )
                        inputs = torch.tensor(
                            [[2.0], [5.0]],
                            requires_grad=True,
                        )
                        probabilities = torch.tensor(
                            [0.25, 0.75],
                            requires_grad=True,
                        )

                        output, _skip_mask, _loss = model(
                            inputs,
                            probabilities=probabilities,
                            indices=torch.zeros(2, dtype=torch.long),
                        )
                        output.sum().backward()

                        expected_input_gradient = 0.0
                        if behavior == DroppedTokenOptions.IDENTITY:
                            expected_input_gradient = 0.75 if weighted else 1.0
                        self.assertEqual(
                            inputs.grad[1].item(),
                            expected_input_gradient,
                        )
                        if weighted:
                            expected_probability_gradient = (
                                5.0 if behavior == DroppedTokenOptions.IDENTITY else 0.0
                            )
                            self.assertEqual(
                                probabilities.grad[1].item(),
                                expected_probability_gradient,
                            )
                        else:
                            self.assertIsNone(probabilities.grad)

    def test_affine_weighting_and_reduction_options_match_exact_equations(
        self,
    ) -> None:
        inputs = torch.tensor([[2.0], [3.0]])
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        indices = torch.tensor([[0, 2], [1, 0]])
        unweighted_routes = torch.tensor([[5.0], [8.0], [13.0], [7.0]])
        weighted_routes = {
            ExpertWeightingPositionOptions.BEFORE_EXPERTS: torch.tensor(
                [[2.0], [5.5], [9.4], [3.4]]
            ),
            ExpertWeightingPositionOptions.AFTER_EXPERTS: torch.tensor(
                [[1.25], [6.0], [7.8], [2.8]]
            ),
        }

        for weighted in (False, True):
            for weighting_position in ExpertWeightingPositionOptions:
                for compute_expert_mixture in (False, True):
                    with self.subTest(
                        weighted=weighted,
                        weighting_position=weighting_position,
                        compute_expert_mixture=compute_expert_mixture,
                    ):
                        model = _mixture_config(
                            top_k=2,
                            num_experts=3,
                            compute_expert_mixture=compute_expert_mixture,
                            weighted=weighted,
                            weighting_position=weighting_position,
                        ).build()
                        _set_affine_experts(
                            model,
                            weights=(2.0, 3.0, 5.0),
                            biases=(1.0, 4.0, -2.0),
                        )
                        expected_routes = (
                            weighted_routes[weighting_position]
                            if weighted
                            else unweighted_routes
                        )
                        expected = (
                            expected_routes.reshape(2, 2, 1).sum(dim=1)
                            if compute_expert_mixture
                            else expected_routes
                        )

                        output, _skip_mask, _loss = model(
                            inputs,
                            probabilities=probabilities,
                            indices=indices,
                        )

                        torch.testing.assert_close(output, expected)

    def test_dense_all_expert_routing_matches_affine_option_equations(self) -> None:
        inputs = torch.tensor([[2.0], [3.0]])
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        unweighted_routes = torch.tensor([[5.0], [10.0], [7.0], [13.0]])
        weighted_routes = {
            ExpertWeightingPositionOptions.BEFORE_EXPERTS: torch.tensor(
                [[2.0], [8.5], [4.6], [7.6]]
            ),
            ExpertWeightingPositionOptions.AFTER_EXPERTS: torch.tensor(
                [[1.25], [7.5], [4.2], [5.2]]
            ),
        }

        for weighted in (False, True):
            for weighting_position in ExpertWeightingPositionOptions:
                for compute_expert_mixture in (False, True):
                    with self.subTest(
                        weighted=weighted,
                        weighting_position=weighting_position,
                        compute_expert_mixture=compute_expert_mixture,
                    ):
                        model = _mixture_config(
                            top_k=2,
                            num_experts=2,
                            compute_expert_mixture=compute_expert_mixture,
                            weighted=weighted,
                            weighting_position=weighting_position,
                        ).build()
                        _set_affine_experts(
                            model,
                            weights=(2.0, 3.0),
                            biases=(1.0, 4.0),
                        )
                        expected_routes = (
                            weighted_routes[weighting_position]
                            if weighted
                            else unweighted_routes
                        )
                        expected = (
                            expected_routes.reshape(2, 2, 1).sum(dim=1)
                            if compute_expert_mixture
                            else expected_routes
                        )

                        output, _skip_mask, _loss = model(
                            inputs,
                            probabilities=probabilities,
                            indices=None,
                        )

                        torch.testing.assert_close(output, expected)

    def test_top_one_vector_and_column_routing_have_identical_equations(
        self,
    ) -> None:
        inputs = torch.tensor([[2.0], [3.0]])
        vector_probabilities = torch.tensor([0.25, 0.6])
        vector_indices = torch.tensor([0, 1])

        for weighting_position, expected in (
            (
                ExpertWeightingPositionOptions.BEFORE_EXPERTS,
                torch.tensor([[2.0], [9.4]]),
            ),
            (
                ExpertWeightingPositionOptions.AFTER_EXPERTS,
                torch.tensor([[1.25], [7.8]]),
            ),
        ):
            for compute_expert_mixture in (False, True):
                for as_column in (False, True):
                    with self.subTest(
                        weighting_position=weighting_position,
                        compute_expert_mixture=compute_expert_mixture,
                        as_column=as_column,
                    ):
                        model = _mixture_config(
                            top_k=1,
                            num_experts=2,
                            compute_expert_mixture=compute_expert_mixture,
                            weighted=True,
                            weighting_position=weighting_position,
                        ).build()
                        _set_affine_experts(
                            model,
                            weights=(2.0, 3.0),
                            biases=(1.0, 4.0),
                        )
                        probabilities = (
                            vector_probabilities.reshape(-1, 1)
                            if as_column
                            else vector_probabilities
                        )
                        indices = (
                            vector_indices.reshape(-1, 1)
                            if as_column
                            else vector_indices
                        )

                        output, _skip_mask, _loss = model(
                            inputs,
                            probabilities=probabilities,
                            indices=indices,
                        )

                        torch.testing.assert_close(output, expected)
