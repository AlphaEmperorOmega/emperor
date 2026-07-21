import unittest

import torch

from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    RoutingInitializationMode,
)
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
    LayerState,
)
from emperor.linears import LinearLayerConfig
from emperor.parametric import (
    AdaptiveRouterOptions,
    ClipParameterOptions,
    GeneratorWeightsMixtureConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerConfig,
    ParametricLayerHandlerConfig,
    VectorWeightsMixtureConfig,
)
from emperor.parametric._handlers import (
    ParameterHandlerBase,
    VectorParameterHandler,
)
from emperor.parametric._mixtures.validation import AdaptiveMixtureValidator
from emperor.parametric._mixtures.vector import VectorMixtureBase
from emperor.parametric._validation import (
    ParametricHandlerValidator,
    ParametricLayerValidator,
)
from emperor.sampler import RouterConfig, SamplerConfig


def _linear_stack(input_dim: int, output_dim: int) -> LayerStackConfig:
    return LayerStackConfig(
        input_dim=input_dim,
        hidden_dim=max(input_dim, output_dim),
        output_dim=output_dim,
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
            layer_model_config=LinearLayerConfig(bias_flag=False),
        ),
    )


def _router_config(input_dim: int, num_experts: int) -> RouterConfig:
    return RouterConfig(
        input_dim=input_dim,
        num_experts=num_experts,
        noisy_topk_flag=False,
        model_config=_linear_stack(input_dim, num_experts),
    )


def _sampler_config(top_k: int, num_experts: int) -> SamplerConfig:
    return SamplerConfig(
        top_k=top_k,
        threshold=0.0,
        filter_above_threshold=False,
        num_topk_samples=0,
        normalize_probabilities_flag=False,
        noisy_topk_flag=False,
        num_experts=num_experts,
        coefficient_of_variation_loss_weight=0.0,
        switch_loss_weight=0.0,
        zero_centred_loss_weight=0.0,
        mutual_information_loss_weight=0.0,
        router_config=None,
    )


def _mixture_kwargs(
    *,
    input_dim: int = 2,
    output_dim: int = 2,
    top_k: int = 2,
    num_experts: int = 3,
    weighted: bool = True,
) -> dict[str, object]:
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "top_k": top_k,
        "num_experts": num_experts,
        "weighted_parameters_flag": weighted,
        "clip_parameter_option": ClipParameterOptions.DISABLED,
        "clip_range": 1.0,
    }


def _augmentation_config(
    input_dim: int = 2,
    output_dim: int = 2,
) -> AdaptiveParameterAugmentationConfig:
    return AdaptiveParameterAugmentationConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        weight_config=None,
        diagonal_config=None,
        bias_config=None,
        mask_config=None,
        model_config=None,
    )


def _generator_config(
    *,
    input_dim: int = 2,
    top_k: int = 2,
    num_experts: int = 2,
    sampler_config: SamplerConfig | None = None,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=input_dim,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=False,
        weighted_parameters_flag=False,
        weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=sampler_config,
        expert_model_config=_linear_stack(input_dim, input_dim),
    )


def _parametric_config(
    *,
    weight_config=None,
    bias_config=None,
    routing_mode: AdaptiveRouterOptions = AdaptiveRouterOptions.INDEPENDENT_ROUTER,
    input_dim: int = 2,
    output_dim: int = 2,
    top_k: int = 2,
    num_experts: int = 2,
) -> ParametricLayerConfig:
    if weight_config is None:
        weight_config = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=input_dim,
                output_dim=output_dim,
                top_k=top_k,
                num_experts=num_experts,
            )
        )
    return ParametricLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        weight_mixture_config=weight_config,
        bias_mixture_config=bias_config,
        routing_initialization_mode=routing_mode,
        router_config=_router_config(input_dim, num_experts),
        sampler_config=_sampler_config(top_k, num_experts),
        adaptive_augmentation_config=_augmentation_config(input_dim, output_dim),
    )


class ParametricMixtureBehavioralContractTests(unittest.TestCase):
    def test_matrix_sparse_dense_and_top_one_mixtures_are_exact(self) -> None:
        bank = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, -1.0], [3.0, 0.5]],
                [[-2.0, 4.0], [1.0, -3.0]],
            ]
        )

        sparse = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=3)
        ).build()
        with torch.no_grad():
            sparse.parameter_bank.copy_(bank)
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        indices = torch.tensor([[2, 0], [1, 2]])
        expected_sparse = torch.stack(
            (
                0.25 * bank[2] + 0.75 * bank[0],
                0.6 * bank[1] + 0.4 * bank[2],
            )
        )
        torch.testing.assert_close(
            sparse.compute_mixture(probabilities, indices),
            expected_sparse,
        )

        dense = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=3, num_experts=3)
        ).build()
        with torch.no_grad():
            dense.parameter_bank.copy_(bank)
        dense_probabilities = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]])
        expected_dense = torch.einsum("be,eij->bij", dense_probabilities, bank)
        torch.testing.assert_close(
            dense.compute_mixture(dense_probabilities, None),
            expected_dense,
        )

        explicit_full = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=3, num_experts=3)
        ).build()
        with torch.no_grad():
            explicit_full.parameter_bank.copy_(bank)
        explicit_probabilities = torch.tensor(
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
            requires_grad=True,
        )
        explicit_indices = torch.tensor([[2, 0, 1], [1, 2, 0]])
        expected_explicit = torch.stack(
            tuple(
                sum(
                    explicit_probabilities[sample, route].detach()
                    * bank[explicit_indices[sample, route]]
                    for route in range(3)
                )
                for sample in range(2)
            )
        )
        explicit_output = explicit_full.compute_mixture(
            explicit_probabilities,
            explicit_indices,
        )
        torch.testing.assert_close(explicit_output, expected_explicit)
        self.assertEqual(explicit_output.shape, (2, 2, 2))
        self.assertEqual(explicit_output.dtype, bank.dtype)
        self.assertEqual(explicit_output.device, bank.device)
        self.assertFalse(torch.equal(explicit_output[0], explicit_output[1]))
        explicit_output.square().sum().backward()
        self.assertGreater(explicit_probabilities.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            explicit_full.parameter_bank.grad.abs().sum().item(),
            0.0,
        )

        top_one = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(
                top_k=1,
                num_experts=3,
                weighted=False,
            )
        ).build()
        with torch.no_grad():
            top_one.parameter_bank.copy_(bank)
        top_one_indices = torch.tensor([1, 0])
        torch.testing.assert_close(
            top_one.compute_mixture(None, top_one_indices),
            bank[top_one_indices],
        )
        with self.assertRaisesRegex(ValueError, "Probabilities must be provided"):
            sparse.compute_mixture(None, indices)

    def test_matrix_bias_probability_shapes_and_gradients_are_exact(self) -> None:
        mixture = MatrixBiasMixtureConfig(
            **_mixture_kwargs(output_dim=3, top_k=2, num_experts=3)
        ).build()
        bank = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0], [3.0, -1.0, 2.0]])
        with torch.no_grad():
            mixture.parameter_bank.copy_(bank)
        probabilities = torch.tensor(
            [[0.2, 0.8], [0.75, 0.25]],
            requires_grad=True,
        )
        indices = torch.tensor([[0, 2], [1, 0]])
        expected = torch.stack(
            (
                0.2 * bank[0] + 0.8 * bank[2],
                0.75 * bank[1] + 0.25 * bank[0],
            )
        )

        output = mixture.compute_mixture(probabilities, indices)

        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(probabilities.grad).all())
        self.assertGreater(probabilities.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(mixture.parameter_bank.grad).all())
        self.assertGreater(mixture.parameter_bank.grad.abs().sum().item(), 0.0)

        top_one = MatrixBiasMixtureConfig(
            **_mixture_kwargs(
                output_dim=3,
                top_k=1,
                num_experts=3,
                weighted=False,
            )
        ).build()
        with torch.no_grad():
            top_one.parameter_bank.copy_(bank)
        torch.testing.assert_close(
            top_one.compute_mixture(None, torch.tensor([2, 0])),
            bank[torch.tensor([2, 0])],
        )

    def test_dense_unweighted_matrix_mixtures_reduce_the_expert_axis(self) -> None:
        weight_bank = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[2.0, 3.0, 5.0, 7.0], [11.0, 13.0, 17.0, 19.0]],
                [[3.0, 5.0, 8.0, 13.0], [21.0, 34.0, 55.0, 89.0]],
            ],
            dtype=torch.float64,
        )
        bias_bank = torch.tensor(
            [
                [1.0, 2.0, 4.0, 8.0],
                [3.0, 5.0, 7.0, 11.0],
                [13.0, 17.0, 19.0, 23.0],
            ],
            dtype=torch.float64,
        )
        probabilities = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
            dtype=torch.float64,
        )
        weights = (
            MatrixWeightsMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=4,
                    top_k=3,
                    num_experts=3,
                    weighted=False,
                )
            )
            .build()
            .double()
        )
        bias = (
            MatrixBiasMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=4,
                    top_k=3,
                    num_experts=3,
                    weighted=False,
                )
            )
            .build()
            .double()
        )
        with torch.no_grad():
            weights.parameter_bank.copy_(weight_bank)
            bias.parameter_bank.copy_(bias_bank)

        generated_weights = weights.compute_mixture(probabilities, None)
        generated_bias = bias.compute_mixture(probabilities, None)

        torch.testing.assert_close(generated_weights, weight_bank.sum(dim=0))
        torch.testing.assert_close(generated_bias, bias_bank.sum(dim=0))
        self.assertEqual(generated_weights.shape, (2, 4))
        self.assertEqual(generated_bias.shape, (4,))
        self.assertEqual(generated_weights.dtype, torch.float64)
        self.assertEqual(generated_bias.device, weight_bank.device)
        (generated_weights.square().sum() + generated_bias.square().sum()).backward()
        self.assertGreater(weights.parameter_bank.grad.abs().sum().item(), 0.0)
        self.assertGreater(bias.parameter_bank.grad.abs().sum().item(), 0.0)

    def test_vector_mixtures_preserve_axis_sample_isolation_exactly(self) -> None:
        bank = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 1.0], [-1.0, 3.0]],
                [[0.0, 1.0], [4.0, -2.0], [2.0, 2.0]],
            ]
        )
        mixture = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=2,
                num_experts=3,
            )
        ).build()
        with torch.no_grad():
            mixture.parameter_bank.copy_(bank)
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
        expected = torch.empty(2, 2, 2)
        for axis in range(2):
            for sample in range(2):
                expected[sample, axis] = sum(
                    probabilities[axis, sample, route]
                    * bank[axis, indices[axis, sample, route]]
                    for route in range(2)
                )

        output = mixture.compute_mixture(probabilities, indices)

        torch.testing.assert_close(output, expected)
        sentinel = torch.arange(12.0).reshape(2, 2, 3)
        torch.testing.assert_close(
            VectorMixtureBase._handle_mixture_output(mixture, sentinel),
            sentinel,
        )
        torch.testing.assert_close(
            mixture._handle_mixture_output(sentinel),
            sentinel.transpose(-1, -2),
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "_compute_weighted_parameters.*child class",
        ):
            VectorMixtureBase._compute_weighted_parameters(
                mixture,
                torch.ones(1),
                torch.ones(1),
            )

        dense = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=3,
                num_experts=3,
            )
        ).build()
        with torch.no_grad():
            dense.parameter_bank.copy_(bank)
        dense_probabilities = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],
                [[0.5, 0.25, 0.25], [0.2, 0.6, 0.2]],
            ]
        )
        expected_dense = torch.empty(2, 2, 2)
        for axis in range(2):
            for sample in range(2):
                expected_dense[sample, axis] = torch.einsum(
                    "e,eo->o",
                    dense_probabilities[axis, sample],
                    bank[axis],
                )
        torch.testing.assert_close(
            dense.compute_mixture(dense_probabilities, None),
            expected_dense,
        )

        unweighted = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=2,
                num_experts=3,
                weighted=False,
            )
        ).build()
        with torch.no_grad():
            unweighted.parameter_bank.copy_(bank)
        expected_unweighted = torch.empty(2, 2, 2)
        for axis in range(2):
            for sample in range(2):
                expected_unweighted[sample, axis] = bank[
                    axis,
                    indices[axis, sample],
                ].sum(dim=0)
        torch.testing.assert_close(
            unweighted.compute_mixture(None, indices),
            expected_unweighted,
        )

        top_one = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=3,
            )
        ).build()
        with torch.no_grad():
            top_one.parameter_bank.copy_(bank)
        top_one_probabilities = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4]],
            requires_grad=True,
        )
        top_one_indices = torch.tensor([[2, 0], [1, 2]])
        expected_top_one = torch.stack(
            (
                torch.stack((0.25 * bank[0, 2], 0.6 * bank[1, 1])),
                torch.stack((0.75 * bank[0, 0], 0.4 * bank[1, 2])),
            )
        )

        top_one_output = top_one.compute_mixture(
            top_one_probabilities,
            top_one_indices,
        )

        torch.testing.assert_close(top_one_output, expected_top_one)
        top_one_output.sum().backward()
        self.assertTrue(torch.isfinite(top_one_probabilities.grad).all())
        self.assertGreater(top_one_probabilities.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            top_one.parameter_bank.grad[
                torch.tensor([[0], [1]]),
                torch.tensor([[0, 2], [1, 2]]),
            ]
            .abs()
            .sum()
            .item(),
            0.0,
        )
        torch.testing.assert_close(
            top_one.parameter_bank.grad[0, 1],
            torch.zeros(2),
        )
        torch.testing.assert_close(
            top_one.parameter_bank.grad[1, 0],
            torch.zeros(2),
        )

        with self.assertRaisesRegex(ValueError, "Probabilities must be provided"):
            mixture.compute_mixture(None, indices)

    def test_generator_clipping_weighting_and_top_one_math_are_exact(self) -> None:
        config = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=2,
                num_experts=2,
            ),
            generator_config=_generator_config(),
        )
        config.clip_parameter_option = ClipParameterOptions.BEFORE
        config.clip_range = 0.5
        mixture = config.build()
        input_weights = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, -1.0], [1.0, 0.5]]),
        )
        output_weights = (
            torch.tensor([[0.5, 2.0], [-1.0, 1.0]]),
            torch.tensor([[1.0, -2.0], [3.0, 0.25]]),
        )
        with torch.no_grad():
            for expert, weight in zip(
                mixture.input_vector_generator.expert_modules,
                input_weights,
                strict=True,
            ):
                expert[0].model.weight_params.copy_(weight)
            for expert, weight in zip(
                mixture.output_vector_generator.expert_modules,
                output_weights,
                strict=True,
            ):
                expert[0].model.weight_params.copy_(weight)

        inputs = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        expected_parts = []
        for sample in range(2):
            routes = []
            for expert in range(2):
                left = (inputs[sample] @ input_weights[expert]).clamp(-0.5, 0.5)
                right = (inputs[sample] @ output_weights[expert]).clamp(-0.5, 0.5)
                routes.append(torch.outer(left, right))
            expected_parts.append(
                probabilities[sample, 0] * routes[0]
                + probabilities[sample, 1] * routes[1]
            )
        expected = torch.stack(expected_parts)

        output, loss = mixture.compute_mixture(probabilities, None, inputs)

        torch.testing.assert_close(output, expected)
        self.assertEqual(loss.item(), 0.0)
        already_grouped_inputs = torch.tensor([[[2.0, -1.0], [0.25, 0.75]]])
        already_grouped_outputs = torch.tensor([[[1.5, -2.0], [-0.5, 1.0]]])
        torch.testing.assert_close(
            mixture._GeneratorWeightsMixture__compute_outer_product(
                already_grouped_inputs,
                already_grouped_outputs,
            ),
            torch.einsum(
                "bki,bkj->bkij",
                already_grouped_inputs.clamp(-0.5, 0.5),
                already_grouped_outputs.clamp(-0.5, 0.5),
            ),
        )

        after_config = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=2,
                num_experts=2,
            ),
            generator_config=_generator_config(),
        )
        after_config.clip_parameter_option = ClipParameterOptions.AFTER
        after_config.clip_range = 0.25
        after = after_config.build()
        with torch.no_grad():
            for expert, weight in zip(
                after.input_vector_generator.expert_modules,
                input_weights,
                strict=True,
            ):
                expert[0].model.weight_params.copy_(weight)
            for expert, weight in zip(
                after.output_vector_generator.expert_modules,
                output_weights,
                strict=True,
            ):
                expert[0].model.weight_params.copy_(weight)
        after_parts = []
        for sample in range(2):
            routes = []
            for expert in range(2):
                left = inputs[sample] @ input_weights[expert]
                right = inputs[sample] @ output_weights[expert]
                routes.append(torch.outer(left, right).clamp(-0.25, 0.25))
            after_parts.append(
                probabilities[sample, 0] * routes[0]
                + probabilities[sample, 1] * routes[1]
            )
        torch.testing.assert_close(
            after.compute_mixture(probabilities, None, inputs)[0],
            torch.stack(after_parts),
        )

        top_one_config = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=3,
                weighted=False,
            ),
            generator_config=_generator_config(top_k=1, num_experts=3),
        )
        top_one = top_one_config.build()
        top_one_input_weights = (
            torch.eye(2),
            torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[4.0, -1.0], [2.0, 3.0]]),
        )
        top_one_output_weights = (
            torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
            torch.tensor([[0.0, 2.0], [1.0, 0.0]]),
            torch.tensor([[-2.0, 1.0], [3.0, 2.0]]),
        )
        with torch.no_grad():
            for expert, weight in zip(
                top_one.input_vector_generator.expert_modules,
                top_one_input_weights,
                strict=True,
            ):
                expert[0].model.weight_params.copy_(weight)
            for expert, weight in zip(
                top_one.output_vector_generator.expert_modules,
                top_one_output_weights,
                strict=True,
            ):
                expert[0].model.weight_params.copy_(weight)
        selected = torch.tensor([1, 0])
        top_one_probabilities = torch.tensor([0.9, 0.8])
        top_one_inputs = inputs.detach().clone().requires_grad_()
        top_one_output, top_one_loss = top_one.compute_mixture(
            top_one_probabilities,
            selected,
            top_one_inputs,
        )

        torch.testing.assert_close(
            top_one_output,
            torch.tensor(
                [
                    [[4.0, 4.0], [4.0, 4.0]],
                    [[1.0, 0.5], [-0.5, -0.25]],
                ]
            ),
        )
        self.assertEqual(top_one_loss.item(), 0.0)
        top_one_output.sum().backward()
        self.assertTrue(torch.isfinite(top_one_inputs.grad).all())
        self.assertGreater(top_one_inputs.grad.abs().sum().item(), 0.0)
        for generators in (
            top_one.input_vector_generator.expert_modules,
            top_one.output_vector_generator.expert_modules,
        ):
            for selected_expert in generators[:2]:
                gradient = selected_expert[0].model.weight_params.grad
                self.assertIsNotNone(gradient)
                self.assertGreater(gradient.abs().sum().item(), 0.0)
            self.assertIsNone(generators[2][0].model.weight_params.grad)


class ParametricLayerBehavioralContractTests(unittest.TestCase):
    def test_shared_dense_router_affine_equation_state_and_optimizer_are_exact(
        self,
    ) -> None:
        weight_config = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2)
        )
        bias_config = MatrixBiasMixtureConfig(**_mixture_kwargs(top_k=2, num_experts=2))
        model = ParametricLayer(
            _parametric_config(
                weight_config=weight_config,
                bias_config=bias_config,
                routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
            )
        ).double()
        router_weights = torch.tensor(
            [[1.0, -0.5], [0.25, 0.75]],
            dtype=torch.float64,
        )
        weight_bank = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[-1.0, 3.0], [2.0, 0.5]],
            ],
            dtype=torch.float64,
        )
        bias_bank = torch.tensor(
            [[0.5, -1.0], [2.0, 0.25]],
            dtype=torch.float64,
        )
        with torch.no_grad():
            model.weights_router.model[0].model.weight_params.copy_(router_weights)
            model.weight_mixture_model.parameter_bank.copy_(weight_bank)
            model.bias_mixture_model.parameter_bank.copy_(bias_bank)

        inputs = torch.tensor(
            [[1.0, 2.0], [-2.0, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        probabilities = torch.softmax(inputs.detach() @ router_weights, dim=-1)
        generated_weights = torch.einsum("be,eij->bij", probabilities, weight_bank)
        generated_bias = probabilities @ bias_bank
        expected = (
            torch.einsum(
                "bi,bij->bj",
                inputs.detach(),
                generated_weights,
            )
            + generated_bias
        )

        output, skip_mask, loss = model(inputs)

        torch.testing.assert_close(output, expected)
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNone(model.bias_router)
        output.square().mean().backward()
        for parameter in (
            inputs,
            model.weights_router.model[0].model.weight_params,
            model.weight_mixture_model.parameter_bank,
            model.bias_mixture_model.parameter_bank,
        ):
            gradient = parameter.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.weight_mixture_model.parameter_bank.detach().clone()
        optimizer.step()
        self.assertFalse(
            torch.equal(before, model.weight_mixture_model.parameter_bank.detach())
        )

        restored = ParametricLayer(
            _parametric_config(
                weight_config=MatrixWeightsMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                ),
                bias_config=MatrixBiasMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                ),
                routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
            )
        ).double()
        restored.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        restored.eval()
        with torch.no_grad():
            expected_after_step = model(inputs.detach())[0]
            actual_after_step = restored(inputs.detach())[0]
        torch.testing.assert_close(actual_after_step, expected_after_step)

    def test_dense_unweighted_matrix_layer_broadcasts_exact_affine_parameters(
        self,
    ) -> None:
        weight_config = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=4,
                top_k=3,
                num_experts=3,
                weighted=False,
            )
        )
        bias_config = MatrixBiasMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=4,
                top_k=3,
                num_experts=3,
                weighted=False,
            )
        )
        model = ParametricLayer(
            _parametric_config(
                weight_config=weight_config,
                bias_config=bias_config,
                routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
                input_dim=2,
                output_dim=4,
                top_k=3,
                num_experts=3,
            )
        ).double()
        weight_bank = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0, 7.0]],
                [[3.0, 5.0, 7.0, 11.0], [13.0, 17.0, 19.0, 23.0]],
                [[5.0, 8.0, 13.0, 21.0], [34.0, 55.0, 89.0, 144.0]],
            ],
            dtype=torch.float64,
        )
        bias_bank = torch.tensor(
            [
                [1.0, 2.0, 3.0, 5.0],
                [7.0, 11.0, 13.0, 17.0],
                [19.0, 23.0, 29.0, 31.0],
            ],
            dtype=torch.float64,
        )
        with torch.no_grad():
            model.weights_router.model[0].model.weight_params.copy_(
                torch.tensor(
                    [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]],
                    dtype=torch.float64,
                )
            )
            model.weight_mixture_model.parameter_bank.copy_(weight_bank)
            model.bias_mixture_model.parameter_bank.copy_(bias_bank)
        inputs = torch.tensor(
            [[1.0, 2.0], [-3.0, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        expected = inputs.detach() @ weight_bank.sum(dim=0) + bias_bank.sum(dim=0)

        output, skip_mask, loss = model(inputs)

        torch.testing.assert_close(output, expected)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device, inputs.device)
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.item(), 0.0)
        output.square().mean().backward()
        for parameter in (
            inputs,
            model.weight_mixture_model.parameter_bank,
            model.bias_mixture_model.parameter_bank,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)
        self.assertIsNone(
            model.weights_router.model[0].model.weight_params.grad,
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        weight_before = model.weight_mixture_model.parameter_bank.detach().clone()
        router_before = (
            model.weights_router.model[0].model.weight_params.detach().clone()
        )
        optimizer.step()
        self.assertFalse(
            torch.equal(
                weight_before,
                model.weight_mixture_model.parameter_bank.detach(),
            )
        )
        torch.testing.assert_close(
            model.weights_router.model[0].model.weight_params.detach(),
            router_before,
        )

    def test_singleton_vector_public_path_preserves_samples_and_gradients(
        self,
    ) -> None:
        model = ParametricLayer(
            _parametric_config(
                weight_config=VectorWeightsMixtureConfig(
                    **_mixture_kwargs(
                        input_dim=2,
                        output_dim=2,
                        top_k=1,
                        num_experts=1,
                    )
                ),
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=1,
            )
        ).double()
        weight_bank = torch.tensor(
            [
                [[2.0, 3.0]],
                [[5.0, 7.0]],
            ],
            dtype=torch.float64,
        )
        with torch.no_grad():
            model.weight_mixture_model.parameter_bank.copy_(weight_bank)
        inputs = torch.tensor(
            [[1.0, 2.0], [-3.0, 0.5], [4.0, -2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        expected = inputs.detach() @ weight_bank.squeeze(1)

        output, skip_mask, loss = model(inputs)

        torch.testing.assert_close(output, expected)
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device, inputs.device)
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        bank_gradient = model.weight_mixture_model.parameter_bank.grad
        self.assertIsNotNone(bank_gradient)
        self.assertTrue(torch.isfinite(bank_gradient).all())
        self.assertGreater(bank_gradient.abs().sum().item(), 0.0)

    def test_vector_skip_mask_helpers_preserve_batch_and_axis_semantics(self) -> None:
        vector_config = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=2,
            )
        )
        model = ParametricLayer(
            _parametric_config(
                weight_config=vector_config,
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=2,
            )
        )
        flatten = model._ParametricLayer__flatten_vector_skip_mask
        unflatten = model._ParametricLayer__unflatten_vector_skip_mask
        reshape = model._ParametricLayer__reshape_vector_sample

        batch_mask = torch.tensor([[1.0], [0.0]])
        torch.testing.assert_close(
            flatten(batch_mask, 2, 2),
            torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        )
        axis_mask = torch.tensor([[1.0, 0.0], [0.5, 0.25]])
        torch.testing.assert_close(
            flatten(axis_mask, 2, 2),
            torch.tensor([[1.0], [0.0], [0.5], [0.25]]),
        )
        self.assertIsNone(unflatten(None, 2, 2))
        torch.testing.assert_close(
            unflatten(
                torch.tensor([[1.0], [0.0], [0.5], [0.25]]),
                2,
                2,
            ),
            torch.tensor([[0.0], [0.25]]),
        )
        self.assertIsNone(reshape(None, 2, 2))
        torch.testing.assert_close(
            reshape(torch.tensor([0, 1, 2, 3]), 2, 2),
            torch.tensor([[0, 2], [1, 3]]),
        )
        torch.testing.assert_close(
            reshape(torch.arange(8).reshape(4, 2), 2, 2),
            torch.tensor(
                [
                    [[0, 1], [4, 5]],
                    [[2, 3], [6, 7]],
                ]
            ),
        )

        probabilities, indices, returned_mask, loss = (
            model._ParametricLayer__sample_bias_probabilities_and_indices(
                torch.ones(2, 2),
                batch_mask,
            )
        )
        self.assertIsNone(probabilities)
        self.assertIsNone(indices)
        self.assertIs(returned_mask, batch_mask)
        self.assertEqual(loss.item(), 0.0)

    def test_abstract_handlers_and_validated_fallbacks_fail_exactly(self) -> None:
        cfg = _parametric_config()
        base_handler = ParameterHandlerBase(cfg)
        with self.assertRaisesRegex(
            NotImplementedError,
            "_init_shared_sampler.*child class",
        ):
            base_handler._init_shared_sampler()
        with self.assertRaisesRegex(
            NotImplementedError,
            "_init_independent_sampler.*child class",
        ):
            base_handler._init_independent_sampler()

        vector_cfg = _parametric_config(
            weight_config=VectorWeightsMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=1,
                    num_experts=2,
                )
            ),
            top_k=1,
            num_experts=2,
        )
        vector_handler = VectorParameterHandler(vector_cfg)
        with self.assertRaisesRegex(
            ValueError,
            "VectorWeightsMixtureConfig does not support SHARED_ROUTER",
        ):
            vector_handler._init_shared_sampler()

        model = ParametricLayer(cfg)
        model.weight_mixture_config = object()
        with self.assertRaisesRegex(
            TypeError,
            "weight_mixture_config must be a supported parametric weight config",
        ):
            model.get_parameter_handler()

        matrix_bias = MatrixBiasMixtureConfig(**_mixture_kwargs(top_k=2, num_experts=3))
        fallback = ParametricLayer(
            _parametric_config(
                bias_config=matrix_bias,
                top_k=2,
                num_experts=3,
            )
        )
        fallback.bias_router = None
        probabilities = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        indices = torch.tensor([[0, 1], [1, 0]])
        bias, _, loss = fallback._ParametricLayer__generate_bias_parameters(
            torch.ones(2, 2),
            None,
            probabilities,
            indices,
        )
        self.assertEqual(bias.shape, (2, 2))
        self.assertEqual(loss.item(), 0.0)


class ParametricValidationBehavioralContractTests(unittest.TestCase):
    def test_runtime_and_mixture_guards_report_exact_failures(self) -> None:
        with self.assertRaisesRegex(TypeError, "input_batch must be a Tensor"):
            AdaptiveMixtureValidator.validate_input_batch_2d([[1.0]])
        with self.assertRaisesRegex(ValueError, "Input batch must be a 2D tensor"):
            AdaptiveMixtureValidator.validate_input_batch_2d(torch.ones(1, 1, 1))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            AdaptiveMixtureValidator._validate_positive_integer("top_k", True)
        weighted_cfg = MatrixWeightsMixtureConfig(**_mixture_kwargs())
        with self.assertRaisesRegex(
            ValueError,
            "weighted_parameters_flag is True",
        ):
            AdaptiveMixtureValidator.validate_weighted_probabilities(
                weighted_cfg,
                None,
            )

        with self.assertRaisesRegex(TypeError, "input_batch must be a Tensor"):
            ParametricLayerValidator.validate_forward_inputs([[1.0]], 2)
        with self.assertRaisesRegex(ValueError, "feature dimension must match"):
            ParametricLayerValidator.validate_forward_inputs(torch.ones(2, 3), 2)

        model = ParametricLayer(_parametric_config())
        model.router_config = object()
        with self.assertRaisesRegex(TypeError, "router_config must be a RouterConfig"):
            ParametricLayerValidator._validate_router_and_sampler_configs(model)
        model.router_config = _router_config(2, 2)
        model.sampler_config = object()
        with self.assertRaisesRegex(
            TypeError,
            "sampler_config must be a SamplerConfig",
        ):
            ParametricLayerValidator._validate_router_and_sampler_configs(model)
        model.adaptive_augmentation_config = object()
        with self.assertRaisesRegex(
            TypeError,
            "adaptive_augmentation_config must be an",
        ):
            ParametricLayerValidator._validate_adaptive_augmentation_config(model)

        handler = ParameterHandlerBase(_parametric_config())
        handler.cfg = object()
        with self.assertRaisesRegex(
            TypeError,
            "ParameterHandlerBase cfg must be ParametricLayerConfig",
        ):
            ParametricHandlerValidator._validate_parameter_handler(handler)
        with self.assertRaisesRegex(
            TypeError,
            "state must be a LayerState",
        ):
            ParametricHandlerValidator.validate_state(object())
        with self.assertRaisesRegex(TypeError, "state.hidden must be a Tensor"):
            ParametricHandlerValidator.validate_state(LayerState(hidden=object()))

        missing_router = ParameterHandlerBase(_parametric_config())
        missing_router.router_config = None
        with self.assertRaisesRegex(
            ValueError,
            "router_config and sampler_config are required",
        ):
            ParametricHandlerValidator._validate_parameter_handler(missing_router)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            ParametricLayerValidator._validate_positive_integer("input_dim", True)
        with self.assertRaisesRegex(
            TypeError,
            "bias_mixture_config must be None or a bias mixture config",
        ):
            ParametricLayerValidator._validate_bias_mixture_config(object())

        conflicting = ParametricLayer(
            _parametric_config(
                bias_config=MatrixBiasMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                )
            )
        )
        conflicting.adaptive_augmentation_config.bias_config = object()
        with self.assertRaisesRegex(ValueError, "bias_config can only be used"):
            ParametricLayerValidator._validate_adaptive_augmentation_config(conflicting)

        handler_config = ParametricLayerHandlerConfig(
            input_dim=2,
            output_dim=2,
            activation=ActivationOptions.DISABLED,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=_parametric_config(),
        )
        layer_handler = handler_config.build()
        layer_handler.cfg = object()
        with self.assertRaisesRegex(
            TypeError,
            "ParametricLayerHandler cfg must be ParametricLayerHandlerConfig",
        ):
            ParametricHandlerValidator._validate_layer_handler(layer_handler)
        layer_handler.cfg = handler_config
        layer_handler.layer_model_config = object()
        with self.assertRaisesRegex(
            TypeError,
            "ParametricLayerHandler.layer_model_config must be ParametricLayerConfig",
        ):
            ParametricHandlerValidator._validate_layer_handler(layer_handler)

    def test_generator_configuration_relationships_are_exact(self) -> None:
        top_k_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(top_k=1),
        )
        with self.assertRaisesRegex(ValueError, "generator_config.top_k must match"):
            AdaptiveMixtureValidator._validate_generator_config(top_k_mismatch)

        num_experts_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(top_k=2, num_experts=3),
        )
        with self.assertRaisesRegex(
            ValueError,
            "generator_config.num_experts must match",
        ):
            AdaptiveMixtureValidator._validate_generator_config(num_experts_mismatch)

        bad_type = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=object(),
        )
        with self.assertRaisesRegex(
            TypeError,
            "generator_config must be a MixtureOfExpertsConfig",
        ):
            AdaptiveMixtureValidator._validate_generator_config(bad_type)

        sampler_top_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(sampler_config=_sampler_config(1, 2)),
        )
        with self.assertRaisesRegex(
            ValueError,
            "sampler_config.top_k must match",
        ):
            AdaptiveMixtureValidator._validate_generator_config(sampler_top_mismatch)

        sampler_count_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(sampler_config=_sampler_config(2, 3)),
        )
        with self.assertRaisesRegex(
            ValueError,
            "sampler_config.num_experts must match",
        ):
            AdaptiveMixtureValidator._validate_generator_config(sampler_count_mismatch)

        invalid_top_k = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=3, num_experts=2)
        )
        with self.assertRaisesRegex(ValueError, "top_k cannot exceed num_experts"):
            AdaptiveMixtureValidator._validate_top_k(invalid_top_k)


if __name__ == "__main__":
    unittest.main()
