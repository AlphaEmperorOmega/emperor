import unittest

import torch

from emperor.config import ModelConfig
from emperor.experts import RoutingInitializationMode
from emperor.parametric import (
    AdaptiveMixtureConfig,
    AdaptiveRouterOptions,
    ClipParameterOptions,
    GeneratorBiasMixture,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixture,
    GeneratorWeightsMixtureConfig,
    MatrixBiasMixture,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixture,
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerConfig,
    VectorRouterConfig,
    VectorWeightsMixture,
    VectorWeightsMixtureConfig,
)
from emperor.parametric._mixtures.validation import AdaptiveMixtureValidator
from emperor.parametric._mixtures.vector import VectorMixtureBase
from emperor.sampler import RouterConfig
from tests.unit.test_expert_mutation_contracts import _halting_expert_stack
from tests.unit.test_parametric_behavioral_contracts import (
    _generator_config,
    _mixture_kwargs,
    _parametric_config,
    _router_config,
    _sampler_config,
)


def _owned_halting_generator_config(
    *,
    input_dim: int = 2,
    top_k: int = 1,
    num_experts: int = 2,
):
    sampler_config = _sampler_config(top_k, num_experts)
    sampler_config.router_config = _router_config(input_dim, num_experts)
    generator_config = _generator_config(
        input_dim=input_dim,
        top_k=top_k,
        num_experts=num_experts,
        sampler_config=sampler_config,
    )
    generator_config.routing_initialization_mode = RoutingInitializationMode.LAYER
    generator_config.expert_model_config = _halting_expert_stack(input_dim)
    return generator_config


class ParametricRuntimeMutationContractTests(unittest.TestCase):
    def assert_exact_error(
        self,
        exception_type: type[Exception],
        expected_message: str,
        callback,
    ) -> None:
        with self.assertRaises(exception_type) as error:
            callback()
        self.assertEqual(str(error.exception), expected_message)

    def test_mixture_constructors_honor_wrappers_overrides_and_exact_state(
        self,
    ) -> None:
        base_weights = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=2,
            )
        )
        wrapper = ModelConfig()
        wrapper.mixture_model_config = base_weights
        weights = MatrixWeightsMixture(
            wrapper,
            AdaptiveMixtureConfig(input_dim=3, output_dim=4),
        )

        self.assertIs(weights.main_cfg, wrapper)
        self.assertEqual(weights.input_dim, 3)
        self.assertEqual(weights.output_dim, 4)
        self.assertEqual(weights.top_k, 1)
        self.assertEqual(weights.num_experts, 2)
        self.assertTrue(weights.weighted_parameters_flag)
        self.assertEqual(
            weights.clip_parameter_option,
            ClipParameterOptions.DISABLED,
        )
        self.assertEqual(weights.clip_range, 1.0)
        self.assertEqual(weights.depth_dim, 2)
        self.assertEqual(weights.parameter_mixture_dim, -2)
        self.assertEqual(weights.probability_shape, (-1, 1, 1))
        self.assertEqual(weights.parameter_bank_shape, (2, 3, 4))
        self.assertEqual(tuple(weights.parameter_bank.shape), (2, 3, 4))

        explicit_main = ModelConfig(input_dim=91)
        base_with_main = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=1, num_experts=2)
        )
        base_with_main.override_config = explicit_main
        wrapper_with_main = ModelConfig()
        wrapper_with_main.mixture_model_config = base_with_main
        with_main = MatrixWeightsMixture(wrapper_with_main)
        self.assertIs(with_main.main_cfg, explicit_main)

        bias = MatrixBiasMixture(
            MatrixBiasMixtureConfig(
                **_mixture_kwargs(output_dim=2, top_k=1, num_experts=2)
            ),
            AdaptiveMixtureConfig(output_dim=4),
        )
        self.assertEqual(bias.output_dim, 4)
        self.assertEqual(bias.depth_dim, 2)
        self.assertEqual(bias.parameter_mixture_dim, -1)
        self.assertEqual(bias.probability_shape, (-1, 1))
        self.assertEqual(bias.parameter_bank_shape, (2, 4))
        self.assertEqual(tuple(bias.parameter_bank.shape), (2, 4))

        vector = VectorWeightsMixture(
            VectorWeightsMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=3,
                    num_experts=3,
                )
            ),
            AdaptiveMixtureConfig(input_dim=3, output_dim=3),
        )
        self.assertEqual(vector.input_dim, 3)
        self.assertEqual(vector.output_dim, 3)
        self.assertEqual(vector.depth_dim, 3)
        self.assertEqual(vector.range_dim, 3)
        self.assertEqual(vector.parameter_mixture_dim, -2)
        self.assertEqual(vector.parameter_bank_shape, (3, 3, 3))
        self.assertEqual(tuple(vector.parameter_bank.shape), (3, 3, 3))
        self.assertEqual(tuple(vector.select_range.shape), (1, 3))
        torch.testing.assert_close(
            vector.select_range,
            torch.tensor([[0, 1, 2]]),
        )

        fully_selected = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=2,
                num_experts=2,
                weighted=False,
            )
        ).build()
        fully_selected_bank = torch.arange(8.0).reshape(2, 2, 2)
        with torch.no_grad():
            fully_selected.parameter_bank.copy_(fully_selected_bank)
        fully_selected_indices = torch.tensor([[[0, 1]], [[1, 0]]])
        torch.testing.assert_close(
            fully_selected.compute_mixture(
                None,
                fully_selected_indices,
            ),
            torch.tensor([[[2.0, 4.0], [10.0, 12.0]]]),
        )

        sparse_vector = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=3,
                output_dim=3,
                top_k=2,
                num_experts=3,
            )
        ).build()
        self.assertEqual(tuple(sparse_vector.select_range.shape), (1, 3, 1))
        torch.testing.assert_close(
            sparse_vector.select_range,
            torch.tensor([[[0], [1], [2]]]),
        )

        weight_bank = torch.arange(24.0).reshape(2, 3, 4)
        with torch.no_grad():
            weights.parameter_bank.copy_(weight_bank)
        probabilities = torch.tensor([0.25, 0.75])
        indices = torch.tensor([1, 0])
        torch.testing.assert_close(
            weights.compute_mixture(probabilities, indices),
            torch.stack((0.25 * weight_bank[1], 0.75 * weight_bank[0])),
        )

        bias_bank = torch.arange(8.0).reshape(2, 4)
        with torch.no_grad():
            bias.parameter_bank.copy_(bias_bank)
        torch.testing.assert_close(
            bias.compute_mixture(probabilities, indices),
            torch.stack((0.25 * bias_bank[1], 0.75 * bias_bank[0])),
        )

    def test_vector_router_honors_overrides_and_computes_exact_logits(
        self,
    ) -> None:
        config = VectorRouterConfig(
            input_dim=2,
            num_experts=2,
            noisy_topk_flag=False,
            model_config=_router_config(2, 2).model_config,
        )
        router = config.build(RouterConfig(input_dim=3, num_experts=2))
        parameter_bank = torch.tensor(
            [
                [[1.0, -1.0], [0.0, 2.0], [3.0, 0.5]],
                [[-2.0, 1.0], [4.0, -3.0], [0.25, 2.0]],
                [[0.5, 2.0], [-1.0, 0.75], [2.0, -4.0]],
            ]
        )
        with torch.no_grad():
            router.parameter_bank.copy_(parameter_bank)
        inputs = torch.tensor(
            [[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]],
            requires_grad=True,
        )
        expected = torch.empty(2, 3, 2)
        for sample in range(2):
            for axis in range(3):
                expected[sample, axis] = sum(
                    inputs.detach()[sample, input_axis]
                    * parameter_bank[input_axis, axis]
                    for input_axis in range(3)
                )

        logits = router.compute_logit_scores(inputs)

        self.assertEqual(router.input_dim, 3)
        self.assertEqual(router.num_experts, 2)
        self.assertEqual(tuple(router.parameter_bank.shape), (3, 3, 2))
        torch.testing.assert_close(logits, expected)
        logits.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(router.parameter_bank.grad).all())
        self.assertGreater(router.parameter_bank.grad.abs().sum().item(), 0.0)

    def test_generator_constructors_override_internal_dimensions_and_modes(
        self,
    ) -> None:
        generator_config = _generator_config(
            input_dim=5,
            top_k=2,
            num_experts=2,
        )
        generator_config.compute_expert_mixture_flag = True
        generator_config.weighted_parameters_flag = True
        generator_config.routing_initialization_mode = RoutingInitializationMode.LAYER
        owned_sampler = _sampler_config(2, 2)
        owned_sampler.router_config = _router_config(5, 2)
        generator_config.sampler_config = owned_sampler
        mixture = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=3,
                top_k=2,
                num_experts=2,
            ),
            generator_config=generator_config,
        ).build()

        self.assertEqual(mixture.range_dim, 2)
        self.assertEqual(mixture.parameter_mixture_dim, -2)
        self.assertEqual(mixture.probability_shape, (-1, 2, 1, 1))
        for generator, output_dim in (
            (mixture.input_vector_generator, 2),
            (mixture.output_vector_generator, 3),
        ):
            self.assertEqual(generator.input_dim, 2)
            self.assertEqual(generator.output_dim, output_dim)
            self.assertEqual(generator.top_k, 2)
            self.assertEqual(generator.num_experts, 2)
            self.assertFalse(generator.compute_expert_mixture_flag)
            self.assertFalse(generator.weighted_parameters_flag)
            self.assertEqual(
                generator.routing_initialization_mode,
                RoutingInitializationMode.DISABLED,
            )
            self.assertIsNone(generator.sampler)

        overridden_weights = GeneratorWeightsMixture(
            GeneratorWeightsMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=2,
                    num_experts=2,
                ),
                generator_config=_generator_config(
                    input_dim=2,
                    top_k=2,
                    num_experts=2,
                ),
            ),
            AdaptiveMixtureConfig(input_dim=3, output_dim=4),
        )
        self.assertEqual(overridden_weights.input_dim, 3)
        self.assertEqual(overridden_weights.output_dim, 4)
        self.assertEqual(overridden_weights.range_dim, 3)
        self.assertEqual(
            overridden_weights.input_vector_generator.input_dim,
            3,
        )
        self.assertEqual(
            overridden_weights.input_vector_generator.output_dim,
            3,
        )
        self.assertEqual(
            overridden_weights.output_vector_generator.input_dim,
            3,
        )
        self.assertEqual(
            overridden_weights.output_vector_generator.output_dim,
            4,
        )

        overridden_bias = GeneratorBiasMixture(
            GeneratorBiasMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=2,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=_generator_config(
                    input_dim=2,
                    top_k=2,
                    num_experts=2,
                ),
            ),
            AdaptiveMixtureConfig(input_dim=3, output_dim=4),
        )
        self.assertEqual(overridden_bias.input_dim, 3)
        self.assertEqual(overridden_bias.output_dim, 4)
        self.assertEqual(overridden_bias.range_dim, 4)
        self.assertEqual(overridden_bias.bias_generator.input_dim, 3)
        self.assertEqual(overridden_bias.bias_generator.output_dim, 4)

    def test_generator_bias_constructor_honors_explicit_overrides(self) -> None:
        mixture = GeneratorBiasMixture(
            GeneratorBiasMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=2,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=_generator_config(
                    input_dim=2,
                    top_k=2,
                    num_experts=2,
                ),
            ),
            AdaptiveMixtureConfig(input_dim=3, output_dim=4),
        )

        self.assertEqual(mixture.input_dim, 3)
        self.assertEqual(mixture.output_dim, 4)
        self.assertEqual(mixture.range_dim, 4)
        self.assertEqual(mixture.bias_generator.input_dim, 3)
        self.assertEqual(mixture.bias_generator.output_dim, 4)

    def test_parametric_constructor_honors_wrapper_and_explicit_overrides(
        self,
    ) -> None:
        base_config = _parametric_config(
            weight_config=MatrixWeightsMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=2,
                    num_experts=2,
                )
            ),
            bias_config=MatrixBiasMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=2,
                    num_experts=2,
                )
            ),
            input_dim=2,
            output_dim=2,
            top_k=2,
            num_experts=2,
        )
        wrapper = ModelConfig()
        wrapper.parameter_generator_model_config = base_config
        model = ParametricLayer(
            wrapper,
            ParametricLayerConfig(input_dim=3, output_dim=4),
        )

        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.output_dim, 4)
        self.assertIs(model.weight_mixture_config, model.cfg.weight_mixture_config)
        self.assertIs(model.bias_mixture_config, model.cfg.bias_mixture_config)
        self.assertEqual(model.adaptive_augmentation_model.input_dim, 3)
        self.assertEqual(model.adaptive_augmentation_model.output_dim, 4)
        self.assertEqual(model.weight_mixture_model.input_dim, 3)
        self.assertEqual(model.weight_mixture_model.output_dim, 4)
        self.assertEqual(
            model.weight_mixture_model.parameter_bank_shape,
            (2, 3, 4),
        )
        self.assertEqual(model.bias_mixture_model.input_dim, 3)
        self.assertEqual(model.bias_mixture_model.output_dim, 4)
        self.assertEqual(
            model.bias_mixture_model.parameter_bank_shape,
            (2, 4),
        )
        self.assertEqual(model.weights_router.input_dim, 3)
        self.assertEqual(model.bias_router.input_dim, 3)
        self.assertIs(model.parameter_handler.cfg, model.cfg)

        original_config = model.weight_mixture_config
        model.weight_mixture_config = object()
        self.assert_exact_error(
            TypeError,
            "weight_mixture_config must be a supported parametric weight config, "
            "got object.",
            model.get_parameter_handler,
        )
        model.weight_mixture_config = original_config

    def test_generator_bias_math_loss_and_gradients_are_exact(self) -> None:
        mixture = GeneratorBiasMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=3,
                top_k=2,
                num_experts=2,
                weighted=False,
            ),
            generator_config=_generator_config(
                input_dim=2,
                top_k=2,
                num_experts=2,
            ),
        ).build()
        expert_weights = (
            torch.tensor([[1.0, 0.0, 2.0], [0.0, -1.0, 0.5]]),
            torch.tensor([[-2.0, 1.0, 0.0], [3.0, 0.25, -1.0]]),
        )
        with torch.no_grad():
            for expert, weight in zip(
                mixture.bias_generator.expert_modules,
                expert_weights,
                strict=True,
            ):
                expert[0].model.weight_params.copy_(weight)
        inputs = torch.tensor(
            [[1.0, 2.0], [-2.0, 0.5]],
            requires_grad=True,
        )
        probabilities = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4]],
            requires_grad=True,
        )
        expected = torch.stack(
            tuple(
                sum(
                    probabilities.detach()[sample, expert_index]
                    * (inputs.detach()[sample] @ expert_weights[expert_index])
                    for expert_index in range(2)
                )
                for sample in range(2)
            )
        )

        output, loss = mixture.compute_mixture(probabilities, None, inputs)

        self.assertEqual(mixture.range_dim, 3)
        self.assertEqual(mixture.bias_generator.input_dim, 2)
        self.assertEqual(mixture.bias_generator.output_dim, 3)
        self.assertTrue(mixture.bias_generator.compute_expert_mixture_flag)
        self.assertTrue(mixture.bias_generator.weighted_parameters_flag)
        torch.testing.assert_close(output, expected)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        for tensor in (inputs, probabilities):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.0)
        for expert in mixture.bias_generator.expert_modules:
            gradient = expert[0].model.weight_params.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_singleton_generator_column_routing_is_exact_for_both_slots(
        self,
    ) -> None:
        generator_config = _generator_config(
            input_dim=2,
            top_k=1,
            num_experts=1,
        )
        weight_mixture = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=1,
                weighted=True,
            ),
            generator_config=generator_config,
        ).build()
        bias_mixture = GeneratorBiasMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=1,
                weighted=False,
            ),
            generator_config=_generator_config(
                input_dim=2,
                top_k=1,
                num_experts=1,
            ),
        ).build()
        input_transform = torch.eye(2)
        output_transform = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        bias_transform = torch.tensor([[1.0, 2.0], [-2.0, 0.25]])
        with torch.no_grad():
            weight_mixture.input_vector_generator.expert_modules[0][
                0
            ].model.weight_params.copy_(input_transform)
            weight_mixture.output_vector_generator.expert_modules[0][
                0
            ].model.weight_params.copy_(output_transform)
            bias_mixture.bias_generator.expert_modules[0][0].model.weight_params.copy_(
                bias_transform
            )
        inputs = torch.tensor(
            [[1.0, 2.0], [-3.0, 0.5]],
            requires_grad=True,
        )
        probabilities = torch.tensor(
            [[0.25], [0.75]],
            requires_grad=True,
        )
        indices = torch.zeros(2, 1, dtype=torch.long)
        input_vectors = inputs.detach() @ input_transform
        output_vectors = inputs.detach() @ output_transform
        expected_weights = probabilities.detach().reshape(-1, 1, 1) * torch.einsum(
            "bi,bj->bij",
            input_vectors,
            output_vectors,
        )
        expected_bias = probabilities.detach() * (inputs.detach() @ bias_transform)

        weights, weight_loss = weight_mixture.compute_mixture(
            probabilities,
            indices,
            inputs,
        )
        bias, bias_loss = bias_mixture.compute_mixture(
            probabilities,
            indices,
            inputs,
        )

        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(bias, expected_bias)
        self.assertEqual(weight_loss.item(), 0.0)
        self.assertEqual(bias_loss.item(), 0.0)
        (weights.square().sum() + bias.square().sum()).backward()
        for tensor in (
            inputs,
            probabilities,
            weight_mixture.input_vector_generator.expert_modules[0][
                0
            ].model.weight_params,
            weight_mixture.output_vector_generator.expert_modules[0][
                0
            ].model.weight_params,
            bias_mixture.bias_generator.expert_modules[0][0].model.weight_params,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.0)

    def test_generator_weight_auxiliary_losses_are_added(self) -> None:
        generator_config = _generator_config(
            input_dim=2,
            top_k=1,
            num_experts=2,
        )
        generator_config.expert_model_config = _halting_expert_stack(2)
        mixture = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=2,
                weighted=False,
            ),
            generator_config=generator_config,
        ).build()
        mixture.eval()
        inputs = torch.tensor(
            [[1.0, 2.0], [3.0, -1.0], [2.0, 0.5], [-2.0, 4.0]],
            requires_grad=True,
        )
        probabilities = torch.ones(4)
        indices = torch.tensor([0, 1, 0, 1])
        expected_input_vectors, _, expected_input_loss = mixture.input_vector_generator(
            inputs, probabilities, indices
        )
        expected_output_vectors, _, expected_output_loss = (
            mixture.output_vector_generator(inputs, probabilities, indices)
        )
        expected_weights = torch.einsum(
            "bi,bj->bij",
            expected_input_vectors,
            expected_output_vectors,
        )

        weights, loss = mixture.compute_mixture(
            probabilities,
            indices,
            inputs,
        )

        self.assertGreater(expected_input_loss.item(), 0.0)
        self.assertGreater(expected_output_loss.item(), 0.0)
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(
            loss,
            expected_input_loss + expected_output_loss,
        )
        (weights.square().mean() + loss).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)

    def test_parametric_layer_adds_real_weight_and_bias_generator_losses(
        self,
    ) -> None:
        weight_config = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=2,
                weighted=False,
            ),
            generator_config=_owned_halting_generator_config(),
        )
        bias_config = GeneratorBiasMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=1,
                num_experts=2,
                weighted=False,
            ),
            generator_config=_owned_halting_generator_config(),
        )
        model = ParametricLayer(
            _parametric_config(
                weight_config=weight_config,
                bias_config=bias_config,
                routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
                top_k=1,
                num_experts=2,
            )
        )
        model.eval()
        inputs = torch.tensor(
            [[1.0, 2.0], [3.0, -1.0], [2.0, 0.5], [-2.0, 4.0]],
            requires_grad=True,
        )
        expected_weights, expected_weight_loss = (
            model.weight_mixture_model.compute_mixture(None, None, inputs)
        )
        expected_bias, expected_bias_loss = model.bias_mixture_model.compute_mixture(
            None, None, inputs
        )
        expected_output = (
            torch.einsum(
                "bi,bij->bj",
                inputs.detach(),
                expected_weights.detach(),
            )
            + expected_bias.detach()
        )

        output, skip_mask, loss = model(inputs)

        self.assertIsNone(skip_mask)
        self.assertGreater(expected_weight_loss.item(), 0.0)
        self.assertGreater(expected_bias_loss.item(), 0.0)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(
            loss,
            expected_weight_loss + expected_bias_loss,
        )
        (output.square().mean() + loss).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)

    def test_independent_bias_router_adds_its_real_auxiliary_loss(self) -> None:
        config = _parametric_config(
            weight_config=MatrixWeightsMixtureConfig(
                **_mixture_kwargs(top_k=2, num_experts=3)
            ),
            bias_config=MatrixBiasMixtureConfig(
                **_mixture_kwargs(top_k=2, num_experts=3)
            ),
            routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            top_k=2,
            num_experts=3,
        )
        config.sampler_config.coefficient_of_variation_loss_weight = 1.0
        model = ParametricLayer(config)
        bias_router_parameters = torch.tensor([[3.0, 0.0, -2.0], [-1.0, 0.0, 1.0]])
        bias_bank = torch.tensor([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]])
        with torch.no_grad():
            model.bias_router.model[0].model.weight_params.copy_(bias_router_parameters)
            model.bias_mixture_model.parameter_bank.copy_(bias_bank)
        inputs = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
        logits = model.bias_router.compute_logit_scores(inputs)
        (
            expected_probabilities,
            expected_indices,
            expected_skip_mask,
            expected_loss,
        ) = model.sampler.sample_probabilities_and_indices(logits, None)
        expected_bias = model.bias_mixture_model.compute_mixture(
            expected_probabilities,
            expected_indices,
            inputs,
        )

        bias, skip_mask, loss = model._ParametricLayer__generate_bias_parameters(
            inputs,
            None,
            torch.zeros_like(expected_probabilities),
            expected_indices,
        )

        self.assertGreater(expected_loss.item(), 0.0)
        torch.testing.assert_close(bias, expected_bias)
        self.assertIs(skip_mask, expected_skip_mask)
        torch.testing.assert_close(loss, expected_loss)

    def test_independent_routers_drive_distinct_exact_weight_and_bias_mixtures(
        self,
    ) -> None:
        model = ParametricLayer(
            _parametric_config(
                weight_config=MatrixWeightsMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                ),
                bias_config=MatrixBiasMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                ),
                routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            )
        ).double()
        weight_router_parameters = torch.tensor(
            [[1.0, -0.5], [0.25, 0.75]],
            dtype=torch.float64,
        )
        bias_router_parameters = torch.tensor(
            [[-0.25, 1.5], [1.0, -0.75]],
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
            model.weights_router.model[0].model.weight_params.copy_(
                weight_router_parameters
            )
            model.bias_router.model[0].model.weight_params.copy_(bias_router_parameters)
            model.weight_mixture_model.parameter_bank.copy_(weight_bank)
            model.bias_mixture_model.parameter_bank.copy_(bias_bank)
        inputs = torch.tensor(
            [[1.0, 2.0], [-2.0, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        weight_probabilities = torch.softmax(
            inputs.detach() @ weight_router_parameters,
            dim=-1,
        )
        bias_probabilities = torch.softmax(
            inputs.detach() @ bias_router_parameters,
            dim=-1,
        )
        expected_weights = torch.einsum(
            "be,eij->bij",
            weight_probabilities,
            weight_bank,
        )
        expected_bias = bias_probabilities @ bias_bank
        expected = (
            torch.einsum(
                "bi,bij->bj",
                inputs.detach(),
                expected_weights,
            )
            + expected_bias
        )

        output, skip_mask, loss = model(inputs)

        torch.testing.assert_close(output, expected)
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        for parameter in (
            inputs,
            model.weights_router.model[0].model.weight_params,
            model.bias_router.model[0].model.weight_params,
            model.weight_mixture_model.parameter_bank,
            model.bias_mixture_model.parameter_bank,
        ):
            gradient = parameter.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_vector_public_path_preserves_noncontiguous_batches_and_skip_masks(
        self,
    ) -> None:
        config = _parametric_config(
            weight_config=VectorWeightsMixtureConfig(
                **_mixture_kwargs(
                    input_dim=2,
                    output_dim=2,
                    top_k=1,
                    num_experts=2,
                )
            ),
            input_dim=2,
            output_dim=2,
            top_k=1,
            num_experts=2,
        )
        config.sampler_config.threshold = 0.2
        model = ParametricLayer(config)
        router_bank = torch.zeros(2, 2, 2)
        router_bank[0, 0] = torch.tensor([2.0, -2.0])
        router_bank[1, 1] = torch.tensor([-1.0, 1.0])
        parameter_bank = torch.tensor(
            [
                [[1.0, 2.0], [10.0, 20.0]],
                [[-3.0, 1.0], [4.0, -2.0]],
            ]
        )
        with torch.no_grad():
            model.weights_router.parameter_bank.copy_(router_bank)
            model.weight_mixture_model.parameter_bank.copy_(parameter_bank)
        input_storage = torch.tensor(
            [[1.0, 9.0, 2.0, 8.0], [3.0, 7.0, -1.0, 6.0]],
            requires_grad=True,
        )
        inputs = input_storage[:, ::2]
        self.assertFalse(inputs.is_contiguous())
        skip_mask = torch.tensor([[1.0], [0.0]])
        active_logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        active_probabilities = torch.softmax(active_logits, dim=-1).amax(dim=-1)
        expected_active_weights = torch.stack(
            (
                active_probabilities[0] * parameter_bank[0, 0],
                active_probabilities[1] * parameter_bank[1, 1],
            )
        )
        expected = torch.stack(
            (
                torch.einsum(
                    "i,ij->j",
                    inputs.detach()[0],
                    expected_active_weights,
                ),
                torch.zeros(2),
            )
        )

        output, returned_skip_mask, loss = model(inputs, skip_mask)

        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(returned_skip_mask, skip_mask)
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(input_storage.grad).all())
        self.assertGreater(input_storage.grad[:, ::2].abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(model.weights_router.parameter_bank.grad).all())
        self.assertGreater(
            model.weights_router.parameter_bank.grad.abs().sum().item(),
            0.0,
        )
        self.assertTrue(
            torch.isfinite(model.weight_mixture_model.parameter_bank.grad).all()
        )
        self.assertGreater(
            model.weight_mixture_model.parameter_bank.grad.abs().sum().item(),
            0.0,
        )

    def test_mixture_weighting_decisions_and_exact_errors(self) -> None:
        matrix = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=3)
        ).build()
        matrix_indices = torch.tensor([[0, 1]])
        self.assert_exact_error(
            ValueError,
            "Probabilities must be provided when 'weighted_parameters_flag' "
            "is set to True.",
            lambda: matrix.compute_mixture(None, matrix_indices),
        )

        vector = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=2,
                num_experts=3,
            )
        ).build()
        vector_indices = torch.tensor([[[0, 1]], [[1, 2]]])
        self.assert_exact_error(
            ValueError,
            "Probabilities must be provided when 'weighted_parameters_flag' "
            "is set to True.",
            lambda: vector.compute_mixture(None, vector_indices),
        )
        self.assert_exact_error(
            NotImplementedError,
            "The method '_compute_weighted_parameters' must be implemented in "
            "the child class.",
            lambda: VectorMixtureBase._compute_weighted_parameters(
                vector,
                torch.ones(1),
                torch.ones(1),
            ),
        )

        unweighted_matrix = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(
                top_k=2,
                num_experts=3,
                weighted=False,
            )
        ).build()
        matrix_bank = torch.arange(12.0).reshape(3, 2, 2)
        with torch.no_grad():
            unweighted_matrix.parameter_bank.copy_(matrix_bank)
        ignored_matrix_probabilities = torch.tensor([[0.99, 0.01]])
        torch.testing.assert_close(
            unweighted_matrix.compute_mixture(
                ignored_matrix_probabilities,
                matrix_indices,
            ),
            (matrix_bank[0] + matrix_bank[1]).unsqueeze(0),
        )

        unweighted_vector = VectorWeightsMixtureConfig(
            **_mixture_kwargs(
                input_dim=2,
                output_dim=2,
                top_k=2,
                num_experts=3,
                weighted=False,
            )
        ).build()
        vector_bank = torch.arange(12.0).reshape(2, 3, 2)
        with torch.no_grad():
            unweighted_vector.parameter_bank.copy_(vector_bank)
        ignored_vector_probabilities = torch.tensor([[[0.99, 0.01]], [[0.2, 0.8]]])
        torch.testing.assert_close(
            unweighted_vector.compute_mixture(
                ignored_vector_probabilities,
                vector_indices,
            ),
            torch.stack(
                (
                    torch.stack(
                        (
                            vector_bank[0, 0] + vector_bank[0, 1],
                            vector_bank[1, 1] + vector_bank[1, 2],
                        )
                    ),
                )
            ),
        )

    def test_adaptive_mixture_validation_reports_exact_contracts(self) -> None:
        self.assert_exact_error(
            TypeError,
            "input_batch must be a Tensor, received list.",
            lambda: AdaptiveMixtureValidator.validate_input_batch_2d([[1.0]]),
        )
        self.assert_exact_error(
            ValueError,
            "Input batch must be a 2D tensor, got 3D with shape (1, 2, 3).",
            lambda: AdaptiveMixtureValidator.validate_input_batch_2d(
                torch.ones(1, 2, 3)
            ),
        )
        weighted_config = MatrixWeightsMixtureConfig(**_mixture_kwargs())
        self.assert_exact_error(
            ValueError,
            "Probabilities must be provided when weighted_parameters_flag is True.",
            lambda: AdaptiveMixtureValidator.validate_weighted_probabilities(
                weighted_config,
                None,
            ),
        )
        self.assert_exact_error(
            ValueError,
            "top_k must be a positive integer, received 0.",
            lambda: AdaptiveMixtureValidator._validate_positive_integer(
                "top_k",
                0,
            ),
        )

        equal_top_k = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2)
        )
        AdaptiveMixtureValidator._validate_top_k(equal_top_k)
        invalid_top_k = MatrixWeightsMixtureConfig(
            **_mixture_kwargs(top_k=3, num_experts=2)
        )
        self.assert_exact_error(
            ValueError,
            "top_k cannot exceed num_experts for AdaptiveMixtureConfig, "
            "received top_k=3, num_experts=2.",
            lambda: AdaptiveMixtureValidator._validate_top_k(invalid_top_k),
        )

        zero_clip = MatrixWeightsMixtureConfig(**_mixture_kwargs())
        zero_clip.clip_range = 0.0
        AdaptiveMixtureValidator._validate_clip_range(zero_clip)
        negative_clip = MatrixWeightsMixtureConfig(**_mixture_kwargs())
        negative_clip.clip_range = -0.25
        self.assert_exact_error(
            ValueError,
            "clip_range must be non-negative, received -0.25.",
            lambda: AdaptiveMixtureValidator._validate_clip_range(negative_clip),
        )

        invalid_vector = VectorWeightsMixtureConfig(
            **_mixture_kwargs(input_dim=2, output_dim=3)
        )
        self.assert_exact_error(
            ValueError,
            "input_dim and output_dim must match for VectorWeightsMixtureConfig, "
            "received input_dim=2, output_dim=3.",
            lambda: invalid_vector.build(),
        )

        valid_model = MatrixWeightsMixtureConfig(**_mixture_kwargs()).build()
        for name in ("input_dim", "output_dim", "top_k", "num_experts"):
            with self.subTest(dimension=name, value=0):
                original = getattr(valid_model, name)
                setattr(valid_model, name, 0)
                self.assert_exact_error(
                    ValueError,
                    f"{name} must be greater than 0, received 0",
                    lambda: AdaptiveMixtureValidator.validate(valid_model),
                )
                setattr(valid_model, name, original)
            with self.subTest(dimension=name, value=True):
                original = getattr(valid_model, name)
                setattr(valid_model, name, True)
                self.assert_exact_error(
                    ValueError,
                    f"{name} must be a positive integer, received True.",
                    lambda: AdaptiveMixtureValidator.validate(valid_model),
                )
                setattr(valid_model, name, original)

        bad_generator_type = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=object(),
        )
        self.assert_exact_error(
            TypeError,
            "generator_config must be a MixtureOfExpertsConfig for generator "
            "mixtures, got object.",
            lambda: bad_generator_type.build(),
        )

        generator_top_k_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(top_k=1, num_experts=2),
        )
        self.assert_exact_error(
            ValueError,
            "generator_config.top_k must match the mixture top_k, received 1 and 2.",
            lambda: generator_top_k_mismatch.build(),
        )

        generator_count_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(top_k=2, num_experts=3),
        )
        self.assert_exact_error(
            ValueError,
            "generator_config.num_experts must match the mixture num_experts, "
            "received 3 and 2.",
            lambda: generator_count_mismatch.build(),
        )

        sampler_top_k_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(
                top_k=2,
                num_experts=2,
                sampler_config=_sampler_config(1, 2),
            ),
        )
        self.assert_exact_error(
            ValueError,
            "generator_config.sampler_config.top_k must match the mixture top_k, "
            "received 1 and 2.",
            lambda: sampler_top_k_mismatch.build(),
        )

        sampler_count_mismatch = GeneratorWeightsMixtureConfig(
            **_mixture_kwargs(top_k=2, num_experts=2),
            generator_config=_generator_config(
                top_k=2,
                num_experts=2,
                sampler_config=_sampler_config(2, 3),
            ),
        )
        self.assert_exact_error(
            ValueError,
            "generator_config.sampler_config.num_experts must match the mixture "
            "num_experts, received 3 and 2.",
            lambda: sampler_count_mismatch.build(),
        )


if __name__ == "__main__":
    unittest.main()
