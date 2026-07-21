import unittest

import torch

from emperor.experts import RoutingInitializationMode
from emperor.layers import (
    ActivationOptions,
    LayerNormPositionOptions,
    LayerState,
)
from emperor.parametric import (
    AdaptiveRouterOptions,
    GeneratorWeightsMixtureConfig,
    MatrixBiasMixtureConfig,
    MatrixWeightsMixtureConfig,
    ParametricLayer,
    ParametricLayerHandler,
    ParametricLayerHandlerConfig,
    VectorWeightsMixtureConfig,
)
from emperor.parametric._handlers import (
    ParameterHandlerBase,
    VectorParameterHandler,
)
from emperor.sampler import RouterConfig
from tests.unit.test_parametric_behavioral_contracts import (
    _generator_config,
    _mixture_kwargs,
    _parametric_config,
    _router_config,
    _sampler_config,
)


def _handler_config(
    *,
    input_dim: int,
    output_dim: int,
    layer_model_config,
) -> ParametricLayerHandlerConfig:
    return ParametricLayerHandlerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        activation=ActivationOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )


def _owned_linear_generator_config():
    sampler_config = _sampler_config(top_k=1, num_experts=2)
    sampler_config.router_config = _router_config(input_dim=2, num_experts=2)
    generator_config = _generator_config(
        input_dim=2,
        top_k=1,
        num_experts=2,
        sampler_config=sampler_config,
    )
    generator_config.routing_initialization_mode = RoutingInitializationMode.LAYER
    return generator_config


class ParametricHandlerMutationContractTests(unittest.TestCase):
    def test_base_handler_preserves_configuration_and_fallback_contracts(
        self,
    ) -> None:
        handler = ParameterHandlerBase(_parametric_config(input_dim=2, output_dim=3))

        self.assertEqual(handler.input_dim, 2)
        self.assertEqual(handler.output_dim, 3)
        cloned_router = handler._clone_router_config(RouterConfig)
        self.assertEqual(cloned_router.input_dim, 2)
        self.assertEqual(cloned_router.num_experts, 2)
        self.assertIs(
            cloned_router.model_config,
            handler.router_config.model_config,
        )

        with self.assertRaises(NotImplementedError) as shared_error:
            handler._init_shared_sampler()
        self.assertEqual(
            str(shared_error.exception),
            "The method `_init_shared_sampler` must be implemented in the child class.",
        )
        with self.assertRaises(NotImplementedError) as independent_error:
            handler._init_independent_sampler()
        self.assertEqual(
            str(independent_error.exception),
            "The method `_init_independent_sampler` must be implemented in "
            "the child class.",
        )

    def test_vector_handler_private_shared_boundary_has_exact_error(self) -> None:
        handler = VectorParameterHandler(
            _parametric_config(
                weight_config=VectorWeightsMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                )
            )
        )

        with self.assertRaises(ValueError) as error:
            handler._init_shared_sampler()

        self.assertEqual(
            str(error.exception),
            "VectorWeightsMixtureConfig does not support SHARED_ROUTER.",
        )

    def test_handler_overrides_execute_exact_plain_layer_state_math(self) -> None:
        base = _handler_config(
            input_dim=2,
            output_dim=2,
            layer_model_config=_parametric_config(),
        )
        overrides = ParametricLayerHandlerConfig(input_dim=3, output_dim=3)
        handler = ParametricLayerHandler(base, overrides)
        identity = torch.eye(3)
        with torch.no_grad():
            handler.model.weights_router.model[0].model.weight_params.zero_()
            handler.model.weight_mixture_model.parameter_bank.copy_(
                torch.stack((identity, 2.0 * identity))
            )
        inputs = torch.tensor(
            [[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]],
            requires_grad=True,
        )
        state = LayerState(hidden=inputs)

        returned = handler(state)

        self.assertIs(returned, state)
        torch.testing.assert_close(returned.hidden, 1.5 * inputs.detach())
        self.assertIsNone(returned.skip_mask)
        self.assertEqual(returned.loss.item(), 0.0)
        returned.hidden.square().sum().backward()
        torch.testing.assert_close(inputs.grad, 4.5 * inputs.detach())
        self.assertGreater(
            handler.model.weight_mixture_model.parameter_bank.grad.abs().sum().item(),
            0.0,
        )

    def test_real_handler_accumulates_existing_and_parametric_losses_exactly(
        self,
    ) -> None:
        parametric_config = _parametric_config(
            weight_config=MatrixWeightsMixtureConfig(
                **_mixture_kwargs(top_k=2, num_experts=3)
            ),
            top_k=2,
            num_experts=3,
        )
        parametric_config.sampler_config.coefficient_of_variation_loss_weight = 1.0
        handler = ParametricLayerHandler(
            _handler_config(
                input_dim=2,
                output_dim=2,
                layer_model_config=parametric_config,
            )
        )
        router_weights = torch.tensor(
            [[3.0, 0.0, -2.0], [-1.0, 0.5, 2.0]],
        )
        with torch.no_grad():
            handler.model.weights_router.model[0].model.weight_params.copy_(
                router_weights
            )
            handler.model.weight_mixture_model.parameter_bank.copy_(
                torch.stack((torch.eye(2), 2.0 * torch.eye(2), 3.0 * torch.eye(2)))
            )
        inputs = torch.tensor(
            [[1.0, 0.0], [0.5, 2.0], [-1.0, 3.0]],
            requires_grad=True,
        )
        expected_logits = handler.model.weights_router.compute_logit_scores(
            inputs.detach()
        )
        _, _, _, expected_parametric_loss = (
            handler.model.sampler.sample_probabilities_and_indices(
                expected_logits,
                None,
            )
        )
        self.assertGreater(expected_parametric_loss.item(), 0.0)
        state = LayerState(
            hidden=inputs,
            loss=torch.tensor(2.0),
        )

        returned = handler(state)

        self.assertIs(returned, state)
        torch.testing.assert_close(
            returned.loss,
            torch.tensor(2.0) + expected_parametric_loss,
        )
        self.assertEqual(returned.hidden.shape, (3, 2))
        self.assertTrue(torch.isfinite(returned.hidden).all())
        (returned.hidden.square().mean() + returned.loss).backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        router_gradient = handler.model.weights_router.model[0].model.weight_params.grad
        self.assertIsNotNone(router_gradient)
        self.assertTrue(torch.isfinite(router_gradient).all())
        self.assertGreater(router_gradient.abs().sum().item(), 0.0)

    def test_vector_weights_and_independent_bias_match_exact_affine_math(
        self,
    ) -> None:
        model = ParametricLayer(
            _parametric_config(
                weight_config=VectorWeightsMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                ),
                bias_config=MatrixBiasMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                ),
                routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            )
        )
        weight_bank = torch.tensor(
            [
                [[1.0, 0.0], [3.0, 2.0]],
                [[0.0, 2.0], [2.0, 4.0]],
            ]
        )
        bias_bank = torch.tensor([[1.0, -1.0], [3.0, 1.0]])
        with torch.no_grad():
            model.weights_router.parameter_bank.zero_()
            model.bias_router.model[0].model.weight_params.zero_()
            model.weight_mixture_model.parameter_bank.copy_(weight_bank)
            model.bias_mixture_model.parameter_bank.copy_(bias_bank)
        inputs = torch.tensor(
            [[1.0, 2.0], [-1.0, 3.0]],
            requires_grad=True,
        )

        output, skip_mask, loss = model(inputs)

        self.assertIsNotNone(model.bias_router)
        self.assertIsNotNone(model.sampler)
        torch.testing.assert_close(
            output,
            torch.tensor([[6.0, 7.0], [3.0, 8.0]]),
        )
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.bias_router.model[0].model.weight_params.grad.abs().sum().item(),
            0.0,
        )

    def test_generator_weights_and_independent_bias_match_exact_affine_math(
        self,
    ) -> None:
        model = ParametricLayer(
            _parametric_config(
                weight_config=GeneratorWeightsMixtureConfig(
                    **_mixture_kwargs(
                        top_k=1,
                        num_experts=2,
                        weighted=False,
                    ),
                    generator_config=_owned_linear_generator_config(),
                ),
                bias_config=MatrixBiasMixtureConfig(
                    **_mixture_kwargs(top_k=1, num_experts=2)
                ),
                routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
                top_k=1,
                num_experts=2,
            )
        )
        generators = (
            model.weight_mixture_model.input_vector_generator,
            model.weight_mixture_model.output_vector_generator,
        )
        with torch.no_grad():
            for generator in generators:
                generator.sampler.router.model[0].model.weight_params.zero_()
                generator.expert_modules[0][0].model.weight_params.copy_(torch.eye(2))
            model.bias_router.model[0].model.weight_params.zero_()
            model.bias_mixture_model.parameter_bank.copy_(
                torch.tensor([[2.0, -4.0], [8.0, 6.0]])
            )
        inputs = torch.tensor(
            [[1.0, 2.0], [-1.0, 1.0]],
            requires_grad=True,
        )

        output, skip_mask, loss = model(inputs)

        self.assertIsNone(model.weights_router)
        self.assertIsNotNone(model.bias_router)
        self.assertIsNotNone(model.sampler)
        torch.testing.assert_close(
            output,
            torch.tensor([[6.0, 8.0], [-1.0, 0.0]]),
        )
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.bias_mixture_model.parameter_bank.grad[0].abs().sum().item(),
            0.0,
        )
        torch.testing.assert_close(
            model.bias_mixture_model.parameter_bank.grad[1],
            torch.zeros(2),
        )


if __name__ == "__main__":
    unittest.main()
