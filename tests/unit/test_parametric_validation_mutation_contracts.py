import unittest

import torch

from emperor.experts import RoutingInitializationMode
from emperor.layers import (
    ActivationOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerState,
)
from emperor.linears import LinearLayerConfig
from emperor.neuron._cluster.state import NeuronClusterRouteState
from emperor.parametric import (
    AdaptiveRouterOptions,
    GeneratorBiasMixtureConfig,
    GeneratorWeightsMixtureConfig,
    MatrixBiasMixtureConfig,
    ParametricLayer,
    ParametricLayerHandler,
    ParametricLayerHandlerConfig,
    VectorWeightsMixtureConfig,
)
from emperor.parametric._handlers import (
    ParameterHandlerBase,
    VectorParameterHandler,
)
from emperor.parametric._validation import (
    ParametricHandlerValidator,
    ParametricLayerValidator,
)
from tests.unit.test_parametric_behavioral_contracts import (
    _generator_config,
    _mixture_kwargs,
    _parametric_config,
    _router_config,
    _sampler_config,
)


def _handler_config(
    layer_model_config,
) -> ParametricLayerHandlerConfig:
    return ParametricLayerHandlerConfig(
        input_dim=2,
        output_dim=2,
        activation=ActivationOptions.DISABLED,
        residual_config=None,
        dropout_probability=0.0,
        layer_norm_position=LayerNormPositionOptions.DISABLED,
        gate_config=None,
        halting_config=None,
        memory_config=None,
        layer_model_config=layer_model_config,
    )


class ParametricValidationMutationContractTests(unittest.TestCase):
    def assert_exact_error(
        self,
        exception_type: type[Exception],
        expected_message: str,
        callback,
    ) -> None:
        with self.assertRaises(exception_type) as error:
            callback()
        self.assertEqual(str(error.exception), expected_message)

    def assert_rejected_before_rng_consumption(
        self,
        exception_type: type[Exception],
        expected_message: str,
        callback,
    ) -> None:
        with torch.random.fork_rng():
            torch.manual_seed(1901)
            expected_next_values = torch.randn(8)

            torch.manual_seed(1901)
            self.assert_exact_error(
                exception_type,
                expected_message,
                callback,
            )
            actual_next_values = torch.randn(8)

        torch.testing.assert_close(actual_next_values, expected_next_values)

    def test_zero_input_dimension_reports_exact_public_error(self) -> None:
        self.assert_exact_error(
            ValueError,
            "input_dim must be greater than 0, received 0",
            lambda: ParametricLayer(_parametric_config(input_dim=0)),
        )

    def test_zero_output_dimension_reports_exact_public_error(self) -> None:
        self.assert_exact_error(
            ValueError,
            "output_dim must be greater than 0, received 0",
            lambda: ParametricLayer(_parametric_config(output_dim=0)),
        )

    def test_private_positive_integer_boundary_rejects_zero_exactly(self) -> None:
        self.assert_exact_error(
            ValueError,
            "input_dim must be a positive integer, received 0.",
            lambda: ParametricLayerValidator._validate_positive_integer(
                "input_dim",
                0,
            ),
        )

    def test_invalid_bias_mixture_reports_exact_public_error(self) -> None:
        config = _parametric_config()
        config.bias_mixture_config = object()
        self.assert_exact_error(
            TypeError,
            "bias_mixture_config must be None or a bias mixture config, got object.",
            lambda: ParametricLayer(config),
        )

    def test_forward_input_type_rank_and_dimension_errors_are_exact(self) -> None:
        model = ParametricLayer(_parametric_config())
        self.assert_exact_error(
            TypeError,
            "input_batch must be a Tensor, received list.",
            lambda: model([[1.0, 2.0]]),
        )
        self.assert_exact_error(
            ValueError,
            "Input must be a 2D matrix (batch, input_dim), got 3D tensor "
            "with shape (1, 2, 3).",
            lambda: model(torch.ones(1, 2, 3)),
        )
        self.assert_exact_error(
            ValueError,
            "Input feature dimension must match input_dim, received input_dim=2 "
            "and input shape (2, 3).",
            lambda: model(torch.ones(2, 3)),
        )

    def test_dimension_one_executes_exact_public_affine_math(self) -> None:
        model = ParametricLayer(
            _parametric_config(
                input_dim=1,
                output_dim=1,
                top_k=1,
                num_experts=1,
            )
        )
        with torch.no_grad():
            model.weight_mixture_model.parameter_bank.fill_(2.0)
        inputs = torch.tensor([[3.0], [-1.0]], requires_grad=True)

        output, skip_mask, loss = model(inputs)

        torch.testing.assert_close(output, torch.tensor([[6.0], [-2.0]]))
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        torch.testing.assert_close(inputs.grad, torch.tensor([[24.0], [-8.0]]))
        self.assertGreater(
            model.weight_mixture_model.parameter_bank.grad.abs().item(),
            0.0,
        )

    def test_invalid_router_config_reports_exact_public_error(self) -> None:
        config = _parametric_config()
        config.router_config = object()
        self.assert_exact_error(
            TypeError,
            "router_config must be a RouterConfig for ParametricLayer, got object.",
            lambda: ParametricLayer(config),
        )

    def test_invalid_sampler_config_reports_exact_public_error(self) -> None:
        config = _parametric_config()
        config.sampler_config = object()
        self.assert_exact_error(
            TypeError,
            "sampler_config must be a SamplerConfig for ParametricLayer, got object.",
            lambda: ParametricLayer(config),
        )

    def test_invalid_routing_mode_is_rejected_before_rng_consumption(self) -> None:
        config = _parametric_config()
        config.routing_initialization_mode = "independent"
        self.assert_rejected_before_rng_consumption(
            TypeError,
            "routing_initialization_mode must be an AdaptiveRouterOptions value "
            "for ParametricLayer, got str.",
            lambda: ParametricLayer(config),
        )

    def test_router_and_sampler_relationships_are_rejected_before_rng(self) -> None:
        count_mismatch = _parametric_config()
        count_mismatch.router_config.num_experts = 3
        self.assert_rejected_before_rng_consumption(
            ValueError,
            "router_config.num_experts must match sampler_config.num_experts, "
            "received 3 and 2.",
            lambda: ParametricLayer(count_mismatch),
        )

        noise_mismatch = _parametric_config()
        noise_mismatch.router_config.noisy_topk_flag = True
        self.assert_rejected_before_rng_consumption(
            ValueError,
            "router_config.noisy_topk_flag must match "
            "sampler_config.noisy_topk_flag, received True and False.",
            lambda: ParametricLayer(noise_mismatch),
        )

    def test_generator_routing_ownership_is_rejected_before_rng(self) -> None:
        invalid_generator_type = _parametric_config(
            weight_config=GeneratorWeightsMixtureConfig(
                **_mixture_kwargs(top_k=1, num_experts=2),
                generator_config=object(),
            ),
            top_k=1,
            num_experts=2,
        )
        self.assert_rejected_before_rng_consumption(
            TypeError,
            "generator_config must be a MixtureOfExpertsConfig for generator "
            "mixtures, got object.",
            lambda: ParametricLayer(invalid_generator_type),
        )

        invalid_nested_mode_config = _generator_config(top_k=1, num_experts=2)
        invalid_nested_mode_config.routing_initialization_mode = "layer"
        invalid_nested_mode = _parametric_config(
            weight_config=GeneratorWeightsMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=invalid_nested_mode_config,
            ),
            top_k=1,
            num_experts=2,
        )
        self.assert_rejected_before_rng_consumption(
            TypeError,
            "weight_mixture_config.generator_config.routing_initialization_mode "
            "must be a RoutingInitializationMode value, got str.",
            lambda: ParametricLayer(invalid_nested_mode),
        )

        independently_weighted = _parametric_config(
            weight_config=GeneratorWeightsMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=True,
                ),
                generator_config=_generator_config(top_k=1, num_experts=2),
            ),
            routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            top_k=1,
            num_experts=2,
        )
        self.assert_rejected_before_rng_consumption(
            ValueError,
            "GeneratorWeightsMixtureConfig with weighted_parameters_flag=True "
            "requires SHARED_ROUTER routing.",
            lambda: ParametricLayer(independently_weighted),
        )

        nested_disabled = _parametric_config(
            weight_config=GeneratorWeightsMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=_generator_config(top_k=1, num_experts=2),
            ),
            routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            top_k=1,
            num_experts=2,
        )
        self.assert_rejected_before_rng_consumption(
            ValueError,
            "weight_mixture_config.generator_config.routing_initialization_mode "
            "must be RoutingInitializationMode.LAYER when ParametricLayer uses "
            "INDEPENDENT_ROUTER, received RoutingInitializationMode.DISABLED.",
            lambda: ParametricLayer(nested_disabled),
        )

        owned_sampler = _sampler_config(1, 2)
        owned_sampler.router_config = _router_config(2, 2)
        owned_generator = _generator_config(
            top_k=1,
            num_experts=2,
            sampler_config=owned_sampler,
        )
        owned_generator.routing_initialization_mode = RoutingInitializationMode.LAYER
        nested_owned_with_shared_outer = _parametric_config(
            weight_config=GeneratorWeightsMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=owned_generator,
            ),
            routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
            top_k=1,
            num_experts=2,
        )
        self.assert_rejected_before_rng_consumption(
            ValueError,
            "weight_mixture_config.generator_config.routing_initialization_mode "
            "must use external routing when ParametricLayer uses SHARED_ROUTER, "
            "received RoutingInitializationMode.LAYER.",
            lambda: ParametricLayer(nested_owned_with_shared_outer),
        )

        independent_bias = _parametric_config(
            bias_config=GeneratorBiasMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=_generator_config(top_k=1, num_experts=2),
            ),
            routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            top_k=1,
            num_experts=2,
        )
        self.assert_rejected_before_rng_consumption(
            ValueError,
            "bias_mixture_config.generator_config.routing_initialization_mode "
            "must be RoutingInitializationMode.LAYER when ParametricLayer uses "
            "INDEPENDENT_ROUTER, received RoutingInitializationMode.DISABLED.",
            lambda: ParametricLayer(independent_bias),
        )

        shared_bias_generator = _generator_config(
            top_k=1,
            num_experts=2,
            sampler_config=owned_sampler,
        )
        shared_bias_generator.routing_initialization_mode = (
            RoutingInitializationMode.LAYER
        )
        shared_bias = _parametric_config(
            bias_config=GeneratorBiasMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=shared_bias_generator,
            ),
            routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
            top_k=1,
            num_experts=2,
        )
        self.assert_rejected_before_rng_consumption(
            ValueError,
            "bias_mixture_config.generator_config.routing_initialization_mode "
            "must use external routing when ParametricLayer uses SHARED_ROUTER, "
            "received RoutingInitializationMode.LAYER.",
            lambda: ParametricLayer(shared_bias),
        )

    def test_independent_generator_routing_validates_weight_and_bias_slots(
        self,
    ) -> None:
        owned_sampler = _sampler_config(1, 2)
        owned_sampler.router_config = _router_config(2, 2)
        owned_weight_generator = _generator_config(
            top_k=1,
            num_experts=2,
            sampler_config=owned_sampler,
        )
        owned_weight_generator.routing_initialization_mode = (
            RoutingInitializationMode.LAYER
        )
        invalid_bias_generator = _generator_config(top_k=1, num_experts=2)
        config = _parametric_config(
            weight_config=GeneratorWeightsMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=owned_weight_generator,
            ),
            bias_config=GeneratorBiasMixtureConfig(
                **_mixture_kwargs(
                    top_k=1,
                    num_experts=2,
                    weighted=False,
                ),
                generator_config=invalid_bias_generator,
            ),
            routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            top_k=1,
            num_experts=2,
        )

        self.assert_rejected_before_rng_consumption(
            ValueError,
            "bias_mixture_config.generator_config.routing_initialization_mode "
            "must be RoutingInitializationMode.LAYER when ParametricLayer uses "
            "INDEPENDENT_ROUTER, received RoutingInitializationMode.DISABLED.",
            lambda: ParametricLayer(config),
        )

    def test_shared_singleton_generators_apply_exact_weight_and_bias(
        self,
    ) -> None:
        owned_sampler = _sampler_config(1, 1)
        owned_sampler.router_config = _router_config(2, 1)
        owned_generator = _generator_config(
            top_k=1,
            num_experts=1,
            sampler_config=owned_sampler,
        )
        owned_generator.routing_initialization_mode = RoutingInitializationMode.LAYER
        model = ParametricLayer(
            _parametric_config(
                weight_config=GeneratorWeightsMixtureConfig(
                    **_mixture_kwargs(
                        top_k=1,
                        num_experts=1,
                        weighted=True,
                    ),
                    generator_config=owned_generator,
                ),
                bias_config=GeneratorBiasMixtureConfig(
                    **_mixture_kwargs(
                        top_k=1,
                        num_experts=1,
                        weighted=False,
                    ),
                    generator_config=_generator_config(
                        top_k=1,
                        num_experts=1,
                    ),
                ),
                routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
                top_k=1,
                num_experts=1,
            )
        ).double()
        input_transform = torch.eye(2, dtype=torch.float64)
        output_transform = torch.tensor(
            [[2.0, -1.0], [0.5, 3.0]],
            dtype=torch.float64,
        )
        bias_transform = torch.tensor(
            [[1.0, 2.0], [-2.0, 0.25]],
            dtype=torch.float64,
        )
        input_generator = model.weight_mixture_model.input_vector_generator
        output_generator = model.weight_mixture_model.output_vector_generator
        bias_generator = model.bias_mixture_model.bias_generator
        with torch.no_grad():
            input_generator.expert_modules[0][0].model.weight_params.copy_(
                input_transform
            )
            output_generator.expert_modules[0][0].model.weight_params.copy_(
                output_transform
            )
            bias_generator.expert_modules[0][0].model.weight_params.copy_(
                bias_transform
            )
        inputs = torch.tensor(
            [[1.0, 2.0], [-3.0, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        expected_output_vectors = inputs.detach() @ output_transform
        expected = (
            inputs.detach().square().sum(dim=1, keepdim=True) * expected_output_vectors
            + inputs.detach() @ bias_transform
        )

        output, skip_mask, loss = model(inputs)

        for generator in (input_generator, output_generator, bias_generator):
            self.assertEqual(
                generator.routing_initialization_mode,
                RoutingInitializationMode.DISABLED,
            )
            self.assertIsNone(generator.sampler)
        torch.testing.assert_close(output, expected)
        self.assertIsNone(skip_mask)
        self.assertEqual(loss.item(), 0.0)
        output.square().sum().backward()
        for parameter in (
            inputs,
            input_generator.expert_modules[0][0].model.weight_params,
            output_generator.expert_modules[0][0].model.weight_params,
            bias_generator.expert_modules[0][0].model.weight_params,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_weight_top_k_mismatch_reports_exact_public_error(self) -> None:
        config = _parametric_config()
        config.sampler_config.top_k = 1
        self.assert_exact_error(
            ValueError,
            "sampler_config.top_k must match weight_mixture_config.top_k, "
            "received 1 and 2.",
            lambda: ParametricLayer(config),
        )

    def test_weight_expert_count_mismatch_reports_exact_public_error(self) -> None:
        config = _parametric_config()
        config.sampler_config.num_experts = 3
        self.assert_exact_error(
            ValueError,
            "sampler_config.num_experts must match "
            "weight_mixture_config.num_experts, received 3 and 2.",
            lambda: ParametricLayer(config),
        )

    def test_bias_top_k_mismatch_reports_exact_public_error(self) -> None:
        config = _parametric_config(
            bias_config=MatrixBiasMixtureConfig(
                **_mixture_kwargs(top_k=1, num_experts=2)
            )
        )
        self.assert_exact_error(
            ValueError,
            "sampler_config.top_k must match bias_mixture_config.top_k, "
            "received 2 and 1.",
            lambda: ParametricLayer(config),
        )

    def test_bias_expert_count_mismatch_reports_exact_public_error(self) -> None:
        config = _parametric_config(
            bias_config=MatrixBiasMixtureConfig(
                **_mixture_kwargs(top_k=2, num_experts=3)
            )
        )
        self.assert_exact_error(
            ValueError,
            "sampler_config.num_experts must match "
            "bias_mixture_config.num_experts, received 2 and 3.",
            lambda: ParametricLayer(config),
        )

    def test_vector_shared_router_reports_exact_public_error(self) -> None:
        config = _parametric_config(
            weight_config=VectorWeightsMixtureConfig(
                **_mixture_kwargs(top_k=2, num_experts=2)
            ),
            routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
        )
        self.assert_exact_error(
            ValueError,
            "VectorWeightsMixtureConfig does not support SHARED_ROUTER routing.",
            lambda: ParametricLayer(config),
        )

    def test_invalid_adaptive_augmentation_reports_exact_public_error(self) -> None:
        config = _parametric_config()
        config.adaptive_augmentation_config = object()
        self.assert_exact_error(
            TypeError,
            "adaptive_augmentation_config must be an "
            "AdaptiveParameterAugmentationConfig for ParametricLayer, got object.",
            lambda: ParametricLayer(config),
        )

    def test_duplicate_bias_sources_report_exact_public_error(self) -> None:
        config = _parametric_config(
            bias_config=MatrixBiasMixtureConfig(
                **_mixture_kwargs(top_k=2, num_experts=2)
            )
        )
        config.adaptive_augmentation_config.bias_config = object()
        self.assert_exact_error(
            ValueError,
            "adaptive_augmentation_config.bias_config can only be used when "
            "bias_mixture_config is None.",
            lambda: ParametricLayer(config),
        )

    def test_handler_rejects_real_non_layer_state_exactly(self) -> None:
        handler = _handler_config(_parametric_config()).build()
        route_state = NeuronClusterRouteState(
            hidden=torch.ones(1, 2),
            positions=torch.zeros(1, 3, dtype=torch.long),
            active_mask=torch.ones(1, dtype=torch.bool),
            escaped_mask=torch.zeros(1, dtype=torch.bool),
            final_mask=torch.zeros(1, dtype=torch.bool),
            halting_state=None,
            loss=torch.tensor(0.0),
        )
        self.assert_exact_error(
            TypeError,
            "state must be a LayerState for ParametricLayerHandler, "
            "got NeuronClusterRouteState.",
            lambda: handler(route_state),
        )

    def test_handler_rejects_generic_state_before_attribute_access(self) -> None:
        handler = _handler_config(_parametric_config()).build()
        self.assert_exact_error(
            TypeError,
            "state must be a LayerState for ParametricLayerHandler, got object.",
            lambda: handler(object()),
        )

    def test_handler_rejects_non_tensor_hidden_exactly(self) -> None:
        handler = _handler_config(_parametric_config()).build()
        self.assert_exact_error(
            TypeError,
            "state.hidden must be a Tensor for ParametricLayerHandler, got object.",
            lambda: handler(LayerState(hidden=object())),
        )

    def test_handler_rejects_generic_layer_config_exactly(self) -> None:
        config = LayerConfig(
            input_dim=2,
            output_dim=2,
            activation=ActivationOptions.DISABLED,
            residual_config=None,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=_parametric_config(),
        )
        self.assert_exact_error(
            TypeError,
            "ParametricLayerHandler cfg must be ParametricLayerHandlerConfig, "
            "got LayerConfig.",
            lambda: ParametricLayerHandler(config),
        )

    def test_handler_rejects_non_parametric_nested_config_exactly(self) -> None:
        config = _handler_config(LinearLayerConfig(bias_flag=False))
        self.assert_exact_error(
            TypeError,
            "ParametricLayerHandler.layer_model_config must be "
            "ParametricLayerConfig, got LinearLayerConfig.",
            lambda: ParametricLayerHandler(config),
        )

    def test_parameter_handler_private_validator_errors_are_exact(self) -> None:
        invalid_config_handler = ParameterHandlerBase(_parametric_config())
        invalid_config_handler.cfg = _handler_config(LinearLayerConfig(bias_flag=False))
        self.assert_exact_error(
            TypeError,
            "ParameterHandlerBase cfg must be ParametricLayerConfig, "
            "got ParametricLayerHandlerConfig.",
            lambda: ParametricHandlerValidator._validate_parameter_handler(
                invalid_config_handler
            ),
        )

        missing_routing_handler = ParameterHandlerBase(_parametric_config())
        missing_routing_handler.router_config = None
        self.assert_exact_error(
            ValueError,
            "router_config and sampler_config are required for parametric routing.",
            lambda: ParametricHandlerValidator._validate_parameter_handler(
                missing_routing_handler
            ),
        )

        vector_handler = VectorParameterHandler(
            _parametric_config(
                weight_config=VectorWeightsMixtureConfig(
                    **_mixture_kwargs(top_k=2, num_experts=2)
                )
            )
        )
        vector_handler.routing_initialization_mode = AdaptiveRouterOptions.SHARED_ROUTER
        self.assert_exact_error(
            ValueError,
            "VectorWeightsMixtureConfig does not support SHARED_ROUTER routing.",
            lambda: ParametricHandlerValidator._validate_parameter_handler(
                vector_handler
            ),
        )


if __name__ == "__main__":
    unittest.main()
