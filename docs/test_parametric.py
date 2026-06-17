from emperor.base.layer.residual import ResidualConnectionOptions
import unittest

import torch
import torch.nn as nn

import emperor.parametric as parametric
from emperor.augmentations.adaptive_parameters import (
    AdaptiveParameterAugmentationConfig,
)
from emperor.base.layer import LayerConfig, LayerStack, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)
from emperor.linears.core.config import LinearLayerConfig
from emperor.parametric import (
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
    ParametricLayerHandler,
    ParametricLayerHandlerConfig,
    ParametricLayerState,
    VectorRouterConfig,
    VectorRouterModel,
    VectorWeightsMixture,
    VectorWeightsMixtureConfig,
)
from emperor.sampler.core.config import RouterConfig, SamplerConfig


class FixedParametricModel(nn.Module):
    def forward(self, input_batch, skip_mask=None):
        return input_batch + 1.0, skip_mask, input_batch.new_tensor(0.75)


class ParametricPresetMixin:
    def layer_stack_config(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        hidden_dim: int | None = None,
        num_layers: int = 1,
    ) -> LayerStackConfig:
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    bias_flag=False,
                ),
            ),
        )

    def router_config(
        self,
        input_dim: int = 4,
        num_experts: int = 4,
        config_cls: type[RouterConfig] = RouterConfig,
    ) -> RouterConfig:
        return config_cls(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=self.layer_stack_config(
                input_dim=input_dim,
                output_dim=num_experts,
            ),
        )

    def sampler_config(
        self,
        top_k: int = 2,
        num_experts: int = 4,
        threshold: float = 0.0,
        router_config: RouterConfig | None = None,
    ) -> SamplerConfig:
        return SamplerConfig(
            top_k=top_k,
            threshold=threshold,
            filter_above_threshold=False,
            num_topk_samples=0,
            normalize_probabilities_flag=False,
            noisy_topk_flag=False,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=0.0,
            switch_loss_weight=0.0,
            zero_centred_loss_weight=0.0,
            mutual_information_loss_weight=0.0,
            router_config=router_config,
        )

    def augmentation_config(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
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

    def mixture_kwargs(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        top_k: int = 2,
        num_experts: int = 4,
        weighted_parameters_flag: bool = True,
    ) -> dict:
        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "top_k": top_k,
            "num_experts": num_experts,
            "weighted_parameters_flag": weighted_parameters_flag,
            "clip_parameter_option": ClipParameterOptions.DISABLED,
            "clip_range": 1.0,
        }

    def vector_weights_config(self, top_k: int = 2) -> VectorWeightsMixtureConfig:
        return VectorWeightsMixtureConfig(
            **self.mixture_kwargs(
                input_dim=4,
                output_dim=4,
                top_k=top_k,
                num_experts=4,
            )
        )

    def matrix_weights_config(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        top_k: int = 2,
        num_experts: int = 4,
    ) -> MatrixWeightsMixtureConfig:
        return MatrixWeightsMixtureConfig(
            **self.mixture_kwargs(
                input_dim=input_dim,
                output_dim=output_dim,
                top_k=top_k,
                num_experts=num_experts,
            )
        )

    def matrix_bias_config(
        self,
        output_dim: int = 3,
        top_k: int = 2,
        num_experts: int = 4,
    ) -> MatrixBiasMixtureConfig:
        return MatrixBiasMixtureConfig(
            **self.mixture_kwargs(
                input_dim=4,
                output_dim=output_dim,
                top_k=top_k,
                num_experts=num_experts,
            )
        )

    def moe_config(
        self,
        input_dim: int = 4,
        output_dim: int = 3,
        top_k: int = 2,
        num_experts: int = 4,
        routing_mode: RoutingInitializationMode = RoutingInitializationMode.LAYER,
    ) -> MixtureOfExpertsConfig:
        sampler_config = None
        if routing_mode == RoutingInitializationMode.LAYER:
            sampler_config = self.sampler_config(
                top_k=top_k,
                num_experts=num_experts,
                router_config=self.router_config(
                    input_dim=input_dim,
                    num_experts=num_experts,
                ),
            )
        return MixtureOfExpertsConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            top_k=top_k,
            num_experts=num_experts,
            capacity_factor=0.0,
            dropped_token_behavior=DroppedTokenOptions.ZEROS,
            compute_expert_mixture_flag=False,
            weighted_parameters_flag=False,
            weighting_position_option=ExpertWeightingPositionOptions.BEFORE_EXPERTS,
            routing_initialization_mode=routing_mode,
            sampler_config=sampler_config,
            expert_model_config=self.layer_stack_config(
                input_dim=input_dim,
                output_dim=output_dim,
            ),
        )

    def generator_weights_config(
        self,
        routing_mode: RoutingInitializationMode = RoutingInitializationMode.LAYER,
    ) -> GeneratorWeightsMixtureConfig:
        return GeneratorWeightsMixtureConfig(
            **self.mixture_kwargs(weighted_parameters_flag=False),
            generator_config=self.moe_config(routing_mode=routing_mode),
        )

    def generator_bias_config(
        self,
        routing_mode: RoutingInitializationMode = RoutingInitializationMode.LAYER,
    ) -> GeneratorBiasMixtureConfig:
        return GeneratorBiasMixtureConfig(
            **self.mixture_kwargs(weighted_parameters_flag=False),
            generator_config=self.moe_config(routing_mode=routing_mode),
        )

    def parametric_config(
        self,
        weight_mixture_config=None,
        bias_mixture_config=None,
        routing_mode: AdaptiveRouterOptions = AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        input_dim: int = 4,
        output_dim: int = 3,
        sampler_threshold: float = 0.0,
        sampler_top_k: int | None = None,
        sampler_num_experts: int | None = None,
    ) -> ParametricLayerConfig:
        weight_mixture_config = weight_mixture_config or self.matrix_weights_config(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        top_k = sampler_top_k or weight_mixture_config.top_k
        num_experts = sampler_num_experts or weight_mixture_config.num_experts
        return ParametricLayerConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            weight_mixture_config=weight_mixture_config,
            bias_mixture_config=bias_mixture_config,
            routing_initialization_mode=routing_mode,
            router_config=self.router_config(
                input_dim=input_dim,
                num_experts=num_experts,
            ),
            sampler_config=self.sampler_config(
                top_k=top_k,
                num_experts=num_experts,
                threshold=sampler_threshold,
            ),
            adaptive_augmentation_config=self.augmentation_config(
                input_dim=input_dim,
                output_dim=output_dim,
            ),
        )


class TestParametricPublicApi(ParametricPresetMixin, unittest.TestCase):
    def test_public_exports_and_leaf_build_dispatch(self):
        removed_stack_name = "Parametric" + "LayerStack"
        self.assertFalse(hasattr(parametric, removed_stack_name))
        self.assertNotIn(removed_stack_name, parametric.__all__)
        self.assertEqual(ClipParameterOptions.DISABLED.name, "DISABLED")

        cases = [
            (self.vector_weights_config(), VectorWeightsMixture),
            (self.matrix_weights_config(), MatrixWeightsMixture),
            (self.matrix_bias_config(), MatrixBiasMixture),
            (self.generator_weights_config(), GeneratorWeightsMixture),
            (self.generator_bias_config(), GeneratorBiasMixture),
        ]
        for cfg, expected_type in cases:
            with self.subTest(config=type(cfg).__name__):
                self.assertIsInstance(cfg.build(), expected_type)

    def test_vector_router_config_builds_vector_router(self):
        cfg = self.router_config(config_cls=VectorRouterConfig)
        model = cfg.build()
        logits = model.compute_logit_scores(torch.randn(3, cfg.input_dim))

        self.assertIsInstance(model, VectorRouterModel)
        self.assertEqual(logits.shape, (3, cfg.input_dim, cfg.num_experts))


class TestParametricMixtures(ParametricPresetMixin, unittest.TestCase):
    def test_weight_mixture_forward_shapes(self):
        batch_size = 3
        vector = self.vector_weights_config().build()
        vector_probs = torch.softmax(torch.randn(4, batch_size, 2), dim=-1)
        vector_indices = torch.randint(0, 4, (4, batch_size, 2))
        self.assertEqual(
            vector.compute_mixture(vector_probs, vector_indices).shape,
            (batch_size, 4, 4),
        )

        matrix = self.matrix_weights_config().build()
        matrix_probs = torch.softmax(torch.randn(batch_size, 2), dim=-1)
        matrix_indices = torch.randint(0, 4, (batch_size, 2))
        self.assertEqual(
            matrix.compute_mixture(matrix_probs, matrix_indices).shape,
            (batch_size, 4, 3),
        )

        generator = self.generator_weights_config().build()
        weights, loss = generator.compute_mixture(None, None, torch.randn(batch_size, 4))
        self.assertEqual(weights.shape, (batch_size, 4, 3))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_layer_bias_paths(self):
        x = torch.randn(3, 4)
        bias_cases = [
            (None, 3),
            (self.matrix_bias_config(), 3),
            (self.generator_bias_config(), 3),
        ]
        for bias_config, output_dim in bias_cases:
            with self.subTest(bias=type(bias_config).__name__):
                cfg = self.parametric_config(
                    bias_mixture_config=bias_config,
                    output_dim=output_dim,
                )
                output, skip_mask, loss = ParametricLayer(cfg)(x)

                self.assertEqual(output.shape, (3, output_dim))
                self.assertIsNone(skip_mask)
                self.assertEqual(loss.shape, torch.Size([]))


class TestParametricLayerRouting(ParametricPresetMixin, unittest.TestCase):
    def test_shared_and_independent_routing_models(self):
        shared = ParametricLayer(
            self.parametric_config(routing_mode=AdaptiveRouterOptions.SHARED_ROUTER)
        )
        self.assertIsNotNone(shared.weights_router)
        self.assertIsNone(shared.bias_router)

        independent = ParametricLayer(
            self.parametric_config(
                bias_mixture_config=self.matrix_bias_config(),
                routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            )
        )
        self.assertIsNotNone(independent.weights_router)
        self.assertIsNotNone(independent.bias_router)

        generator = ParametricLayer(
            self.parametric_config(
                weight_mixture_config=self.generator_weights_config(),
                routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
            )
        )
        self.assertIsNone(generator.weights_router)
        self.assertIsNone(generator.bias_router)

        vector_cfg = self.parametric_config(
            weight_mixture_config=self.vector_weights_config(),
            routing_mode=AdaptiveRouterOptions.SHARED_ROUTER,
            input_dim=4,
            output_dim=4,
        )
        with self.assertRaises(ValueError):
            ParametricLayer(vector_cfg)

    def test_generic_layer_stack_with_parametric_handler_config(self):
        stack_config = LayerStackConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=3,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
            layer_config=ParametricLayerHandlerConfig(
                input_dim=4,
                output_dim=3,
                activation=ActivationOptions.DISABLED,
                residual_connection_option=ResidualConnectionOptions.DISABLED,
                dropout_probability=0.0,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=self.parametric_config(),
            ),
        )
        model = stack_config.build()
        state = ParametricLayerState(
            hidden=torch.randn(2, 4),
            skip_mask=torch.ones(2, 1),
        )

        output_state = model(state)

        self.assertIsInstance(model, LayerStack)
        self.assertIsInstance(model[0], ParametricLayerHandler)
        self.assertEqual(output_state.hidden.shape, (2, 3))
        self.assertIsNotNone(output_state.loss)
        self.assertIsNotNone(output_state.skip_mask)

    def test_skip_mask_propagates_through_parametric_state(self):
        cfg = self.parametric_config(sampler_threshold=0.99)
        model = ParametricLayerHandlerConfig(
            input_dim=4,
            output_dim=3,
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=cfg,
        ).build()
        state = ParametricLayerState(
            hidden=torch.randn(3, 4),
            skip_mask=torch.ones(3, 1),
        )

        output_state = model(state)

        self.assertEqual(output_state.skip_mask.shape, (3, 1))
        self.assertTrue(torch.equal(output_state.skip_mask, torch.zeros(3, 1)))

    def test_handler_accumulates_existing_loss(self):
        handler = ParametricLayerHandlerConfig(
            input_dim=4,
            output_dim=4,
            activation=ActivationOptions.DISABLED,
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=0.0,
            layer_norm_position=LayerNormPositionOptions.DISABLED,
            gate_config=None,
            halting_config=None,
            memory_config=None,
            layer_model_config=self.parametric_config(output_dim=4),
        ).build()
        handler.model = FixedParametricModel()
        state = ParametricLayerState(
            hidden=torch.zeros(2, 4),
            loss=torch.tensor(2.0),
            skip_mask=torch.ones(2, 1),
        )

        output_state = handler(state)

        torch.testing.assert_close(output_state.loss, torch.tensor(2.75))
        self.assertEqual(output_state.hidden.shape, (2, 4))


class TestParametricValidation(ParametricPresetMixin, unittest.TestCase):
    def test_validation_failures(self):
        cases = []

        missing_weight = self.parametric_config()
        missing_weight.weight_mixture_config = None
        cases.append((missing_weight, ValueError))

        wrong_router = self.parametric_config()
        wrong_router.router_config = object()
        cases.append((wrong_router, TypeError))

        bad_dim = self.parametric_config()
        bad_dim.input_dim = 0
        cases.append((bad_dim, ValueError))

        mismatched_counts = self.parametric_config(sampler_top_k=1)
        cases.append((mismatched_counts, ValueError))

        for cfg, error_type in cases:
            with self.subTest(error=error_type.__name__):
                with self.assertRaises(error_type):
                    ParametricLayer(cfg)

    def test_invalid_forward_rank_raises(self):
        model = ParametricLayer(self.parametric_config())

        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 4))

    def test_bad_clip_range_raises(self):
        cfg = self.matrix_weights_config()
        cfg.clip_range = -1.0

        with self.assertRaises(ValueError):
            cfg.build()

    def test_vector_requires_matching_input_and_output_dims(self):
        cfg = VectorWeightsMixtureConfig(
            **self.mixture_kwargs(input_dim=4, output_dim=3)
        )

        with self.assertRaises(ValueError):
            cfg.build()

    def test_zero_losses_use_input_device_scalar(self):
        cfg = self.parametric_config(
            weight_mixture_config=self.generator_weights_config(),
            routing_mode=AdaptiveRouterOptions.INDEPENDENT_ROUTER,
        )
        model = ParametricLayer(cfg)
        x = torch.randn(2, 4)

        _, _, loss = model(x)

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.device, x.device)
