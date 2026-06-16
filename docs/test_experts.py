from emperor.base.layer.residual import ResidualConnectionOptions
import torch
import unittest

from dataclasses import replace
from emperor.base.layer import Layer, LayerConfig, LayerStack, LayerStackConfig
from emperor.base.options import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerNormPositionOptions,
)
from emperor.sampler.model import SamplerModel
from emperor.sampler.core.routers import RouterModel
from emperor.sampler.core.config import RouterConfig, SamplerConfig
from emperor.experts.config import MixtureOfExpertsModelConfig
from emperor.experts.core.config import MixtureOfExpertsConfig
from emperor.experts.core.layers import (
    MixtureOfExperts,
    MixtureOfExpertsMap,
    MixtureOfExpertsReduce,
    ExpertInputData,
)
from emperor.experts.core._validator import MixtureOfExpertsValidator
from emperor.experts.core._expert_capacity import ExpertCapacityHandler
from emperor.linears.core.config import LinearLayerConfig
from emperor.experts.model import MixtureOfExpertsModel
from emperor.experts.core.config import MixtureOfExpertsLayerConfig
from emperor.experts.core.state import MixtureOfExpertsLayerState
from emperor.sampler.core.samplers import SamplerFull, SamplerSparse, SamplerTopk
from emperor.experts.core.options import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    RoutingInitializationMode,
)


class MixtureOfExpertsPresetMixin:
    def config_values(self, cfg):
        return {
            field_name: getattr(cfg, field_name)
            for field_name in cfg.__dataclass_fields__
        }

    def router_config(
        self,
        input_dim: int = 8,
        num_experts: int = 6,
        bias_flag: bool = False,
        noisy_topk_flag: bool = False,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        stack_dropout_probability: float = 0.0,
    ) -> RouterConfig:
        hidden_dim = (
            stack_hidden_dim if stack_hidden_dim > 0 else max(input_dim, num_experts)
        )
        output_dim = num_experts * 2 if noisy_topk_flag else num_experts
        return RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=noisy_topk_flag,
            model_config=LayerStackConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=stack_num_layers,
                last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
                apply_output_pipeline_flag=False,
                layer_config=LayerConfig(
                    activation=stack_activation,
                    layer_norm_position=LayerNormPositionOptions.DISABLED,
                    residual_connection_option=stack_residual_connection_option,
                    dropout_probability=stack_dropout_probability,
                    gate_config=None,
                    halting_config=None,
                    memory_config=None,
                    layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
                ),
            ),
        )

    def sampler_config(
        self,
        num_experts: int = 6,
        top_k: int = 3,
        threshold: float = 0.0,
        filter_above_threshold: bool = False,
        num_topk_samples: int = 0,
        normalize_probabilities_flag: bool = False,
        noisy_topk_flag: bool = False,
        coefficient_of_variation_loss_weight: float = 0.0,
        switch_loss_weight: float = 0.0,
        zero_centred_loss_weight: float = 0.0,
        mutual_information_loss_weight: float = 0.0,
        router_config: "RouterConfig | None" = None,
    ) -> SamplerConfig:
        return SamplerConfig(
            top_k=top_k,
            threshold=threshold,
            filter_above_threshold=filter_above_threshold,
            num_topk_samples=num_topk_samples,
            normalize_probabilities_flag=normalize_probabilities_flag,
            noisy_topk_flag=noisy_topk_flag,
            num_experts=num_experts,
            coefficient_of_variation_loss_weight=coefficient_of_variation_loss_weight,
            switch_loss_weight=switch_loss_weight,
            zero_centred_loss_weight=zero_centred_loss_weight,
            mutual_information_loss_weight=mutual_information_loss_weight,
            router_config=router_config,
        )

    def expert_model_config(
        self,
        input_dim: int = 8,
        output_dim: int = 6,
        bias_flag: bool = False,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        stack_dropout_probability: float = 0.0,
    ) -> LayerStackConfig:
        hidden_dim = (
            stack_hidden_dim if stack_hidden_dim > 0 else max(input_dim, output_dim)
        )
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=stack_num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=LayerConfig(
                activation=stack_activation,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_connection_option=stack_residual_connection_option,
                dropout_probability=stack_dropout_probability,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=LinearLayerConfig(bias_flag=bias_flag),
            ),
        )

    def preset(
        self,
        input_dim: int = 8,
        output_dim: int = 6,
        router_model_bias_flag: bool = False,
        router_model_noisy_topk_flag: bool = False,
        sampler_threshold: float = 0.0,
        sampler_filter_above_threshold: bool = False,
        sampler_num_topk_samples: int = 0,
        sampler_normalize_probabilities_flag: bool = False,
        sampler_switch_loss_weight: float = 0.0,
        sampler_zero_centred_loss_weight: float = 0.0,
        sampler_mutual_information_loss_weight: float = 0.0,
        sampler_coefficient_of_variation_loss_weight: float = 0.0,
        experts_top_k: int = 3,
        experts_num_experts: int = 6,
        experts_compute_expert_mixture_flag: bool = False,
        experts_weighting_position_option: ExpertWeightingPositionOptions = ExpertWeightingPositionOptions.BEFORE_EXPERTS,
        experts_routing_initialization_mode: RoutingInitializationMode = RoutingInitializationMode.DISABLED,
        experts_weighted_parameters_flag: bool = False,
        experts_capacity_factor: float = 0.0,
        experts_dropped_token_behavior: DroppedTokenOptions = DroppedTokenOptions.ZEROS,
        experts_model_bias_flag: bool = False,
        stack_num_layers: int = 2,
        stack_hidden_dim: int = 0,
        stack_activation: ActivationOptions = ActivationOptions.RELU,
        stack_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        stack_dropout_probability: float = 0.0,
    ) -> MixtureOfExpertsConfig:
        return MixtureOfExpertsConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            expert_model_config=self.expert_model_config(
                input_dim=input_dim,
                output_dim=output_dim,
                bias_flag=experts_model_bias_flag,
                stack_num_layers=stack_num_layers,
                stack_hidden_dim=stack_hidden_dim,
                stack_activation=stack_activation,
                stack_residual_connection_option=stack_residual_connection_option,
                stack_dropout_probability=stack_dropout_probability,
            ),
            top_k=experts_top_k,
            num_experts=experts_num_experts,
            capacity_factor=experts_capacity_factor,
            dropped_token_behavior=experts_dropped_token_behavior,
            compute_expert_mixture_flag=experts_compute_expert_mixture_flag,
            weighted_parameters_flag=experts_weighted_parameters_flag,
            weighting_position_option=experts_weighting_position_option,
            routing_initialization_mode=experts_routing_initialization_mode,
            sampler_config=self.sampler_config(
                num_experts=experts_num_experts,
                top_k=experts_top_k,
                threshold=sampler_threshold,
                filter_above_threshold=sampler_filter_above_threshold,
                num_topk_samples=sampler_num_topk_samples,
                normalize_probabilities_flag=sampler_normalize_probabilities_flag,
                noisy_topk_flag=router_model_noisy_topk_flag,
                coefficient_of_variation_loss_weight=sampler_coefficient_of_variation_loss_weight,
                switch_loss_weight=sampler_switch_loss_weight,
                zero_centred_loss_weight=sampler_zero_centred_loss_weight,
                mutual_information_loss_weight=sampler_mutual_information_loss_weight,
                router_config=self.router_config(
                    input_dim=input_dim,
                    num_experts=experts_num_experts,
                    bias_flag=router_model_bias_flag,
                    noisy_topk_flag=router_model_noisy_topk_flag,
                    stack_num_layers=stack_num_layers,
                    stack_hidden_dim=stack_hidden_dim,
                    stack_activation=stack_activation,
                    stack_residual_connection_option=stack_residual_connection_option,
                    stack_dropout_probability=stack_dropout_probability,
                ),
            ),
        )

    def stack_preset(
        self,
        input_dim: int = 8,
        output_dim: int = 6,
        experts_stack_num_layers: int = 2,
        experts_stack_activation: ActivationOptions = ActivationOptions.RELU,
        experts_stack_residual_connection_option: ResidualConnectionOptions = (
            ResidualConnectionOptions.DISABLED
        ),
        experts_stack_dropout_probability: float = 0.0,
        **kwargs: object,
    ) -> LayerStackConfig:
        hidden_dim = max(input_dim, output_dim)
        return LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=experts_stack_num_layers,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=MixtureOfExpertsLayerConfig(
                activation=experts_stack_activation,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_connection_option=experts_stack_residual_connection_option,
                dropout_probability=experts_stack_dropout_probability,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=self.preset(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    **kwargs,
                ),
            ),
        )

    def model_preset(self, **kwargs: object) -> MixtureOfExpertsModelConfig:
        stack_config = self.stack_preset(**kwargs)
        leaf_config = stack_config.layer_config.layer_model_config
        return MixtureOfExpertsModelConfig(
            input_dim=stack_config.input_dim,
            output_dim=stack_config.output_dim,
            top_k=leaf_config.top_k,
            routing_initialization_mode=leaf_config.routing_initialization_mode,
            sampler_config=leaf_config.sampler_config,
            stack_config=stack_config,
        )

    def external_routing_inputs(
        self,
        input_batch: torch.Tensor,
        sampler_config: SamplerConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        router = RouterModel(sampler_config.router_config)
        sampler = SamplerModel(replace(sampler_config, router_config=None))
        logits = router.compute_logit_scores(input_batch)
        probabilities, indices, _, _ = sampler.sample_probabilities_and_indices(logits)
        return probabilities, indices


class TestMixtureOfExperts(MixtureOfExpertsPresetMixin, unittest.TestCase):
    def test_init_with_different_configs(self):
        top_k_options = [1, 3, 6]
        num_experts = 6
        routing_initialization_modes = [
            RoutingInitializationMode.DISABLED,
            RoutingInitializationMode.LAYER,
        ]

        for top_k in top_k_options:
            for routing_initialization_mode in routing_initialization_modes:
                message = f"Testing configuration with num_experts={num_experts}, top_k={top_k}, and routing_initialization_mode={routing_initialization_mode}"
                with self.subTest(msg=message):
                    c = self.preset(
                        experts_num_experts=num_experts,
                        experts_top_k=top_k,
                        experts_routing_initialization_mode=routing_initialization_mode,
                    )

                    m = MixtureOfExperts(c)
                    cfg = m.cfg
                    self.assertIsInstance(m, MixtureOfExperts)
                    self.assertEqual(m.input_dim, cfg.input_dim)
                    self.assertEqual(m.output_dim, cfg.output_dim)
                    self.assertEqual(m.expert_model_config, cfg.expert_model_config)
                    self.assertEqual(m.top_k, top_k)
                    self.assertEqual(m.num_experts, num_experts)
                    self.assertEqual(m.capacity_factor, cfg.capacity_factor)
                    self.assertEqual(
                        m.dropped_token_behavior,
                        cfg.dropped_token_behavior or DroppedTokenOptions.ZEROS,
                    )
                    self.assertEqual(
                        m.compute_expert_mixture_flag,
                        cfg.compute_expert_mixture_flag,
                    )
                    self.assertEqual(
                        m.weighted_parameters_flag, cfg.weighted_parameters_flag
                    )
                    self.assertEqual(
                        m.routing_initialization_mode, cfg.routing_initialization_mode
                    )
                    self.assertEqual(
                        m.weighting_position_option,
                        cfg.weighting_position_option,
                    )
                    self.assertEqual(m.sampler_config, cfg.sampler_config)

    def test_validator_rejects_invalid_config_values(self):
        cases = [
            {"input_dim": 0},
            {"input_dim": True},
            {"output_dim": 0},
            {"experts_top_k": 0},
            {"experts_top_k": True},
            {"experts_top_k": 7, "experts_num_experts": 6},
            {"experts_num_experts": 0},
            {"experts_capacity_factor": -0.1},
            {
                "input_dim": 8,
                "output_dim": 6,
                "experts_capacity_factor": 1.0,
            },
            {
                "experts_top_k": 6,
                "experts_num_experts": 6,
                "experts_capacity_factor": 1.0,
            },
        ]

        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    MixtureOfExperts(self.preset(**overrides))

    def test_validator_rejects_invalid_forward_reference_types(self):
        cases = [
            {"expert_model_config": object()},
            {"weighting_position_option": object()},
            {"routing_initialization_mode": object()},
            {"dropped_token_behavior": object()},
        ]
        cfg = self.preset()

        for override in cases:
            field_name = next(iter(override))
            with self.subTest(field_name=field_name):
                values = self.config_values(cfg)
                values.update(override)
                with self.assertRaises(TypeError):
                    MixtureOfExperts(MixtureOfExpertsConfig(**values))

        cfg = self.preset(
            experts_routing_initialization_mode=RoutingInitializationMode.LAYER
        )

        with self.subTest(field_name="sampler_config"):
            values = self.config_values(cfg)
            values.update(sampler_config=object())
            with self.assertRaises(TypeError):
                MixtureOfExperts(MixtureOfExpertsConfig(**values))

        with self.subTest(field_name="sampler_config.router_config"):
            values = self.config_values(cfg)
            values.update(sampler_config=self.sampler_config(router_config=object()))
            with self.assertRaises(TypeError):
                MixtureOfExperts(MixtureOfExpertsConfig(**values))

    def test_validator_allows_missing_owned_routing_configs_when_not_layer_owned(self):
        for routing_initialization_mode in (
            RoutingInitializationMode.DISABLED,
            RoutingInitializationMode.SHARED,
        ):
            with self.subTest(routing_initialization_mode=routing_initialization_mode):
                cfg = self.preset(
                    experts_routing_initialization_mode=routing_initialization_mode
                )
                values = self.config_values(cfg)
                values.update(sampler_config=None)

                model = MixtureOfExperts(MixtureOfExpertsConfig(**values))

                self.assertIsNone(model.sampler_config)

    def test_validator_allows_optional_dropped_token_behavior(self):
        cfg = self.preset(experts_dropped_token_behavior=None)

        model = MixtureOfExperts(cfg)

        self.assertIsNone(model.cfg.dropped_token_behavior)
        self.assertEqual(model.dropped_token_behavior, DroppedTokenOptions.ZEROS)

    def test_forward_rejects_external_routing_when_layer_owns_routing(self):
        cfg = self.preset(
            experts_routing_initialization_mode=RoutingInitializationMode.LAYER,
            experts_top_k=3,
            experts_num_experts=6,
        )
        model = MixtureOfExperts(cfg)
        input_batch = torch.randn(5, cfg.input_dim)
        probabilities = torch.rand(5, cfg.top_k)
        indices = torch.randint(0, cfg.num_experts, (5, cfg.top_k))

        with self.assertRaises(ValueError):
            model.forward(input_batch, probabilities=probabilities, indices=indices)

    def test_forward_validator_rejects_invalid_runtime_inputs(self):
        cfg = self.preset(
            experts_routing_initialization_mode=RoutingInitializationMode.DISABLED,
            experts_top_k=3,
            experts_num_experts=6,
        )
        model = MixtureOfExperts(cfg)
        input_batch = torch.randn(5, cfg.input_dim)
        probabilities = torch.rand(5, cfg.top_k)
        indices = torch.randint(0, cfg.num_experts, (5, cfg.top_k))

        cases = [
            (
                "input_batch",
                [[1.0] * cfg.input_dim],
                probabilities,
                indices,
                TypeError,
            ),
            ("input_batch", torch.randn(5), probabilities, indices, ValueError),
            (
                "input_batch",
                torch.randn(5, cfg.input_dim + 1),
                probabilities,
                indices,
                ValueError,
            ),
            ("probabilities", input_batch, [0.1, 0.2], indices, TypeError),
            (
                "probabilities",
                input_batch,
                torch.rand(4, cfg.top_k),
                indices,
                ValueError,
            ),
            (
                "probabilities",
                input_batch,
                torch.rand(5, cfg.top_k + 1),
                indices,
                ValueError,
            ),
            (
                "indices",
                input_batch,
                probabilities,
                torch.rand(5, cfg.top_k),
                TypeError,
            ),
            (
                "indices",
                input_batch,
                probabilities,
                torch.randint(0, cfg.num_experts, (4, cfg.top_k)),
                ValueError,
            ),
            (
                "indices",
                input_batch,
                probabilities,
                torch.randint(0, cfg.num_experts, (5, cfg.top_k + 1)),
                ValueError,
            ),
            (
                "indices",
                input_batch,
                probabilities,
                torch.full((5, cfg.top_k), cfg.num_experts),
                ValueError,
            ),
        ]

        for field_name, batch, probs, route_indices, error_type in cases:
            with self.subTest(field_name=field_name):
                with self.assertRaises(error_type):
                    model.forward(batch, probs, route_indices)

    def test_forward_validator_requires_external_routing_inputs(self):
        cfg = self.preset(
            experts_routing_initialization_mode=RoutingInitializationMode.DISABLED,
            experts_top_k=3,
            experts_num_experts=6,
        )
        model = MixtureOfExperts(cfg)
        input_batch = torch.randn(5, cfg.input_dim)
        probabilities = torch.rand(5, cfg.top_k)

        with self.assertRaises(ValueError):
            model.forward(input_batch)

        with self.assertRaises(ValueError):
            model.forward(input_batch, probabilities=probabilities)

    def test_forward_validator_accepts_dense_routing_without_indices(self):
        cfg = self.preset(
            experts_routing_initialization_mode=RoutingInitializationMode.DISABLED,
            experts_top_k=6,
            experts_num_experts=6,
            experts_compute_expert_mixture_flag=True,
        )
        model = MixtureOfExperts(cfg)
        input_batch = torch.randn(5, cfg.input_dim)
        probabilities = torch.rand(5, cfg.top_k)

        output, loss = model.forward(input_batch, probabilities=probabilities)

        self.assertEqual(output.shape, (5, cfg.output_dim))
        self.assertEqual(loss.item(), 0.0)

    def test__create_experts(self):
        c = self.preset()

        m = MixtureOfExperts(c)
        expert_models = m._MixtureOfExperts__create_experts()
        self.assertEqual(len(m.expert_modules), m.num_experts)
        for expert in expert_models:
            self.assertIsInstance(expert, LayerStack)
            for layer in expert:
                self.assertIsInstance(layer, Layer)

    def test__maybe_create_sampler(self):
        num_experts = 6
        expert_options = [1, 3, 6]
        routing_initialization_modes = [
            RoutingInitializationMode.DISABLED,
            RoutingInitializationMode.LAYER,
        ]
        sampler_options = [SamplerSparse, SamplerTopk, SamplerFull]

        for routing_initialization_mode in routing_initialization_modes:
            for sampler_option, expert_option in zip(sampler_options, expert_options):
                message = f"Testing configuration with sampler_option={sampler_option.__name__}, num_experts={num_experts}, top_k={expert_option}"
                with self.subTest(msg=message):
                    c = self.preset(
                        experts_routing_initialization_mode=routing_initialization_mode,
                        experts_num_experts=num_experts,
                        experts_top_k=expert_option,
                    )

                    m = MixtureOfExperts(c)
                    sampler = m._MixtureOfExperts__maybe_create_sampler()
                    if routing_initialization_mode == RoutingInitializationMode.LAYER:
                        self.assertIsInstance(sampler, SamplerModel)
                        self.assertIsInstance(sampler.router, RouterModel)
                        self.assertIsInstance(sampler.sampler_model, sampler_option)
                        self.assertEqual(sampler.sampler_model.top_k, expert_option)
                        continue
                    self.assertIsNone(sampler)

    def test__maybe_compute_expert_indices(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        routing_initialization_mode_options = [
            RoutingInitializationMode.DISABLED,
            RoutingInitializationMode.LAYER,
        ]

        for top_k in top_k_options:
            for routing_initialization_mode in routing_initialization_mode_options:
                message = f"Testing configuration with routing_initialization_mode={routing_initialization_mode}, top_k={top_k}"
                with self.subTest(msg=message):
                    c = self.preset(
                        experts_routing_initialization_mode=routing_initialization_mode,
                        experts_num_experts=num_experts,
                        experts_top_k=top_k,
                    )

                    m = MixtureOfExperts(c)
                    inputs = torch.randn(5, c.input_dim)
                    if routing_initialization_mode == RoutingInitializationMode.LAYER:
                        input_indices = None
                        input_probabilities = None
                        probabilities, indices, sampler_loss = (
                            m._maybe_compute_expert_indices(
                                inputs, input_probabilities, input_indices
                            )
                        )
                        if top_k == num_experts:
                            self.assertIsNone(indices)
                        else:
                            self.assertIsInstance(indices, torch.Tensor)
                        self.assertIsInstance(probabilities, torch.Tensor)
                        self.assertIsInstance(sampler_loss, torch.Tensor)
                        self.assertEqual(sampler_loss.item(), 0.0)
                    elif (
                        routing_initialization_mode
                        == RoutingInitializationMode.DISABLED
                    ):
                        probabilities, indices, sampler_loss = (
                            m._maybe_compute_expert_indices(inputs)
                        )
                        self.assertIsNone(probabilities)
                        self.assertIsNone(indices)
                        self.assertEqual(sampler_loss.item(), 0.0)

                    probabilities_input = torch.rand(5, top_k)
                    indices_input = (
                        None
                        if top_k == num_experts
                        else torch.randint(0, m.num_experts, (5, top_k))
                    )
                    probabilities, indices, sampler_loss = (
                        m._maybe_compute_expert_indices(
                            inputs,
                            probabilities_input,
                            indices_input,
                        )
                    )
                    self.assertIs(probabilities, probabilities_input)
                    self.assertIs(indices, indices_input)
                    self.assertEqual(sampler_loss.item(), 0.0)

    def test_get_expert_token_indices(self):
        num_experts = 6
        top_k_options = [1, 3]
        capacity_factor_options = [0.0, 0.5, 1.0, 2.0]

        for top_k in top_k_options:
            for capacity_factor in capacity_factor_options:
                for expert_index in range(num_experts):
                    message = f"Testing with top_k={top_k}, capacity_factor={capacity_factor}, expert_index={expert_index}"
                    with self.subTest(msg=message):
                        dim = 8
                        c = self.preset(
                            input_dim=dim,
                            output_dim=dim,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_capacity_factor=capacity_factor,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 30
                        rows = []
                        for _ in range(batch_size):
                            row = torch.randperm(m.num_experts)[:top_k]
                            rows.append(row)
                        indices = torch.stack(rows)

                        sample_indices, dropped_indices = m._get_expert_token_indices(
                            indices, expert_index
                        )

                        self.assertIsInstance(sample_indices, torch.Tensor)
                        self.assertIsInstance(dropped_indices, torch.Tensor)

                        total = sample_indices.size(0) + dropped_indices.size(0)
                        if capacity_factor > 0 and dropped_indices.size(0) > 0:
                            expected_capacity = max(
                                1, int(batch_size / num_experts * capacity_factor)
                            )
                            self.assertLess(sample_indices.size(0), total)
                            self.assertEqual(sample_indices.size(0), expected_capacity)
                            self.assertEqual(
                                dropped_indices.size(0), total - expected_capacity
                            )

    def test_get_expert_routing_positions(self):
        num_experts = 6
        top_k_options = [1, 3]
        capacity_factor_options = [0.0, 0.5, 1.0, 2.0]

        for top_k in top_k_options:
            for capacity_factor in capacity_factor_options:
                for expert_index in range(num_experts):
                    message = f"Testing with top_k={top_k}, capacity_factor={capacity_factor}, expert_index={expert_index}"
                    with self.subTest(msg=message):
                        dim = 8
                        c = self.preset(
                            input_dim=dim,
                            output_dim=dim,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_capacity_factor=capacity_factor,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 30
                        rows = []
                        for _ in range(batch_size):
                            row = torch.randperm(m.num_experts)[:top_k]
                            rows.append(row)
                        indices = torch.stack(rows)
                        if top_k == 1:
                            indices = indices.squeeze(-1)

                        m._get_expert_token_indices(indices, expert_index)
                        sample_positions, dropped_positions = (
                            m._get_expert_routing_positions(indices, expert_index)
                        )

                        self.assertIsInstance(sample_positions, torch.Tensor)
                        self.assertIsInstance(dropped_positions, torch.Tensor)

                        total = sample_positions.size(0) + dropped_positions.size(0)
                        if capacity_factor > 0 and dropped_positions.size(0) > 0:
                            expected_capacity = max(
                                1, int(batch_size / num_experts * capacity_factor)
                            )
                            self.assertLess(sample_positions.size(0), total)
                            self.assertEqual(
                                sample_positions.size(0), expected_capacity
                            )
                            self.assertEqual(
                                dropped_positions.size(0), total - expected_capacity
                            )

    def test_maybe_get_expert_probabilities(self):
        num_experts = 6
        top_k_options = [1, 3, num_experts]

        for weighting_position_option in ExpertWeightingPositionOptions:
            for top_k in top_k_options:
                for expert_index in range(num_experts):
                    message = f"Testing with weighting_position_option={weighting_position_option.name}, top_k={top_k}, expert_index={expert_index}"
                    with self.subTest(msg=message):
                        c = self.preset(
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_weighting_position_option=weighting_position_option,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 10
                        probabilities = torch.randn(batch_size, top_k)

                        if top_k == num_experts:
                            indices = None
                        else:
                            indices = torch.randperm(batch_size * top_k)[:batch_size]

                        result = (
                            m.expert_weighting_handler.maybe_get_expert_probabilities(
                                indices, probabilities, expert_index
                            )
                        )

                        if (
                            weighting_position_option
                            == ExpertWeightingPositionOptions.AFTER_EXPERTS
                        ):
                            self.assertIsNone(result)
                        elif (
                            weighting_position_option
                            == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                        ):
                            self.assertIsInstance(result, torch.Tensor)
                            if top_k == num_experts:
                                self.assertTrue(
                                    torch.equal(result, probabilities[:, expert_index])
                                )
                            else:
                                self.assertTrue(
                                    torch.equal(
                                        result, probabilities.flatten()[indices]
                                    )
                                )

    def test_select_expert_and_dropped_samples(self):
        num_experts = 6
        top_k = 3
        dropped_indices_options = [
            torch.tensor([2, 5, 8]),
            torch.tensor([], dtype=torch.long),
        ]

        for dropped_token_behavior in DroppedTokenOptions:
            for dropped_indices in dropped_indices_options:
                message = f"Testing with dropped_token_behavior={dropped_token_behavior.name}, dropped_indices_size={dropped_indices.size(0)}"
                with self.subTest(msg=message):
                    input_dim = 8
                    c = self.preset(
                        input_dim=input_dim,
                        output_dim=input_dim,
                        experts_num_experts=num_experts,
                        experts_top_k=top_k,
                        experts_dropped_token_behavior=dropped_token_behavior,
                    )

                    m = MixtureOfExperts(c)

                    batch_size = 10
                    input_batch = torch.randn(batch_size, input_dim)
                    indices = torch.randperm(batch_size)[:top_k]

                    expert_samples, dropped_samples = (
                        m.capacity_handler.select_expert_and_dropped_samples(
                            input_batch, indices, dropped_indices
                        )
                    )

                    self.assertIsInstance(expert_samples, torch.Tensor)
                    self.assertIsInstance(dropped_samples, torch.Tensor)
                    self.assertTrue(torch.equal(expert_samples, input_batch[indices]))

                    if dropped_token_behavior == DroppedTokenOptions.ZEROS:
                        self.assertTrue(
                            torch.equal(
                                dropped_samples,
                                torch.zeros_like(input_batch[dropped_indices]),
                            )
                        )
                    else:
                        self.assertTrue(
                            torch.equal(dropped_samples, input_batch[dropped_indices])
                        )

    def test__build_routed_expert_inputs(self):
        num_experts = 6
        top_k_options = [1, 3]
        capacity_factor_options = [0.0, 1.0, 1.5, 2.0]

        for top_k in top_k_options:
            for capacity_factor in capacity_factor_options:
                for weighting_position_option in ExpertWeightingPositionOptions:
                    message = f"Testing with top_k={top_k}, capacity_factor={capacity_factor}, weighting_position_option={weighting_position_option.name}"
                    with self.subTest(msg=message):
                        input_dim = 8
                        c = self.preset(
                            input_dim=input_dim,
                            output_dim=input_dim,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                            experts_capacity_factor=capacity_factor,
                            experts_weighting_position_option=weighting_position_option,
                        )

                        m = MixtureOfExperts(c)

                        batch_size = 30
                        rows = []
                        for _ in range(batch_size):
                            row = torch.randperm(num_experts)[:top_k]
                            rows.append(row)
                        indices = torch.stack(rows)
                        if top_k == 1:
                            indices = indices.squeeze(-1)

                        probabilities = torch.rand(batch_size, top_k)
                        input_batch = torch.randn(batch_size, input_dim)

                        result = m._MixtureOfExperts__build_routed_expert_inputs(
                            input_batch, probabilities, indices
                        )

                        self.assertIsInstance(result, list)
                        self.assertLessEqual(len(result), num_experts)
                        seen_expert_indices = [item.expert_index for item in result]
                        self.assertEqual(
                            len(seen_expert_indices), len(set(seen_expert_indices))
                        )

                        for item in result:
                            self.assertIsInstance(item, ExpertInputData)
                            self.assertGreaterEqual(item.expert_index, 0)
                            self.assertLess(item.expert_index, num_experts)

                            self.assertIsInstance(item.expert_samples, torch.Tensor)
                            self.assertGreater(item.expert_samples.numel(), 0)
                            self.assertEqual(item.expert_samples.shape[-1], input_dim)

                            self.assertIsInstance(item.dropped_samples, torch.Tensor)
                            if capacity_factor == 0.0:
                                self.assertEqual(item.dropped_samples.numel(), 0)

                            self.assertIsInstance(
                                item.expert_routing_positions, torch.Tensor
                            )
                            self.assertIsInstance(
                                item.dropped_routing_positions, torch.Tensor
                            )

                            if (
                                weighting_position_option
                                == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                            ):
                                self.assertIsInstance(item.probabilities, torch.Tensor)
                            else:
                                self.assertIsNone(item.probabilities)

    def test__build_routed_expert_inputs_skips_empty_experts(self):
        input_dim = 8
        num_experts = 6
        c = self.preset(
            input_dim=input_dim,
            output_dim=input_dim,
            experts_num_experts=num_experts,
            experts_top_k=1,
        )

        m = MixtureOfExperts(c)

        assigned_experts = {0, 2, 4}
        indices = torch.tensor([0, 2, 4, 0, 2, 4])
        probabilities = torch.rand(indices.size(0), 1)
        input_batch = torch.randn(indices.size(0), input_dim)

        result = m._MixtureOfExperts__build_routed_expert_inputs(
            input_batch, probabilities, indices
        )

        result_expert_indices = {item.expert_index for item in result}
        self.assertEqual(len(result), len(assigned_experts))
        self.assertEqual(result_expert_indices, assigned_experts)
        for item in result:
            self.assertGreater(item.expert_samples.numel(), 0)

    def test__build_dense_expert_inputs(self):
        num_experts = 6
        input_dim = 8
        batch_size = 10

        for weighting_position_option in ExpertWeightingPositionOptions:
            message = f"Testing with weighting_position_option={weighting_position_option.name}"
            with self.subTest(msg=message):
                c = self.preset(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    experts_num_experts=num_experts,
                    experts_top_k=num_experts,
                    experts_weighting_position_option=weighting_position_option,
                )

                m = MixtureOfExperts(c)

                input_batch = torch.randn(batch_size, input_dim)
                probabilities = torch.rand(batch_size, num_experts)

                result = m._MixtureOfExperts__build_dense_expert_inputs(
                    input_batch, probabilities
                )

                self.assertIsInstance(result, list)
                self.assertEqual(len(result), num_experts)

                for expert_index, item in enumerate(result):
                    self.assertIsInstance(item, ExpertInputData)
                    self.assertEqual(item.expert_index, expert_index)

                    self.assertTrue(torch.equal(item.expert_samples, input_batch))

                    self.assertIsInstance(item.dropped_samples, torch.Tensor)
                    self.assertEqual(item.dropped_samples.numel(), 0)

                    self.assertIsNone(item.expert_routing_positions)
                    self.assertIsNone(item.dropped_routing_positions)

                    if (
                        weighting_position_option
                        == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                    ):
                        assert isinstance(item.probabilities, torch.Tensor)
                        self.assertTrue(
                            torch.equal(
                                item.probabilities, probabilities[:, expert_index]
                            )
                        )
                    else:
                        self.assertIsNone(item.probabilities)

    def test__split_tokens_per_expert(self):
        num_experts = 6
        input_dim = 8
        batch_size = 10
        top_k_options = [1, 3, num_experts]

        for top_k in top_k_options:
            message = f"Testing with top_k={top_k}"
            with self.subTest(msg=message):
                c = self.preset(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    experts_num_experts=num_experts,
                    experts_top_k=top_k,
                )

                m = MixtureOfExperts(c)

                input_batch = torch.randn(batch_size, input_dim)
                probabilities = torch.rand(batch_size, top_k)

                if top_k == num_experts:
                    indices = None
                else:
                    rows = [
                        torch.randperm(num_experts)[:top_k] for _ in range(batch_size)
                    ]
                    indices = torch.stack(rows)
                    if top_k == 1:
                        indices = indices.squeeze(-1)

                result = m._split_tokens_per_expert(input_batch, probabilities, indices)

                self.assertIsInstance(result, list)

                if top_k == num_experts:
                    self.assertEqual(len(result), num_experts)
                    for item in result:
                        self.assertTrue(torch.equal(item.expert_samples, input_batch))
                        self.assertIsNone(item.expert_routing_positions)
                        self.assertIsNone(item.dropped_routing_positions)
                else:
                    self.assertLessEqual(len(result), num_experts)
                    for item in result:
                        self.assertGreater(item.expert_samples.numel(), 0)
                        self.assertIsInstance(
                            item.expert_routing_positions, torch.Tensor
                        )
                        self.assertIsInstance(
                            item.dropped_routing_positions, torch.Tensor
                        )

    def test__compute_experts_output(self):
        num_experts = 6
        batch_size = 10
        top_k_options = [1, 3, 6]
        weighted_parameters_flag_options = [True, False]

        for weighting_position_option in ExpertWeightingPositionOptions:
            for top_k in top_k_options:
                for weighted_parameters_flag in weighted_parameters_flag_options:
                    message = f"Testing configuration with weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, weighting_position={weighting_position_option}"
                    with self.subTest(msg=message):
                        c = self.preset(
                            experts_weighted_parameters_flag=weighted_parameters_flag,
                            experts_weighting_position_option=weighting_position_option,
                            experts_num_experts=num_experts,
                            experts_top_k=top_k,
                        )

                        m = MixtureOfExperts(c)

                        assert c.input_dim is not None
                        assert c.output_dim is not None

                        num_samples = batch_size * top_k
                        expert_samples = torch.randn(num_samples, c.input_dim)
                        probabilities = torch.randperm(num_samples).float()
                        expert_input_slice = ExpertInputData(
                            expert_index=0,
                            expert_samples=expert_samples,
                            dropped_samples=torch.zeros(0),
                            expert_routing_positions=None,
                            dropped_routing_positions=None,
                            probabilities=probabilities,
                        )

                        output, loss = m._MixtureOfExperts__compute_expert_output(  # type: ignore[operator]
                            expert_input_slice
                        )

                        self.assertIsInstance(output, torch.Tensor)
                        self.assertEqual(output.shape, (num_samples, c.output_dim))
                        self.assertTrue(torch.isfinite(output).all())
                        self.assertEqual(loss.item(), 0.0)

                        applies_before = (
                            weighting_position_option
                            == ExpertWeightingPositionOptions.BEFORE_EXPERTS
                        )
                        if applies_before and weighted_parameters_flag:
                            zero_probs_slice = ExpertInputData(
                                expert_index=0,
                                expert_samples=expert_samples,
                                dropped_samples=torch.zeros(0),
                                expert_routing_positions=None,
                                dropped_routing_positions=None,
                                probabilities=torch.zeros(num_samples),
                            )
                            zero_output, _ = (
                                m._MixtureOfExperts__compute_expert_output(  # type: ignore[operator]
                                    zero_probs_slice
                                )
                            )
                            expert_model: torch.nn.Module = m.expert_modules[0]  # type: ignore[assignment]
                            expected = Layer.run_model_returning_hidden(
                                expert_model, torch.zeros_like(expert_samples)
                            )
                            self.assertTrue(torch.allclose(zero_output, expected))

    def test__compute_expert_mixture(self):
        num_experts = 6
        top_k_options = [1, 3, num_experts]
        flag_options = [True, False]

        for top_k in top_k_options:
            for compute_expert_mixture_flag in flag_options:
                for weighted_parameters_flag in flag_options:
                    for weighting_position_option in ExpertWeightingPositionOptions:
                        message = (
                            f"Testing with weighted_parameters_flag={weighted_parameters_flag}, "
                            f"compute_expert_mixture_flag={compute_expert_mixture_flag}, "
                            f"top_k={top_k}, "
                            f"weighting_position_option={weighting_position_option}"
                        )
                        with self.subTest(msg=message):
                            c = self.preset(
                                experts_weighted_parameters_flag=weighted_parameters_flag,
                                experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                experts_weighting_position_option=weighting_position_option,
                                experts_num_experts=num_experts,
                                experts_top_k=top_k,
                            )

                            m = MixtureOfExperts(c)

                            batch_size = 8
                            experts_output = torch.randn(
                                batch_size * top_k, c.output_dim
                            )
                            sample_indices = torch.cat(
                                [
                                    torch.randperm(num_experts)[:top_k]
                                    for _ in range(batch_size)
                                ]
                            )
                            probabilities = torch.softmax(
                                torch.randn(batch_size * top_k), dim=-1
                            )

                            output = m._MixtureOfExperts__compute_expert_mixture(
                                experts_output, sample_indices, probabilities
                            )

                            expected = experts_output.clone()
                            if top_k != num_experts:
                                _, sort_order = sample_indices.sort(dim=0)
                                expected = expected[sort_order]

                            applies_after = (
                                weighted_parameters_flag
                                and weighting_position_option
                                == ExpertWeightingPositionOptions.AFTER_EXPERTS
                            )
                            if applies_after:
                                expected = expected * probabilities.reshape(-1, 1)

                            if compute_expert_mixture_flag and top_k > 1:
                                expected = expected.view(-1, top_k, c.output_dim).sum(
                                    dim=1
                                )

                            self.assertEqual(output.shape, expected.shape)
                            self.assertTrue(torch.allclose(output, expected))

    def test__compute_expert_mixture_sorting_correctness(self):
        output_dim = 4
        batch_size = 2
        top_k = 3
        indices = torch.tensor([2, 0, 1, 1, 2, 0])
        num_experts_options = [6, 3]

        for num_experts in num_experts_options:
            should_sort = top_k != num_experts
            with self.subTest(
                msg=f"sorting num_experts={num_experts}, should_sort={should_sort}"
            ):
                c = self.preset(
                    experts_weighted_parameters_flag=False,
                    experts_compute_expert_mixture_flag=False,
                    experts_num_experts=num_experts,
                    experts_top_k=top_k,
                    output_dim=output_dim,
                )
                m = MixtureOfExperts(c)

                experts_output = torch.arange(
                    batch_size * top_k * output_dim, dtype=torch.float
                ).view(batch_size * top_k, output_dim)

                output = m._MixtureOfExperts__compute_expert_mixture(
                    experts_output, indices, probabilities=None
                )

                if should_sort:
                    _, sort_order = indices.sort(dim=0)
                    expected_output = experts_output[sort_order]
                else:
                    expected_output = experts_output
                self.assertTrue(torch.equal(output, expected_output))

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        routing_initialization_modes = [
            RoutingInitializationMode.DISABLED,
            RoutingInitializationMode.LAYER,
        ]
        num_layers_options = [1, 2, 3]
        capacity_factor_options = [0.0, 1.0, 1.5]
        dropped_token_behavior_options = [
            DroppedTokenOptions.ZEROS,
            DroppedTokenOptions.IDENTITY,
        ]

        for num_layers in num_layers_options:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    for routing_initialization_mode in routing_initialization_modes:
                        for compute_expert_mixture_flag in flag_options:
                            for weighted_parameters_flag in flag_options:
                                for capacity_factor in capacity_factor_options:
                                    for (
                                        dropped_token_behavior
                                    ) in dropped_token_behavior_options:
                                        message = f"Testing with weighting_position_option={weighting_position_option.name}, routing_initialization_mode={routing_initialization_mode}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}, capacity_factor={capacity_factor}, dropped_token_behavior={dropped_token_behavior}"
                                        with self.subTest(msg=message):
                                            if (
                                                capacity_factor > 0
                                                and top_k == num_experts
                                            ):
                                                continue  # validator rejects capacity + top_k==num_experts
                                            output_dim = 8 if capacity_factor > 0 else 6
                                            c = self.preset(
                                                input_dim=8,
                                                output_dim=output_dim,
                                                experts_top_k=top_k,
                                                experts_weighting_position_option=weighting_position_option,
                                                experts_routing_initialization_mode=routing_initialization_mode,
                                                experts_weighted_parameters_flag=weighted_parameters_flag,
                                                experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                                experts_num_experts=num_experts,
                                                experts_capacity_factor=capacity_factor,
                                                experts_dropped_token_behavior=dropped_token_behavior,
                                                stack_num_layers=num_layers,
                                            )

                                            m = MixtureOfExperts(c)

                                            input = torch.randn(10, c.input_dim)
                                            indices = probabilities = None
                                            if (
                                                routing_initialization_mode
                                                == RoutingInitializationMode.DISABLED
                                            ):
                                                router_cfg = (
                                                    c.sampler_config.router_config
                                                )
                                                sampler_cfg = replace(
                                                    c.sampler_config, router_config=None
                                                )
                                                router = RouterModel(router_cfg)
                                                sampler = SamplerModel(sampler_cfg)

                                                logits = router.compute_logit_scores(
                                                    input
                                                )
                                                probabilities, indices, _, _ = (
                                                    sampler.sample_probabilities_and_indices(
                                                        logits
                                                    )
                                                )

                                            output, total_loss = m.forward(
                                                input, probabilities, indices
                                            )

                                            expected_shape = (
                                                10 * top_k,
                                                c.output_dim,
                                            )
                                            if compute_expert_mixture_flag:
                                                expected_shape = (
                                                    10,
                                                    c.output_dim,
                                                )
                                            self.assertEqual(
                                                output.shape, expected_shape
                                            )
                                            self.assertEqual(total_loss.item(), 0.0)

    def test_forward_backpropagates_to_each_dense_expert(self):
        num_experts = 4
        cfg = self.preset(
            input_dim=4,
            output_dim=4,
            experts_top_k=num_experts,
            experts_num_experts=num_experts,
            experts_compute_expert_mixture_flag=True,
            experts_routing_initialization_mode=RoutingInitializationMode.DISABLED,
            stack_num_layers=1,
            stack_activation=ActivationOptions.DISABLED,
        )
        model = MixtureOfExperts(cfg)
        input_batch = torch.ones(3, cfg.input_dim, requires_grad=True)
        probabilities = torch.full(
            (input_batch.size(0), cfg.top_k),
            1.0 / cfg.top_k,
        )

        output, total_loss = model.forward(input_batch, probabilities=probabilities)
        (output.sum() + total_loss).backward()

        self.assertIsNotNone(input_batch.grad)
        self.assertTrue(torch.any(input_batch.grad.abs() > 0))
        for expert in model.expert_modules:
            with self.subTest(expert=expert.__class__.__name__):
                nonzero_grads = [
                    parameter.grad
                    for parameter in expert.parameters()
                    if parameter.requires_grad
                    and parameter.grad is not None
                    and torch.any(parameter.grad.abs() > 0)
                ]
                self.assertTrue(len(nonzero_grads) > 0)


class TestMixtureOfExpertsStack(MixtureOfExpertsPresetMixin, unittest.TestCase):
    def test_init_with_default_config(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            message = f"Testing configuration with num_layers={num_layers}"
            with self.subTest(msg=message):
                c = self.stack_preset(
                    experts_stack_num_layers=num_layers,
                )
                m = c.build()
                self.assertIsInstance(m, LayerStack)

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        num_layers_options = [1, 2, 3]
        routing_initialization_modes = [
            RoutingInitializationMode.DISABLED,
            RoutingInitializationMode.LAYER,
        ]

        for num_layers in num_layers_options:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    for routing_initialization_mode in routing_initialization_modes:
                        for weighted_parameters_flag in flag_options:
                            message = f"Testing with weighting_position_option={weighting_position_option.name}, routing_initialization_mode={routing_initialization_mode}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                            with self.subTest(msg=message):
                                c = self.stack_preset(
                                    experts_top_k=top_k,
                                    experts_weighting_position_option=weighting_position_option,
                                    experts_routing_initialization_mode=routing_initialization_mode,
                                    experts_weighted_parameters_flag=weighted_parameters_flag,
                                    experts_compute_expert_mixture_flag=True,
                                    experts_num_experts=num_experts,
                                    experts_stack_num_layers=num_layers,
                                )
                                m = c.build()

                                batch_size = 10

                                input = torch.randn(batch_size, c.input_dim)
                                indices = probabilities = None
                                if (
                                    routing_initialization_mode
                                    == RoutingInitializationMode.DISABLED
                                ):
                                    moe_cfg = c.layer_config.layer_model_config
                                    router_cfg = moe_cfg.sampler_config.router_config
                                    sampler_cfg = replace(
                                        moe_cfg.sampler_config, router_config=None
                                    )
                                    router = RouterModel(router_cfg)
                                    sampler = SamplerModel(sampler_cfg)

                                    logits = router.compute_logit_scores(input)
                                    probabilities, indices, _, _ = (
                                        sampler.sample_probabilities_and_indices(logits)
                                    )

                                loss = torch.tensor(0.0)
                                state = MixtureOfExpertsLayerState(
                                    hidden=input,
                                    probabilities=probabilities,
                                    indices=indices,
                                    loss=loss,
                                )
                                state = m(state)
                                output, loss = state.hidden, state.loss

                                expected_shape = (
                                    batch_size,
                                    c.output_dim,
                                )
                                self.assertEqual(output.shape, expected_shape)
                                self.assertEqual(loss.item(), 0.0)

    def test_forward_accumulates_existing_state_loss_and_layer_sampler_loss(self):
        cfg = self.stack_preset(
            input_dim=4,
            output_dim=4,
            experts_top_k=2,
            experts_num_experts=4,
            experts_compute_expert_mixture_flag=True,
            experts_routing_initialization_mode=RoutingInitializationMode.LAYER,
            sampler_zero_centred_loss_weight=0.25,
            experts_stack_num_layers=1,
            stack_num_layers=1,
            stack_activation=ActivationOptions.DISABLED,
        )
        model = cfg.build()
        initial_loss = torch.tensor(2.0)
        state = MixtureOfExpertsLayerState(
            hidden=torch.ones(5, cfg.input_dim),
            loss=initial_loss,
        )

        result_state = model(state)

        self.assertEqual(result_state.hidden.shape, (5, cfg.output_dim))
        self.assertIsNotNone(result_state.loss)
        self.assertGreater(result_state.loss.item(), initial_loss.item())


class TestMixtureOfExpertsModel(MixtureOfExpertsPresetMixin, unittest.TestCase):
    def test_init_with_default_config(self):
        num_layer_options = [1, 2, 3]

        for num_layers in num_layer_options:
            message = f"Testing configuration with num_layers={num_layers}"
            with self.subTest(msg=message):
                c = self.model_preset(
                    stack_num_layers=num_layers,
                    experts_stack_num_layers=num_layers,
                )
                m = MixtureOfExpertsModel(c)

                self.assertIsInstance(m.expert_stack, LayerStack)
                moe_layer = m.expert_stack[0]

                for expert in moe_layer.model.expert_modules:
                    self.assertIsInstance(expert, LayerStack)

    def test_shared_routing_creates_model_owned_sampler(self):
        num_layers = 3
        c = self.model_preset(
            experts_stack_num_layers=num_layers,
            experts_routing_initialization_mode=RoutingInitializationMode.SHARED,
            experts_compute_expert_mixture_flag=True,
            stack_num_layers=1,
        )

        model = MixtureOfExpertsModel(c)

        self.assertIsInstance(model.shared_sampler, SamplerModel)
        self.assertIsInstance(model.shared_sampler.router, RouterModel)
        self.assertIsInstance(model.expert_stack, LayerStack)
        self.assertEqual(len(model.expert_stack), num_layers)
        for moe_layer in model.expert_stack:
            self.assertEqual(
                moe_layer.model.routing_initialization_mode,
                RoutingInitializationMode.SHARED,
            )
            self.assertIsNone(moe_layer.model.sampler)

        hidden = torch.randn(10, model.input_dim)
        result_state = Layer.run_model_returning_state(model, hidden)
        self.assertEqual(result_state.hidden.shape, (10, model.cfg.output_dim))

    def test_validator_rejects_invalid_model_reference_types(self):
        cfg = self.model_preset(
            experts_routing_initialization_mode=RoutingInitializationMode.SHARED,
            experts_compute_expert_mixture_flag=True,
        )
        cases = [
            ("stack_config", object()),
            ("sampler_config", object()),
            ("sampler_config", None),
            (
                "sampler_config.router_config",
                self.sampler_config(router_config=object()),
            ),
            ("sampler_config.router_config", self.sampler_config(router_config=None)),
        ]

        for field_name, invalid_value in cases:
            with self.subTest(field_name=field_name):
                values = self.config_values(cfg)
                if field_name == "stack_config":
                    values.update(stack_config=invalid_value)
                else:
                    values.update(sampler_config=invalid_value)
                with self.assertRaises(TypeError):
                    MixtureOfExpertsModel(MixtureOfExpertsModelConfig(**values))

    def test_validator_allows_missing_model_sampler_config_when_not_shared(self):
        for routing_initialization_mode in (
            RoutingInitializationMode.DISABLED,
            RoutingInitializationMode.LAYER,
        ):
            with self.subTest(routing_initialization_mode=routing_initialization_mode):
                cfg = self.model_preset(
                    experts_routing_initialization_mode=routing_initialization_mode
                )
                values = self.config_values(cfg)
                values.update(sampler_config=None)

                model = MixtureOfExpertsModel(MixtureOfExpertsModelConfig(**values))

                self.assertIsNone(model.sampler_config)
                self.assertIsNone(model.shared_sampler)

    def test_stack_config_override_is_applied(self):
        c = self.model_preset(
            experts_routing_initialization_mode=RoutingInitializationMode.LAYER,
            experts_compute_expert_mixture_flag=True,
            experts_num_experts=6,
            experts_top_k=3,
            stack_num_layers=2,
            experts_stack_num_layers=2,
        )
        overrides = MixtureOfExpertsModelConfig(
            stack_config=replace(c.stack_config, num_layers=3)
        )

        model = MixtureOfExpertsModel(c, overrides)

        self.assertEqual(model.stack_config.num_layers, 3)
        hidden = torch.randn(10, c.input_dim)
        result_state = Layer.run_model_returning_state(model, hidden)
        self.assertEqual(result_state.hidden.shape, (10, model.cfg.output_dim))

    def test_top_k_returns_configured_value(self):
        for top_k in (1, 3, 6):
            with self.subTest(top_k=top_k):
                c = self.model_preset(experts_num_experts=6, experts_top_k=top_k)
                model = MixtureOfExpertsModel(c)
                self.assertEqual(model.top_k, top_k)

    def test__combine_losses_handles_missing_and_additive_losses(self):
        model = MixtureOfExpertsModel(self.model_preset())
        shared_sampler_loss = torch.tensor(2.0)
        expert_stack_loss = torch.tensor(3.0)
        combine_losses = model._MixtureOfExpertsModel__combine_losses

        self.assertIsNone(combine_losses(None, None))
        self.assertIs(combine_losses(None, expert_stack_loss), expert_stack_loss)
        self.assertIs(combine_losses(shared_sampler_loss, None), shared_sampler_loss)
        torch.testing.assert_close(
            combine_losses(shared_sampler_loss, expert_stack_loss),
            torch.tensor(5.0),
        )

    def test_forward_combines_existing_state_loss_with_shared_sampler_loss(self):
        cfg = self.model_preset(
            input_dim=4,
            output_dim=4,
            experts_top_k=2,
            experts_num_experts=4,
            experts_routing_initialization_mode=RoutingInitializationMode.SHARED,
            experts_compute_expert_mixture_flag=True,
            sampler_zero_centred_loss_weight=0.25,
            experts_stack_num_layers=1,
            stack_num_layers=1,
            stack_activation=ActivationOptions.DISABLED,
        )
        model = MixtureOfExpertsModel(cfg)
        initial_loss = torch.tensor(2.0)
        input_state = MixtureOfExpertsLayerState(
            hidden=torch.ones(5, cfg.input_dim),
            loss=initial_loss,
        )

        result_state = model(input_state)

        self.assertEqual(result_state.hidden.shape, (5, cfg.output_dim))
        self.assertIsNotNone(result_state.loss)
        self.assertGreater(result_state.loss.item(), initial_loss.item())

    def test_forward_via_layer_stack_runs_state_branch(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    message = f"Testing with weighting_position_option={weighting_position_option.name}, top_k={top_k}, num_layers={num_layers}"
                    with self.subTest(msg=message):
                        c = self.model_preset(
                            experts_top_k=top_k,
                            experts_weighting_position_option=weighting_position_option,
                            experts_routing_initialization_mode=RoutingInitializationMode.LAYER,
                            experts_compute_expert_mixture_flag=True,
                            experts_num_experts=num_experts,
                            stack_num_layers=num_layers,
                            experts_stack_num_layers=num_layers,
                        )

                        m = MixtureOfExpertsModel(c)

                        self.assertIsInstance(m.expert_stack, LayerStack)
                        self.assertNotIsInstance(m.expert_stack, MixtureOfExperts)

                        batch_size = 10
                        input = torch.randn(batch_size, c.input_dim)
                        result_state = Layer.run_model_returning_state(m, input)

                        self.assertEqual(
                            result_state.hidden.shape,
                            (batch_size, c.output_dim),
                        )
                        self.assertEqual(result_state.loss.item(), 0.0)

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    for routing_initialization_mode in RoutingInitializationMode:
                        for weighted_parameters_flag in flag_options:
                            message = f"Testing with weighting_position_option={weighting_position_option.name}, routing_initialization_mode={routing_initialization_mode}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                            with self.subTest(msg=message):
                                c = self.model_preset(
                                    experts_top_k=top_k,
                                    experts_weighting_position_option=weighting_position_option,
                                    experts_routing_initialization_mode=routing_initialization_mode,
                                    experts_weighted_parameters_flag=weighted_parameters_flag,
                                    experts_compute_expert_mixture_flag=True,
                                    experts_num_experts=num_experts,
                                    experts_stack_num_layers=num_layers,
                                )

                                m = MixtureOfExpertsModel(c)

                                batch_size = 10

                                input = torch.randn(batch_size, c.input_dim)
                                indices = probabilities = None
                                if (
                                    routing_initialization_mode
                                    == RoutingInitializationMode.DISABLED
                                ):
                                    leaf_cfg = (
                                        c.stack_config.layer_config.layer_model_config
                                    )
                                    probabilities, indices = self.external_routing_inputs(
                                        input, leaf_cfg.sampler_config
                                    )

                                input_state = MixtureOfExpertsLayerState(
                                    hidden=input,
                                    probabilities=probabilities,
                                    indices=indices,
                                )
                                result_state = m(input_state)

                                expected_shape = (
                                    batch_size,
                                    c.output_dim,
                                )
                                self.assertEqual(
                                    result_state.hidden.shape, expected_shape
                                )
                                self.assertEqual(result_state.loss.item(), 0.0)

    def test_forward_without_expert_mixture_expands_single_layer_output(self):
        num_experts = 6
        batch_size = 10

        for routing_initialization_mode in RoutingInitializationMode:
            for top_k in (1, 3, 6):
                message = f"Testing with routing_initialization_mode={routing_initialization_mode}, top_k={top_k}"
                with self.subTest(msg=message):
                    c = self.model_preset(
                        experts_top_k=top_k,
                        experts_routing_initialization_mode=routing_initialization_mode,
                        experts_compute_expert_mixture_flag=False,
                        experts_num_experts=num_experts,
                        experts_stack_num_layers=1,
                    )
                    model = MixtureOfExpertsModel(c)
                    input = torch.randn(batch_size, c.input_dim)
                    probabilities = indices = None
                    if routing_initialization_mode == RoutingInitializationMode.DISABLED:
                        leaf_cfg = c.stack_config.layer_config.layer_model_config
                        probabilities, indices = self.external_routing_inputs(
                            input, leaf_cfg.sampler_config
                        )
                    input_state = MixtureOfExpertsLayerState(
                        hidden=input,
                        probabilities=probabilities,
                        indices=indices,
                    )

                    result_state = model(input_state)

                    self.assertEqual(
                        result_state.hidden.shape,
                        (batch_size * top_k, c.output_dim),
                    )
                    self.assertEqual(result_state.loss.item(), 0.0)


class TestMixtureOfExpertsMap(MixtureOfExpertsPresetMixin, unittest.TestCase):
    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3, 6]
        flag_options = [True, False]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    for compute_expert_mixture_flag in flag_options:
                        for weighted_parameters_flag in flag_options:
                            message = f"Testing with weighting_position_option={weighting_position_option.name}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                            with self.subTest(msg=message):
                                c = self.preset(
                                    experts_top_k=top_k,
                                    experts_weighting_position_option=weighting_position_option,
                                    experts_weighted_parameters_flag=weighted_parameters_flag,
                                    experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                    experts_num_experts=num_experts,
                                    stack_num_layers=num_layers,
                                )

                                m = MixtureOfExpertsMap(c)

                                input = torch.randn(10, c.input_dim)
                                indices = probabilities = None
                                router_cfg = c.sampler_config.router_config
                                sampler_cfg = replace(
                                    c.sampler_config, router_config=None
                                )
                                router = RouterModel(router_cfg)
                                sampler = SamplerModel(sampler_cfg)

                                logits = router.compute_logit_scores(input)
                                probabilities, indices, _, _ = (
                                    sampler.sample_probabilities_and_indices(logits)
                                )

                                output, total_loss = m.forward(
                                    input, probabilities, indices
                                )

                                expected_shape = (
                                    10 * top_k,
                                    c.output_dim,
                                )

                                self.assertEqual(output.shape, expected_shape)


class TestMixtureOfExpertsReduce(MixtureOfExpertsPresetMixin, unittest.TestCase):
    def test_forward_validator_rejects_invalid_runtime_inputs(self):
        cfg = self.preset(
            input_dim=6,
            output_dim=8,
            experts_top_k=3,
            experts_num_experts=6,
            experts_weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
            experts_weighted_parameters_flag=True,
            experts_compute_expert_mixture_flag=True,
        )
        model = MixtureOfExpertsReduce(cfg)
        input_batch = torch.randn(15, cfg.input_dim)
        probabilities = torch.rand(5, cfg.top_k)
        indices = torch.randint(0, cfg.num_experts, (5, cfg.top_k))

        cases = [
            ("input_batch", torch.randn(15), probabilities, indices, ValueError),
            (
                "input_batch",
                torch.randn(15, cfg.input_dim + 1),
                probabilities,
                indices,
                ValueError,
            ),
            ("probabilities", input_batch, [0.1, 0.2], indices, TypeError),
            (
                "probabilities",
                input_batch,
                torch.rand(5, cfg.top_k + 1),
                indices,
                ValueError,
            ),
            (
                "probabilities",
                torch.randn(14, cfg.input_dim),
                probabilities,
                indices,
                ValueError,
            ),
            (
                "indices",
                input_batch,
                probabilities,
                torch.rand(5, cfg.top_k),
                TypeError,
            ),
            (
                "indices",
                input_batch,
                probabilities,
                torch.randint(0, cfg.num_experts, (5, cfg.top_k + 1)),
                ValueError,
            ),
            (
                "indices",
                input_batch,
                probabilities,
                torch.full((5, cfg.top_k), cfg.num_experts),
                ValueError,
            ),
            (
                "indices",
                torch.randn(14, cfg.input_dim),
                probabilities,
                indices,
                ValueError,
            ),
        ]

        for field_name, batch, probs, route_indices, error_type in cases:
            with self.subTest(field_name=field_name):
                with self.assertRaises(error_type):
                    model.forward(batch, probs, route_indices)

    def test_forward(self):
        num_experts = 6
        top_k_options = [1, 3]
        flag_options = [True, False]
        # routing_initialization_modes = [RoutingInitializationMode.DISABLED, RoutingInitializationMode.LAYER]
        num_layers_options = [1, 2, 3]

        for num_layers in num_layers_options:
            for weighting_position_option in ExpertWeightingPositionOptions:
                for top_k in top_k_options:
                    for compute_expert_mixture_flag in flag_options:
                        for weighted_parameters_flag in flag_options:
                            message = f"Testing with weighting_position_option={weighting_position_option.name}, compute_expert_mixture_flag={compute_expert_mixture_flag}, weighted_parameters_flag={weighted_parameters_flag}, top_k={top_k}, num_layers={num_layers}"
                            with self.subTest(msg=message):
                                c = self.preset(
                                    input_dim=8,
                                    output_dim=6,
                                    experts_top_k=top_k,
                                    experts_weighting_position_option=weighting_position_option,
                                    experts_weighted_parameters_flag=weighted_parameters_flag,
                                    experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                    experts_num_experts=num_experts,
                                    stack_num_layers=num_layers,
                                )

                                m = MixtureOfExpertsMap(c)

                                rc = self.preset(
                                    input_dim=6,
                                    output_dim=8,
                                    experts_top_k=top_k,
                                    experts_weighting_position_option=weighting_position_option,
                                    experts_weighted_parameters_flag=weighted_parameters_flag,
                                    experts_compute_expert_mixture_flag=compute_expert_mixture_flag,
                                    experts_num_experts=num_experts,
                                    stack_num_layers=num_layers,
                                )
                                r = MixtureOfExpertsReduce(rc)

                                input = torch.randn(10, c.input_dim)
                                indices = probabilities = None

                                router_cfg = c.sampler_config.router_config
                                sampler_cfg = replace(
                                    c.sampler_config, router_config=None
                                )
                                router = RouterModel(router_cfg)
                                sampler = SamplerModel(sampler_cfg)

                                logits = router.compute_logit_scores(input)
                                probabilities, indices, _, _ = (
                                    sampler.sample_probabilities_and_indices(logits)
                                )

                                output, total_loss = m.forward(
                                    input, probabilities, indices
                                )
                                output, total_loss = r.forward(
                                    output, probabilities, indices
                                )

                                expected_shape = (10, rc.output_dim)

                                self.assertEqual(output.shape, expected_shape)


class TestExpertCapacityHandler(unittest.TestCase):
    def test_token_indices_within_capacity_returns_all(self):
        capacity_factors = [0, 1.0, 1.5, 2.0]
        token_indices = torch.tensor([3, 7])
        for capacity_factor in capacity_factors:
            message = f"Testing with capacity_factor={capacity_factor}"
            with self.subTest(msg=message):
                cfg = MixtureOfExpertsConfig(
                    capacity_factor=capacity_factor,
                    num_experts=5,
                    top_k=2,
                    dropped_token_behavior=DroppedTokenOptions.ZEROS,
                )
                handler = ExpertCapacityHandler(cfg)
                expert_tokens, dropped_tokens = (
                    handler.maybe_apply_capacity_limit_token_indices(
                        token_indices, batch_size=10
                    )
                )
                self.assertTrue(torch.equal(expert_tokens, token_indices))
                self.assertEqual(dropped_tokens.numel(), 0)
                self.assertIsNone(handler.shuffle_indices)

    def test_token_indices_exceeds_capacity_drops_tokens(self):
        capacity_factors = [1.0, 1.5]
        for capacity_factor in capacity_factors:
            message = f"Testing with capacity_factor={capacity_factor}"
            with self.subTest(msg=message):
                cfg = MixtureOfExpertsConfig(
                    capacity_factor=capacity_factor,
                    num_experts=5,
                    top_k=2,
                    dropped_token_behavior=DroppedTokenOptions.ZEROS,
                )
                handler = ExpertCapacityHandler(cfg)
                capacity = max(1, int(10 / 5 * capacity_factor))
                tokens_to_drop = 2
                token_indices = torch.arange(capacity + tokens_to_drop)
                expert_tokens, dropped_tokens = (
                    handler.maybe_apply_capacity_limit_token_indices(
                        token_indices, batch_size=10
                    )
                )
                self.assertEqual(expert_tokens.numel(), capacity)
                self.assertEqual(dropped_tokens.numel(), tokens_to_drop)
                all_returned = torch.cat([expert_tokens, dropped_tokens]).sort().values
                self.assertTrue(torch.equal(all_returned, token_indices.sort().values))
                self.assertIsInstance(handler.shuffle_indices, torch.Tensor)

    def test_routing_positions_no_shuffle_returns_unchanged(self):
        cfg = MixtureOfExpertsConfig(
            capacity_factor=1.0,
            num_experts=5,
            top_k=2,
            dropped_token_behavior=DroppedTokenOptions.ZEROS,
        )
        handler = ExpertCapacityHandler(cfg)
        self.assertIsNone(handler.shuffle_indices)
        token_indices = torch.tensor([0, 1, 2, 3])
        result, dropped = handler.maybe_apply_capacity_limit_routing_positions(
            token_indices, batch_size=10
        )
        self.assertTrue(torch.equal(result, token_indices))
        self.assertEqual(dropped.numel(), 0)

    def test_routing_positions_uses_stored_shuffle_indices(self):
        cfg = MixtureOfExpertsConfig(
            capacity_factor=1.0,
            num_experts=5,
            top_k=2,
            dropped_token_behavior=DroppedTokenOptions.ZEROS,
        )
        handler = ExpertCapacityHandler(cfg)
        handler.shuffle_indices = torch.tensor([1, 0, 2, 3])
        token_indices = torch.tensor([10, 20, 30, 40])
        result, dropped = handler.maybe_apply_capacity_limit_routing_positions(
            token_indices, batch_size=10
        )
        self.assertEqual(result.numel(), 2)
        self.assertEqual(dropped.numel(), 2)
        self.assertTrue(torch.equal(result, torch.tensor([20, 10])))
        self.assertTrue(torch.equal(dropped, torch.tensor([30, 40])))

    def test_select_expert_and_dropped_samples(self):
        input_batch = torch.randn(4, 8)
        indices = torch.tensor([0, 1])
        dropped_indices = torch.tensor([2, 3])
        expected_dropped = {
            DroppedTokenOptions.ZEROS: torch.zeros(2, 8),
            DroppedTokenOptions.IDENTITY: input_batch[dropped_indices],
        }
        for behavior in DroppedTokenOptions:
            message = f"Testing with behavior={behavior.name}"
            with self.subTest(msg=message):
                cfg = MixtureOfExpertsConfig(
                    capacity_factor=1.0,
                    num_experts=5,
                    top_k=2,
                    dropped_token_behavior=behavior,
                )
                handler = ExpertCapacityHandler(cfg)
                expert_samples, dropped = handler.select_expert_and_dropped_samples(
                    input_batch, indices, dropped_indices
                )
                self.assertTrue(torch.equal(expert_samples, input_batch[indices]))
                self.assertTrue(torch.equal(dropped, expected_dropped[behavior]))
