import unittest

import torch

from emperor.experts import (
    DroppedTokenOptions,
    ExpertWeightingPositionOptions,
    MixtureOfExpertsConfig,
    MixtureOfExpertsLayerConfig,
    MixtureOfExpertsModelConfig,
    RoutingInitializationMode,
)
from emperor.experts._layers.map import MixtureOfExpertsMap
from emperor.experts._layers.mixture import MixtureOfExperts
from emperor.experts._layers.reduce import MixtureOfExpertsReduce
from emperor.experts._model import MixtureOfExpertsModel
from emperor.layers import (
    ActivationOptions,
    LastLayerBiasOptions,
    LayerConfig,
    LayerNormPositionOptions,
    LayerStackConfig,
)
from emperor.linears import LinearLayerConfig
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
            layer_model_config=LinearLayerConfig(bias_flag=True),
        ),
    )


def _mixture_config(
    *,
    input_dim: int = 2,
    output_dim: int = 2,
    top_k: int = 1,
    num_experts: int = 2,
) -> MixtureOfExpertsConfig:
    return MixtureOfExpertsConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        num_experts=num_experts,
        capacity_factor=0.0,
        dropped_token_behavior=DroppedTokenOptions.ZEROS,
        compute_expert_mixture_flag=True,
        weighted_parameters_flag=True,
        weighting_position_option=ExpertWeightingPositionOptions.AFTER_EXPERTS,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        expert_model_config=_linear_stack(input_dim, output_dim),
    )


def _model_config(
    *,
    input_dim: int = 2,
    output_dim: int = 2,
    top_k: int = 1,
) -> MixtureOfExpertsModelConfig:
    mixture_config = _mixture_config(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
    )
    return MixtureOfExpertsModelConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
        routing_initialization_mode=RoutingInitializationMode.DISABLED,
        sampler_config=None,
        stack_config=LayerStackConfig(
            input_dim=input_dim,
            hidden_dim=max(input_dim, output_dim),
            output_dim=output_dim,
            num_layers=1,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=False,
            layer_config=MixtureOfExpertsLayerConfig(
                activation=ActivationOptions.DISABLED,
                layer_norm_position=LayerNormPositionOptions.DISABLED,
                residual_config=None,
                dropout_probability=0.0,
                gate_config=None,
                halting_config=None,
                memory_config=None,
                layer_model_config=mixture_config,
            ),
        ),
    )


def _sampler_config(
    *,
    input_dim: int = 2,
    top_k: int = 1,
    num_experts: int = 2,
) -> SamplerConfig:
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
        router_config=RouterConfig(
            input_dim=input_dim,
            num_experts=num_experts,
            noisy_topk_flag=False,
            model_config=_linear_stack(input_dim, num_experts),
        ),
    )


class ExpertValidationContractTests(unittest.TestCase):
    def test_expert_constructors_reject_malformed_config_before_dereference(
        self,
    ) -> None:
        for module_type in (
            MixtureOfExperts,
            MixtureOfExpertsMap,
            MixtureOfExpertsReduce,
        ):
            with self.subTest(module_type=module_type.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "Configuration Error: `cfg` must be of type "
                    "MixtureOfExpertsConfig, received type object",
                ):
                    module_type(object())  # type: ignore[arg-type]

    def test_expert_constructors_reject_malformed_overrides_before_dereference(
        self,
    ) -> None:
        for module_type in (
            MixtureOfExperts,
            MixtureOfExpertsMap,
            MixtureOfExpertsReduce,
        ):
            with self.subTest(module_type=module_type.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "Configuration Error: `overrides` must be of type "
                    "MixtureOfExpertsConfig or None, received type object",
                ):
                    module_type(
                        _mixture_config(),
                        object(),  # type: ignore[arg-type]
                    )

    def test_model_constructor_rejects_a_malformed_config_before_rng_work(
        self,
    ) -> None:
        rng_state = torch.random.get_rng_state()

        with self.assertRaisesRegex(
            TypeError,
            "Configuration Error: `cfg` must be of type "
            "MixtureOfExpertsModelConfig, received type object",
        ):
            MixtureOfExpertsModel(object())  # type: ignore[arg-type]

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))

    def test_model_constructor_rejects_malformed_overrides_before_rng_work(
        self,
    ) -> None:
        rng_state = torch.random.get_rng_state()

        with self.assertRaisesRegex(
            TypeError,
            "Configuration Error: `overrides` must be of type "
            "MixtureOfExpertsModelConfig or None, received type object",
        ):
            MixtureOfExpertsModel(
                _model_config(),
                object(),  # type: ignore[arg-type]
            )

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
