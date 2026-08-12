from dataclasses import dataclass

import models.vit.expert_linear_adaptive.config as config
from emperor.attention import MixtureOfAttentionHeadsConfig
from emperor.experts import MixtureOfExpertsModelConfig
from emperor.layers import LayerStackConfig, RecurrentLayerConfig
from models.vit.expert_linear_adaptive import _config_defaults as config_defaults
from models.vit.expert_linear_adaptive._vit_expert_config_factory import (
    VitExpertAdaptiveConfigDependencies as _ExpertAdaptiveDependencies,
)
from models.vit.expert_linear_adaptive._vit_expert_config_factory import (
    VitExpertAdaptiveConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsSubmoduleStackOptions,
    HiddenAdaptiveBiasOptions,
    HiddenAdaptiveDiagonalOptions,
    HiddenAdaptiveMaskOptions,
    HiddenAdaptiveWeightOptions,
    SubmoduleStackOptions,
    TransformerAttentionOptions,
    TransformerEncoderOptions,
    TransformerFeedForwardOptions,
)


@dataclass(frozen=True)
class ExpertAdaptiveConfigDependencies:
    hidden_dim: int
    encoder_options: TransformerEncoderOptions | None
    attention_options: TransformerAttentionOptions | None
    feed_forward_options: TransformerFeedForwardOptions | None
    mixture_options: ExpertsMixtureOptions | None
    mixture_submodule_stack_options: ExpertsSubmoduleStackOptions | None
    mixture_layer_controller_options: ExpertsLayerControllerOptions | None
    mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    mixture_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_stack_options: ExpertsSubmoduleStackOptions | None
    sampler_options: ExpertsSamplerOptions | None
    router_options: ExpertsRouterOptions | None
    router_stack_options: ExpertsSubmoduleStackOptions | None
    router_layer_controller_options: ExpertsLayerControllerOptions | None
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    expert_layer_controller_options: ExpertsLayerControllerOptions | None
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions | None
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions | None
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions | None
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions | None
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions | None
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions | None
    expert_attention_use_kv_expert_models_flag: bool


class ExpertAdaptiveConfigFactory:
    def __init__(self, dependencies: ExpertAdaptiveConfigDependencies) -> None:
        self.dependencies = dependencies
        config_module = config
        self.encoder_options = (
            dependencies.encoder_options
            if dependencies.encoder_options is not None
            else config_defaults.vit_encoder_options(config_module)
        )
        self.attention_options = (
            dependencies.attention_options
            if dependencies.attention_options is not None
            else config_defaults.vit_attention_options(config_module)
        )
        self.feed_forward_options = (
            dependencies.feed_forward_options
            if dependencies.feed_forward_options is not None
            else config_defaults.vit_feed_forward_options(config_module)
        )
        self.mixture_options = (
            dependencies.mixture_options
            if dependencies.mixture_options is not None
            else config_defaults.experts_mixture_options(config_module)
        )
        self.mixture_submodule_stack_options = (
            dependencies.mixture_submodule_stack_options
            if dependencies.mixture_submodule_stack_options is not None
            else config_defaults.experts_submodule_stack_options(
                config_module, config_defaults.ExpertStackRole.MAIN
            )
        )
        self.mixture_layer_controller_options = (
            dependencies.mixture_layer_controller_options
            if dependencies.mixture_layer_controller_options is not None
            else config_defaults.experts_layer_controller_options(
                config_module, config_defaults.ExpertControlRole.MAIN
            )
        )
        self.mixture_dynamic_memory_options = (
            dependencies.mixture_dynamic_memory_options
            if dependencies.mixture_dynamic_memory_options is not None
            else config_defaults.experts_dynamic_memory_options(
                config_module, config_defaults.ExpertControlRole.MAIN
            )
        )
        self.mixture_recurrent_controller_options = (
            dependencies.mixture_recurrent_controller_options
            if dependencies.mixture_recurrent_controller_options is not None
            else config_defaults.experts_recurrent_controller_options(
                config_module, config_defaults.ExpertControlRole.MAIN
            )
        )
        self.expert_stack_options = (
            dependencies.expert_stack_options
            if dependencies.expert_stack_options is not None
            else config_defaults.experts_submodule_stack_options(
                config_module, config_defaults.ExpertStackRole.EXPERT
            )
        )
        self.sampler_options = (
            dependencies.sampler_options
            if dependencies.sampler_options is not None
            else config_defaults.experts_sampler_options(config_module)
        )
        self.router_options = (
            dependencies.router_options
            if dependencies.router_options is not None
            else config_defaults.experts_router_options(config_module)
        )
        self.router_stack_options = (
            dependencies.router_stack_options
            if dependencies.router_stack_options is not None
            else config_defaults.experts_submodule_stack_options(
                config_module, config_defaults.ExpertStackRole.ROUTER
            )
        )
        self.router_layer_controller_options = (
            dependencies.router_layer_controller_options
            if dependencies.router_layer_controller_options is not None
            else config_defaults.experts_layer_controller_options(
                config_module, config_defaults.ExpertControlRole.ROUTER
            )
        )
        self.router_dynamic_memory_options = (
            dependencies.router_dynamic_memory_options
            if dependencies.router_dynamic_memory_options is not None
            else config_defaults.experts_dynamic_memory_options(
                config_module, config_defaults.ExpertControlRole.ROUTER
            )
        )
        self.router_recurrent_controller_options = (
            dependencies.router_recurrent_controller_options
            if dependencies.router_recurrent_controller_options is not None
            else config_defaults.experts_recurrent_controller_options(
                config_module, config_defaults.ExpertControlRole.ROUTER
            )
        )
        self.expert_layer_controller_options = (
            dependencies.expert_layer_controller_options
            if dependencies.expert_layer_controller_options is not None
            else config_defaults.experts_layer_controller_options(
                config_module, config_defaults.ExpertControlRole.EXPERT
            )
        )
        self.expert_dynamic_memory_options = (
            dependencies.expert_dynamic_memory_options
            if dependencies.expert_dynamic_memory_options is not None
            else config_defaults.experts_dynamic_memory_options(
                config_module, config_defaults.ExpertControlRole.EXPERT
            )
        )
        self.expert_recurrent_controller_options = (
            dependencies.expert_recurrent_controller_options
            if dependencies.expert_recurrent_controller_options is not None
            else config_defaults.experts_recurrent_controller_options(
                config_module, config_defaults.ExpertControlRole.EXPERT
            )
        )
        self.adaptive_generator_stack_options = (
            dependencies.adaptive_generator_stack_options
            if dependencies.adaptive_generator_stack_options is not None
            else config_defaults.adaptive_generator_stack_options(config_module)
        )
        self.hidden_adaptive_weight_options = (
            dependencies.hidden_adaptive_weight_options
            if dependencies.hidden_adaptive_weight_options is not None
            else config_defaults.hidden_adaptive_weight_options(config_module)
        )
        self.hidden_adaptive_bias_options = (
            dependencies.hidden_adaptive_bias_options
            if dependencies.hidden_adaptive_bias_options is not None
            else config_defaults.hidden_adaptive_bias_options(config_module)
        )
        self.hidden_adaptive_diagonal_options = (
            dependencies.hidden_adaptive_diagonal_options
            if dependencies.hidden_adaptive_diagonal_options is not None
            else config_defaults.hidden_adaptive_diagonal_options(config_module)
        )
        self.hidden_adaptive_mask_options = (
            dependencies.hidden_adaptive_mask_options
            if dependencies.hidden_adaptive_mask_options is not None
            else config_defaults.hidden_adaptive_mask_options(config_module)
        )
        self.router_adaptive_weight_options = (
            dependencies.router_adaptive_weight_options
            if dependencies.router_adaptive_weight_options is not None
            else config_defaults.hidden_adaptive_weight_options(
                config_module, config_defaults.AdaptiveRole.ROUTER
            )
        )
        self.router_adaptive_bias_options = (
            dependencies.router_adaptive_bias_options
            if dependencies.router_adaptive_bias_options is not None
            else config_defaults.hidden_adaptive_bias_options(
                config_module, config_defaults.AdaptiveRole.ROUTER
            )
        )
        self.router_adaptive_diagonal_options = (
            dependencies.router_adaptive_diagonal_options
            if dependencies.router_adaptive_diagonal_options is not None
            else config_defaults.hidden_adaptive_diagonal_options(
                config_module, config_defaults.AdaptiveRole.ROUTER
            )
        )
        self.router_adaptive_mask_options = (
            dependencies.router_adaptive_mask_options
            if dependencies.router_adaptive_mask_options is not None
            else config_defaults.hidden_adaptive_mask_options(
                config_module, config_defaults.AdaptiveRole.ROUTER
            )
        )

    def build_feed_forward_base_stack_config(
        self,
        feed_forward_stack_options: SubmoduleStackOptions,
    ) -> MixtureOfExpertsModelConfig:
        return self.__expert_config_factory().build_feed_forward_base_stack_config(
            feed_forward_stack_options
        )

    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config: LayerStackConfig | RecurrentLayerConfig,
    ) -> MixtureOfAttentionHeadsConfig:
        return self.__expert_config_factory().build_attention_config(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            sequence_length=sequence_length,
            projection_model_config=projection_model_config,
        )

    def __expert_config_factory(self) -> VitExpertAdaptiveConfigFactory:
        dependencies = self.dependencies
        return VitExpertAdaptiveConfigFactory(
            _ExpertAdaptiveDependencies(
                hidden_dim=dependencies.hidden_dim,
                encoder_options=self.encoder_options,
                attention_options=self.attention_options,
                feed_forward_options=self.feed_forward_options,
                mixture_options=self.mixture_options,
                expert_stack_options=self.expert_stack_options,
                sampler_options=self.sampler_options,
                router_options=self.router_options,
                router_stack_options=self.router_stack_options,
                expert_layer_controller_options=(self.expert_layer_controller_options),
                expert_dynamic_memory_options=self.expert_dynamic_memory_options,
                expert_recurrent_controller_options=(
                    self.expert_recurrent_controller_options
                ),
                expert_attention_use_kv_expert_models_flag=(
                    dependencies.expert_attention_use_kv_expert_models_flag
                ),
                mixture_submodule_stack_options=(self.mixture_submodule_stack_options),
                mixture_layer_controller_options=(
                    self.mixture_layer_controller_options
                ),
                mixture_dynamic_memory_options=(self.mixture_dynamic_memory_options),
                mixture_recurrent_controller_options=(
                    self.mixture_recurrent_controller_options
                ),
                router_layer_controller_options=(self.router_layer_controller_options),
                router_dynamic_memory_options=self.router_dynamic_memory_options,
                router_recurrent_controller_options=(
                    self.router_recurrent_controller_options
                ),
                adaptive_generator_stack_options=(
                    self.adaptive_generator_stack_options
                ),
                hidden_adaptive_weight_options=self.hidden_adaptive_weight_options,
                hidden_adaptive_bias_options=self.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=(
                    self.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=self.hidden_adaptive_mask_options,
                router_adaptive_weight_options=self.router_adaptive_weight_options,
                router_adaptive_bias_options=self.router_adaptive_bias_options,
                router_adaptive_diagonal_options=(
                    self.router_adaptive_diagonal_options
                ),
                router_adaptive_mask_options=self.router_adaptive_mask_options,
            )
        )
