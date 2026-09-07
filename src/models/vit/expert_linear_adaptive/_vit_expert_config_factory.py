from dataclasses import dataclass, field

import torch

from emperor.attention import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.augmentations.adaptive_parameters import GroupingConfig
from emperor.experts import MixtureOfExpertsConfig, MixtureOfExpertsModelConfig
from emperor.layers import LastLayerBiasOptions, LayerStackConfig, RecurrentLayerConfig
from models.vit.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigDependencies as ExpertAdaptiveControlConfigDependencies,
)
from models.vit.expert_linear_adaptive._expert_control_config_factory import (
    ControlConfigFactory as ExpertAdaptiveControlConfigFactory,
)
from models.vit.expert_linear_adaptive.runtime_options import (
    AdaptiveGeneratorStackOptions,
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
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
class VitExpertConfigDependencies:
    hidden_dim: int
    encoder_options: TransformerEncoderOptions
    attention_options: TransformerAttentionOptions
    feed_forward_options: TransformerFeedForwardOptions
    mixture_options: ExpertsMixtureOptions
    expert_stack_options: ExpertsSubmoduleStackOptions
    sampler_options: ExpertsSamplerOptions
    router_options: ExpertsRouterOptions
    router_stack_options: ExpertsSubmoduleStackOptions
    expert_layer_controller_options: ExpertsLayerControllerOptions
    expert_dynamic_memory_options: ExpertsDynamicMemoryOptions
    expert_recurrent_controller_options: ExpertsRecurrentControllerOptions
    expert_attention_use_kv_expert_models_flag: bool


@dataclass(frozen=True)
class VitExpertAdaptiveConfigDependencies(VitExpertConfigDependencies):
    grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    router_grouping_config: GroupingConfig | None = field(default=None, kw_only=True)
    mixture_submodule_stack_options: ExpertsSubmoduleStackOptions
    mixture_layer_controller_options: ExpertsLayerControllerOptions
    mixture_dynamic_memory_options: ExpertsDynamicMemoryOptions
    mixture_recurrent_controller_options: ExpertsRecurrentControllerOptions
    router_layer_controller_options: ExpertsLayerControllerOptions
    router_dynamic_memory_options: ExpertsDynamicMemoryOptions
    router_recurrent_controller_options: ExpertsRecurrentControllerOptions
    adaptive_generator_stack_options: AdaptiveGeneratorStackOptions
    hidden_adaptive_weight_options: HiddenAdaptiveWeightOptions
    hidden_adaptive_bias_options: HiddenAdaptiveBiasOptions
    hidden_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    hidden_adaptive_mask_options: HiddenAdaptiveMaskOptions
    router_adaptive_weight_options: HiddenAdaptiveWeightOptions
    router_adaptive_bias_options: HiddenAdaptiveBiasOptions
    router_adaptive_diagonal_options: HiddenAdaptiveDiagonalOptions
    router_adaptive_mask_options: HiddenAdaptiveMaskOptions


class _VitExpertConfigFactoryBase:
    def __init__(self, dependencies: VitExpertConfigDependencies) -> None:
        self.dependencies = dependencies

    def build_feed_forward_base_stack_config(
        self,
        feed_forward_stack_options: SubmoduleStackOptions,
    ) -> MixtureOfExpertsModelConfig:
        return self._build_expert_model_config(
            feed_forward_stack_options,
            use_feed_forward_stack_options=True,
        )

    def build_attention_config(
        self,
        *,
        batch_size: int,
        hidden_dim: int,
        sequence_length: int,
        projection_model_config: LayerStackConfig | RecurrentLayerConfig,
    ) -> MixtureOfAttentionHeadsConfig:
        dependencies = self.dependencies
        encoder_options = dependencies.encoder_options
        attention_options = dependencies.attention_options
        return MixtureOfAttentionHeadsConfig(
            batch_size=batch_size,
            num_heads=attention_options.num_heads,
            embedding_dim=hidden_dim,
            query_key_projection_dim=hidden_dim,
            value_projection_dim=hidden_dim,
            target_sequence_length=sequence_length,
            source_sequence_length=sequence_length,
            target_dtype=torch.float32,
            dropout_probability=encoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            batch_first_flag=True,
            projection_model_config=projection_model_config,
            experts_config=self._build_attention_experts_config(),
            use_kv_expert_models_flag=(
                dependencies.expert_attention_use_kv_expert_models_flag
            ),
        )

    def _build_attention_experts_config(self) -> MixtureOfExpertsConfig:
        model_config = self._build_expert_model_config(
            None,
            use_feed_forward_stack_options=False,
        )
        return model_config.stack_config.layer_config.layer_model_config

    def _build_expert_model_config(
        self,
        feed_forward_stack_options: SubmoduleStackOptions | None,
        *,
        use_feed_forward_stack_options: bool,
    ) -> MixtureOfExpertsModelConfig:
        if use_feed_forward_stack_options:
            if feed_forward_stack_options is None:
                raise ValueError("feed_forward_stack_options is required.")
            stack_options = self._feed_forward_experts_stack_options(
                feed_forward_stack_options
            )
        else:
            stack_options = self._attention_experts_stack_options()
        model_config = self._build_control_config(stack_options).build()
        if isinstance(model_config, MixtureOfExpertsModelConfig):
            return model_config
        return model_config.block_config

    def _feed_forward_experts_stack_options(
        self,
        feed_forward_stack_options: SubmoduleStackOptions,
    ) -> ExpertsStackOptions:
        return ExpertsStackOptions(
            hidden_dim=feed_forward_stack_options.hidden_dim,
            bias_flag=feed_forward_stack_options.bias_flag,
            layer_norm_position=feed_forward_stack_options.layer_norm_position,
            num_layers=feed_forward_stack_options.num_layers,
            activation=feed_forward_stack_options.activation,
            residual_connection_option=(
                feed_forward_stack_options.residual_connection_option
            ),
            residual_model_flag=feed_forward_stack_options.residual_model_flag,
            residual_stack_options=feed_forward_stack_options.residual_stack_options,
            dropout_probability=feed_forward_stack_options.dropout_probability,
            last_layer_bias_option=feed_forward_stack_options.last_layer_bias_option,
            apply_output_postprocessing_flag=(
                feed_forward_stack_options.apply_output_postprocessing_flag
            ),
        )

    def _attention_experts_stack_options(self) -> ExpertsStackOptions:
        dependencies = self.dependencies
        return ExpertsStackOptions(
            hidden_dim=dependencies.hidden_dim,
            bias_flag=dependencies.feed_forward_options.bias_flag,
            layer_norm_position=dependencies.encoder_options.layer_norm_position,
            num_layers=dependencies.feed_forward_options.num_layers,
            activation=dependencies.encoder_options.activation,
            residual_connection_option=None,
            residual_model_flag=False,
            residual_stack_options=(
                dependencies.expert_stack_options.residual_stack_options
            ),
            dropout_probability=dependencies.encoder_options.dropout_probability,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_postprocessing_flag=True,
        )

    def _build_control_config(
        self,
        stack_options: ExpertsStackOptions,
    ) -> ExpertAdaptiveControlConfigFactory:
        raise NotImplementedError


class VitExpertAdaptiveConfigFactory(_VitExpertConfigFactoryBase):
    dependencies: VitExpertAdaptiveConfigDependencies

    def __init__(
        self,
        dependencies: VitExpertAdaptiveConfigDependencies,
    ) -> None:
        super().__init__(dependencies)

    def _build_control_config(
        self,
        stack_options: ExpertsStackOptions,
    ) -> ExpertAdaptiveControlConfigFactory:
        dependencies = self.dependencies
        return ExpertAdaptiveControlConfigFactory(
            ExpertAdaptiveControlConfigDependencies(
                stack_options=stack_options,
                submodule_stack_options=dependencies.mixture_submodule_stack_options,
                mixture_options=dependencies.mixture_options,
                expert_stack_options=dependencies.expert_stack_options,
                sampler_options=dependencies.sampler_options,
                router_options=dependencies.router_options,
                router_stack_options=dependencies.router_stack_options,
                router_layer_controller_options=(
                    dependencies.router_layer_controller_options
                ),
                router_dynamic_memory_options=(
                    dependencies.router_dynamic_memory_options
                ),
                router_recurrent_controller_options=(
                    dependencies.router_recurrent_controller_options
                ),
                layer_controller_options=(
                    dependencies.mixture_layer_controller_options
                ),
                dynamic_memory_options=dependencies.mixture_dynamic_memory_options,
                recurrent_controller_options=(
                    dependencies.mixture_recurrent_controller_options
                ),
                expert_layer_controller_options=(
                    dependencies.expert_layer_controller_options
                ),
                expert_dynamic_memory_options=(
                    dependencies.expert_dynamic_memory_options
                ),
                expert_recurrent_controller_options=(
                    dependencies.expert_recurrent_controller_options
                ),
                adaptive_generator_stack_options=(
                    dependencies.adaptive_generator_stack_options
                ),
                grouping_config=dependencies.grouping_config,
                hidden_adaptive_weight_options=(
                    dependencies.hidden_adaptive_weight_options
                ),
                hidden_adaptive_bias_options=dependencies.hidden_adaptive_bias_options,
                hidden_adaptive_diagonal_options=(
                    dependencies.hidden_adaptive_diagonal_options
                ),
                hidden_adaptive_mask_options=dependencies.hidden_adaptive_mask_options,
                router_grouping_config=dependencies.router_grouping_config,
                router_adaptive_weight_options=(
                    dependencies.router_adaptive_weight_options
                ),
                router_adaptive_bias_options=dependencies.router_adaptive_bias_options,
                router_adaptive_diagonal_options=(
                    dependencies.router_adaptive_diagonal_options
                ),
                router_adaptive_mask_options=dependencies.router_adaptive_mask_options,
                hidden_dim=dependencies.hidden_dim,
                output_dim=dependencies.hidden_dim,
            )
        )
