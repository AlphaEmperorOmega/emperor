from dataclasses import dataclass
from typing import Any

import torch
from emperor.attention.core.variants.mixture_of_attention_heads.config import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.base.layer.residual import ResidualConnectionOptions
from emperor.base.options import LastLayerBiasOptions
from emperor.experts.config import MixtureOfExpertsModelConfig

from models.vit.expert_linear._expert_control_config_factory import (
    ControlConfigDependencies as ExpertLinearControlConfigDependencies,
)
from models.vit.expert_linear._expert_control_config_factory import (
    ControlConfigFactory as ExpertLinearControlConfigFactory,
)
from models.vit.expert_linear.runtime_options import (
    ExpertsStackOptions,
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
    mixture_options: Any
    expert_stack_options: Any
    sampler_options: Any
    router_options: Any
    router_stack_options: Any
    expert_layer_controller_options: Any
    expert_dynamic_memory_options: Any
    expert_recurrent_controller_options: Any
    expert_attention_use_kv_expert_models_flag: bool


@dataclass(frozen=True)
class VitExpertAdaptiveConfigDependencies(VitExpertConfigDependencies):
    expert_attention_flag: bool
    mixture_submodule_stack_options: Any
    mixture_layer_controller_options: Any
    mixture_dynamic_memory_options: Any
    mixture_recurrent_controller_options: Any
    router_layer_controller_options: Any
    router_dynamic_memory_options: Any
    router_recurrent_controller_options: Any
    adaptive_generator_stack_options: Any
    hidden_adaptive_weight_options: Any
    hidden_adaptive_bias_options: Any
    hidden_adaptive_diagonal_options: Any
    hidden_adaptive_mask_options: Any
    router_adaptive_weight_options: Any
    router_adaptive_bias_options: Any
    router_adaptive_diagonal_options: Any
    router_adaptive_mask_options: Any


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
        projection_model_config,
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
            projection_model_config=projection_model_config,
            experts_config=self._build_attention_experts_config(),
            use_kv_expert_models_flag=(
                dependencies.expert_attention_use_kv_expert_models_flag
            ),
        )

    def _build_attention_experts_config(self):
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
            dropout_probability=feed_forward_stack_options.dropout_probability,
            last_layer_bias_option=feed_forward_stack_options.last_layer_bias_option,
            apply_output_pipeline_flag=(
                feed_forward_stack_options.apply_output_pipeline_flag
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
            residual_connection_option=ResidualConnectionOptions.DISABLED,
            dropout_probability=dependencies.encoder_options.dropout_probability,
            last_layer_bias_option=LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
        )

    def _build_control_config(self, stack_options: ExpertsStackOptions):
        raise NotImplementedError


class VitExpertConfigFactory(_VitExpertConfigFactoryBase):
    def _build_control_config(
        self,
        stack_options: ExpertsStackOptions,
    ) -> ExpertLinearControlConfigFactory:
        dependencies = self.dependencies
        return ExpertLinearControlConfigFactory(
            ExpertLinearControlConfigDependencies(
                stack_options=stack_options,
                submodule_stack_options=None,
                mixture_options=dependencies.mixture_options,
                expert_stack_options=dependencies.expert_stack_options,
                sampler_options=dependencies.sampler_options,
                router_options=dependencies.router_options,
                router_stack_options=dependencies.router_stack_options,
                layer_controller_options=None,
                dynamic_memory_options=None,
                recurrent_controller_options=None,
                expert_layer_controller_options=(
                    dependencies.expert_layer_controller_options
                ),
                expert_dynamic_memory_options=(
                    dependencies.expert_dynamic_memory_options
                ),
                expert_recurrent_controller_options=(
                    dependencies.expert_recurrent_controller_options
                ),
                hidden_dim=dependencies.hidden_dim,
                output_dim=dependencies.hidden_dim,
            )
        )
