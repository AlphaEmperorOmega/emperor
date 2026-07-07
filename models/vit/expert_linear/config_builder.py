import torch
from emperor.attention.core.variants.mixture_of_attention_heads.config import (
    MixtureOfAttentionHeadsConfig,
)
from emperor.experts.config import MixtureOfExpertsModelConfig

import models.experts.linear.config as expert_defaults
import models.vit.expert_linear.config as config
from models.experts._builder_adapter import linear_builder_kwargs_from_flat
from models.experts._builder_options import (
    ExpertsDynamicMemoryOptions,
    ExpertsLayerControllerOptions,
    ExpertsMixtureOptions,
    ExpertsRecurrentControllerOptions,
    ExpertsRouterOptions,
    ExpertsSamplerOptions,
    ExpertsStackOptions,
    ExpertsSubmoduleStackOptions,
)
from models.experts.linear._control_config_factory import (
    ControlConfigDependencies,
    ControlConfigFactory,
)
from models.vit.expert_linear.experiment_config import ExperimentConfig
from models.vit.linear.config_builder import VitLinearConfigBuilder


class VitExpertLinearConfigBuilder(VitLinearConfigBuilder):
    def __init__(
        self,
        *args,
        mixture_options: ExpertsMixtureOptions | None = None,
        expert_stack_options: ExpertsSubmoduleStackOptions | None = None,
        sampler_options: ExpertsSamplerOptions | None = None,
        router_options: ExpertsRouterOptions | None = None,
        router_stack_options: ExpertsSubmoduleStackOptions | None = None,
        expert_layer_controller_options: ExpertsLayerControllerOptions | None = None,
        expert_dynamic_memory_options: ExpertsDynamicMemoryOptions | None = None,
        expert_recurrent_controller_options: (
            ExpertsRecurrentControllerOptions | None
        ) = None,
        expert_attention_flag: bool = config.EXPERT_ATTENTION_FLAG,
        expert_attention_use_kv_expert_models_flag: bool = (
            config.EXPERT_ATTENTION_USE_KV_EXPERT_MODELS_FLAG
        ),
        **kwargs,
    ) -> None:
        expert_defaults_kwargs = linear_builder_kwargs_from_flat({}, expert_defaults)
        self.mixture_options = (
            mixture_options or expert_defaults_kwargs["mixture_options"]
        )
        self.expert_stack_options = (
            expert_stack_options or expert_defaults_kwargs["expert_stack_options"]
        )
        self.sampler_options = (
            sampler_options or expert_defaults_kwargs["sampler_options"]
        )
        self.router_options = (
            router_options or expert_defaults_kwargs["router_options"]
        )
        self.router_stack_options = (
            router_stack_options or expert_defaults_kwargs["router_stack_options"]
        )
        self.expert_layer_controller_options = (
            expert_layer_controller_options
            or expert_defaults_kwargs["expert_layer_controller_options"]
        )
        self.expert_dynamic_memory_options = (
            expert_dynamic_memory_options
            or expert_defaults_kwargs["expert_dynamic_memory_options"]
        )
        self.expert_recurrent_controller_options = (
            expert_recurrent_controller_options
            or expert_defaults_kwargs["expert_recurrent_controller_options"]
        )
        self.expert_attention_flag = expert_attention_flag
        self.expert_attention_use_kv_expert_models_flag = (
            expert_attention_use_kv_expert_models_flag
        )
        super().__init__(*args, **kwargs)
        self.experiment_config_type = ExperimentConfig

    def _build_feed_forward_stack_config(self) -> MixtureOfExpertsModelConfig:
        return self._build_expert_model_config()

    def _build_attention_config(self):
        if not self.expert_attention_flag:
            return super()._build_attention_config()
        encoder_options = self.encoder_options
        attention_options = self.attention_options
        return MixtureOfAttentionHeadsConfig(
            batch_size=self.batch_size,
            num_heads=attention_options.num_heads,
            embedding_dim=self.hidden_dim,
            query_key_projection_dim=self.hidden_dim,
            value_projection_dim=self.hidden_dim,
            target_sequence_length=self.sequence_length,
            source_sequence_length=self.sequence_length,
            target_dtype=torch.float32,
            dropout_probability=encoder_options.dropout_probability,
            zero_attention_flag=False,
            causal_attention_mask_flag=False,
            add_key_value_bias_flag=attention_options.add_key_value_bias_flag,
            average_attention_weights_flag=False,
            return_attention_weights_flag=False,
            projection_model_config=self._build_attention_projection_stack_config(),
            experts_config=self._build_attention_experts_config(),
            use_kv_expert_models_flag=self.expert_attention_use_kv_expert_models_flag,
        )

    def _build_attention_experts_config(self):
        model_config = self._build_expert_model_config()
        return model_config.stack_config.layer_config.layer_model_config

    def _build_expert_model_config(self) -> MixtureOfExpertsModelConfig:
        stack_options = ExpertsStackOptions(
            hidden_dim=self.hidden_dim,
            bias_flag=self.feed_forward_options.bias_flag,
            layer_norm_position=self.encoder_options.layer_norm_position,
            num_layers=self.feed_forward_options.num_layers,
            activation=self.encoder_options.activation,
            residual_connection_option=config.ResidualConnectionOptions.DISABLED,
            dropout_probability=self.encoder_options.dropout_probability,
            last_layer_bias_option=config.LastLayerBiasOptions.DEFAULT,
            apply_output_pipeline_flag=True,
        )
        factory = ControlConfigFactory(
            ControlConfigDependencies(
                stack_options=stack_options,
                submodule_stack_options=None,
                mixture_options=self.mixture_options,
                expert_stack_options=self.expert_stack_options,
                sampler_options=self.sampler_options,
                router_options=self.router_options,
                router_stack_options=self.router_stack_options,
                layer_controller_options=None,
                dynamic_memory_options=None,
                recurrent_controller_options=None,
                expert_layer_controller_options=self.expert_layer_controller_options,
                expert_dynamic_memory_options=self.expert_dynamic_memory_options,
                expert_recurrent_controller_options=(
                    self.expert_recurrent_controller_options
                ),
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
        )
        model_config = factory.build()
        if isinstance(model_config, MixtureOfExpertsModelConfig):
            return model_config
        return model_config.block_config
